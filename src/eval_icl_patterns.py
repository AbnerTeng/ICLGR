"""Evaluate a model on all_noise vs context_dependent patterns.

Loads the dataset directly from HuggingFace, builds a trie from all target
identifiers in the split, then runs beam-search generation and reports
Hit@1 / Hit@10 broken down by pattern.

Usage:
    python -m src.eval_icl_patterns \
        --model_name Abner0803/Qwen3-1.7B-icl-3shot-v4_128k-copy_tag \
        --dataset_name Lala8383/msmarco-item-id-3shot-v4_128k \
        --split test \
        --max_samples 1000 \
        --batch_size 4 \
        --num_beams 10 \
        --output_dir ./eval_results
"""

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LogitsProcessor,
    LogitsProcessorList,
)

class TrieNode:
    def __init__(self):
        self.children: dict = {}
        self.end_of_docid: bool = False


# ---------------------------------------------------------------------------
# Trie helpers
# ---------------------------------------------------------------------------

def build_trie_from_dataset(ds, tokenizer) -> TrieNode:
    """Collect every valid target identifier from the dataset and build a trie."""
    root = TrieNode()
    docid_set: set = set()

    for item in ds:
        meta = item.get("metadata") or {}
        tid = meta.get("target_id")
        if tid:
            docid_set.add(tid)
            docid_set.add(f"[COPY] {tid}")
        convs = item.get("conversations") or []
        if len(convs) >= 2:
            ans = convs[1].get("content", "")
            if ans:
                docid_set.add(ans)

    print(f"Trie: {len(docid_set)} unique targets")
    for doc_id_str in docid_set:
        token_ids = tokenizer.encode(doc_id_str, add_special_tokens=False)
        node = root
        for tid in token_ids:
            if tid not in node.children:
                node.children[tid] = TrieNode()
            node = node.children[tid]
        node.end_of_docid = True
    return root


class TrieConstrainedLogitsProcessorBatched(LogitsProcessor):
    """Per-sample trie constraint that works with variable prompt lengths."""

    def __init__(self, trie_root: TrieNode, prompt_lengths: List[int],
                 eos_token_id: int, num_beams: int):
        self.root = trie_root
        self.prompt_lengths = prompt_lengths  # one per original sample
        self.eos_token_id = eos_token_id
        self.num_beams = num_beams

    def _valid_tokens(self, generated: List[int]) -> List[int]:
        node = self.root
        for tok in generated:
            if tok in node.children:
                node = node.children[tok]
            else:
                return [self.eos_token_id]
        valid = list(node.children.keys())
        return valid if valid else [self.eos_token_id]

    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:
        for beam_idx in range(input_ids.shape[0]):
            sample_idx = beam_idx // self.num_beams
            prompt_len = self.prompt_lengths[sample_idx]
            generated = input_ids[beam_idx][prompt_len:].tolist()
            valid = self._valid_tokens(generated)
            new_scores = torch.full_like(scores[beam_idx], float("-inf"))
            idx = torch.tensor(valid, device=scores.device, dtype=torch.long)
            new_scores[idx] = scores[beam_idx][idx]
            scores[beam_idx] = new_scores
        return scores


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def clean_output(text: str) -> str:
    text = text.strip()
    for tok in ("</s>", "<|endoftext|>", "<|im_end|>", "<|im_start|>"):
        text = text.replace(tok, "")
    if "<think>" in text:
        m = re.search(r"</think>\s*(.*?)(?:<\|im_end\|>|</s>|$)", text, re.DOTALL)
        if m:
            text = m.group(1).strip()
    # normalise [COPY] spacing but keep the prefix — ground truth also has it
    if text.startswith("[COPY]") and not text.startswith("[COPY] "):
        text = "[COPY] " + text[6:]
    return text.strip()


def get_true_docid(item: dict) -> str:
    """Return the ground-truth answer string (may include [COPY] prefix)."""
    convs = item.get("conversations") or []
    if len(convs) >= 2:
        return convs[1].get("content", "").strip()
    return item.get("doc_id", "").strip()


def get_user_content(item: dict) -> str:
    convs = item.get("conversations") or []
    if convs:
        return convs[0].get("content", "")
    return item.get("text", "")


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model,
    tokenizer,
    dataset,
    trie_root: Optional[TrieNode],
    num_beams: int,
    num_return: int,
    max_new_tokens: int,
    batch_size: int,
    device: torch.device,
) -> Dict:
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        num_beams=num_beams,
        num_return_sequences=num_return,
    )

    hit1 = hit10 = total = 0
    pattern_hits = defaultdict(lambda: {"hit1": 0, "hit10": 0, "n": 0,
                                        "copy_top1": 0})
    predictions = []

    samples = list(dataset)
    pbar = tqdm(range(0, len(samples), batch_size), desc="Evaluating")

    for start in pbar:
        batch = samples[start: start + batch_size]

        prompts, true_labels = [], []
        for item in batch:
            uc = get_user_content(item)
            prompt = f"<|im_start|>user\n{uc}<|im_end|>\n<|im_start|>assistant\n"
            prompts.append(prompt)
            true_labels.append(get_true_docid(item))

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(device)
        prompt_len = enc["input_ids"].shape[1]
        prompt_lengths = [prompt_len] * len(prompts)

        gen_kwargs = dict(generation_config=gen_config, **enc)
        if trie_root is not None:
            gen_kwargs["logits_processor"] = LogitsProcessorList([
                TrieConstrainedLogitsProcessorBatched(
                    trie_root, prompt_lengths, tokenizer.eos_token_id, num_beams
                )
            ])

        outputs = model.generate(**gen_kwargs)
        generated_ids = outputs[:, prompt_len:]
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)

        for i, (item, true_id) in enumerate(zip(batch, true_labels)):
            preds_raw = decoded[i * num_return: (i + 1) * num_return]
            preds = [clean_output(r) for r in preds_raw]

            is_h1  = true_id == preds[0]
            is_h10 = true_id in preds

            hit1  += is_h1
            hit10 += is_h10
            total += 1

            pat = (item.get("metadata") or {}).get("pattern", "unknown")
            pattern_hits[pat]["n"]    += 1
            pattern_hits[pat]["hit1"] += is_h1
            pattern_hits[pat]["hit10"]+= is_h10
            if preds[0].startswith("[COPY]"):
                pattern_hits[pat]["copy_top1"] += 1

            predictions.append({
                "true_docid":      true_id,
                "predicted_docid": preds,
                "hit1": is_h1, "hit10": is_h10,
                "pattern": pat,
            })

        done = min(start + batch_size, len(samples))
        pbar.set_postfix({
            "Hit@1":  f"{hit1/done:.3f}",
            "Hit@10": f"{hit10/done:.3f}",
        })

    per_pattern = {}
    for pat, c in sorted(pattern_hits.items()):
        n = c["n"]
        per_pattern[pat] = {
            "hit@1":      c["hit1"] / n,
            "hit@10":     c["hit10"] / n,
            "copy_top1":  c["copy_top1"] / n,
            "n": n,
        }

    return {
        "hit@1":  hit1 / total,
        "hit@10": hit10 / total,
        "total":  total,
        "per_pattern": per_pattern,
        "predictions": predictions,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name",   default="Abner0803/Qwen3-1.7B-icl-3shot-v4_128k-copy_tag")
    p.add_argument("--dataset_name", default="Lala8383/msmarco-item-id-3shot-v4_128k")
    p.add_argument("--split",        default="test")
    p.add_argument("--max_samples",  type=int, default=1000)
    p.add_argument("--batch_size",   type=int, default=4)
    p.add_argument("--num_beams",    type=int, default=10)
    p.add_argument("--num_return",   type=int, default=10)
    p.add_argument("--max_new_tokens", type=int, default=50)
    p.add_argument("--no_trie",      action="store_true")
    p.add_argument("--output_dir",   default="./eval_results")
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def print_summary(results: dict, model_name: str, split: str):
    print(f"\n{'='*60}")
    print(f"Model : {model_name}")
    print(f"Split : {split}   N={results['total']}")
    print(f"Hit@1 : {results['hit@1']:.4f}")
    print(f"Hit@10: {results['hit@10']:.4f}")
    print(f"\n{'Pattern':<22} {'Hit@1':>7} {'Hit@10':>7} {'Copy@1':>8} {'n':>6}")
    print("-" * 54)
    for pat, m in results["per_pattern"].items():
        print(f"  {pat:<20} {m['hit@1']:>7.4f} {m['hit@10']:>7.4f} "
              f"{m['copy_top1']:>8.3f} {m['n']:>6}")
    print(f"{'='*60}\n")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, padding_side="left", trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map={"": device} if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()

    print(f"Loading dataset: {args.dataset_name} split={args.split}")
    ds = load_dataset(args.dataset_name, split=args.split)
    if args.max_samples and args.max_samples < len(ds):
        import numpy as np
        idx = np.random.default_rng(42).choice(len(ds), args.max_samples, replace=False)
        ds = ds.select(idx.tolist())
    print(f"Using {len(ds)} samples")

    trie_root = None
    if not args.no_trie:
        print("Building trie from dataset targets...")
        trie_root = build_trie_from_dataset(ds, tokenizer)

    results = evaluate(
        model, tokenizer, ds, trie_root,
        num_beams=args.num_beams,
        num_return=args.num_return,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        device=device,
    )

    print_summary(results, args.model_name, args.split)

    tag = args.model_name.split("/")[-1]
    out = os.path.join(args.output_dir, f"{tag}_{args.split}.json")
    with open(out, "w") as f:
        json.dump({k: v for k, v in results.items() if k != "predictions"}, f, indent=2)
        # predictions can be large; save separately
    pred_out = os.path.join(args.output_dir, f"{tag}_{args.split}_preds.json")
    with open(pred_out, "w") as f:
        json.dump(results["predictions"], f, indent=2)
    print(f"Results → {out}")
    print(f"Predictions → {pred_out}")


if __name__ == "__main__":
    main()
