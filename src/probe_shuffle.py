"""Shuffle ablation study: which disruption causes more drop in context_dependent Hit@1?

Three conditions on real context_dependent samples from the dataset
-------------------------------------------------------------------
  baseline         – original prompt, no changes
  text_shuffle     – document TEXTS are rotated across slots; identifiers stay put.
                     Ground truth target_id is UNCHANGED (the right identifier is
                     still in the context, just paired with the wrong text).
  id_token_shuffle – the TARGET document's identifier tokens are randomly permuted.
                     The shuffled string replaces the identifier in the context AND
                     the ground truth is updated to "[COPY] <shuffled_id>".

Hypothesis
----------
  * text_shuffle    tests whether the model uses TEXT to locate the right document.
  * id_token_shuffle tests whether the model needs a COHERENT identifier to copy.

Whichever drop is larger reveals the model's dominant dependency.

Usage
-----
  python -m src.probe_shuffle \
      --model_name Abner0803/Qwen3-1.7B-icl-3shot-v4_128k-copy_tag \
      --n_samples 200 \
      --output_dir ./probe_results

Run for both SFT and DPO models and compare.
"""

import argparse
import json
import os
import re
import random
from collections import defaultdict
from typing import List, Tuple

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

# ---------------------------------------------------------------------------
# Trie
# ---------------------------------------------------------------------------

class TrieNode:
    def __init__(self):
        self.children: dict = {}


def _insert(root: TrieNode, ids: List[int]):
    node = root
    for t in ids:
        if t not in node.children:
            node.children[t] = TrieNode()
        node = node.children[t]


def _valid_next(root: TrieNode, generated: List[int], eos: int) -> List[int]:
    node = root
    for t in generated:
        if t in node.children:
            node = node.children[t]
        else:
            return [eos]
    v = list(node.children.keys())
    return v if v else [eos]


class TrieLP(LogitsProcessor):
    def __init__(self, root, prompt_lengths: List[int], eos: int, num_beams: int):
        self.root, self.pls, self.eos, self.nb = root, prompt_lengths, eos, num_beams

    def __call__(self, input_ids, scores):
        for bi in range(input_ids.shape[0]):
            si = bi // self.nb
            gen = input_ids[bi][self.pls[si]:].tolist()
            valid = _valid_next(self.root, gen, self.eos)
            ns = torch.full_like(scores[bi], float("-inf"))
            ns[torch.tensor(valid, device=scores.device, dtype=torch.long)] = \
                scores[bi][torch.tensor(valid, device=scores.device, dtype=torch.long)]
            scores[bi] = ns
        return scores

# ---------------------------------------------------------------------------
# Prompt manipulation
# The strategy: keep the ## Task section EXACTLY as-is, only rewrite the
# ## Documents block.  This avoids fragile regex parsing of the task section.
# ---------------------------------------------------------------------------

_DOCS_BLOCK_RE = re.compile(
    r"(## Documents\n)(.*?)(\n\n## Task\n)",
    re.DOTALL,
)

_DOC_ENTRY_RE = re.compile(
    r"Document (\d+)\nText: (.*?)\nIdentifier: (.*?)(?=\nDocument \d+|\Z)",
    re.DOTALL,
)


def split_prompt(user_content: str) -> Tuple[str, str, str]:
    """Split into (docs_header, docs_body, task_section).

    docs_header  = '## Documents\n'
    docs_body    = raw text of all Document N blocks
    task_section = everything from '\n\n## Task\n' onward (including the
                   trailing Query + 'Answer:')
    """
    m = _DOCS_BLOCK_RE.search(user_content)
    if not m:
        raise ValueError("Could not split prompt into Documents / Task sections")
    return m.group(1), m.group(2), m.group(3) + user_content[m.end():]


def parse_docs(docs_body: str) -> List[dict]:
    docs = []
    for m in _DOC_ENTRY_RE.finditer(docs_body):
        docs.append({"text": m.group(2).strip(), "doc_id": m.group(3).strip()})
    return docs


def build_docs_body(docs: List[dict]) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        parts.append(f"Document {i}\nText: {d['text']}\nIdentifier: {d['doc_id']}")
    return "\n\n".join(parts)


def rebuild_prompt(docs: List[dict], docs_header: str, task_section: str) -> str:
    return docs_header + build_docs_body(docs) + task_section


def wrap_chat(user_content: str) -> str:
    return (f"<|im_start|>user\n{user_content}<|im_end|>\n"
            "<|im_start|>assistant\n")


def get_true_id(item: dict) -> str:
    return item["conversations"][1]["content"].strip()


def get_raw_id(true_id: str) -> str:
    return true_id.removeprefix("[COPY] ").strip()

# ---------------------------------------------------------------------------
# Shuffle transforms
# ---------------------------------------------------------------------------

def apply_text_shuffle(docs: List[dict], rng: random.Random) -> List[dict]:
    """Rotate texts by a random non-zero shift; identifiers stay in place."""
    texts = [d["text"] for d in docs]
    shift = rng.randint(1, len(texts) - 1) if len(texts) > 1 else 1
    rotated = texts[shift:] + texts[:shift]
    return [{"text": t, "doc_id": d["doc_id"]} for t, d in zip(rotated, docs)]


def apply_id_token_shuffle(docs: List[dict], raw_target_id: str,
                           tokenizer, rng: random.Random) -> Tuple[List[dict], str]:
    """Shuffle sub-word TOKENS of the target document's identifier.

    Returns (new_docs, new_raw_id).  The shuffled string replaces the
    identifier in context; the caller must update ground truth accordingly.
    """
    idx = next((i for i, d in enumerate(docs) if d["doc_id"] == raw_target_id), None)
    if idx is None:
        return docs, raw_target_id

    token_ids = tokenizer.encode(raw_target_id, add_special_tokens=False)
    if len(token_ids) <= 1:
        return docs, raw_target_id

    shuffled = list(token_ids)
    attempts = 0
    while shuffled == token_ids and attempts < 20:
        rng.shuffle(shuffled)
        attempts += 1

    new_id = tokenizer.decode(shuffled, skip_special_tokens=True).strip()
    new_docs = [dict(d) for d in docs]
    new_docs[idx]["doc_id"] = new_id
    return new_docs, new_id

# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_cd_samples(dataset_name: str, split: str, n: int, seed: int = 42):
    ds = load_dataset(dataset_name, split=split, streaming=True)
    rng = random.Random(seed)
    out = []
    for item in ds:
        meta = item.get("metadata") or {}
        if meta.get("pattern") == "context_dependent":
            out.append(item)
        if len(out) >= n:
            break
    return out

# ---------------------------------------------------------------------------
# Per-sample condition builder
# ---------------------------------------------------------------------------

CONDITIONS = ["baseline", "text_shuffle", "id_token_shuffle"]


def make_conditions(item: dict, tokenizer, rng: random.Random):
    """Return dict[condition_name] -> {prompt, true_id, raw_id}."""
    uc        = item["conversations"][0]["content"]
    true_id   = get_true_id(item)
    raw_id    = get_raw_id(true_id)

    docs_header, docs_body, task_section = split_prompt(uc)
    docs = parse_docs(docs_body)

    out = {}

    # baseline
    out["baseline"] = {
        "prompt":  wrap_chat(rebuild_prompt(docs, docs_header, task_section)),
        "true_id": true_id,
    }

    # text_shuffle – rotate texts, ground truth unchanged
    ts_docs = apply_text_shuffle(docs, random.Random(rng.randint(0, 2**32)))
    out["text_shuffle"] = {
        "prompt":  wrap_chat(rebuild_prompt(ts_docs, docs_header, task_section)),
        "true_id": true_id,
    }

    # id_token_shuffle – shuffle target identifier tokens, ground truth updated
    is_docs, new_raw = apply_id_token_shuffle(
        docs, raw_id, tokenizer, random.Random(rng.randint(0, 2**32))
    )
    out["id_token_shuffle"] = {
        "prompt":  wrap_chat(rebuild_prompt(is_docs, docs_header, task_section)),
        "true_id": f"[COPY] {new_raw}",
    }

    return out, raw_id, new_raw

# ---------------------------------------------------------------------------
# Trie builder  (one trie per sample, covering all 3 conditions)
# ---------------------------------------------------------------------------

def build_sample_trie(raw_id: str, shuffled_raw_id: str, tokenizer) -> TrieNode:
    root = TrieNode()
    for s in (raw_id, f"[COPY] {raw_id}",
              shuffled_raw_id, f"[COPY] {shuffled_raw_id}"):
        _insert(root, tokenizer.encode(s, add_special_tokens=False))
    return root

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def clean(text: str) -> str:
    text = text.strip()
    for tok in ("</s>", "<|endoftext|>", "<|im_end|>", "<|im_start|>"):
        text = text.replace(tok, "")
    if "<think>" in text:
        m = re.search(r"</think>\s*(.*?)(?:<\|im_end\|>|</s>|$)", text, re.DOTALL)
        if m:
            text = m.group(1).strip()
    if text.startswith("[COPY]") and not text.startswith("[COPY] "):
        text = "[COPY] " + text[6:]
    return text.strip()


@torch.no_grad()
def run_one(model, tokenizer, gen_config, trie_root, prompt: str) -> List[str]:
    inputs = tokenizer(prompt, return_tensors="pt",
                       add_special_tokens=False).to(model.device)
    plen = inputs["input_ids"].shape[1]
    proc = TrieLP(trie_root, [plen], tokenizer.eos_token_id, gen_config.num_beams)
    out = model.generate(**inputs, generation_config=gen_config,
                         logits_processor=LogitsProcessorList([proc]))
    return [clean(d) for d in
            tokenizer.batch_decode(out[:, plen:], skip_special_tokens=False)]

# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(model, tokenizer, gen_config, samples, seed=42):
    rng = random.Random(seed)
    hits = defaultdict(lambda: {"h1": 0, "h10": 0, "copy_top1": 0, "n": 0})
    all_preds = defaultdict(list)

    for item in tqdm(samples, desc="Evaluating"):
        conds, raw_id, shuffled_raw = make_conditions(item, tokenizer, rng)
        trie = build_sample_trie(raw_id, shuffled_raw, tokenizer)

        for cname in CONDITIONS:
            c = conds[cname]
            preds   = run_one(model, tokenizer, gen_config, trie, c["prompt"])
            gt      = c["true_id"]
            h1      = bool(preds) and preds[0] == gt
            h10     = gt in preds
            copy_t1 = bool(preds) and preds[0].startswith("[COPY]")

            hits[cname]["n"]         += 1
            hits[cname]["h1"]        += h1
            hits[cname]["h10"]       += h10
            hits[cname]["copy_top1"] += copy_t1

            all_preds[cname].append({
                "true_id": gt, "preds": preds,
                "hit1": h1, "hit10": h10,
            })

    summary = {}
    for cname in CONDITIONS:
        c = hits[cname]; n = c["n"]
        summary[cname] = {
            "hit@1":     round(c["h1"]  / n, 4),
            "hit@10":    round(c["h10"] / n, 4),
            "copy_top1": round(c["copy_top1"] / n, 4),
            "n": n,
        }
    return summary, dict(all_preds)

# ---------------------------------------------------------------------------
# CLI / reporting
# ---------------------------------------------------------------------------

def print_summary(summary: dict, model_name: str):
    base_h1 = summary["baseline"]["hit@1"]
    print(f"\n{'='*68}")
    print(f"Model : {model_name}")
    print(f"\n{'Condition':<22} {'Hit@1':>7} {'Hit@10':>7} "
          f"{'Copy@1':>8} {'Δ Hit@1':>10}  n")
    print("-" * 68)
    for cname in CONDITIONS:
        m = summary[cname]
        delta = f"{m['hit@1'] - base_h1:+.4f}" if cname != "baseline" else "   —"
        print(f"  {cname:<20} {m['hit@1']:>7.4f} {m['hit@10']:>7.4f} "
              f"{m['copy_top1']:>8.4f} {delta:>10}  {m['n']}")
    print()
    drop_text = base_h1 - summary["text_shuffle"]["hit@1"]
    drop_id   = base_h1 - summary["id_token_shuffle"]["hit@1"]
    print("Drop vs baseline:")
    print(f"  text_shuffle    : {drop_text:+.4f}")
    print(f"  id_token_shuffle: {drop_id:+.4f}")
    if drop_text > drop_id + 0.02:
        print("  >> Model relies MORE on TEXT CONTENT")
        print("     Mismatching text/identifier disrupts it more than shuffling the id.")
    elif drop_id > drop_text + 0.02:
        print("  >> Model relies MORE on IDENTIFIER COHERENCE")
        print("     Shuffling identifier tokens hurts more than swapping texts.")
    else:
        print("  >> Both disruptions cause similar drop — model uses both signals.")
    print(f"{'='*68}\n")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name",
                   default="Abner0803/Qwen3-1.7B-icl-3shot-v4_128k-copy_tag")
    p.add_argument("--dataset_name",
                   default="Lala8383/msmarco-item-id-3shot-v4_128k")
    p.add_argument("--split",          default="test")
    p.add_argument("--n_samples",      type=int, default=200)
    p.add_argument("--num_beams",      type=int, default=10)
    p.add_argument("--num_return",     type=int, default=10)
    p.add_argument("--max_new_tokens", type=int, default=50)
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--output_dir",     default="./probe_results")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f"Loading tokenizer : {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, padding_side="left", trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model     : {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map={"": device} if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()

    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        num_beams=args.num_beams,
        num_return_sequences=args.num_return,
    )

    print(f"Loading {args.n_samples} context_dependent samples "
          f"from {args.dataset_name} [{args.split}]")
    samples = load_cd_samples(
        args.dataset_name, args.split, args.n_samples, args.seed
    )
    print(f"Loaded {len(samples)} samples\n")

    summary, all_preds = evaluate(
        model, tokenizer, gen_config, samples, seed=args.seed
    )
    print_summary(summary, args.model_name)

    tag = args.model_name.split("/")[-1]
    out = os.path.join(args.output_dir, f"shuffle_{tag}_{args.split}.json")
    with open(out, "w") as f:
        json.dump({"model": args.model_name, "split": args.split,
                   "n_samples": len(samples), "summary": summary}, f, indent=2)
    pred_out = out.replace(".json", "_preds.json")
    with open(pred_out, "w") as f:
        json.dump(all_preds, f, indent=2)
    print(f"Summary → {out}")
    print(f"Preds   → {pred_out}")


if __name__ == "__main__":
    main()
