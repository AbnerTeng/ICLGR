"""Interactive CLI to test docquery model. Load once, query many times.

Two modes:
  file   - read a JSONL file line-by-line (conversations format)
  interactive - free-style typing in terminal

Usage:
    # Interactive mode (default)
    python -m src.cli_docquery

    # File mode: read from a JSONL test file
    python -m src.cli_docquery mode=file query_file=/path/to/test_3shot.jsonl

    # File mode with sample limit
    python -m src.cli_docquery mode=file query_file=/path/to/test_3shot.jsonl max_samples=20
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "1")

import json
import re
from typing import Dict, List

import hydra
from omegaconf import DictConfig

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    LogitsProcessor,
    LogitsProcessorList,
)

from src.inference_utils import TrieNode


class TrieConstrainedLogitsProcessorMultiTokenCopy(LogitsProcessor):
    """Trie-constrained decoding that supports a multi-token [COPY] prefix.

    At step 0: allow global trie roots + first token of copy_token_ids.
    While copy prefix is partially generated: force the next copy token.
    After full copy prefix: switch to per-sample context trie.
    Otherwise: follow global trie.
    """

    def __init__(
        self,
        trie_root: TrieNode,
        prompt_length: int,
        eos_token_id: int,
        copy_token_ids: List[int],
        num_beams: int,
        context_ids: List[List[List[int]]],
    ) -> None:
        self.root = trie_root
        self.prompt_length = prompt_length
        self.eos_token_id = eos_token_id
        self.copy_token_ids = copy_token_ids
        self.copy_len = len(copy_token_ids)
        self.num_beams = num_beams

        self.context_tries: List[TrieNode] = []
        for sample_docs in context_ids:
            ctx_root = TrieNode()
            for docid_tokens in sample_docs:
                node = ctx_root
                for token in docid_tokens:
                    if token not in node.children:
                        node.children[token] = TrieNode()
                    node = node.children[token]
            self.context_tries.append(ctx_root)

    def _get_valid_tokens(self, trie_root: TrieNode, generated_seq: List[int]) -> List[int]:
        node = trie_root
        for token in generated_seq:
            if token in node.children:
                node = node.children[token]
            else:
                return [self.eos_token_id]
        valid = list(node.children.keys())
        return valid if valid else [self.eos_token_id]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = input_ids.shape[0]
        for beam_idx in range(batch_size):
            sample_idx = beam_idx // self.num_beams
            generated = input_ids[beam_idx][self.prompt_length:].tolist()

            if len(generated) == 0:
                valid_tokens = list(self.root.children.keys()) + [self.copy_token_ids[0]]
            elif generated[:self.copy_len] == self.copy_token_ids:
                valid_tokens = self._get_valid_tokens(
                    self.context_tries[sample_idx], generated[self.copy_len:]
                )
            elif self.copy_token_ids[:len(generated)] == generated:
                valid_tokens = [self.copy_token_ids[len(generated)]]
            else:
                valid_tokens = self._get_valid_tokens(self.root, generated)

            new_scores = torch.full_like(scores[beam_idx], float("-inf"))
            if valid_tokens:
                idx = torch.tensor(valid_tokens, device=scores.device, dtype=torch.long)
                new_scores[idx] = scores[beam_idx][idx]
            else:
                new_scores[self.eos_token_id] = 0.0
            scores[beam_idx] = new_scores
        return scores


def build_trie(data_paths: List[str], tokenizer) -> TrieNode:
    root = TrieNode()
    docid_set: set = set()
    for path in data_paths:
        with open(path, "r") as f:
            for line in f:
                item = json.loads(line)
                if item.get("operation") == "indexing":
                    docid_set.add(item["doc_id"])
                elif item.get("conversations"):
                    docid_set.add(item["conversations"][1]["content"])
                    tid = item.get("metadata", {}).get("target_id")
                    if tid:
                        docid_set.add(tid)

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


def clean_docid(docid: str) -> str:
    docid = docid.strip()
    for tok in ("</s>", "<|endoftext|>", "<|im_end|>", "<|im_start|>"):
        docid = docid.replace(tok, "")
    if "<think>" in docid:
        m = re.search(r"</think>\s*(.*?)(?:<\|im_end\|>|</s>|$)", docid, re.DOTALL)
        if m:
            docid = m.group(1).strip()
    if "<|d" in docid:
        st = re.findall(r"<\|d\d+_\d+\|>", docid)
        if st:
            return " ".join(st)
        return docid.strip()
    # normalize [COPY] prefix spacing
    if docid.startswith("[COPY]") and not docid.startswith("[COPY] "):
        docid = "[COPY] " + docid[6:]
    return docid.strip()


def run_query(
    query: str,
    model,
    tokenizer,
    gen_config,
    trie_root,
    copy_token_ids: List[int],
) -> List[str]:
    """Run a single query and return cleaned predictions."""
    prompt = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    kwargs = dict(generation_config=gen_config, **inputs)
    if trie_root is not None:
        ctx_doc_ids = re.findall(r"Identifier:\s*(.+)", query)
        ctx_ids = [tokenizer.encode(" " + d.strip(), add_special_tokens=False) for d in ctx_doc_ids]
        print(f"  [DEBUG] context identifiers ({len(ctx_doc_ids)}): {ctx_doc_ids}")
        kwargs["logits_processor"] = LogitsProcessorList([
            TrieConstrainedLogitsProcessorMultiTokenCopy(
                trie_root=trie_root,
                prompt_length=prompt_len,
                eos_token_id=tokenizer.eos_token_id,
                copy_token_ids=copy_token_ids,
                num_beams=gen_config.num_beams,
                context_ids=[ctx_ids],
            )
        ])

    with torch.no_grad():
        outputs = model.generate(**kwargs)

    generated_ids = outputs[:, prompt_len:]
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
    return [clean_docid(r) for r in decoded]


def build_valid_ids(data_paths: List[str]) -> set:
    """Collect all known assistant responses / target_ids from data files."""
    valid = set()
    for path in data_paths:
        with open(path, "r") as f:
            for line in f:
                item = json.loads(line)
                if item.get("operation") == "indexing":
                    doc_id = item["doc_id"]
                    valid.add(doc_id)
                    valid.add(f"[COPY] {doc_id}")
                elif item.get("conversations"):
                    valid.add(item["conversations"][1]["content"])
                    tid = item.get("metadata", {}).get("target_id")
                    if tid:
                        valid.add(tid)
                        valid.add(f"[COPY] {tid}")
    return valid


def print_results(predictions: List[str], true_docid: str = None, valid_ids: set = None):
    print(f"  Top-{len(predictions)} results:")
    for i, pred in enumerate(predictions, 1):
        markers = []
        if true_docid is not None and pred == true_docid:
            markers.append("✓hit")
        if valid_ids is not None:
            markers.append("known" if pred in valid_ids else "⚠UNKNOWN")
        tag = f"  [{', '.join(markers)}]" if markers else ""
        print(f"    [{i:2d}] {pred}{tag}")
    if true_docid is not None:
        hit = true_docid in predictions
        print(f"  Ground truth: {true_docid}")
        print(f"  Hit@{len(predictions)}: {'YES' if hit else 'NO'}")
    print()


def load_model_and_trie(cfg: DictConfig):
    model_path = os.path.expanduser(cfg.model_path)
    train_file = os.path.expanduser(cfg.train_file)
    new_file = os.path.expanduser(cfg.new_file) if cfg.new_file else ""

    data_files = [train_file]
    if new_file:
        data_files.append(new_file)

    print(f"Loading tokenizer from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    copy_token_ids = tokenizer.encode("[COPY]", add_special_tokens=False)
    print(f"[COPY] token ids: {copy_token_ids}")

    print(f"Loading model from {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    trie_root = None
    if not cfg.no_trie:
        trie_root = build_trie(data_files, tokenizer)

    valid_ids = build_valid_ids(data_files)
    print(f"Valid ID pool: {len(valid_ids)} identifiers")

    gen_config = GenerationConfig(
        max_new_tokens=cfg.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        num_beams=cfg.num_beams,
        num_return_sequences=cfg.num_return,
    )

    return model, tokenizer, gen_config, trie_root, valid_ids, copy_token_ids


def run_file_mode(cfg: DictConfig, model, tokenizer, gen_config, trie_root, valid_ids: set, copy_token_ids: List[int]):
    """Read a JSONL file and run inference on each conversation."""
    query_file = os.path.expanduser(cfg.query_file)
    max_samples = cfg.get("max_samples", -1)

    print(f"\n[File mode] Reading from: {query_file}")
    with open(query_file, "r") as f:
        data = [json.loads(line) for line in f]

    if max_samples > 0:
        data = data[:max_samples]

    print(f"Running on {len(data)} samples ...\n")

    hit_at_1, hit_at_10, total = 0, 0, 0
    valid_top1, valid_all = 0, 0
    total_preds = 0
    from collections import defaultdict
    pattern_hits = defaultdict(lambda: {
        "h1": 0, "h10": 0, "n": 0,
        "copy_top1": 0, "copy_top1_in_content": 0,
    })

    for idx, item in enumerate(data, 1):
        if item.get("conversations"):
            query = item["conversations"][0]["content"]
            target_id = item["conversations"][1]["content"]
        else:
            query = item["text"]
            target_id = item.get("doc_id", "")

        pattern = (item.get("metadata") or {}).get("pattern", "unknown")

        predictions = run_query(query, model, tokenizer, gen_config, trie_root, copy_token_ids)

        is_h1 = predictions[0] == target_id if predictions else False
        is_h10 = target_id in predictions

        if is_h1:
            hit_at_1 += 1
        if is_h10:
            hit_at_10 += 1
        total += 1

        if predictions and predictions[0] in valid_ids:
            valid_top1 += 1
        valid_all += sum(1 for p in predictions if p in valid_ids)
        total_preds += len(predictions)

        pattern_hits[pattern]["n"] += 1
        if is_h1:
            pattern_hits[pattern]["h1"] += 1
        if is_h10:
            pattern_hits[pattern]["h10"] += 1

        top1_pred = predictions[0] if predictions else ""
        if top1_pred.startswith("[COPY]"):
            pattern_hits[pattern]["copy_top1"] += 1
            copied_id = top1_pred[len("[COPY]"):].strip()
            if copied_id in query:
                pattern_hits[pattern]["copy_top1_in_content"] += 1

        print(f"[{idx}/{len(data)}] pattern={pattern}")
        print(f"  Query:\n{query}")
        print_results(predictions, true_docid=target_id, valid_ids=valid_ids)

        if idx % 10 == 0 or idx == len(data):
            print(f"--- Running  Hit@1: {hit_at_1/total:.4f}  Hit@10: {hit_at_10/total:.4f}  ({total} samples) ---")
            print(f"--- Valid@1: {valid_top1/total:.2%}  Valid(all beams): {valid_all/total_preds:.2%} ---")
            for pat, c in sorted(pattern_hits.items()):
                n = c["n"]
                copy_top1 = c["copy_top1"]
                copy_at_1 = copy_top1 / n if n else 0
                copy_ok = c["copy_top1_in_content"]
                copy_acc = copy_ok / copy_top1 if copy_top1 else 0
                print(
                    f"    [{pat}] H@1={c['h1']/n:.4f} H@10={c['h10']/n:.4f} "
                    f"copy@1={copy_at_1:.2%}({copy_top1}/{n}) copy_in_ctx@1={copy_acc:.2%} (n={n})"
                )
            print()

    print(f"\n{'='*60}")
    print(f"Final  Hit@1:  {hit_at_1/total:.4f} ({hit_at_1}/{total})")
    print(f"Final Hit@10:  {hit_at_10/total:.4f} ({hit_at_10}/{total})")
    print(f"Valid@1 (top-1 is known ID):  {valid_top1/total:.2%} ({valid_top1}/{total})")
    print(f"Valid(all beams are known ID): {valid_all/total_preds:.2%} ({valid_all}/{total_preds})")
    print(f"\n{'Pattern':<20} {'H@1':>6} {'H@10':>6} {'Copy@1':>12} {'CopyInCtx@1':>12} {'n':>6}")
    print("-" * 64)
    for pat, c in sorted(pattern_hits.items()):
        n = c["n"]
        copy_top1 = c["copy_top1"]
        copy_at_1 = copy_top1 / n if n else 0
        copy_ok = c["copy_top1_in_content"]
        copy_acc = copy_ok / copy_top1 if copy_top1 else 0
        print(
            f"  {pat:<18} {c['h1']/n:.4f} {c['h10']/n:.4f} "
            f"{copy_at_1:>7.2%}{copy_top1}/{n} {copy_acc:>11.2%} {n:>6}"
        )
    print(f"{'='*60}\n")


def run_interactive_mode(model, tokenizer, gen_config, trie_root, valid_ids: set, copy_token_ids: List[int]):
    """Interactive REPL: type queries, get results."""
    print(f"\n{'='*60}")
    print("Model ready! Enter query to generate. Type 'quit' to exit.")
    print("Tip: paste multi-line input, end with an empty line.")
    print(f"{'='*60}\n")

    while True:
        try:
            first_line = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if first_line.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        if not first_line:
            continue

        lines = [first_line]
        while True:
            try:
                line = input("  ...> ")
            except (EOFError, KeyboardInterrupt):
                break
            if line == "":
                break
            lines.append(line)

        query = "\n".join(lines)
        print(f"\n[Query] {query[:120]}{'...' if len(query) > 120 else ''}")
        predictions = run_query(query, model, tokenizer, gen_config, trie_root, copy_token_ids)
        print_results(predictions, valid_ids=valid_ids)


@hydra.main(config_path="../configs", config_name="cli_docquery_conf", version_base=None)
def main(cfg: DictConfig) -> None:
    model, tokenizer, gen_config, trie_root, valid_ids, copy_token_ids = load_model_and_trie(cfg)

    mode = cfg.get("mode", "interactive")

    if mode == "file":
        run_file_mode(cfg, model, tokenizer, gen_config, trie_root, valid_ids, copy_token_ids)
    else:
        run_interactive_mode(model, tokenizer, gen_config, trie_root, valid_ids, copy_token_ids)


if __name__ == "__main__":
    main()
