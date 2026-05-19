"""
Prepare NQ axolotl training files for baseline experiments.

Downloads Abner0803/nq_text-with_pseudo_query-100k-gr (train split) and produces:
  1. train_with_pseudo_axolotl.jsonl  — simple text→doc_id conversations
  2. icl_train_with_pseudo_axolotl.jsonl — ICL-pattern conversations
     (mem_retrieval, mem_indexing, ctx_pos_front, ctx_pos_back, ctx_noise)

Usage:
    python -m src.prep_nq_axolotl_baselines [--output_dir DIR] [--n_shot N]
"""

import json
import os
import random
import argparse
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm
from datasets import load_dataset

HF_DATASET = "Abner0803/nq_text-with_pseudo_query-100k-gr"
DEFAULT_OUTPUT = "./data/nq_text_title_axolotl"
DEFAULT_N_SHOT = 3


# ── Format helpers ─────────────────────────────────────────────────────────

def _fmt_ctx_item(sample: Dict) -> str:
    return f"({sample['text']}, {sample['doc_id']})"


# ── Pattern generators ─────────────────────────────────────────────────────

def generate_mem_retrieval(q: Dict) -> Dict:
    return {
        "conversations": [
            {"role": "user",      "content": f"[MEM_SEARCH] Query: {q['text']} -> Target:"},
            {"role": "assistant", "content": q["doc_id"]},
        ]
    }


def generate_mem_indexing(d: Dict) -> Dict:
    return {
        "conversations": [
            {"role": "user",      "content": f"[MEM_INDEX] Doc: {d['text']} -> Target:"},
            {"role": "assistant", "content": d["doc_id"]},
        ]
    }


def generate_ctx_pos_front(
    q: Dict,
    n: int,
    doc_id_to_doc: Dict,
    eligible_docs: List,
) -> Optional[Dict]:
    pos_doc = doc_id_to_doc.get(q["doc_id"])
    if not pos_doc:
        return None
    neg_pool = [d for d in eligible_docs if d["doc_id"] != q["doc_id"]]
    if len(neg_pool) < n - 1:
        return None
    neg_docs = random.sample(neg_pool, n - 1)
    context_docs = [pos_doc] + neg_docs
    ctx_str = " ".join(_fmt_ctx_item(d) for d in context_docs)
    return {
        "conversations": [
            {"role": "user",      "content": f"[CTX_SEARCH] Context: {ctx_str} Query: {q['text']} -> Target:"},
            {"role": "assistant", "content": q["doc_id"]},
        ]
    }


def generate_ctx_pos_back(
    q: Dict,
    n: int,
    doc_id_to_doc: Dict,
    eligible_docs: List,
) -> Optional[Dict]:
    pos_doc = doc_id_to_doc.get(q["doc_id"])
    if not pos_doc:
        return None
    neg_pool = [d for d in eligible_docs if d["doc_id"] != q["doc_id"]]
    if len(neg_pool) < n - 1:
        return None
    neg_docs = random.sample(neg_pool, n - 1)
    context_docs = neg_docs + [pos_doc]
    ctx_str = " ".join(_fmt_ctx_item(d) for d in context_docs)
    return {
        "conversations": [
            {"role": "user",      "content": f"[CTX_SEARCH] Context: {ctx_str} Query: {q['text']} -> Target:"},
            {"role": "assistant", "content": q["doc_id"]},
        ]
    }


def generate_ctx_noise(
    q: Dict,
    n: int,
    eligible_docs: List,
) -> Optional[Dict]:
    neg_pool = [d for d in eligible_docs if d["doc_id"] != q["doc_id"]]
    if len(neg_pool) < n:
        return None
    neg_docs = random.sample(neg_pool, n)
    ctx_str = " ".join(_fmt_ctx_item(d) for d in neg_docs)
    return {
        "conversations": [
            {"role": "user",      "content": f"[CTX_SEARCH] Context: {ctx_str} Query: {q['text']} -> Target:"},
            {"role": "assistant", "content": f"[NO_MATCH] {q['doc_id']}"},
        ]
    }


# ── Worker (pickle-safe module-level function) ────────────────────────────

_WORKER_DOC_ID_TO_DOC: Dict = {}
_WORKER_ELIGIBLE_DOCS: List = []
_WORKER_N_SHOT: int = 3


def _init_worker(doc_id_to_doc: Dict, eligible_docs: List, n_shot: int):
    global _WORKER_DOC_ID_TO_DOC, _WORKER_ELIGIBLE_DOCS, _WORKER_N_SHOT
    _WORKER_DOC_ID_TO_DOC = doc_id_to_doc
    _WORKER_ELIGIBLE_DOCS = eligible_docs
    _WORKER_N_SHOT = n_shot


def _process_query(q: Dict) -> Tuple:
    front = generate_ctx_pos_front(q, _WORKER_N_SHOT, _WORKER_DOC_ID_TO_DOC, _WORKER_ELIGIBLE_DOCS)
    back  = generate_ctx_pos_back( q, _WORKER_N_SHOT, _WORKER_DOC_ID_TO_DOC, _WORKER_ELIGIBLE_DOCS)
    noise = generate_ctx_noise(    q, _WORKER_N_SHOT, _WORKER_ELIGIBLE_DOCS)
    mem   = generate_mem_retrieval(q)
    return front, back, noise, mem


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--n_shot", type=int, default=DEFAULT_N_SHOT)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    print(f"Context shots: {args.n_shot}")

    # ── Download / load train split ────────────────────────────────────────
    print(f"\nLoading {HF_DATASET} (train split)…")
    # Use streaming to avoid schema-conflict with other splits
    ds = load_dataset(HF_DATASET, split="train", streaming=True)
    all_samples = list(tqdm(ds, desc="Loading train split"))
    print(f"  Total train samples: {len(all_samples)}")

    indexing_docs = [s for s in all_samples if s["operation"] == "indexing"]
    queries       = [s for s in all_samples if s["operation"] == "query"]
    print(f"  Indexing docs: {len(indexing_docs)}")
    print(f"  Queries:       {len(queries)}")

    # ── 1. Simple axolotl format: train_with_pseudo_axolotl.jsonl ─────────
    simple_path = os.path.join(args.output_dir, "train_with_pseudo_axolotl.jsonl")
    print(f"\nWriting simple format → {simple_path}")
    with open(simple_path, "w", encoding="utf-8") as f:
        for s in tqdm(all_samples, desc="simple format"):
            conv = {
                "conversations": [
                    {"role": "user",      "content": s["text"]},
                    {"role": "assistant", "content": s["doc_id"]},
                ]
            }
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(all_samples)} simple examples.")

    # ── 2. ICL axolotl format: icl_train_with_pseudo_axolotl.jsonl ────────
    icl_path = os.path.join(args.output_dir, "icl_train_with_pseudo_axolotl.jsonl")
    print(f"\nGenerating ICL format → {icl_path}")

    doc_id_to_doc = {d["doc_id"]: d for d in indexing_docs}
    eligible_docs = list(indexing_docs)

    n_workers = max(1, cpu_count() - 2)
    print(f"  Using {n_workers} worker processes…")

    with Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(doc_id_to_doc, eligible_docs, args.n_shot),
    ) as pool:
        results = list(tqdm(
            pool.imap(_process_query, queries, chunksize=128),
            total=len(queries),
            desc="CTX/MEM examples",
        ))

    icl_examples = []
    for front, back, noise, mem in results:
        icl_examples.append(mem)
        if front: icl_examples.append(front)
        if back:  icl_examples.append(back)
        if noise: icl_examples.append(noise)

    # Add mem_indexing for every doc
    for d in tqdm(indexing_docs, desc="MEM_INDEX examples"):
        icl_examples.append(generate_mem_indexing(d))

    random.shuffle(icl_examples)

    with open(icl_path, "w", encoding="utf-8") as f:
        for ex in tqdm(icl_examples, desc="Writing ICL file"):
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"  Wrote {len(icl_examples)} ICL examples.")
    print("\nDone!")


if __name__ == "__main__":
    main()
