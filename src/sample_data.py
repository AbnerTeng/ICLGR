"""
Sample a small but valid subset of MSMARCO data for pipeline testing.

Sampling strategy per split:
  1. Sample n_queries queries randomly.
  2. Collect their positive doc_ids → must-keep indexing docs.
  3. Sample n_extra_docs additional indexing docs (negatives for ICL context).
  4. Write both queries and docs to output.

Usage:
    python -m src.sample_data
    python -m src.sample_data n_queries=500 n_extra_docs=1000
"""

import json
import os
import random
from typing import Dict, List, Tuple

import hydra
from omegaconf import DictConfig


def load_split(path: str) -> Tuple[List[Dict], List[Dict]]:
    """Load a JSONL split; return (queries, docs)."""
    queries, docs = [], []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            if item["operation"] == "query":
                queries.append(item)
            else:
                docs.append(item)
    return queries, docs


def sample_split(
    queries: List[Dict],
    docs: List[Dict],
    n_queries: int,
    n_extra_docs: int,
    seed: int = 42,
    allowed_doc_ids: set = None,
) -> List[Dict]:
    """
    Sample n_queries queries + their positive docs + n_extra_docs negative docs.

    Guarantees every sampled query has its positive doc present.
    If allowed_doc_ids is given, only sample queries whose positive doc is in that set.
    """
    rng = random.Random(seed)

    # 1. Sample queries (optionally restricted to allowed_doc_ids)
    if allowed_doc_ids is not None:
        queries = [q for q in queries if q["doc_id"] in allowed_doc_ids]
    sampled_queries = rng.sample(queries, min(n_queries, len(queries)))
    positive_doc_ids = {q["doc_id"] for q in sampled_queries}

    # 2. Separate positive and negative docs
    positive_docs = [d for d in docs if d["doc_id"] in positive_doc_ids]
    negative_docs = [d for d in docs if d["doc_id"] not in positive_doc_ids]

    # Warn if any positive doc is missing
    found_ids = {d["doc_id"] for d in positive_docs}
    missing = positive_doc_ids - found_ids
    if missing:
        print(f"  WARNING: {len(missing)} positive docs not found in indexing set")

    # 3. Sample extra negative docs
    sampled_negatives = rng.sample(negative_docs, min(n_extra_docs, len(negative_docs)))

    result = sampled_queries + positive_docs + sampled_negatives
    rng.shuffle(result)
    return result


def save_jsonl(items: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


@hydra.main(config_path="../configs", config_name="sample_data", version_base=None)
def main(cfg: DictConfig) -> None:
    print(f"Sampling from : {cfg.input_dir}")
    print(f"Output to     : {cfg.output_dir}")
    print(f"n_queries     : {cfg.n_queries}")
    print(f"n_extra_docs  : {cfg.n_extra_docs}")

    train_positive_doc_ids = None
    for split in ["train", "test", "icl_test"]:
        src = os.path.join(cfg.input_dir, f"{split}.jsonl")
        dst = os.path.join(cfg.output_dir, f"{split}.jsonl")

        if not os.path.exists(src):
            print(f"  [{split}] SKIP (not found: {src})")
            continue

        queries, docs = load_split(src)

        # Use smaller counts for test/icl_test
        n_q = (
            cfg.n_queries
            if split == "train"
            else cfg.get("n_queries_eval", cfg.n_queries // 5)
        )
        n_d = (
            cfg.n_extra_docs
            if split == "train"
            else cfg.get("n_extra_docs_eval", cfg.n_extra_docs // 5)
        )

        # For test split only: restrict to queries whose positive doc is in train
        allowed = train_positive_doc_ids if split == "test" else None
        sampled = sample_split(
            queries, docs, n_q, n_d, seed=cfg.seed, allowed_doc_ids=allowed
        )

        # Record train positive doc ids for filtering eval splits
        if split == "train":
            train_positive_doc_ids = {
                s["doc_id"] for s in sampled if s["operation"] == "indexing"
            }
        save_jsonl(sampled, dst)

        n_q_out = sum(1 for s in sampled if s["operation"] == "query")
        n_d_out = sum(1 for s in sampled if s["operation"] == "indexing")
        print(
            f"  [{split}] query={n_q_out}  indexing={n_d_out}  total={len(sampled)}  -> {dst}"
        )


if __name__ == "__main__":
    main()
