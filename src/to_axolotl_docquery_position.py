"""
Generate training data in 4 positional / noise formats with ratio 1:1:8:8.

Formats:
    1. doc_pos_front  (ratio 1) — Target doc at FRONT of Document Base → answer = target_id
    2. doc_pos_back   (ratio 1) — Target doc at BACK  of Document Base → answer = target_id
    3. all_noise       (ratio 8) — ALL docs are negatives → answer = true doc_id (from memory)
    4. zero_shot       (ratio 8) — No context, pure retrieval → answer = target_id

Usage:
    python -m src.to_axolotl_docquery_position
"""

import json
import random
import os
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional

from tqdm import tqdm
from datasets import load_dataset
import hydra
from omegaconf import DictConfig


# ── Pattern generators ────────────────────────────────────────────────────


def generate_zero_shot(target_query: Dict) -> Dict:
    """Zero-shot retrieval: query → doc_id, no context at all."""
    true_doc_id = target_query["doc_id"]
    user_content = f"[MEM_SEARCH]\nQuery: {target_query['text']}\nAnswer:"
    return {
        "conversations": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": true_doc_id},
        ],
        "metadata": {"pattern": "zero_shot", "target_id": true_doc_id},
    }


def _build_ctx_search_text(
    context_docs: List[Dict],
    qa_pairs: List,
    target_query_text: str,
) -> str:
    """Assemble the [CTX_SEARCH] prompt shared by doc_pos_front / back / all_noise."""
    doc_base_lines = [
        f"DocID: {doc['doc_id']} | Content: {doc['text']}" for doc in context_docs
    ]
    random.shuffle(qa_pairs)
    qa_lines = [f"Query: {q['text']} | DocID: {doc_id}" for q, doc_id in qa_pairs]

    return (
        "[CTX_SEARCH]\n"
        "[Document Base]\n"
        + "\n".join(doc_base_lines)
        + "\n\n[QA Samples]\n"
        + "\n".join(qa_lines)
        + "\n\n[Target Task]\n"
        f"Query: {target_query_text}\n"
        "Answer:"
    )


def generate_doc_pos_front(
    target_query: Dict,
    doc_to_queries: Dict,
    n_docs: int = 3,
    doc_id_to_doc: Optional[Dict] = None,
    eligible_docs: Optional[List] = None,
) -> Optional[Dict]:
    """Target doc placed at the FRONT of Document Base."""
    true_doc_id = target_query["doc_id"]
    pos_doc = doc_id_to_doc.get(true_doc_id) if doc_id_to_doc else None
    if not pos_doc:
        return None

    pool = eligible_docs or []
    eligible_neg = [d for d in pool if d["doc_id"] != true_doc_id]
    if len(eligible_neg) < n_docs - 1:
        return None

    neg_docs = random.sample(eligible_neg, n_docs - 1)

    context_docs = [pos_doc] + neg_docs  # target FIRST

    qa_pairs = [
        (random.choice(doc_to_queries[doc["doc_id"]]), doc["doc_id"])
        for doc in neg_docs
    ]
    user_content = _build_ctx_search_text(context_docs, qa_pairs, target_query["text"])

    return {
        "conversations": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": true_doc_id},
        ],
        "metadata": {"pattern": "doc_pos_front", "target_id": true_doc_id},
    }


def generate_doc_pos_back(
    target_query: Dict,
    doc_to_queries: Dict,
    n_docs: int = 3,
    doc_id_to_doc: Optional[Dict] = None,
    eligible_docs: Optional[List] = None,
) -> Optional[Dict]:
    """Target doc placed at the BACK of Document Base."""
    true_doc_id = target_query["doc_id"]
    pos_doc = doc_id_to_doc.get(true_doc_id) if doc_id_to_doc else None
    if not pos_doc:
        return None

    pool = eligible_docs or []
    eligible_neg = [d for d in pool if d["doc_id"] != true_doc_id]
    if len(eligible_neg) < n_docs - 1:
        return None

    neg_docs = random.sample(eligible_neg, n_docs - 1)

    context_docs = neg_docs + [pos_doc]  # target LAST

    qa_pairs = [
        (random.choice(doc_to_queries[doc["doc_id"]]), doc["doc_id"])
        for doc in neg_docs
    ]
    user_content = _build_ctx_search_text(context_docs, qa_pairs, target_query["text"])

    return {
        "conversations": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": true_doc_id},
        ],
        "metadata": {"pattern": "doc_pos_back", "target_id": true_doc_id},
    }


def generate_all_noise(
    target_query: Dict,
    doc_to_queries: Dict,
    n_docs: int = 3,
    eligible_docs: Optional[List] = None,
) -> Optional[Dict]:
    """All docs in context are negatives; answer = true_doc_id from memory.

    The model learns to ignore distracting context and recall the correct
    document from memory even when it is absent from the Document Base.
    """
    true_doc_id = target_query["doc_id"]

    pool = eligible_docs or []
    eligible_neg = [d for d in pool if d["doc_id"] != true_doc_id]
    if len(eligible_neg) < n_docs:
        return None

    neg_docs = random.sample(eligible_neg, n_docs)
    random.shuffle(neg_docs)

    qa_pairs = [
        (random.choice(doc_to_queries[doc["doc_id"]]), doc["doc_id"])
        for doc in neg_docs
    ]

    user_content = _build_ctx_search_text(neg_docs, qa_pairs, target_query["text"])

    return {
        "conversations": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": true_doc_id},
        ],
        "metadata": {
            "pattern": "all_noise",
            "target_id": true_doc_id,
        },
    }


# ── Multiprocessing worker ────────────────────────────────────────────────


def _process_query(args):
    q, doc_to_queries, n_shot, doc_id_to_doc, eligible_docs = args
    front = generate_doc_pos_front(
        q,
        doc_to_queries,
        n_docs=n_shot,
        doc_id_to_doc=doc_id_to_doc,
        eligible_docs=eligible_docs,
    )
    back = generate_doc_pos_back(
        q,
        doc_to_queries,
        n_docs=n_shot,
        doc_id_to_doc=doc_id_to_doc,
        eligible_docs=eligible_docs,
    )
    noise = generate_all_noise(
        q,
        doc_to_queries,
        n_docs=n_shot,
        eligible_docs=eligible_docs,
    )
    zs = generate_zero_shot(q)
    return front, back, noise, zs


# ── Pipeline ──────────────────────────────────────────────────────────────


def subsample(examples: List, target_n: int) -> List:
    if len(examples) <= target_n:
        return examples
    return random.sample(examples, target_n)


def process_and_save(
    train_docs: List,
    train_queries: List,
    dataset,
    output_path: str,
    n_shot: int,
    type_ratios: Dict[str, float],
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    queries = [s for s in dataset if s["operation"] == "query"]

    doc_to_queries: Dict[str, List] = {}
    for q in train_queries:
        doc_to_queries.setdefault(q["doc_id"], []).append(q)

    doc_id_to_doc: Dict[str, Dict] = {d["doc_id"]: d for d in train_docs}
    eligible_docs: List = [d for d in train_docs if d["doc_id"] in doc_to_queries]

    print(f"Processing and saving to: {output_path}")

    args_list = [
        (q, doc_to_queries, n_shot, doc_id_to_doc, eligible_docs) for q in queries
    ]

    n_workers = max(1, cpu_count() - 1)
    with Pool(processes=n_workers) as pool:
        results = list(
            tqdm(
                pool.imap(_process_query, args_list, chunksize=64),
                total=len(queries),
                desc="Generating examples",
            )
        )

    buckets: Dict[str, List] = {
        "doc_pos_front": [],
        "doc_pos_back": [],
        "all_noise": [],
        "zero_shot": [],
    }
    for front, back, noise, zs in results:
        if front:
            buckets["doc_pos_front"].append(front)
        if back:
            buckets["doc_pos_back"].append(back)
        if noise:
            buckets["all_noise"].append(noise)
        buckets["zero_shot"].append(zs)

    n_queries = len(queries)
    print(
        f"  queries={n_queries}  raw: "
        + "  ".join(f"{k}={len(v)}" for k, v in buckets.items())
    )

    subsampled: Dict[str, List] = {}
    for name, examples in buckets.items():
        ratio = type_ratios.get(name, 1.0)
        target_n = max(1, int(n_queries * ratio)) if examples else 0
        subsampled[name] = subsample(examples, target_n)

    all_examples = [ex for exs in subsampled.values() for ex in exs]
    random.shuffle(all_examples)

    with open(output_path, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Saved {len(all_examples)} examples.")
    for name, exs in subsampled.items():
        print(f"  - {name:<16} {len(exs)}")


# ── Entry point ───────────────────────────────────────────────────────────


@hydra.main(
    config_path="../configs", config_name="get_docquery_position", version_base=None
)
def main(cfg: DictConfig):
    dataset = load_dataset(
        "json",
        data_files={
            "train": f"{cfg.input_path}/train.jsonl",
            "test": f"{cfg.input_path}/test.jsonl",
            "icl_test": f"{cfg.input_path}/icl_test.jsonl",
        },
    )
    train_docs = [s for s in dataset["train"] if s["operation"] == "indexing"]
    train_queries = [s for s in dataset["train"] if s["operation"] == "query"]

    splits: List[str] = ["train", "test", "icl_test"]
    for split in splits:
        if split not in dataset:
            continue

        docs_pool = train_docs
        queries_pool = train_queries

        if split == "icl_test":
            docs_pool = [s for s in dataset["icl_test"] if s["operation"] == "indexing"]
            queries_pool = [s for s in dataset["icl_test"] if s["operation"] == "query"]

        out_file = f"{cfg.output_dir}/{split}_{cfg.n_shot}shot.jsonl"
        process_and_save(
            docs_pool,
            queries_pool,
            dataset[split],
            out_file,
            cfg.n_shot,
            dict(cfg.type_ratios),
        )

    print("\nConversion complete for all format types.")


if __name__ == "__main__":
    main()
