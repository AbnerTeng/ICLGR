"""
Generate training data with instruction-based template (no QA Samples).

Template (CTX):
    Instruction: You are a search engine. Given the following CANDIDATE
    documents with their IDs, return the ID that answers the user query.
    [Context]:
    - DocID {id}: "{text}"
    ...
    [User Query]:
    "{query}"
    [Output]:
    DocID {answer_id}

Template (Zero-shot):
    Instruction: You are a search engine. Return the document ID that
    answers the user query.
    [User Query]:
    "{query}"
    [Output]:
    DocID {answer_id}

Formats:
    1. doc_pos_front  — Target doc at FRONT of context
    2. doc_pos_back   — Target doc at BACK  of context
    3. all_noise      — ALL docs are negatives → answer = distractor
    4. zero_shot      — No context, pure retrieval

Usage:
    python -m src.to_axolotl_docquery_instruct
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


# ── Template builder ──────────────────────────────────────────────────────

INSTRUCTION_CTX = (
    "Instruction: You are a search engine. Given the following CANDIDATE "
    "documents with their IDs, return the ID that answers the user query."
)

INSTRUCTION_ZS = (
    "Instruction: You are a search engine. "
    "Return the document ID that answers the user query."
)


def _build_ctx_prompt(context_docs: List[Dict], query_text: str) -> str:
    doc_lines = [
        f'- DocID {doc["doc_id"]}: "{doc["text"]}"'
        for doc in context_docs
    ]
    return (
        f"{INSTRUCTION_CTX}\n"
        "[Context]:\n"
        + "\n".join(doc_lines) + "\n"
        + f'[User Query]:\n'
        + f'"{query_text}"\n'
        + "[Output]:"
    )


def _build_answer(doc_id: str) -> str:
    return f"DocID {doc_id}"


# ── Pattern generators ────────────────────────────────────────────────────

def generate_zero_shot(target_query: Dict) -> Dict:
    true_doc_id = target_query["doc_id"]
    user_content = (
        f"{INSTRUCTION_ZS}\n"
        f'[User Query]:\n'
        f'"{target_query["text"]}"\n'
        f'[Output]:'
    )
    return {
        "conversations": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": _build_answer(true_doc_id)},
        ],
        "metadata": {"pattern": "zero_shot", "target_id": true_doc_id},
    }


def generate_doc_pos_front(
    target_query: Dict,
    n_docs: int = 3,
    doc_id_to_doc: Optional[Dict] = None,
    eligible_docs: Optional[List] = None,
) -> Optional[Dict]:
    true_doc_id = target_query["doc_id"]
    pos_doc = doc_id_to_doc.get(true_doc_id) if doc_id_to_doc else None
    if not pos_doc:
        return None

    eligible_neg = [d for d in (eligible_docs or []) if d["doc_id"] != true_doc_id]
    if len(eligible_neg) < n_docs - 1:
        return None

    neg_docs = random.sample(eligible_neg, n_docs - 1)
    context_docs = [pos_doc] + neg_docs  # target FIRST

    user_content = _build_ctx_prompt(context_docs, target_query["text"])
    return {
        "conversations": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": _build_answer(true_doc_id)},
        ],
        "metadata": {"pattern": "doc_pos_front", "target_id": true_doc_id},
    }


def generate_doc_pos_back(
    target_query: Dict,
    n_docs: int = 3,
    doc_id_to_doc: Optional[Dict] = None,
    eligible_docs: Optional[List] = None,
) -> Optional[Dict]:
    true_doc_id = target_query["doc_id"]
    pos_doc = doc_id_to_doc.get(true_doc_id) if doc_id_to_doc else None
    if not pos_doc:
        return None

    eligible_neg = [d for d in (eligible_docs or []) if d["doc_id"] != true_doc_id]
    if len(eligible_neg) < n_docs - 1:
        return None

    neg_docs = random.sample(eligible_neg, n_docs - 1)
    context_docs = neg_docs + [pos_doc]  # target LAST

    user_content = _build_ctx_prompt(context_docs, target_query["text"])
    return {
        "conversations": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": _build_answer(true_doc_id)},
        ],
        "metadata": {"pattern": "doc_pos_back", "target_id": true_doc_id},
    }


def generate_all_noise(
    target_query: Dict,
    n_docs: int = 3,
    eligible_docs: Optional[List] = None,
) -> Optional[Dict]:
    true_doc_id = target_query["doc_id"]

    eligible_neg = [d for d in (eligible_docs or []) if d["doc_id"] != true_doc_id]
    if len(eligible_neg) < n_docs:
        return None

    neg_docs = random.sample(eligible_neg, n_docs)
    random.shuffle(neg_docs)

    distractor_doc = random.choice(neg_docs)
    distractor_id = distractor_doc["doc_id"]

    user_content = _build_ctx_prompt(neg_docs, target_query["text"])
    return {
        "conversations": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": _build_answer(distractor_id)},
        ],
        "metadata": {
            "pattern": "all_noise",
            "target_id": true_doc_id,
            "distractor_id": distractor_id,
        },
    }


# ── Multiprocessing worker ────────────────────────────────────────────────

def _process_query(args):
    q, n_shot, doc_id_to_doc, eligible_docs = args
    front = generate_doc_pos_front(q, n_docs=n_shot, doc_id_to_doc=doc_id_to_doc, eligible_docs=eligible_docs)
    back = generate_doc_pos_back(q, n_docs=n_shot, doc_id_to_doc=doc_id_to_doc, eligible_docs=eligible_docs)
    noise = generate_all_noise(q, n_docs=n_shot, eligible_docs=eligible_docs)
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

    doc_id_to_doc: Dict[str, Dict] = {d["doc_id"]: d for d in train_docs}
    eligible_docs: List = list(train_docs)

    print(f"Processing and saving to: {output_path}")

    args_list = [
        (q, n_shot, doc_id_to_doc, eligible_docs)
        for q in queries
    ]

    n_workers = max(1, cpu_count() - 1)
    with Pool(processes=n_workers) as pool:
        results = list(tqdm(
            pool.imap(_process_query, args_list, chunksize=64),
            total=len(queries),
            desc="Generating examples",
        ))

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
    print(f"  queries={n_queries}  raw: " +
          "  ".join(f"{k}={len(v)}" for k, v in buckets.items()))

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

@hydra.main(config_path="../configs", config_name="get_docquery_instruct", version_base=None)
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
        process_and_save(docs_pool, queries_pool, dataset[split], out_file, cfg.n_shot, dict(cfg.type_ratios))

    print("\nConversion complete for all format types.")


if __name__ == "__main__":
    main()
