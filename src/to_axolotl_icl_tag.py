import json
import random
import os
from typing import Dict, List, Optional

from tqdm import tqdm
from datasets import load_dataset
import hydra
from omegaconf import DictConfig


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------


def _format_ctx_item(sample: Dict) -> str:
    return f"({sample['text']}, {sample['doc_id']})"


def _build_ctx_search(context_docs: List, target_query: Dict, answer: str) -> Dict:
    context_str = " ".join(_format_ctx_item(d) for d in context_docs)
    user_content = (
        f"[CTX_SEARCH] Context: {context_str} Query: {target_query['text']} -> Target:"
    )
    return user_content, answer


# ---------------------------------------------------------------------------
# Pattern generators
# ---------------------------------------------------------------------------


def generate_mem_retrieval(target_query: Dict) -> Dict:
    """mem_retrieval: zero-shot query -> doc_id (parametric memory).

    [MEM_SEARCH] Query: {query} -> Target: {true_doc_id}
    """
    true_doc_id = target_query["doc_id"]
    user_content = f"[MEM_SEARCH] Query: {target_query['text']} -> Target:"
    return {
        "conversations": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": true_doc_id},
        ],
        "metadata": {"pattern": "mem_retrieval", "target_id": true_doc_id},
    }


def generate_mem_indexing(target_doc: Dict) -> Dict:
    """mem_indexing: zero-shot doc -> doc_id (parametric memory).

    [MEM_INDEX] Doc: {text} -> Target: {doc_id}
    """
    doc_id = target_doc["doc_id"]
    user_content = f"[MEM_INDEX] Doc: {target_doc['text']} -> Target:"
    return {
        "conversations": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": doc_id},
        ],
        "metadata": {"pattern": "mem_indexing", "target_id": doc_id},
    }


def generate_ctx_pos_front(
    target_query: Dict, all_docs: List, n: int
) -> Optional[Dict]:
    """ctx_pos_front: CTX_SEARCH with positive doc placed first in context.

    Answer: {true_doc_id}
    """
    true_doc_id = target_query["doc_id"]
    pos_doc = next((d for d in all_docs if d["doc_id"] == true_doc_id), None)
    if not pos_doc:
        return None

    neg_pool = [d for d in all_docs if d["doc_id"] != true_doc_id]
    if len(neg_pool) < n - 1:
        return None

    neg_docs = random.sample(neg_pool, n - 1)
    context_docs = [pos_doc] + neg_docs
    user_content, answer = _build_ctx_search(context_docs, target_query, true_doc_id)
    return {
        "conversations": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer},
        ],
        "metadata": {"pattern": "ctx_pos_front", "target_id": true_doc_id},
    }


def generate_ctx_pos_back(target_query: Dict, all_docs: List, n: int) -> Optional[Dict]:
    """ctx_pos_back: CTX_SEARCH with positive doc placed last in context.

    Answer: {true_doc_id}
    """
    true_doc_id = target_query["doc_id"]
    pos_doc = next((d for d in all_docs if d["doc_id"] == true_doc_id), None)
    if not pos_doc:
        return None

    neg_pool = [d for d in all_docs if d["doc_id"] != true_doc_id]
    if len(neg_pool) < n - 1:
        return None

    neg_docs = random.sample(neg_pool, n - 1)
    context_docs = neg_docs + [pos_doc]
    user_content, answer = _build_ctx_search(context_docs, target_query, true_doc_id)
    return {
        "conversations": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer},
        ],
        "metadata": {"pattern": "ctx_pos_back", "target_id": true_doc_id},
    }


def generate_ctx_noise(target_query: Dict, all_docs: List, n: int) -> Optional[Dict]:
    """ctx_noise: CTX_SEARCH with no positive doc in context.

    Answer: [NO_MATCH] {true_doc_id}
    """
    true_doc_id = target_query["doc_id"]
    neg_pool = [d for d in all_docs if d["doc_id"] != true_doc_id]
    if len(neg_pool) < n:
        return None

    context_docs = random.sample(neg_pool, n)
    answer = f"[NO_MATCH] {true_doc_id}"
    user_content, answer = _build_ctx_search(context_docs, target_query, answer)
    return {
        "conversations": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer},
        ],
        "metadata": {"pattern": "ctx_noise", "target_id": true_doc_id},
    }


# ---------------------------------------------------------------------------
# Subsample helper
# ---------------------------------------------------------------------------


def subsample(examples: List, target_n: int) -> List:
    if len(examples) <= target_n:
        return examples
    return random.sample(examples, target_n)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def process_and_save(
    train_docs, _train_queries, dataset, output_path, n_shot, type_ratios
):
    """Generates all pattern types with configurable ratios and saves to JSONL."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    docs = [s for s in dataset if s["operation"] == "indexing"]
    queries = [s for s in dataset if s["operation"] == "query"]

    print(f"Processing and saving to: {output_path}")

    mem_retrieval_examples = []
    ctx_pos_front_examples = []
    ctx_pos_back_examples = []
    ctx_noise_examples = []

    for q in tqdm(queries, desc="Generating query examples"):
        mem_retrieval_examples.append(generate_mem_retrieval(q))

        ex = generate_ctx_pos_front(q, train_docs, n_shot)
        if ex:
            ctx_pos_front_examples.append(ex)

        ex = generate_ctx_pos_back(q, train_docs, n_shot)
        if ex:
            ctx_pos_back_examples.append(ex)

        ex = generate_ctx_noise(q, train_docs, n_shot)
        if ex:
            ctx_noise_examples.append(ex)

    mem_indexing_examples = [
        generate_mem_indexing(d) for d in tqdm(docs, desc="Generating doc examples")
    ]

    # Subsample each type according to ratios
    pool = {
        "mem_retrieval": mem_retrieval_examples,
        "mem_indexing": mem_indexing_examples,
        "ctx_pos_front": ctx_pos_front_examples,
        "ctx_pos_back": ctx_pos_back_examples,
        "ctx_noise": ctx_noise_examples,
    }
    total_raw = sum(len(v) for v in pool.values())
    subsampled = {}
    for name, examples in pool.items():
        ratio = type_ratios.get(name, 1.0)
        target_n = max(1, int(total_raw * ratio)) if examples else 0
        subsampled[name] = subsample(examples, target_n)

    all_examples = [ex for exs in subsampled.values() for ex in exs]
    random.shuffle(all_examples)

    with open(output_path, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Saved {len(all_examples)} examples.")
    for name, exs in subsampled.items():
        print(f"  - {name:<16} {len(exs)}")


@hydra.main(config_path="../configs", config_name="get_icl_axolotl", version_base=None)
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

    print("\nConversion complete for all ICL patterns.")


if __name__ == "__main__":
    main()
