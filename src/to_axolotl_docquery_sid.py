import json
import random
import re
import os
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm
from datasets import load_dataset
import hydra
from omegaconf import DictConfig

# ---------------------------------------------------------------------------
# Shuffled ID helpers
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"<\|(\w+)_\d+\|>")


def _parse_dims(doc_id: str) -> List[str]:
    """Return ordered list of dimension names in a doc_id string."""
    return [m.group(1) for m in _TOKEN_RE.finditer(doc_id)]


def _apply_shuffle(doc_id: str, rand_vals: Dict[str, int]) -> str:
    """Replace each <|dN_xxx|> token's number with rand_vals[dN]."""
    def _replace(m):
        dim = m.group(1)
        return f"<|{dim}_{rand_vals[dim]}|>"
    return _TOKEN_RE.sub(_replace, doc_id)


def make_shuffled_id_map(doc_ids: List[str]) -> Dict[str, str]:
    """Build a per-sample mapping: original_doc_id -> shuffled_doc_id.

    For each dimension (d0, d1, d2, ...) present across all docs,
    assigns n unique random integers (0-255) so no two docs share the
    same value in any dimension.
    """
    n = len(doc_ids)

    # Collect all dimensions across docs
    all_dims: set = set()
    for doc_id in doc_ids:
        all_dims.update(_parse_dims(doc_id))

    # Per dimension: n unique values sampled from 0-255
    dim_pools = {dim: random.sample(range(256), min(n, 256)) for dim in all_dims}

    shuffled_map: Dict[str, str] = {}
    for i, doc_id in enumerate(doc_ids):
        rand_vals = {dim: dim_pools[dim][i] for dim in all_dims}
        shuffled_map[doc_id] = _apply_shuffle(doc_id, rand_vals)

    return shuffled_map


# ---------------------------------------------------------------------------
# Pattern generators
# ---------------------------------------------------------------------------

def generate_mem_retrieval(target_query: Dict) -> Dict:
    """mem_retrieval: query -> doc_id (parametric memory, no shuffling).

    [MEM_SEARCH]
    Query: {query}
    Answer: {true_doc_id}
    """
    true_doc_id = target_query["doc_id"]
    user_content = (
        "[MEM_SEARCH]\n"
        f"Query: {target_query['text']}\n"
        "Answer:"
    )
    return {
        "conversations": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": true_doc_id},
        ],
        "metadata": {"pattern": "mem_retrieval", "target_id": true_doc_id},
    }


def generate_mem_indexing(target_doc: Dict) -> Dict:
    """mem_indexing: doc -> doc_id (parametric memory, no shuffling).

    [MEM_SEARCH]
    Content: {doc}
    Answer: {true_doc_id}
    """
    doc_id = target_doc["doc_id"]
    user_content = (
        "[MEM_SEARCH]\n"
        f"Content: {target_doc['text']}\n"
        "Answer:"
    )
    return {
        "conversations": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": doc_id},
        ],
        "metadata": {"pattern": "mem_indexing", "target_id": doc_id},
    }


def generate_ctx_match(
    target_query: Dict, all_docs: List, doc_to_queries: Dict, n_docs: int = 3
) -> Optional[Dict]:
    """ctx_match: CTX_SEARCH with target doc INCLUDED in Document Base.

    Doc IDs are shuffled per-sample so the model must use context.
    Answer: {shuffled_id of target doc}

    [CTX_SEARCH]
    [Document Base]
    DocID: {sid} | Content: {text}   (n_docs entries, shuffled order)

    [QA Samples]
    Query: {q} | DocID: {sid}        (n_docs-1 demonstrations)

    [Target Task]
    Query: {target_query}
    Answer: {target_sid}
    """
    true_doc_id = target_query["doc_id"]

    pos_doc = next((d for d in all_docs if d["doc_id"] == true_doc_id), None)
    if not pos_doc:
        return None

    eligible_neg_docs = [
        d for d in all_docs
        if d["doc_id"] != true_doc_id and d["doc_id"] in doc_to_queries
    ]
    if len(eligible_neg_docs) < n_docs - 1:
        return None

    neg_docs = random.sample(eligible_neg_docs, n_docs - 1)

    context_docs = neg_docs + [pos_doc]
    random.shuffle(context_docs)

    # Build shuffled ID map for all context docs
    sid_map = make_shuffled_id_map([d["doc_id"] for d in context_docs])

    doc_base_lines = [
        f"DocID: {sid_map[doc['doc_id']]} | Content: {doc['text']}"
        for doc in context_docs
    ]
    qa_pairs = [
        (random.choice(doc_to_queries[doc["doc_id"]]), sid_map[doc["doc_id"]])
        for doc in neg_docs
    ]
    random.shuffle(qa_pairs)
    qa_lines = [f"Query: {q['text']} | DocID: {sid}" for q, sid in qa_pairs]

    target_sid = sid_map[true_doc_id]
    user_content = (
        "[CTX_SEARCH]\n"
        "[Document Base]\n"
        + "\n".join(doc_base_lines)
        + "\n\n[QA Samples]\n"
        + "\n".join(qa_lines)
        + "\n\n[Target Task]\n"
        f"Query: {target_query['text']}\n"
        "Answer:"
    )
    return {
        "conversations": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": target_sid},
        ],
        "metadata": {
            "pattern": "ctx_match",
            "target_id": true_doc_id,
            "shuffled_id": target_sid,
        },
    }


def generate_ctx_nomatch(
    target_query: Dict, all_docs: List, doc_to_queries: Dict, n_docs: int = 3
) -> Optional[Dict]:
    """ctx_nomatch: CTX_SEARCH with target doc NOT in Document Base.

    Context docs use shuffled IDs; target doc is absent so the model
    must recall from parametric memory.
    Answer: [NO_MATCH] {true_doc_id}  (original ID, not shuffled)
    """
    true_doc_id = target_query["doc_id"]

    eligible_for_qa = [
        d for d in all_docs
        if d["doc_id"] != true_doc_id and d["doc_id"] in doc_to_queries
    ]
    if len(eligible_for_qa) < n_docs:
        return None

    sampled = random.sample(eligible_for_qa, n_docs)
    qa_docs = sampled[: n_docs - 1]
    noise_doc = sampled[n_docs - 1]

    context_docs = qa_docs + [noise_doc]
    random.shuffle(context_docs)

    sid_map = make_shuffled_id_map([d["doc_id"] for d in context_docs])

    doc_base_lines = [
        f"DocID: {sid_map[doc['doc_id']]} | Content: {doc['text']}"
        for doc in context_docs
    ]
    qa_pairs = [
        (random.choice(doc_to_queries[doc["doc_id"]]), sid_map[doc["doc_id"]])
        for doc in qa_docs
    ]
    random.shuffle(qa_pairs)
    qa_lines = [f"Query: {q['text']} | DocID: {sid}" for q, sid in qa_pairs]

    user_content = (
        "[CTX_SEARCH]\n"
        "[Document Base]\n"
        + "\n".join(doc_base_lines)
        + "\n\n[QA Samples]\n"
        + "\n".join(qa_lines)
        + "\n\n[Target Task]\n"
        f"Query: {target_query['text']}\n"
        "Answer:"
    )
    return {
        "conversations": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": f"[NO_MATCH] {true_doc_id}"},
        ],
        "metadata": {"pattern": "ctx_nomatch", "target_id": true_doc_id},
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def subsample(examples: List, target_n: int) -> List:
    if len(examples) <= target_n:
        return examples
    return random.sample(examples, target_n)


def process_and_save(train_docs, train_queries, dataset, output_path, n_shot, type_ratios):
    """Generates all 4 format types with shuffled IDs for CTX patterns."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    docs = [s for s in dataset if s["operation"] == "indexing"]
    queries = [s for s in dataset if s["operation"] == "query"]

    doc_to_queries: Dict[str, List] = {}
    for q in train_queries:
        doc_to_queries.setdefault(q["doc_id"], []).append(q)

    print(f"Processing and saving to: {output_path}")

    mem_retrieval_examples = []
    ctx_match_examples = []
    ctx_nomatch_examples = []

    for q in tqdm(queries, desc="Generating query examples"):
        mem_retrieval_examples.append(generate_mem_retrieval(q))

        ex = generate_ctx_match(q, train_docs, doc_to_queries, n_docs=n_shot)
        if ex:
            ctx_match_examples.append(ex)

        ex = generate_ctx_nomatch(q, train_docs, doc_to_queries, n_docs=n_shot)
        if ex:
            ctx_nomatch_examples.append(ex)

    mem_indexing_examples = [
        generate_mem_indexing(d)
        for d in tqdm(docs, desc="Generating doc examples")
    ]

    pool = {
        "mem_retrieval": mem_retrieval_examples,
        "mem_indexing":  mem_indexing_examples,
        "ctx_match":     ctx_match_examples,
        "ctx_nomatch":   ctx_nomatch_examples,
    }
    n_queries = len(queries)
    print(f"  queries={n_queries}  docs={len(docs)}  raw: " +
          "  ".join(f"{k}={len(v)}" for k, v in pool.items()))
    subsampled = {}
    for name, examples in pool.items():
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


@hydra.main(config_path="../configs", config_name="get_docquery_sid_axolotl", version_base=None)
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
        ratios = dict(cfg.test_type_ratios) if split == "test" and hasattr(cfg, "test_type_ratios") else dict(cfg.type_ratios)
        process_and_save(docs_pool, queries_pool, dataset[split], out_file, cfg.n_shot, ratios)

    print("\nConversion complete for all format types.")


if __name__ == "__main__":
    main()
