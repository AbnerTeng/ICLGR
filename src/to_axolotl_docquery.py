import json
import random
import os
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional

from tqdm import tqdm
from datasets import load_dataset
import hydra
from omegaconf import DictConfig


def generate_mem_retrieval(target_query: Dict) -> Dict:
    """mem_retrieval: query -> doc_id (retrieval, parametric memory).

    [MEM_SEARCH]
    Query: {query}
    Answer: {true_doc_id}
    """
    true_doc_id = target_query["doc_id"]
    user_content = f"[MEM_SEARCH]\nQuery: {target_query['text']}\nAnswer:"
    return {
        "conversations": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": true_doc_id},
        ],
        "metadata": {"pattern": "mem_retrieval", "target_id": true_doc_id},
    }


def generate_mem_indexing(target_doc: Dict) -> Dict:
    """mem_indexing: doc -> doc_id (indexing, parametric memory).

    [MEM_SEARCH]
    Content: {doc}
    Answer: {true_doc_id}
    """
    doc_id = target_doc["doc_id"]
    user_content = f"[MEM_SEARCH]\nContent: {target_doc['text']}\nAnswer:"
    return {
        "conversations": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": doc_id},
        ],
        "metadata": {"pattern": "mem_indexing", "target_id": doc_id},
    }


def generate_ctx_match(
    target_query: Dict,
    all_docs: List,
    doc_to_queries: Dict,
    n_docs: int = 3,
    doc_id_to_doc: Optional[Dict] = None,
    eligible_docs: Optional[List] = None,
) -> Optional[Dict]:
    """ctx_match: CTX_SEARCH with target doc INCLUDED in Document Base.

    Model retrieves target doc_id from context.
    Answer: {true_doc_id}

    [CTX_SEARCH]
    [Document Base]
    DocID: {id} | Content: {text}   (n_docs entries, shuffled)

    [QA Samples]
    Query: {q} | DocID: {id}        (n_docs-1 demonstrations for neg docs)

    [Target Task]
    Query: {target_query}
    Answer: {true_doc_id}
    """
    true_doc_id = target_query["doc_id"]

    if doc_id_to_doc is not None:
        pos_doc = doc_id_to_doc.get(true_doc_id)
    else:
        pos_doc = next((d for d in all_docs if d["doc_id"] == true_doc_id), None)
    if not pos_doc:
        return None

    pool = (
        eligible_docs
        if eligible_docs is not None
        else [d for d in all_docs if d["doc_id"] in doc_to_queries]
    )
    eligible_neg_docs = [d for d in pool if d["doc_id"] != true_doc_id]
    if len(eligible_neg_docs) < n_docs - 1:
        return None

    neg_docs = random.sample(eligible_neg_docs, n_docs - 1)

    # Shuffle Document Base so target position is not predictable
    context_docs = neg_docs + [pos_doc]
    random.shuffle(context_docs)

    doc_base_lines = [
        f"DocID: {doc['doc_id']} | Content: {doc['text']}" for doc in context_docs
    ]
    qa_pairs = [
        (random.choice(doc_to_queries[doc["doc_id"]]), doc["doc_id"])
        for doc in neg_docs
    ]
    random.shuffle(qa_pairs)
    qa_lines = [f"Query: {q['text']} | DocID: {doc_id}" for q, doc_id in qa_pairs]

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
            {"role": "assistant", "content": true_doc_id},
        ],
        "metadata": {"pattern": "ctx_match", "target_id": true_doc_id},
    }


def generate_ctx_nomatch(
    target_query: Dict,
    all_docs: List,
    doc_to_queries: Dict,
    n_docs: int = 3,
    eligible_docs: Optional[List] = None,
) -> Optional[Dict]:
    """ctx_nomatch: CTX_SEARCH with target doc NOT in Document Base.

    Model must recognize the doc is absent and recall from parametric memory.
    Answer: [NO_MATCH] {true_doc_id}
    """
    true_doc_id = target_query["doc_id"]

    pool = (
        eligible_docs
        if eligible_docs is not None
        else [d for d in all_docs if d["doc_id"] in doc_to_queries]
    )
    # Need n_docs docs total: n_docs-1 have queries for QA demos, 1 is noise
    eligible_for_qa = [
        d
        for d in all_docs
        if d["doc_id"] != true_doc_id and d["doc_id"] in doc_to_queries
    ]
    if len(eligible_for_qa) < n_docs:
        return None

    sampled = random.sample(eligible_for_qa, n_docs)
    qa_docs = sampled[: n_docs - 1]
    noise_doc = sampled[n_docs - 1]

    context_docs = qa_docs + [noise_doc]
    random.shuffle(context_docs)

    doc_base_lines = [
        f"DocID: {doc['doc_id']} | Content: {doc['text']}" for doc in context_docs
    ]
    qa_pairs = [
        (random.choice(doc_to_queries[doc["doc_id"]]), doc["doc_id"]) for doc in qa_docs
    ]
    random.shuffle(qa_pairs)
    qa_lines = [f"Query: {q['text']} | DocID: {doc_id}" for q, doc_id in qa_pairs]

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


def _process_query(args):
    """Worker function for multiprocessing."""
    q, train_docs, doc_to_queries, n_shot, doc_id_to_doc, eligible_docs = args
    mem = generate_mem_retrieval(q)
    ctx_m = generate_ctx_match(
        q,
        train_docs,
        doc_to_queries,
        n_docs=n_shot,
        doc_id_to_doc=doc_id_to_doc,
        eligible_docs=eligible_docs,
    )
    ctx_nm = generate_ctx_nomatch(
        q, train_docs, doc_to_queries, n_docs=n_shot, eligible_docs=eligible_docs
    )
    return mem, ctx_m, ctx_nm


def subsample(examples: List, target_n: int) -> List:
    if len(examples) <= target_n:
        return examples
    return random.sample(examples, target_n)


def process_and_save(
    train_docs, train_queries, dataset, output_path, n_shot, type_ratios
):
    """Generates all 4 format types with configurable ratios and saves to JSONL."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    docs = [s for s in dataset if s["operation"] == "indexing"]
    queries = [s for s in dataset if s["operation"] == "query"]

    # Build doc_id -> queries mapping for ICL demonstrations
    doc_to_queries: Dict[str, List] = {}
    for q in train_queries:
        doc_to_queries.setdefault(q["doc_id"], []).append(q)

    # Pre-compute lookups to avoid O(N) scans inside each query iteration
    doc_id_to_doc: Dict[str, Dict] = {d["doc_id"]: d for d in train_docs}
    eligible_docs: List = [d for d in train_docs if d["doc_id"] in doc_to_queries]

    print(f"Processing and saving to: {output_path}")

    mem_retrieval_examples = []
    ctx_match_examples = []
    ctx_nomatch_examples = []

    n_workers = max(1, cpu_count() - 1)
    args_list = [
        (q, train_docs, doc_to_queries, n_shot, doc_id_to_doc, eligible_docs)
        for q in queries
    ]

    with Pool(processes=n_workers) as pool:
        results = list(
            tqdm(
                pool.imap(_process_query, args_list, chunksize=64),
                total=len(queries),
                desc="Generating query examples",
            )
        )

    for mem, ctx_m, ctx_nm in results:
        mem_retrieval_examples.append(mem)
        if ctx_m:
            ctx_match_examples.append(ctx_m)
        if ctx_nm:
            ctx_nomatch_examples.append(ctx_nm)

    mem_indexing_examples = [
        generate_mem_indexing(d) for d in tqdm(docs, desc="Generating doc examples")
    ]

    # Subsample each type according to ratios
    pool = {
        "mem_retrieval": mem_retrieval_examples,
        "mem_indexing": mem_indexing_examples,
        "ctx_match": ctx_match_examples,
        "ctx_nomatch": ctx_nomatch_examples,
    }
    n_queries = len(queries)
    print(
        f"  queries={n_queries}  docs={len(docs)}  raw: "
        + "  ".join(f"{k}={len(v)}" for k, v in pool.items())
    )
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


@hydra.main(
    config_path="../configs", config_name="get_docquery_axolotl", version_base=None
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
