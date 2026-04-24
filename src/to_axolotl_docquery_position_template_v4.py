"""
Stage 2: three pattern types
    → trains ICL ability

-----------------
Type 1 — Context-dependent  (doc_pos_front / doc_pos_back)
    The relevant document appears in the context; model must copy its identifier.

    context: [doc_A→title_A, doc_B→title_B, doc_C→title_C]
    query:   <query related to doc_A>
    output:  [COPY] title_A

    Mix: ~70-80% pseudo queries, ~20-30% real queries.
    Target size: ~30-40k (augmented by pairing the same query with different context orderings).

Type 2 — Context-independent  (all_noise)
    All context documents are unrelated to the query; model must fall back to
    parametric memory and ignore the context.

    context: [doc_X→title_X, doc_Y→title_Y]   ← unrelated
    query:   <some query>
    output:  <title from parametric memory>

    Mix: primarily real queries.
    Target size: ~30-40k.

Type 3 — Empty context  (stage1_retrieval / stage1_indexing)
    Direct replay of Stage 1 examples with no context window.

    context: (empty)
    query:   <query>
    output:  <title>

    Mix: both pseudo and real queries.
    Target size: ~15-20k (sampled from Stage 1 data).

Total: ~75-100k examples across all three types.
"""

import json
import random
import os
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import hydra
from omegaconf import DictConfig


# ── Template builder ──────────────────────────────────────────────────────

def _build_ctx_prompt(context_docs: List[Dict], query_text: str, example: Optional[Dict] = None) -> str:
    doc_blocks = []
    for i, doc in enumerate(context_docs):
        doc_blocks.append(
            f"Document {i + 1}\n"
            f"Text: {doc['text']}\n"
            f"Identifier: {doc['item']}"
        )
    example_block = ""
    if example:
        example_block = f"Example:\nQuery: {example['text']}\nAnswer: {example['item']}\n\n"
    return (
        "## Documents\n"
        + "\n\n".join(doc_blocks) + "\n\n\n"
        + "## Task\n"
        + example_block
        + f"Query: {query_text}\n"
        + "Answer:"
    )


def _build_answer(item: str, copy: bool = False) -> str:
    if copy:
        return f"[COPY] {item}"
    return item


# ── Pattern generators ────────────────────────────────────────────────────

def generate_zero_shot(target_query: Dict, example: Optional[Dict] = None) -> Dict:
    true_item = target_query["item"]
    example_block = ""
    if example:
        example_block = f"Example:\nQuery: {example['text']}\nAnswer: {example['item']}\n\n"
    user_content = (
        "## Task\n"
        + example_block
        + f"Query: {target_query['text']}\n"
        + "Answer:"
    )
    return {
        "conversations": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": _build_answer(true_item, copy=False)},
        ],
        "metadata": {"pattern": "stage1_retrieval", "target_id": true_item},
    }


def generate_zero_shot_indexing(target_doc: Dict) -> Dict:
    true_item = target_doc["item"]
    user_content = (
        "## Task\n"
        f"Document: {target_doc['text']}\n"
        "Answer:"
    )
    return {
        "conversations": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": _build_answer(true_item, copy=False)},
        ],
        "metadata": {"pattern": "stage1_indexing", "target_id": true_item},
    }


def _pick_neg_docs(
    n_needed: int,
    true_doc_id: str,
    pos_doc: Optional[Dict],
    eligible_docs: Optional[List],
    use_retrieval_hn: bool,
) -> Optional[List[Dict]]:
    """Pick n_needed negatives. If use_retrieval_hn, take from pos_doc['hard_negatives']."""
    if use_retrieval_hn:
        hn = (pos_doc or {}).get("hard_negatives") or []
        hn = [d for d in hn if d.get("doc_id") != true_doc_id]
        if len(hn) < n_needed:
            return None
        return random.sample(hn, n_needed)
    eligible_neg = [d for d in (eligible_docs or []) if d["doc_id"] != true_doc_id]
    if len(eligible_neg) < n_needed:
        return None
    return random.sample(eligible_neg, n_needed)


def generate_doc_pos_front(
    target_query: Dict,
    n_docs: int = 3,
    doc_id_to_doc: Optional[Dict] = None,
    eligible_docs: Optional[List] = None,
    use_retrieval_hn: bool = False,
    example: Optional[Dict] = None,
) -> Optional[Dict]:
    true_doc_id = target_query["doc_id"]
    true_item = target_query["item"]
    pos_doc = doc_id_to_doc.get(true_doc_id) if doc_id_to_doc else None
    if not pos_doc:
        return None

    neg_docs = _pick_neg_docs(n_docs - 1, true_doc_id, pos_doc, eligible_docs, use_retrieval_hn)
    if neg_docs is None:
        return None
    context_docs = [pos_doc] + neg_docs  # target FIRST

    user_content = _build_ctx_prompt(context_docs, target_query["text"], example=example)
    return {
        "conversations": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": _build_answer(true_item, copy=True)},
        ],
        "metadata": {"pattern": "doc_pos_front", "target_id": true_item},
    }


def generate_doc_pos_back(
    target_query: Dict,
    n_docs: int = 3,
    doc_id_to_doc: Optional[Dict] = None,
    eligible_docs: Optional[List] = None,
    use_retrieval_hn: bool = False,
    example: Optional[Dict] = None,
) -> Optional[Dict]:
    true_doc_id = target_query["doc_id"]
    true_item = target_query["item"]
    pos_doc = doc_id_to_doc.get(true_doc_id) if doc_id_to_doc else None
    if not pos_doc:
        return None

    neg_docs = _pick_neg_docs(n_docs - 1, true_doc_id, pos_doc, eligible_docs, use_retrieval_hn)
    if neg_docs is None:
        return None
    context_docs = neg_docs + [pos_doc]  # target LAST

    user_content = _build_ctx_prompt(context_docs, target_query["text"], example=example)
    return {
        "conversations": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": _build_answer(true_item, copy=True)},
        ],
        "metadata": {"pattern": "doc_pos_back", "target_id": true_item},
    }


def generate_all_noise(
    target_query: Dict,
    n_docs: int = 3,
    eligible_docs: Optional[List] = None,
    doc_id_to_doc: Optional[Dict] = None,
    use_retrieval_hn: bool = False,
    example: Optional[Dict] = None,
) -> Optional[Dict]:
    true_doc_id = target_query["doc_id"]
    pos_doc = doc_id_to_doc.get(true_doc_id) if doc_id_to_doc else None

    neg_docs = _pick_neg_docs(n_docs, true_doc_id, pos_doc, eligible_docs, use_retrieval_hn)
    if neg_docs is None:
        return None
    random.shuffle(neg_docs)

    true_item = target_query["item"]

    user_content = _build_ctx_prompt(neg_docs, target_query["text"], example=example)
    return {
        "conversations": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": _build_answer(true_item, copy=False)},
        ],
        "metadata": {
            "pattern": "all_noise",
            "target_id": true_item,
        },
    }


# ── BM25 Hard Negative ───────────────────────────────────────────────────

def build_hard_negative_map(
    queries: List[Dict],
    docs: List[Dict],
    top_k: int = 20,
) -> Dict[str, List[int]]:
    """Build a mapping from query doc_id -> list of hard negative doc indices.

    Uses BM25 to find the most similar (but incorrect) documents for each query.
    Returns indices into the `docs` list.
    """
    from rank_bm25 import BM25Okapi

    print(f"  Building BM25 index over {len(docs)} docs...")
    tokenized_docs = [d["text"].lower().split() for d in docs]
    bm25 = BM25Okapi(tokenized_docs)

    doc_id_to_indices: Dict[str, List[int]] = {}
    for i, d in enumerate(docs):
        doc_id_to_indices.setdefault(d["doc_id"], []).append(i)

    hard_neg_map: Dict[str, List[int]] = {}
    print(f"  Computing hard negatives for {len(queries)} queries (top_k={top_k})...")
    for q in tqdm(queries, desc="BM25 scoring"):
        tokenized_q = q["text"].lower().split()
        scores = bm25.get_scores(tokenized_q)
        true_doc_id = q["doc_id"]
        true_indices = set(doc_id_to_indices.get(true_doc_id, []))
        ranked = np.argsort(scores)[::-1]
        neg_indices = [int(idx) for idx in ranked if idx not in true_indices][:top_k]
        hard_neg_map[q["doc_id"] + "||" + q["text"]] = neg_indices

    return hard_neg_map


def generate_all_noise_hard(
    target_query: Dict,
    n_docs: int,
    docs: List[Dict],
    hard_neg_indices: List[int],
    example: Optional[Dict] = None,
) -> Optional[Dict]:
    """Generate all_noise using pre-computed BM25 hard negatives."""
    if len(hard_neg_indices) < n_docs:
        return None

    selected_indices = random.sample(hard_neg_indices, n_docs)
    neg_docs = [docs[i] for i in selected_indices]

    true_item = target_query["item"]
    user_content = _build_ctx_prompt(neg_docs, target_query["text"], example=example)
    return {
        "conversations": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": _build_answer(true_item, copy=False)},
        ],
        "metadata": {
            "pattern": "all_noise",
            "target_id": true_item,
        },
    }


# ── Multiprocessing worker ────────────────────────────────────────────────

def _process_query(args):
    q, n_shot, doc_id_to_doc, eligible_docs, use_retrieval_hn, example = args
    front = generate_doc_pos_front(
        q, n_docs=n_shot, doc_id_to_doc=doc_id_to_doc,
        eligible_docs=eligible_docs, use_retrieval_hn=use_retrieval_hn, example=example,
    )
    back = generate_doc_pos_back(
        q, n_docs=n_shot, doc_id_to_doc=doc_id_to_doc,
        eligible_docs=eligible_docs, use_retrieval_hn=use_retrieval_hn, example=example,
    )
    noise = generate_all_noise(
        q, n_docs=n_shot, eligible_docs=eligible_docs,
        doc_id_to_doc=doc_id_to_doc, use_retrieval_hn=use_retrieval_hn, example=example,
    )
    return front, back, noise


def _process_noise_only(args):
    q, n_shot, eligible_docs, example = args
    return generate_all_noise(q, n_docs=n_shot, eligible_docs=eligible_docs, example=example)


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
    zs_docs: Optional[List] = None,
    zs_queries: Optional[List] = None,
    oversample: Optional[Dict[str, int]] = None,
    zs_noise: bool = False,
    hard_negative: bool = False,
    hard_negative_k: int = 20,
    retrieval_hard_neg: bool = False,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    queries = list(train_queries)

    doc_id_to_doc: Dict[str, Dict] = {d["doc_id"]: d for d in train_docs}
    eligible_docs: List = list(train_docs)

    print(f"Processing and saving to: {output_path}")

    if retrieval_hard_neg:
        n_with_hn = sum(1 for d in train_docs if d.get("hard_negatives"))
        print(f"  retrieval_hard_neg=True  ({n_with_hn}/{len(train_docs)} docs have hard_negatives)")

    example_pool = queries
    args_list = [
        (q, n_shot, doc_id_to_doc, eligible_docs, retrieval_hard_neg,
         random.choice([x for x in example_pool if x["doc_id"] != q["doc_id"]]))
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
    }
    for front, back, noise in results:
        if front:
            buckets["doc_pos_front"].append(front)
        if back:
            buckets["doc_pos_back"].append(back)
        if noise:
            buckets["all_noise"].append(noise)

    # Replace all_noise with stage1 queries but use train_docs as noise pool (same distribution as test)
    if zs_noise and zs_docs is not None and zs_queries is not None:
        print(f"  Generating all_noise from stage1 queries ({len(zs_queries)} queries), noise docs from train ({len(eligible_docs)} docs)...")
        buckets["all_noise"] = []
        zs_noise_args = [
            (q, n_shot, eligible_docs,
             random.choice([x for x in example_pool if x["doc_id"] != q["doc_id"]]))
            for q in zs_queries
        ]
        with Pool(processes=n_workers) as pool:
            zs_noise_results = list(tqdm(
                pool.imap(_process_noise_only, zs_noise_args, chunksize=256),
                total=len(zs_queries),
                desc="Generating stage1 all_noise",
            ))
        buckets["all_noise"] = [ex for ex in zs_noise_results if ex is not None]
        print(f"  stage1 all_noise: {len(buckets['all_noise'])} examples")

    # Generate hard negative all_noise using BM25 (always use train_docs as noise pool)
    if hard_negative:
        hn_queries = zs_queries if (zs_noise and zs_queries is not None) else queries
        hn_docs = train_docs
        hard_neg_map = build_hard_negative_map(hn_queries, hn_docs, top_k=hard_negative_k)
        print(f"  Generating hard negative all_noise...")
        buckets["all_noise"] = []
        for q in tqdm(hn_queries, desc="Hard negative all_noise"):
            key = q["doc_id"] + "||" + q["text"]
            neg_indices = hard_neg_map.get(key, [])
            eg = random.choice([x for x in example_pool if x["doc_id"] != q["doc_id"]])
            ex = generate_all_noise_hard(q, n_shot, hn_docs, neg_indices, example=eg)
            if ex:
                buckets["all_noise"].append(ex)
        print(f"  hard negative all_noise: {len(buckets['all_noise'])} examples")

    _zs_queries = zs_queries if zs_queries is not None else list(train_queries)
    _zs_docs = zs_docs if zs_docs is not None else train_docs

    stage1_retrieval = [
        generate_zero_shot(
            q,
            example=random.choice([x for x in example_pool if x["doc_id"] != q["doc_id"]]),
        )
        for q in _zs_queries
    ]
    stage1_indexing = [generate_zero_shot_indexing(d) for d in _zs_docs]

    n_queries = len(queries)

    # Handle stage1: ratio is a fraction of the original stage1 data (preserves retrieval:indexing ratio)
    stage1_ratio = type_ratios.pop("stage1", None)
    if stage1_ratio is not None:
        ret_target = max(1, int(len(stage1_retrieval) * stage1_ratio))
        idx_target = max(1, int(len(stage1_indexing) * stage1_ratio))
        buckets["stage1_retrieval"] = subsample(stage1_retrieval, ret_target)
        buckets["stage1_indexing"] = subsample(stage1_indexing, idx_target)
    else:
        buckets["stage1_retrieval"] = stage1_retrieval
        buckets["stage1_indexing"] = stage1_indexing

    print(f"  queries={n_queries}  raw: " +
          "  ".join(f"{k}={len(v)}" for k, v in buckets.items()))

    subsampled: Dict[str, List] = {}
    for name, examples in buckets.items():
        if name not in type_ratios:
            subsampled[name] = examples
            continue
        ratio = type_ratios[name]
        target_n = max(1, int(n_queries * ratio)) if examples else 0
        subsampled[name] = subsample(examples, target_n)

    if oversample:
        for name, factor in oversample.items():
            if name in subsampled and factor > 1:
                original = subsampled[name]
                subsampled[name] = original * factor
                print(f"  oversample {name}: {len(original)} x {factor} = {len(subsampled[name])}")

    all_examples = [ex for exs in subsampled.values() for ex in exs]
    random.shuffle(all_examples)

    with open(output_path, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Saved {len(all_examples)} examples.")
    for name, exs in subsampled.items():
        print(f"  - {name:<22} {len(exs)}")


# ── Entry point ───────────────────────────────────────────────────────────

@hydra.main(config_path="../configs", config_name="get_docquery_position_template_v4", version_base=None)
def main(cfg: DictConfig):
    dataset = load_dataset(
        "json",
        data_files={
            "train": f"{cfg.input_path}/train.jsonl",
            "test": f"{cfg.input_path}/test.jsonl",
            "icl_test": f"{cfg.input_path}/icl_test.jsonl",
        },
    )
    def _alias_item(row):
        if "item" not in row:
            row["item"] = row["doc_id"]
        row["item"] = row["item"].lower()
        for hn in row.get("hard_negatives") or []:
            if "item" not in hn:
                hn["item"] = hn["doc_id"]
            hn["item"] = hn["item"].lower()
        return row

    train_docs = [_alias_item(s) for s in dataset["train"] if s["operation"] == "indexing"]
    train_queries = [_alias_item(s) for s in dataset["train"] if s["operation"] == "query"]

    zs_docs: Optional[List] = None
    zs_queries: Optional[List] = None
    if cfg.get("zs_input_path"):
        print(f"Loading zero-shot data from: {cfg.zs_input_path}")
        zs_ds = load_dataset("json", data_files={"train": cfg.zs_input_path})["train"]
        zs_docs = [s for s in zs_ds if s["operation"] == "indexing"]
        zs_queries = [s for s in zs_ds if s["operation"] == "query"]
        print(f"  zero-shot pool: {len(zs_docs)} docs, {len(zs_queries)} queries")

    debug_max_queries = cfg.get("debug_max_queries")
    if debug_max_queries:
        debug_max_queries = int(debug_max_queries)
        train_queries = train_queries[:debug_max_queries]
        if zs_queries is not None:
            zs_queries = zs_queries[:debug_max_queries]
        if zs_docs is not None:
            zs_docs = zs_docs[:debug_max_queries]
        print(
            f"Debug mode: limiting queries/docs to first {debug_max_queries} "
            f"(train_queries={len(train_queries)}, "
            f"zs_queries={len(zs_queries) if zs_queries is not None else 0}, "
            f"zs_docs={len(zs_docs) if zs_docs is not None else 0})"
        )

    splits: List[str] = ["train"]
    for split in splits:
        if split not in dataset:
            continue

        docs_pool = train_docs
        queries_pool = train_queries

        if split == "icl_test":
            docs_pool = [_alias_item(s) for s in dataset["icl_test"] if s["operation"] == "indexing"]
            queries_pool = [_alias_item(s) for s in dataset["icl_test"] if s["operation"] == "query"]

        split_zs_docs = zs_docs if split == "train" else None
        split_zs_queries = zs_queries if split == "train" else None

        oversample_cfg = dict(cfg.oversample) if cfg.get("oversample") else None
        use_zs_noise = cfg.get("zs_noise", False) and split == "train"
        use_hard_neg = cfg.get("hard_negative", False) and split == "train"
        hard_neg_k = cfg.get("hard_negative_k", 20)
        use_retrieval_hn = cfg.get("retrieval_hard_neg", False) and split == "train"
        if use_hard_neg and use_retrieval_hn:
            raise ValueError("hard_negative (BM25) and retrieval_hard_neg cannot both be true")

        out_file = f"{cfg.output_dir}/{split}_{cfg.n_shot}shot.jsonl"
        process_and_save(
            docs_pool, queries_pool, dataset[split], out_file,
            cfg.n_shot, dict(cfg.type_ratios),
            zs_docs=split_zs_docs, zs_queries=split_zs_queries,
            oversample=oversample_cfg,
            zs_noise=use_zs_noise,
            hard_negative=use_hard_neg,
            hard_negative_k=hard_neg_k,
            retrieval_hard_neg=use_retrieval_hn,
        )

    print("\nConversion complete for all format types.")


if __name__ == "__main__":
    main()
