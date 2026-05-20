"""
Run:
  python -m src.build_test_docquery_v4 n_shot=100 n_examples=3
"""

import json
import os
import random
from typing import Dict, Iterable, List, Optional, Tuple

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from .to_axolotl_docquery_position_template_v4 import _build_answer


def _alias_item(row: Dict) -> Dict:
    if "item" not in row:
        row["item"] = row["doc_id"]
    row["item"] = row["item"].lower()
    for hn in row.get("hard_negatives") or []:
        if "item" not in hn:
            hn["item"] = hn["doc_id"]
        hn["item"] = hn["item"].lower()
    return row


def _iter_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_split(path: str) -> Tuple[List[Dict], List[Dict]]:
    docs: List[Dict] = []
    queries: List[Dict] = []
    for row in tqdm(_iter_jsonl(path), desc=f"read {os.path.basename(path)}"):
        row = _alias_item(row)
        operation = row.get("operation")
        if operation == "indexing":
            docs.append(row)
        elif operation == "query":
            queries.append(row)
    return docs, queries


def _build_ctx_prompt(
    context_docs: List[Dict],
    query_text: str,
    examples: Optional[List[Dict]] = None,
) -> str:
    doc_blocks = [
        f"Document {i + 1}\nText: {doc['text']}\nIdentifier: {doc['item']}"
        for i, doc in enumerate(context_docs)
    ]
    example_block = ""
    if examples:
        ex_lines = [
            f"Example {i + 1}:\nQuery: {ex['text']}\nAnswer: {ex['item']}"
            for i, ex in enumerate(examples)
        ]
        example_block = "\n\n".join(ex_lines) + "\n\n"

    return (
        "## Documents\n"
        + "\n\n".join(doc_blocks)
        + "\n\n\n"
        + "## Task\n"
        + example_block
        + f"Query: {query_text}\n"
        + "Answer:"
    )


def generate_context_dependent(
    target_query: Dict,
    n_shot: int,
    doc_id_to_doc: Dict[str, Dict],
    eligible_docs: List[Dict],
    examples: List[Dict],
    retrieval_hard_neg: bool = False,
) -> Optional[Dict]:
    true_doc_id = target_query["doc_id"]
    true_item = target_query["item"]
    pos_doc = doc_id_to_doc.get(true_doc_id)
    if not pos_doc:
        return None

    if retrieval_hard_neg:
        eligible_neg = [
            d
            for d in pos_doc.get("hard_negatives", [])
            if d["doc_id"] != true_doc_id
            and str(d.get("item", d.get("doc_id", ""))).lower() != true_item
        ]
    else:
        eligible_neg = [
            d
            for d in eligible_docs
            if d["doc_id"] != true_doc_id
            and str(d.get("item", d.get("doc_id", ""))).lower() != true_item
        ]
    if len(eligible_neg) < n_shot - 1:
        return None
    if retrieval_hard_neg:
        neg_docs = eligible_neg[: n_shot - 1]
    else:
        neg_docs = random.sample(eligible_neg, n_shot - 1)

    target_position = random.randint(0, n_shot - 1)
    context_docs = list(neg_docs)
    context_docs.insert(target_position, pos_doc)

    return {
        "conversations": [
            {
                "role": "user",
                "content": _build_ctx_prompt(
                    context_docs, target_query["text"], examples
                ),
            },
            {"role": "assistant", "content": _build_answer(true_item, copy=True)},
        ],
        "metadata": {
            "pattern": "context_dependent",
            "target_id": true_item,
            "target_position": target_position,
        },
    }


def generate_all_noise(
    target_query: Dict,
    n_shot: int,
    eligible_docs: List[Dict],
    examples: List[Dict],
    doc_id_to_doc: Optional[Dict[str, Dict]] = None,
    retrieval_hard_neg: bool = False,
) -> Optional[Dict]:
    true_doc_id = target_query["doc_id"]
    true_item = target_query["item"]
    if retrieval_hard_neg:
        pos_doc = doc_id_to_doc.get(true_doc_id) if doc_id_to_doc else None
        if not pos_doc:
            return None
        eligible_neg = [
            d
            for d in pos_doc.get("hard_negatives", [])
            if d["doc_id"] != true_doc_id
            and str(d.get("item", d.get("doc_id", ""))).lower() != true_item
        ]
    else:
        eligible_neg = [
            d
            for d in eligible_docs
            if d["doc_id"] != true_doc_id
            and str(d.get("item", d.get("doc_id", ""))).lower() != true_item
        ]
    if len(eligible_neg) < n_shot:
        return None
    if retrieval_hard_neg:
        neg_docs = eligible_neg[:n_shot]
    else:
        neg_docs = random.sample(eligible_neg, n_shot)

    return {
        "conversations": [
            {
                "role": "user",
                "content": _build_ctx_prompt(neg_docs, target_query["text"], examples),
            },
            {"role": "assistant", "content": _build_answer(true_item, copy=False)},
        ],
        "metadata": {"pattern": "all_noise", "target_id": true_item},
    }


def _sample_examples(
    example_pool: List[Dict], target_query: Dict, k: int
) -> List[Dict]:
    eligible = [x for x in example_pool if x["doc_id"] != target_query["doc_id"]]
    if len(eligible) < k:
        return eligible
    return random.sample(eligible, k)


def process_and_save(
    queries: List[Dict],
    docs_pool: List[Dict],
    example_pool: List[Dict],
    output_path: str,
    n_shot: int,
    n_examples: int,
    retrieval_hard_neg: bool = False,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc_id_to_doc = {d["doc_id"]: d for d in docs_pool}
    if retrieval_hard_neg:
        n_with_hn = sum(1 for d in docs_pool if d.get("hard_negatives"))
        print(
            f"retrieval_hard_neg=True ({n_with_hn}/{len(docs_pool)} docs have hard_negatives)"
        )

    n_saved = 0
    n_skipped_cd = n_skipped_an = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for q in tqdm(queries, desc=f"build {os.path.basename(output_path)}"):
            examples = _sample_examples(example_pool, q, n_examples)
            cd = generate_context_dependent(
                q, n_shot, doc_id_to_doc, docs_pool, examples, retrieval_hard_neg
            )
            an = generate_all_noise(
                q, n_shot, docs_pool, examples, doc_id_to_doc, retrieval_hard_neg
            )

            to_write: List[Dict] = []
            if cd:
                to_write.append(cd)
            else:
                n_skipped_cd += 1
            if an:
                to_write.append(an)
            else:
                n_skipped_an += 1

            random.shuffle(to_write)
            for ex in to_write:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                n_saved += 1
    print(
        f"Saved {n_saved} -> {output_path}  "
        f"(skipped: cd={n_skipped_cd}, all_noise={n_skipped_an})"
    )


@hydra.main(
    config_path="../configs",
    config_name="build_test_docquery_v4",
    version_base=None,
)
def main(cfg: DictConfig):
    random.seed(cfg.get("seed", 42))

    n_shot = int(cfg.n_shot)
    n_examples = int(cfg.get("n_examples", 3))
    splits = cfg.get("splits", ["test", "icl_test"])
    retrieval_hard_neg = bool(cfg.get("retrieval_hard_neg", False))
    debug_max_queries = cfg.get("debug_max_queries")
    debug_max_queries = int(debug_max_queries) if debug_max_queries else None

    train_docs: Optional[List[Dict]] = None
    train_queries: Optional[List[Dict]] = None
    test_queries: Optional[List[Dict]] = None
    icl_test_docs: Optional[List[Dict]] = None
    icl_test_queries: Optional[List[Dict]] = None

    if any(split in splits for split in ["train", "test"]):
        train_docs, train_queries = _load_split(f"{cfg.input_path}/train.jsonl")

    if "test" in splits:
        _, test_queries = _load_split(f"{cfg.input_path}/test.jsonl")

    if "icl_test" in splits:
        icl_test_docs, icl_test_queries = _load_split(
            f"{cfg.input_path}/icl_test.jsonl"
        )

    if debug_max_queries:
        if train_queries is not None:
            train_queries = train_queries[:debug_max_queries]
        if test_queries is not None:
            test_queries = test_queries[:debug_max_queries]
        if icl_test_queries is not None:
            icl_test_queries = icl_test_queries[:debug_max_queries]
        print(f"Debug mode: limiting query pools to first {debug_max_queries}")

    print(
        f"pools: train_docs={len(train_docs) if train_docs is not None else 0}  "
        f"train_q={len(train_queries) if train_queries is not None else 0}  "
        f"test_q={len(test_queries) if test_queries is not None else 0}  "
        f"icl_test_docs={len(icl_test_docs) if icl_test_docs is not None else 0}  "
        f"icl_test_q={len(icl_test_queries) if icl_test_queries is not None else 0}"
    )

    if "train" in splits:
        assert train_docs is not None and train_queries is not None
        process_and_save(
            queries=train_queries,
            docs_pool=train_docs,
            example_pool=train_queries,
            output_path=f"{cfg.output_dir}/train_{n_shot}shot.jsonl",
            n_shot=n_shot,
            n_examples=n_examples,
            retrieval_hard_neg=retrieval_hard_neg,
        )
    if "test" in splits:
        assert train_docs is not None and test_queries is not None
        process_and_save(
            queries=test_queries,
            docs_pool=train_docs,
            example_pool=test_queries,
            output_path=f"{cfg.output_dir}/test_{n_shot}shot.jsonl",
            n_shot=n_shot,
            n_examples=n_examples,
            retrieval_hard_neg=retrieval_hard_neg,
        )
    if "icl_test" in splits:
        assert icl_test_docs is not None and icl_test_queries is not None
        process_and_save(
            queries=icl_test_queries,
            docs_pool=icl_test_docs,
            example_pool=icl_test_queries,
            output_path=f"{cfg.output_dir}/icl_test_{n_shot}shot.jsonl",
            n_shot=n_shot,
            n_examples=n_examples,
            retrieval_hard_neg=retrieval_hard_neg,
        )


if __name__ == "__main__":
    main()
