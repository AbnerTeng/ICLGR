"""
Build few-shot train data with the v4 doc-query template.

This is the train-only companion to src/build_test_docquery_v4.py.  It is
designed for large train.jsonl files, so it only reads the train split instead
of loading test/icl_test as well.

Run:
  python -m src.build_train_docquery_v4
  python -m src.build_train_docquery_v4 n_shot=100 n_examples=3 retrieval_hard_neg=true
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


def _load_train_split(train_path: str) -> Tuple[List[Dict], List[Dict]]:
    docs: List[Dict] = []
    queries: List[Dict] = []
    for row in tqdm(_iter_jsonl(train_path), desc=f"read {os.path.basename(train_path)}"):
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


def _sample_examples(example_pool: List[Dict], target_query: Dict, k: int) -> List[Dict]:
    if not example_pool or k <= 0:
        return []

    examples: List[Dict] = []
    seen_doc_ids = {target_query["doc_id"]}
    max_attempts = max(k * 10, 20)

    for _ in range(max_attempts):
        if len(examples) >= k:
            return examples
        candidate = random.choice(example_pool)
        doc_id = candidate["doc_id"]
        if doc_id in seen_doc_ids:
            continue
        examples.append(candidate)
        seen_doc_ids.add(doc_id)

    for candidate in example_pool:
        if len(examples) >= k:
            break
        doc_id = candidate["doc_id"]
        if doc_id in seen_doc_ids:
            continue
        examples.append(candidate)
        seen_doc_ids.add(doc_id)
    return examples


def _pick_neg_docs(
    n_needed: int,
    true_doc_id: str,
    true_item: str,
    pos_doc: Optional[Dict],
    eligible_docs: List[Dict],
    retrieval_hard_neg: bool,
) -> Optional[List[Dict]]:
    true_item = true_item.lower()

    def is_negative(doc: Dict) -> bool:
        doc_item = doc.get("item") or doc.get("doc_id")
        return doc.get("doc_id") != true_doc_id and str(doc_item).lower() != true_item

    if retrieval_hard_neg:
        hard_negatives = (pos_doc or {}).get("hard_negatives") or []
        eligible_neg = [d for d in hard_negatives if is_negative(d)]
    else:
        eligible_neg = [d for d in eligible_docs if is_negative(d)]

    if len(eligible_neg) < n_needed:
        return None
    if retrieval_hard_neg:
        return eligible_neg[:n_needed]
    return random.sample(eligible_neg, n_needed)


def generate_context_dependent(
    target_query: Dict,
    n_shot: int,
    doc_id_to_doc: Dict[str, Dict],
    eligible_docs: List[Dict],
    examples: List[Dict],
    retrieval_hard_neg: bool,
) -> Optional[Dict]:
    true_doc_id = target_query["doc_id"]
    true_item = target_query["item"]
    pos_doc = doc_id_to_doc.get(true_doc_id)
    if not pos_doc:
        return None

    neg_docs = _pick_neg_docs(
        n_needed=n_shot - 1,
        true_doc_id=true_doc_id,
        true_item=true_item,
        pos_doc=pos_doc,
        eligible_docs=eligible_docs,
        retrieval_hard_neg=retrieval_hard_neg,
    )
    if neg_docs is None:
        return None

    target_position = random.randint(0, n_shot - 1)
    context_docs = list(neg_docs)
    context_docs.insert(target_position, pos_doc)

    return {
        "conversations": [
            {
                "role": "user",
                "content": _build_ctx_prompt(context_docs, target_query["text"], examples),
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
    doc_id_to_doc: Dict[str, Dict],
    eligible_docs: List[Dict],
    examples: List[Dict],
    retrieval_hard_neg: bool,
) -> Optional[Dict]:
    true_doc_id = target_query["doc_id"]
    true_item = target_query["item"]
    pos_doc = doc_id_to_doc.get(true_doc_id)

    neg_docs = _pick_neg_docs(
        n_needed=n_shot,
        true_doc_id=true_doc_id,
        true_item=true_item,
        pos_doc=pos_doc,
        eligible_docs=eligible_docs,
        retrieval_hard_neg=retrieval_hard_neg,
    )
    if neg_docs is None:
        return None

    return {
        "conversations": [
            {
                "role": "user",
                "content": _build_ctx_prompt(neg_docs, target_query["text"], examples),
            },
            {"role": "assistant", "content": _build_answer(true_item, copy=False)},
        ],
        "metadata": {
            "pattern": "all_noise",
            "target_id": true_item,
        },
    }


def process_and_save(
    queries: List[Dict],
    docs_pool: List[Dict],
    output_path: str,
    n_shot: int,
    n_examples: int,
    retrieval_hard_neg: bool = True,
    max_queries: Optional[int] = None,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if max_queries is not None and max_queries > 0:
        queries = queries[:max_queries]
        print(f"Debug mode: limiting train queries to first {len(queries)}")

    doc_id_to_doc = {d["doc_id"]: d for d in docs_pool}
    if retrieval_hard_neg:
        n_with_hn = sum(1 for d in docs_pool if d.get("hard_negatives"))
        print(
            f"retrieval_hard_neg=True "
            f"({n_with_hn}/{len(docs_pool)} docs have hard_negatives)"
        )

    n_saved = 0
    n_skipped_cd = 0
    n_skipped_an = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for q in tqdm(queries, desc=f"build {os.path.basename(output_path)}"):
            examples = _sample_examples(queries, q, n_examples)
            cd = generate_context_dependent(
                q, n_shot, doc_id_to_doc, docs_pool, examples, retrieval_hard_neg
            )
            an = generate_all_noise(
                q, n_shot, doc_id_to_doc, docs_pool, examples, retrieval_hard_neg
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
    config_name="build_train_docquery_v4_100shot_hardneg",
    version_base=None,
)
def main(cfg: DictConfig):
    random.seed(cfg.get("seed", 42))

    input_path = str(cfg.input_path)
    train_path = f"{input_path}/train.jsonl"
    docs_pool, train_queries = _load_train_split(train_path)
    print(f"pools: train_docs={len(docs_pool)}  train_q={len(train_queries)}")

    n_shot = int(cfg.n_shot)
    splits = cfg.get("splits", ["train"])
    if "train" not in splits:
        print(f"No train split requested in splits={list(splits)}")
        return

    process_and_save(
        queries=train_queries,
        docs_pool=docs_pool,
        output_path=f"{cfg.output_dir}/train_{n_shot}shot.jsonl",
        n_shot=n_shot,
        n_examples=int(cfg.get("n_examples", 3)),
        retrieval_hard_neg=bool(cfg.get("retrieval_hard_neg", True)),
        max_queries=cfg.get("debug_max_queries"),
    )


if __name__ == "__main__":
    main()
