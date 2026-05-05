"""
Build few-shot test/icl_test data with v4 template (multi-example block + n docs).

Differences from src/to_axolotl_docquery_position_template_v4.py:
  - Generates evaluation data only (test, icl_test), no train/stage1/hard-neg.
  - Example block supports MULTIPLE demonstrations (Example 1: ... Example N:),
    matching the existing test_3shot.jsonl format.
  - Each query produces two examples: context_dependent (target at random
    position 0..n_shot-1) and all_noise (target NOT in context).

Sourcing rule:
  test     — queries from test.jsonl;     docs sampled from train.jsonl indexing;
             demo examples sampled from test queries (leave-one-out).
  icl_test — queries from icl_test.jsonl; docs sampled from icl_test indexing;
             demo examples sampled from icl_test queries (leave-one-out).

Run:
  python -m src.build_test_docquery_v4 n_shot=10
"""

import json
import os
import random
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional

import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from datasets import load_dataset

from .to_axolotl_docquery_position_template_v4 import _build_answer


def _alias_item(row: Dict) -> Dict:
    if "item" not in row:
        row["item"] = row["doc_id"]
    row["item"] = row["item"].lower()
    return row


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
) -> Optional[Dict]:
    true_doc_id = target_query["doc_id"]
    true_item = target_query["item"]
    pos_doc = doc_id_to_doc.get(true_doc_id)
    if not pos_doc:
        return None

    eligible_neg = [d for d in eligible_docs if d["doc_id"] != true_doc_id]
    if len(eligible_neg) < n_shot - 1:
        return None
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
) -> Optional[Dict]:
    true_doc_id = target_query["doc_id"]
    eligible_neg = [d for d in eligible_docs if d["doc_id"] != true_doc_id]
    if len(eligible_neg) < n_shot:
        return None
    neg_docs = random.sample(eligible_neg, n_shot)

    true_item = target_query["item"]
    return {
        "conversations": [
            {
                "role": "user",
                "content": _build_ctx_prompt(
                    neg_docs, target_query["text"], examples
                ),
            },
            {"role": "assistant", "content": _build_answer(true_item, copy=False)},
        ],
        "metadata": {"pattern": "all_noise", "target_id": true_item},
    }


def _process_query(args):
    q, n_shot, doc_id_to_doc, eligible_docs, examples = args
    cd = generate_context_dependent(q, n_shot, doc_id_to_doc, eligible_docs, examples)
    an = generate_all_noise(q, n_shot, eligible_docs, examples)
    return cd, an


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
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc_id_to_doc = {d["doc_id"]: d for d in docs_pool}

    args_list = [
        (
            q,
            n_shot,
            doc_id_to_doc,
            docs_pool,
            _sample_examples(example_pool, q, n_shot),
        )
        for q in queries
    ]

    n_workers = max(1, cpu_count() - 1)
    with Pool(processes=n_workers) as pool:
        results = list(
            tqdm(
                pool.imap(_process_query, args_list, chunksize=64),
                total=len(queries),
                desc=f"build {os.path.basename(output_path)}",
            )
        )

    out: List[Dict] = []
    n_skipped_cd = n_skipped_an = 0
    for cd, an in results:
        if cd:
            out.append(cd)
        else:
            n_skipped_cd += 1
        if an:
            out.append(an)
        else:
            n_skipped_an += 1

    random.shuffle(out)
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in out:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(
        f"Saved {len(out)} -> {output_path}  "
        f"(skipped: cd={n_skipped_cd}, all_noise={n_skipped_an})"
    )


@hydra.main(
    config_path="../configs",
    config_name="build_test_docquery_v4",
    version_base=None,
)
def main(cfg: DictConfig):
    random.seed(cfg.get("seed", 42))

    dataset = load_dataset(
        "json",
        data_files={
            "train": f"{cfg.input_path}/train.jsonl",
            "test": f"{cfg.input_path}/test.jsonl",
            "icl_test": f"{cfg.input_path}/icl_test.jsonl",
        },
    )

    train_docs = [
        _alias_item(dict(s)) for s in dataset["train"] if s["operation"] == "indexing"
    ]
    test_queries = [
        _alias_item(dict(s)) for s in dataset["test"] if s["operation"] == "query"
    ]
    icl_test_docs = [
        _alias_item(dict(s))
        for s in dataset["icl_test"]
        if s["operation"] == "indexing"
    ]
    icl_test_queries = [
        _alias_item(dict(s)) for s in dataset["icl_test"] if s["operation"] == "query"
    ]

    print(
        f"pools: train_docs={len(train_docs)}  "
        f"test_q={len(test_queries)}  "
        f"icl_test_docs={len(icl_test_docs)}  icl_test_q={len(icl_test_queries)}"
    )

    n_shot = int(cfg.n_shot)
    splits = cfg.get("splits", ["test", "icl_test"])

    if "test" in splits:
        process_and_save(
            queries=test_queries,
            docs_pool=train_docs,
            example_pool=test_queries,
            output_path=f"{cfg.output_dir}/test_{n_shot}shot.jsonl",
            n_shot=n_shot,
        )
    if "icl_test" in splits:
        process_and_save(
            queries=icl_test_queries,
            docs_pool=icl_test_docs,
            example_pool=icl_test_queries,
            output_path=f"{cfg.output_dir}/icl_test_{n_shot}shot.jsonl",
            n_shot=n_shot,
        )


if __name__ == "__main__":
    main()
