"""
Build high-shot (e.g. 200-shot) ICL dataset in id_only format.

Replicates the msmarco-icl-100shot-v4-id_only format:
  - Documents block shows only "Identifier: <item>" (no text)
  - Task block has 3 ICL examples (query → answer, no [COPY])
  - context_dependent: target doc placed at random position, answer = [COPY] <item>
  - all_noise: no target doc in context, answer = <item> (parametric memory)

Usage:
    python src/build_highshot_icl_id_only.py \
        --input_file  data/msmarco_text/train.jsonl \
        --output_dir  data/msmarco-icl-200shot-v4-id_only \
        --n_shot      200 \
        --n_icl_examples 3 \
        --seed        42

    # also build test/icl_test splits:
    python src/build_highshot_icl_id_only.py \
        --input_file  data/msmarco_text/train.jsonl \
        --test_file   data/msmarco_text/test.jsonl \
        --icl_test_file data/msmarco_text/icl_test.jsonl \
        --output_dir  data/msmarco-icl-200shot-v4-id_only \
        --n_shot      200
"""

import argparse
import json
import random
from pathlib import Path
from typing import Optional

from tqdm import tqdm


def load_split(path: str):
    docs, queries = [], []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d["operation"] == "indexing":
                docs.append(d)
            elif d["operation"] == "query":
                queries.append(d)
    return docs, queries


def build_prompt(context_docs: list, query_text: str, icl_examples: list) -> str:
    doc_blocks = "\n\n".join(
        f"Document {i + 1}\nIdentifier: {doc['item'].lower()}"
        for i, doc in enumerate(context_docs)
    )
    example_block = "\n\n".join(
        f"Example {i + 1}:\nQuery: {ex['text']}\nAnswer: {ex['item'].lower()}"
        for i, ex in enumerate(icl_examples)
    )
    return (
        f"## Documents\n{doc_blocks}\n\n\n"
        f"## Task\n{example_block}\n\n"
        f"Query: {query_text}\nAnswer:"
    )


def generate_context_dependent(
    query: dict,
    docs: list,
    doc_id_to_doc: dict,
    n_shot: int,
    icl_examples: list,
    rng: random.Random,
) -> Optional[dict]:
    true_doc_id = query["doc_id"]
    target_item = query["item"].lower()

    pos_doc = doc_id_to_doc.get(true_doc_id)
    if pos_doc is None:
        return None

    neg_pool = [d for d in docs if d["doc_id"] != true_doc_id]
    if len(neg_pool) < n_shot - 1:
        return None

    neg_docs = rng.sample(neg_pool, n_shot - 1)
    target_pos = rng.randint(0, n_shot - 1)  # 0-indexed insert position
    context_docs = neg_docs[:target_pos] + [pos_doc] + neg_docs[target_pos:]

    return {
        "conversations": [
            {"role": "user", "content": build_prompt(context_docs, query["text"], icl_examples)},
            {"role": "assistant", "content": f"[COPY] {target_item}"},
        ],
        "metadata": {
            "pattern": "context_dependent",
            "target_id": target_item,
            "target_position": target_pos + 1,  # 1-indexed
        },
    }


def generate_all_noise(
    query: dict,
    docs: list,
    n_shot: int,
    icl_examples: list,
    rng: random.Random,
) -> Optional[dict]:
    true_doc_id = query["doc_id"]
    target_item = query["item"].lower()

    neg_pool = [d for d in docs if d["doc_id"] != true_doc_id]
    if len(neg_pool) < n_shot:
        return None

    context_docs = rng.sample(neg_pool, n_shot)

    return {
        "conversations": [
            {"role": "user", "content": build_prompt(context_docs, query["text"], icl_examples)},
            {"role": "assistant", "content": target_item},
        ],
        "metadata": {
            "pattern": "all_noise",
            "target_id": target_item,
        },
    }


def build_split(
    docs: list,
    queries: list,
    example_pool: list,
    n_shot: int,
    n_icl_examples: int,
    rng: random.Random,
    cd_only: bool = False,
) -> list:
    doc_id_to_doc = {d["doc_id"]: d for d in docs}
    examples = []

    for query in tqdm(queries, desc="generating"):
        # sample ICL examples: pick n_icl_examples random indices, re-roll if same doc
        icl_pool = [q for q in example_pool if q["doc_id"] != query["doc_id"]]
        if len(icl_pool) < n_icl_examples:
            continue
        icl_examples = rng.sample(icl_pool, n_icl_examples)

        cd = generate_context_dependent(query, docs, doc_id_to_doc, n_shot, icl_examples, rng)
        if cd:
            examples.append(cd)

        if not cd_only:
            an = generate_all_noise(query, docs, n_shot, icl_examples, rng)
            if an:
                examples.append(an)

    rng.shuffle(examples)
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, help="train.jsonl (operation: indexing/query)")
    parser.add_argument("--test_file", default=None)
    parser.add_argument("--icl_test_file", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_shot", type=int, default=200)
    parser.add_argument("--n_icl_examples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cd_only", action="store_true", help="generate context_dependent samples only")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_docs, train_queries = load_split(args.input_file)
    print(f"train: {len(train_docs)} docs, {len(train_queries)} queries")

    splits = {"train": (train_docs, train_queries)}

    if args.test_file:
        test_docs, test_queries = load_split(args.test_file)
        print(f"test:  {len(test_docs)} docs, {len(test_queries)} queries")
        splits["test"] = (test_docs, test_queries)

    if args.icl_test_file:
        icl_docs, icl_queries = load_split(args.icl_test_file)
        print(f"icl_test: {len(icl_docs)} docs, {len(icl_queries)} queries")
        splits["icl_test"] = (icl_docs, icl_queries)

    for split_name, (docs, queries) in splits.items():
        print(f"\nBuilding {split_name} ({args.n_shot}-shot id_only)...")
        examples = build_split(
            docs, queries,
            example_pool=train_queries,
            n_shot=args.n_shot,
            n_icl_examples=args.n_icl_examples,
            rng=rng,
            cd_only=args.cd_only,
        )

        out_path = out_dir / f"{split_name}.jsonl"
        with open(out_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        cd = sum(1 for e in examples if e["metadata"]["pattern"] == "context_dependent")
        an = sum(1 for e in examples if e["metadata"]["pattern"] == "all_noise")
        print(f"  wrote {len(examples)} samples -> {out_path}")
        print(f"  context_dependent: {cd}, all_noise: {an}")


if __name__ == "__main__":
    main()
