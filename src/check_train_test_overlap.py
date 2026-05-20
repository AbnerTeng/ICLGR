"""Check overlap between test sets and training data.

For each test file, computes how many rows have:
  - doc_id that appears in any train.jsonl indexing row
  - query text that appears in any train.jsonl query row
  - (query, doc_id) pair that appears exactly in any train.jsonl query row

Usage:
    python -m src.check_train_test_overlap \\
        --train data/nq-item-id/data/train.jsonl \\
        --test  data/nq-item-id/data/test.jsonl \\
        --test  data/nq-item-id/data/icl_test_query_only.jsonl

    # Show a few overlapping/non-overlapping examples per test file
    python -m src.check_train_test_overlap ... --show 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_train(path: Path) -> tuple[set[str], set[str], set[tuple[str, str]]]:
    """Return (indexing_docids, query_texts, query_pairs) from a train file."""
    indexing_docids: set[str] = set()
    query_texts: set[str] = set()
    query_pairs: set[tuple[str, str]] = set()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            op = d.get("operation")
            if op == "indexing":
                indexing_docids.add(d["doc_id"])
            elif op == "query":
                query_texts.add(d["text"])
                query_pairs.add((d["text"], d["doc_id"]))
    return indexing_docids, query_texts, query_pairs


def check_test(
    test_path: Path,
    indexing_docids: set[str],
    query_texts: set[str],
    query_pairs: set[tuple[str, str]],
    show: int = 0,
) -> dict:
    n = 0
    doc_hits = 0
    qtext_hits = 0
    pair_hits = 0
    examples_dup: list[dict] = []
    examples_uniq: list[dict] = []

    with test_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            n += 1
            text, doc_id = d["text"], d["doc_id"]
            doc_in = doc_id in indexing_docids
            qtext_in = text in query_texts
            pair_in = (text, doc_id) in query_pairs

            if doc_in:
                doc_hits += 1
            if qtext_in:
                qtext_hits += 1
            if pair_in:
                pair_hits += 1

            if show:
                if pair_in and len(examples_dup) < show:
                    examples_dup.append({"text": text, "doc_id": doc_id})
                elif not pair_in and len(examples_uniq) < show:
                    examples_uniq.append({"text": text, "doc_id": doc_id})

    return {
        "n": n,
        "doc_hits": doc_hits,
        "qtext_hits": qtext_hits,
        "pair_hits": pair_hits,
        "examples_dup": examples_dup,
        "examples_uniq": examples_uniq,
    }


def fmt_pct(num: int, denom: int) -> str:
    return f"{num}/{denom} ({num / denom:.1%})" if denom else "n/a"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True, help="Path to train.jsonl")
    parser.add_argument(
        "--test",
        type=Path,
        action="append",
        required=True,
        help="Test jsonl path (repeat for multiple)",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=0,
        help="Show up to N overlapping / N non-overlapping examples per test file",
    )
    args = parser.parse_args()

    print(f"Loading train: {args.train}")
    indexing_docids, query_texts, query_pairs = load_train(args.train)
    print(
        f"  -> {len(indexing_docids):,} indexing docids, "
        f"{len(query_texts):,} unique query texts, "
        f"{len(query_pairs):,} (query,doc_id) pairs\n"
    )

    for test_path in args.test:
        if not test_path.exists():
            print(f"WARN: {test_path} not found, skipping")
            continue
        result = check_test(
            test_path, indexing_docids, query_texts, query_pairs, show=args.show
        )
        n = result["n"]
        print(f"=== {test_path} ===")
        print(f"  samples: {n}")
        print(f"  doc_id ∈ train.indexing:       {fmt_pct(result['doc_hits'], n)}")
        print(f"  query text ∈ train.query:      {fmt_pct(result['qtext_hits'], n)}")
        print(f"  (query, doc_id) ∈ train.query: {fmt_pct(result['pair_hits'], n)}")
        if args.show and result["examples_dup"]:
            print(f"\n  -- examples that ARE in train ({len(result['examples_dup'])}):")
            for ex in result["examples_dup"]:
                print(f"     text  : {ex['text']!r}")
                print(f"     doc_id: {ex['doc_id']!r}")
        if args.show and result["examples_uniq"]:
            print(f"\n  -- examples NOT in train ({len(result['examples_uniq'])}):")
            for ex in result["examples_uniq"]:
                print(f"     text  : {ex['text']!r}")
                print(f"     doc_id: {ex['doc_id']!r}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
