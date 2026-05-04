"""Convert flat-format JSONL (with `operation` field) to chat `conversations` format,
then merge with another already-formatted JSONL.

Flat input record (e.g. msmarco_text-with_pseudo_query-100k-gr):
    {
      "item": "...", "text": "...", "doc_id": "...",
      "operation": "query" | "indexing", "source": "msmarco", ...
    }

Output record (chat_template format used by axolotl):
    {
      "conversations": [
        {"role": "user", "content": <text>},
        {"role": "assistant", "content": <doc_id>}
      ],
      "metadata": {"pattern": "<operation>", "target_id": "<doc_id>", "source": "..."}
    }

Usage:
    python -m src.merge_train_to_conversations \
        --already_chat ./data/msmarco-item-id-3shot-v4_128k/train_3shot.jsonl \
        --flat ./data/msmarco_text-with_pseudo_query-100k-gr/train.jsonl \
        --output ./data/msmarco-item-id-3shot-v4_128k_mix/train_with_pseudo.jsonl \
        --shuffle
"""

import argparse
import json
import os
import random
from typing import Iterable


def convert_flat_record(rec: dict) -> dict | None:
    op = rec.get("operation")
    text = rec.get("text")
    doc_id = rec.get("doc_id")
    if not text or not doc_id or op not in ("query", "indexing"):
        return None
    return {
        "conversations": [
            {"role": "user", "content": str(text)},
            {"role": "assistant", "content": str(doc_id)},
        ],
        "metadata": {
            "pattern": op,
            "target_id": str(doc_id),
            "source": rec.get("source", ""),
        },
    }


def iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--already_chat", type=str, required=True,
                        help="JSONL already in `conversations` format (kept as-is).")
    parser.add_argument("--flat", type=str, required=True,
                        help="JSONL with `operation` field; converted to chat format.")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    n_chat, n_flat_ok, n_flat_skip = 0, 0, 0
    pattern_count: dict[str, int] = {}

    records: list[dict] = []

    for rec in iter_jsonl(args.already_chat):
        if "conversations" not in rec:
            n_flat_skip += 1
            continue
        records.append(rec)
        n_chat += 1
        pat = (rec.get("metadata") or {}).get("pattern", "chat_passthrough")
        pattern_count[pat] = pattern_count.get(pat, 0) + 1

    for rec in iter_jsonl(args.flat):
        out = convert_flat_record(rec)
        if out is None:
            n_flat_skip += 1
            continue
        records.append(out)
        n_flat_ok += 1
        pat = out["metadata"]["pattern"]
        pattern_count[pat] = pattern_count.get(pat, 0) + 1

    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(records)

    with open(args.output, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"already_chat (passthrough): {n_chat}")
    print(f"flat -> chat:               {n_flat_ok}")
    print(f"skipped:                    {n_flat_skip}")
    print(f"total written:              {len(records)} -> {args.output}")
    print("Pattern breakdown:")
    for k, v in sorted(pattern_count.items(), key=lambda x: -x[1]):
        print(f"  {k:25s} {v}")


if __name__ == "__main__":
    main()
