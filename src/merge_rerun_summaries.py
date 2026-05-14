"""Merge rerun compressed indexing rows back into a full JSONL file.

Example:
    python3 -m src.merge_rerun_summaries \
        --base data/nq-item-id-llm-compressed-final/data/train.jsonl \
        --rerun outputs/bad_rerun_300.jsonl \
        --out data/nq-item-id-llm-compressed-final/data/train.fixed.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}") from exc


def load_replacements(path: Path) -> dict[str, dict]:
    replacements = {}
    for _, row in iter_jsonl(path):
        if row.get("operation") == "indexing":
            replacements[row.get("doc_id", "")] = row
    return replacements


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--rerun", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    replacements = load_replacements(Path(args.rerun))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    replaced = 0
    with Path(args.base).open("r", encoding="utf-8") as fin, out_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line_no, line in enumerate(fin, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {args.base}:{line_no}") from exc

            total += 1
            doc_id = row.get("doc_id", "")
            if row.get("operation") == "indexing" and doc_id in replacements:
                row = replacements[doc_id]
                replaced += 1
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {out_path}")
    print(f"  rows={total} replacements_loaded={len(replacements)} replaced={replaced}")


if __name__ == "__main__":
    main()
