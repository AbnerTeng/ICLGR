"""Compare original and compressed JSONL indexing rows.

Example:
    python3 -m src.compare_compressed_jsonl \
        --before data/nq-item-id/data/train.jsonl \
        --after data/nq-item-id-llm-compressed/data/train.jsonl \
        --out outputs/compressed_compare.md \
        --examples 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Iterable


def token_count(text: str) -> int:
    return len(text.split())


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}") from exc


def percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    return values[int((len(values) - 1) * q)]


def summarize_lengths(values: list[int]) -> str:
    if not values:
        return "| count | avg | min | p25 | p50 | p75 | p90 | p95 | max |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |"
    return "\n".join(
        [
            "| count | avg | min | p25 | p50 | p75 | p90 | p95 | max |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            (
                f"| {len(values)} | {mean(values):.2f} | {min(values)} | "
                f"{percentile(values, 0.25)} | {percentile(values, 0.50)} | "
                f"{percentile(values, 0.75)} | {percentile(values, 0.90)} | "
                f"{percentile(values, 0.95)} | {max(values)} |"
            ),
        ]
    )


def truncate(text: str, max_chars: int) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ..."


def load_after_rows(path: Path) -> list[dict]:
    rows = []
    for row in iter_jsonl(path):
        if row.get("operation") == "indexing":
            rows.append(row)
    return rows


def load_before_by_doc_id(path: Path, doc_ids: set[str]) -> dict[str, dict]:
    rows = {}
    for row in iter_jsonl(path):
        doc_id = row.get("doc_id")
        if row.get("operation") == "indexing" and doc_id in doc_ids:
            rows[doc_id] = row
            if len(rows) == len(doc_ids):
                break
    return rows


def build_report(before_path: Path, after_path: Path, examples: int, preview_chars: int) -> str:
    after_rows = load_after_rows(after_path)
    doc_ids = {row.get("doc_id", "") for row in after_rows}
    before_rows = load_before_by_doc_id(before_path, doc_ids)

    matched = []
    missing = []
    for after in after_rows:
        doc_id = after.get("doc_id", "")
        before = before_rows.get(doc_id)
        if before is None:
            missing.append(doc_id)
            continue
        before_tokens = token_count(before.get("text", ""))
        after_tokens = token_count(after.get("text", ""))
        matched.append((doc_id, before, after, before_tokens, after_tokens))

    before_lengths = [item[3] for item in matched]
    after_lengths = [item[4] for item in matched]

    lines = [
        "# Compressed JSONL Comparison",
        "",
        f"- before: `{before_path}`",
        f"- after: `{after_path}`",
        f"- after indexing rows: `{len(after_rows)}`",
        f"- matched doc_ids: `{len(matched)}`",
        f"- missing doc_ids: `{len(missing)}`",
        "",
        "## Before Token Lengths",
        "",
        summarize_lengths(before_lengths),
        "",
        "## After Token Lengths",
        "",
        summarize_lengths(after_lengths),
        "",
        "## Examples",
        "",
    ]

    for i, (doc_id, before, after, before_tokens, after_tokens) in enumerate(matched[:examples], start=1):
        reduction = 0.0
        if before_tokens:
            reduction = 100.0 * (1.0 - after_tokens / before_tokens)
        lines.extend(
            [
                f"### {i}. {doc_id}",
                "",
                f"- before_tokens: `{before_tokens}`",
                f"- after_tokens: `{after_tokens}`",
                f"- reduction: `{reduction:.2f}%`",
                "",
                "**Before Preview**",
                "",
                truncate(before.get("text", ""), preview_chars),
                "",
                "**After**",
                "",
                after.get("text", "").strip(),
                "",
            ]
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--examples", type=int, default=10)
    parser.add_argument("--preview-chars", type=int, default=900)
    args = parser.parse_args()

    report = build_report(
        before_path=Path(args.before),
        after_path=Path(args.after),
        examples=args.examples,
        preview_chars=args.preview_chars,
    )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report + "\n", encoding="utf-8")
        print(f"Wrote {out_path}")
    else:
        print(report)


if __name__ == "__main__":
    main()
