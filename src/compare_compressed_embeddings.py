"""Compare original and compressed JSONL rows with embedding cosine similarity.

Example:
    uv run --with 'huggingface-hub>=1.5.0,<2.0' \
        python -m src.compare_compressed_embeddings \
        --before data/nq-item-id/data/train.jsonl \
        --after data/nq-item-id-llm-compressed-final/train.jsonl \
        --out outputs/nq_compressed_final_embedding_cosine.md \
        --worst-out outputs/nq_compressed_final_embedding_cosine_worst.jsonl \
        --sample-size 2000
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}") from exc


def token_count(text: str) -> int:
    return len(text.split())


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return values[int((len(values) - 1) * q)]


def summarize(values: list[float]) -> str:
    if not values:
        return "| count | avg | min | p1 | p5 | p25 | p50 | p75 | p95 | max |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |"
    return "\n".join(
        [
            "| count | avg | min | p1 | p5 | p25 | p50 | p75 | p95 | max |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            (
                f"| {len(values)} | {mean(values):.4f} | {min(values):.4f} | "
                f"{percentile(values, 0.01):.4f} | {percentile(values, 0.05):.4f} | "
                f"{percentile(values, 0.25):.4f} | {percentile(values, 0.50):.4f} | "
                f"{percentile(values, 0.75):.4f} | {percentile(values, 0.95):.4f} | "
                f"{max(values):.4f} |"
            ),
        ]
    )


def truncate(text: str, max_chars: int) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ..."


def load_indexing_by_doc_id(path: Path) -> dict[str, dict]:
    rows = {}
    for row in iter_jsonl(path):
        if row.get("operation") == "indexing":
            rows[row["doc_id"]] = row
    return rows


def select_doc_ids(
    doc_ids: list[str],
    before_rows: dict[str, dict],
    after_rows: dict[str, dict],
    sample_size: int | None,
    seed: int,
    include_anomalies: bool,
) -> list[str]:
    if sample_size is None or sample_size >= len(doc_ids):
        return doc_ids

    selected: set[str] = set()
    if include_anomalies:
        for doc_id in doc_ids:
            before_tokens = token_count(before_rows[doc_id].get("text", ""))
            after_tokens = token_count(after_rows[doc_id].get("text", ""))
            if after_tokens < 30 or after_tokens > before_tokens:
                selected.add(doc_id)

    remaining = [doc_id for doc_id in doc_ids if doc_id not in selected]
    rng = random.Random(seed)
    rng.shuffle(remaining)
    selected.update(remaining[: max(0, sample_size - len(selected))])
    return [doc_id for doc_id in doc_ids if doc_id in selected]


def write_worst(path: Path, rows: list[dict], limit: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows[:limit]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", default=None)
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out", required=True)
    parser.add_argument("--worst-out", default=None)
    parser.add_argument("--worst", type=int, default=50)
    parser.add_argument("--preview-chars", type=int, default=700)
    parser.add_argument("--no-include-anomalies", action="store_true")
    args = parser.parse_args()

    before_path = Path(args.before)
    after_path = Path(args.after)
    before_rows = load_indexing_by_doc_id(before_path)
    after_rows = load_indexing_by_doc_id(after_path)
    doc_ids = sorted(set(before_rows) & set(after_rows))
    selected_doc_ids = select_doc_ids(
        doc_ids=doc_ids,
        before_rows=before_rows,
        after_rows=after_rows,
        sample_size=args.sample_size,
        seed=args.seed,
        include_anomalies=not args.no_include_anomalies,
    )

    before_texts = [before_rows[doc_id].get("text", "") for doc_id in selected_doc_ids]
    after_texts = [after_rows[doc_id].get("text", "") for doc_id in selected_doc_ids]

    model = SentenceTransformer(args.model, device=args.device)
    before_embeddings = model.encode(
        before_texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    after_embeddings = model.encode(
        after_texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    cosines = np.sum(before_embeddings * after_embeddings, axis=1).tolist()

    result_rows = []
    for doc_id, cosine in zip(selected_doc_ids, cosines):
        before_text = before_rows[doc_id].get("text", "")
        after_text = after_rows[doc_id].get("text", "")
        before_tokens = token_count(before_text)
        after_tokens = token_count(after_text)
        reduction = 0.0 if before_tokens == 0 else 100.0 * (1.0 - after_tokens / before_tokens)
        result_rows.append(
            {
                "doc_id": doc_id,
                "cosine": float(cosine),
                "before_tokens": before_tokens,
                "after_tokens": after_tokens,
                "reduction_pct": reduction,
                "before_preview": truncate(before_text, args.preview_chars),
                "after": after_text.strip(),
            }
        )

    result_rows.sort(key=lambda row: row["cosine"])
    values = [row["cosine"] for row in result_rows]
    below_80 = sum(value < 0.80 for value in values)
    below_70 = sum(value < 0.70 for value in values)
    below_60 = sum(value < 0.60 for value in values)

    lines = [
        "# Compressed Embedding Cosine Comparison",
        "",
        f"- before: `{before_path}`",
        f"- after: `{after_path}`",
        f"- model: `{args.model}`",
        f"- matched indexing rows: `{len(doc_ids)}`",
        f"- evaluated rows: `{len(selected_doc_ids)}`",
        f"- sample seed: `{args.seed}`",
        f"- cosine < 0.80: `{below_80}`",
        f"- cosine < 0.70: `{below_70}`",
        f"- cosine < 0.60: `{below_60}`",
        "",
        "## Cosine Summary",
        "",
        summarize(values),
        "",
        "## Worst Examples",
        "",
    ]

    for i, row in enumerate(result_rows[: args.worst], start=1):
        lines.extend(
            [
                f"### {i}. {row['doc_id']}",
                "",
                f"- cosine: `{row['cosine']:.4f}`",
                f"- before_tokens: `{row['before_tokens']}`",
                f"- after_tokens: `{row['after_tokens']}`",
                f"- reduction: `{row['reduction_pct']:.2f}%`",
                "",
                "**Before Preview**",
                "",
                row["before_preview"],
                "",
                "**After**",
                "",
                row["after"],
                "",
            ]
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")

    if args.worst_out:
        write_worst(Path(args.worst_out), result_rows, args.worst)
        print(f"Wrote {args.worst_out}")


if __name__ == "__main__":
    main()
