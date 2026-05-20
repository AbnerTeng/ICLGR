"""Find questionable compressed summaries and create a rerun JSONL.

Example:
    python3 -m src.find_bad_compressed_summaries \
        --before data/nq-item-id/data/train.jsonl \
        --after data/nq-item-id-llm-compressed-final/data/train.jsonl \
        --bad-out outputs/bad_summaries.jsonl \
        --source-out outputs/bad_source_for_rerun.jsonl \
        --report-out outputs/bad_summaries_report.md
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Iterable

from src.compress_indexing_text import lead_focused_text, normalize_text, strip_front_matter


BAD_MARKERS = (
    "jump to : navigation",
    "jump to: navigation",
    "contents ( hide )",
    "external links",
    "references",
    "retrieved from",
    "categories :",
    "hidden categories :",
    "this article needs",
    "this article is about",
    "full - power stations",
    "tv market city of license",
    "channel number digital number call letters",
    "housemates name entered exited",
    "talk contents about wikipedia",
    "terms may apply",
    "privacy policy",
)

BAD_ENDINGS = (" the", " of", " for", " and", " to", " in", " with", " by")


def token_count(text: str) -> int:
    return len(text.split())


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}") from exc


def normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def token_ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def copied_ngram_ratio(source_text: str, summary: str, n: int = 12) -> float:
    summary_ngrams = token_ngrams(summary, n)
    if not summary_ngrams:
        return 0.0
    source_ngrams = token_ngrams(source_text, n)
    if not source_ngrams:
        return 0.0
    copied = len(summary_ngrams & source_ngrams)
    return copied / len(summary_ngrams)


def source_echo_position(summary: str, source_text: str, doc_id: str) -> int | None:
    summary_norm = normalize_for_match(summary)
    source_text = normalize_text(source_text)
    title = doc_id.strip().lower()
    markers = [
        "jump to : navigation",
        "jump to: navigation",
        "contents ( hide )",
        "references",
        "external links",
        "see also",
    ]
    if title:
        markers.extend([f"{title} ", f"{title} jump to", f"{title} {title}"])

    source_prefixes = [
        " ".join(normalize_text(source_text).split()[:50]).strip(),
        " ".join(strip_front_matter(source_text, doc_id=doc_id).split()[:50]).strip(),
        " ".join(lead_focused_text(source_text, doc_id=doc_id).split()[:50]).strip(),
    ]
    markers.extend(prefix for prefix in source_prefixes if prefix)

    best_pos: int | None = None
    for marker in markers:
        marker_norm = normalize_for_match(marker)
        if not marker_norm:
            continue
        pos = summary_norm.find(marker_norm)
        if pos > 80 and (best_pos is None or pos < best_pos):
            best_pos = pos
    return best_pos


def judge_summary(before: dict, after: dict, target_tokens: int) -> list[str]:
    doc_id = after.get("doc_id", "")
    source_text = before.get("text", "")
    summary = after.get("text", "")
    lowered = normalize_for_match(summary)
    reasons: list[str] = []
    n_tokens = token_count(summary)

    if n_tokens < 60:
        reasons.append("too_short")
    if target_tokens == 300 and n_tokens > 330:
        reasons.append("too_long")
    if target_tokens == 150 and n_tokens > 180:
        reasons.append("too_long")

    for marker in BAD_MARKERS:
        if marker in lowered:
            reasons.append(f"bad_marker:{marker}")
            break

    if source_echo_position(summary, source_text=source_text, doc_id=doc_id) is not None:
        reasons.append("source_echo")

    title = doc_id.strip().lower()
    if title and lowered.count(title) >= 2:
        reasons.append("title_repeated")

    if lowered.endswith(BAD_ENDINGS):
        reasons.append("truncated_sentence")

    ratio = copied_ngram_ratio(source_text, summary)
    if n_tokens >= 90 and ratio >= 0.35:
        reasons.append(f"verbatim_copy_ratio:{ratio:.2f}")

    return reasons


def load_before_indexing(path: Path) -> dict[str, dict]:
    rows = {}
    for _, row in iter_jsonl(path):
        if row.get("operation") == "indexing":
            rows[row.get("doc_id", "")] = row
    return rows


def truncate(text: str, max_chars: int) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ..."


def build_report(bad_items: list[dict], reason_counts: Counter, examples: int) -> str:
    lines = [
        "# Bad Summary Report",
        "",
        f"- bad_count: `{len(bad_items)}`",
        "",
        "## Reasons",
        "",
        "| reason | count |",
        "|---|---:|",
    ]
    for reason, count in reason_counts.most_common():
        lines.append(f"| `{reason}` | {count} |")

    if bad_items:
        lengths = [item["after_tokens"] for item in bad_items]
        lines.extend(
            [
                "",
                "## Bad Lengths",
                "",
                f"- avg: `{mean(lengths):.2f}`",
                f"- min: `{min(lengths)}`",
                f"- max: `{max(lengths)}`",
                "",
                "## Examples",
                "",
            ]
        )

    for item in bad_items[:examples]:
        lines.extend(
            [
                f"### line {item['line_no']} | {item['doc_id']}",
                "",
                f"- reasons: `{', '.join(item['reasons'])}`",
                f"- before_tokens: `{item['before_tokens']}`",
                f"- after_tokens: `{item['after_tokens']}`",
                "",
                "**After**",
                "",
                truncate(item["after_text"], 1200),
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--bad-out", default="outputs/bad_summaries.jsonl")
    parser.add_argument("--source-out", default="outputs/bad_source_for_rerun.jsonl")
    parser.add_argument("--report-out", default="outputs/bad_summaries_report.md")
    parser.add_argument("--target-tokens", type=int, default=300)
    parser.add_argument("--examples", type=int, default=30)
    args = parser.parse_args()

    before_by_doc_id = load_before_indexing(Path(args.before))
    bad_items = []
    reason_counts: Counter = Counter()
    checked = 0
    missing_before = 0

    for line_no, after in iter_jsonl(Path(args.after)):
        if after.get("operation") != "indexing":
            continue
        checked += 1
        doc_id = after.get("doc_id", "")
        before = before_by_doc_id.get(doc_id)
        if before is None:
            missing_before += 1
            continue

        reasons = judge_summary(before, after, target_tokens=args.target_tokens)
        if not reasons:
            continue
        reason_counts.update(reasons)
        bad_items.append(
            {
                "line_no": line_no,
                "doc_id": doc_id,
                "reasons": reasons,
                "before_tokens": token_count(before.get("text", "")),
                "after_tokens": token_count(after.get("text", "")),
                "after_text": after.get("text", ""),
            }
        )

    bad_out = Path(args.bad_out)
    bad_out.parent.mkdir(parents=True, exist_ok=True)
    with bad_out.open("w", encoding="utf-8") as f:
        for item in bad_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    source_out = Path(args.source_out)
    source_out.parent.mkdir(parents=True, exist_ok=True)
    bad_doc_ids = {item["doc_id"] for item in bad_items}
    with source_out.open("w", encoding="utf-8") as f:
        for _, row in iter_jsonl(Path(args.before)):
            if row.get("operation") == "indexing" and row.get("doc_id", "") in bad_doc_ids:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = build_report(bad_items, reason_counts, examples=args.examples)
    report += f"\n\nChecked indexing rows: `{checked}`\n"
    report += f"Missing before rows: `{missing_before}`\n"
    Path(args.report_out).write_text(report + "\n", encoding="utf-8")

    print(f"checked_indexing={checked}")
    print(f"bad_count={len(bad_items)}")
    print(f"bad_out={bad_out}")
    print(f"source_out={source_out}")
    print(f"report_out={args.report_out}")


if __name__ == "__main__":
    main()
