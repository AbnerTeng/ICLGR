"""Summarize eval logs into a single table.

Usage:
    python -m src.summarize_logs --log_dir logs/v3_pseudo
    python -m src.summarize_logs --log_dir logs/v3_pseudo logs/stage1_test_trie
"""

import argparse
import os
import re
from pathlib import Path
from collections import OrderedDict


def parse_log(log_path: str) -> dict:
    with open(log_path, "r") as f:
        content = f.read()

    results = {}

    h1 = re.search(r"Final\s+Hit@1:\s+([\d.]+)", content)
    h10 = re.search(r"Final\s+Hit@10:\s+([\d.]+)", content)
    if h1:
        results["overall/H@1"] = float(h1.group(1))
    if h10:
        results["overall/H@10"] = float(h10.group(1))

    for m in re.finditer(
        r"\[(\w+)\]\s+H@1=([\d.]+)\s+H@10=([\d.]+)\s+(copy@1|copy_rate)=([\d.]+)%",
        content,
    ):
        pat = m.group(1)
        results[f"{pat}/H@1"] = float(m.group(2))
        results[f"{pat}/H@10"] = float(m.group(3))
        metric_name = m.group(4)
        metric_value = float(m.group(5))
        if metric_name == "copy@1":
            results[f"{pat}/copy@1%"] = metric_value
        else:
            results[f"{pat}/beam_copy%"] = metric_value

    if not any(k.endswith("/copy@1%") for k in results):
        lines = content.splitlines()
        pattern_counts = {}
        current_pattern = None

        for line in lines:
            m = re.match(r"^\[(\d+)/(\d+)\]\s+pattern=(\w+)", line)
            if m:
                current_pattern = m.group(3)
                stats = pattern_counts.setdefault(
                    current_pattern, {"n": 0, "copy_top1": 0}
                )
                stats["n"] += 1
                continue

            if current_pattern is None:
                continue

            m = re.match(r"^\s+\[\s*1\]\s+(.*?)\s*(?:\s+\[[^\]]+\])?\s*$", line)
            if m:
                top1 = m.group(1).strip()
                if top1.startswith("[COPY]"):
                    pattern_counts[current_pattern]["copy_top1"] += 1
                current_pattern = None

        for pat, stats in pattern_counts.items():
            n = stats["n"]
            if n:
                results[f"{pat}/copy@1%"] = 100.0 * stats["copy_top1"] / n

    return results


def fmt(val, width=8, pct=False):
    if val < 0:
        return "-".rjust(width)
    if pct:
        return f"{val:.2f}%".rjust(width)
    return f"{val:.4f}".rjust(width)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, nargs="+", required=True)
    parser.add_argument("--csv", type=str, default="", help="Save as TSV file")
    args = parser.parse_args()

    patterns = ["doc_pos_front", "doc_pos_back", "all_noise"]

    rows = []
    for log_dir in args.log_dir:
        log_files = sorted(Path(log_dir).glob("*.log"))
        for log_path in log_files:
            name = log_path.stem.replace("eval_", "")
            r = parse_log(str(log_path))
            if not r:
                continue
            rows.append((log_dir, name, r))

    groups = OrderedDict()
    for log_dir, name, r in rows:
        key = os.path.basename(log_dir)
        if key not in groups:
            groups[key] = []
        groups[key].append((name, r))

    short_groups = {g for g in groups if "stage1" in g}

    W = 20  # model name width
    C = 8  # metric column width
    SEP = " | "

    all_tsv = []

    for group_name, group_rows in groups.items():
        is_short = group_name in short_groups

        # --- terminal header ---
        hdr = f"{'model':<{W}}{SEP}{'H@1':>{C}} {'H@10':>{C}}"
        if not is_short:
            for pat in patterns:
                short = pat.replace("doc_pos_", "").replace("all_noise", "noise")
                hdr += (
                    f"{SEP}{short + '/H@1':>{C}} {short + '/H@10':>{C}} "
                    f"{short + '/cp@1%':>{C}} {short + '/beam_cp%':>{C}}"
                )
        line_w = len(hdr)

        print(f"\n{'=' * line_w}")
        print(f"  {group_name}")
        print(f"{'=' * line_w}")
        print(hdr)
        print("-" * line_w)

        # --- TSV header ---
        T = "\t"
        tsv_hdr_parts = ["model", "H@1", "H@10"]
        if not is_short:
            for pat in patterns:
                short = pat.replace("doc_pos_", "").replace("all_noise", "noise")
                tsv_hdr_parts += [
                    f"{short}/H@1",
                    f"{short}/H@10",
                    f"{short}/cp@1%",
                    f"{short}/beam_cp%",
                ]
        all_tsv.append(f"\n{group_name}")
        all_tsv.append(T.join(tsv_hdr_parts))

        for name, r in group_rows:
            h1 = r.get("overall/H@1", -1)
            h10 = r.get("overall/H@10", -1)

            # terminal line
            line = f"{name:<{W}}{SEP}{fmt(h1, C)} {fmt(h10, C)}"
            # TSV line
            tsv_parts = [name, f"{h1:.4f}", f"{h10:.4f}"]

            if not is_short:
                for pat in patterns:
                    ph1 = r.get(f"{pat}/H@1", -1)
                    ph10 = r.get(f"{pat}/H@10", -1)
                    pcp = r.get(f"{pat}/copy@1%", -1)
                    pbcp = r.get(f"{pat}/beam_copy%", -1)
                    line += f"{SEP}{fmt(ph1, C)} {fmt(ph10, C)} {fmt(pcp, C, pct=True)} {fmt(pbcp, C, pct=True)}"
                    tsv_parts.append(f"{ph1:.4f}" if ph1 >= 0 else "-")
                    tsv_parts.append(f"{ph10:.4f}" if ph10 >= 0 else "-")
                    tsv_parts.append(f"{pcp:.2f}%" if pcp >= 0 else "-")
                    tsv_parts.append(f"{pbcp:.2f}%" if pbcp >= 0 else "-")

            print(line)
            all_tsv.append(T.join(tsv_parts))

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("\n".join(all_tsv) + "\n")
        print(f"\nSaved to: {args.csv}")


if __name__ == "__main__":
    main()
