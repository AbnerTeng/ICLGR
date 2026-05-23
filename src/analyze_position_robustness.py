"""Analyze and plot Hit@1 vs golden document position.

Reads inference results from retrieve_results/position_robustness/ and
produces a position-robustness plot.

Usage:
    python -m src.analyze_position_robustness \
        --results_dir retrieve_results/position_robustness \
        --output      nshot_results/position_robustness.png
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _parse_filename(name: str):
    """Extract (tag, split, position) from result filename."""
    m = re.search(r"(.+)-(test|icl_test)-pos(\d+)-", name)
    if not m:
        return None
    return m.group(1), m.group(2), int(m.group(3))


def load_results(results_dir: Path) -> Dict:
    """Return {tag: {split: {position: metrics}}}"""
    data: Dict = {}
    for f in sorted(results_dir.glob("*.json")):
        parsed = _parse_filename(f.stem)
        if parsed is None:
            continue
        tag, split, pos = parsed
        with open(f) as fh:
            r = json.load(fh)

        pp = r.get("per_pattern", {})
        cd = pp.get("context_dependent", {})

        data.setdefault(tag, {}).setdefault(split, {})[pos] = {
            "hit@1":       cd.get("hit_at_1", r.get("hit_at_1")),
            "hit@10":      cd.get("hit_at_10", r.get("hit_at_10")),
            "n":           cd.get("total", r.get("total")),
        }
    return data


def plot(data: Dict, output: Path):
    splits = ["test", "icl_test"]
    split_labels = {"test": "test (context_dependent)", "icl_test": "icl_test (context_dependent)"}
    colors = ["#2ca02c", "#d62728", "#ff7f0e", "#9467bd"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, split in zip(axes, splits):
        for ci, (tag, splits_data) in enumerate(sorted(data.items())):
            split_data = splits_data.get(split, {})
            if not split_data:
                continue
            positions = sorted(split_data.keys())
            hits = [split_data[p]["hit@1"] for p in positions]
            ns = [split_data[p]["n"] for p in positions]
            color = colors[ci % len(colors)]

            ax.plot(positions, hits, marker="o", color=color,
                    label=tag, lw=2, ms=8)
            for p, h, n in zip(positions, hits, ns):
                ax.annotate(f"{h:.2f}\n(n={n})", (p, h),
                            textcoords="offset points", xytext=(0, 8),
                            fontsize=7, ha="center", color=color)

        ax.set_xlabel("Golden document position (1-indexed)", fontsize=11)
        ax.set_ylabel("Hit@1", fontsize=11)
        ax.set_title(f"{split_labels.get(split, split)}", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(sorted({p for sd in data.values()
                               for p in sd.get(split, {}).keys()}))
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.axhline(0, color="grey", lw=0.5)

    fig.suptitle("Position Robustness: Hit@1 vs Golden Doc Position (100-shot)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {output}")


def _fmt(v) -> str:
    return f"{v:.4f}" if v is not None else "   N/A"


def print_table(data: Dict):
    for tag, splits_data in sorted(data.items()):
        print(f"\n{'='*60}")
        print(f"Model: {tag}")
        for split, pos_data in sorted(splits_data.items()):
            print(f"\n  Split: {split}")
            print(f"  {'Pos':>5} {'Hit@1':>8} {'Hit@10':>8} {'n':>7}")
            print(f"  {'-'*34}")
            for pos in sorted(pos_data.keys()):
                m = pos_data[pos]
                print(f"  {pos:>5} {_fmt(m['hit@1']):>8} {_fmt(m['hit@10']):>8} {str(m['n']):>7}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="retrieve_results/position_robustness")
    p.add_argument("--output",      default="nshot_results/position_robustness.png")
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    data = load_results(results_dir)
    if not data:
        print(f"No results found in {results_dir}")
        return
    print_table(data)
    plot(data, Path(args.output))


if __name__ == "__main__":
    main()
