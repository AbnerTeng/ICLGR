"""Three-model comparison: SFT vs DPO vs DPO-balanced.

Loads pre-computed per-sample records from ece_copy_token.py JSON outputs and
eval_icl_patterns.py JSON outputs, then generates:

  1. ece_results/comparison_3model.png
       6-panel reliability diagram grid (3 models × 2 splits)
  2. ece_results/comparison_histogram.png
       6-panel confidence histogram grid (3 models × 2 splits)
  3. ece_results/comparison_summary.json
       Machine-readable table of all ECE / generation metrics

Usage:
    python -m src.compare_ece_models \
        --ece_root ./ece_results \
        --eval_dir ./eval_results \
        --output_dir ./ece_results
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS = [
    {
        "label": "SFT",
        "hf_id":  "Abner0803/Qwen3-1.7B-icl-3shot-v4_128k-copy_tag",
        "tag":    "Qwen3-1.7B-icl-3shot-v4_128k-copy_tag",
        "dir":    "sft",
        "color":  "#2ca02c",
    },
    {
        "label": "DPO",
        "hf_id":  "Abner0803/Qwen3-1.7B-icl-3shot-v4_128k-copy_tag-dpo",
        "tag":    "Qwen3-1.7B-icl-3shot-v4_128k-copy_tag-dpo",
        "dir":    "dpo",
        "color":  "#d62728",
    },
    {
        "label": "DPO-balanced",
        "hf_id":  "Abner0803/Qwen3-1.7B-icl-3shot-v4_128k-copy_tag-dpo-balanced",
        "tag":    "Qwen3-1.7B-icl-3shot-v4_128k-copy_tag-dpo-balanced",
        "dir":    "dpo_balanced",
        "color":  "#ff7f0e",
    },
]

SPLITS = ["test", "icl_test"]


# ---------------------------------------------------------------------------
# ECE helpers  (mirror of src/ece_copy_token.py — no import to avoid deps)
# ---------------------------------------------------------------------------

def compute_ece(
    confidences: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_mids  = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    bin_confs  = np.zeros(n_bins)
    bin_accs   = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= lo) & (confidences < hi)
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_counts[i] = mask.sum()
        bin_confs[i]  = confidences[mask].mean()
        bin_accs[i]   = labels[mask].mean()

    bin_weights = bin_counts / max(bin_counts.sum(), 1)
    ece = float(np.sum(bin_weights * np.abs(bin_confs - bin_accs)))
    return ece, bin_mids, bin_confs, bin_accs, bin_weights


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ece_records(ece_root: str, model_dir: str, split: str) -> Dict:
    """Load the JSON produced by ece_copy_token.py."""
    path = os.path.join(ece_root, model_dir, f"ece_results_{split}.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def load_eval_results(eval_dir: str, model_tag: str, split: str) -> Dict:
    """Load the JSON produced by eval_icl_patterns.py."""
    path = os.path.join(eval_dir, f"{model_tag}_{split}.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Plot 1: 3×2 reliability diagram grid
# ---------------------------------------------------------------------------

def plot_reliability_grid(
    all_data: Dict,          # all_data[label][split] = {confidences, labels, ece, ...}
    n_bins: int,
    output_path: str,
):
    n_models = len(MODELS)
    n_splits = len(SPLITS)

    fig, axes = plt.subplots(
        n_models, n_splits,
        figsize=(5 * n_splits, 4 * n_models),
        squeeze=False,
    )

    for mi, m in enumerate(MODELS):
        label = m["label"]
        color = m["color"]
        for si, split in enumerate(SPLITS):
            ax = axes[mi][si]
            data = all_data.get(label, {}).get(split, {})
            if not data:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=11)
                ax.set_title(f"{label} | {split}")
                continue

            confs  = np.array(data["confidences"])
            labels_arr = np.array(data["labels"])
            ece, bin_mids, bin_confs, bin_accs, bin_weights = compute_ece(
                confs, labels_arr, n_bins
            )

            mask = bin_weights > 0
            bar_w = 1.0 / n_bins * 0.85

            ax.bar(bin_mids[mask], bin_accs[mask], width=bar_w,
                   alpha=0.75, color=color, label="Fraction positive")
            ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect calibration")

            # Gap overlay
            gap = bin_confs[mask] - bin_accs[mask]
            gap_colors = ["#d62728" if g > 0 else "#2ca02c" for g in gap]
            for bm, g, bc, ba in zip(
                bin_mids[mask], gap, gap_colors, bin_accs[mask]
            ):
                ax.bar(bm, g, bottom=ba, width=bar_w, alpha=0.35, color=bc)

            n = len(confs)
            ax.set_title(
                f"{label} | {split}\nECE = {ece:.4f}  (n={n})",
                fontsize=10,
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("P([COPY] first token)", fontsize=8)
            if si == 0:
                ax.set_ylabel("Fraction positive (doc in context)", fontsize=8)
            if mi == 0 and si == 0:
                ax.legend(fontsize=7)

    fig.suptitle(
        "Reliability Diagrams: SFT vs DPO vs DPO-balanced\n"
        "[COPY] token calibration  |  MS MARCO 3-shot ICL",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Reliability diagram grid saved to: {output_path}")


# ---------------------------------------------------------------------------
# Plot 2: 3×2 confidence histogram grid
# ---------------------------------------------------------------------------

def plot_histogram_grid(
    all_data: Dict,
    output_path: str,
):
    n_models = len(MODELS)
    n_splits = len(SPLITS)

    fig, axes = plt.subplots(
        n_models, n_splits,
        figsize=(5 * n_splits, 3.5 * n_models),
        squeeze=False,
    )

    for mi, m in enumerate(MODELS):
        label = m["label"]
        color = m["color"]
        for si, split in enumerate(SPLITS):
            ax = axes[mi][si]
            data = all_data.get(label, {}).get(split, {})
            if not data:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_title(f"{label} | {split}")
                continue

            confs      = np.array(data["confidences"])
            labels_arr = np.array(data["labels"])
            mean_pos   = confs[labels_arr == 1].mean() if (labels_arr == 1).any() else 0
            mean_neg   = confs[labels_arr == 0].mean() if (labels_arr == 0).any() else 0

            ax.hist(confs[labels_arr == 1], bins=40, alpha=0.65,
                    color="steelblue", density=True,
                    label=f"doc in ctx (μ={mean_pos:.3f})")
            ax.hist(confs[labels_arr == 0], bins=40, alpha=0.65,
                    color="tomato", density=True,
                    label=f"no match  (μ={mean_neg:.3f})")

            ece = data.get("ece", float("nan"))
            ax.set_title(
                f"{label} | {split}  ECE={ece:.4f}", fontsize=10
            )
            ax.set_xlabel("P([COPY] first token)", fontsize=8)
            if si == 0:
                ax.set_ylabel("Density", fontsize=8)
            ax.legend(fontsize=7)

    fig.suptitle(
        "P([COPY]) distributions: SFT vs DPO vs DPO-balanced",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Histogram grid saved to: {output_path}")


# ---------------------------------------------------------------------------
# Plot 3: ECE bar chart summary
# ---------------------------------------------------------------------------

def plot_ece_bar(summary: List[Dict], output_path: str):
    """Bar chart: ECE per model, grouped by split."""
    splits  = SPLITS
    n_s     = len(splits)
    n_m     = len(MODELS)
    labels  = [m["label"] for m in MODELS]
    colors  = [m["color"]  for m in MODELS]

    x   = np.arange(n_s)
    w   = 0.22
    off = np.linspace(-(n_m - 1) / 2 * w, (n_m - 1) / 2 * w, n_m)

    fig, ax = plt.subplots(figsize=(7, 4))
    for mi, (label, color) in enumerate(zip(labels, colors)):
        ece_vals = []
        for split in splits:
            row = next(
                (r for r in summary
                 if r["model"] == label and r["split"] == split),
                {}
            )
            ece_vals.append(row.get("ece", float("nan")))
        bars = ax.bar(x + off[mi], ece_vals, width=w, color=color,
                      alpha=0.85, label=label)
        for bar, v in zip(bars, ece_vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("ECE (lower is better)")
    ax.set_title("[COPY] Token ECE: SFT vs DPO vs DPO-balanced")
    ax.set_ylim(0, max(
        r.get("ece", 0) for r in summary if r.get("ece") is not None
    ) * 1.3 + 0.05)
    ax.axhline(0, color="black", lw=0.8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"ECE bar chart saved to: {output_path}")


# ---------------------------------------------------------------------------
# Plot 4: Generation metrics bar chart
# ---------------------------------------------------------------------------

def plot_generation_bar(summary: List[Dict], output_path: str):
    """Bar chart: Hit@1 / context_dep Hit@1 / all_noise Hit@1 per model."""
    eval_rows = [r for r in summary if r.get("hit@1") is not None]
    if not eval_rows:
        print("No eval results found, skipping generation bar chart.")
        return

    metrics = [
        ("hit@1",              "Overall Hit@1"),
        ("context_dep_hit@1",  "context_dep Hit@1"),
        ("all_noise_hit@1",    "all_noise Hit@1"),
        ("context_dep_copy@1", "context_dep Copy@1"),
    ]

    n_m  = len(MODELS)
    n_me = len(metrics)
    x    = np.arange(n_me)
    w    = 0.22
    off  = np.linspace(-(n_m - 1) / 2 * w, (n_m - 1) / 2 * w, n_m)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for mi, m in enumerate(MODELS):
        label = m["label"]
        color = m["color"]
        # pick test split only
        row = next(
            (r for r in eval_rows
             if r["model"] == label and r.get("split") == "test"),
            {}
        )
        vals = [row.get(k, float("nan")) for k, _ in metrics]
        bars = ax.bar(x + off[mi], vals, width=w, color=color,
                      alpha=0.85, label=label)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([desc for _, desc in metrics], fontsize=9)
    ax.set_ylabel("Score (higher is better)")
    ax.set_title("Generation Metrics (test split): SFT vs DPO vs DPO-balanced")
    ax.set_ylim(0, 1.12)
    ax.axhline(1, color="grey", lw=0.6, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Generation bar chart saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="3-model ECE comparison")
    p.add_argument("--ece_root",   default="./ece_results")
    p.add_argument("--eval_dir",   default="./eval_results")
    p.add_argument("--output_dir", default="./ece_results")
    p.add_argument("--n_bins",     type=int, default=15)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Collect data ─────────────────────────────────────────────────────────
    all_data: Dict = {}   # all_data[model_label][split] = {confidences, labels, ece, ...}
    summary:  List[Dict] = []

    for m in MODELS:
        label = m["label"]
        all_data[label] = {}

        for split in SPLITS:
            rec = load_ece_records(args.ece_root, m["dir"], split)
            if not rec:
                print(f"  [WARN] No ECE data for {label}/{split}")
                continue

            records    = rec.get("records", [])
            confs      = np.array([r["copy_prob"]   for r in records], dtype=np.float32)
            labels_arr = np.array([r["label"]        for r in records], dtype=np.float32)

            ece, *_ = compute_ece(confs, labels_arr, args.n_bins)

            all_data[label][split] = {
                "confidences": confs,
                "labels":      labels_arr,
                "ece":         ece,
            }

            # per-pattern means
            pat      = [r["pattern"] for r in records]
            cd_mask  = np.array([p == "context_dependent" for p in pat])
            an_mask  = np.array([p == "all_noise"         for p in pat])
            mean_cd  = float(confs[cd_mask].mean()) if cd_mask.any() else float("nan")
            mean_an  = float(confs[an_mask].mean()) if an_mask.any() else float("nan")

            summary.append({
                "model":       label,
                "split":       split,
                "ece":         ece,
                "spearman_rho":rec.get("spearman_rho"),
                "n":           len(records),
                "mean_p_copy_context_dep": mean_cd,
                "mean_p_copy_all_noise":   mean_an,
            })

        # ── Generation metrics (test split only) ─────────────────────────────
        eval_rec = load_eval_results(args.eval_dir, m["tag"], "test")
        if eval_rec:
            pp = eval_rec.get("per_pattern", {})
            cd = pp.get("context_dependent", {})
            an = pp.get("all_noise",         {})
            summary.append({
                "model":              label,
                "split":              "test",
                "hit@1":              eval_rec.get("hit@1"),
                "hit@10":             eval_rec.get("hit@10"),
                "context_dep_hit@1":  cd.get("hit@1"),
                "context_dep_hit@10": cd.get("hit@10"),
                "context_dep_copy@1": cd.get("copy_top1"),
                "all_noise_hit@1":    an.get("hit@1"),
                "all_noise_hit@10":   an.get("hit@10"),
                "n_eval":             eval_rec.get("total"),
            })

    # ── Print summary table ───────────────────────────────────────────────────
    _print_summary(summary)

    # ── Save summary JSON ─────────────────────────────────────────────────────
    out_json = os.path.join(args.output_dir, "comparison_summary.json")
    with open(out_json, "w") as f:
        def _ser(o):
            if isinstance(o, np.floating): return float(o)
            if isinstance(o, np.integer):  return int(o)
            raise TypeError(type(o))
        json.dump(summary, f, indent=2, default=_ser)
    print(f"\nSummary JSON saved to: {out_json}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_reliability_grid(
        all_data, args.n_bins,
        os.path.join(args.output_dir, "comparison_3model.png"),
    )
    plot_histogram_grid(
        all_data,
        os.path.join(args.output_dir, "comparison_histogram.png"),
    )
    plot_ece_bar(
        summary,
        os.path.join(args.output_dir, "comparison_ece_bar.png"),
    )
    plot_generation_bar(
        summary,
        os.path.join(args.output_dir, "comparison_generation_bar.png"),
    )


def _print_summary(summary: List[Dict]):
    # ECE table
    print("\n" + "=" * 72)
    print("ECE SUMMARY")
    print("-" * 72)
    print(f"{'Model':<18} {'Split':<12} {'ECE':>8} {'ρ':>8} {'P([C]|ctx)':>12} {'P([C]|noise)':>14}")
    print("-" * 72)
    for r in summary:
        if "ece" not in r or r.get("hit@1") is not None:
            continue
        print(
            f"{r['model']:<18} {r['split']:<12} "
            f"{r['ece']:>8.4f} "
            f"{(r.get('spearman_rho') or float('nan')):>8.4f} "
            f"{(r.get('mean_p_copy_context_dep') or float('nan')):>12.4f} "
            f"{(r.get('mean_p_copy_all_noise') or float('nan')):>14.4f}"
        )

    # Generation table
    eval_rows = [r for r in summary if r.get("hit@1") is not None]
    if eval_rows:
        print("\n" + "=" * 72)
        print("GENERATION METRICS  (test split, 1000 samples)")
        print("-" * 72)
        print(
            f"{'Model':<18} {'H@1':>6} {'H@10':>6} "
            f"{'ctx H@1':>9} {'ctx H@10':>10} {'ctx Copy@1':>11} "
            f"{'ns H@1':>8} {'ns H@10':>8}"
        )
        print("-" * 72)
        for r in eval_rows:
            def _f(v): return f"{v:.4f}" if v is not None else "  N/A"
            print(
                f"{r['model']:<18} {_f(r.get('hit@1')):>6} {_f(r.get('hit@10')):>6} "
                f"{_f(r.get('context_dep_hit@1')):>9} {_f(r.get('context_dep_hit@10')):>10} "
                f"{_f(r.get('context_dep_copy@1')):>11} "
                f"{_f(r.get('all_noise_hit@1')):>8} {_f(r.get('all_noise_hit@10')):>8}"
            )
    print("=" * 72)


if __name__ == "__main__":
    main()
