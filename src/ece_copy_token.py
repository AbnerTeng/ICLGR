"""ECE curve study for [COPY] token log probability.

The model is trained to emit "[COPY] <id>" as its FIRST token(s) when the
positive document IS present in the context window (pattern=context_dependent).
When no matching doc is in context (pattern=all_noise) it must NOT emit [COPY].

For each test sample we run a single forward pass (no generation) and read
the log-probability the model assigns to "[COPY]" being the first output token.
Ground-truth label = 1 for context_dependent samples, 0 for all_noise.

Settings studied
----------------
  context_dependent  – The positive doc IS in the context window.
                       Label = 1; model should output [COPY].

  all_noise          – Context contains only unrelated (noise) documents.
                       The query's correct doc is absent but the model can
                       fall back on its parametric memory to answer.
                       Label = 0; model should NOT output [COPY].

  ctx_no_answer      – (new) Context contains noise docs taken from a
                       *different* query's sample (cross-query pairing).
                       Neither the context documents nor parametric memory
                       for the foreign query provides the correct answer.
                       This is the "genuinely unknown" scenario: both recall
                       paths are blocked.  Label = 0.
                       Enable with --include_ctx_no_answer.

Usage:
    python -m src.ece_copy_token \
        --dataset_name Lala8383/msmarco-item-id-3shot-v4_128k \
        --split test \
        --model_name Abner0803/Qwen3-1.7B-icl-3shot-v4_128k-copy_tag-dpo \
        --max_samples 2000 \
        --n_bins 15 \
        --output_dir ./ece_results

    # Include the new ctx_no_answer scenario:
    python -m src.ece_copy_token \
        --dataset_name Lala8383/msmarco-item-id-3shot-v4_128k \
        --split test \
        --model_name Abner0803/Qwen3-1.7B-icl-3shot-v4_128k-copy_tag-dpo \
        --max_samples 2000 \
        --n_bins 15 \
        --output_dir ./ece_results \
        --include_ctx_no_answer
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Patterns where the positive doc IS in context → label 1
# label=1  → model SHOULD output [COPY] (answer doc is in the context window)
# label=0  → model should NOT output [COPY] (all_noise: no matching doc)
# ctx_no_answer is also label=0: constructed at eval-time by cross-matching
# queries from all_noise samples so that neither context nor parametric memory
# provides the answer (see build_ctx_no_answer_samples()).
# ---------------------------------------------------------------------------
COPY_POSITIVE_PATTERNS = {"context_dependent"}
COPY_NEGATIVE_PATTERNS = {"all_noise", "ctx_no_answer"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_prompt(user_content: str, tokenizer) -> str:
    """Wrap user content in the Qwen3 chat template used during training."""
    messages = [{"role": "user", "content": user_content}]
    # apply_chat_template adds the assistant turn prefix automatically when
    # add_generation_prompt=True, so the model sees "<|im_start|>assistant\n"
    # as the very next tokens it must predict.
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def extract_query_from_content(user_content: str) -> str:
    """Extract the target query text from the user-turn content.

    Both v3 ("Query: <q>\\nAnswer:") and v4 ("Query: <q>\\nThe most relevant
    ID is:") formats embed the query as the last "Query: " line.  We take the
    last occurrence so that few-shot examples in the prompt are ignored.
    """
    idx = user_content.rfind("Query: ")
    if idx == -1:
        return ""
    rest = user_content[idx + len("Query: "):]
    return rest.split("\n")[0].strip()


def replace_query_in_content(user_content: str, new_query: str) -> str:
    """Replace the last "Query: <text>" line in user_content with new_query.

    Returns the original string unchanged if the marker is not found.
    """
    idx = user_content.rfind("Query: ")
    if idx == -1:
        return user_content
    eol = user_content.find("\n", idx)
    replacement = f"Query: {new_query}"
    if eol == -1:
        return user_content[:idx] + replacement
    return user_content[:idx] + replacement + user_content[eol:]


def build_ctx_no_answer_samples(noise_items: List[Dict]) -> List[Dict]:
    """Build synthetic "ctx_no_answer" samples from all_noise items.

    Strategy — circular-shift cross-query pairing:
      Sample i keeps its original context documents (noise docs for query_i)
      but receives the query text from sample i+1 (mod N).

    Why this creates a "genuinely unknown" scenario:
      • The context docs are noise for query_{i+1}: they are NEVER the correct
        document for query_{i+1}, so the context provides no answer.
      • The model's parametric memory was not trained to map query_{i+1} to any
        doc ID present in this sample's context, so memory also cannot help.
      • Expected calibrated behaviour: P([COPY]) should remain LOW (≈ 0),
        because copying from this unrelated context cannot retrieve the answer.
      • If P([COPY]) is unexpectedly HIGH the model is spuriously triggered by
        the mere presence of context documents, regardless of their relevance.

    All returned items have metadata.pattern = "ctx_no_answer" and label = 0.
    """
    if len(noise_items) < 2:
        return []

    # Extract the target query from each all_noise item
    queries = [extract_query_from_content(
        (item.get("conversations") or [{}])[0].get("content", "")
    ) for item in noise_items]

    synthetic: List[Dict] = []
    n = len(noise_items)
    for i, item in enumerate(noise_items):
        # Take the query from the next sample (circular)
        new_query = queries[(i + 1) % n]
        if not new_query:
            continue

        orig_convs = item.get("conversations") or []
        if not orig_convs:
            continue
        orig_content = orig_convs[0].get("content", "")
        new_content = replace_query_in_content(orig_content, new_query)

        # Build a synthetic item: same context docs, swapped query, label=0
        new_item: Dict = {
            **item,
            "conversations": [
                {"role": orig_convs[0].get("role", "user"), "content": new_content},
                # Keep the assistant answer slot for record-keeping (not used for label)
                *(orig_convs[1:]),
            ],
            "metadata": {
                **(item.get("metadata") or {}),
                "pattern": "ctx_no_answer",
                # Track provenance: which sample donated the query
                "cross_query_src": (i + 1) % n,
            },
        }
        synthetic.append(new_item)

    return synthetic


def get_copy_first_token_ids(tokenizer) -> List[int]:
    """Return the token-id(s) that could be the first token of '[COPY]'.

    Qwen3 usually splits '[COPY]' into multiple tokens.  We only need the
    *first* token because that is what the model generates at step-0.
    We also try ' [COPY]' (leading space) in case the tokenizer merges them.
    """
    candidates = set()
    for text in ("[COPY]", " [COPY]", "[COPY] "):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            candidates.add(ids[0])
    return sorted(candidates)


@torch.no_grad()
def compute_copy_logprob_batch(
    prompts: List[str],
    copy_first_token_ids: List[int],
    model,
    tokenizer,
    device: torch.device,
) -> List[float]:
    """Return log P([COPY] first token) for each prompt in the batch.

    We run a forward pass on the prompt and look at the logits at the LAST
    position (which corresponds to the first generated token).
    """
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    ).to(device)

    outputs = model(**enc)
    # logits: (batch, seq_len, vocab_size)
    # Last non-padding position per sample → first generation step
    # Because padding is on the LEFT (padding_side="left") we take [:, -1, :]
    last_logits = outputs.logits[:, -1, :]  # (batch, vocab)

    log_probs = F.log_softmax(last_logits.float(), dim=-1)

    copy_ids_tensor = torch.tensor(copy_first_token_ids, device=device)
    # Sum probabilities over all candidate first tokens of [COPY]
    copy_log_probs = torch.logsumexp(log_probs[:, copy_ids_tensor], dim=-1)  # (batch,)

    return copy_log_probs.cpu().tolist()


# ---------------------------------------------------------------------------
# ECE computation
# ---------------------------------------------------------------------------

def compute_ece(
    confidences: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute ECE and per-bin statistics.

    Returns:
        ece          – scalar ECE
        bin_mids     – bin midpoints (for plotting)
        bin_confs    – mean confidence per bin
        bin_accs     – fraction of positives per bin
        bin_weights  – fraction of samples per bin
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_mids = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    bin_confs = np.zeros(n_bins)
    bin_accs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= lo) & (confidences < hi)
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_counts[i] = mask.sum()
        bin_confs[i] = confidences[mask].mean()
        bin_accs[i] = labels[mask].mean()

    bin_weights = bin_counts / bin_counts.sum()
    ece = float(np.sum(bin_weights * np.abs(bin_confs - bin_accs)))
    return ece, bin_mids, bin_confs, bin_accs, bin_weights


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_ece(
    confidences: np.ndarray,
    labels: np.ndarray,
    n_bins: int,
    output_path: str,
    model_name: str,
    split: str,
):
    ece, bin_mids, bin_confs, bin_accs, bin_weights = compute_ece(
        confidences, labels, n_bins
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: reliability diagram
    ax = axes[0]
    bar_width = 1.0 / n_bins * 0.9
    mask = bin_weights > 0
    ax.bar(
        bin_mids[mask],
        bin_accs[mask],
        width=bar_width,
        alpha=0.7,
        color="steelblue",
        label="Fraction positive (label=1)",
    )
    ax.plot([0, 1], [0, 1], "r--", lw=1.5, label="Perfect calibration")
    ax.set_xlabel("Model confidence P([COPY] first token)")
    ax.set_ylabel("Fraction of samples with doc in context")
    ax.set_title(f"Reliability Diagram\nECE = {ece:.4f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)

    # Gap bars (calibration error per bin)
    gap = bin_confs[mask] - bin_accs[mask]
    bar_colors = ["#d62728" if g > 0 else "#2ca02c" for g in gap]
    for bm, g, bc in zip(bin_mids[mask], gap, bar_colors):
        ax.bar(bm, g, bottom=bin_accs[mask][gap == g][0], width=bar_width,
               alpha=0.35, color=bc)

    # Right: confidence histogram
    ax2 = axes[1]
    ax2.hist(confidences[labels == 1], bins=40, alpha=0.6, color="steelblue",
             density=True, label="doc in context (label=1)")
    ax2.hist(confidences[labels == 0], bins=40, alpha=0.6, color="tomato",
             density=True, label="doc NOT in context (label=0)")
    ax2.set_xlabel("P([COPY] first token)")
    ax2.set_ylabel("Density")
    ax2.set_title("Confidence distribution by label")
    ax2.legend(fontsize=8)

    short_model = model_name.split("/")[-1]
    fig.suptitle(
        f"[COPY] token ECE study — {short_model} | split={split} | n={len(confidences)}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="ECE curve for [COPY] token log prob")
    p.add_argument("--dataset_name", default="Lala8383/msmarco-item-id-3shot-v4_128k")
    p.add_argument("--split", default="test", choices=["test", "icl_test"])
    p.add_argument("--model_name", default="Abner0803/Qwen3-1.7B-icl-3shot-v4_128k-copy_tag-dpo")
    p.add_argument("--max_samples", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_bins", type=int, default=15)
    p.add_argument("--output_dir", default="./ece_results")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--cache_dir", default=None)
    p.add_argument(
        "--include_ctx_no_answer",
        action="store_true",
        default=False,
        help=(
            "Add a synthetic 'ctx_no_answer' evaluation scenario.  "
            "Built from all_noise samples by circular-shift cross-query pairing: "
            "each sample keeps its original noise context documents but receives "
            "the query from the next sample.  Neither the context nor parametric "
            "memory can supply the correct answer.  Label = 0 (should not [COPY]).  "
            "Measures whether P([COPY]) stays low when the model is genuinely stuck."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load tokenizer & model ──────────────────────────────────────────────
    print(f"Loading tokenizer from {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        padding_side="left",
        trust_remote_code=True,
        cache_dir=args.cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    copy_first_token_ids = get_copy_first_token_ids(tokenizer)
    print(f"[COPY] first-token IDs: {copy_first_token_ids} → "
          f"{[tokenizer.decode([t]) for t in copy_first_token_ids]}")

    print(f"Loading model from {args.model_name} ...")
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map={"": device} if device.type == "cuda" else None,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
    )
    model.eval()
    print("Model loaded.")

    # ── Load dataset ────────────────────────────────────────────────────────
    print(f"Loading dataset {args.dataset_name} split={args.split} ...")
    ds = load_dataset(
        args.dataset_name,
        split=args.split,
        cache_dir=args.cache_dir,
    )
    print(f"Dataset size: {len(ds)}")

    # Sample
    if args.max_samples and args.max_samples < len(ds):
        indices = np.random.default_rng(42).choice(len(ds), args.max_samples, replace=False)
        ds = ds.select(indices.tolist())
        print(f"Sampled {len(ds)} examples.")

    # ── Forward passes ──────────────────────────────────────────────────────
    all_log_probs: List[float] = []
    all_labels: List[int] = []
    all_patterns: List[str] = []
    raw_records: List[Dict] = []

    def get_label(item) -> int:
        """Return 1 if answer doc is in context (should produce [COPY]), else 0.

        Returns -1 for samples with unrecognised patterns so they can be skipped.
        """
        meta = item.get("metadata") or {}
        pattern = meta.get("pattern", "")
        if pattern in COPY_POSITIVE_PATTERNS:
            return 1
        if pattern in COPY_NEGATIVE_PATTERNS:
            return 0
        # Fallback: infer from the assistant's ground-truth answer
        convs = item.get("conversations") or []
        if len(convs) >= 2:
            ans = convs[1].get("content", "")
            return 1 if ans.strip().startswith("[COPY]") else 0
        return -1  # unknown — skip

    def get_user_content(item) -> str:
        convs = item.get("conversations") or []
        if convs:
            return convs[0].get("content", "")
        return item.get("text", "")

    batch_prompts, batch_labels, batch_patterns, batch_items = [], [], [], []

    def flush_batch():
        nonlocal batch_prompts, batch_labels, batch_patterns, batch_items
        if not batch_prompts:
            return
        log_probs = compute_copy_logprob_batch(
            batch_prompts, copy_first_token_ids, model, tokenizer, device
        )
        for lp, lab, pat, item in zip(log_probs, batch_labels, batch_patterns, batch_items):
            prob = float(np.exp(lp))
            all_log_probs.append(lp)
            all_labels.append(lab)
            all_patterns.append(pat)
            raw_records.append({
                "pattern": pat,
                "label": lab,
                "copy_log_prob": lp,
                "copy_prob": prob,
                "true_answer": (item.get("conversations") or [{}])[1].get("content", "")
                    if item.get("conversations") else "",
            })
        batch_prompts.clear()
        batch_labels.clear()
        batch_patterns.clear()
        batch_items.clear()

    # Accumulate all_noise items for later ctx_no_answer construction
    noise_items_for_cross: List[Dict] = []

    for item in tqdm(ds, desc="Computing copy log-probs"):
        user_content = get_user_content(item)
        if not user_content:
            continue
        label = get_label(item)
        if label == -1:
            continue  # skip unrecognised patterns
        prompt = build_prompt(user_content, tokenizer)
        meta = item.get("metadata") or {}
        pattern = meta.get("pattern", "unknown")

        batch_prompts.append(prompt)
        batch_labels.append(label)
        batch_patterns.append(pattern)
        batch_items.append(item)

        # Stash all_noise items so we can build ctx_no_answer samples later
        if args.include_ctx_no_answer and pattern == "all_noise":
            noise_items_for_cross.append(item)

        if len(batch_prompts) >= args.batch_size:
            flush_batch()

    flush_batch()

    # ── ctx_no_answer pass (optional) ───────────────────────────────────────
    if args.include_ctx_no_answer:
        print(
            f"\nBuilding ctx_no_answer samples from {len(noise_items_for_cross)} "
            "all_noise items (circular-shift cross-query pairing) ..."
        )
        ctx_no_answer_items = build_ctx_no_answer_samples(noise_items_for_cross)
        print(f"  Generated {len(ctx_no_answer_items)} ctx_no_answer samples.")

        if ctx_no_answer_items:
            for item in tqdm(ctx_no_answer_items, desc="ctx_no_answer forward passes"):
                user_content = get_user_content(item)
                if not user_content:
                    continue
                prompt = build_prompt(user_content, tokenizer)
                batch_prompts.append(prompt)
                batch_labels.append(0)   # doc NOT in context → should not [COPY]
                batch_patterns.append("ctx_no_answer")
                batch_items.append(item)

                if len(batch_prompts) >= args.batch_size:
                    flush_batch()

            flush_batch()

    print(f"\nCollected {len(all_labels)} samples.")
    label_arr = np.array(all_labels)
    print(f"  label=1 (doc in ctx): {label_arr.sum()} ({label_arr.mean():.2%})")
    print(f"  label=0 (no doc):     {(1 - label_arr).sum()} ({(1 - label_arr).mean():.2%})")

    # Pattern breakdown
    from collections import Counter
    pat_counts = Counter(all_patterns)
    print("  Pattern counts:", dict(pat_counts))

    # ── Confidence = P([COPY] first token) ─────────────────────────────────
    confidences = np.exp(np.array(all_log_probs, dtype=np.float32))

    # ── ECE ─────────────────────────────────────────────────────────────────
    ece, bin_mids, bin_confs, bin_accs, bin_weights = compute_ece(
        confidences, label_arr, args.n_bins
    )
    print(f"\nECE = {ece:.4f}")

    # AUC-ROC as additional metric
    from scipy.stats import spearmanr
    rho, pval = spearmanr(confidences, label_arr)
    print(f"Spearman ρ (conf vs label): {rho:.4f}  (p={pval:.3e})")

    # ── Per-pattern summary stats ────────────────────────────────────────────
    unique_pats = sorted(set(all_patterns))
    per_pattern_stats: Dict = {}
    for pat in unique_pats:
        idx = [i for i, p in enumerate(all_patterns) if p == pat]
        if not idx:
            continue
        pc = confidences[idx]
        pl = label_arr[idx]
        n_bins_pat = min(args.n_bins, max(5, len(idx) // 20))
        pat_ece, *_ = compute_ece(pc, pl, n_bins_pat)
        per_pattern_stats[pat] = {
            "n": len(idx),
            "mean_copy_prob": float(pc.mean()),
            "std_copy_prob": float(pc.std()),
            "frac_positive": float(pl.mean()),
            "ece": float(pat_ece),
        }

    # ── Save results ─────────────────────────────────────────────────────────
    split_tag = args.split
    model_tag = args.model_name.replace("/", "_")

    results_path = os.path.join(args.output_dir, f"ece_results_{split_tag}.json")
    with open(results_path, "w") as f:
        json.dump(
            {
                "ece": ece,
                "spearman_rho": float(rho),
                "n_samples": len(all_labels),
                "n_positive": int(label_arr.sum()),
                "model": args.model_name,
                "dataset": args.dataset_name,
                "split": args.split,
                "ctx_no_answer_included": args.include_ctx_no_answer,
                "per_pattern": per_pattern_stats,
                "bins": [
                    {
                        "bin_mid": float(bm),
                        "mean_conf": float(bc),
                        "fraction_positive": float(ba),
                        "weight": float(bw),
                    }
                    for bm, bc, ba, bw in zip(bin_mids, bin_confs, bin_accs, bin_weights)
                ],
                "records": raw_records,
            },
            f,
            indent=2,
        )
    print(f"Results saved to: {results_path}")

    # Per-pattern ECE
    print("\nPer-pattern analysis:")
    for pat in unique_pats:
        stats = per_pattern_stats.get(pat, {})
        print(
            f"  [{pat:<20}] n={stats.get('n', 0):5d}  "
            f"mean_P([COPY])={stats.get('mean_copy_prob', 0):.3f}  "
            f"frac_pos={stats.get('frac_positive', 0):.3f}  "
            f"ECE={stats.get('ece', 0):.4f}"
        )

    # Highlight the ctx_no_answer finding explicitly
    if args.include_ctx_no_answer and "ctx_no_answer" in per_pattern_stats:
        na_stats = per_pattern_stats["ctx_no_answer"]
        noise_stats = per_pattern_stats.get("all_noise", {})
        print("\n── ctx_no_answer vs all_noise comparison ──────────────────────────────")
        print(
            f"  all_noise     mean P([COPY]) = {noise_stats.get('mean_copy_prob', float('nan')):.4f}  "
            f"(n={noise_stats.get('n', 0)}, model CAN fall back to parametric memory)"
        )
        print(
            f"  ctx_no_answer mean P([COPY]) = {na_stats['mean_copy_prob']:.4f}  "
            f"(n={na_stats['n']}, NEITHER context NOR memory provides the answer)"
        )
        delta = na_stats["mean_copy_prob"] - noise_stats.get("mean_copy_prob", 0.0)
        # Use a single threshold for both the label and interpretation text
        direction = "HIGHER" if delta > 0.005 else ("LOWER" if delta < -0.005 else "similar")
        print(
            f"  Δ = {delta:+.4f}  →  ctx_no_answer P([COPY]) is {direction} than all_noise"
        )
        if delta > 0.005:
            print(
                "  Interpretation: model spuriously raises P([COPY]) when confused "
                "(context present but irrelevant), suggesting the copy trigger is "
                "partially driven by the presence of context docs, not their relevance."
            )
        elif delta < -0.005:
            print(
                "  Interpretation: model actually suppresses P([COPY]) more strongly "
                "when context docs are completely unrelated to the query."
            )
        else:
            print(
                "  Interpretation: P([COPY]) is insensitive to whether context docs "
                "are topically related or completely unrelated — suppression is robust."
            )

    # ── Plot ─────────────────────────────────────────────────────────────────
    plot_path = os.path.join(args.output_dir, f"ece_plot_{split_tag}.png")
    plot_ece(confidences, label_arr, args.n_bins, plot_path, args.model_name, args.split)

    # Per-pattern confidence histograms
    # Color coding per pattern to make the ctx_no_answer scenario visually distinct
    pat_colors = {
        "context_dependent": ("steelblue", "steelblue"),   # all label=1
        "all_noise":         ("tomato",    "tomato"),       # all label=0
        "ctx_no_answer":     ("darkorange","darkorange"),   # all label=0, new scenario
    }
    fig, axes = plt.subplots(
        1, len(unique_pats), figsize=(4 * len(unique_pats), 4), sharey=True
    )
    if len(unique_pats) == 1:
        axes = [axes]
    for ax, pat in zip(axes, unique_pats):
        idx = [i for i, p in enumerate(all_patterns) if p == pat]
        pc = confidences[idx]
        pl = label_arr[idx]
        col_pos, col_neg = pat_colors.get(pat, ("steelblue", "tomato"))
        stats = per_pattern_stats.get(pat, {})
        mean_p = stats.get("mean_copy_prob", float("nan"))

        if pl.sum() > 0:
            ax.hist(pc[pl == 1], bins=30, alpha=0.65, color=col_pos,
                    density=True, label="label=1 (doc in ctx)")
        if (1 - pl).sum() > 0:
            ax.hist(pc[pl == 0], bins=30, alpha=0.65, color=col_neg,
                    density=True, label="label=0 (no doc)")

        # Annotate mean P([COPY]) as a vertical line
        ax.axvline(mean_p, color="black", linestyle="--", lw=1.2,
                   label=f"mean={mean_p:.3f}")

        subtitle = ""
        if pat == "ctx_no_answer":
            subtitle = "\n(cross-query: no ctx answer, no mem)"
        ax.set_title(f"{pat}{subtitle}\nn={len(idx)}", fontsize=8)
        ax.set_xlabel("P([COPY])")
        if ax == axes[0]:
            ax.set_ylabel("Density")
        ax.legend(fontsize=7)

    extra = " | +ctx_no_answer" if args.include_ctx_no_answer else ""
    fig.suptitle(f"[COPY] confidence by pattern{extra}", fontsize=11)
    fig.tight_layout()
    pp = os.path.join(args.output_dir, f"per_pattern_{split_tag}.png")
    fig.savefig(pp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Per-pattern plot saved to: {pp}")


if __name__ == "__main__":
    main()
