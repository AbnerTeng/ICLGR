"""
Build hard-negative DPO pairs for context-dependent H@1-success cases.

Fixes an inconsistency in the original dpo_h1success_cd_balanced.jsonl:
  - SFT trains the model to output "[COPY] <docid>" for context-dependent queries
  - But the original DPO pairs use plain docids (no [COPY]) for both chosen and
    rejected, because the inference trie constrains [COPY] away — this contradicts
    the SFT target and produces a weak gradient signal.

This script corrects that:
  chosen   = "[COPY] <target_id>"       (matches SFT training format)
  rejected = <output without [COPY]>    (hard negative, one of three strategies)

Three rejected strategies (combined by default):
  strip_copy       — rejected = target_id (same docid but without [COPY])
                     teaches "same content without [COPY] is wrong"
  zero_shot_unseen — rejected = wrong zero-shot prediction on unseen queries
                     (natural output when no context is provided)
  zero_shot_same   — rejected = wrong zero-shot prediction on the same query
                     (contrast: with-context→[COPY]right vs no-context→wrong)

Usage:
    # all three strategies combined (strip_copy only, no inference files needed):
    python src/build_hard_cd_dpo.py \\
        --h1success_file data/dpo-v4-copy-full/dpo_h1success.jsonl \\
        --dpo_file       data/dpo-v4-copy-full/dpo.jsonl \\
        --output_file    data/dpo-v4-copy-full/dpo_h1success_cd_hard.jsonl

    # with zero-shot inference files:
    python src/build_hard_cd_dpo.py \\
        --h1success_file       data/dpo-v4-copy-full/dpo_h1success.jsonl \\
        --dpo_file             data/dpo-v4-copy-full/dpo.jsonl \\
        --output_file          data/dpo-v4-copy-full/dpo_h1success_cd_hard.jsonl \\
        --zero_shot_unseen     retrieve_results/zero_shot_unseen.json \\
        --zero_shot_same       retrieve_results/zero_shot_same_queries.json \\
        --strategy all \\
        --seed 42
"""

import argparse
import json
import random
import re
from pathlib import Path


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def extract_context_docids(prompt: str) -> set[str]:
    """Extract identifiers listed under 'Identifier:' in the prompt's document block."""
    return set(re.findall(r"^Identifier:\s*(.+)$", prompt, re.MULTILINE))


def load_zero_shot_wrong_predictions(inference_file: str) -> list[str]:
    """Return predicted docids where H@1 is False from a zero-shot inference JSON."""
    with open(inference_file) as f:
        data = json.load(f)
    wrong = []
    for pred in data["predictions"]:
        if not pred["hit_at_1"] and pred["predicted_docid"]:
            wrong.append(pred["predicted_docid"][0])
    return wrong


def load_zero_shot_same_map(inference_file: str) -> dict[str, str]:
    """
    Return a mapping of query text → zero-shot wrong prediction.
    Expects the inference JSON to have a "text" field per prediction matching
    the user content used to build the DPO prompt.
    """
    with open(inference_file) as f:
        data = json.load(f)
    mapping = {}
    for pred in data["predictions"]:
        if not pred["hit_at_1"] and pred["predicted_docid"]:
            mapping[pred["text"].strip()] = pred["predicted_docid"][0]
    return mapping


def extract_query_from_prompt(prompt: str) -> str:
    """Pull the final 'Query: ...' line from the few-shot prompt."""
    match = re.search(r"Query:\s*(.+)\nAnswer:", prompt)
    return match.group(1).strip() if match else ""


def build_strip_copy_pairs(cd_pairs: list[dict]) -> list[dict]:
    """chosen=[COPY] docid, rejected=docid (same content, no [COPY])."""
    out = []
    for p in cd_pairs:
        target = p["metadata"]["target_id"]
        out.append({
            "prompt": p["prompt"],
            "chosen": f"[COPY] {target}",
            "rejected": target,
            "metadata": {**p["metadata"], "strategy": "strip_copy"},
        })
    return out


def build_zero_shot_unseen_pairs(
    cd_pairs: list[dict], wrong_preds: list[str], rng: random.Random
) -> list[dict]:
    """chosen=[COPY] docid, rejected=wrong zero-shot prediction on an unseen query."""
    if not wrong_preds:
        return []
    out = []
    for p in cd_pairs:
        target = p["metadata"]["target_id"]
        rejected = rng.choice(wrong_preds)
        # avoid accidentally picking the correct answer
        while rejected == target and len(wrong_preds) > 1:
            rejected = rng.choice(wrong_preds)
        out.append({
            "prompt": p["prompt"],
            "chosen": f"[COPY] {target}",
            "rejected": rejected,
            "metadata": {**p["metadata"], "strategy": "zero_shot_unseen"},
        })
    return out


def build_zero_shot_same_pairs(
    cd_pairs: list[dict],
    query_to_wrong: dict[str, str],
    fallback_preds: list[str],
    rng: random.Random,
) -> list[dict]:
    """chosen=[COPY] docid, rejected=zero-shot wrong prediction on the same query."""
    if not query_to_wrong and not fallback_preds:
        return []
    out = []
    for p in cd_pairs:
        target = p["metadata"]["target_id"]
        query = extract_query_from_prompt(p["prompt"])
        if query in query_to_wrong:
            rejected = query_to_wrong[query]
        elif fallback_preds:
            rejected = rng.choice(fallback_preds)
        else:
            continue
        if rejected == target:
            continue
        out.append({
            "prompt": p["prompt"],
            "chosen": f"[COPY] {target}",
            "rejected": rejected,
            "metadata": {**p["metadata"], "strategy": "zero_shot_same"},
        })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h1success_file", required=True)
    parser.add_argument("--dpo_file", required=True,
                        help="used to determine target output size")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--zero_shot_unseen", default=None,
                        help="zero-shot inference JSON on unseen queries")
    parser.add_argument("--zero_shot_same", default=None,
                        help="zero-shot inference JSON on the same test queries")
    parser.add_argument(
        "--strategy",
        default="all",
        choices=["strip_copy", "zero_shot_unseen", "zero_shot_same", "all"],
    )
    parser.add_argument("--target_size", type=int, default=None,
                        help="output size; defaults to line count of --dpo_file")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # load context-dependent H@1-success pairs
    all_h1 = load_jsonl(args.h1success_file)
    cd_pairs = [p for p in all_h1 if p["metadata"]["pattern"] == "context_dependent"]
    print(f"context_dependent h1success pairs: {len(cd_pairs)}")

    # target size
    if args.target_size is not None:
        target = args.target_size
    else:
        with open(args.dpo_file) as f:
            target = sum(1 for _ in f)
    print(f"target output size: {target}")

    # build pools per strategy
    use_all = args.strategy == "all"

    pools: dict[str, list[dict]] = {}

    if args.strategy in ("strip_copy", "all"):
        pools["strip_copy"] = build_strip_copy_pairs(cd_pairs)
        print(f"strip_copy pool: {len(pools['strip_copy'])}")

    if args.strategy in ("zero_shot_unseen", "all"):
        wrong_preds: list[str] = []
        if args.zero_shot_unseen:
            wrong_preds = load_zero_shot_wrong_predictions(args.zero_shot_unseen)
            print(f"zero_shot_unseen wrong predictions available: {len(wrong_preds)}")
        else:
            print("zero_shot_unseen: no inference file provided, skipping")
        if wrong_preds:
            pools["zero_shot_unseen"] = build_zero_shot_unseen_pairs(cd_pairs, wrong_preds, rng)
            print(f"zero_shot_unseen pool: {len(pools['zero_shot_unseen'])}")

    if args.strategy in ("zero_shot_same", "all"):
        query_map: dict[str, str] = {}
        fallback: list[str] = []
        if args.zero_shot_same:
            query_map = load_zero_shot_same_map(args.zero_shot_same)
            fallback = list(query_map.values())
            print(f"zero_shot_same query map size: {len(query_map)}")
        else:
            print("zero_shot_same: no inference file provided, skipping")
        if query_map:
            pools["zero_shot_same"] = build_zero_shot_same_pairs(cd_pairs, query_map, fallback, rng)
            print(f"zero_shot_same pool: {len(pools['zero_shot_same'])}")

    if not pools:
        raise ValueError("No strategies produced any pairs. Check your inputs.")

    # combine: if multiple strategies, allocate target evenly across them
    strategy_names = list(pools.keys())
    n_strategies = len(strategy_names)
    quota = target // n_strategies
    remainder = target % n_strategies

    final: list[dict] = []
    for i, name in enumerate(strategy_names):
        pool = pools[name]
        alloc = quota + (1 if i < remainder else 0)
        if len(pool) >= alloc:
            sampled = rng.sample(pool, alloc)
        else:
            print(f"warning: {name} pool has {len(pool)} < {alloc}, using all")
            sampled = pool
        final.extend(sampled)
        print(f"  {name}: {len(sampled)} pairs")

    rng.shuffle(final)

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for pair in final:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\nwrote {len(final)} pairs -> {out_path}")
    strategy_counts = {}
    for p in final:
        s = p["metadata"].get("strategy", "unknown")
        strategy_counts[s] = strategy_counts.get(s, 0) + 1
    for s, c in strategy_counts.items():
        print(f"  {s}: {c}")


if __name__ == "__main__":
    main()
