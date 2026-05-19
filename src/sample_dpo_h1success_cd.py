"""
Sample context_dependent pairs from dpo_h1success.jsonl to match dpo.jsonl size.

Output: dpo_h1success_cd_balanced.jsonl
  - only context_dependent pairs from h1success
  - randomly subsampled to --target_size (default: matches dpo.jsonl line count)

Usage:
    python src/sample_dpo_h1success_cd.py \
        --h1success_file data/dpo-v4-copy-full/dpo_h1success.jsonl \
        --dpo_file data/dpo-v4-copy-full/dpo.jsonl \
        --output_file data/dpo-v4-copy-full/dpo_h1success_cd_balanced.jsonl \
        [--target_size 5050] \
        [--seed 42]
"""

import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h1success_file", required=True)
    parser.add_argument("--dpo_file", required=True, help="used to infer target size if --target_size not set")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--target_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.h1success_file) as f:
        cd_pairs = [json.loads(line) for line in f if json.loads(line)["metadata"]["pattern"] == "context_dependent"]

    if args.target_size is not None:
        target = args.target_size
    else:
        with open(args.dpo_file) as f:
            target = sum(1 for _ in f)

    print(f"context_dependent h1success pairs available: {len(cd_pairs)}")
    print(f"target size: {target}")

    if len(cd_pairs) < target:
        print(f"warning: only {len(cd_pairs)} pairs available, using all of them")
        sampled = cd_pairs
    else:
        random.seed(args.seed)
        sampled = random.sample(cd_pairs, target)

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for pair in sampled:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"wrote {len(sampled)} pairs -> {out_path}")


if __name__ == "__main__":
    main()
