"""Merge pseudo-query dataset with precomputed hard_negatives.

Pseudo dataset provides queries + indexing rows.
HN dataset provides per-doc hard_negatives (keyed by original-case title).

Join: pseudo.item (title) == hn.doc_id (title)
Output: data/<dst_dir>/ with train.jsonl, test.jsonl, icl_test.jsonl.
"""

import argparse
import json
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pseudo_file",
        default="data/msmarco_text-with_pseudo_query-100k-gr/train_with_pseudo.jsonl",
    )
    ap.add_argument("--hn_dir", default="data/msmarco-item-id-hardneg")
    ap.add_argument("--dst_dir", default="data/msmarco-pseudo-hardneg")
    args = ap.parse_args()

    pseudo_file = Path(args.pseudo_file)
    hn_dir = Path(args.hn_dir)
    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build title -> hard_negatives map from HN data
    print(f"Loading HN map from {hn_dir / 'train.jsonl'}")
    title_to_hn = {}
    with open(hn_dir / "train.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r.get("operation") == "indexing" and "hard_negatives" in r:
                title_to_hn[r["doc_id"]] = r["hard_negatives"]
    print(f"  {len(title_to_hn)} titles have hard_negatives")

    # 2. Walk pseudo file, attach hard_negatives to indexing rows
    n_queries = 0
    n_indexing = 0
    n_attached = 0
    n_missing = 0
    with open(pseudo_file) as fin, open(dst_dir / "train.jsonl", "w") as fout:
        for line in fin:
            r = json.loads(line)
            op = r.get("operation")
            if op == "indexing":
                n_indexing += 1
                hn = title_to_hn.get(r["item"])
                if hn is not None:
                    r["hard_negatives"] = hn
                    n_attached += 1
                else:
                    n_missing += 1
            elif op == "query":
                n_queries += 1
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {dst_dir / 'train.jsonl'}")
    print(
        f"  queries={n_queries}  indexing={n_indexing}  attached={n_attached}  missing={n_missing}"
    )

    # 3. Copy test.jsonl and icl_test.jsonl from HN dir (template loader requires all 3)
    for fname in ["test.jsonl", "icl_test.jsonl"]:
        shutil.copy(hn_dir / fname, dst_dir / fname)
        print(f"Copied {fname}")


if __name__ == "__main__":
    main()
