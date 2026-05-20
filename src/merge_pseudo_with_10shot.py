"""Merge pseudo-query dataset (flat GR) into a 10-shot ICL conversation dataset.

10shot file (ShareGPT format) provides ICL training examples.
Pseudo file (flat GR) provides additional indexing/query rows; we convert each
row to ShareGPT format using stage1_indexing / stage1_query templates, then
concatenate with the 10shot file.

Output: data/<dst_dir>/train.jsonl
"""
import argparse
import json
from pathlib import Path


def pseudo_to_conv(r):
    op = r.get("operation")
    item = r["item"]
    text = r["text"]
    if op == "indexing":
        user_content = f"## Task\nDocument: {text}\nAnswer:"
        pattern = "stage1_indexing"
    elif op == "query":
        user_content = f"## Task\nQuery: {text}\nAnswer:"
        pattern = "stage1_query"
    else:
        return None
    return {
        "conversations": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": item},
        ],
        "metadata": {"pattern": pattern, "target_id": item},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pseudo_file", default="data/msmarco_text-with_pseudo_query-100k-gr/train_with_pseudo.jsonl")
    ap.add_argument("--shot_file", default="data/msmarco-item-id-10shot_v4/train_10shot.jsonl")
    ap.add_argument("--dst_dir", default="data/msmarco-item-id-10shot_v4-with-pseudo")
    args = ap.parse_args()

    pseudo_file = Path(args.pseudo_file)
    shot_file = Path(args.shot_file)
    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    out = dst_dir / "train.jsonl"

    n_pseudo_i = n_pseudo_q = n_pseudo_skip = n_shot = 0
    with open(out, "w") as fout:
        print(f"Converting pseudo rows from {pseudo_file}")
        with open(pseudo_file) as f:
            for line in f:
                r = json.loads(line)
                conv = pseudo_to_conv(r)
                if conv is None:
                    n_pseudo_skip += 1
                    continue
                if r["operation"] == "indexing":
                    n_pseudo_i += 1
                else:
                    n_pseudo_q += 1
                fout.write(json.dumps(conv, ensure_ascii=False) + "\n")

        print(f"Appending shot rows from {shot_file}")
        with open(shot_file) as f:
            for line in f:
                if not line.endswith("\n"):
                    line += "\n"
                fout.write(line)
                n_shot += 1

    total = n_pseudo_i + n_pseudo_q + n_shot
    print(f"Wrote {out}")
    print(f"  pseudo_indexing={n_pseudo_i}  pseudo_query={n_pseudo_q}  skipped={n_pseudo_skip}")
    print(f"  shot={n_shot}  total={total}")


if __name__ == "__main__":
    main()
