"""
Construct DPO pair datasets from inference output.

dpo.jsonl        — H@10=True, H@1=False: chosen=true_docid, rejected=predicted[0]
dpo_h1success.jsonl — H@1=True: chosen=predicted[0], rejected=predicted[1]

Usage:
    python src/construct_dpo_pairs.py \
        --inference_file retrieve_results/v4-5e-5-5epochs-copy-test.json \
        --test_file data/msmarco-icl-3shot-v4/test.jsonl \
        --output_dir data/dpo-v4-copy
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_file", required=True)
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    with open(args.inference_file) as f:
        inference = json.load(f)
    predictions = inference["predictions"]

    with open(args.test_file) as f:
        test_data = [json.loads(line) for line in f]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dpo_pairs = []
    h1success_pairs = []
    stats = {"dpo": 0, "h1success": 0, "skipped": 0}

    for pred, item in zip(predictions, test_data):
        user_content = item["conversations"][0]["content"]
        # store raw user content — chatml.default wraps it with ChatML tokens
        true_docid = pred["true_docid"]
        predicted = pred["predicted_docid"]
        hit1 = pred["hit_at_1"]
        hit10 = pred["hit_at_10"]

        # dpo: ranking failure — correct answer in top-10 but not top-1
        if hit10 and not hit1:
            dpo_pairs.append({
                "prompt": user_content,
                "chosen": true_docid,
                "rejected": predicted[0],
                "metadata": pred.get("metadata"),
            })
            stats["dpo"] += 1

        # dpo_h1success: model got it right — push beam 1 further from beam 2
        elif hit1 and len(predicted) > 1:
            h1success_pairs.append({
                "prompt": user_content,
                "chosen": predicted[0],
                "rejected": predicted[1],
                "metadata": pred.get("metadata"),
            })
            stats["h1success"] += 1

        else:
            stats["skipped"] += 1

    dpo_path = out_dir / "dpo.jsonl"
    with open(dpo_path, "w") as f:
        for pair in dpo_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    h1_path = out_dir / "dpo_h1success.jsonl"
    with open(h1_path, "w") as f:
        for pair in h1success_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"dpo.jsonl:          {stats['dpo']:>6} pairs  -> {dpo_path}")
    print(f"dpo_h1success.jsonl:{stats['h1success']:>6} pairs  -> {h1_path}")
    print(f"skipped (H@10=False, H@1=False): {stats['skipped']}")


if __name__ == "__main__":
    main()
