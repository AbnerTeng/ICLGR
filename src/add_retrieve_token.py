"""
Add [RETRIEVE] token to all_noise samples in the dataset.

For all_noise pattern, the assistant answer becomes:
    [RETRIEVE] {target_id}

For context_dependent pattern (already has [COPY]), no change.

Usage:
    python -m src.add_retrieve_token \
        --src Lala8383/msmarco-item-id-3shot-v4_128k \
        --out data/msmarco-icl-3shot-v4-retrieve \
        [--push --dst Lala8383/msmarco-item-id-3shot-v4_128k_retrieve]
"""

import argparse
import copy
import json
import os

from datasets import load_dataset, DatasetDict


def add_retrieve_token(example: dict) -> dict:
    if example["metadata"]["pattern"] != "all_noise":
        return example

    example = copy.deepcopy(example)
    assistant_turn = example["conversations"][-1]
    assert assistant_turn["role"] == "assistant"
    assistant_turn["content"] = f"[RETRIEVE] {assistant_turn['content']}"
    return example


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="Lala8383/msmarco-item-id-3shot-v4_128k")
    parser.add_argument("--out", default="data/msmarco-icl-3shot-v4-retrieve")
    parser.add_argument("--push", action="store_true", help="Push to HuggingFace Hub")
    parser.add_argument(
        "--dst", default="Lala8383/msmarco-item-id-3shot-v4_128k_retrieve"
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    splits = ["train", "test", "icl_test"]
    dataset_dict = {}
    for split in splits:
        print(f"Processing {split}...")
        ds = load_dataset(args.src, split=split)
        ds = ds.map(add_retrieve_token, desc=f"Adding [RETRIEVE] to {split}")

        n_retrieve = sum(
            1 for x in ds if x["conversations"][-1]["content"].startswith("[RETRIEVE]")
        )
        n_copy = sum(
            1 for x in ds if x["conversations"][-1]["content"].startswith("[COPY]")
        )
        print(f"  [RETRIEVE]={n_retrieve}  [COPY]={n_copy}  total={len(ds)}")

        out_file = os.path.join(args.out, f"{split}.jsonl")
        with open(out_file, "w", encoding="utf-8") as f:
            for ex in ds:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"  Saved -> {out_file}")

        dataset_dict[split] = ds

    if args.push:
        print(f"\nPushing to {args.dst} ...")
        DatasetDict(dataset_dict).push_to_hub(args.dst)
        print("Done.")
    else:
        print(
            f"\nSaved all splits to {args.out}/  (pass --push to also upload to HuggingFace)"
        )


if __name__ == "__main__":
    main()
