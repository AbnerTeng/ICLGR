import json
from argparse import ArgumentParser, Namespace

from datasets import load_from_disk


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/msmarco_icl_split_clean",
    )
    parser.add_argument("--dataset_name", type=str, default="msmarco")

    return parser.parse_args()


def convert_to_axolotl_format(dataset, output_path):
    with open(output_path, "w") as f:
        for sample in dataset:
            conv = {
                "conversations": [
                    {"role": "user", "content": sample["text"]},
                    {"role": "assistant", "content": sample["doc_id"]},
                ]
            }
            f.write(json.dumps(conv) + "\n")


if __name__ == "__main__":
    args = get_args()
    dataset = load_from_disk(args.input_path)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    icl_test_dataset = dataset["icl_test"]

    convert_to_axolotl_format(
        train_dataset, f"data/{args.dataset_name}_axolotl/train_axolotl.jsonl"
    )
    convert_to_axolotl_format(
        test_dataset, f"data/{args.dataset_name}_axolotl/test_axolotl.jsonl"
    )
    convert_to_axolotl_format(
        icl_test_dataset, f"data/{args.dataset_name}_axolotl/icl_test_axolotl.jsonl"
    )
