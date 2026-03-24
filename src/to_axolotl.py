import json
import os

import hydra
from omegaconf import DictConfig


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


@hydra.main(config_path="../configs", config_name="to_axolotl_conf", version_base=None)
def main(cfg: DictConfig) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)

    for split in cfg.splits:
        input_file = os.path.join(cfg.input_dir, f"{split}.jsonl")
        output_file = os.path.join(cfg.output_dir, f"{split}_axolotl.jsonl")

        with open(input_file, "r") as f:
            data = [json.loads(line) for line in f]

        convert_to_axolotl_format(data, output_file)
        print(f"Converted {split}: {input_file} -> {output_file}")


if __name__ == "__main__":
    main()
