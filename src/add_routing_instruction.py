"""
Add explicit routing instruction to all_noise examples in the ICL dataset.

For all_noise samples, inserts before the final "Query: ..." line:
  "Note: If none of the documents are relevant to the query, generate the answer from memory."

Saves to a new directory without modifying the original dataset.
"""

import json
import os
from pathlib import Path

INSTRUCTION = (
    "Note: If none of the documents are relevant to the query, "
    "generate the answer from memory."
)

SRC_DIR = Path("./data/msmarco-icl-3shot-v4")
DST_DIR = Path("./data/msmarco-icl-3shot-v4-routing")


def add_instruction(user_content: str) -> str:
    marker = "\nQuery:"
    idx = user_content.rfind(marker)
    if idx == -1:
        return user_content
    return user_content[:idx] + f"\n{INSTRUCTION}" + user_content[idx:]


def process_file(src: Path, dst: Path) -> None:
    total = modified = 0
    dst.parent.mkdir(parents=True, exist_ok=True)

    with open(src) as f_in, open(dst, "w") as f_out:
        for line in f_in:
            item = json.loads(line)
            total += 1

            if (item.get("metadata") or {}).get("pattern") == "all_noise":
                convs = item["conversations"]
                convs[0]["content"] = add_instruction(convs[0]["content"])
                modified += 1

            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"{src.name}: {total} total, {modified} all_noise modified → {dst}")


if __name__ == "__main__":
    DST_DIR.mkdir(parents=True, exist_ok=True)

    for split in ("train.jsonl", "test.jsonl", "icl_test.jsonl"):
        src = SRC_DIR / split
        dst = DST_DIR / split
        if src.exists():
            process_file(src, dst)
        else:
            print(f"Skipping {src} (not found)")
