"""Convert title-based identifiers in id_only ICL datasets to semantic IDs.

Substitutes in three locations:
  1. "Identifier: {title}" lines in Documents section
  2. "Answer: {title}" lines in Task examples
  3. Assistant response: "[COPY] {title}" or plain "{title}"
"""

import argparse
import json
import re
from pathlib import Path


def load_mapper(mapper_path: str) -> dict[str, str]:
    with open(mapper_path) as f:
        return json.load(f)


def substitute_content(user_content: str, asst_content: str, mapper: dict[str, str]):
    missing = set()

    def lookup(title: str) -> str:
        key = title.strip().lower()
        semid = mapper.get(key)
        if semid is None:
            missing.add(key)
            return title  # keep original if not found
        return semid

    # Replace "Identifier: {title}" — title runs to end of line
    def replace_identifier(m):
        return "Identifier: " + lookup(m.group(1))

    user_content = re.sub(r"^Identifier: (.+)$", replace_identifier, user_content, flags=re.MULTILINE)

    # Replace "Answer: {title}" — title runs to end of line (skip blank Answer: lines)
    def replace_answer(m):
        title = m.group(1)
        if not title.strip():
            return m.group(0)
        return "Answer: " + lookup(title)

    user_content = re.sub(r"^Answer: (.+)$", replace_answer, user_content, flags=re.MULTILINE)

    # Replace assistant content
    asst = asst_content.strip()
    if asst.startswith("[COPY] "):
        title = asst[len("[COPY] "):]
        asst_content = "[COPY] " + lookup(title)
    else:
        asst_content = lookup(asst)

    return user_content, asst_content, missing


def convert_file(src: Path, dst: Path, mapper: dict[str, str]):
    total = 0
    total_missing = set()
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            d = json.loads(line)
            convs = d["conversations"]
            user_idx = next(i for i, c in enumerate(convs) if c["role"] in ("user", "human"))
            asst_idx = next(i for i, c in enumerate(convs) if c["role"] in ("assistant", "gpt", "model"))

            new_user, new_asst, missing = substitute_content(
                convs[user_idx]["content"], convs[asst_idx]["content"], mapper
            )
            convs[user_idx]["content"] = new_user
            convs[asst_idx]["content"] = new_asst
            total_missing |= missing

            fout.write(json.dumps(d, ensure_ascii=False) + "\n")
            total += 1

    print(f"  {src.name}: {total} examples written, {len(total_missing)} unique titles missing from mapper")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", default="data/msmarco-icl-100shot-v4-id_only")
    ap.add_argument("--dst_dir", default="data/msmarco-icl-100shot-v4-sid_only")
    ap.add_argument("--mapper", default="data/msmarco_title_to_semid.json")
    args = ap.parse_args()

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading mapper from {args.mapper} ...")
    mapper = load_mapper(args.mapper)
    print(f"  {len(mapper)} entries")

    for split_file in sorted(src_dir.glob("*.jsonl")):
        print(f"Converting {split_file.name} ...")
        convert_file(split_file, dst_dir / split_file.name, mapper)

    print("Done. Output →", dst_dir)


if __name__ == "__main__":
    main()
