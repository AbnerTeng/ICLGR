import json
import random
import os
from typing import Dict, List

from tqdm import tqdm
from datasets import load_dataset
import hydra
from omegaconf import DictConfig


def format_item(sample: Dict):
    """Formats a single data point into a (text, doc_id) string."""
    return f"({sample['text']}, {sample['doc_id']})"


def generate_icl_variants(target_query, all_docs, all_queries, n):
    """
    Generates different ICL patterns for a specific target query.
    """
    if n == 0:
        user_content = f"({target_query['text']}, )"
        return [
            {
                "conversations": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": target_query["doc_id"]},
                ],
                "metadata": {
                    "pattern": "zero-shot",
                    "target_id": target_query["doc_id"],
                },
            }
        ]

    pos_doc = next((d for d in all_docs if d["doc_id"] == target_query["doc_id"]), None)
    neg_docs = random.sample(
        [d for d in all_docs if d["doc_id"] != target_query["doc_id"]], n + 1
    )
    # neg_queries = random.sample(
    #     [q for q in all_queries if q["doc_id"] != target_query["doc_id"]], n + 1
    # )

    if not pos_doc:
        return []

    variants = []

    patterns = {
        "Doc-Positive Front": [pos_doc] + neg_docs[: n - 1],
        "Doc-Positive Back": neg_docs[: n - 1] + [pos_doc],
        "All-Noise Docs": neg_docs[:n],
        # "Doc-Positive + Query Front": [pos_doc] + neg_docs[: n - 1] + neg_queries[:n],
        # "Doc-Positive + Query Back": neg_docs[:n] + neg_queries[: n - 1] + [pos_doc],
        # "All-Noise Docs + Query": neg_docs[:n] + neg_queries[:n],
        # "Query Only (Noise)": neg_queries[:n],
    }

    for name, samples in patterns.items():
        context_str = " ".join([format_item(s) for s in samples])
        user_content = f"{context_str} ({target_query['text']}, )"
        variants.append(
            {
                "conversations": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": target_query["doc_id"]},
                ],
                "metadata": {"pattern": name, "target_id": target_query["doc_id"]},
            }
        )

    return variants


def subsample_by_pattern(examples, ratio_map=None, keep_all_patterns=None):
    ratio_map = ratio_map or {}
    keep_all_patterns = keep_all_patterns or set()

    grouped = {}
    for ex in examples:
        pattern = ex["metadata"]["pattern"]
        grouped.setdefault(pattern, []).append(ex)

    final_examples = []
    for pattern, items in grouped.items():
        if pattern in keep_all_patterns:
            final_examples.extend(items)
        elif pattern in ratio_map:
            keep_n = (
                max(1, int(len(items) * ratio_map[pattern])) if len(items) > 0 else 0
            )
            final_examples.extend(random.sample(items, keep_n))
        else:
            final_examples.extend(items)

    random.shuffle(final_examples)
    return final_examples


def process_and_save(train_docs, train_queries, dataset, output_path, n_shot):
    """Splits dataset by operation and applies n-shot conversion."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # docs = [s for s in dataset if s["operation"] == "indexing"]
    queries = [s for s in dataset if s["operation"] == "query"]

    print(f"Processing and saving to: {output_path}")

    nshot_examples = []
    for q in tqdm(queries, desc=f"Generating {n_shot}-shot"):
        variants = generate_icl_variants(q, train_docs, train_queries, n_shot)
        nshot_examples.extend(variants)
    nshot_examples = subsample_by_pattern(
        nshot_examples,
        ratio_map={
            "Doc-Positive Front": 0.2,
            "Doc-Positive Back": 0.2,
        },
        keep_all_patterns={"All-Noise Docs"},
    )
    zero_shot_examples = []
    for q in tqdm(queries, desc="Generating zero-shot"):
        zero_shot_examples.extend(
            generate_icl_variants(q, train_docs, train_queries, 0)
        )

    all_examples = nshot_examples + zero_shot_examples
    random.shuffle(all_examples)

    with open(output_path, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Saved {len(all_examples)} examples.")
    print(f"  - n-shot kept: {len(nshot_examples)}")
    print(f"  - zero-shot added: {len(zero_shot_examples)}")


@hydra.main(config_path="../configs", config_name="get_icl_axolotl", version_base=None)
def main(cfg: DictConfig):
    dataset = load_dataset(
        "json",
        data_files={
            "train": f"{cfg.input_path}/train.jsonl",
            "test": f"{cfg.input_path}/test.jsonl",
            "icl_test": f"{cfg.input_path}/icl_test.jsonl",
        },
    )
    train_docs = [s for s in dataset["train"] if s["operation"] == "indexing"]
    train_queries = [s for s in dataset["train"] if s["operation"] == "query"]

    splits: List[str] = ["train", "test", "icl_test"]
    for split in splits:
        if split not in dataset:
            continue

        docs_pool = train_docs
        queries_pool = train_queries

        if split == "icl_test":
            docs_pool = [s for s in dataset["icl_test"] if s["operation"] == "indexing"]
            queries_pool = [s for s in dataset["icl_test"] if s["operation"] == "query"]

        out_file = f"{cfg.output_dir}/{split}_{cfg.n_shot}shot.jsonl"
        process_and_save(docs_pool, queries_pool, dataset[split], out_file, cfg.n_shot)

    print("\nConversion complete for all 7 ICL patterns.")


if __name__ == "__main__":
    main()
