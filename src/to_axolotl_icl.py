import json
import random
import os
from argparse import ArgumentParser, Namespace
from datasets import load_from_disk
from tqdm import tqdm
from datasets import load_dataset

def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default="/semantic-search-pvc/users/lala/ICLGR_project/data/msmarco-ICL-100k/data",
    )
    parser.add_argument("--dataset_name", type=str, default="msmarco")
    parser.add_argument("--n_shot", type=int, default=0, help="Number of examples in context")
    return parser.parse_args()

def format_item(sample):
    """Formats a single data point into a (text, doc_id) string."""
    return f"({sample['text']}, {sample['doc_id']})"

def generate_icl_variants(target_query, all_docs, all_queries, n):
    """
    Generates 7 different ICL patterns for a specific target query.
    """
    # Zero-shot
    if n == 0:
        user_content = f"({target_query['text']}, )"
        return [{
            "conversations": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": target_query["doc_id"]}
            ],
            "metadata": {"pattern": "zero-shot", "target_id": target_query["doc_id"]}
        }]
        
    # Find the matching document for the current query (Positive Sample)
    pos_doc = next((d for d in all_docs if d['doc_id'] == target_query['doc_id']), None)
    
    # Randomly pick negative documents (ID does not match target)
    neg_docs = random.sample([d for d in all_docs if d['doc_id'] != target_query['doc_id']], n + 1)
    
    # Randomly pick negative queries (ID does not match target)
    neg_queries = random.sample([q for q in all_queries if q['doc_id'] != target_query['doc_id']], n + 1)

    if not pos_doc:
        return []

    variants = []
    
    # Define the 7 patterns based on your requirements
    patterns = {
        "Doc-Positive Front": [pos_doc] + neg_docs[:n-1],
        "Doc-Positive Back": neg_docs[:n-1] + [pos_doc],
        "All-Noise Docs": neg_docs[:n],
        # "Doc-Positive + Query Front": [pos_doc] + neg_docs[:n-1] + neg_queries[:n],
        # "Doc-Positive + Query Back": neg_docs[:n] + neg_queries[:n-1] + [pos_doc],
        # "All-Noise Docs + Query": neg_docs[:n] + neg_queries[:n],
        # "Query Only (Noise)": neg_queries[:n]
    }

    for name, samples in patterns.items():
        # Concatenate examples into a single context string
        context_str = " ".join([format_item(s) for s in samples])
        
        # Format for Axolotl/ShareGPT: User provides context + query, Assistant provides DocID
        user_content = f"{context_str} ({target_query['text']}, )"
        
        variants.append({
            "conversations": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": target_query["doc_id"]}
            ],
            "metadata": {"pattern": name, "target_id": target_query["doc_id"]}
        })
    
    return variants

def process_and_save(train_docs, train_queries, dataset, output_path, n_shot):
    """Splits dataset by operation and applies n-shot conversion."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Separate samples by their operation type
    # docs = [s for s in dataset if s["operation"] == "indexing"]
    queries = [s for s in dataset if s["operation"] == "query"]
    
    print(f"Processing and saving to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        # Iterate through queries to create ICL training samples
        for q in tqdm(queries):
            variants = generate_icl_variants(q, train_docs, train_queries, n_shot)
            for v in variants:
                f.write(json.dumps(v, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    args = get_args()
    
    dataset = load_dataset(
        "json",
        data_files={
            "train": f"{args.input_path}/train.jsonl",
            "test": f"{args.input_path}/test.jsonl",
            "icl_test": f"{args.input_path}/icl_test.jsonl",
        },
    )
    train_docs = [s for s in dataset["train"] if s["operation"] == "indexing"]
    train_queries = [s for s in dataset["train"] if s["operation"] == "query"]
    
    splits = ["train", "test", "icl_test"] 
    for split in splits:
        if split not in dataset:
            continue

        # default
        docs_pool = train_docs
        queries_pool = train_queries

        # special rule for icl_teset
        if split == "icl_test":
            docs_pool = [s for s in dataset["icl_test"] if s["operation"] == "indexing"]
            queries_pool = [s for s in dataset["icl_test"] if s["operation"] == "query"]

        out_file = f"/semantic-search-pvc/users/lala/ICLGR_project/data/{args.dataset_name}_axolotl/{split}_{args.n_shot}shot.jsonl"
        process_and_save(
            docs_pool,
            queries_pool,
            dataset[split],
            out_file,
            args.n_shot
        )

    print("\n✅ Conversion complete for all 7 ICL patterns.")