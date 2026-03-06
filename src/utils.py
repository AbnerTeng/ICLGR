import json
import os
import random

import numpy as np
import torch


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_test_split(
    docs: list,
    queries: list,
    test_size: float = 0.2,
    icl_test_size: float = 0.1,
    folder_name: str = "NQ_100k",
) -> None:
    assert len(docs) == len(queries), "Docs and queries must have the same length"
    data = list(zip(docs, queries))
    random.shuffle(data)
    seed_all(42)

    queries_with_docid = []

    for doc, query in zip(docs, queries):
        if doc["text_id"] == query["text_id"]:
            query_copy = query.copy()
            query_copy["doc_id"] = doc["doc_id"]
            queries_with_docid.append(query_copy)
        else:
            raise ValueError(
                f"Doc ID {doc['doc_id']} does not match Query ID {query['text_id']}"
            )

    train_docs_id = int(len(data) * (1 - icl_test_size))
    train_queries_id = int(len(data) * (1 - test_size - icl_test_size))
    test_queries_id = train_queries_id + int(len(data) * test_size)

    train = docs[:train_docs_id] + queries_with_docid[:train_queries_id]
    test = queries_with_docid[train_queries_id:test_queries_id]
    icl_test = docs[train_docs_id:] + queries_with_docid[test_queries_id:]

    if not os.path.exists(f"./data/{folder_name}"):
        os.makedirs(f"./data/{folder_name}")

    with open(f"./data/{folder_name}/train.jsonl", "w") as f:
        for item in train:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(f"./data/{folder_name}/test.jsonl", "w") as f:
        for item in test:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(f"./data/{folder_name}/icl_test.jsonl", "w") as f:
        for item in icl_test:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    with open("./data/NQ_gen_semid/train_semantic_docids.jsonl", "r") as f:
        train_docid = [json.loads(line) for line in f]

    with open("./data/NQ_gen_semid/NQ_100k_multi_task_queries.jsonl", "r") as f:
        train_query = [json.loads(line) for line in f]

    train_test_split(
        docs=train_docid,
        queries=train_query,
        test_size=0.2,
        icl_test_size=0.1,
        folder_name="NQ",
    )
