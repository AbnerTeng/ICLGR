"""
Generate hierarchical cluster label -> docid training samples.

For each indexing document in train.jsonl, its existing PQ cluster path
(d0, d1, d2) is used to group documents. TF-IDF keywords are extracted per
cluster to produce a human-readable label. The resulting training sample:
    input:  "biology evolution reptile"  (top_k keywords per level, default 1)
    output: "<|d0_143|> <|d1_216|> <|d2_182|>"

Usage:
    python -m src.gen_hierarchical_samples --top_k 1
"""

import argparse
import json
import re
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


TRAIN_PATH = "data/msmarco/train.jsonl"
OUTPUT_SAMPLES_PATH = "data/msmarco/train_hierarchical.jsonl"
OUTPUT_MAPPING_PATH = "data/msmarco/cluster_keywords.json"


def parse_docid(doc_id: str) -> tuple[int, int, int]:
    """'<|d0_136|> <|d1_162|> <|d2_147|>' -> (136, 162, 147)"""
    codes = [int(x) for x in re.findall(r"<\|d\d+_(\d+)\|>", doc_id)]
    return tuple(
        codes[:3]
    )  # take only first 3 levels (some have 4 due to disambiguation)


def top_keywords(
    tfidf_matrix, indices: list[int], feature_names, top_k: int, exclude: set = None
) -> list[str]:
    """Return top_k TF-IDF keywords for a cluster, skipping any word in exclude."""
    mean_scores = np.asarray(tfidf_matrix[indices].mean(axis=0)).flatten()
    sorted_idx = mean_scores.argsort()[::-1]
    exclude = exclude or set()
    result = []
    for i in sorted_idx:
        if mean_scores[i] <= 0:
            break
        word = feature_names[i]
        if word not in exclude:
            result.append(word)
        if len(result) == top_k:
            break
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--top_k", type=int, default=1, help="Keywords per cluster level"
    )
    parser.add_argument(
        "--sample_docs",
        type=int,
        default=5,
        help="Sample docs per cluster in mapping JSON",
    )
    args = parser.parse_args()

    # ── Load indexing documents ──────────────────────────────────────────────
    print("Loading documents...")
    with open(TRAIN_PATH) as f:
        all_data = [json.loads(line) for line in f]

    docs = [d for d in all_data if d["operation"] == "indexing"]
    doc_texts = [d["text"] for d in docs]
    doc_ids = [d["doc_id"] for d in docs]
    print(f"  {len(docs)} indexing documents")

    # ── Build TF-IDF over all doc texts ─────────────────────────────────────
    print("Building TF-IDF...")
    tfidf = TfidfVectorizer(
        max_features=20000, stop_words="english", min_df=2, max_df=0.85
    )
    tfidf_matrix = tfidf.fit_transform(doc_texts)
    feature_names = tfidf.get_feature_names_out()

    # ── Group documents by existing cluster codes ────────────────────────────
    d0_groups: dict = defaultdict(list)
    d01_groups: dict = defaultdict(list)
    d012_groups: dict = defaultdict(list)

    for i, doc_id in enumerate(doc_ids):
        d0, d1, d2 = parse_docid(doc_id)
        d0_groups[d0].append(i)
        d01_groups[(d0, d1)].append(i)
        d012_groups[(d0, d1, d2)].append(i)

    # ── Extract keywords per cluster (each level excludes words from parent) ──
    print("Extracting cluster keywords...")
    d0_kw = {
        k: top_keywords(tfidf_matrix, v, feature_names, args.top_k)
        for k, v in d0_groups.items()
    }

    d01_kw = {}
    for (d0, d1), indices in d01_groups.items():
        exclude = set(d0_kw.get(d0, []))
        d01_kw[(d0, d1)] = top_keywords(
            tfidf_matrix, indices, feature_names, args.top_k, exclude
        )

    d012_kw = {}
    for (d0, d1, d2), indices in d012_groups.items():
        exclude = set(d0_kw.get(d0, [])) | set(d01_kw.get((d0, d1), []))
        d012_kw[(d0, d1, d2)] = top_keywords(
            tfidf_matrix, indices, feature_names, args.top_k, exclude
        )

    # ── Save cluster → keyword + sample doc mapping ──────────────────────────
    print("Saving cluster keyword mapping...")
    mapping = {"d0": {}, "d1": {}, "d2": {}}

    for d0, indices in d0_groups.items():
        mapping["d0"][str(d0)] = {
            "keywords": d0_kw[d0],
            "doc_count": len(indices),
            "sample_docs": [doc_texts[i][:200] for i in indices[: args.sample_docs]],
        }

    for (d0, d1), indices in d01_groups.items():
        mapping["d1"][f"{d0}_{d1}"] = {
            "keywords": d01_kw[(d0, d1)],
            "doc_count": len(indices),
            "sample_docs": [doc_texts[i][:200] for i in indices[: args.sample_docs]],
        }

    for (d0, d1, d2), indices in d012_groups.items():
        mapping["d2"][f"{d0}_{d1}_{d2}"] = {
            "keywords": d012_kw[(d0, d1, d2)],
            "doc_count": len(indices),
            "sample_docs": [doc_texts[i][:200] for i in indices[: args.sample_docs]],
        }

    with open(OUTPUT_MAPPING_PATH, "w") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    print(f"Saved cluster mapping to {OUTPUT_MAPPING_PATH}")

    # ── Create hierarchical training samples ─────────────────────────────────
    print("Creating training samples...")
    samples = []
    skipped = 0

    for i, doc in enumerate(docs):
        d0, d1, d2 = parse_docid(doc_ids[i])
        kw0 = " ".join(d0_kw.get(d0, []))
        kw1 = " ".join(d01_kw.get((d0, d1), []))
        kw2 = " ".join(d012_kw.get((d0, d1, d2), []))
        label = " ".join(filter(None, [kw0, kw1, kw2]))

        if not label:
            skipped += 1
            continue

        samples.append(
            {
                "text": label,
                "doc_id": doc["doc_id"],
                "operation": "hierarchical_indexing",
                "source": "msmarco",
            }
        )

    print(f"  Created {len(samples)} samples, skipped {skipped} (no keywords)")

    with open(OUTPUT_SAMPLES_PATH, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Saved training samples to {OUTPUT_SAMPLES_PATH}")


if __name__ == "__main__":
    main()
