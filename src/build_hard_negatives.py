"""Mine 3 hard negatives per row using bge-base embeddings + FAISS.

Pool = train.jsonl indexing rows. Gold doc_id excluded. Output writes
hard_negatives = [{"doc_id", "text"}, ...] alongside original fields.
"""

import argparse
import json
import random
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def embed(model, texts, batch_size, device):
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
        device=device,
    ).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", default="data/msmarco-item-id")
    ap.add_argument("--dst_dir", default="data/msmarco-item-id-hardneg")
    ap.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--k", type=int, default=3, help="Number of hard negatives per row")
    ap.add_argument(
        "--pool_depth",
        type=int,
        default=0,
        help="Retrieve this many before sampling; 0 = just take top-k directly",
    )
    ap.add_argument(
        "--skip_top",
        type=int,
        default=0,
        help="Skip this many top hits (likely false negatives) before sampling",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument(
        "--files", nargs="+", default=["train.jsonl", "test.jsonl", "icl_test.jsonl"]
    )
    args = ap.parse_args()
    random.seed(args.seed)

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Loading model {args.model} on {device}")
    model = SentenceTransformer(args.model, device=device)

    # 1. Build pool from train.jsonl indexing rows
    train_rows = load_jsonl(src_dir / "train.jsonl")
    pool_rows = [r for r in train_rows if r.get("operation") == "indexing"]
    pool_texts = [r["text"] for r in pool_rows]
    pool_docids = [r["doc_id"] for r in pool_rows]
    print(f"Pool size (train indexing): {len(pool_rows)}")

    # 2. Embed pool and build FAISS index (inner product on L2-normalized = cosine)
    print("Embedding pool...")
    pool_emb = embed(model, pool_texts, args.batch_size, device)
    dim = pool_emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(pool_emb)

    # 3. Per-file: only embed/search INDEXING rows; query rows pass through unchanged
    sample_mode = args.pool_depth > args.k
    retrieve_k = (
        max(args.pool_depth, args.k) + args.skip_top + 8
    )  # extra buffer for gold filtering
    print(
        f"Mode: {'SAMPLE' if sample_mode else 'TOP-K'}  "
        f"k={args.k} pool_depth={args.pool_depth} skip_top={args.skip_top} retrieve={retrieve_k}"
    )
    for fname in args.files:
        print(f"\n=== {fname} ===")
        rows = train_rows if fname == "train.jsonl" else load_jsonl(src_dir / fname)

        idx_rows = [r for r in rows if r.get("operation") == "indexing"]
        print(f"rows={len(rows)}  indexing anchors={len(idx_rows)}")

        if idx_rows:
            if fname == "train.jsonl":
                # train indexing anchors == pool; reuse embeddings
                anchor_emb = pool_emb
            else:
                print(f"Embedding {len(idx_rows)} indexing anchors...")
                anchor_emb = embed(
                    model, [r["text"] for r in idx_rows], args.batch_size, device
                )

            print(f"Searching top-{retrieve_k}...")
            D, I = index.search(anchor_emb, retrieve_k)

            for local_i, r in enumerate(idx_rows):
                gold = r["doc_id"]
                # Gold-filtered candidate list in rank order
                cands = [int(j) for j in I[local_i] if pool_docids[int(j)] != gold]
                if sample_mode:
                    # Sample k from [skip_top : pool_depth] (post gold-filter)
                    window = cands[args.skip_top : args.skip_top + args.pool_depth]
                    if len(window) < args.k:
                        window = cands[
                            : args.skip_top + args.pool_depth
                        ]  # fallback: expand
                    picks = random.sample(window, min(args.k, len(window)))
                else:
                    picks = cands[: args.k]
                r["hard_negatives"] = [
                    {"doc_id": pool_docids[j], "text": pool_texts[j]} for j in picks
                ]
        else:
            print("No indexing rows in this file — passing through unchanged")

        write_jsonl(dst_dir / fname, rows)
        print(f"Wrote {dst_dir / fname}")


if __name__ == "__main__":
    main()
