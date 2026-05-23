"""BM25 and DPR retrieval baselines for item-id JSONL datasets."""

from __future__ import annotations

import argparse
import heapq
import json
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    title: str


@dataclass(frozen=True)
class Query:
    text: str
    doc_id: str
    operation: str


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_corpus(
    path: Path,
    operations: set[str],
    include_doc_id_in_text: bool,
) -> list[Document]:
    docs: dict[str, Document] = {}
    for row in read_jsonl(path):
        operation = str(row.get("operation", ""))
        if operations and operation not in operations:
            continue

        doc_id = str(row["doc_id"]).strip()
        text = str(row["text"]).strip()
        # if include_doc_id_in_text:
        #     text = f"{doc_id} {text}"

        if doc_id not in docs:
            docs[doc_id] = Document(doc_id=doc_id, text=text, title="")

    if not docs:
        raise ValueError(f"No corpus documents loaded from {path}")
    return list(docs.values())


def merge_corpus_docs(
    base_docs: list[Document],
    extra_docs: list[Document],
) -> list[Document]:
    docs = {doc.doc_id: doc for doc in base_docs}
    for doc in extra_docs:
        if doc.doc_id not in docs:
            docs[doc.doc_id] = doc
    return list(docs.values())


def load_queries(
    path: Path, operations: set[str] | None, max_samples: int | None
) -> list[Query]:
    queries: list[Query] = []
    for row in read_jsonl(path):
        operation = str(row.get("operation", ""))
        if operations is not None and operation not in operations:
            continue

        queries.append(
            Query(
                text=str(row["text"]).strip(),
                doc_id=str(row["doc_id"]).strip(),
                operation=operation,
            )
        )
        if max_samples is not None and len(queries) >= max_samples:
            break

    if not queries:
        raise ValueError(f"No queries loaded from {path}")
    return queries


class BM25Index:
    def __init__(
        self,
        docs: list[Document],
        k1: float = 1.5,
        b: float = 0.75,
        min_idf: float = 0.0,
    ) -> None:
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.min_idf = min_idf
        self.doc_len: list[int] = []
        term_frequencies: dict[str, list[tuple[int, int]]] = defaultdict(list)

        for doc_idx, doc in enumerate(docs):
            counts = Counter(tokenize(doc.text))
            length = sum(counts.values())
            self.doc_len.append(length)
            for term, tf in counts.items():
                term_frequencies[term].append((doc_idx, tf))

        self.avgdl = sum(self.doc_len) / len(self.doc_len)
        self.idf = {
            term: math.log(
                1.0 + (len(docs) - len(postings) + 0.5) / (len(postings) + 0.5)
            )
            for term, postings in term_frequencies.items()
        }
        self.inverted: dict[str, list[tuple[int, float]]] = {}
        for term, postings in term_frequencies.items():
            idf = self.idf[term]
            if idf < self.min_idf:
                continue
            weighted_postings = []
            for doc_idx, tf in postings:
                denom = tf + self.k1 * (
                    1.0 - self.b + self.b * self.doc_len[doc_idx] / self.avgdl
                )
                weighted_postings.append((doc_idx, idf * tf * (self.k1 + 1.0) / denom))
            self.inverted[term] = weighted_postings

    def search(self, query: str, top_k: int) -> list[str]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            postings = self.inverted.get(term)
            if not postings:
                continue
            idf = self.idf[term]
            if idf < self.min_idf:
                continue
            for doc_idx, score in postings:
                scores[doc_idx] += score

        ranked = heapq.nlargest(top_k, scores.items(), key=lambda item: item[1])
        return [self.docs[doc_idx].doc_id for doc_idx, _ in ranked]


def calculate_metrics(
    predictions: list[list[str]],
    queries: list[Query],
    cutoffs: list[int],
    corpus_doc_ids: set[str],
) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {
        "total": len(queries),
        "missing_gold": sum(
            1 for query in queries if query.doc_id not in corpus_doc_ids
        ),
    }

    for cutoff in cutoffs:
        hits = []
        reciprocal_ranks = []
        ndcg = []
        for predicted_doc_ids, query in zip(predictions, queries):
            try:
                rank = predicted_doc_ids.index(query.doc_id) + 1
            except ValueError:
                rank = math.inf

            hits.append(1.0 if rank <= cutoff else 0.0)
            reciprocal_ranks.append(1.0 / rank if rank <= cutoff else 0.0)
            ndcg.append(1.0 / math.log2(rank + 1.0) if rank <= cutoff else 0.0)

        hit_score = sum(hits) / len(hits)
        metrics[f"hit@{cutoff}"] = hit_score
        metrics[f"hits@{cutoff}"] = hit_score
        metrics[f"recall@{cutoff}"] = hit_score
        metrics[f"mrr@{cutoff}"] = sum(reciprocal_ranks) / len(reciprocal_ranks)
        metrics[f"ndcg@{cutoff}"] = sum(ndcg) / len(ndcg)

    return metrics


def write_predictions(
    path: Path, queries: list[Query], predictions: list[list[str]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for query, predicted_doc_ids in zip(queries, predictions):
            f.write(
                json.dumps(
                    {
                        "text": query.text,
                        "true_docid": query.doc_id,
                        "operation": query.operation,
                        "predicted_docids": predicted_doc_ids,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def run_bm25(
    args: argparse.Namespace, docs: list[Document], queries: list[Query]
) -> list[list[str]]:
    try:
        import bm25s
    except ImportError as exc:
        raise SystemExit(
            "BM25 baseline requires bm25s. Run `uv sync` or `uv add bm25s`, "
            "then rerun with --method bm25."
        ) from exc

    corpus_texts = [doc.text for doc in docs]
    print(len(corpus_texts))

    doc_ids = [doc.doc_id for doc in docs]
    corpus_tokens = bm25s.tokenize(
        corpus_texts,
        stopwords=args.bm25_stopwords,
        show_progress=args.bm25_show_progress,
    )
    retriever = bm25s.BM25(k1=args.bm25_k1, b=args.bm25_b, corpus=doc_ids)
    retriever.index(corpus_tokens, show_progress=args.bm25_show_progress)

    predictions = []
    for start in range(0, len(queries), args.query_batch_size):
        batch = queries[start : start + args.query_batch_size]
        query_tokens = bm25s.tokenize(
            [query.text for query in batch],
            stopwords=args.bm25_stopwords,
            show_progress=False,
        )
        results = retriever.retrieve(
            query_tokens,
            k=args.top_k,
            return_as="documents",
            show_progress=False,
        )
        predictions.extend(
            [[str(doc_id) for doc_id in row] for row in results.tolist()]
        )
        if args.progress_every and len(predictions) % args.progress_every == 0:
            print(
                f"BM25 searched {len(predictions)}/{len(queries)} queries",
                file=sys.stderr,
            )
        for index in range(10):
            print(queries[index].doc_id in predictions[index][:10])
            print(predictions[index][:10], queries[index].doc_id, queries[index].text)
        print("hi")
        breakpoint()
    return predictions


def run_bm25_py(
    args: argparse.Namespace, docs: list[Document], queries: list[Query]
) -> list[list[str]]:
    index = BM25Index(docs, k1=args.bm25_k1, b=args.bm25_b, min_idf=args.bm25_min_idf)
    predictions = []
    for idx, query in enumerate(queries, start=1):
        predictions.append(index.search(query.text, args.top_k))
        if args.progress_every and idx % args.progress_every == 0:
            print(f"BM25-py searched {idx}/{len(queries)} queries", file=sys.stderr)
    return predictions


def run_bm25_rank(
    args: argparse.Namespace, docs: list[Document], queries: list[Query]
) -> list[list[str]]:
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        raise SystemExit(
            "BM25 rank baseline requires rank_bm25. Run `uv add rank_bm25`, "
            "then rerun with --method bm25_rank."
        ) from exc

    print(f"Tokenizing {len(docs)} docs...", file=sys.stderr)
    tokenized_docs = [tokenize(doc.text) for doc in docs]
    doc_ids = [doc.doc_id for doc in docs]
    print(
        f"Building BM25Okapi index (k1={args.bm25_k1}, b={args.bm25_b})...",
        file=sys.stderr,
    )
    bm25 = BM25Okapi(tokenized_docs, k1=args.bm25_k1, b=args.bm25_b)

    predictions: list[list[str]] = []
    for idx, query in enumerate(queries, start=1):
        tokenized_q = tokenize(query.text)
        scores = bm25.get_scores(tokenized_q)
        top_indices = heapq.nlargest(
            args.top_k, range(len(scores)), key=lambda i: scores[i]
        )
        predictions.append([doc_ids[i] for i in top_indices])
        if args.progress_every and idx % args.progress_every == 0:
            print(
                f"BM25-rank searched {idx}/{len(queries)} queries",
                file=sys.stderr,
            )
    return predictions


def run_dpr(
    args: argparse.Namespace, docs: list[Document], queries: list[Query]
) -> list[list[str]]:
    try:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader
        from transformers import (
            DPRContextEncoder,
            DPRContextEncoderTokenizerFast,
            DPRQuestionEncoder,
            DPRQuestionEncoderTokenizerFast,
        )
    except ImportError as exc:
        raise SystemExit(
            "DPR baseline requires torch and transformers. Install them in this env, "
            "then rerun with --method dpr."
        ) from exc

    device = torch.device(args.device)
    ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(
        args.dpr_context_model
    )
    ctx_encoder = DPRContextEncoder.from_pretrained(args.dpr_context_model).to(device)
    q_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained(
        args.dpr_question_model
    )
    q_encoder = DPRQuestionEncoder.from_pretrained(args.dpr_question_model).to(device)
    ctx_encoder.eval()
    q_encoder.eval()

    cache_path = Path(args.dpr_cache) if args.dpr_cache else None
    if cache_path and cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu")
        doc_ids = cached["doc_ids"]
        ctx_embeddings = cached["embeddings"]
    else:
        doc_ids = [doc.doc_id for doc in docs]
        ctx_embeddings_parts = []
        loader = DataLoader(
            docs, batch_size=args.batch_size, shuffle=False, collate_fn=list
        )
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader, start=1):
                encoded = ctx_tokenizer(
                    [doc.title for doc in batch],
                    [doc.text for doc in batch],
                    truncation=True,
                    padding=True,
                    max_length=args.max_length,
                    return_tensors="pt",
                ).to(device)
                embeddings = ctx_encoder(**encoded).pooler_output
                if args.normalize_embeddings:
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                ctx_embeddings_parts.append(embeddings.cpu())
                if (
                    args.progress_every
                    and batch_idx * args.batch_size % args.progress_every == 0
                ):
                    print(
                        f"DPR encoded {min(batch_idx * args.batch_size, len(docs))}/{len(docs)} docs",
                        file=sys.stderr,
                    )
        ctx_embeddings = torch.cat(ctx_embeddings_parts, dim=0)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"doc_ids": doc_ids, "embeddings": ctx_embeddings}, cache_path)

    predictions: list[list[str]] = []
    ctx_embeddings = ctx_embeddings.to(device)
    loader = DataLoader(
        queries, batch_size=args.query_batch_size, shuffle=False, collate_fn=list
    )
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            encoded = q_tokenizer(
                [query.text for query in batch],
                truncation=True,
                padding=True,
                max_length=args.max_length,
                return_tensors="pt",
            ).to(device)
            query_embeddings = q_encoder(**encoded).pooler_output
            if args.normalize_embeddings:
                query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
            scores = query_embeddings @ ctx_embeddings.T
            top_indices = torch.topk(scores, k=args.top_k, dim=1).indices.cpu().tolist()
            predictions.extend([[doc_ids[idx] for idx in row] for row in top_indices])
            if (
                args.progress_every
                and batch_idx * args.query_batch_size % args.progress_every == 0
            ):
                print(
                    f"DPR searched {min(batch_idx * args.query_batch_size, len(queries))}/{len(queries)} queries",
                    file=sys.stderr,
                )

    return predictions


def write_pyserini_corpus(path: Path, docs: list[Document]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(
                json.dumps(
                    {
                        "id": doc.doc_id,
                        "contents": doc.text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def build_pyserini_dense_index(args: argparse.Namespace, docs: list[Document]) -> None:
    corpus_file = Path(args.pyserini_corpus_file)
    index_dir = Path(args.pyserini_index_dir)
    if (
        index_dir.exists()
        and any(index_dir.iterdir())
        and not args.pyserini_rebuild_index
    ):
        return

    write_pyserini_corpus(corpus_file, docs)
    index_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "pyserini.encode",
        "input",
        "--corpus",
        str(corpus_file),
        "--fields",
        "text",
        "output",
        "--embeddings",
        str(index_dir),
        "--to-faiss",
        "encoder",
        "--encoder",
        args.pyserini_encoder,
        "--fields",
        "text",
        "--batch",
        str(args.batch_size),
    ]
    if args.pyserini_fp16:
        cmd.append("--fp16")
    subprocess.run(cmd, check=True)


def make_pyserini_query_encoder(args: argparse.Namespace):
    from pyserini.search.faiss import (
        AnceQueryEncoder,
        DprQueryEncoder,
        FaissSearcher,
        TctColBertQueryEncoder,
    )

    encoder_name = args.pyserini_query_encoder or args.pyserini_encoder
    lower_name = encoder_name.lower()
    if "ance" in lower_name:
        return FaissSearcher, AnceQueryEncoder(encoder_name, device=args.device)
    if "tct_colbert" in lower_name or "tct-colbert" in lower_name:
        return FaissSearcher, TctColBertQueryEncoder(encoder_name, device=args.device)
    if "dpr" in lower_name:
        return FaissSearcher, DprQueryEncoder(encoder_name, device=args.device)
    return FaissSearcher, encoder_name


def run_dpr_pyserini(
    args: argparse.Namespace, docs: list[Document], queries: list[Query]
) -> list[list[str]]:
    try:
        from pyserini.search.faiss import FaissSearcher
    except ImportError as exc:
        raise SystemExit(
            "Pyserini DPR baseline requires pyserini and faiss-cpu. "
            "Run `uv sync`, then rerun with --method dpr_pyserini."
        ) from exc

    build_pyserini_dense_index(args, docs)
    searcher_cls, query_encoder = make_pyserini_query_encoder(args)
    if searcher_cls is not FaissSearcher:
        raise RuntimeError("Unexpected Pyserini searcher class.")
    searcher = FaissSearcher(str(args.pyserini_index_dir), query_encoder)

    predictions: list[list[str]] = []
    for idx, query in enumerate(queries, start=1):
        hits = searcher.search(query.text, k=args.top_k)
        predictions.append([hit.docid for hit in hits])
        if args.progress_every and idx % args.progress_every == 0:
            print(
                f"Pyserini DPR searched {idx}/{len(queries)} queries",
                file=sys.stderr,
            )
    return predictions


def parse_operations(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=["bm25", "bm25_py", "bm25_rank", "dpr", "dpr_pyserini"],
        required=True,
    )
    parser.add_argument(
        "--train_file", type=Path, default=Path("data/msmarco-item-id/train.jsonl")
    )
    parser.add_argument(
        "--test_file", type=Path, default=Path("data/msmarco-item-id/test.jsonl")
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=Path("retrieve_results/baseline_predictions.jsonl"),
    )
    parser.add_argument("--metrics_file", type=Path, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[1, 5, 10, 20, 100])
    parser.add_argument("--progress_every", type=int, default=1000)
    parser.add_argument(
        "--corpus_operations",
        type=parse_operations,
        default=parse_operations("indexing"),
    )
    parser.add_argument("--query_operations", type=parse_operations, default=None)
    parser.add_argument(
        "--include_doc_id_in_text", action=argparse.BooleanOptionalAction, default=True
    )

    parser.add_argument("--bm25_k1", type=float, default=1.5)
    parser.add_argument("--bm25_b", type=float, default=0.75)
    parser.add_argument("--bm25_min_idf", type=float, default=0.1)
    parser.add_argument("--bm25_stopwords", default=None)
    parser.add_argument("--bm25_show_progress", action="store_true")

    parser.add_argument(
        "--dpr_question_model", default="facebook/dpr-question_encoder-multiset-base"
    )
    parser.add_argument(
        "--dpr_context_model", default="facebook/dpr-ctx_encoder-multiset-base"
    )
    parser.add_argument(
        "--dpr_cache", default="retrieve_results/dpr_corpus_embeddings.pt"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--query_batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--normalize_embeddings", action="store_true")

    parser.add_argument(
        "--pyserini_encoder",
        default="castorini/ance-msmarco-passage",
        help="Document encoder for Pyserini dense indexing.",
    )
    parser.add_argument(
        "--pyserini_query_encoder",
        default=None,
        help="Optional separate query encoder for Pyserini dense search.",
    )
    parser.add_argument(
        "--pyserini_corpus_file",
        type=Path,
        default=Path("retrieve_results/pyserini_dense_corpus.jsonl"),
    )
    parser.add_argument(
        "--pyserini_index_dir",
        type=Path,
        default=Path("retrieve_results/pyserini_dense_index"),
    )
    parser.add_argument("--pyserini_rebuild_index", action="store_true")
    parser.add_argument("--pyserini_fp16", action="store_true")
    return parser.parse_args()


def resolve_query_operations(args: argparse.Namespace) -> set[str]:
    if args.query_operations is not None:
        return args.query_operations
    return {"query"}


def should_add_test_indexing_to_corpus(test_file: Path) -> bool:
    return "icl_test" in test_file.name


def main() -> int:
    args = parse_args()
    query_operations = resolve_query_operations(args)
    docs = load_corpus(
        args.train_file,
        operations=args.corpus_operations,
        include_doc_id_in_text=args.include_doc_id_in_text,
    )
    added_test_indexing = False
    if should_add_test_indexing_to_corpus(args.test_file):
        test_indexing_docs = load_corpus(
            args.test_file,
            operations={"indexing"},
            include_doc_id_in_text=args.include_doc_id_in_text,
        )
        docs = merge_corpus_docs(docs, test_indexing_docs)
        added_test_indexing = True
    queries = load_queries(
        args.test_file, operations=query_operations, max_samples=args.max_samples
    )
    if args.top_k < max(args.cutoffs):
        raise ValueError("--top_k must be >= max(--cutoffs)")

    if args.method == "bm25":
        predictions = run_bm25(args, docs, queries)
    elif args.method == "bm25_py":
        predictions = run_bm25_py(args, docs, queries)
    elif args.method == "bm25_rank":
        predictions = run_bm25_rank(args, docs, queries)
    elif args.method == "dpr_pyserini":
        predictions = run_dpr_pyserini(args, docs, queries)
    else:
        predictions = run_dpr(args, docs, queries)

    write_predictions(args.output_file, queries, predictions)
    metrics = calculate_metrics(
        predictions,
        queries,
        cutoffs=args.cutoffs,
        corpus_doc_ids={doc.doc_id for doc in docs},
    )
    metrics["method"] = args.method
    metrics["train_file"] = str(args.train_file)
    metrics["test_file"] = str(args.test_file)
    metrics["corpus_size"] = len(docs)
    metrics["query_operations"] = sorted(query_operations)
    metrics["added_test_indexing_to_corpus"] = added_test_indexing

    if args.metrics_file:
        args.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_file.write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
