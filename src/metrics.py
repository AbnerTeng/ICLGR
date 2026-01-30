from typing import Dict, List, Union

import numpy as np


class GRMetrics:
    def __init__(
        self, beam_searched_outputs: List[List[str]], goldens: List[str]
    ) -> None:
        self.beam_searched_outputs = beam_searched_outputs
        self.goldens = goldens
        self.metric_type = ["hits", "mrr", "ndcg"]

    def _calculate_hits_at_k(self, output: List[str], golden: str, k: int) -> float:
        try:
            rank = output.index(golden) + 1
        except ValueError:
            rank = np.inf

        return 1.0 if rank <= k else 0.0

    def _calculate_mrr_at_k(self, output: List[str], golden: str, k: int) -> float:
        try:
            rank = output.index(golden) + 1
        except ValueError:
            rank = np.inf

        return 1.0 / rank if rank <= k else 0.0

    def _calculate_ndcg_at_k(self, output: List[str], golden: str, k: int) -> float:
        try:
            rank = output.index(golden) + 1
        except ValueError:
            rank = np.inf

        return 1.0 / np.log2(rank + 1) if rank <= k else 0.0

    def calculate_metrics(self, k: List[int]) -> Dict[str, float]:
        if max(k) > len(self.beam_searched_outputs):
            raise ValueError(
                f"Max k value {max(k)} exceeds number of beam searched outputs {len(self.beam_searched_outputs)}"
            )

        metrics_dict = {}

        for model_output, golden in zip(self.beam_searched_outputs, self.goldens):
            for metric in self.metric_type:
                for cutoff in k:
                    if metric == "hits":
                        if f"hits@{cutoff}" not in metrics_dict:
                            metrics_dict[f"hits@{cutoff}"] = []

                        metrics_dict[f"hits@{cutoff}"].append(
                            self._calculate_hits_at_k(model_output, golden, cutoff)
                        )
                    elif metric == "mrr":
                        if f"mrr@{cutoff}" not in metrics_dict:
                            metrics_dict[f"mrr@{cutoff}"] = []

                        metrics_dict[f"mrr@{cutoff}"].append(
                            self._calculate_mrr_at_k(model_output, golden, cutoff)
                        )
                    elif metric == "ndcg":
                        if f"ndcg@{cutoff}" not in metrics_dict:
                            metrics_dict[f"ndcg@{cutoff}"] = []

                        metrics_dict[f"ndcg@{cutoff}"].append(
                            self._calculate_ndcg_at_k(model_output, golden, cutoff)
                        )

        for metric in metrics_dict:
            metrics_dict[metric] = np.mean(metrics_dict[metric])

        return metrics_dict


def dcg_at_k(relevance_scores: List[float], k: int) -> float:
    """
    Calculate Discounted Cumulative Gain at k

    Args:
        relevance_scores: List of relevance scores for ranked documents
        k: Cut-off rank position

    Returns:
        DCG@k value
    """
    relevance_scores = np.array(relevance_scores[:k])
    if len(relevance_scores) == 0:
        return 0.0

    positions = np.arange(1, len(relevance_scores) + 1)
    dcg = np.sum(relevance_scores / np.log2(positions + 1))

    return dcg


def ndcg_at_k(relevance_scores: List[float], k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at k

    Args:
        relevance_scores: List of relevance scores for ranked documents
        k: Cut-off rank position

    Returns:
        NDCG@k value (between 0 and 1)
    """
    dcg_k = dcg_at_k(relevance_scores, k)
    ideal_relevance = sorted(relevance_scores, reverse=True)
    idcg_k = dcg_at_k(ideal_relevance, k)

    if idcg_k == 0:
        return 0.0

    return dcg_k / idcg_k


def mrr(relevant_positions: List[int]) -> float:
    """
    Calculate Mean Reciprocal Rank for a single query

    Args:
        relevant_positions: List of 0-indexed positions where relevant documents appear
        k: Cut-off rank position

    Returns:
        RR@k value (reciprocal rank of first relevant document within top-k)
    """
    valid_positions = [pos + 1 for pos in relevant_positions if pos >= 0]

    if not valid_positions:
        return 0.0

    return 1.0 / min(valid_positions)


def calculate_ndcg_metrics(
    results: List[Dict],
    k_values: List[int] = [1, 5, 10],
) -> Dict[str, float]:
    """
    Calculate NDCG@K metrics for a list of query results

    Args:
        results: List of result dictionaries containing rankings and relevance scores
        k_values: List of k values to compute NDCG for
        relevance_field: Field name containing relevance scores in results

    Returns:
        Dictionary with NDCG@k values for each k
    """
    ndcg_metrics_reranked = {f"ndcg@{k}": 0.0 for k in k_values}
    valid_queries = 0

    for result in results:
        reranked_docs = result.get("reranked_docs", [])

        if not reranked_docs:
            continue

        relevance_scores = [
            pair["score"]
            for pair in result.get("reranking_scores", [])
            if pair["doc_id"] in reranked_docs
        ]
        for k in k_values:
            ndcg_k = ndcg_at_k(relevance_scores, k)
            ndcg_metrics_reranked[f"ndcg@{k}"] += ndcg_k

        valid_queries += 1

    if valid_queries > 0:
        for k in k_values:
            ndcg_metrics_reranked[f"ndcg@{k}"] /= valid_queries

    return ndcg_metrics_reranked


def calculate_mrr_metrics(
    results: List[Dict],
    ground_truth_field: str = "query_id",
) -> Dict[str, float]:
    """
    Calculate MRR@K metrics for a list of query results

    Args:
        results: List of result dictionaries containing rankings
        k_values: List of k values to compute MRR for
        ground_truth_field: Field containing ground truth document ID

    Returns:
        Dictionary with MRR@k values for each k
    """
    mrr_metrics = {"mrr": 0.0}
    valid_queries = 0

    for result in results:
        ground_truth = result.get(ground_truth_field)
        reranked_docs = result.get("reranked_docs", [])

        if not ground_truth or not reranked_docs:
            continue

        relevant_positions = []
        for i, doc_id in enumerate(reranked_docs):
            if doc_id == ground_truth:
                relevant_positions.append(i)
                break  # Only need first occurrence for MRR

        rr = mrr(relevant_positions)
        mrr_metrics["mrr"] += rr

        valid_queries += 1

    if valid_queries > 0:
        mrr_metrics["mrr"] /= valid_queries

    return mrr_metrics


def calculate_comprehensive_metrics(
    results: List[Dict], k_values: List[int] = [1, 5, 10]
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Calculate comprehensive ranking metrics including Hits@K, NDCG@K, and MRR@K

    Args:
        results: List of result dictionaries
        k_values: List of k values to compute metrics for

    Returns:
        Dictionary containing all computed metrics
    """
    hits_dsi, hits_reranked, total_queries = calculate_hits_metrics(results, k_values)
    ndcg_metrics = calculate_ndcg_metrics(results, k_values)
    mrr_metrics = calculate_mrr_metrics(results)

    comprehensive_metrics = {
        "total_queries": total_queries,
        "hits_dsi": hits_dsi,
        "hits_reranked": hits_reranked,
        "ndcg": ndcg_metrics,
        "mrr": mrr_metrics,
    }

    return comprehensive_metrics


def calculate_hits_metrics(results: List[Dict], k_values: List[int] = [1, 10]):
    """
    Calculate Hits@K metrics using the same logic as QueryEvalCallback in trainer.py
    For reranker evaluation, we'll use a proxy for ground truth:
    - If we have explicit relevance judgments, use those
    - Otherwise, assume the first DSI result as the "correct" answer for comparison
    """
    hits_metrics_dsi = {f"hits@{k}": 0 for k in k_values}
    hits_metrics_reranked = {f"hits@{k}": 0 for k in k_values}
    total_queries = len(results)

    for result in results:
        dsi_ranking = result.get("dsi_ranking", [])
        reranked_docs = result.get("reranked_docs", [])

        if not dsi_ranking or not reranked_docs:
            continue

        ground_truth_doc = result.get("query_id", None)
        rank_list = np.array(reranked_docs)  # Top 10 as in trainer
        dsi_output = np.array(dsi_ranking)
        hits_dsi = np.where(dsi_output == ground_truth_doc)[0]
        hits_rerank = np.where(rank_list == ground_truth_doc)[0]

        if len(hits_dsi) != 0:
            for k in k_values:
                if hits_dsi[0] < k:
                    hits_metrics_dsi[f"hits@{k}"] += 1

        if len(hits_rerank) != 0:
            for k in k_values:
                if hits_rerank[0] < k:
                    hits_metrics_reranked[f"hits@{k}"] += 1

    for k in k_values:
        hits_metrics_dsi[f"hits@{k}"] = hits_metrics_dsi[f"hits@{k}"] / total_queries
        hits_metrics_reranked[f"hits@{k}"] = (
            hits_metrics_reranked[f"hits@{k}"] / total_queries
        )

    return hits_metrics_dsi, hits_metrics_reranked, total_queries


def test_metrics():
    """Test function to verify metric calculations"""

    # Test NDCG calculation
    print("Testing NDCG calculation...")
    relevance_scores = [3.0, 2.0, 3.0, 0.0, 1.0, 2.0]  # Example relevance scores
    ndcg_5 = ndcg_at_k(relevance_scores, 5)
    print(f"NDCG@5 for scores {relevance_scores[:5]}: {ndcg_5:.4f}")

    # Test MRR calculation
    print("\nTesting MRR calculation...")
    relevant_positions = [2]  # Relevant document at position 2 (0-indexed)
    mrr = mrr(relevant_positions)
    print(f"MRR for relevant position {relevant_positions}: {mrr:.4f}")

    # Test with mock results
    print("\nTesting with mock results...")
    mock_results = [
        {
            "query_id": "q1",
            "reranked_docs": ["doc1", "doc2", "doc3"],
            "ranking_scores": [
                {"doc_id": "doc1", "score": 0.9},
                {"doc_id": "doc2", "score": 0.7},
                {"doc_id": "doc3", "score": 0.5},
            ],
        }
    ]

    comprehensive = calculate_comprehensive_metrics(mock_results, [1, 5, 10])
    print("Comprehensive metrics:")
    for metric_type, values in comprehensive.items():
        if isinstance(values, dict):
            print(f"  {metric_type}:")
            for k, v in values.items():
                print(f"    {k}: {v:.4f}")
        else:
            print(f"  {metric_type}: {values}")


if __name__ == "__main__":
    test_metrics()
