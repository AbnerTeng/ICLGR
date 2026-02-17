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
