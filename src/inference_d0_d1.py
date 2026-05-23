import os
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
from omegaconf import DictConfig
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    LogitsProcessorList,
)

from .inference_utils import (
    TrieNode,
    build_semantic_docid_trie,
    TrieConstrainedLogitsProcessor,
)
from .metrics import GRMetrics

os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("stage-1_decoder_inference_title.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class DecoderInference:
    """
    Inference class for decoder-only models trained on docid generation task.

    Supports semantic docid tokens:
    - Semantic: "<|d0_253|> <|d1_56|> <|d2_174|>"

    For semantic docids, the model must be trained with semantic tokens
    added as special tokens (see config/axolotl_semantic_docids.yml).
    """

    def __init__(
        self,
        model_path: str,
        from_hf: bool,
        train_data_path: str,
        new_data_path: str = "",
        base_model_path: Optional[str] = None,
        subsample_size: int = 0,
        seed: int = 42,
        clusters_file: str = "",
        query_assignments_file: str = "",
        query_prefix: str = "question: ",
    ) -> None:
        if from_hf:
            self.model_path = model_path
        else:
            self.model_path = Path(model_path)

        self.base_model_path = base_model_path
        self.train_data_path = train_data_path
        self.new_data_path = new_data_path
        self.subsample_size = subsample_size
        self.query_prefix = query_prefix
        self.rng = random.Random(seed)
        self.device = self._setup_device()

        logger.info("Initializing Decoder Inference...")
        logger.info(f"Loading model from: {self.model_path}")

        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.trie_root = self._build_trie()
        self.seen_trie = self._build_seen_trie()
        self.generation_config = self._setup_generation_config()

        if subsample_size > 0:
            self.unseen_docids = self._load_unseen_docids()
            logger.info(
                f"Random subsampling enabled: {subsample_size} candidates per query "
                f"from {len(self.unseen_docids)} unseen docids"
            )
        else:
            self.unseen_docids: List[str] = []

        if clusters_file and query_assignments_file:
            self.clusters, self.query_to_cluster = self._load_cluster_data(
                clusters_file, query_assignments_file
            )
            logger.info(
                f"Cluster-based candidate selection enabled: "
                f"{len(self.clusters)} clusters loaded"
            )
        else:
            self.clusters: Dict[str, List[str]] = {}
            self.query_to_cluster: Dict[str, int] = {}

        logger.info("Model loaded successfully!")

    def _setup_device(self) -> torch.device:
        """Setup the computation device."""
        return torch.device("cuda")

    def _load_tokenizer(self) -> AutoTokenizer:
        """Load the tokenizer with special semantic tokens if present."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, padding_side="left", trust_remote_code=True
            )
            logger.info(f"Loaded tokenizer from {self.model_path}")

            if any(token.startswith("<|d") for token in tokenizer.get_vocab().keys()):
                logger.info("Detected semantic docid tokens in vocabulary")
                logger.info(f"Vocabulary size: {len(tokenizer)}")

        except Exception as e:
            logger.warning(f"Loading tokenizer from base Qwen model due to: {e}")
            tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen3-1.7B", padding_side="left", trust_remote_code=True
            )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def _load_model(self) -> AutoModelForCausalLM:
        """Load the trained model."""
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=(
                torch.float16 if self.device.type == "cuda" else torch.float32
            ),
            device_map="auto" if self.device.type == "cuda" else None,
        )
        model.eval()

        return model

    def _build_trie(self) -> TrieNode:
        """Build the trie from training data."""
        logger.info("Building full docid trie (seen + unseen)...")
        files = [self.train_data_path]
        if self.new_data_path:
            files.append(self.new_data_path)
        trie_root = build_semantic_docid_trie(files, self.tokenizer)
        return trie_root

    def _build_seen_trie(self) -> TrieNode:
        """Build a trie from seen (training) docs only, used alongside cluster tries."""
        logger.info("Building seen-only docid trie...")
        return build_semantic_docid_trie([self.train_data_path], self.tokenizer)

    def _load_unseen_docids(self) -> List[str]:
        """Load all unseen docids from new_data_path for per-query subsampling."""
        if not self.new_data_path:
            return []
        docids = []
        with open(self.new_data_path) as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("operation") == "indexing":
                    docids.append(entry["doc_id"])
        return docids

    def _build_mini_trie(self, docids: List[str]) -> TrieNode:
        """Build a trie over a small candidate set for per-query constrained decoding."""
        root = TrieNode()
        for doc_id_str in docids:
            token_ids = self.tokenizer.encode(doc_id_str, add_special_tokens=False)
            node = root
            for token_id in token_ids:
                if token_id not in node.children:
                    node.children[token_id] = TrieNode()
                node = node.children[token_id]
            node.end_of_docid = True
        return root

    def _load_cluster_data(
        self, clusters_file: str, query_assignments_file: str
    ) -> tuple[Dict[str, List[str]], Dict[str, int]]:
        """Load pre-built KMeans clusters and query→cluster assignments."""
        with open(os.path.expanduser(clusters_file)) as f:
            clusters = json.load(f)  # {cluster_id_str: [doc_id, ...]}

        query_to_cluster: Dict[str, int] = {}
        with open(os.path.expanduser(query_assignments_file)) as f:
            for line in f:
                entry = json.loads(line)
                query_to_cluster[entry["text"]] = entry["cluster_id"]

        return clusters, query_to_cluster

    def _get_cluster_candidates(self, text: str, true_docid: str) -> List[str]:
        """Return 99 docs from the query's cluster + the true doc (100 total)."""
        cluster_id = self.query_to_cluster.get(text)
        if cluster_id is None:
            # fallback: random subsample if query not found in assignments
            pool = [d for d in self.unseen_docids if d.strip() != true_docid]
            sampled = self.rng.sample(pool, min(99, len(pool)))
            return sampled + [true_docid]

        cluster_docs = self.clusters.get(str(cluster_id), [])
        # take up to 99 cluster-mates excluding the true doc
        pool = [d for d in cluster_docs if d.strip() != true_docid]
        candidates = pool[:99] + [true_docid]
        return candidates

    def _sample_candidates(self, true_docid: str) -> List[str]:
        """Sample subsample_size candidates from unseen docids, always including true_docid."""
        pool = [d for d in self.unseen_docids if d.strip() != true_docid]
        sampled = self.rng.sample(pool, min(self.subsample_size - 1, len(pool)))
        return sampled + [true_docid]

    def _create_logits_processor(
        self,
        prompt_length: int,
        trie: Optional[TrieNode] = None,
        extra_trie: Optional[TrieNode] = None,
    ) -> LogitsProcessorList:
        """Create a logits processor with specific prompt length for the current batch."""
        processor = TrieConstrainedLogitsProcessor(
            trie if trie is not None else self.trie_root,
            prompt_length,
            self.tokenizer.eos_token_id,
            extra_root=extra_trie,
        )
        return LogitsProcessorList([processor])

    def _setup_generation_config(self) -> GenerationConfig:
        """Setup generation configuration for docid generation."""
        return GenerationConfig(
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            num_beams=10,
            num_return_sequences=10,
        )

    @torch.no_grad()
    def generate_docid(
        self, text: str, trie: Optional[TrieNode] = None, extra_trie: Optional[TrieNode] = None
    ) -> List[str] | str:
        """
        Generate document ID for given text.

        Args:
            text: Input text (document or question)
            trie: Optional per-query trie override; uses self.trie_root when None.

        Returns:
            Generated document ID(s) as string or list of strings.
            For semantic docids, returns format like "<|d0_253|> <|d1_56|> <|d2_174|>"
        """
        inputs_str = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
        inputs = self.tokenizer.encode(inputs_str, return_tensors="pt").to(self.device)
        # Create logits processor with the actual prompt length for this input
        prompt_length = inputs.shape[1]

        logits_processor = self._create_logits_processor(prompt_length, trie=trie, extra_trie=extra_trie)
        outputs = self.model.generate(
            inputs,
            generation_config=self.generation_config,
            logits_processor=logits_processor,
        )
        generated_ids = [output_ids[prompt_length:] for output_ids in outputs]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)

        if len(response) == 1:
            raise ValueError("Expected multiple generated docids for retrieval task.")
        else:
            cleaned_response = [self._clean_docid(resp) for resp in response]

        return cleaned_response

    def _clean_docid(self, docid: str) -> str:
        """
        Clean the generated docid string to extract only semantic tokens.

        For semantic docids: extracts and preserves only <|dX_Y|> tokens
        For numeric docids: removes <| and |> delimiters

        Args:
            docid: Raw docid string from model

        Returns:
            Cleaned docid string containing only semantic tokens
        """
        import re

        docid = docid.strip()
        docid = docid.replace("</s>", "").replace("<|endoftext|>", "")
        docid = docid.replace("<|im_end|>", "").replace("<|im_start|>", "")

        if "<think>" in docid:
            match = re.search(
                r"</think>\s*(.*?)(?:<|im_end|>|</s>|$)", docid, re.DOTALL
            )
            if match:
                docid = match.group(1).strip()

        if "<|d" in docid:
            semantic_tokens = re.findall(r"<\|d\d+_\d+\|>", docid)
            if semantic_tokens:
                return " ".join(semantic_tokens)
            else:
                return docid.strip()
        else:
            return docid.replace("<|", "").replace("|>", "").strip()

    def evaluate_on_test_set(
        self, test_files: List[str], max_samples: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluate the model on a test set and compute accuracy.

        Args:
            test_files: Path to several test JSON file
            max_samples: Maximum number of samples to evaluate (None for all)

        Returns:
            Dictionary with evaluation results
        """
        full_results: List[Dict[str, Any]] = []

        for test_file in test_files:
            logger.info(f"Loading test data from: {test_file}")

            with open(test_file, "r") as f:
                test_data = [json.loads(line) for line in f]

            if max_samples:
                test_data = test_data[:max_samples]

            logger.info(f"Evaluating on {len(test_data)} samples...")

            hit_at_1: float = 0.0
            hit_at_10: float = 0.0
            recall_at_10: float = 0.0
            mrr_at_10: float = 0.0
            total: int = len(test_data)
            predictions: list = []

            with tqdm(total=total, desc="Evaluating", unit="sample") as pbar:
                for idx, item in enumerate(test_data):
                    if item.get("conversations") is None:
                        text = item["text"]
                        true_docid = item["doc_id"]
                    else:
                        text = item["conversations"][0]["content"]
                        true_docid = item["conversations"][1]["content"]

                    true_docid_normalized = str(true_docid).strip()

                    if self.clusters:
                        candidates = self._get_cluster_candidates(text, true_docid_normalized)
                        mini_trie = self._build_mini_trie(candidates)
                        # decode space = seen docs (memory) ∪ cluster candidates (context)
                        # matches ICLGR's decode space so the [COPY] routing advantage is measured fairly
                        predicted_docids = self.generate_docid(
                            self.query_prefix + text, trie=mini_trie, extra_trie=self.seen_trie
                        )
                    elif self.subsample_size > 0 and self.unseen_docids:
                        candidates = self._sample_candidates(true_docid_normalized)
                        mini_trie = self._build_mini_trie(candidates)
                        predicted_docids = self.generate_docid(
                            self.query_prefix + text, trie=mini_trie, extra_trie=self.seen_trie
                        )
                    else:
                        predicted_docids = self.generate_docid(self.query_prefix + text)

                    if isinstance(predicted_docids, list):
                        predicted_docids_normalized = [
                            str(p).strip() for p in predicted_docids
                        ]
                    else:
                        predicted_docids_normalized = [str(predicted_docids).strip()]

                    rank = (
                        predicted_docids_normalized.index(true_docid_normalized) + 1
                        if true_docid_normalized in predicted_docids_normalized
                        else float("inf")
                    )
                    hit_at_1 += 1.0 if rank <= 1 else 0.0
                    hit_at_10 += 1.0 if rank <= 10 else 0.0
                    recall_at_10 += 1.0 if rank <= 10 else 0.0
                    mrr_at_10 += 1.0 / rank if rank <= 10 else 0.0

                    current_hit_1 = hit_at_1 / (idx + 1)
                    current_hit_10 = hit_at_10 / (idx + 1)
                    current_recall_at_10 = recall_at_10 / (idx + 1)
                    current_mrr_at_10 = mrr_at_10 / (idx + 1)

                    pbar.set_postfix(
                        {
                            "Hit@1": f"{current_hit_1:.4f}",
                            "Hit@10": f"{current_hit_10:.4f}",
                            "Recall@10": f"{current_recall_at_10:.4f}",
                            "MRR@10": f"{current_mrr_at_10:.4f}",
                        }
                    )
                    pbar.update(1)

                    predictions.append(
                        {
                            "text": text,
                            "true_docid": true_docid,
                            "predicted_docid": predicted_docids,
                            "hit_at_1": rank <= 1,
                            "hit_at_10": rank <= 10,
                            "recall_at_10": rank <= 10,
                            "mrr_at_10": 1.0 / rank if rank <= 10 else 0.0,
                        }
                    )

            hit_at_1_score = hit_at_1 / total
            hit_at_10_score = hit_at_10 / total
            recall_at_10_score = recall_at_10 / total
            mrr_at_10_score = mrr_at_10 / total

            results = {
                "hit_at_1": hit_at_1_score,
                "hit_at_10": hit_at_10_score,
                "recall_at_10": recall_at_10_score,
                "mrr_at_10": mrr_at_10_score,
                "total": total,
                "predictions": predictions,
            }

            logger.info(f"Hit@1: {hit_at_1_score:.4f} ({hit_at_1}/{total})")
            logger.info(f"Hit@10: {hit_at_10_score:.4f} ({hit_at_10}/{total})")
            logger.info(f"Recall@10: {recall_at_10_score:.4f} ({recall_at_10}/{total})")
            logger.info(f"MRR@10: {mrr_at_10_score:.4f} ({mrr_at_10}/{total})")

            full_results.append(results)

        return full_results


@hydra.main(
    config_path="../configs", config_name="inference_conf_d0_d1", version_base=None
)
def main(cfg: DictConfig) -> int:
    max_samples = cfg.max_samples if cfg.max_samples > 0 else None
    output_file = os.path.expanduser(cfg.output_file)

    test_files = [os.path.expanduser(f) for f in cfg.test_files]

    if not os.path.exists(output_file) or cfg.force:
        inference = DecoderInference(
            model_path=os.path.expanduser(cfg.model_path),
            from_hf=cfg.from_hf,
            train_data_path=os.path.expanduser(cfg.train_file),
            new_data_path=os.path.expanduser(cfg.new_file) if cfg.new_file else "",
            base_model_path=(
                os.path.expanduser(cfg.base_model_path) if cfg.base_model_path else None
            ),
            subsample_size=cfg.get("subsample_size", 0),
            seed=cfg.get("seed", 42),
            clusters_file=os.path.expanduser(cfg.get("clusters_file", "") or ""),
            query_assignments_file=os.path.expanduser(cfg.get("query_assignments_file", "") or ""),
            query_prefix=cfg.get("query_prefix", "question: "),
        )
        full_results = inference.evaluate_on_test_set(test_files, max_samples)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(full_results, f, indent=2)
        logger.info(f"Evaluation results saved to: {output_file}")
    else:
        with open(output_file, "r") as f:
            full_results = json.load(f)

        if isinstance(full_results, dict):
            all_predictions = full_results["predictions"]
        else:
            all_predictions = [pred for r in full_results for pred in r["predictions"]]
        model_outputs = [res["predicted_docid"] for res in all_predictions]
        goldens = [res["true_docid"] for res in all_predictions]
        metrics_calculator = GRMetrics(model_outputs, goldens)
        metrics = metrics_calculator.calculate_metrics(k=[1, 10])
        logger.info(metrics)

    return 0


if __name__ == "__main__":
    exit(main())
