import os
import json
import logging
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
    Supports batched inference for improved GPU utilization.

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
        new_data_path: str,
        base_model_path: Optional[str] = None,
    ) -> None:
        if from_hf:
            self.model_path = model_path
        else:
            self.model_path = Path(model_path)

        self.base_model_path = base_model_path
        self.train_data_path = train_data_path
        self.new_data_path = new_data_path
        self.device = self._setup_device()

        logger.info("Initializing Decoder Inference...")
        logger.info(f"Loading model from: {self.model_path}")
        logger.info(f"Using device: {self.device}")

        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.trie_root = self._build_trie()
        self.generation_config = self._setup_generation_config()

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
        logger.info("Building semantic docid trie...")

        files = [self.train_data_path]

        if self.new_data_path:
            files.append(self.new_data_path)

        trie_root = build_semantic_docid_trie(
            files,
            self.tokenizer,
        )

        return trie_root

    def _create_logits_processor(self, prompt_length: int) -> LogitsProcessorList:
        """Create a logits processor with specific prompt length for the current batch."""
        processor = TrieConstrainedLogitsProcessor(
            self.trie_root, prompt_length, self.tokenizer.eos_token_id
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

    def _format_input(self, text: str, context: Optional[List[str]] = None) -> str:
        if context:
            context_str = " ".join(context)
            user_content = (
                f"[CTX_SEARCH] Context: {context_str} Query: {text} -> Target:"
            )
        else:
            user_content = f"[MEM_SEARCH] Query: {text} -> Target:"
        return f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"

    @torch.no_grad()
    def generate_docids_batch(
        self,
        texts: List[str],
        contexts: Optional[List[Optional[List[str]]]] = None,
    ) -> List[List[str]]:
        """
        Generate document IDs for a batch of queries.

        Args:
            texts: List of query texts.
            contexts: Optional list of context lists, one per query.
                      If None, all queries use [MEM_SEARCH].

        Returns:
            List of lists: for each query, num_return_sequences predicted docids.
        """
        if contexts is None:
            contexts = [None] * len(texts)

        inputs_strs = [self._format_input(t, c) for t, c in zip(texts, contexts)]

        encoded = self.tokenizer(
            inputs_strs,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        # With left-padding all sequences share the same padded prompt_length,
        # so TrieConstrainedLogitsProcessor can use a single prompt_length value.
        prompt_length = encoded["input_ids"].shape[1]
        logits_processor = self._create_logits_processor(prompt_length)

        outputs = self.model.generate(
            **encoded,
            generation_config=self.generation_config,
            logits_processor=logits_processor,
        )

        # outputs: (batch_size * num_return_sequences, seq_len)
        num_seqs = self.generation_config.num_return_sequences
        results = []
        for i in range(len(texts)):
            batch_outputs = outputs[i * num_seqs : (i + 1) * num_seqs]
            generated_ids = [out[prompt_length:] for out in batch_outputs]
            responses = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=False
            )
            results.append([self._clean_docid(r) for r in responses])

        return results

    @torch.no_grad()
    def generate_docid(
        self, text: str, context: Optional[List[str]] = None
    ) -> List[str]:
        """Generate document IDs for a single query (wraps generate_docids_batch)."""
        return self.generate_docids_batch([text], [context])[0]

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
        self,
        test_file: str,
        max_samples: Optional[int] = None,
        batch_size: int = 8,
    ) -> Dict[str, Any]:
        """
        Evaluate the model on a test set and compute accuracy.

        Args:
            test_file: Path to test JSON file
            max_samples: Maximum number of samples to evaluate (None for all)
            batch_size: Number of queries to process in parallel

        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Loading test data from: {test_file}")

        with open(test_file, "r") as f:
            test_data = [json.loads(line) for line in f]

        if max_samples:
            test_data = test_data[:max_samples]

        logger.info(
            f"Evaluating on {len(test_data)} samples (batch_size={batch_size})..."
        )

        hit_at_1 = 0
        hit_at_10 = 0
        total = len(test_data)
        predictions = []

        pbar = tqdm(range(0, total, batch_size), desc="Evaluating")

        for batch_start in pbar:
            batch = test_data[batch_start : batch_start + batch_size]

            texts, true_docids = [], []
            for item in batch:
                if item.get("conversations") is None:
                    texts.append(item["text"])
                    true_docids.append(item["doc_id"])
                else:
                    texts.append(item["conversations"][0]["content"])
                    true_docids.append(item["conversations"][1]["content"])

            batch_predicted = self.generate_docids_batch(texts)

            for item, text, true_docid, predicted_docids in zip(
                batch, texts, true_docids, batch_predicted
            ):
                logger.debug(f"Predicted: {predicted_docids}, True: {true_docid}")

                true_docid_normalized = str(true_docid).strip()
                predicted_docids_normalized = [str(p).strip() for p in predicted_docids]

                is_hit_1 = true_docid_normalized == predicted_docids_normalized[0]
                is_hit_10 = true_docid_normalized in predicted_docids_normalized

                if is_hit_1:
                    hit_at_1 += 1
                if is_hit_10:
                    hit_at_10 += 1

                predictions.append(
                    {
                        "text": text,
                        "true_docid": true_docid,
                        "predicted_docid": predicted_docids,
                        "hit_at_1": is_hit_1,
                        "hit_at_10": is_hit_10,
                        "metadata": item.get("metadata"),
                    }
                )

            done = min(batch_start + batch_size, total)
            pbar.set_postfix(
                {
                    "Hit@1": f"{hit_at_1 / done:.4f}",
                    "Hit@10": f"{hit_at_10 / done:.4f}",
                }
            )

        hit_at_1_score = hit_at_1 / total
        hit_at_10_score = hit_at_10 / total

        results = {
            "hit_at_1": hit_at_1_score,
            "hit_at_10": hit_at_10_score,
            "hit_at_1_count": hit_at_1,
            "hit_at_10_count": hit_at_10,
            "total": total,
            "predictions": predictions,
        }

        logger.info(f"Hit@1: {hit_at_1_score:.4f} ({hit_at_1}/{total})")
        logger.info(f"Hit@10: {hit_at_10_score:.4f} ({hit_at_10}/{total})")

        return results


@hydra.main(config_path="../configs", config_name="inference_conf", version_base=None)
def main(cfg: DictConfig) -> int:
    max_samples = cfg.max_samples if cfg.max_samples > 0 else None
    batch_size = cfg.get("batch_size", 8)
    output_file = os.path.expanduser(cfg.output_file)

    inference = DecoderInference(
        model_path=os.path.expanduser(cfg.model_path),
        from_hf=cfg.from_hf,
        train_data_path=os.path.expanduser(cfg.train_file),
        new_data_path=os.path.expanduser(cfg.new_file) if cfg.new_file else "",
        base_model_path=(
            os.path.expanduser(cfg.base_model_path) if cfg.base_model_path else None
        ),
    )

    if not os.path.exists(output_file):
        results = inference.evaluate_on_test_set(
            os.path.expanduser(cfg.test_file), max_samples, batch_size
        )

        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Evaluation results saved to: {output_file}")
        else:
            logger.info(
                f"Hit@1: {results['hit_at_1']:.4f} ({results['hit_at_1_count']}/{results['total']})"
            )
            logger.info(
                f"Hit@10: {results['hit_at_10']:.4f} ({results['hit_at_10_count']}/{results['total']})"
            )
    else:
        with open(output_file, "r") as f:
            results = json.load(f)["predictions"]

        model_outputs = [res["predicted_docid"] for res in results]
        goldens = [res["true_docid"] for res in results]
        metrics_calculator = GRMetrics(model_outputs, goldens)
        metrics = metrics_calculator.calculate_metrics(k=[1, 10])

        logger.info(metrics)

    return 0


if __name__ == "__main__":
    main()
