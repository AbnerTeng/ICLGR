import os
import json
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    LogitsProcessorList,
)

from .decoder_inference_utils import (
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
        device: str = "auto",
        base_model_path: Optional[str] = None,
    ) -> None:
        if from_hf:
            self.model_path = model_path
        else:
            self.model_path = Path(model_path)

        self.base_model_path = base_model_path
        self.train_data_path = train_data_path
        self.device = self._setup_device(device)
        self.use_lora = use_lora
        self.constraint_type = constraint_type

        logger.info("Initializing Decoder Inference...")
        logger.info(f"Loading model from: {self.model_path}")
        logger.info(f"Using device: {self.device}")

        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.trie_root = self._build_trie()
        self.generation_config = self._setup_generation_config()

        logger.info("Model loaded successfully!")

    def _setup_device(self, device: str) -> torch.device:
        """Setup the computation device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")

        return torch.device(device)

    def _load_tokenizer(self) -> AutoTokenizer:
        """Load the tokenizer with special semantic tokens if present."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, padding_side="left", trust_remote_code=True
            )
            logger.info(f"Loaded tokenizer from {self.model_path}")

            if any(
                token.startswith("<|d") for token in tokenizer.get_vocab().keys()
            ):
                logger.info("Detected semantic docid tokens in vocabulary")
                logger.info(f"Vocabulary size: {len(tokenizer)}")

        except Exception as e:
            logger.warning(f"Loading tokenizer from base Qwen model due to: {e}")
            tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen3-0.6B", padding_side="left", trust_remote_code=True
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
        trie_root = build_semantic_docid_trie(
            self.train_data_path,
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

    @torch.no_grad()
    def generate_docid(self, text: str) -> List[str] | str:
        """
        Generate document ID for given text.

        Args:
            text: Input text (document or question)

        Returns:
            Generated document ID(s) as string or list of strings.
            For semantic docids, returns format like "<|d0_253|> <|d1_56|> <|d2_174|>"
        """
        inputs_str = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
        inputs = self.tokenizer.encode(inputs_str, return_tensors="pt").to(self.device)
        # Create logits processor with the actual prompt length for this input
        prompt_length = inputs.shape[1]

        logits_processor = self._create_logits_processor(prompt_length)
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

        return clean_response

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
        self, test_file: str, max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the model on a test set and compute accuracy.

        Args:
            test_file: Path to test JSON file
            max_samples: Maximum number of samples to evaluate (None for all)

        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Loading test data from: {test_file}")

        with open(test_file, "r") as f:
            test_data = [json.loads(line) for line in f]

        if max_samples:
            test_data = test_data[:max_samples]

        logger.info(f"Evaluating on {len(test_data)} samples...")

        hit_at_1 = 0
        hit_at_10 = 0
        total = len(test_data)
        predictions = []

        pbar = tqdm(test_data, desc="Evaluating")
        for idx, item in enumerate(pbar, 1):
            if item.get("conversations") is None:
                text = item["text"]
                true_docid = item["doc_id"]
            else:
                text = item["conversations"][0]["content"]
                true_docid = item["conversations"][1]["content"]

            predicted_docids = self.generate_docid("question: " + text)
            print(f"Predicted_docids: {predicted_docids}, True_docid: {true_docid}")

            true_docid_normalized = str(true_docid).strip()

            if isinstance(predicted_docids, list):
                predicted_docids_normalized = [str(p).strip() for p in predicted_docids]
            else:
                predicted_docids_normalized = [str(predicted_docids).strip()]

            is_hit_1 = true_docid_normalized == predicted_docids_normalized[0]

            if is_hit_1:
                hit_at_1 += 1

            is_hit_10 = true_docid_normalized in predicted_docids_normalized

            if is_hit_10:
                hit_at_10 += 1

            current_hit_1 = hit_at_1 / idx
            current_hit_10 = hit_at_10 / idx
            pbar.set_postfix(
                {"Hit@1": f"{current_hit_1:.4f}", "Hit@10": f"{current_hit_10:.4f}"}
            )

            predictions.append(
                {
                    "text": text,
                    "true_docid": true_docid,
                    "predicted_docid": predicted_docids,
                    "hit_at_1": is_hit_1,
                    "hit_at_10": is_hit_10,
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


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint",
    )
    parser.add_argument(
        "--from_hf",
        type=bool,
        required=True,
        default=False,
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        help="Path to the base model checkpoint (for LoRA models)",
    )
    parser.add_argument("--input", type=str, help="Input text for single inference")
    parser.add_argument(
        "--train_file",
        type=str,
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="data/CORAL/CORAL_1k_test.json",
        help="Path to test file for batch inference",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate model on test set and compute accuracy",
    )
    parser.add_argument(
        "--max_samples", type=int, help="Maximum number of samples for evaluation"
    )
    parser.add_argument("--output_file", type=str, help="Output file to save results")

    return parser.parse_args()


def main():
    device = "auto"
    args = get_args()
    inference = DecoderInference(
        args.model_path,
        args.from_hf,
        args.train_file,
        device,
        args.base_model_path,
    )

    if args.input:
        logger.info(f"Input text: {args.input}")
        docid = inference.generate_docid(args.input)
        logger.info(f"Generated DocID: {docid}")

    elif args.evaluate:
        if not os.path.exists(args.output_file):
            results = inference.evaluate_on_test_set(args.test_file, args.max_samples)

            if args.output_file:
                with open(args.output_file, "w") as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Evaluation results saved to: {args.output_file}")
            else:
                logger.info(
                    f"Hit@1: {results['hit_at_1']:.4f} ({results['hit_at_1_count']}/{results['total']})"
                )
                logger.info(
                    f"Hit@10: {results['hit_at_10']:.4f} ({results['hit_at_10_count']}/{results['total']})"
                )
        else:
            with open(args.output_file, "r") as f:
                results = json.load(f)["predictions"]

            model_outputs = [res["predicted_docid"] for res in results]
            goldens = [res["true_docid"] for res in results]
            metrics_calculator = GRMetrics(model_outputs, goldens)
            metrics = metrics_calculator.calculate_metrics(k=[1, 10])

            print(metrics)

    else:
        logger.error("Please specify either --input, --batch_inference, or --evaluate")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
