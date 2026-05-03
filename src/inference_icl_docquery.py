import os
import re
import json
import logging
from pathlib import Path
from collections import defaultdict
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
    TrieConstrainedLogitsProcessor,
)
from .metrics import GRMetrics


os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("inference_icl_docquery.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def build_docquery_trie(data_paths: List[str], tokenizer) -> TrieNode:
    """Build trie from conversations-format data (docquery / ICL).

    Collects all unique generation targets so the trie covers both
    bare identifiers (zero-shot) and ``[COPY] <id>`` (doc-position) patterns.
    """
    root = TrieNode()

    if isinstance(data_paths, str):
        data_paths = [data_paths]

    docid_set: set = set()

    for path in data_paths:
        with open(path, "r") as f:
            for line in f:
                item = json.loads(line)

                if item.get("operation") == "indexing":
                    docid_set.add(item["doc_id"])
                elif item.get("conversations"):
                    assistant_content = item["conversations"][1]["content"]
                    docid_set.add(assistant_content)
                    target_id = item.get("metadata", {}).get("target_id")
                    if target_id:
                        docid_set.add(target_id)
                        docid_set.add(f"[COPY] {target_id}")

    logger.info(f"Building trie from {len(docid_set)} unique targets")

    for doc_id_str in docid_set:
        token_ids = tokenizer.encode(doc_id_str, add_special_tokens=False)
        node = root
        for token_id in token_ids:
            if token_id not in node.children:
                node.children[token_id] = TrieNode()
            node = node.children[token_id]
        node.end_of_docid = True

    return root


class DocQueryInference:
    """Inference for models trained on the docquery / ICL conversation format.

    Expected data schema (JSONL)::

        {
          "conversations": [
            {"role": "user",      "content": "## Documents ...\\n## Task\\nQuery: ...\\nAnswer:"},
            {"role": "assistant", "content": "[COPY] Some Document Title"}
          ],
          "metadata": {"pattern": "doc_pos_back", "target_id": "Some Document Title"}
        }
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

        logger.info("Initializing DocQuery Inference...")
        logger.info(f"Loading model from: {self.model_path}")
        logger.info(f"Using device: {self.device}")

        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.trie_root = self._build_trie()
        self.generation_config = self._setup_generation_config()

        logger.info("Model loaded successfully!")

    def _setup_device(self) -> torch.device:
        return torch.device("cuda")

    def _load_tokenizer(self) -> AutoTokenizer:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, padding_side="left", trust_remote_code=True
            )
            logger.info(f"Loaded tokenizer from {self.model_path}")
        except Exception as e:
            logger.warning(f"Loading tokenizer from base Qwen model due to: {e}")
            tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen3-1.7B", padding_side="left", trust_remote_code=True
            )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def _load_model(self) -> AutoModelForCausalLM:
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=(
                torch.float16 if self.device.type == "cuda" else torch.float32
            ),
            device_map={"": self.device} if self.device.type == "cuda" else None,
        )
        model.eval()
        return model

    def _build_trie(self) -> TrieNode:
        logger.info("Building docquery trie...")

        files = [self.train_data_path]
        if self.new_data_path:
            files.append(self.new_data_path)

        return build_docquery_trie(files, self.tokenizer)

    def _create_logits_processor(self, prompt_length: int) -> LogitsProcessorList:
        processor = TrieConstrainedLogitsProcessor(
            self.trie_root, prompt_length, self.tokenizer.eos_token_id
        )
        return LogitsProcessorList([processor])

    def _setup_generation_config(self) -> GenerationConfig:
        return GenerationConfig(
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            num_beams=10,
            num_return_sequences=10,
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_docid(self, text: str) -> List[str]:
        """Generate document ID(s) for a single user content string."""
        return self._generate_from_prompts(
            [f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"]
        )[0]

    @torch.no_grad()
    def _generate_from_prompts(self, prompts: List[str]) -> List[List[str]]:
        """Run generation on already-formatted prompt strings."""
        encoding = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(self.device)
        prompt_length = encoding["input_ids"].shape[1]

        logits_processor = self._create_logits_processor(prompt_length)
        outputs = self.model.generate(
            **encoding,
            generation_config=self.generation_config,
            logits_processor=logits_processor,
        )

        num_ret = self.generation_config.num_return_sequences
        generated_ids = outputs[:, prompt_length:]
        decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)

        results = []
        for i in range(len(prompts)):
            sample_responses = decoded[i * num_ret : (i + 1) * num_ret]
            results.append([self._clean_docid(r) for r in sample_responses])
        return results

    # ------------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_docid(docid: str) -> str:
        """Strip special tokens and ``[COPY]`` prefix from a generated string."""
        docid = docid.strip()
        docid = docid.replace("</s>", "").replace("<|endoftext|>", "")
        docid = docid.replace("<|im_end|>", "").replace("<|im_start|>", "")

        if "<think>" in docid:
            match = re.search(
                r"</think>\s*(.*?)(?:<\|im_end\|>|</s>|$)", docid, re.DOTALL
            )
            if match:
                docid = match.group(1).strip()

        if "<|d" in docid:
            semantic_tokens = re.findall(r"<\|d\d+_\d+\|>", docid)
            if semantic_tokens:
                return " ".join(semantic_tokens)
            return docid.strip()

        docid = docid.strip()
        if docid.startswith("[COPY]"):
            docid = docid[len("[COPY]") :].strip()

        return docid

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_on_test_set(
        self,
        test_file: str,
        max_samples: Optional[int] = None,
        batch_size: int = 8,
    ) -> Dict[str, Any]:
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
        predictions: List[Dict[str, Any]] = []
        pattern_hits = defaultdict(lambda: {"hit_at_1": 0, "hit_at_10": 0, "total": 0})

        pbar = tqdm(range(0, total, batch_size), desc="Evaluating")

        for batch_start in pbar:
            batch = test_data[batch_start : batch_start + batch_size]

            prompts, true_docids = [], []
            for item in batch:
                if item.get("conversations") is None:
                    prompts.append(
                        f"<|im_start|>user\n{item['text']}<|im_end|>\n<|im_start|>assistant\n"
                    )
                    true_docids.append(item["doc_id"])
                else:
                    prompts.append(
                        f"<|im_start|>user\n{item['conversations'][0]['content']}<|im_end|>\n<|im_start|>assistant\n"
                    )
                    target_id = item.get("metadata", {}).get("target_id")
                    if target_id:
                        true_docids.append(target_id)
                    else:
                        true_docids.append(
                            self._clean_docid(item["conversations"][1]["content"])
                        )

            batch_predicted = self._generate_from_prompts(prompts)

            for item, text, true_docid, predicted_docids in zip(
                batch, prompts, true_docids, batch_predicted
            ):
                logger.debug(
                    f"Predicted_docids: {predicted_docids}, True_docid: {true_docid}"
                )

                true_docid_normalized = str(true_docid).strip()
                predicted_docids_normalized = [str(p).strip() for p in predicted_docids]

                is_hit_1 = true_docid_normalized == predicted_docids_normalized[0]
                is_hit_10 = true_docid_normalized in predicted_docids_normalized

                if is_hit_1:
                    hit_at_1 += 1
                if is_hit_10:
                    hit_at_10 += 1

                pattern = (item.get("metadata") or {}).get("pattern", "unknown")
                pattern_hits[pattern]["total"] += 1
                if is_hit_1:
                    pattern_hits[pattern]["hit_at_1"] += 1
                if is_hit_10:
                    pattern_hits[pattern]["hit_at_10"] += 1

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

            idx = len(predictions)
            postfix = {
                "Hit@1": f"{hit_at_1 / idx:.2f}",
                "Hit@10": f"{hit_at_10 / idx:.2f}",
            }
            for pat, counts in sorted(pattern_hits.items()):
                n = counts["total"]
                postfix[f"{pat}/H@1"] = f"{counts['hit_at_1'] / n:.2f}"
                postfix[f"{pat}/H@10"] = f"{counts['hit_at_10'] / n:.2f}"
            pbar.set_postfix(postfix)

        hit_at_1_score = hit_at_1 / total
        hit_at_10_score = hit_at_10 / total

        per_pattern = {}
        for pat, counts in sorted(pattern_hits.items()):
            n = counts["total"]
            per_pattern[pat] = {
                "hit_at_1": counts["hit_at_1"] / n,
                "hit_at_10": counts["hit_at_10"] / n,
                "hit_at_1_count": counts["hit_at_1"],
                "hit_at_10_count": counts["hit_at_10"],
                "total": n,
            }

        results = {
            "hit_at_1": hit_at_1_score,
            "hit_at_10": hit_at_10_score,
            "hit_at_1_count": hit_at_1,
            "hit_at_10_count": hit_at_10,
            "total": total,
            "per_pattern": per_pattern,
            "predictions": predictions,
        }

        logger.info(f"Hit@1: {hit_at_1_score:.4f} ({hit_at_1}/{total})")
        logger.info(f"Hit@10: {hit_at_10_score:.4f} ({hit_at_10}/{total})")
        for pat, m in per_pattern.items():
            logger.info(
                f"  [{pat}] Hit@1: {m['hit_at_1']:.4f} ({m['hit_at_1_count']}/{m['total']})  "
                f"Hit@10: {m['hit_at_10']:.4f} ({m['hit_at_10_count']}/{m['total']})"
            )

        return results


@hydra.main(config_path="../configs", config_name="inference_conf", version_base=None)
def main(cfg: DictConfig) -> int:
    max_samples = cfg.max_samples if cfg.max_samples > 0 else None
    output_file = os.path.expanduser(cfg.output_file)

    inference = DocQueryInference(
        model_path=os.path.expanduser(cfg.model_path),
        from_hf=cfg.from_hf,
        train_data_path=os.path.expanduser(cfg.train_file),
        new_data_path=os.path.expanduser(cfg.new_file) if cfg.new_file else "",
        base_model_path=(
            os.path.expanduser(cfg.base_model_path) if cfg.base_model_path else None
        ),
    )

    if not os.path.exists(output_file):
        batch_size = cfg.batch_size if hasattr(cfg, "batch_size") else 8
        results = inference.evaluate_on_test_set(
            os.path.expanduser(cfg.test_file), max_samples, batch_size=batch_size
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
