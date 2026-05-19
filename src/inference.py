import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import hydra
from omegaconf import DictConfig
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

logger = logging.getLogger(__name__)


class DecoderInference:
    """Inference class for decoder-only models trained on docid generation task."""

    def __init__(
        self,
        model_path: str,
        from_hf: bool,
        train_data_path: str,
        num_beams: int = 10,
        num_return: int = 10,
        max_new_tokens: int = 50,
        device: str = "cuda",
        base_model_path: Optional[str] = None,
    ) -> None:
        self.model_path = model_path if from_hf else Path(model_path)
        self.base_model_path = base_model_path
        self.train_data_path = train_data_path
        self.num_beams = num_beams
        self.num_return = num_return
        self.max_new_tokens = max_new_tokens
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading model from: {self.model_path}")
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.trie_root = self._build_trie()
        self.generation_config = self._setup_generation_config()
        logger.info("Model loaded successfully!")

    def _load_tokenizer(self) -> AutoTokenizer:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, padding_side="left", trust_remote_code=True
            )
            if any(token.startswith("<|d") for token in tokenizer.get_vocab().keys()):
                logger.info("Detected semantic docid tokens in vocabulary")
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
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map="auto" if self.device.type == "cuda" else None,
        )
        model.eval()
        return model

    def _build_trie(self) -> TrieNode:
        logger.info("Building semantic docid trie...")
        return build_semantic_docid_trie(self.train_data_path, self.tokenizer)

    def _create_logits_processor(self, prompt_length: int) -> LogitsProcessorList:
        processor = TrieConstrainedLogitsProcessor(
            self.trie_root, prompt_length, self.tokenizer.eos_token_id
        )
        return LogitsProcessorList([processor])

    def _setup_generation_config(self) -> GenerationConfig:
        return GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            num_beams=self.num_beams,
            num_return_sequences=self.num_return,
        )

    @torch.no_grad()
    def generate_docid(self, text: str) -> List[str]:
        inputs_str = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
        inputs = self.tokenizer.encode(inputs_str, return_tensors="pt").to(self.device)
        prompt_length = inputs.shape[1]
        logits_processor = self._create_logits_processor(prompt_length)
        outputs = self.model.generate(
            inputs,
            generation_config=self.generation_config,
            logits_processor=logits_processor,
        )
        generated_ids = [output_ids[prompt_length:] for output_ids in outputs]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        return [self._clean_docid(r) for r in response]

    def _clean_docid(self, docid: str) -> str:
        import re

        docid = docid.strip()
        for tok in ("</s>", "<|endoftext|>", "<|im_end|>", "<|im_start|>"):
            docid = docid.replace(tok, "")

        if "<think>" in docid:
            match = re.search(
                r"</think>\s*(.*?)(?:<\|im_end\|>|</s>|$)", docid, re.DOTALL
            )
            if match:
                docid = match.group(1).strip()

        if "<|d" in docid:
            tokens = re.findall(r"<\|d\d+_\d+\|>", docid)
            return " ".join(tokens) if tokens else docid.strip()

        return docid.replace("<|", "").replace("|>", "").strip()

    def evaluate_on_test_set(
        self, test_file: str, max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        logger.info(f"Loading test data from: {test_file}")
        with open(test_file) as f:
            test_data = [json.loads(line) for line in f]

        if max_samples:
            test_data = test_data[:max_samples]

        logger.info(f"Evaluating on {len(test_data)} samples...")

        hit_at_1 = hit_at_10 = recall_at_10 = mrr_at_10 = 0.0
        total = len(test_data)
        predictions = []

        with tqdm(total=total, desc="Evaluating", unit="sample") as pbar:
            for idx, item in enumerate(test_data):
                if item.get("conversations") is None:
                    text = item["text"]
                    true_docid = item["doc_id"]
                else:
                    text = item["conversations"][0]["content"]
                    true_docid = item["conversations"][1]["content"]

                predicted_docids = self.generate_docid("question: " + text)
                # logger.info(f"Predicted: {predicted_docids}  True: {true_docid}")

                true_norm = str(true_docid).strip()
                pred_norm = [str(p).strip() for p in predicted_docids]

                rank = (
                    pred_norm.index(true_norm) + 1
                    if true_norm in pred_norm
                    else float("inf")
                )
                hit_at_1 += 1.0 if rank <= 1 else 0.0
                hit_at_10 += 1.0 if rank <= 10 else 0.0
                recall_at_10 += 1.0 if rank <= 10 else 0.0
                mrr_at_10 += 1.0 / rank if rank <= 10 else 0.0

                pbar.set_postfix(
                    {
                        "Hit@1": f"{hit_at_1 / (idx + 1):.4f}",
                        "Hit@10": f"{hit_at_10 / (idx + 1):.4f}",
                        "MRR@10": f"{mrr_at_10 / (idx + 1):.4f}",
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
                        "mrr_at_10": 1.0 / rank if rank <= 10 else 0.0,
                    }
                )

        results = {
            "hit_at_1": hit_at_1 / total,
            "hit_at_10": hit_at_10 / total,
            "recall_at_10": recall_at_10 / total,
            "mrr_at_10": mrr_at_10 / total,
            "total": total,
            "predictions": predictions,
        }
        logger.info(f"Hit@1:  {results['hit_at_1']:.4f} ({hit_at_1}/{total})")
        logger.info(f"Hit@10: {results['hit_at_10']:.4f} ({hit_at_10}/{total})")
        logger.info(f"MRR@10: {results['mrr_at_10']:.4f} ({mrr_at_10}/{total})")
        return results


@hydra.main(config_path="../configs", config_name="inference_stage1", version_base=None)
def main(cfg: DictConfig) -> None:
    inference = DecoderInference(
        model_path=cfg.model_path,
        from_hf=cfg.get("from_hf", False),
        train_data_path=cfg.train_file,
        num_beams=cfg.get("num_beams", 10),
        num_return=cfg.get("num_return", 10),
        max_new_tokens=cfg.get("max_new_tokens", 50),
        base_model_path=cfg.get("base_model_path", None),
    )

    output_file = cfg.get("output_file", None)

    if output_file and Path(output_file).exists():
        with open(output_file) as f:
            saved = json.load(f)
        model_outputs = [r["predicted_docid"] for r in saved["predictions"]]
        goldens = [r["true_docid"] for r in saved["predictions"]]
        metrics = GRMetrics(model_outputs, goldens).calculate_metrics(k=[1, 10])
        print(metrics)
        return

    results = inference.evaluate_on_test_set(
        cfg.test_file, cfg.get("max_samples", None)
    )

    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
