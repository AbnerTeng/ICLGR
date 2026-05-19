import os
import re
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

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

from datasets import load_dataset
from .inference_utils import (
    TrieNode,
    build_semantic_docid_trie,
    TrieConstrainedLogitsProcessor,
    TrieConstrainedLogitsProcessorWithTag,
    TrieConstrainedLogitsProcessorWithDualTag,
)
from .metrics import GRMetrics


os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("inference_icl_with_tag.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

ProcessorType = Literal["base", "copy", "dual_tag"]


class DecoderInferenceWithTag:
    def __init__(
        self,
        model_path: str,
        from_hf: bool,
        train_data_path: str,
        new_data_path: str,
        processor_type: ProcessorType = "base",
        base_model_path: Optional[str] = None,
        hf_corpus: Optional[str] = None,
    ) -> None:
        self.model_path = model_path if from_hf else Path(model_path)
        self.base_model_path = base_model_path
        self.train_data_path = train_data_path
        self.new_data_path = new_data_path
        self.hf_corpus = hf_corpus
        self.processor_type = processor_type
        self.device = torch.device("cuda")

        logger.info(f"Loading model from: {self.model_path}")
        logger.info(f"Processor type: {self.processor_type}")

        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.trie_root = self._build_trie()
        self.generation_config = self._setup_generation_config()

        self.copy_token_id = self.tokenizer.convert_tokens_to_ids("[COPY]")
        self.retrieve_token_id = self.tokenizer.convert_tokens_to_ids("[RETRIEVE]")

        if (
            processor_type != "base"
            and self.copy_token_id == self.tokenizer.unk_token_id
        ):
            logger.warning("[COPY] not in vocabulary — tag-based decoding may fail")
        if (
            processor_type == "dual_tag"
            and self.retrieve_token_id == self.tokenizer.unk_token_id
        ):
            logger.warning("[RETRIEVE] not in vocabulary — dual_tag decoding may fail")

        logger.info("Model loaded successfully!")

    def _load_tokenizer(self) -> AutoTokenizer:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, padding_side="left", trust_remote_code=True
            )
        except Exception as e:
            logger.warning(f"Falling back to base tokenizer due to: {e}")
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
            device_map={"": self.device} if self.device.type == "cuda" else None,
        )
        model.eval()
        return model

    def _build_trie(self) -> TrieNode:
        # dual_tag generates docids after [RETRIEVE], so needs a leading space
        # to match BPE tokenization of the first token after the tag.
        # copy/base generate directly (no preceding tag), so no leading space.
        leading_space = self.processor_type == "dual_tag"
        if self.hf_corpus:
            return self._build_trie_from_hf(self.hf_corpus, leading_space=leading_space)
        logger.info("Building docid trie from local files...")
        files = [self.train_data_path]
        if self.new_data_path:
            files.append(self.new_data_path)
        return build_semantic_docid_trie(
            files, self.tokenizer, id_field="item", leading_space=leading_space
        )

    def _build_trie_from_hf(
        self, hf_repo: str, leading_space: bool = False
    ) -> TrieNode:
        logger.info(
            f"Building docid trie from HuggingFace corpus: {hf_repo} (leading_space={leading_space})"
        )
        ds = load_dataset(hf_repo, split="train")
        root = TrieNode()
        for item in ds:
            if item.get("operation") != "indexing":
                continue
            text = (" " if leading_space else "") + item["doc_id"]
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            node = root
            for tid in token_ids:
                if tid not in node.children:
                    node.children[tid] = TrieNode()
                node = node.children[tid]
            node.end_of_docid = True
        logger.info(f"Trie built from HF corpus ({hf_repo})")
        return root

    def _setup_generation_config(self) -> GenerationConfig:
        return GenerationConfig(
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            num_beams=10,
            num_return_sequences=10,
        )

    def _extract_context_ids(self, user_content: str) -> list[list[int]]:
        """Parse 'Identifier: <docid>' lines from the prompt and return token ID lists.

        Uses a leading space so tokenization matches what the model generates
        after the [COPY] tag token (BPE splits differ with/without preceding space).
        """
        identifiers = re.findall(r"Identifier:\s*(.+)", user_content)
        return [
            self.tokenizer.encode(" " + ident.strip(), add_special_tokens=False)
            for ident in identifiers
        ]

    def _create_logits_processor(
        self,
        prompt_length: int,
        context_ids_batch: Optional[list[list[list[int]]]] = None,
    ) -> LogitsProcessorList:
        num_beams = self.generation_config.num_beams

        if self.processor_type == "base":
            processor = TrieConstrainedLogitsProcessor(
                self.trie_root, prompt_length, self.tokenizer.eos_token_id
            )
        elif self.processor_type == "copy":
            processor = TrieConstrainedLogitsProcessorWithTag(
                trie_root=self.trie_root,
                prompt_length=prompt_length,
                eos_token_id=self.tokenizer.eos_token_id,
                copy_token_id=self.copy_token_id,
                num_beams=num_beams,
                context_ids=context_ids_batch,
            )
        elif self.processor_type == "dual_tag":
            processor = TrieConstrainedLogitsProcessorWithDualTag(
                trie_root=self.trie_root,
                prompt_length=prompt_length,
                eos_token_id=self.tokenizer.eos_token_id,
                copy_token_id=self.copy_token_id,
                retrieve_token_id=self.retrieve_token_id,
                num_beams=num_beams,
                context_ids=context_ids_batch,
            )
        else:
            raise ValueError(f"Unknown processor_type: {self.processor_type}")

        return LogitsProcessorList([processor])

    def _clean_docid(self, docid: str) -> str:
        docid = docid.strip()
        for tok in [
            "</s>",
            "<|endoftext|>",
            "<|im_end|>",
            "<|im_start|>",
            "[COPY]",
            "[RETRIEVE]",
        ]:
            docid = docid.replace(tok, "")

        if "<think>" in docid:
            match = re.search(
                r"</think>\s*(.*?)(?:<\|im_end\|>|</s>|$)", docid, re.DOTALL
            )
            if match:
                docid = match.group(1).strip()

        if "<|d" in docid:
            semantic_tokens = re.findall(r"<\|d\d+_\d+\|>", docid)
            return " ".join(semantic_tokens) if semantic_tokens else docid.strip()

        return docid.replace("<|", "").replace("|>", "").strip()

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: List[str],
        context_ids_batch: Optional[list[list[list[int]]]] = None,
    ) -> List[List[str]]:
        encoding = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        prompt_length = encoding["input_ids"].shape[1]
        logits_processor = self._create_logits_processor(
            prompt_length, context_ids_batch
        )

        outputs = self.model.generate(
            **encoding,
            generation_config=self.generation_config,
            logits_processor=logits_processor,
            use_cache=True,
        )

        num_ret = self.generation_config.num_return_sequences
        generated_ids = outputs[:, prompt_length:]
        decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)

        results = []
        for i in range(len(prompts)):
            sample_responses = decoded[i * num_ret : (i + 1) * num_ret]
            results.append([self._clean_docid(r) for r in sample_responses])
        return results

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
            f"Evaluating {len(test_data)} samples  batch_size={batch_size}  processor={self.processor_type}"
        )

        hit_at_1, hit_at_10 = 0, 0
        total = len(test_data)
        predictions = []
        pattern_hits = defaultdict(lambda: {"hit_at_1": 0, "hit_at_10": 0, "total": 0})
        needs_context = self.processor_type in ("copy", "dual_tag")

        pbar = tqdm(range(0, total, batch_size), desc="Evaluating")
        for batch_start in pbar:
            batch = test_data[batch_start : batch_start + batch_size]

            prompts, true_docids, context_ids_batch = [], [], []
            for item in batch:
                if item.get("conversations") is None:
                    user_content = item["text"]
                    true_docids.append(item["doc_id"])
                else:
                    user_content = item["conversations"][0]["content"]
                    true_docids.append(
                        self._clean_docid(item["conversations"][1]["content"])
                    )

                prompts.append(
                    f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
                )
                if needs_context:
                    context_ids_batch.append(self._extract_context_ids(user_content))

            predicted_batch = self.generate_batch(
                prompts,
                context_ids_batch if needs_context else None,
            )

            for item, true_docid, predicted_docids in zip(
                batch, true_docids, predicted_batch
            ):
                true_norm = str(true_docid).strip()
                pred_norm = [str(p).strip() for p in predicted_docids]

                is_hit_1 = true_norm == pred_norm[0]
                is_hit_10 = true_norm in pred_norm

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
            pbar.set_postfix(postfix)

        per_pattern = {
            pat: {
                "hit_at_1": counts["hit_at_1"] / counts["total"],
                "hit_at_10": counts["hit_at_10"] / counts["total"],
                "hit_at_1_count": counts["hit_at_1"],
                "hit_at_10_count": counts["hit_at_10"],
                "total": counts["total"],
            }
            for pat, counts in sorted(pattern_hits.items())
        }

        results = {
            "hit_at_1": hit_at_1 / total,
            "hit_at_10": hit_at_10 / total,
            "hit_at_1_count": hit_at_1,
            "hit_at_10_count": hit_at_10,
            "total": total,
            "per_pattern": per_pattern,
            "predictions": predictions,
        }

        logger.info(f"Hit@1: {results['hit_at_1']:.4f} ({hit_at_1}/{total})")
        logger.info(f"Hit@10: {results['hit_at_10']:.4f} ({hit_at_10}/{total})")
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

    inference = DecoderInferenceWithTag(
        model_path=os.path.expanduser(cfg.model_path),
        from_hf=cfg.from_hf,
        train_data_path=os.path.expanduser(cfg.train_file),
        new_data_path=os.path.expanduser(cfg.new_file) if cfg.new_file else "",
        processor_type=cfg.get("processor_type", "base"),
        base_model_path=os.path.expanduser(cfg.base_model_path)
        if cfg.base_model_path
        else None,
        hf_corpus=cfg.get("hf_corpus", None) or None,
    )

    if not os.path.exists(output_file):
        results = inference.evaluate_on_test_set(
            os.path.expanduser(cfg.test_file),
            max_samples,
            batch_size=cfg.get("batch_size", 8),
        )
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to: {output_file}")
    else:
        with open(output_file, "r") as f:
            results = json.load(f)
        model_outputs = [res["predicted_docid"] for res in results["predictions"]]
        goldens = [res["true_docid"] for res in results["predictions"]]
        metrics = GRMetrics(model_outputs, goldens).calculate_metrics(k=[1, 10])
        logger.info(metrics)

    return 0


if __name__ == "__main__":
    main()
