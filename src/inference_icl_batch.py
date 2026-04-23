import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
from omegaconf import DictConfig

import torch
import torch.multiprocessing as mp
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
    Supports batched inference and multi-GPU data parallelism.

    Supports semantic docid tokens:
    - Semantic: "<|d0_253|> <|d1_56|> <|d2_174|>"
    """

    def __init__(
        self,
        model_path: str,
        from_hf: bool,
        train_data_path: str,
        new_data_path: str,
        base_model_path: Optional[str] = None,
        gpu_id: Optional[int] = None,
    ) -> None:
        if from_hf:
            self.model_path = model_path
        else:
            self.model_path = Path(model_path)

        self.base_model_path = base_model_path
        self.train_data_path = train_data_path
        self.new_data_path = new_data_path
        self.gpu_id = gpu_id
        self.device = self._setup_device()

        logger.info(f"[GPU {gpu_id}] Initializing Decoder Inference on {self.device}")

        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.trie_root = self._build_trie()
        self.generation_config = self._setup_generation_config()

        logger.info(f"[GPU {gpu_id}] Model loaded successfully!")

    def _setup_device(self) -> torch.device:
        if self.gpu_id is not None:
            return torch.device(f"cuda:{self.gpu_id}")
        return torch.device("cuda")

    def _load_tokenizer(self) -> AutoTokenizer:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, padding_side="left", trust_remote_code=True
            )
            if any(token.startswith("<|d") for token in tokenizer.get_vocab().keys()):
                logger.info(f"[GPU {self.gpu_id}] Detected semantic docid tokens, vocab size: {len(tokenizer)}")
        except Exception as e:
            logger.warning(f"[GPU {self.gpu_id}] Falling back to base tokenizer: {e}")
            tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen3-1.7B", padding_side="left", trust_remote_code=True
            )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def _load_model(self) -> AutoModelForCausalLM:
        load_kwargs: Dict[str, Any] = {
            "torch_dtype": torch.float16,
            # Pin each model to its own GPU; no inter-GPU tensor sharding.
            "device_map": {"": self.device},
        }

        try:
            load_kwargs["attn_implementation"] = "flash_attention_2"
            model = AutoModelForCausalLM.from_pretrained(self.model_path, **load_kwargs)
            logger.info(f"[GPU {self.gpu_id}] Loaded with Flash Attention 2")
        except Exception:
            load_kwargs.pop("attn_implementation")
            model = AutoModelForCausalLM.from_pretrained(self.model_path, **load_kwargs)
            logger.info(f"[GPU {self.gpu_id}] Flash Attention 2 unavailable, using default")

        model.eval()

        try:
            model = torch.compile(model, mode="reduce-overhead")
            logger.info(f"[GPU {self.gpu_id}] torch.compile applied")
        except Exception as e:
            logger.warning(f"[GPU {self.gpu_id}] torch.compile skipped: {e}")

        return model

    def _build_trie(self) -> TrieNode:
        files = [self.train_data_path]
        if self.new_data_path:
            files.append(self.new_data_path)
        return build_semantic_docid_trie(files, self.tokenizer)

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

    def _format_input(self, text: str, context: Optional[List[str]] = None) -> str:
        if context:
            context_str = " ".join(context)
            user_content = f"[CTX_SEARCH] Context: {context_str} Query: {text} -> Target:"
        else:
            user_content = f"[MEM_SEARCH] Query: {text} -> Target:"
        return f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"

    @torch.no_grad()
    def generate_docids_batch(
        self,
        texts: List[str],
        contexts: Optional[List[Optional[List[str]]]] = None,
    ) -> List[List[str]]:
        if contexts is None:
            contexts = [None] * len(texts)

        inputs_strs = [self._format_input(t, c) for t, c in zip(texts, contexts)]

        encoded = self.tokenizer(
            inputs_strs,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        prompt_length = encoded["input_ids"].shape[1]
        logits_processor = self._create_logits_processor(prompt_length)

        outputs = self.model.generate(
            **encoded,
            generation_config=self.generation_config,
            logits_processor=logits_processor,
        )

        num_seqs = self.generation_config.num_return_sequences
        results = []
        for i in range(len(texts)):
            batch_outputs = outputs[i * num_seqs : (i + 1) * num_seqs]
            generated_ids = [out[prompt_length:] for out in batch_outputs]
            responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
            results.append([self._clean_docid(r) for r in responses])

        return results

    @torch.no_grad()
    def generate_docid(self, text: str, context: Optional[List[str]] = None) -> List[str]:
        return self.generate_docids_batch([text], [context])[0]

    def _clean_docid(self, docid: str) -> str:
        import re

        docid = docid.strip()
        docid = docid.replace("</s>", "").replace("<|endoftext|>", "")
        docid = docid.replace("<|im_end|>", "").replace("<|im_start|>", "")

        if "<think>" in docid:
            match = re.search(r"</think>\s*(.*?)(?:<|im_end|>|</s>|$)", docid, re.DOTALL)
            if match:
                docid = match.group(1).strip()

        if "<|d" in docid:
            semantic_tokens = re.findall(r"<\|d\d+_\d+\|>", docid)
            return " ".join(semantic_tokens) if semantic_tokens else docid.strip()
        else:
            return docid.replace("<|", "").replace("|>", "").strip()

    def _evaluate_data(
        self, data: List[Dict], batch_size: int
    ) -> Tuple[int, int, List[Dict]]:
        """Process a pre-loaded list of samples. Returns (hit1, hit10, predictions)."""
        def _item_text(item):
            return item["conversations"][0]["content"] if item.get("conversations") else item["text"]

        order = sorted(range(len(data)), key=lambda i: len(_item_text(data[i])))
        data_sorted = [data[i] for i in order]

        hit_at_1 = hit_at_10 = 0
        predictions_sorted = []

        pbar = tqdm(
            range(0, len(data), batch_size),
            desc=f"GPU {self.gpu_id}",
            position=self.gpu_id or 0,
            leave=True,
        )
        for batch_start in pbar:
            batch = data_sorted[batch_start : batch_start + batch_size]
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
                true_norm = str(true_docid).strip()
                pred_norm = [str(p).strip() for p in predicted_docids]
                is_hit_1 = true_norm == pred_norm[0]
                is_hit_10 = true_norm in pred_norm
                if is_hit_1:
                    hit_at_1 += 1
                if is_hit_10:
                    hit_at_10 += 1
                predictions_sorted.append({
                    "text": text,
                    "true_docid": true_docid,
                    "predicted_docid": predicted_docids,
                    "hit_at_1": is_hit_1,
                    "hit_at_10": is_hit_10,
                    "metadata": item.get("metadata"),
                })

            done = min(batch_start + batch_size, len(data))
            pbar.set_postfix({
                "Hit@1": f"{hit_at_1/done:.4f}",
                "Hit@10": f"{hit_at_10/done:.4f}",
            })

        # Restore sorted order
        predictions = [None] * len(data)
        for sorted_idx, orig_idx in enumerate(order):
            predictions[orig_idx] = predictions_sorted[sorted_idx]

        return hit_at_1, hit_at_10, predictions

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

        logger.info(f"Evaluating on {len(test_data)} samples (batch_size={batch_size})...")
        hit_at_1, hit_at_10, predictions = self._evaluate_data(test_data, batch_size)
        total = len(test_data)

        results = {
            "hit_at_1": hit_at_1 / total,
            "hit_at_10": hit_at_10 / total,
            "hit_at_1_count": hit_at_1,
            "hit_at_10_count": hit_at_10,
            "total": total,
            "predictions": predictions,
        }
        logger.info(f"Hit@1: {results['hit_at_1']:.4f} ({hit_at_1}/{total})")
        logger.info(f"Hit@10: {results['hit_at_10']:.4f} ({hit_at_10}/{total})")
        return results


# ---------------------------------------------------------------------------
# Multi-GPU worker (must be top-level for pickle)
# ---------------------------------------------------------------------------

def _mp_worker(
    gpu_id: int,
    model_path: str,
    from_hf: bool,
    train_data_path: str,
    new_data_path: str,
    data_chunk: List[Dict],
    batch_size: int,
    return_dict,
) -> None:
    """Spawned per-GPU worker: loads its own model copy and processes data_chunk."""
    inference = DecoderInference(
        model_path=model_path,
        from_hf=from_hf,
        train_data_path=train_data_path,
        new_data_path=new_data_path,
        gpu_id=gpu_id,
    )
    hit1, hit10, preds = inference._evaluate_data(data_chunk, batch_size)
    return_dict[gpu_id] = (hit1, hit10, preds)


def evaluate_multi_gpu(
    model_path: str,
    from_hf: bool,
    train_data_path: str,
    new_data_path: str,
    test_file: str,
    max_samples: Optional[int],
    batch_size: int,
) -> Dict[str, Any]:
    """Split test data across all available GPUs and run inference in parallel."""
    with open(test_file, "r") as f:
        test_data = [json.loads(line) for line in f]
    if max_samples:
        test_data = test_data[:max_samples]

    n_gpus = torch.cuda.device_count()
    logger.info(f"Running data-parallel inference on {n_gpus} GPUs, {len(test_data)} samples")

    # Split data into n_gpus chunks, preserving original indices for reassembly
    chunks = [[] for _ in range(n_gpus)]
    chunk_indices = [[] for _ in range(n_gpus)]
    for i, item in enumerate(test_data):
        gpu = i % n_gpus
        chunks[gpu].append(item)
        chunk_indices[gpu].append(i)

    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    return_dict = manager.dict()

    processes = []
    for gpu_id in range(n_gpus):
        p = ctx.Process(
            target=_mp_worker,
            args=(
                gpu_id,
                model_path,
                from_hf,
                train_data_path,
                new_data_path,
                chunks[gpu_id],
                batch_size,
                return_dict,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Reassemble results in original order
    total = len(test_data)
    predictions = [None] * total
    hit_at_1 = hit_at_10 = 0

    for gpu_id in range(n_gpus):
        h1, h10, preds = return_dict[gpu_id]
        hit_at_1 += h1
        hit_at_10 += h10
        for local_idx, orig_idx in enumerate(chunk_indices[gpu_id]):
            predictions[orig_idx] = preds[local_idx]

    results = {
        "hit_at_1": hit_at_1 / total,
        "hit_at_10": hit_at_10 / total,
        "hit_at_1_count": hit_at_1,
        "hit_at_10_count": hit_at_10,
        "total": total,
        "predictions": predictions,
    }
    logger.info(f"Hit@1: {results['hit_at_1']:.4f} ({hit_at_1}/{total})")
    logger.info(f"Hit@10: {results['hit_at_10']:.4f} ({hit_at_10}/{total})")
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(config_path="../configs", config_name="inference_batch_conf", version_base=None)
def main(cfg: DictConfig) -> int:
    max_samples = cfg.max_samples if cfg.max_samples > 0 else None
    batch_size = cfg.get("batch_size", 8)
    output_file = os.path.expanduser(cfg.output_file)

    if not os.path.exists(output_file):
        n_gpus = torch.cuda.device_count()
        model_path = os.path.expanduser(cfg.model_path)
        train_data_path = os.path.expanduser(cfg.train_file)
        new_data_path = os.path.expanduser(cfg.new_file) if cfg.new_file else ""

        if n_gpus > 1:
            results = evaluate_multi_gpu(
                model_path=model_path,
                from_hf=cfg.from_hf,
                train_data_path=train_data_path,
                new_data_path=new_data_path,
                test_file=os.path.expanduser(cfg.test_file),
                max_samples=max_samples,
                batch_size=batch_size,
            )
        else:
            inference = DecoderInference(
                model_path=model_path,
                from_hf=cfg.from_hf,
                train_data_path=train_data_path,
                new_data_path=new_data_path,
                gpu_id=0,
            )
            results = inference.evaluate_on_test_set(
                os.path.expanduser(cfg.test_file), max_samples, batch_size
            )

        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to: {output_file}")
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
