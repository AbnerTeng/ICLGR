"""Refine compressed indexing summaries using both original and compressed text.

This pass reads aligned before/after JSONL files. Query rows are copied from
the after file unchanged; indexing rows are rewritten by vLLM using the
original document as evidence and the current compressed text as a draft.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

from src.compress_indexing_text import (
    VllmCompressor,
    clean_generation,
    durable_flush,
    enforce_length,
    lead_focused_text,
    maybe_durable_flush,
    normalize_text,
    target_min_tokens,
    trim_tail,
)


def build_refine_prompt(
    source_text: str,
    draft_summary: str,
    doc_id: str,
    target_tokens: int,
) -> str:
    if target_tokens == 300:
        target_range = "220-300 tokens"
    else:
        target_range = f"{target_min_tokens(target_tokens)}-{target_tokens} tokens"

    return (
        "You are improving a compressed summary for Natural Questions retrieval training.\n\n"
        "Rewrite the current summary into ONE clean retrieval-oriented summary.\n\n"
        f"Target length: {target_range}.\n\n"
        "Rules:\n"
        "- Use the current summary as a draft, but fix noise, copied text, missing context, and truncation.\n"
        "- Use the source document only to verify and enrich factual details.\n"
        "- Output ONLY the rewritten summary.\n"
        "- Do NOT copy source text verbatim.\n"
        "- Do NOT continue the original document.\n"
        "- Do NOT include headings, table of contents, navigation text, references, external links, infobox fields, or maintenance notices.\n"
        "- If the page is primarily a list or table, summarize what the list contains, its categories, and representative important entities. Do not copy rows.\n"
        "- Preserve key entities, dates, definitions, aliases, relationships, major events, and facts likely to answer questions.\n"
        "- Use complete sentences and stop after the summary.\n\n"
        f"Title: {doc_id}\n\n"
        "CURRENT SUMMARY:\n"
        "<<<\n"
        f"{draft_summary}\n"
        ">>>\n\n"
        "SOURCE DOCUMENT:\n"
        "<<<\n"
        f"{source_text}\n"
        ">>>\n\n"
        "REWRITTEN SUMMARY:"
    )


class RefineCompressor(VllmCompressor):
    def refine_batch(
        self,
        pairs: list[tuple[dict, dict]],
        target_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
    ) -> list[str]:
        from vllm import SamplingParams

        prompts = []
        min_tokens = target_min_tokens(target_tokens)
        for before, after in pairs:
            doc_id = before.get("doc_id", "")
            source_text = self._truncate_document(
                lead_focused_text(
                    trim_tail(
                        normalize_text(before.get("text", "")),
                        min_tokens=min_tokens,
                    ),
                    doc_id=doc_id,
                )
            )
            draft_summary = clean_generation(after.get("text", ""))
            prompts.append(
                self._truncate_prompt(
                    self._format_prompt(
                        build_refine_prompt(
                            source_text=source_text,
                            draft_summary=draft_summary,
                            doc_id=doc_id,
                            target_tokens=target_tokens,
                        )
                    )
                )
            )

        outputs = self.llm.generate(
            prompts,
            SamplingParams(
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_tokens=self.max_new_tokens,
            ),
        )

        decoded = [output.outputs[0].text for output in outputs]
        return [
            enforce_length(
                text,
                source_text=before.get("text", ""),
                doc_id=before.get("doc_id", ""),
                min_tokens=min_tokens,
                max_tokens=target_tokens,
            )
            for text, (before, _) in zip(decoded, pairs)
        ]


def flush_refine_batch(
    compressor: RefineCompressor,
    batch: list[tuple[int, dict, dict]],
    buffered_rows: dict[int, dict],
    fout,
    next_to_write: int,
    written_rows: int,
    flush_every: int,
    target_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> tuple[int, int]:
    if not batch:
        return next_to_write, written_rows

    summaries = compressor.refine_batch(
        [(before, after) for _, before, after in batch],
        target_tokens=target_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    for (idx, _, after), summary in zip(batch, summaries):
        after["text"] = summary
        buffered_rows[idx] = after

    while next_to_write in buffered_rows:
        fout.write(json.dumps(buffered_rows.pop(next_to_write), ensure_ascii=False) + "\n")
        next_to_write += 1
        written_rows += 1
        maybe_durable_flush(fout, written_rows, flush_every)
    return next_to_write, written_rows


def refine_file_vllm(
    before_path: Path,
    after_path: Path,
    dst: Path,
    model_name: str,
    target_tokens: int,
    batch_size: int,
    dtype: str,
    max_input_tokens: int,
    max_new_tokens: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    enforce_eager: bool,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    start_line: int,
    end_line: Optional[int],
    flush_every: int,
    limit: Optional[int],
) -> None:
    from tqdm import tqdm

    dst.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    compressor = RefineCompressor(
        model_name=model_name,
        dtype=dtype,
        max_input_tokens=max_input_tokens,
        max_new_tokens=max_new_tokens,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
    )

    total = 0
    indexing = 0
    batch: list[tuple[int, dict, dict]] = []
    buffered_rows: dict[int, dict] = {}
    next_to_write = 0
    written_rows = 0
    local_idx = 0

    with (
        before_path.open("r", encoding="utf-8") as f_before,
        after_path.open("r", encoding="utf-8") as f_after,
        dst.open("w", encoding="utf-8") as fout,
    ):
        for line_no, (before_line, after_line) in enumerate(
            tqdm(zip(f_before, f_after), desc="Refining indexing rows"),
            start=1,
        ):
            if line_no < start_line:
                continue
            if end_line is not None and line_no > end_line:
                break
            if limit is not None and total >= limit:
                break
            if not before_line.strip() or not after_line.strip():
                continue

            before = json.loads(before_line)
            after = json.loads(after_line)
            if before.get("operation") != after.get("operation"):
                raise ValueError(f"Operation mismatch at line {line_no}")
            if before.get("doc_id") != after.get("doc_id"):
                raise ValueError(f"doc_id mismatch at line {line_no}")

            idx = local_idx
            local_idx += 1
            total += 1

            if after.get("operation") != "indexing":
                buffered_rows[idx] = after
            else:
                indexing += 1
                batch.append((idx, before, after))
                if len(batch) >= batch_size:
                    next_to_write, written_rows = flush_refine_batch(
                        compressor,
                        batch,
                        buffered_rows,
                        fout,
                        next_to_write,
                        written_rows,
                        flush_every,
                        target_tokens,
                        temperature,
                        top_p,
                        repetition_penalty,
                    )
                    batch = []

            while next_to_write in buffered_rows:
                fout.write(json.dumps(buffered_rows.pop(next_to_write), ensure_ascii=False) + "\n")
                next_to_write += 1
                written_rows += 1
                maybe_durable_flush(fout, written_rows, flush_every)

        next_to_write, written_rows = flush_refine_batch(
            compressor,
            batch,
            buffered_rows,
            fout,
            next_to_write,
            written_rows,
            flush_every,
            target_tokens,
            temperature,
            top_p,
            repetition_penalty,
        )
        while next_to_write in buffered_rows:
            fout.write(json.dumps(buffered_rows.pop(next_to_write), ensure_ascii=False) + "\n")
            next_to_write += 1
            written_rows += 1
            maybe_durable_flush(fout, written_rows, flush_every)

        durable_flush(fout)

    print(f"Wrote {dst}")
    print(f"  rows={total} refined_indexing={indexing}")
    elapsed = time.perf_counter() - start
    if elapsed > 0:
        print(f"  elapsed_sec={elapsed:.2f}")
        print(f"  indexing_rows_per_sec={indexing / elapsed:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", default="data/nq-item-id/data/train.jsonl")
    parser.add_argument("--after", default="data/nq-item-id-llm-compressed-final/data/train.jsonl")
    parser.add_argument("--dst", default="data/nq-item-id-llm-refined/data/train.jsonl")
    parser.add_argument("--target-tokens", type=int, choices=[150, 300], default=300)
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.97)
    parser.add_argument("--disable-enforce-eager", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.08)
    parser.add_argument("--start-line", type=int, default=1)
    parser.add_argument("--end-line", type=int, default=None)
    parser.add_argument("--flush-every", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    refine_file_vllm(
        before_path=Path(args.before),
        after_path=Path(args.after),
        dst=Path(args.dst),
        model_name=args.model,
        target_tokens=args.target_tokens,
        batch_size=args.batch_size,
        dtype=args.dtype,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=not args.disable_enforce_eager,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        start_line=args.start_line,
        end_line=args.end_line,
        flush_every=args.flush_every,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
