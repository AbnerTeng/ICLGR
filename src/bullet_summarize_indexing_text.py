"""Rewrite indexing rows as bullet point summaries.

Only rows with ``operation == "indexing"`` are changed; query rows are copied
unchanged. Two backends are supported:

* ``extractive``: deterministic, dependency-free bullet extraction.
* ``vllm``: local vLLM abstractive bullet summaries.

Examples:
    python -m src.bullet_summarize_indexing_text \
        --backend extractive \
        --src data/nq-item-id/data/train.jsonl \
        --dst data/nq-item-id-bullets/data/train.jsonl \
        --num-bullets 5 \
        --max-bullet-tokens 32

    CUDA_VISIBLE_DEVICES=0 python -m src.bullet_summarize_indexing_text \
        --backend vllm \
        --model Qwen/Qwen2.5-14B-Instruct \
        --src data/nq-item-id/data/train.jsonl \
        --dst data/nq-item-id-llm-bullets/data/train.jsonl \
        --num-bullets 5 \
        --batch-size 4 \
        --limit 20
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Optional

from src.compress_indexing_text import (
    clamp_tokens,
    durable_flush,
    lead_focused_text,
    maybe_durable_flush,
    normalize_text,
    sentence_like_chunks,
    token_count,
    trim_tail,
)


BAD_BULLET_PATTERNS = (
    r"\bjump to\s*:?\s*navigation\b",
    r"\bcontents\s*\(\s*hide\s*\)",
    r"\breferences\b",
    r"\bexternal links\b",
    r"\bsee also\b",
    r"\bretrieved from\b",
    r"\bcategories\s*:",
    r"\bhidden categories\s*:",
    r"\bthis article needs\b",
    r"\blearn how and when to remove this template message\b",
)


def clean_bullet(text: str, max_bullet_tokens: int) -> str:
    text = normalize_text(text)
    text = re.sub(r"^[\-*•\d.)\s]+", "", text).strip()
    text = clamp_tokens(text.split(), max_bullet_tokens)
    return text.rstrip(" ;,")


def is_good_bullet(text: str, doc_id: str) -> bool:
    if token_count(text) < 4:
        return False
    lowered = text.lower()
    if any(re.search(pattern, lowered) for pattern in BAD_BULLET_PATTERNS):
        return False
    title_terms = [
        token
        for token in re.findall(r"[a-z0-9]+", doc_id.lower())
        if len(token) > 2
    ]
    if title_terms and len(text.split()) < 8:
        return any(term in lowered for term in title_terms[:3])
    return True


def format_bullets(bullets: list[str]) -> str:
    return "\n".join(f"- {bullet}" for bullet in bullets if bullet.strip())


def extractive_bullet_summary(
    text: str,
    doc_id: str,
    num_bullets: int,
    max_bullet_tokens: int,
    max_source_tokens: int,
) -> str:
    """Build a compact bullet summary from early high-signal sentences."""
    text = lead_focused_text(
        trim_tail(normalize_text(text), min_tokens=80),
        doc_id=doc_id,
    )
    text = clamp_tokens(text.split(), max_source_tokens)

    bullets: list[str] = []
    seen: set[str] = set()
    for chunk in sentence_like_chunks(text):
        bullet = clean_bullet(chunk, max_bullet_tokens=max_bullet_tokens)
        key = re.sub(r"\W+", " ", bullet.lower()).strip()
        if not key or key in seen:
            continue
        if not is_good_bullet(bullet, doc_id=doc_id):
            continue
        bullets.append(bullet)
        seen.add(key)
        if len(bullets) >= num_bullets:
            break

    if not bullets:
        fallback = clean_bullet(text, max_bullet_tokens=max_bullet_tokens)
        if fallback:
            bullets.append(fallback)

    return format_bullets(bullets)


def build_llm_prompt(text: str, doc_id: str, num_bullets: int, max_bullet_tokens: int) -> str:
    return (
        "You are compressing a noisy Wikipedia-style document for retrieval training.\n\n"
        f"Write exactly {num_bullets} bullet points.\n\n"
        "Rules:\n"
        "- Output ONLY bullet points, one per line, each starting with '- '.\n"
        f"- Each bullet must be at most {max_bullet_tokens} words.\n"
        "- Do NOT copy long spans from the source text.\n"
        "- Do NOT include headings, table of contents, navigation text, references, external links, infobox fields, or maintenance notices.\n"
        "- Preserve key entities, dates, definitions, aliases, relationships, major events, and facts likely to answer questions.\n"
        "- If the document is a list or table, summarize what the list contains and representative important entities instead of copying rows.\n\n"
        f"Title: {doc_id}\n\n"
        "SOURCE DOCUMENT:\n"
        "<<<\n"
        f"{text}\n"
        ">>>\n\n"
        "BULLETS:"
    )


def clean_generated_bullets(
    text: str,
    source_text: str,
    doc_id: str,
    num_bullets: int,
    max_bullet_tokens: int,
) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:text)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    text = re.sub(r"^(?:BULLETS|SUMMARY)\s*:\s*", "", text, flags=re.IGNORECASE)

    bullets: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        bullet = clean_bullet(line, max_bullet_tokens=max_bullet_tokens)
        key = re.sub(r"\W+", " ", bullet.lower()).strip()
        if not key or key in seen:
            continue
        if is_good_bullet(bullet, doc_id=doc_id):
            bullets.append(bullet)
            seen.add(key)
        if len(bullets) >= num_bullets:
            break

    if len(bullets) < max(1, min(2, num_bullets)):
        fallback = extractive_bullet_summary(
            source_text,
            doc_id=doc_id,
            num_bullets=num_bullets,
            max_bullet_tokens=max_bullet_tokens,
            max_source_tokens=512,
        )
        return fallback

    return format_bullets(bullets)


class VllmBulletSummarizer:
    def __init__(
        self,
        model_name: str,
        dtype: str,
        max_input_tokens: int,
        max_new_tokens: int,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        enforce_eager: bool,
    ) -> None:
        from vllm import LLM

        self.llm = LLM(
            model=model_name,
            dtype=dtype,
            max_model_len=max_input_tokens + max_new_tokens,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            trust_remote_code=True,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens

    def _format_prompt(self, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "You write concise bullet summaries for retrieval/indexing datasets.",
            },
            {"role": "user", "content": prompt},
        ]
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt + "\n\n- "

    def _truncate_document(self, text: str) -> str:
        max_document_tokens = max(256, self.max_input_tokens - 512)
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= max_document_tokens:
            return text
        token_ids = token_ids[:max_document_tokens]
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def _truncate_prompt(self, prompt: str) -> str:
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(token_ids) <= self.max_input_tokens:
            return prompt
        token_ids = token_ids[-self.max_input_tokens :]
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def summarize_batch(
        self,
        rows: list[dict],
        num_bullets: int,
        max_bullet_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
    ) -> list[str]:
        from vllm import SamplingParams

        prompts = []
        for row in rows:
            source = lead_focused_text(
                trim_tail(normalize_text(row.get("text", "")), min_tokens=80),
                doc_id=row.get("doc_id", ""),
            )
            prompts.append(
                self._truncate_prompt(
                    self._format_prompt(
                        build_llm_prompt(
                            self._truncate_document(source),
                            doc_id=row.get("doc_id", ""),
                            num_bullets=num_bullets,
                            max_bullet_tokens=max_bullet_tokens,
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
            clean_generated_bullets(
                generated,
                source_text=row.get("text", ""),
                doc_id=row.get("doc_id", ""),
                num_bullets=num_bullets,
                max_bullet_tokens=max_bullet_tokens,
            )
            for generated, row in zip(decoded, rows)
        ]


def summarize_file_extractive(
    src: Path,
    dst: Path,
    num_bullets: int,
    max_bullet_tokens: int,
    max_source_tokens: int,
    start_line: int,
    end_line: Optional[int],
    flush_every: int,
    limit: Optional[int],
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    summarized = 0
    with (
        src.open("r", encoding="utf-8") as fin,
        dst.open("w", encoding="utf-8") as fout,
    ):
        for line_no, line in enumerate(fin, start=1):
            if line_no < start_line:
                continue
            if end_line is not None and line_no > end_line:
                break
            if limit is not None and total >= limit:
                break
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {src}:{line_no}") from exc

            total += 1
            if row.get("operation") == "indexing":
                row["text"] = extractive_bullet_summary(
                    row.get("text", ""),
                    doc_id=row.get("doc_id", ""),
                    num_bullets=num_bullets,
                    max_bullet_tokens=max_bullet_tokens,
                    max_source_tokens=max_source_tokens,
                )
                summarized += 1

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            maybe_durable_flush(fout, total, flush_every)

        durable_flush(fout)

    print(f"Wrote {dst}")
    print(f"  rows={total} indexing_summarized={summarized}")


def flush_vllm_batch(
    summarizer: VllmBulletSummarizer,
    batch: list[tuple[int, dict]],
    buffered_rows: dict[int, dict],
    fout,
    next_to_write: int,
    written_rows: int,
    flush_every: int,
    num_bullets: int,
    max_bullet_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> tuple[int, int]:
    if not batch:
        return next_to_write, written_rows

    rows = [row for _, row in batch]
    summaries = summarizer.summarize_batch(
        rows,
        num_bullets=num_bullets,
        max_bullet_tokens=max_bullet_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    for (idx, row), summary in zip(batch, summaries):
        row["text"] = summary
        buffered_rows[idx] = row

    while next_to_write in buffered_rows:
        fout.write(json.dumps(buffered_rows.pop(next_to_write), ensure_ascii=False) + "\n")
        next_to_write += 1
        written_rows += 1
        maybe_durable_flush(fout, written_rows, flush_every)
    return next_to_write, written_rows


def summarize_file_vllm(
    src: Path,
    dst: Path,
    model_name: str,
    num_bullets: int,
    max_bullet_tokens: int,
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
    summarizer = VllmBulletSummarizer(
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
    batch: list[tuple[int, dict]] = []
    buffered_rows: dict[int, dict] = {}
    next_to_write = 0
    written_rows = 0

    with (
        src.open("r", encoding="utf-8") as fin,
        dst.open("w", encoding="utf-8") as fout,
    ):
        local_idx = 0
        for line_no, line in enumerate(tqdm(fin, desc="Summarizing indexing rows"), start=1):
            if line_no < start_line:
                continue
            if end_line is not None and line_no > end_line:
                break
            if limit is not None and total >= limit:
                break
            if not line.strip():
                continue

            idx = local_idx
            local_idx += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {src}:{line_no}") from exc

            total += 1
            if row.get("operation") != "indexing":
                buffered_rows[idx] = row
            else:
                indexing += 1
                batch.append((idx, row))
                if len(batch) >= batch_size:
                    next_to_write, written_rows = flush_vllm_batch(
                        summarizer,
                        batch,
                        buffered_rows,
                        fout,
                        next_to_write,
                        written_rows,
                        flush_every,
                        num_bullets,
                        max_bullet_tokens,
                        temperature,
                        top_p,
                        repetition_penalty,
                    )
                    batch = []

            while next_to_write in buffered_rows:
                fout.write(
                    json.dumps(buffered_rows.pop(next_to_write), ensure_ascii=False) + "\n"
                )
                next_to_write += 1
                written_rows += 1
                maybe_durable_flush(fout, written_rows, flush_every)

        next_to_write, written_rows = flush_vllm_batch(
            summarizer,
            batch,
            buffered_rows,
            fout,
            next_to_write,
            written_rows,
            flush_every,
            num_bullets,
            max_bullet_tokens,
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
    print(f"  rows={total} llm_summarized_indexing={indexing}")
    elapsed = time.perf_counter() - start
    if elapsed > 0:
        print(f"  elapsed_sec={elapsed:.2f}")
        print(f"  indexing_rows_per_sec={indexing / elapsed:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["extractive", "vllm"], default="vllm")
    parser.add_argument("--src", default="data/nq-item-id/data/train.jsonl")
    parser.add_argument("--dst", default="data/nq-item-id-bullets/data/train.jsonl")
    parser.add_argument("--num-bullets", type=int, default=5)
    parser.add_argument("--max-bullet-tokens", type=int, default=32)
    parser.add_argument("--max-source-tokens", type=int, default=768)
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=256)
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

    if args.num_bullets < 1:
        raise ValueError("--num-bullets must be >= 1")
    if args.max_bullet_tokens < 4:
        raise ValueError("--max-bullet-tokens must be >= 4")

    if args.backend == "vllm":
        summarize_file_vllm(
            src=Path(args.src),
            dst=Path(args.dst),
            model_name=args.model,
            num_bullets=args.num_bullets,
            max_bullet_tokens=args.max_bullet_tokens,
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
    else:
        summarize_file_extractive(
            src=Path(args.src),
            dst=Path(args.dst),
            num_bullets=args.num_bullets,
            max_bullet_tokens=args.max_bullet_tokens,
            max_source_tokens=args.max_source_tokens,
            start_line=args.start_line,
            end_line=args.end_line,
            flush_every=args.flush_every,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
