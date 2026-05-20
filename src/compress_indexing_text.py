"""Compress indexing rows in item-id JSONL datasets.

It only changes rows with ``operation == "indexing"`` and leaves query rows
untouched. Two backends are supported:

* ``extractive``: deterministic, dependency-free truncation/compression.
* ``vllm``: local vLLM abstractive compression with Qwen2.5-Instruct.

Example:
    CUDA_VISIBLE_DEVICES=0 python -m src.compress_indexing_text \
        --backend extractive \
        --src data/nq-item-id/data/train.jsonl \
        --dst data/nq-item-id-compressed/data/train.jsonl \
        --target-tokens 300

    python -m src.compress_indexing_text \
        --backend vllm \
        --model Qwen/Qwen2.5-14B-Instruct \
        --src data/nq-item-id/data/train.jsonl \
        --dst data/nq-item-id-llm-compressed_300/data/train.jsonl \
        --target-tokens 300 \
        --batch-size 4 \
        --limit 20

    python -m src.compress_indexing_text \
        --backend vllm \
        --model Qwen/Qwen2.5-14B-Instruct \
        --src data/nq-item-id/data/train.jsonl \
        --dst data/nq-item-id-llm-compressed-300/data/train.jsonl \
        --target-tokens 300 \
        --batch-size 4
        --limit 20
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Iterable, Optional

TAIL_MARKERS = (
    " References ",
    " External links ",
    " See also ",
    " Notes ",
    " Retrieved from ",
    " Categories :",
    " Hidden categories :",
    " Talk Contents About Wikipedia",
)

NOISE_PATTERNS = (
    r"\bJump to : navigation , search\b",
    r"\bContents \( hide \)",
    r"\b\( edit \)",
    r"\bJump up \^",
    r"\bLearn how and when to remove this template message\b",
)

BAD_SUMMARY_PATTERNS = (
    r"\bterms may apply\b",
    r"\bprivacy policy\b",
    r"\bwikipedia\s*[®]?\s+is a registered trademark\b",
    r"\bretrieved from\b",
    r"\bcategories\s*:",
    r"\bhidden categories\s*:",
    r"\btalk contents about wikipedia\b",
    r"\bexternal links\b",
    r"\bsee also\b",
    r"\breferences\b",
    r"\bfor other uses\b",
    r"\bnot to be confused with\b",
    r"\bhousemates name entered exited\b",
    r"\bchannel number digital number call letters\b",
    r"\bthis article needs\b",
    r"\bthis article is about\b",
    r"\bfull - power stations\b",
    r"\btv market city of license\b",
)

LEAD_VERBS = (
    " is ",
    " was ",
    " are ",
    " were ",
    " refers to ",
    " known as ",
    " consists of ",
)


def token_count(text: str) -> int:
    return len(text.split())


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    for pattern in NOISE_PATTERNS:
        text = re.sub(pattern, " ", text)
    return re.sub(r"\s+", " ", text).strip()


def trim_tail(text: str, min_tokens: int) -> str:
    """Drop common Wikipedia tail sections when enough content remains."""
    best_cut = len(text)
    for marker in TAIL_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            best_cut = min(best_cut, idx)

    if best_cut == len(text):
        return text

    candidate = text[:best_cut].strip()
    if token_count(candidate) >= min_tokens:
        return candidate
    return text


def strip_front_matter(text: str, doc_id: str) -> str:
    """Remove common navigation/template clutter while preserving the lead."""
    text = normalize_text(text)
    for pattern in (
        r"^.*?\bJump to : navigation , search\b",
        r"^This article .*?\( Learn how and when to remove this template message \)",
    ):
        candidate = re.sub(pattern, "", text, count=1).strip()
        if token_count(candidate) >= 50:
            text = candidate

    title = doc_id.strip()
    if title and not text.lower().startswith(title.lower()):
        title_idx = text.lower().find(title.lower())
        if 0 <= title_idx <= 400:
            text = text[title_idx:].strip()
    return text


def doc_terms(doc_id: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", doc_id.lower())
        if len(token) > 2
    ]


def flexible_title_pattern(doc_id: str) -> str:
    parts = [re.escape(token) for token in doc_id.split() if token]
    return r"\s+".join(parts)


def find_lead_offset(text: str, doc_id: str) -> int | None:
    title = flexible_title_pattern(doc_id)
    if not title:
        return None

    patterns = (
        rf"\b{title}\b\s*,\s*(?:officially|also known as|formerly)\b",
        rf"\b{title}\b.{{0,180}}\b(?:is|was|are|were|refers to|consists of)\b",
        rf"\bthe\s+{title}\b.{{0,180}}\b(?:is|was|are|were|refers to|consists of)\b",
        r"\bThis is a list of\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match and "this article" not in match.group(0).lower():
            return match.start()
    return None


def is_lead_candidate(chunk: str, doc_id: str) -> bool:
    lowered = f" {chunk.lower()} "
    if len(chunk.split()) < 8:
        return False
    if any(re.search(pattern, lowered) for pattern in BAD_SUMMARY_PATTERNS):
        return False
    if lowered.count(" edit ") > 1:
        return False
    if re.search(r"\b(name|entered|exited|channel|network|country|language|headquarters)\b", lowered):
        if not any(verb in lowered for verb in LEAD_VERBS):
            return False

    terms = doc_terms(doc_id)
    title_hit = any(term in lowered for term in terms[:3])
    definition_hit = any(verb in lowered for verb in LEAD_VERBS)
    list_hit = " this is a list " in lowered or " is a list of " in lowered
    return (title_hit and definition_hit) or list_hit


def lead_focused_text(text: str, doc_id: str) -> str:
    """Start from the first real lead sentence instead of an infobox/list."""
    text = strip_front_matter(text, doc_id=doc_id)
    offset = find_lead_offset(text, doc_id)
    if offset is not None:
        text = text[offset:].strip()

    chunks = sentence_like_chunks(text)
    if not chunks:
        return text

    for idx, chunk in enumerate(chunks):
        if is_lead_candidate(chunk, doc_id):
            return " ".join(chunks[idx:]).strip()
    return text


def sentence_like_chunks(text: str) -> list[str]:
    chunks = re.split(r"(?<=[.!?])\s+", text)
    return [c.strip() for c in chunks if c.strip()]


def clamp_tokens(tokens: list[str], max_tokens: int) -> str:
    return " ".join(tokens[:max_tokens]).strip()


def clamp_to_sentence_boundary(text: str, max_tokens: int) -> str:
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text.strip()

    clamped = clamp_tokens(tokens, max_tokens)
    sentence_end = max(clamped.rfind("."), clamped.rfind("!"), clamped.rfind("?"))
    if sentence_end > max(80, len(clamped) // 2):
        return clamped[: sentence_end + 1].strip()
    return clamped


def compress_text(text: str, doc_id: str, min_tokens: int, max_tokens: int) -> str:
    """Compress text into the target token range.

    The strategy is intentionally conservative for retrieval/indexing data:
    keep the title/doc_id signal, prefer the article lead, then extend with
    early sentence-like chunks until the minimum length is reached.
    """
    if max_tokens < min_tokens:
        raise ValueError("--max-tokens must be >= --min-tokens")

    text = lead_focused_text(
        trim_tail(normalize_text(text), min_tokens=min_tokens),
        doc_id=doc_id,
    )
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text

    prefix = doc_id.strip()
    lowered = text.lower()
    prefix_tokens = prefix.split()

    if prefix and not lowered.startswith(prefix.lower()):
        selected_tokens = prefix_tokens + [":"]
    else:
        selected_tokens = []

    chunks = sentence_like_chunks(text)
    for chunk in chunks:
        chunk_tokens = chunk.split()
        if not chunk_tokens:
            continue
        remaining = max_tokens - len(selected_tokens)
        if remaining <= 0:
            break
        if len(chunk_tokens) <= remaining:
            selected_tokens.extend(chunk_tokens)
        else:
            selected_tokens.extend(chunk_tokens[:remaining])
            break
        if len(selected_tokens) >= min_tokens:
            break

    if len(selected_tokens) < min_tokens:
        selected_tokens = tokens[:max_tokens]

    return clamp_tokens(selected_tokens, max_tokens)


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}") from exc


def durable_flush(fout) -> None:
    fout.flush()
    os.fsync(fout.fileno())


def maybe_durable_flush(fout, written_rows: int, flush_every: int) -> None:
    if flush_every > 0 and written_rows > 0 and written_rows % flush_every == 0:
        durable_flush(fout)


def compress_file(
    src: Path,
    dst: Path,
    min_tokens: int,
    max_tokens: int,
    start_line: int,
    end_line: Optional[int],
    flush_every: int,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    compressed = 0
    with (
        src.open("r", encoding="utf-8") as fin,
        dst.open("w", encoding="utf-8") as fout,
    ):
        for line_no, line in enumerate(fin, start=1):
            if line_no < start_line:
                continue
            if end_line is not None and line_no > end_line:
                break
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {src}:{line_no}") from exc

            total += 1
            if row.get("operation") == "indexing":
                before = token_count(row.get("text", ""))
                row["text"] = compress_text(
                    row.get("text", ""),
                    doc_id=row.get("doc_id", ""),
                    min_tokens=min_tokens,
                    max_tokens=max_tokens,
                )
                after = token_count(row["text"])
                if after < before:
                    compressed += 1

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            maybe_durable_flush(fout, total, flush_every)

        durable_flush(fout)

    print(f"Wrote {dst}")
    print(f"  rows={total} indexing_compressed={compressed}")


def target_min_tokens(target_tokens: int) -> int:
    if target_tokens not in {150, 300}:
        raise ValueError("--target-tokens must be either 150 or 300")
    if target_tokens == 300:
        return 220
    return int(target_tokens * 0.8)


def build_llm_prompt(text: str, doc_id: str, target_tokens: int) -> str:
    if target_tokens == 300:
        target_range = "220-300 tokens"
    else:
        min_tokens = target_min_tokens(target_tokens)
        target_range = f"{min_tokens}-{target_tokens} tokens"

    return (
        "You are compressing a noisy Wikipedia-style document for Natural Questions retrieval training.\n\n"
        "Write ONE retrieval-oriented summary.\n\n"
        f"Target length: {target_range}.\n\n"
        "Rules:\n"
        "- Output ONLY the summary.\n"
        "- Do NOT copy the source text verbatim.\n"
        "- Do NOT continue the original document.\n"
        "- Do NOT include headings, table of contents, navigation text, references, external links, infobox fields, or maintenance notices.\n"
        "- If the document is primarily a list or table, summarize what the list contains, its categories, and representative important entities. Do not copy rows.\n"
        "- Preserve key entities, dates, definitions, aliases, relationships, major events, and facts likely to answer questions.\n"
        "- Use complete sentences.\n"
        "- Stop after the summary.\n\n"
        f"Title: {doc_id}\n\n"
        "SOURCE DOCUMENT:\n"
        "<<<\n"
        f"{text}\n"
        ">>>\n\n"
        "SUMMARY:"
    )


def clean_generation(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:text)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    prefixes = (
        "Compressed document:",
        "Compressed text:",
        "Summary:",
        "Answer:",
    )
    for prefix in prefixes:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix) :].strip()
    return normalize_text(text)


def remove_source_echo(summary: str, source_text: str, doc_id: str) -> str:
    summary = clean_generation(summary)
    source_text = normalize_text(source_text)
    lowered = summary.lower()

    markers = [
        "jump to : navigation",
        "jump to: navigation",
        "contents ( hide )",
        "contents",
        "references ( edit )",
        "references",
        "external links",
        "see also",
    ]

    title = doc_id.strip()
    if title:
        markers.extend(
            [
                f"{title} ",
                f"{title} jump to",
                f"{title} {title}",
            ]
        )

    source_prefixes = [
        " ".join(normalize_text(source_text).split()[:50]).strip(),
        " ".join(strip_front_matter(source_text, doc_id=doc_id).split()[:50]).strip(),
        " ".join(lead_focused_text(source_text, doc_id=doc_id).split()[:50]).strip(),
    ]
    markers.extend(prefix for prefix in source_prefixes if prefix)

    best_pos: int | None = None
    for marker in markers:
        marker = marker.strip()
        if not marker:
            continue
        pos = lowered.find(marker.lower())
        if pos > 80 and (best_pos is None or pos < best_pos):
            best_pos = pos

    if best_pos is not None:
        summary = summary[:best_pos].strip()
    return summary


def is_bad_summary(text: str, doc_id: str, min_tokens: int) -> bool:
    text = clean_generation(text)
    if token_count(text) < max(50, int(min_tokens * 0.35)):
        return True

    lowered = text.lower()
    if any(re.search(pattern, lowered) for pattern in BAD_SUMMARY_PATTERNS):
        return True

    title = doc_id.strip().lower()
    if title and lowered.count(title) >= 2:
        return True

    if text.strip().lower().endswith((" the", " of", " for", " and", " to", " in")):
        return True

    doc_tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", doc_id.lower())
        if len(token) > 2
    ]
    if doc_tokens and not any(token in lowered for token in doc_tokens[:3]):
        return True
    return False


def enforce_length(
    text: str, source_text: str, doc_id: str, min_tokens: int, max_tokens: int
) -> str:
    """Keep generated text within max_tokens and provide a fallback if too short."""
    text = remove_source_echo(text, source_text=source_text, doc_id=doc_id)
    n_tokens = token_count(text)
    if min_tokens <= n_tokens <= max_tokens and not is_bad_summary(
        text, doc_id=doc_id, min_tokens=min_tokens
    ):
        return text
    if n_tokens > max_tokens:
        clamped = clamp_to_sentence_boundary(text, max_tokens)
        if not is_bad_summary(clamped, doc_id=doc_id, min_tokens=min_tokens):
            return clamped
    if text and not is_bad_summary(text, doc_id=doc_id, min_tokens=min_tokens):
        return text

    fallback = compress_text(
        source_text, doc_id=doc_id, min_tokens=min_tokens, max_tokens=max_tokens
    )
    if not text or is_bad_summary(text, doc_id=doc_id, min_tokens=min_tokens):
        return fallback
    merged = f"{text} {fallback}"
    return clamp_tokens(merged.split(), max_tokens)


class VllmCompressor:
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
                "content": "You compress documents for retrieval/indexing datasets.",
            },
            {"role": "user", "content": prompt},
        ]
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt + "\n\nCompressed document:"

    def _truncate_document(self, text: str) -> str:
        # Keep room for instructions and the chat template so the assistant
        # generation marker is never truncated away.
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

    def compress_batch(
        self,
        rows: list[dict],
        target_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
    ) -> list[str]:
        from vllm import SamplingParams

        prompts = [
            self._truncate_prompt(
                self._format_prompt(
                    build_llm_prompt(
                        self._truncate_document(
                            lead_focused_text(
                                trim_tail(
                                    normalize_text(row.get("text", "")),
                                    min_tokens=target_min_tokens(target_tokens),
                                ),
                                doc_id=row.get("doc_id", ""),
                            )
                        ),
                        doc_id=row.get("doc_id", ""),
                        target_tokens=target_tokens,
                    )
                )
            )
            for row in rows
        ]
        outputs = self.llm.generate(
            prompts,
            SamplingParams(
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_tokens=self.max_new_tokens,
            ),
        )
        min_tokens = target_min_tokens(target_tokens)
        decoded = [output.outputs[0].text for output in outputs]
        return [
            enforce_length(
                text,
                source_text=row.get("text", ""),
                doc_id=row.get("doc_id", ""),
                min_tokens=min_tokens,
                max_tokens=target_tokens,
            )
            for text, row in zip(decoded, rows)
        ]


def flush_vllm_batch(
    compressor: VllmCompressor,
    batch: list[tuple[int, dict]],
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

    rows = [row for _, row in batch]
    summaries = compressor.compress_batch(
        rows,
        target_tokens=target_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    for (idx, row), summary in zip(batch, summaries):
        row["text"] = summary
        buffered_rows[idx] = row

    while next_to_write in buffered_rows:
        fout.write(
            json.dumps(buffered_rows.pop(next_to_write), ensure_ascii=False) + "\n"
        )
        next_to_write += 1
        written_rows += 1
        maybe_durable_flush(fout, written_rows, flush_every)
    return next_to_write, written_rows


def compress_file_vllm(
    src: Path,
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
    compressor = VllmCompressor(
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
        for line_no, line in enumerate(tqdm(fin, desc="Compressing indexing rows"), start=1):
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
            row = json.loads(line)
            total += 1
            if row.get("operation") != "indexing":
                buffered_rows[idx] = row
            else:
                indexing += 1
                batch.append((idx, row))
                if len(batch) >= batch_size:
                    next_to_write, written_rows = flush_vllm_batch(
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
                fout.write(
                    json.dumps(buffered_rows.pop(next_to_write), ensure_ascii=False)
                    + "\n"
                )
                next_to_write += 1
                written_rows += 1
                maybe_durable_flush(fout, written_rows, flush_every)

        next_to_write, written_rows = flush_vllm_batch(
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
            fout.write(
                json.dumps(buffered_rows.pop(next_to_write), ensure_ascii=False) + "\n"
            )
            next_to_write += 1
            written_rows += 1
            maybe_durable_flush(fout, written_rows, flush_every)

        durable_flush(fout)

    print(f"Wrote {dst}")
    print(f"  rows={total} llm_compressed_indexing={indexing}")
    elapsed = time.perf_counter() - start
    if elapsed > 0:
        print(f"  elapsed_sec={elapsed:.2f}")
        print(f"  indexing_rows_per_sec={indexing / elapsed:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["extractive", "vllm"], default="vllm")
    parser.add_argument("--src", default="data/nq-item-id/data/train.jsonl")
    parser.add_argument("--dst", default="data/nq-item-id-compressed/data/train.jsonl")
    parser.add_argument("--target-tokens", type=int, choices=[150, 300], default=300)
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="bfloat16",
    )
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

    if args.backend == "vllm":
        compress_file_vllm(
            src=Path(args.src),
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
    else:
        compress_file(
            src=Path(args.src),
            dst=Path(args.dst),
            min_tokens=target_min_tokens(args.target_tokens),
            max_tokens=args.target_tokens,
            start_line=args.start_line,
            end_line=args.end_line,
            flush_every=args.flush_every,
        )


if __name__ == "__main__":
    main()
