# count_tokens_qwen3_hf.py
import json
import argparse
import random
from tqdm import tqdm
from transformers import AutoTokenizer


def _extract_text(
    item,
    data_format: str,
    text_field: str,
    message_scope: str,
) -> str | None:
    if data_format == "text":
        return item.get(text_field)

    if data_format != "conversations":
        raise ValueError(f"Unsupported format: {data_format}")

    conversations = item.get("conversations")
    if not isinstance(conversations, list):
        return None

    pieces = []
    for message in conversations:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if content is None:
            continue
        if message_scope == "all" or role == message_scope:
            pieces.append(str(content))
    return "\n".join(pieces) if pieces else None


def count_tokens_jsonl(
    input_path: str,
    tokenizer_name: str,
    text_field: str = "text",
    operation: str = "indexing",
    sample_size: int = 10000,
    seed: int = 42,
    add_special_tokens: bool = False,
    data_format: str = "text",
    message_scope: str = "all",
):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=True,
    )

    rng = random.Random(seed)
    sampled_texts = []
    eligible_examples = 0
    total_tokens = 0
    total_examples = 0
    skipped = 0
    skipped_operation = 0

    max_tokens = 0
    min_tokens = None

    token_counts = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Sampling examples"):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            if operation is not None and item.get("operation") != operation:
                skipped_operation += 1
                continue

            text = _extract_text(
                item,
                data_format=data_format,
                text_field=text_field,
                message_scope=message_scope,
            )
            if text is None:
                skipped += 1
                continue

            eligible_examples += 1
            if sample_size <= 0 or len(sampled_texts) < sample_size:
                sampled_texts.append(text)
            else:
                replace_idx = rng.randrange(eligible_examples)
                if replace_idx < sample_size:
                    sampled_texts[replace_idx] = text

    for text in tqdm(sampled_texts, desc="Counting tokens"):
        token_ids = tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
        )

        n_tokens = len(token_ids)

        token_counts.append(n_tokens)
        total_tokens += n_tokens
        total_examples += 1

        max_tokens = max(max_tokens, n_tokens)
        min_tokens = n_tokens if min_tokens is None else min(min_tokens, n_tokens)

    avg_tokens = total_tokens / total_examples if total_examples > 0 else 0

    token_counts_sorted = sorted(token_counts)

    def percentile(p):
        if not token_counts_sorted:
            return 0
        idx = int(len(token_counts_sorted) * p / 100)
        idx = min(idx, len(token_counts_sorted) - 1)
        return token_counts_sorted[idx]

    print("===== Token Statistics =====")
    print(f"Tokenizer: {tokenizer_name}")
    print(f"Input file: {input_path}")
    print(f"Format: {data_format}")
    print(f"Text field: {text_field}")
    print(f"Message scope: {message_scope}")
    print(f"Operation filter: {operation}")
    print(f"Eligible examples: {eligible_examples}")
    print(f"Sample size: {total_examples}")
    print(f"Random seed: {seed}")
    print(f"Add special tokens: {add_special_tokens}")
    print()
    print(f"Examples: {total_examples}")
    print(f"Skipped: {skipped}")
    print(f"Skipped by operation: {skipped_operation}")
    print(f"Total tokens: {total_tokens}")
    print(f"Average tokens/example: {avg_tokens:.2f}")
    print(f"Min tokens/example: {min_tokens}")
    print(f"Max tokens/example: {max_tokens}")
    print()
    print(f"P50 tokens/example: {percentile(50)}")
    print(f"P90 tokens/example: {percentile(90)}")
    print(f"P95 tokens/example: {percentile(95)}")
    print(f"P99 tokens/example: {percentile(99)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to jsonl dataset",
    )

    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="Abner0803/Qwen3-1.7B-icl-3shot-v4_128k-copy_tag",
    )

    parser.add_argument(
        "--text_field",
        type=str,
        default="text",
    )

    parser.add_argument(
        "--operation",
        type=str,
        default="indexing",
        help="Only sample rows whose operation matches this value. Use '' to disable.",
    )

    parser.add_argument(
        "--sample_size",
        type=int,
        default=10000,
        help="Number of matching rows to randomly sample before counting tokens. Use 0 to count all matching rows.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )

    parser.add_argument(
        "--add_special_tokens",
        action="store_true",
        help="Whether to include special tokens",
    )

    parser.add_argument(
        "--format",
        choices=["text", "conversations"],
        default="text",
        help="Use 'conversations' for Axolotl-style conversation JSONL.",
    )

    parser.add_argument(
        "--message_scope",
        choices=["user", "assistant", "all"],
        default="all",
        help="Which conversation messages to count when --format conversations.",
    )

    args = parser.parse_args()
    operation = args.operation if args.operation != "" else None

    count_tokens_jsonl(
        input_path=args.input_path,
        tokenizer_name=args.tokenizer_name,
        text_field=args.text_field,
        operation=operation,
        sample_size=args.sample_size,
        seed=args.seed,
        add_special_tokens=args.add_special_tokens,
        data_format=args.format,
        message_scope=args.message_scope,
    )
