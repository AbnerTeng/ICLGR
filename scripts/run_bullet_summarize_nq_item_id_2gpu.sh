#!/usr/bin/env bash
set -euo pipefail

# Summarize indexing rows in the NQ item-id dataset as bullet points.
#
# Usage:
#   bash scripts/run_bullet_summarize_nq_item_id_2gpu.sh
#
# Common overrides:
#   CUDA_VISIBLE_DEVICES=2,3 OUT_DIR=data/nq-item-id-llm-bullets/data \
#     bash scripts/run_bullet_summarize_nq_item_id_2gpu.sh
#   NUM_BULLETS=6 MAX_BULLET_TOKENS=28 BATCH_SIZE=8 \
#     bash scripts/run_bullet_summarize_nq_item_id_2gpu.sh

SRC_DIR=${SRC_DIR:-data/nq-item-id/data}
OUT_DIR=${OUT_DIR:-data/nq-item-id-llm-bullets/data}
LOG_DIR=${LOG_DIR:-logs}
PYTHON=${PYTHON:-python3}

BACKEND=${BACKEND:-vllm}
MODEL=${MODEL:-Qwen/Qwen2.5-14B-Instruct}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-2}

NUM_BULLETS=${NUM_BULLETS:-5}
MAX_BULLET_TOKENS=${MAX_BULLET_TOKENS:-32}
MAX_SOURCE_TOKENS=${MAX_SOURCE_TOKENS:-768}
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
MAX_INPUT_TOKENS=${MAX_INPUT_TOKENS:-2048}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.97}
DTYPE=${DTYPE:-bfloat16}
TEMPERATURE=${TEMPERATURE:-0.0}
TOP_P=${TOP_P:-1.0}
REPETITION_PENALTY=${REPETITION_PENALTY:-1.08}
FLUSH_EVERY=${FLUSH_EVERY:-1000}
FILES=${FILES:-train.jsonl icl_test.jsonl test.jsonl}

mkdir -p "$OUT_DIR" "$LOG_DIR"
export CUDA_VISIBLE_DEVICES

count_indexing_rows() {
  local src=$1
  "$PYTHON" -c '
import json
import sys

count = 0
with open(sys.argv[1], encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        row = json.loads(line)
        count += row.get("operation") == "indexing"
print(count)
' "$src"
}

run_one_file() {
  local name=$1
  local src="$SRC_DIR/$name"
  local dst="$OUT_DIR/$name"
  local log="$LOG_DIR/bullet_summarize_${name%.jsonl}.log"

  if [[ ! -f "$src" ]]; then
    echo "[miss] $src -- skipping"
    return
  fi

  local indexing_count
  indexing_count=$(count_indexing_rows "$src")
  if [[ "$indexing_count" == "0" ]]; then
    echo "[copy] $name has no indexing rows"
    cp "$src" "$dst"
    return
  fi

  echo "=== summarize $name indexing_rows=$indexing_count cuda=$CUDA_VISIBLE_DEVICES tp=$TENSOR_PARALLEL_SIZE ==="
  "$PYTHON" -m src.bullet_summarize_indexing_text \
    --backend "$BACKEND" \
    --src "$src" \
    --dst "$dst" \
    --num-bullets "$NUM_BULLETS" \
    --max-bullet-tokens "$MAX_BULLET_TOKENS" \
    --max-source-tokens "$MAX_SOURCE_TOKENS" \
    --model "$MODEL" \
    --batch-size "$BATCH_SIZE" \
    --dtype "$DTYPE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --max-input-tokens "$MAX_INPUT_TOKENS" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --repetition-penalty "$REPETITION_PENALTY" \
    --flush-every "$FLUSH_EVERY" \
    2>&1 | tee "$log"
}

for name in $FILES; do
  run_one_file "$name"
done

echo
echo "Done. Bullet-summary JSONL files are in $OUT_DIR"
