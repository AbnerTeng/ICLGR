#!/usr/bin/env bash
set -euo pipefail

SRC=${SRC:-data/nq-item-id/data/train.jsonl}
OUT_DIR=${OUT_DIR:-data/nq-item-id-llm-compressed-shards/data}
LOG_DIR=${LOG_DIR:-logs}
TARGET_TOKENS=${TARGET_TOKENS:-300}
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
MAX_INPUT_TOKENS=${MAX_INPUT_TOKENS:-2048}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.97}
TEMPERATURE=${TEMPERATURE:-0.0}
TOP_P=${TOP_P:-1.0}
REPETITION_PENALTY=${REPETITION_PENALTY:-1.08}
FLUSH_EVERY=${FLUSH_EVERY:-1000}
MODEL=${MODEL:-Qwen/Qwen2.5-14B-Instruct}

mkdir -p "$OUT_DIR" "$LOG_DIR"

run_shard() {
  local shard_id=$1
  local gpu_id=$2
  local start_line=$3
  local end_line=${4:-}
  local dst="$OUT_DIR/train.shard${shard_id}.jsonl"
  local log="$LOG_DIR/compress_shard${shard_id}.log"

  local -a cmd=(
    python3 -m src.compress_indexing_text
    --backend vllm
    --src "$SRC"
    --dst "$dst"
    --target-tokens "$TARGET_TOKENS"
    --model "$MODEL"
    --batch-size "$BATCH_SIZE"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --max-input-tokens "$MAX_INPUT_TOKENS"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --temperature "$TEMPERATURE"
    --top-p "$TOP_P"
    --repetition-penalty "$REPETITION_PENALTY"
    --flush-every "$FLUSH_EVERY"
    --start-line "$start_line"
  )

  if [[ -n "$end_line" ]]; then
    cmd+=(--end-line "$end_line")
  fi

  echo "Starting shard ${shard_id} on GPU ${gpu_id}: lines ${start_line}-${end_line:-end}"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu_id" nohup "${cmd[@]}" > "$log" 2>&1 &
  echo "  pid=$! log=$log dst=$dst"
}

run_shard 0 0 1 29984
run_shard 1 1 29985 59968
run_shard 2 3 59969

echo
echo "All shard jobs started."
echo "Monitor with:"
echo "  tail -f $LOG_DIR/compress_shard0.log"
echo "  tail -f $LOG_DIR/compress_shard1.log"
echo "  tail -f $LOG_DIR/compress_shard2.log"
echo
echo "After all shards finish, merge with:"
echo "  mkdir -p data/nq-item-id-llm-compressed-final/data"
echo "  cat $OUT_DIR/train.shard0.jsonl $OUT_DIR/train.shard1.jsonl $OUT_DIR/train.shard2.jsonl > data/nq-item-id-llm-compressed-final/data/train.jsonl"
