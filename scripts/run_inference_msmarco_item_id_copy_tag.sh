#!/bin/bash
set -euo pipefail

# Inference for Abner0803/Qwen3-1.7B-icl-3shot-v4_128k-copy_tag on
#   /mnt/raid0/home/lalako/ICLGR/data/msmarco-item-id/test.jsonl
#   /mnt/raid0/home/lalako/ICLGR/data/msmarco-item-id/icl_test.jsonl
#
# These files are flat (text -> doc_id) with no ICL prompt, so processor_type=base.
# Trie is built from the HF corpus Lala8383/msmarco-item-id-lower (item-id docids).
#
# Env overrides:
#   GPU=5            CUDA_VISIBLE_DEVICES
#   BATCH_SIZE=8     inference batch size
#   MAX_SAMPLES=-1   -1 means all; set to 50 for a quick smoke test
#   TAG=base         appended to output filename

cd /mnt/raid0/home/lalako/ICLGR

GPU=${GPU:-5}
MODEL=${MODEL:-Abner0803/Qwen3-1.7B-icl-3shot-v4_128k-copy_tag}
MODEL_TAG=${MODEL_TAG:-$(basename "$MODEL")}
DATA_DIR="/mnt/raid0/home/lalako/ICLGR/data/msmarco-item-id-lower"
OUT_DIR="./retrieve_results"
BATCH_SIZE=${BATCH_SIZE:-8}
MAX_SAMPLES=${MAX_SAMPLES:--1}
TAG=${TAG:-base}

mkdir -p "$OUT_DIR"

run_one() {
    local test_file=$1
    local out_name=$2
    echo "========== Inference on $test_file (max_samples=$MAX_SAMPLES) =========="
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" python -m src.inference_icl_with_tag \
        model_path="$MODEL" \
        from_hf=true \
        train_file="$DATA_DIR/train.jsonl" \
        new_file="$DATA_DIR/icl_test.jsonl" \
        test_file="$test_file" \
        processor_type=base \
        hf_corpus="Lala8383/msmarco-item-id-lower" \
        batch_size="$BATCH_SIZE" \
        max_samples="$MAX_SAMPLES" \
        output_file="$OUT_DIR/$out_name"
}

run_one "$DATA_DIR/test.jsonl"                "${MODEL_TAG}_msmarco-item-id_test_${TAG}.json"
run_one "$DATA_DIR/icl_test_query_only.jsonl" "${MODEL_TAG}_msmarco-item-id_icl_test_${TAG}.json"

echo "========== All done =========="
