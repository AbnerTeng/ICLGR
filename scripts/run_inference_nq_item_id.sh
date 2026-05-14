#!/bin/bash
set -euo pipefail

# Inference for Abner0803/Qwen3-1.7B-nq-text-100k-with_pseudo_queries on
#   /mnt/raid0/home/lalako/ICLGR/data/nq-item-id/data/test.jsonl
#   /mnt/raid0/home/lalako/ICLGR/data/nq-item-id/data/icl_test_query_only.jsonl
#
# No HF corpus is used; the docid trie is built from local
# train.jsonl + icl_test.jsonl (operation=="indexing" rows, id_field=doc_id).
#
# Env overrides:
#   GPU=5            CUDA_VISIBLE_DEVICES (PCI order)
#   BATCH_SIZE=8
#   MAX_SAMPLES=-1   -1 means all; set e.g. 50 for a smoke test
#   TAG=base         appended to output filename
#   MODEL=...        override model id

cd /mnt/raid0/home/lalako/ICLGR

GPU=${GPU:-5}
MODEL=${MODEL:-Abner0803/Qwen3-1.7B-nq-text-100k-with_pseudo_queries}
MODEL_TAG=${MODEL_TAG:-$(basename "$MODEL")}
DATA_DIR="/mnt/raid0/home/lalako/ICLGR/data/nq-item-id/data"
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
        hf_corpus=null \
        +id_field=doc_id \
        batch_size="$BATCH_SIZE" \
        max_samples="$MAX_SAMPLES" \
        output_file="$OUT_DIR/$out_name"
}

run_one "$DATA_DIR/test.jsonl"                "${MODEL_TAG}_nq-item-id_test_${TAG}.json"
run_one "$DATA_DIR/icl_test_query_only.jsonl" "${MODEL_TAG}_nq-item-id_icl_test_${TAG}.json"

echo "========== All done =========="
