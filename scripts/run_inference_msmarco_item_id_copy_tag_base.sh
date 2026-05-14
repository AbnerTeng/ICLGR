#!/bin/bash
set -euo pipefail

# Inference for Abner0803/Qwen3-1.7B-icl-3shot-v4_128k-copy_tag on
#   /mnt/raid0/home/lalako/ICLGR/data/msmarco-item-id/test.jsonl
#   /mnt/raid0/home/lalako/ICLGR/data/msmarco-item-id/icl_test.jsonl
#
# These files are flat (text -> doc_id) with no ICL prompt, so processor_type=base.
# Trie is built from the HF corpus Lala8383/msmarco-item-id-lower (item-id docids).

cd /mnt/raid0/home/lalako/ICLGR

GPU=${GPU:-5}
MODEL="Abner0803/Qwen3-1.7B-icl-3shot-v4_128k-copy_tag"
DATA_DIR="/mnt/raid0/home/lalako/ICLGR/data/msmarco-item-id"
OUT_DIR="./retrieve_results"
BATCH_SIZE=${BATCH_SIZE:-8}

mkdir -p "$OUT_DIR"

run_one() {
    local test_file=$1
    local out_name=$2
    echo "========== Inference on $test_file =========="
    CUDA_VISIBLE_DEVICES="$GPU" python -m src.inference_icl_with_tag \
        model_path="$MODEL" \
        from_hf=true \
        train_file="$DATA_DIR/train.jsonl" \
        new_file="$DATA_DIR/icl_test.jsonl" \
        test_file="$test_file" \
        processor_type=base \
        hf_corpus="Lala8383/msmarco-item-id-lower" \
        batch_size="$BATCH_SIZE" \
        max_samples=-1 \
        output_file="$OUT_DIR/$out_name"
}

run_one "$DATA_DIR/test.jsonl"     "Qwen3-1.7B-icl-3shot-v4_128k-copy_tag_msmarco-item-id_test_base.json"
run_one "$DATA_DIR/icl_test.jsonl" "Qwen3-1.7B-icl-3shot-v4_128k-copy_tag_msmarco-item-id_icl_test_base.json"

echo "========== All done =========="
