#!/bin/bash
# Run DPR multiset baselines for MS MARCO and NQ test/icl_test splits.
#
# Usage:
#   bash scripts/run_dpr_baselines.sh
#   CUDA_VISIBLE_DEVICES=2 bash scripts/run_dpr_baselines.sh
#   MAX_SAMPLES=1000 bash scripts/run_dpr_baselines.sh

set -euo pipefail

CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
RESULTS_DIR="${RESULTS_DIR:-retrieve_results}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
BATCH_SIZE="${BATCH_SIZE:-128}"
QUERY_BATCH_SIZE="${QUERY_BATCH_SIZE:-128}"
TOP_K="${TOP_K:-100}"
CUTOFFS="${CUTOFFS:-1 5 10 20 100}"
DPR_QUESTION_MODEL="${DPR_QUESTION_MODEL:-facebook/dpr-question_encoder-multiset-base}"
DPR_CONTEXT_MODEL="${DPR_CONTEXT_MODEL:-facebook/dpr-ctx_encoder-multiset-base}"

export CUDA_DEVICE_ORDER
export CUDA_VISIBLE_DEVICES

mkdir -p "$RESULTS_DIR"

COMMON_ARGS=(
    --method dpr
    --dpr_question_model "$DPR_QUESTION_MODEL"
    --dpr_context_model "$DPR_CONTEXT_MODEL"
    --top_k "$TOP_K"
    --cutoffs $CUTOFFS
    --no-include_doc_id_in_text
    --batch_size "$BATCH_SIZE"
    --query_batch_size "$QUERY_BATCH_SIZE"
    --device cuda
)

if [[ -n "$MAX_SAMPLES" ]]; then
    COMMON_ARGS+=(--max_samples "$MAX_SAMPLES")
    SAMPLE_TAG="_${MAX_SAMPLES}"
else
    SAMPLE_TAG=""
fi

run_dpr() {
    local dataset="$1"
    local split="$2"
    local train_file="$3"
    local test_file="$4"
    local cache_file="${RESULTS_DIR}/dpr_multiset_${dataset}_${split}_corpus_embeddings_no_title.pt"
    local output_file="${RESULTS_DIR}/dpr_multiset_${dataset}_${split}${SAMPLE_TAG}_predictions.jsonl"
    local metrics_file="${RESULTS_DIR}/dpr_multiset_${dataset}_${split}${SAMPLE_TAG}_metrics.json"

    echo "=== DPR dataset=${dataset} split=${split} cuda=${CUDA_VISIBLE_DEVICES} samples=${MAX_SAMPLES:-all} ==="
    "$PYTHON_BIN" -m src.retrieval_baselines \
        "${COMMON_ARGS[@]}" \
        --train_file "$train_file" \
        --test_file "$test_file" \
        --dpr_cache "$cache_file" \
        --output_file "$output_file" \
        --metrics_file "$metrics_file"
}

# run_dpr "msmarco_item_id" "test" \
#     "/mnt/raid0/home/lalako/ICLGR/data/msmarco-item-id/train.jsonl" \
#     "/mnt/raid0/home/lalako/ICLGR/data/msmarco-item-id/test.jsonl"

# run_dpr "msmarco_item_id" "icl_test" \
#     "/mnt/raid0/home/lalako/ICLGR/data/msmarco-item-id/train.jsonl" \
#     "/mnt/raid0/home/lalako/ICLGR/data/msmarco-item-id/icl_test.jsonl"

run_dpr "nq_item_id" "test" \
    "/mnt/raid0/home/lalako/ICLGR/data/nq-item-id/data/train.jsonl" \
    "/mnt/raid0/home/lalako/ICLGR/data/nq-item-id/data/test.jsonl"

run_dpr "nq_item_id" "icl_test" \
    "/mnt/raid0/home/lalako/ICLGR/data/nq-item-id/data/train.jsonl" \
    "/mnt/raid0/home/lalako/ICLGR/data/nq-item-id/data/icl_test.jsonl"

echo ""
echo "=== Done. DPR results written to ${RESULTS_DIR} ==="
