#!/bin/bash
# Evaluate position robustness: runs inference for each (position, split) pair
# sequentially on a single GPU, then all results land in RESULTS_DIR for
# analysis with src/analyze_position_robustness.py.
#
# Usage:
#   bash scripts/run_position_robustness.sh
#
# Key overrides:
#   GPU=1 MODEL_PATH=./checkpoint/my-model TAG=my-model FROM_HF=false \
#     bash scripts/run_position_robustness.sh

set -euo pipefail

# --- configurable via environment ---
MODEL_PATH="${MODEL_PATH:-Abner0803/Qwen3-1.7B-icl-3shot-dpo-irr_doc}"
FROM_HF="${FROM_HF:-true}"
TAG="${TAG:-dpo-irr-doc-pos}"
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"          # keep at 1 for 100-shot (long contexts)
MAX_SAMPLES="${MAX_SAMPLES:--1}"       # -1 = full dataset
PROCESSOR_TYPE="${PROCESSOR_TYPE:-copy}"
INFERENCE_MODULE="${INFERENCE_MODULE:-src.inference_icl_with_tag}"
PYTHON="${PYTHON:-.venv/bin/python}"

DATA_DIR="${DATA_DIR:-./data/msmarco-icl-100shot-v4-pos_variants}"
RESULTS_DIR="${RESULTS_DIR:-./retrieve_results/position_robustness}"
LOG_DIR="${LOG_DIR:-./logs/position_robustness}"

POSITIONS="${POSITIONS:-1 25 50 75 100}"
SPLITS="${SPLITS:-test icl_test}"
# ------------------------------------

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SAMPLES_TAG="$( [[ "$MAX_SAMPLES" == "-1" ]] && echo "all" || echo "$MAX_SAMPLES" )"

echo "[pos-robust] model=${MODEL_PATH}  tag=${TAG}"
echo "[pos-robust] gpu=${GPU}  batch=${BATCH_SIZE}  samples=${SAMPLES_TAG}"
echo "[pos-robust] positions=[${POSITIONS}]  splits=[${SPLITS}]"
echo "[pos-robust] started at $(date)"

for POS in $POSITIONS; do
    POS_PAD=$(printf "%03d" "$POS")
    for SPLIT in $SPLITS; do
        TEST_FILE="${DATA_DIR}/${SPLIT}_pos${POS_PAD}.jsonl"
        OUT_FILE="${RESULTS_DIR}/${TAG}-${SPLIT}-pos${POS_PAD}-${SAMPLES_TAG}.json"
        LOG="${LOG_DIR}/${TAG}-${SPLIT}-pos${POS_PAD}.log"

        if [[ ! -f "$TEST_FILE" ]]; then
            echo "[miss] $TEST_FILE -- skip"
            continue
        fi
        if [[ -f "$OUT_FILE" ]]; then
            echo "[skip] $OUT_FILE already exists"
            continue
        fi

        echo "[run] pos=${POS}  split=${SPLIT}"
        "$PYTHON" -m "${INFERENCE_MODULE}" \
            model_path="${MODEL_PATH}" \
            from_hf="${FROM_HF}" \
            test_file="${TEST_FILE}" \
            output_file="${OUT_FILE}" \
            batch_size="${BATCH_SIZE}" \
            max_samples="${MAX_SAMPLES}" \
            processor_type="${PROCESSOR_TYPE}" \
            > "$LOG" 2>&1
        echo "[done] pos=${POS}  split=${SPLIT}"
    done
done

echo ""
echo "[pos-robust] all done at $(date)"
echo "Results in ${RESULTS_DIR}/"
