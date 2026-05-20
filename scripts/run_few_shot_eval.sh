#!/bin/bash
# Build few-shot test/icl_test datasets and run inference for each shot.
#
# Usage:
#   bash scripts/run_few_shot_eval.sh
#   CUDA_VISIBLE_DEVICES=4 bash scripts/run_few_shot_eval.sh
#   SHOTS="10 100" bash scripts/run_few_shot_eval.sh           # subset
#   MAX_SAMPLES=50 BATCH_SIZE=1 bash scripts/run_few_shot_eval.sh

set -euo pipefail

# ── config ─────────────────────────────────────────────────────────────────
# SHOTS="${SHOTS:-100 80 60 50 40 30 20 10 5 2 1}"
SHOTS="${SHOTS:-60 50 40 30}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
DATA_DIR="${DATA_DIR:-./data/msmarco-item-id-3shot-v4_128k}"
RESULTS_DIR="${RESULTS_DIR:-./retrieve_results}"
TAG="${TAG:-v4_128k-copy_tag}"

BATCH_SIZE="${BATCH_SIZE:-1}"

# Sample-count tag for output filename: "all" when MAX_SAMPLES=-1, else the number.
if [[ "$MAX_SAMPLES" == "-1" ]]; then
    SAMPLES_TAG="all"
else
    SAMPLES_TAG="${MAX_SAMPLES}"
fi

mkdir -p "$RESULTS_DIR"
export CUDA_VISIBLE_DEVICES

# ── step 1: build datasets ─────────────────────────────────────────────────
for N in $SHOTS; do
    TEST="${DATA_DIR}/test_${N}shot.jsonl"
    ICL="${DATA_DIR}/icl_test_${N}shot.jsonl"
    if [[ -f "$TEST" && -f "$ICL" ]]; then
        echo "[skip-build] n_shot=${N} (both files exist)"
        continue
    fi
    echo "=== build n_shot=${N} ==="
    python -m src.build_test_docquery_v4 n_shot="${N}"
done

# ── step 2: inference for each (n_shot, split) ─────────────────────────────
for N in $SHOTS; do
    for SPLIT in test icl_test; do
        TEST_FILE="${DATA_DIR}/${SPLIT}_${N}shot.jsonl"
        OUT_FILE="${RESULTS_DIR}/${TAG}-${SPLIT}_${N}shot-${SAMPLES_TAG}.json"
        if [[ ! -f "$TEST_FILE" ]]; then
            echo "[miss] $TEST_FILE  -- skipping"
            continue
        fi
        echo "=== infer split=${SPLIT}  n_shot=${N}  bs=${BATCH_SIZE}  samples=${SAMPLES_TAG}  cuda=${CUDA_VISIBLE_DEVICES} ==="
        python -m src.inference_icl_with_tag \
            test_file="${TEST_FILE}" \
            output_file="${OUT_FILE}" \
            batch_size="${BATCH_SIZE}" \
            max_samples="${MAX_SAMPLES}"
    done
done

echo ""
echo "=== Done. Results in ${RESULTS_DIR}/${TAG}-{test,icl_test}_Nshot-${SAMPLES_TAG}.json ==="

# CUDA_VISIBLE_DEVICES=0 MAX_SAMPLES=500 bash scripts/run_few_shot_eval.sh 2>&1 | tee /tmp/few_shot_eval_10.log
