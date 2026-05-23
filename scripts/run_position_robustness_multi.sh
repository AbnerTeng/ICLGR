#!/bin/bash
# Evaluate position robustness across multiple models in one run.
# Delegates each model to run_position_robustness.sh via env-var overrides.
#
# Edit the MODELS associative array below to add or remove models, then:
#
#   bash scripts/run_position_robustness_multi.sh
#
# Results land in RESULTS_DIR (default: retrieve_results/position_robustness/)
# and can be compared with src/analyze_position_robustness.py.

set -euo pipefail

# --- models to evaluate: TAG -> MODEL_PATH ---
declare -A MODELS=(
    ["dpo-irr-doc-pos"]="Abner0803/Qwen3-1.7B-icl-3shot-dpo-irr_doc"
    ["sft-100shot"]="./checkpoint/Qwen3-1.7B-icl-100shot-v4-5e-5-5epochs"
    ["pos-template-sft"]="./checkpoint/Qwen3-1.7B-icl-3shot-position-template-v4-5e-5-5epochs"
    ["pos-template-dpo"]="./checkpoint/Qwen3-1.7B-icl-3shot-position-template-v4-dpo"
)

# --- shared settings (override via env as usual) ---
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
POSITIONS="${POSITIONS:-1 25 50 75 100}"
SPLITS="${SPLITS:-test icl_test}"
RESULTS_DIR="${RESULTS_DIR:-./retrieve_results/position_robustness}"
LOG_DIR="${LOG_DIR:-./logs/position_robustness}"

export GPU BATCH_SIZE MAX_SAMPLES POSITIONS SPLITS RESULTS_DIR LOG_DIR

echo "[multi] evaluating ${#MODELS[@]} models"
echo "[multi] gpu=${GPU}  batch=${BATCH_SIZE}  samples=${MAX_SAMPLES}"
echo "[multi] started at $(date)"

for TAG in "${!MODELS[@]}"; do
    MODEL_PATH="${MODELS[$TAG]}"
    # local checkpoints don't need HF download
    if [[ "$MODEL_PATH" == ./* || "$MODEL_PATH" == /* ]]; then
        FROM_HF=false
    else
        FROM_HF=true
    fi

    echo ""
    echo "========================================"
    echo "[multi] tag=${TAG}"
    echo "[multi] model=${MODEL_PATH}"
    echo "========================================"

    TAG="$TAG" MODEL_PATH="$MODEL_PATH" FROM_HF="$FROM_HF" \
        bash scripts/run_position_robustness.sh
done

echo ""
echo "[multi] all models done at $(date)"
echo ""
echo "To analyze results:"
echo "  .venv/bin/python -m src.analyze_position_robustness \\"
echo "      --results_dir ${RESULTS_DIR} \\"
echo "      --output nshot_results/position_robustness.png"
