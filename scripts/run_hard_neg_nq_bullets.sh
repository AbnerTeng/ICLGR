#!/bin/bash
# Mine hard negatives for nq-item-id-llm-bullets, then build few-shot train data.
#
# Usage:
#   bash scripts/run_hard_neg_nq_bullets.sh
#   SHOTS="3 10" bash scripts/run_hard_neg_nq_bullets.sh
#   CUDA_VISIBLE_DEVICES=3 bash scripts/run_hard_neg_nq_bullets.sh

set -euo pipefail

# Pin CUDA device enumeration to PCI bus order so "cuda:2" == PCI bus 2.
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

# ── config ─────────────────────────────────────────────────────────────────
SRC_DIR="${SRC_DIR:-./data/nq-item-id-llm-bullets/data}"
DST_DIR="${DST_DIR:-./data/nq-item-id-llm-bullets-hardneg}"
K="${K:-100}"
DEVICE="${DEVICE:-cuda:2}"
SHOTS="${SHOTS:-3 10 20 50 100}"
FILES="${FILES:-train.jsonl test.jsonl icl_test.jsonl}"

# ── step 1: hard negative mining ───────────────────────────────────────────
if [[ -f "${DST_DIR}/train.jsonl" ]]; then
    echo "[skip-mine] ${DST_DIR}/train.jsonl already exists"
else
    echo "=== mine hard negatives  src=${SRC_DIR}  dst=${DST_DIR}  k=${K}  device=${DEVICE} ==="
    python -m src.build_hard_negatives \
        --src_dir "${SRC_DIR}" \
        --dst_dir "${DST_DIR}" \
        --k "${K}" \
        --device "${DEVICE}" \
        --files ${FILES}
fi

# ── step 2: build few-shot train data per shot ─────────────────────────────
for N in ${SHOTS}; do
    OUT_DIR="./data/nq-item-id-llm-bullets-hardneg-${N}shot-v4"
    if [[ -f "${OUT_DIR}/train.jsonl" ]]; then
        echo "[skip-build] n_shot=${N} (${OUT_DIR}/train.jsonl exists)"
        continue
    fi
    CFG="build_train_docquery_v4_nq_bullets_${N}shot_hardneg"
    echo "=== build few-shot train  n_shot=${N}  config=${CFG} ==="
    python -m src.build_train_docquery_v4 --config-name "${CFG}"
done

echo ""
echo "=== Done. Outputs:"
echo "  hardneg:   ${DST_DIR}"
for N in ${SHOTS}; do
    echo "  ${N}shot:    ./data/nq-item-id-llm-bullets-hardneg-${N}shot-v4"
done
