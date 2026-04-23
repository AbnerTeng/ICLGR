#!/bin/bash
set -e

# ============================================================================
# Quick iteration pipeline: datagen → train (2ep) → inference (trie) → summary
#
# Usage:
#   bash run_experiment.sh <exp_name> <datagen_config>
#
# Examples:
#   bash run_experiment.sh balanced get_docquery_position_template_v3
#   bash run_experiment.sh noise_heavy get_docquery_position_template_v3_noise_heavy
# ============================================================================

EXP_NAME="${1:?Usage: bash run_experiment.sh <exp_name> <datagen_config>}"
DATAGEN_CONFIG="${2:?Usage: bash run_experiment.sh <exp_name> <datagen_config>}"

export AXOLOTL_DO_NOT_TRACK=1

if [ -f "wandb.txt" ]; then
    export WANDB_API_KEY=$(cat wandb.txt)
    echo "WANDB_API_KEY loaded from wandb.txt"
fi

uv pip install hydra-core --quiet

BASE_MODEL="Abner0803/Qwen3-1.7B-msmarco-text-100k-with_pseudo_queries"
GPUS="0 1 2 3 4 5 6 7"
MAX_SAMPLES=200
REMOTE_DATA_ROOT="/semantic-search-pvc/users/lala/ICLGR_local"

CKPT_DIR="./checkpoint/Qwen3-1.7B-icl-3shot-${EXP_NAME}-quick"
TRAIN_CONFIG="/tmp/axolotl_${EXP_NAME}_quick.yml"

# --- Read output_dir from datagen config ---
DATA_DIR=$(python3 -c "
import yaml
with open('configs/${DATAGEN_CONFIG}.yaml') as f:
    cfg = yaml.safe_load(f)
print(cfg['output_dir'])
")
echo "============================================================"
echo "  Experiment: ${EXP_NAME}"
echo "  Datagen config: ${DATAGEN_CONFIG}"
echo "  Data dir: ${DATA_DIR}"
echo "  Checkpoint dir: ${CKPT_DIR}"
echo "============================================================"

# ============================
# Stage 1: Data Generation
# ============================
echo ""
echo "========== [1/4] Data Generation =========="
python -m src.to_axolotl_docquery_position_template_v3 \
    --config-name "${DATAGEN_CONFIG}"

# ============================
# Stage 2: Training (2 epochs)
# ============================
echo ""
echo "========== [2/4] Training (2 epochs) =========="

cat > "${TRAIN_CONFIG}" <<EOF
base_model: ${BASE_MODEL}

datasets:
  - path: ${REMOTE_DATA_ROOT}/${DATA_DIR#./}/train_3shot.jsonl
    type: chat_template
    chat_template: tokenizer_default_fallback_chatml
    field_messages: conversations
    message_property_mappings:
      role: role
      content: content
    roles:
      assistant:
        - assistant
        - gpt
        - model
      user:
        - user
        - human
      system:
        - system

roles_to_train: ["assistant"]
train_on_eos: "turn"

shuffle_merged_datasets: true
output_dir: ${CKPT_DIR}
sequence_len: 2048
sample_packing: true
pad_to_sequence_len: true
flash_attention: true
xformers_attention: false
flex_attention: false
sdp_attention: false

gradient_accumulation_steps: 8
micro_batch_size: 8
dataloader_num_workers: 6
num_epochs: 2

optimizer: adamw_torch
lr_scheduler: cosine
learning_rate: 1e-4
warmup_ratio: 0.1
weight_decay: 0.0
bf16: true
tf32: false
gradient_checkpointing: true

logging_steps: 50
save_strategy: epoch
save_total_limit: 2

wandb_project: "ICLGR"
wandb_entity: "lala83"
wandb_watch: "gradients"
wandb_name: "quick-${EXP_NAME}-2ep"
wandb_log_model: "checkpoint"

val_set_size: 0.0
EOF

echo "Training config: ${TRAIN_CONFIG}"
accelerate launch -m axolotl.cli.train "${TRAIN_CONFIG}"

# ============================
# Stage 3: Inference (trie only, 200 samples)
# ============================
echo ""
echo "========== [3/4] Inference =========="

# --- Generate temp inference configs ---
INFER_TEST="/tmp/cli_dq_${EXP_NAME}_test.yaml"
INFER_ICL="/tmp/cli_dq_${EXP_NAME}_icl_test.yaml"

cat > "${INFER_TEST}" <<EOF
model_path: "placeholder"
train_file: "${DATA_DIR}/train_3shot.jsonl"
new_file: "${DATA_DIR}/icl_test_3shot.jsonl"
num_beams: 10
num_return: 10
max_new_tokens: 50
no_trie: false
mode: "file"
query_file: "${DATA_DIR}/test_3shot.jsonl"
max_samples: ${MAX_SAMPLES}
EOF

cat > "${INFER_ICL}" <<EOF
model_path: "placeholder"
train_file: "${DATA_DIR}/train_3shot.jsonl"
new_file: "${DATA_DIR}/icl_test_3shot.jsonl"
num_beams: 10
num_return: 10
max_new_tokens: 50
no_trie: false
mode: "file"
query_file: "${DATA_DIR}/icl_test_3shot.jsonl"
max_samples: ${MAX_SAMPLES}
EOF

# Copy temp configs to configs/ so Hydra can find them
cp "${INFER_TEST}" "configs/cli_dq_${EXP_NAME}_test.yaml"
cp "${INFER_ICL}" "configs/cli_dq_${EXP_NAME}_icl_test.yaml"

# --- Run 4 eval groups sequentially ---
echo "[3a] Self data - test (trie)"
python -m src.eval_all_checkpoints \
    --ckpt_dir "${CKPT_DIR}" \
    --base_model "${BASE_MODEL}" \
    --config_name "cli_dq_${EXP_NAME}_test" \
    --gpus ${GPUS} \
    --max_samples ${MAX_SAMPLES} \
    --log_dir "logs/${EXP_NAME}_quick_test_trie"

echo "[3b] Self data - icl_test (trie)"
python -m src.eval_all_checkpoints \
    --ckpt_dir "${CKPT_DIR}" \
    --base_model "${BASE_MODEL}" \
    --config_name "cli_dq_${EXP_NAME}_icl_test" \
    --gpus ${GPUS} \
    --max_samples ${MAX_SAMPLES} \
    --log_dir "logs/${EXP_NAME}_quick_icl_test_trie"

echo "[3c] Stage1 data - test (trie)"
python -m src.eval_all_checkpoints \
    --ckpt_dir "${CKPT_DIR}" \
    --base_model "${BASE_MODEL}" \
    --config_name cli_docquery_stage1 \
    --gpus ${GPUS} \
    --max_samples ${MAX_SAMPLES} \
    --log_dir "logs/${EXP_NAME}_quick_stage1_test_trie"

echo "[3d] Stage1 data - icl_test (trie)"
python -m src.eval_all_checkpoints \
    --ckpt_dir "${CKPT_DIR}" \
    --base_model "${BASE_MODEL}" \
    --config_name cli_docquery_stage1_icl_test \
    --gpus ${GPUS} \
    --max_samples ${MAX_SAMPLES} \
    --log_dir "logs/${EXP_NAME}_quick_stage1_icl_test_trie"

# ============================
# Stage 4: Summary
# ============================
echo ""
echo "========== [4/4] Summary =========="
python -m src.summarize_logs --log_dir \
    "logs/${EXP_NAME}_quick_test_trie" \
    "logs/${EXP_NAME}_quick_icl_test_trie" \
    "logs/${EXP_NAME}_quick_stage1_test_trie" \
    "logs/${EXP_NAME}_quick_stage1_icl_test_trie"

# Cleanup temp configs
rm -f "${TRAIN_CONFIG}" "${INFER_TEST}" "${INFER_ICL}"

echo ""
echo "============================================================"
echo "  Experiment '${EXP_NAME}' complete!"
echo "============================================================"
