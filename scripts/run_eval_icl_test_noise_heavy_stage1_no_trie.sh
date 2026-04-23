#!/bin/bash

CKPT_DIR="./checkpoint/Qwen3-1.7B-icl-3shot-v3-noise-heavy-1e-4-8epochs"
BASE_MODEL="Abner0803/Qwen3-1.7B-msmarco-text-100k-with_pseudo_queries"
GPUS="0 1 2 3 4 5 6 7"
MAX_SAMPLES=500

echo "========== noise_heavy stage1 icl_test (no_trie) =========="
python -m src.eval_all_checkpoints \
    --ckpt_dir "$CKPT_DIR" \
    --base_model "$BASE_MODEL" \
    --config_name cli_docquery_stage1_icl_test \
    --gpus $GPUS \
    --max_samples $MAX_SAMPLES \
    --log_dir logs/noise_heavy_stage1_icl_test_no_trie \
    --no_trie

echo "========== Done! =========="
