#!/bin/bash
# Generate all eval scripts for a given experiment.
# Usage: bash gen_eval_scripts.sh <exp_name> <ckpt_dir> <config_test> <config_icl_test>
#
# Example:
#   bash gen_eval_scripts.sh balanced \
#     ./checkpoint/Qwen3-1.7B-icl-3shot-v3-balanced-1e-4-8epochs \
#     cli_docquery_balanced \
#     cli_docquery_balanced_icl_test

EXP=$1
CKPT_DIR=$2
CFG_TEST=$3
CFG_ICL_TEST=$4
BASE_MODEL="Abner0803/Qwen3-1.7B-msmarco-text-100k-with_pseudo_queries"
GPUS="0 1 2 3 4 5 6 7"
MAX_SAMPLES=500

for SPLIT in test icl_test; do
  if [ "$SPLIT" = "test" ]; then
    CFG=$CFG_TEST
  else
    CFG=$CFG_ICL_TEST
  fi

  for TRIE in trie no_trie; do
    FNAME="run_eval_${SPLIT}_${EXP}_${TRIE}.sh"
    LOG_DIR="logs/${EXP}_${SPLIT}_${TRIE}"

    if [ "$TRIE" = "no_trie" ]; then
      TRIE_FLAG="    --no_trie"
    else
      TRIE_FLAG=""
    fi

    cat > "$FNAME" <<SCRIPT
#!/bin/bash

CKPT_DIR="${CKPT_DIR}"
BASE_MODEL="${BASE_MODEL}"
GPUS="${GPUS}"
MAX_SAMPLES=${MAX_SAMPLES}

echo "========== ${EXP} ${SPLIT} (${TRIE}) =========="
python -m src.eval_all_checkpoints \\
    --ckpt_dir "\$CKPT_DIR" \\
    --base_model "\$BASE_MODEL" \\
    --config_name ${CFG} \\
    --gpus \$GPUS \\
    --max_samples \$MAX_SAMPLES \\
    --log_dir ${LOG_DIR}${TRIE_FLAG:+ \\
${TRIE_FLAG}}

echo "========== Done! =========="
SCRIPT

    echo "Created: $FNAME -> $LOG_DIR"
  done
done

# Also generate stage1 eval scripts
for SPLIT in test icl_test; do
  if [ "$SPLIT" = "test" ]; then
    CFG="cli_docquery_stage1"
  else
    CFG="cli_docquery_stage1_icl_test"
  fi

  for TRIE in trie no_trie; do
    FNAME="run_eval_${SPLIT}_${EXP}_stage1_${TRIE}.sh"
    LOG_DIR="logs/${EXP}_stage1_${SPLIT}_${TRIE}"

    if [ "$TRIE" = "no_trie" ]; then
      TRIE_FLAG="    --no_trie"
    else
      TRIE_FLAG=""
    fi

    cat > "$FNAME" <<SCRIPT
#!/bin/bash

CKPT_DIR="${CKPT_DIR}"
BASE_MODEL="${BASE_MODEL}"
GPUS="${GPUS}"
MAX_SAMPLES=${MAX_SAMPLES}

echo "========== ${EXP} stage1 ${SPLIT} (${TRIE}) =========="
python -m src.eval_all_checkpoints \\
    --ckpt_dir "\$CKPT_DIR" \\
    --base_model "\$BASE_MODEL" \\
    --config_name ${CFG} \\
    --gpus \$GPUS \\
    --max_samples \$MAX_SAMPLES \\
    --log_dir ${LOG_DIR}${TRIE_FLAG:+ \\
${TRIE_FLAG}}

echo "========== Done! =========="
SCRIPT

    echo "Created: $FNAME -> $LOG_DIR"
  done
done
