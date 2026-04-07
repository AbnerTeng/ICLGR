#!/bin/bash
set -e

export AXOLOTL_DO_NOT_TRACK=1

VERSION=${1:-"all"}  # "v3", "v4", or "all"

# ── Step 1: Generate data ────────────────────────────────────────────────
if [[ "$VERSION" == "v3" || "$VERSION" == "all" ]]; then
    echo "=== Generating v3 data ==="
    python -m src.to_axolotl_docquery_position_template_v3
fi

if [[ "$VERSION" == "v4" || "$VERSION" == "all" ]]; then
    echo "=== Generating v4 data ==="
    python -m src.to_axolotl_docquery_position_template_v4
fi

# ── Step 2: Train with 8 GPUs ───────────────────────────────────────────
if [[ "$VERSION" == "v3" || "$VERSION" == "all" ]]; then
    echo "=== Training v3 ==="
    accelerate launch --num_processes 8 -m axolotl.cli.train configs/axolotl_icl_msmarco-100k_v3.yml
fi

if [[ "$VERSION" == "v4" || "$VERSION" == "all" ]]; then
    echo "=== Training v4 ==="
    accelerate launch --num_processes 8 -m axolotl.cli.train configs/axolotl_icl_msmarco-100k_v4.yml
fi

echo "=== Done ==="
