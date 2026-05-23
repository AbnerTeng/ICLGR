# Position Robustness Analysis

Evaluates how a model's retrieval performance (Hit@1) changes depending on where the golden document appears in the 100-shot context (positions 1, 25, 50, 75, 100).

## Step 1 — Download the position-variant dataset

The test data is hosted at `Abner0803/msmarco-icl-100shot-v4-pos_variants`.
Download it with the HF CLI:

```bash
huggingface-cli download Abner0803/msmarco-icl-100shot-v4-pos_variants \
    --repo-type dataset \
    --local-dir data/msmarco-icl-100shot-v4-pos_variants
```

This gives you 10 JSONL files (5 positions × 2 splits):

```
data/msmarco-icl-100shot-v4-pos_variants/
    test_pos001.jsonl      icl_test_pos001.jsonl
    test_pos025.jsonl      icl_test_pos025.jsonl
    test_pos050.jsonl      icl_test_pos050.jsonl
    test_pos075.jsonl      icl_test_pos075.jsonl
    test_pos100.jsonl      icl_test_pos100.jsonl
```

## Step 2 — Run inference

Use `scripts/run_position_robustness.sh`. At minimum, set `MODEL_PATH` and `TAG`:

> You can set TAG to whatever you want (model name)

```bash
# HuggingFace model
MODEL_PATH=Abner0803/Qwen3-1.7B-icl-3shot-dpo-irr_doc \
TAG=dpo-irr-doc \
MAX_SAMPLES=500 \
GPU=0 \
bash scripts/run_position_robustness.sh

# Local checkpoint
MODEL_PATH=./checkpoint/Qwen3-1.7B-icl-100shot-v4-5e-5-5epochs \
FROM_HF=false \
TAG=sft-100shot \
MAX_SAMPLES=500 \
GPU=0 \
bash scripts/run_position_robustness.sh
```

Results are written to `retrieve_results/position_robustness/` and logs to `logs/position_robustness/`.

### Key environment variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `Abner0803/Qwen3-1.7B-icl-3shot-dpo-irr_doc` | HF repo ID or local path |
| `FROM_HF` | `true` | Set `false` for local checkpoints |
| `TAG` | `dpo-irr-doc-pos` | Label used in output filenames and plots |
| `GPU` | `0` | CUDA device index |
| `BATCH_SIZE` | `1` | Keep at 1 for 100-shot (long contexts cause OOM at higher values) |
| `MAX_SAMPLES` | `-1` (all) | Number of samples per file; use `500` for a quick run |
| `POSITIONS` | `1 25 50 75 100` | 1-indexed golden doc positions to evaluate |
| `SPLITS` | `test icl_test` | Which splits to run |

### Evaluating multiple models at once

Edit the `MODELS` array in `scripts/run_position_robustness_multi.sh` and run:

```bash
MAX_SAMPLES=500 GPU=0 bash scripts/run_position_robustness_multi.sh
```

## Step 3 — Analyze results

```bash
python -m src.analyze_position_robustness \
    --results_dir retrieve_results/position_robustness \
    --output      nshot_results/position_robustness.png
```

This prints a Hit@1 table per model and saves a comparison plot to `nshot_results/position_robustness.png`.
