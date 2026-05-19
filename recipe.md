## ICGR training recipe

After finish stage-1 and stage-2 training, you'll need to evaluate the model with WHOLE test dataset to collect as much as data for DPO. Also evaluate the stage-1 model with icl queries to get the reject samples for `dpo_h1success_cd_zs_unseen.jsonl`

1. `dpo.jsonl` & `dpo_h1success.jsonl`:

```bash
python src/construct_dpo_pairs.py \
      --inference_file retrieve_results/v4-5e-5-5epochs-test-full.json \
      --test_file      data/msmarco-icl-3shot-v4/test.jsonl \
      --output_dir     data/dpo-v4-copy-full
```

2. `dpo_h1success_cd_zs_unseen.jsonl`:

```bash
python src/build_hard_cd_dpo.py \
      --h1success_file    data/dpo-v4-copy-full/dpo_h1success.jsonl \
      --dpo_file          data/dpo-v4-copy-full/dpo.jsonl \
      --output_file       data/dpo-v4-copy-full/dpo_h1success_cd_zs_unseen.jsonl \
      --zero_shot_unseen  retrieve_results/<your-zero-shot-unseen-inference>.json \
      --strategy          zero_shot_unseen \
      --seed              42
```

Then, you can train DPO using `./configs/axolotl_dpo_msmarco_v4_zs_unseen.yml`

After that, use 100 shot dataset to extend long context with LoRA
