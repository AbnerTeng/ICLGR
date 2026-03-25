"""
Experiment pipeline: data generation -> training stages -> evaluation.

Usage:
    python -m src.pipeline
    python -m src.pipeline experiment_name=my_exp
    python -m src.pipeline train_stages.0.data_types=[mem_retrieval,mem_indexing]
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml
import hydra
from omegaconf import DictConfig, OmegaConf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_cmd(cmd: List[str], desc: str = "") -> None:
    label = desc or " ".join(cmd[:3])
    print(f"\n[RUN] {label}")
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {' '.join(cmd)}")


def find_latest_checkpoint(checkpoint_dir: str) -> str:
    """Return path to the highest-numbered checkpoint-* dir, or checkpoint_dir itself."""
    ckpts = sorted(
        [d for d in Path(checkpoint_dir).iterdir() if d.name.startswith("checkpoint-")],
        key=lambda x: int(x.name.split("-")[1]),
    )
    return str(ckpts[-1]) if ckpts else checkpoint_dir


def filter_jsonl_by_pattern(src: str, dst: str, patterns: List[str]) -> int:
    """Write only lines whose metadata.pattern is in patterns. Returns line count."""
    patterns_set = set(patterns)
    count = 0
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(src) as f_in, open(dst, "w") as f_out:
        for line in f_in:
            ex = json.loads(line)
            if ex.get("metadata", {}).get("pattern") in patterns_set:
                f_out.write(line)
                count += 1
    return count


def patch_axolotl_config(template: str, dst: str, patches: Dict) -> None:
    """Load axolotl yml template, apply key-value patches, save to dst."""
    with open(template) as f:
        cfg = yaml.safe_load(f)
    for k, v in patches.items():
        if v is not None:
            cfg[k] = v
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_generate_data(cfg: DictConfig, experiment_dir: str) -> Dict[str, str]:
    """Run data generation script; return mapping of split -> file path."""
    data_dir = os.path.join(experiment_dir, "data")
    dg = cfg.data_gen
    n_shot = cfg.get("n_shot", 3)

    cmd = [
        sys.executable, "-m", dg.script,
        f"output_dir={data_dir}",
        f"n_shot={n_shot}",
        "hydra.run.dir=.",          # prevent hydra from changing cwd
        "hydra/job_logging=disabled",
        "hydra/hydra_logging=disabled",
    ]
    for k, v in dg.get("overrides", {}).items():
        cmd.append(f"{k}={v}")

    run_cmd(cmd, desc=f"Data generation ({dg.script})")

    return {
        split: os.path.join(data_dir, f"{split}_{n_shot}shot.jsonl")
        for split in ["train", "test", "icl_test"]
    }


def step_train_stage(
    stage: DictConfig,
    data_paths: Dict[str, str],
    stage_dir: str,
    prev_checkpoint: Optional[str],
    experiment_name: str,
) -> str:
    """Filter data, patch axolotl config, run training. Returns latest checkpoint path."""
    os.makedirs(stage_dir, exist_ok=True)

    # Filter training data to the patterns for this stage
    stage_train_file = os.path.join(stage_dir, "train.jsonl")
    n = filter_jsonl_by_pattern(data_paths["train"], stage_train_file, list(stage.data_types))
    print(f"  Filtered {n} examples with patterns {list(stage.data_types)}")

    # Build dataset entry matching axolotl format
    dataset_entry = {
        "path": os.path.abspath(stage_train_file),
        "type": "chat_template",
        "chat_template": "tokenizer_default_fallback_chatml",
        "field_messages": "conversations",
        "message_property_mappings": {"role": "role", "content": "content"},
        "roles": {
            "assistant": ["assistant", "gpt", "model"],
            "user": ["user", "human"],
            "system": ["system"],
        },
    }

    checkpoint_dir = os.path.join(stage_dir, "checkpoint")
    patches = {
        "datasets": [dataset_entry],
        "output_dir": checkpoint_dir,
        "wandb_name": f"{experiment_name}-{stage.name}",
    }
    if prev_checkpoint:
        patches["base_model"] = prev_checkpoint
    elif stage.get("base_model"):
        patches["base_model"] = stage.base_model

    stage_config = os.path.join(stage_dir, "axolotl_config.yml")
    patch_axolotl_config(stage.axolotl_config_template, stage_config, patches)

    run_cmd(
        ["accelerate", "launch", "-m", "axolotl.cli.train", stage_config],
        desc=f"Training stage [{stage.name}]",
    )

    return find_latest_checkpoint(checkpoint_dir)


def step_evaluate(
    eval_cfg: DictConfig,
    model_path: str,
    data_paths: Dict[str, str],
    eval_dir: str,
) -> Dict[str, Dict]:
    """Run inference_icl on each eval target. Returns results dict."""
    os.makedirs(eval_dir, exist_ok=True)
    all_results = {}

    for target in eval_cfg.targets:
        # Resolve test file: prefer explicit path, else use generated split
        test_file = target.get("path") or data_paths.get(target.split_name, "")
        if not test_file or not os.path.exists(test_file):
            print(f"  [SKIP] {target.name}: test file not found ({test_file})")
            continue

        output_file = os.path.join(eval_dir, f"{target.name}_results.json")
        new_file = eval_cfg.get("new_file", "")

        cmd = [
            sys.executable, "-m", "src.inference_icl",
            f"model_path={os.path.abspath(model_path)}",
            "from_hf=false",
            f"train_file={eval_cfg.train_file}",
            f"new_file={new_file}",
            f"test_file={os.path.abspath(test_file)}",
            f"output_file={os.path.abspath(output_file)}",
            f"max_samples={eval_cfg.get('max_samples', -1)}",
            "hydra.run.dir=.",
            "hydra/job_logging=disabled",
            "hydra/hydra_logging=disabled",
        ]
        run_cmd(cmd, desc=f"Evaluation [{target.name}]")

        with open(output_file) as f:
            res = json.load(f)
        all_results[target.name] = {
            "hit_at_1":  res["hit_at_1"],
            "hit_at_10": res["hit_at_10"],
            "total":     res["total"],
        }
        print(f"  [{target.name}] Hit@1={res['hit_at_1']:.4f}  Hit@10={res['hit_at_10']:.4f}  (n={res['total']})")

    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="../configs", config_name="pipeline", version_base=None)
def main(cfg: DictConfig) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{cfg.experiment_name}_{timestamp}"
    exp_dir = os.path.join(cfg.experiment_base_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    OmegaConf.save(cfg, os.path.join(exp_dir, "pipeline_config.yaml"))

    print(f"\n{'='*60}")
    print(f"  Experiment : {exp_name}")
    print(f"  Directory  : {exp_dir}")
    print(f"{'='*60}")

    # Step 1: Generate data
    print("\n[STEP 1] Data Generation")
    data_paths = step_generate_data(cfg, exp_dir)
    for split, path in data_paths.items():
        exists = "OK" if os.path.exists(path) else "MISSING"
        print(f"  {split:10s} -> {path} [{exists}]")

    # Step 2: Training stages
    prev_checkpoint = None
    for i, stage in enumerate(cfg.train_stages, 1):
        print(f"\n[STEP 2.{i}] Training Stage: {stage.name}")
        stage_dir = os.path.join(exp_dir, stage.name)
        prev_checkpoint = step_train_stage(
            stage, data_paths, stage_dir, prev_checkpoint, exp_name
        )
        print(f"  Checkpoint: {prev_checkpoint}")

    # Step 3: Evaluation
    print("\n[STEP 3] Evaluation")
    eval_dir = os.path.join(exp_dir, "eval")
    eval_results = step_evaluate(cfg.eval, prev_checkpoint, data_paths, eval_dir)

    # Summary
    summary = {
        "experiment_name": exp_name,
        "experiment_dir":  exp_dir,
        "final_checkpoint": prev_checkpoint,
        "eval_results": eval_results,
    }
    summary_path = os.path.join(exp_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  Done! Summary: {summary_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
