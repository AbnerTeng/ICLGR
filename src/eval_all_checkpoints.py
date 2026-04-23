"""Evaluate base model + all checkpoints using cli_docquery in parallel across GPUs."""

import argparse
import os
import subprocess
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


def run_eval(model_path, gpu_id, max_samples, log_path, config_name,
             pattern="", no_trie=False):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = [
        "python", "-m", "src.cli_docquery",
        "--config-name", config_name,
        f"model_path={model_path}",
        f"max_samples={max_samples}",
    ]
    if no_trie:
        cmd.append("no_trie=true")
    if pattern:
        cmd.append(f"pattern='{pattern}'")
    label = os.path.basename(model_path) or model_path
    print(f"[GPU {gpu_id}] Starting {label} ...")
    with open(log_path, "w") as log_f:
        proc = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
    status = "OK" if proc.returncode == 0 else f"FAILED (exit={proc.returncode})"
    print(f"[GPU {gpu_id}] Done {label} -> {status}")
    return log_path


def parse_results(log_path):
    results = {"file": os.path.basename(log_path)}
    with open(log_path, "r") as f:
        content = f.read()
    h1 = re.search(r"Final\s+Hit@1:\s+([\d.]+)", content)
    h10 = re.search(r"Final\s+Hit@10:\s+([\d.]+)", content)
    if h1:
        results["Hit@1"] = float(h1.group(1))
    if h10:
        results["Hit@10"] = float(h10.group(1))
    for m in re.finditer(r"^\s+(\S+)\s+([\d.]+)\s+([\d.]+)", content, re.MULTILINE):
        pat, p_h1, p_h10 = m.group(1), m.group(2), m.group(3)
        try:
            results[f"{pat}/H@1"] = float(p_h1)
            results[f"{pat}/H@10"] = float(p_h10)
        except ValueError:
            pass
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="")
    parser.add_argument("--config_name", type=str, default="cli_docquery_conf")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--pattern", type=str, default="")
    parser.add_argument("--no_trie", action="store_true", help="Disable trie-constrained decoding")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    eval_jobs = []
    if args.base_model:
        eval_jobs.append(("base_model", args.base_model))
    ckpt_dirs = sorted(
        [d for d in Path(args.ckpt_dir).iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-")[-1]),
    )
    for ckpt in ckpt_dirs:
        eval_jobs.append((ckpt.name, str(ckpt)))

    if not eval_jobs:
        print("No models to evaluate.")
        return

    names = [name for name, _ in eval_jobs]
    print(f"Models to evaluate ({len(eval_jobs)}): {names}")
    print(f"Config: {args.config_name}, GPUs: {args.gpus}, max_samples: {args.max_samples}\n")

    num_gpus = len(args.gpus)
    log_paths = []

    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = {}
        for i, (name, model_path) in enumerate(eval_jobs):
            gpu = args.gpus[i % num_gpus]
            log_path = os.path.join(args.log_dir, f"eval_{name}.log")
            log_paths.append(log_path)
            futures[executor.submit(
                run_eval, model_path, gpu, args.max_samples, log_path,
                args.config_name, args.pattern, args.no_trie,
            )] = name
        for future in as_completed(futures):
            future.result()

    print(f"\n{'='*120}")
    print("Summary")
    print(f"{'='*120}")
    print(f"{'Model':<20} {'Hit@1':>8} {'Hit@10':>8} | {'zs_ret/H@1':>10} {'zs_idx/H@1':>10} {'front/H@1':>10} {'back/H@1':>10} {'noise/H@1':>10}")
    print("-" * 120)
    for (name, _), log_path in zip(eval_jobs, log_paths):
        r = parse_results(log_path)
        print(
            f"  {name:<18} {r.get('Hit@1', -1):>8.4f} {r.get('Hit@10', -1):>8.4f} | "
            f"{r.get('zero_shot_retrieval/H@1', -1):>10.4f} "
            f"{r.get('zero_shot_indexing/H@1', -1):>10.4f} "
            f"{r.get('doc_pos_front/H@1', -1):>10.4f} "
            f"{r.get('doc_pos_back/H@1', -1):>10.4f} "
            f"{r.get('all_noise/H@1', -1):>10.4f}"
        )
    print(f"{'='*120}")


if __name__ == "__main__":
    main()
