# Repository Guidelines

## Project Structure & Module Organization

This repository implements in-context learning for generative retrieval. Core Python modules live in `src/`, including data preparation, Axolotl conversion, inference, constrained decoding utilities, and evaluation. Hydra YAML configs live in `configs/`. Reusable shell workflows live in `scripts/`. Tests live in `tests/`. Large local artifacts are kept in `data/`, `checkpoint/`, `outputs/`, `logs/`, and `retrieve_results/`; avoid committing generated data, model checkpoints, or run outputs.

## Build, Test, and Development Commands

- `uv sync --dev`: create or update the Python 3.11 environment from `pyproject.toml` and `uv.lock`.
- `uv run pytest`: run the test suite.
- `uv run ruff check src tests`: run lint checks.
- `uv run python -m src.build_test_docquery_v4 n_shot=10`: generate a docquery test split using Hydra overrides.
- `uv run python -m src.inference_icl_with_tag`: run the primary tagged ICL inference entrypoint with `configs/inference_conf.yaml`.
- `bash scripts/run_few_shot_eval.sh`: run few-shot evaluation; override with `SHOTS`, `MAX_SAMPLES`, `BATCH_SIZE`, or `CUDA_VISIBLE_DEVICES`.

## Coding Style & Naming Conventions

Use Python 3.11 syntax and keep modules importable with `python -m src.<module>`. Follow the existing style: 4-space indentation, type hints when practical, and lowercase snake_case for files, functions, variables, and Hydra config keys. Keep experiment names descriptive, for example `balanced`, `noise_heavy`, `stage1`, `trie`, or `no_trie`. Prefer Hydra overrides over hard-coded parameters.

## Testing Guidelines

Tests use `pytest` conventions and should be named `test_*.py`. Add focused unit tests under `tests/` for shared logic such as constrained decoding, metrics, or data transforms. Use fakes or small fixtures instead of loading real models. Before opening a PR, run `uv run pytest`; add `uv run ruff check src tests` when touching Python broadly.

## Commit & Pull Request Guidelines

Recent commits use short imperative summaries, sometimes with Conventional Commit prefixes such as `feat:` or `refactor:`. Keep subjects concise and specific, for example `feat: add query template` or `refactor: clean up docquery CLI`.

Pull requests should describe the experiment or behavior changed, list commands run, and mention any dataset, checkpoint, or config assumptions. Include log snippets or result paths for evaluation changes.

## Security & Configuration Tips

Do not commit secrets, Hugging Face tokens, local machine paths outside the repo, or private model artifacts. Keep environment setup in local shell files such as `env.sh`, and prefer config files in `configs/` for reproducible parameters.
