# Qwen3.5 Verified Subset Runbook

This runbook prepares a small SWE-bench Verified transfer study for:

- `baseline`
- `rys_16_20`
- `rys_32_36`

The default setup uses a fixed shuffled subset and is designed for Delta or another Linux GPU server.

## 0. Clone the expected eval dependencies

This repo auto-discovers these neighboring folders:

- `../mini-swe-agent-upstream`
- `../SWE-bench-upstream`

From the parent directory that will contain this repo:

```bash
git clone https://github.com/SWE-agent/mini-swe-agent.git mini-swe-agent-upstream
git clone https://github.com/SWE-bench/SWE-bench.git SWE-bench-upstream
```

If you only want to generate trajectories first and evaluate later, `mini-swe-agent-upstream` is the required one for the initial run.

## 1. Prepare the bundle

From the repo root:

```bash
uv sync
uv run python scripts/setup_qwen35_verified_subset.py --count 50 --shuffle
```

This writes:

- `results/agent_study/qwen35_verified_50/manifest.json`
- `results/agent_study/qwen35_verified_50/conditions.json`
- `results/agent_study/qwen35_verified_50/model_routes.template.json`

## 2. Edit the route template

Copy the template and fill in the real model names and endpoint URLs for your three served models:

```bash
cp results/agent_study/qwen35_verified_50/model_routes.template.json \
  results/agent_study/qwen35_verified_50/model_routes.json
```

Edit `model_routes.json` so:

- `baseline` points at your baseline endpoint
- `rys_16_20` points at your first relayered endpoint
- `rys_32_36` points at your second relayered endpoint

## 3. Delta / Apptainer config

On Delta, add this repo config on top of the default mini-SWE-agent `swebench.yaml`:

- `configs/agent_eval/swebench_singularity.yaml`

It switches the environment backend from Docker to Apptainer via mini-SWE-agent's `singularity` environment class.

## 4. Run conditions in parallel

The repo runner now supports `--condition-name`, so the easiest parallel strategy is one shell per condition.

Run these from the repo root after cloning `mini-swe-agent` next to this repo or installing it in the environment:

```bash
uv run python scripts/run_mini_swe_experiment.py \
  --manifest results/agent_study/qwen35_verified_50/manifest.json \
  --conditions-file results/agent_study/qwen35_verified_50/conditions.json \
  --model-routes-file results/agent_study/qwen35_verified_50/model_routes.json \
  --output-dir results/agent_study/qwen35_verified_50/runs \
  --base-config swebench.yaml \
  --base-config configs/agent_eval/swebench_singularity.yaml \
  --workers 4 \
  --condition-name baseline
```

```bash
uv run python scripts/run_mini_swe_experiment.py \
  --manifest results/agent_study/qwen35_verified_50/manifest.json \
  --conditions-file results/agent_study/qwen35_verified_50/conditions.json \
  --model-routes-file results/agent_study/qwen35_verified_50/model_routes.json \
  --output-dir results/agent_study/qwen35_verified_50/runs \
  --base-config swebench.yaml \
  --base-config configs/agent_eval/swebench_singularity.yaml \
  --workers 4 \
  --condition-name rys_16_20
```

```bash
uv run python scripts/run_mini_swe_experiment.py \
  --manifest results/agent_study/qwen35_verified_50/manifest.json \
  --conditions-file results/agent_study/qwen35_verified_50/conditions.json \
  --model-routes-file results/agent_study/qwen35_verified_50/model_routes.json \
  --output-dir results/agent_study/qwen35_verified_50/runs \
  --base-config swebench.yaml \
  --base-config configs/agent_eval/swebench_singularity.yaml \
  --workers 4 \
  --condition-name rys_32_36
```

You can run these in three `tmux` panes, three terminals, or background them with `&`.
If the model servers are stable and underutilized, increase `--workers` gradually to `6` or `8`.

## 5. Evaluate and summarize

After all three conditions finish, run evaluation, record building, and summary once:

```bash
uv run python scripts/evaluate_swebench_runs.py \
  --manifest results/agent_study/qwen35_verified_50/manifest.json \
  --conditions-file results/agent_study/qwen35_verified_50/conditions.json \
  --experiment-dir results/agent_study/qwen35_verified_50/runs
```

```bash
uv run python scripts/build_agent_run_records.py \
  --manifest results/agent_study/qwen35_verified_50/manifest.json \
  --conditions-file results/agent_study/qwen35_verified_50/conditions.json \
  --experiment-dir results/agent_study/qwen35_verified_50/runs \
  --output results/agent_study/qwen35_verified_50/run_records.json
```

```bash
uv run python scripts/summarize_agent_runs.py \
  --runs results/agent_study/qwen35_verified_50/run_records.json \
  --out-dir results/agent_study/qwen35_verified_50/summary
```

## Notes

- If you want the official SWE-bench cloud evaluator instead of local harness execution, submit each condition's `preds.json` after generation completes.
- If you want a larger pilot later, rerun the setup script with `--count 100`.
