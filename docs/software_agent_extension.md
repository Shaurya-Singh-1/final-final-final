# Software-Agent Extension

The linked `software-agents` repository is the course website, not an agent implementation. For the actual software-engineering evaluation phase, this repo is prepared to target a lightweight real agent such as [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent).

## Recommended study flow

1. Use this repo to find a promising RYS block on the chosen base model.
2. Export the relayered checkpoint with `hf_export.export_model`.
3. Serve both the baseline and RYS checkpoints through the same inference stack.
4. Freeze a fixed SWE task manifest and route each condition into the same agent scaffold.
5. Summarize task-level results with `scripts/summarize_agent_runs.py`.

## One-command pipeline

If you want the cleanest handoff, use the wrapper:

```bash
uv run python scripts/run_agent_study_pipeline.py \
  --model-routes-file configs/agent_eval/model_routes.example.json \
  --output-root results/agent_study/demo_run \
  --num-layers 64 \
  --base-model-id your/model \
  --block 24,35 \
  --block 29,34 \
  --instance-id astropy__astropy-12907 \
  --instance-id django__django-11019 \
  --dry-run
```

This will:

1. write `conditions.json`
2. write `manifest.json`
3. print the exact run/eval/summary commands it would execute

Drop `--dry-run` once your model endpoints and Docker-backed SWE-bench setup are ready.

## Condition manifests

You can create a clean baseline-vs-RYS experiment manifest with:

```bash
uv run python scripts/build_agent_study_conditions.py \
  --num-layers 64 \
  --base-model-id your/model \
  --block 24,35 \
  --block 29,34 \
  --output results/agent_study/conditions.json
```

Each condition records:

- the canonical relayer layer list
- extra layer count
- relative compute overhead

## Fixed task manifests

Freeze the downstream task set before running any agent experiments:

```bash
uv run python scripts/create_swebench_manifest.py \
  --subset lite \
  --split test \
  --slice 0:25 \
  --output results/agent_study/manifest.json
```

This writes a reproducible list of exact instance IDs, which is much cleaner than relying on an implicit slice later.

## Running the agent study

1. Prepare the RYS conditions with `build_agent_study_conditions.py`, or let `run_agent_study_pipeline.py` generate them for you.
2. Prepare the exact task manifest with `create_swebench_manifest.py`, or let `run_agent_study_pipeline.py` generate it.
3. Create a model-routing JSON that maps each condition name to a `mini-swe-agent` model config.
4. Run the experiment:

```bash
uv run python scripts/run_mini_swe_experiment.py \
  --manifest results/agent_study/manifest.json \
  --conditions-file results/agent_study/conditions.json \
  --model-routes-file configs/agent_eval/model_routes.example.json \
  --output-dir results/agent_study/runs
```

5. Evaluate the generated `preds.json` files:

```bash
uv run python scripts/evaluate_swebench_runs.py \
  --manifest results/agent_study/manifest.json \
  --conditions-file results/agent_study/conditions.json \
  --experiment-dir results/agent_study/runs
```

6. Convert outputs into normalized run records and summarize:

```bash
uv run python scripts/build_agent_run_records.py \
  --manifest results/agent_study/manifest.json \
  --conditions-file results/agent_study/conditions.json \
  --experiment-dir results/agent_study/runs \
  --output results/agent_study/run_records.json

uv run python scripts/summarize_agent_runs.py \
  --runs results/agent_study/run_records.json \
  --out-dir results/agent_study/summary
```

## Expected run-record schema

`scripts/summarize_agent_runs.py` expects per-task records like:

```json
{
  "task_id": "repo__issue_001",
  "condition": "rys_24_35",
  "success": true,
  "steps": 17,
  "execution_errors": 1,
  "runtime_seconds": 302.5,
  "extra_layers": 11,
  "overhead_fraction": 0.171875,
  "evaluation_status": "resolved"
}
```

These can be produced by any agent framework as long as each task yields one final row. The summary script will aggregate:

- success rate
- average steps
- average execution errors
- average runtime
- average RYS overhead

## Why this structure

The project proposal is about whether a relayered model changes agent outcomes under identical settings. The condition manifest plus task-level summary makes that comparison explicit and keeps the agent scaffold fixed, which is the cleanest experimental design.
