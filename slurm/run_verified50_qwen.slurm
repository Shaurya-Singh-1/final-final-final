#!/bin/bash
#SBATCH -A bdes-delta-gpu
#SBATCH -p gpuA100x4,gpuA40x4
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH -J qwen_verified50
#SBATCH -o qwen_verified50-%j.out

set -euxo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
ROOT_DIR="$(cd "$REPO_DIR"/.. && pwd)"
RUNROOT="$ROOT_DIR/qwen35_verified_50"
MODELDIR="$ROOT_DIR/models/Qwen3.5-27B"
RYS16="$ROOT_DIR/exports/qwen35-rys-16-20"
RYS32="$ROOT_DIR/exports/qwen35-rys-32-36"

export UV_CACHE_DIR="$ROOT_DIR/cache/uv"
export XDG_CACHE_HOME="$ROOT_DIR/cache/xdg"
export HF_HOME="$ROOT_DIR/cache/hf"
export HUGGINGFACE_HUB_CACHE="$ROOT_DIR/cache/hf/hub"
export MODELDIR
export RUNROOT

mkdir -p "$REPO_DIR/logs" "$ROOT_DIR/models" "$ROOT_DIR/exports" "$RUNROOT" \
  "$UV_CACHE_DIR" "$XDG_CACHE_HOME" "$HF_HOME"
cd "$REPO_DIR"

hostname
nvidia-smi -L

rm -rf .venv
uv python install 3.12
uv venv --python 3.12
uv sync
uv pip install vllm sb-cli

uv run python -c "import vllm; print('vllm ok')"
uv run python -c "import sb_cli; print('sb-cli ok')"

if [ ! -f "$MODELDIR/model.safetensors.index.json" ]; then
  uv run python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3.5-27B",
    local_dir=os.environ["MODELDIR"],
)
PY
fi

test -f "$MODELDIR/config.json"
test -f "$MODELDIR/model.safetensors.index.json"

uv run python -m hf_export.export_model \
  --source "$MODELDIR" \
  --source-repo-id Qwen/Qwen3.5-27B \
  --output "$RYS16" \
  --blocks "16,20" \
  --overwrite

uv run python -m hf_export.export_model \
  --source "$MODELDIR" \
  --source-repo-id Qwen/Qwen3.5-27B \
  --output "$RYS32" \
  --blocks "32,36" \
  --overwrite

uv run python scripts/setup_qwen35_verified_subset.py \
  --count 50 \
  --shuffle \
  --output-root "$RUNROOT"

python - <<'PY'
import json
import os
from pathlib import Path

runroot = Path(os.environ["RUNROOT"])
payload = {
    "baseline": {
        "config": {
            "model": {
                "model_class": "litellm",
                "model_name": "qwen35-baseline",
                "model_kwargs": {
                    "api_base": "http://127.0.0.1:8000/v1",
                    "api_key": "EMPTY",
                    "temperature": 0.0,
                },
            },
            "agent": {"cost_limit": 0},
        },
        "env": {"OPENAI_API_KEY": "EMPTY"},
    },
    "rys_16_20": {
        "config": {
            "model": {
                "model_class": "litellm",
                "model_name": "qwen35-rys-16-20",
                "model_kwargs": {
                    "api_base": "http://127.0.0.1:8000/v1",
                    "api_key": "EMPTY",
                    "temperature": 0.0,
                },
            },
            "agent": {"cost_limit": 0},
        },
        "env": {"OPENAI_API_KEY": "EMPTY"},
    },
    "rys_32_36": {
        "config": {
            "model": {
                "model_class": "litellm",
                "model_name": "qwen35-rys-32-36",
                "model_kwargs": {
                    "api_base": "http://127.0.0.1:8000/v1",
                    "api_key": "EMPTY",
                    "temperature": 0.0,
                },
            },
            "agent": {"cost_limit": 0},
        },
        "env": {"OPENAI_API_KEY": "EMPTY"},
    },
}
(runroot / "model_routes.json").write_text(json.dumps(payload, indent=2))
PY

run_condition () {
  local NAME="$1"
  local MODEL_DIR="$2"
  local SERVED_NAME="$3"

  pkill -f "vllm.entrypoints.openai.api_server.*--port 8000" 2>/dev/null || true
  sleep 5

  CUDA_VISIBLE_DEVICES=0,1 nohup uv run python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --served-model-name "$SERVED_NAME" \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --trust-remote-code \
    --port 8000 \
    > "$REPO_DIR/logs/${NAME}-server.log" 2>&1 &
  echo $! > "$REPO_DIR/logs/vllm.pid"

  until curl -sf http://127.0.0.1:8000/v1/models >/dev/null; do
    sleep 5
  done

  uv run python scripts/run_mini_swe_experiment.py \
    --manifest "$RUNROOT/manifest.json" \
    --conditions-file "$RUNROOT/conditions.json" \
    --model-routes-file "$RUNROOT/model_routes.json" \
    --output-dir "$RUNROOT/runs" \
    --base-config swebench.yaml \
    --base-config configs/agent_eval/swebench_singularity.yaml \
    --workers 2 \
    --condition-name "$NAME"

  kill "$(cat "$REPO_DIR/logs/vllm.pid")" 2>/dev/null || true
  rm -f "$REPO_DIR/logs/vllm.pid"
  sleep 10
}

run_condition baseline "$MODELDIR" qwen35-baseline
run_condition rys_16_20 "$RYS16" qwen35-rys-16-20
run_condition rys_32_36 "$RYS32" qwen35-rys-32-36
