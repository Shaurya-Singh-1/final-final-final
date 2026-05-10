#!/usr/bin/env bash
set -euxo pipefail

REPO_DIR="$(cd "$(dirname "$0")"/.. && pwd)"
ROOT_DIR="$(cd "$REPO_DIR"/.. && pwd)"
RUNROOT="$ROOT_DIR/qwen35_verified_50"
MODELDIR="$ROOT_DIR/models/Qwen3.5-27B"
RYS16="$ROOT_DIR/exports/qwen35-rys-16-20"
RYS32="$ROOT_DIR/exports/qwen35-rys-32-36"
LOGDIR="$REPO_DIR/logs"

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="$ROOT_DIR/cache/uv"
export XDG_CACHE_HOME="$ROOT_DIR/cache/xdg"
export HF_HOME="$ROOT_DIR/cache/hf"
export HUGGINGFACE_HUB_CACHE="$ROOT_DIR/cache/hf/hub"
export MODELDIR
export RUNROOT

mkdir -p "$LOGDIR" "$ROOT_DIR/models" "$ROOT_DIR/exports" "$RUNROOT" \
  "$UV_CACHE_DIR" "$XDG_CACHE_HOME" "$HF_HOME"
cd "$REPO_DIR"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v docker >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y docker.io
fi

if ! docker version >/dev/null 2>&1; then
  pkill dockerd || true
  nohup dockerd --iptables=false --bridge=none --ip-forward=false --ip-masq=false > "$ROOT_DIR/dockerd.log" 2>&1 &
  for _ in $(seq 1 60); do
    if docker version >/dev/null 2>&1; then
      break
    fi
    sleep 2
  done
  docker version >/dev/null 2>&1
fi

python3 --version
nvidia-smi -L

if [ ! -d .venv ]; then
  uv venv --python python3
fi
uv sync
if ! uv run python -c "import vllm, sb_cli" >/dev/null 2>&1; then
  uv pip install vllm sb-cli
fi

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

if [ ! -f "$RYS16/model.safetensors.index.json" ]; then
  uv run python -m hf_export.export_model \
    --source "$MODELDIR" \
    --source-repo-id Qwen/Qwen3.5-27B \
    --output "$RYS16" \
    --blocks "16,20" \
    --overwrite
fi

if [ ! -f "$RYS32/model.safetensors.index.json" ]; then
  uv run python -m hf_export.export_model \
    --source "$MODELDIR" \
    --source-repo-id Qwen/Qwen3.5-27B \
    --output "$RYS32" \
    --blocks "32,36" \
    --overwrite
fi

uv run python scripts/setup_qwen35_verified_subset.py \
  --count 50 \
  --shuffle \
  --output-root "$RUNROOT"

uv run python - <<'PY'
import json
import os
from pathlib import Path

runroot = Path(os.environ["RUNROOT"])
route_path = runroot / "model_routes.json"
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
                    "api_base": "http://127.0.0.1:8001/v1",
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
                    "api_base": "http://127.0.0.1:8002/v1",
                    "api_key": "EMPTY",
                    "temperature": 0.0,
                },
            },
            "agent": {"cost_limit": 0},
        },
        "env": {"OPENAI_API_KEY": "EMPTY"},
    },
}
route_path.write_text(json.dumps(payload, indent=2))
PY

start_server() {
  local GPUs="$1"
  local MODEL_DIR="$2"
  local SERVED_NAME="$3"
  local PORT="$4"
  local LOG_NAME="$5"

  CUDA_VISIBLE_DEVICES="$GPUs" nohup uv run python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --served-model-name "$SERVED_NAME" \
    --host 127.0.0.1 \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --trust-remote-code \
    --port "$PORT" \
    > "$LOGDIR/$LOG_NAME" 2>&1 &
}

pkill -f "vllm.entrypoints.openai.api_server" || true

start_server "0,1" "$MODELDIR" "qwen35-baseline" 8000 baseline-server.log
start_server "2,3" "$RYS16" "qwen35-rys-16-20" 8001 rys_16_20-server.log
start_server "4,5" "$RYS32" "qwen35-rys-32-36" 8002 rys_32_36-server.log

for port in 8000 8001 8002; do
  for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null; then
      break
    fi
    sleep 5
  done
  curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null
done

nohup uv run python scripts/run_mini_swe_experiment.py \
  --manifest "$RUNROOT/manifest.json" \
  --conditions-file "$RUNROOT/conditions.json" \
  --model-routes-file "$RUNROOT/model_routes.json" \
  --output-dir "$RUNROOT/runs" \
  --base-config swebench.yaml \
  --workers 3 \
  --condition-name baseline \
  > "$LOGDIR/baseline-run.log" 2>&1 &

nohup uv run python scripts/run_mini_swe_experiment.py \
  --manifest "$RUNROOT/manifest.json" \
  --conditions-file "$RUNROOT/conditions.json" \
  --model-routes-file "$RUNROOT/model_routes.json" \
  --output-dir "$RUNROOT/runs" \
  --base-config swebench.yaml \
  --workers 3 \
  --condition-name rys_16_20 \
  > "$LOGDIR/rys_16_20-run.log" 2>&1 &

nohup uv run python scripts/run_mini_swe_experiment.py \
  --manifest "$RUNROOT/manifest.json" \
  --conditions-file "$RUNROOT/conditions.json" \
  --model-routes-file "$RUNROOT/model_routes.json" \
  --output-dir "$RUNROOT/runs" \
  --base-config swebench.yaml \
  --workers 3 \
  --condition-name rys_32_36 \
  > "$LOGDIR/rys_32_36-run.log" 2>&1 &

echo "RunPod experiment started."
echo "Logs:"
echo "  $LOGDIR/baseline-server.log"
echo "  $LOGDIR/rys_16_20-server.log"
echo "  $LOGDIR/rys_32_36-server.log"
echo "  $LOGDIR/baseline-run.log"
echo "  $LOGDIR/rys_16_20-run.log"
echo "  $LOGDIR/rys_32_36-run.log"
