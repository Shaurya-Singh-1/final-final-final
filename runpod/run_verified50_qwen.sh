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
export HF_HUB_ENABLE_HF_TRANSFER=1
export MODELDIR
export RYS16
export RYS32
export RUNROOT
PORT_BASE=18000

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
uv pip install --upgrade \
  "transformers[serving]" \
  sb-cli \
  pillow \
  torchvision

uv run python - <<'PY'
from pathlib import Path

serve_path = Path(".venv/lib/python3.11/site-packages/transformers/cli/serve.py")
text = serve_path.read_text()
needle = "from fastapi import FastAPI, HTTPException"
replacement = "from fastapi import FastAPI, HTTPException, Request"
if needle in text and replacement not in text:
    serve_path.write_text(text.replace(needle, replacement, 1))
    print("patched transformers serve.py Request import")
else:
    print("transformers serve.py already patched")
PY

uv run python -c "import transformers; print(transformers.__version__)"
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
                "model_name": os.environ["MODELDIR"],
                "model_kwargs": {
                    "api_base": f"http://127.0.0.1:{18000}/v1",
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
                "model_name": os.environ["RYS16"],
                "model_kwargs": {
                    "api_base": f"http://127.0.0.1:{18001}/v1",
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
                "model_name": os.environ["RYS32"],
                "model_kwargs": {
                    "api_base": f"http://127.0.0.1:{18002}/v1",
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
  local REQUEST_MODEL_NAME="$3"
  local PORT="$4"
  local LOG_NAME="$5"

  CUDA_VISIBLE_DEVICES="$GPUs" nohup uv run transformers serve \
    --force-model "$REQUEST_MODEL_NAME" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --dtype bfloat16 \
    --trust-remote-code \
    > "$LOGDIR/$LOG_NAME" 2>&1 &
}

wait_for_server() {
  local PORT="$1"
  local REQUEST_MODEL_NAME="$2"
  local LABEL="$3"

  for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null; then
      break
    fi
    sleep 5
  done
  curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null

  curl -sf "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    --data @- >/dev/null <<JSON
{
  "model": "$REQUEST_MODEL_NAME",
  "messages": [{"role": "user", "content": "Say READY in one word."}],
  "max_tokens": 8,
  "temperature": 0.0
}
JSON
  echo "${LABEL} server passed smoke test on port ${PORT}."
}

pkill -f "transformers serve" || true

start_server "0,1" "$MODELDIR" "$MODELDIR" 18000 baseline-server.log
start_server "2,3" "$RYS16" "$RYS16" 18001 rys_16_20-server.log
start_server "4,5" "$RYS32" "$RYS32" 18002 rys_32_36-server.log

wait_for_server 18000 "$MODELDIR" "baseline"
wait_for_server 18001 "$RYS16" "rys_16_20"
wait_for_server 18002 "$RYS32" "rys_32_36"

nohup uv run python scripts/run_mini_swe_experiment.py \
  --manifest "$RUNROOT/manifest.json" \
  --conditions-file "$RUNROOT/conditions.json" \
  --model-routes-file "$RUNROOT/model_routes.json" \
  --output-dir "$RUNROOT/runs" \
  --base-config swebench.yaml \
  --workers 2 \
  --condition-name baseline \
  > "$LOGDIR/baseline-run.log" 2>&1 &

nohup uv run python scripts/run_mini_swe_experiment.py \
  --manifest "$RUNROOT/manifest.json" \
  --conditions-file "$RUNROOT/conditions.json" \
  --model-routes-file "$RUNROOT/model_routes.json" \
  --output-dir "$RUNROOT/runs" \
  --base-config swebench.yaml \
  --workers 2 \
  --condition-name rys_16_20 \
  > "$LOGDIR/rys_16_20-run.log" 2>&1 &

nohup uv run python scripts/run_mini_swe_experiment.py \
  --manifest "$RUNROOT/manifest.json" \
  --conditions-file "$RUNROOT/conditions.json" \
  --model-routes-file "$RUNROOT/model_routes.json" \
  --output-dir "$RUNROOT/runs" \
  --base-config swebench.yaml \
  --workers 2 \
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
