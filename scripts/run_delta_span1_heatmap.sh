#!/usr/bin/env bash

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_DEFAULT="$(cd "$REPO/.." && pwd)"
ROOT="${ROOT:-$ROOT_DEFAULT}"
MODEL_PATH="${1:-$ROOT/models/Qwen3.5-27B}"
GPU_COUNT="${GPU_COUNT:-3}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT:-$ROOT/results/span1_heatmap_$TIMESTAMP}"
export MODEL_PATH

export HF_HOME="${HF_HOME:-$ROOT/cache/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$ROOT/cache/hf/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$ROOT/cache/hf/transformers}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT/cache/xdg}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/cache/uv}"

mkdir -p "$OUT" "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME" "$UV_CACHE_DIR"

cd "$REPO"

AVAILABLE_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
if [[ -z "$AVAILABLE_GPUS" || "$AVAILABLE_GPUS" -lt 1 ]]; then
  echo "No visible NVIDIA GPUs found. Run this on the allocated Delta compute node, not the login node." >&2
  exit 1
fi
if [[ "$GPU_COUNT" -gt "$AVAILABLE_GPUS" ]]; then
  echo "Requested GPU_COUNT=$GPU_COUNT but only found $AVAILABLE_GPUS visible GPUs; lowering to match." >&2
  GPU_COUNT="$AVAILABLE_GPUS"
fi

if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "Model config not found: $MODEL_PATH/config.json" >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  uv python install 3.12
  uv venv --python 3.12
fi

uv sync

NUM_LAYERS="$(
  uv run python - <<'PY'
import json
import os
from pathlib import Path

cfg = json.loads(Path(os.environ["MODEL_PATH"]).joinpath("config.json").read_text())
num_layers = cfg.get("num_hidden_layers") or cfg.get("text_config", {}).get("num_hidden_layers")
if num_layers is None:
    raise SystemExit("Could not infer num_hidden_layers from config.json")
print(int(num_layers))
PY
)"

uv run python scripts/init_queue.py \
  --num-layers "$NUM_LAYERS" \
  --min-span 1 \
  --max-span 1 \
  --queue-file "$OUT/queue.json" \
  --results-file "$OUT/combined_results.pkl"

echo "$OUT" > "$ROOT/results/LATEST_HEATMAP_RUN.txt"

echo
echo "Starting span-1 layer sweep"
echo "  repo:  $REPO"
echo "  model: $MODEL_PATH"
echo "  out:   $OUT"
echo "  layers:$NUM_LAYERS"
echo "  gpus:  $GPU_COUNT"
echo
echo "Monitor with:"
echo "  tail -f $OUT/worker0.log"
echo

launch_worker() {
  local gpu_id="$1"
  local worker_name="h200-$gpu_id"
  local log_file="$OUT/worker${gpu_id}.log"
  local pid_file="$OUT/worker${gpu_id}.pid"

  nohup bash -lc "cd '$REPO' && CUDA_VISIBLE_DEVICES=$gpu_id uv run python scripts/run_hf_math_eq_combined_worker.py \
    --queue-file '$OUT/queue.json' \
    --combined-results-file '$OUT/combined_results.pkl' \
    --math-results-file '$OUT/math_results.pkl' \
    --eq-results-file '$OUT/eq_results.pkl' \
    --model-path '$MODEL_PATH' \
    --math-dataset-path datasets/math_16.json \
    --eq-dataset-path datasets/eq_16.json \
    --batch-size 2 \
    --math-max-new 64 \
    --eq-max-new 128 \
    --dtype bfloat16 \
    --attention-impl sdpa \
    --trust-remote-code \
    --local-files-only \
    --device-map cuda:0 \
    --worker-id '$worker_name'" > "$log_file" 2>&1 &

  echo $! > "$pid_file"
}

for gpu_id in $(seq 0 $((GPU_COUNT - 1))); do
  launch_worker "$gpu_id"
done

sleep 5

echo "Worker startup status:"
for gpu_id in $(seq 0 $((GPU_COUNT - 1))); do
  pid_file="$OUT/worker${gpu_id}.pid"
  log_file="$OUT/worker${gpu_id}.log"
  pid="$(cat "$pid_file")"
  if ps -p "$pid" >/dev/null 2>&1; then
    echo "  worker${gpu_id}: pid=$pid running"
  else
    echo "  worker${gpu_id}: pid=$pid exited early" >&2
  fi
  tail -20 "$log_file" || true
done

for gpu_id in $(seq 0 $((GPU_COUNT - 1))); do
  wait "$(cat "$OUT/worker${gpu_id}.pid")"
done

uv run python scripts/analyze_results.py \
  --math-scores "$OUT/math_results.pkl" \
  --eq-scores "$OUT/eq_results.pkl" \
  --out-dir "$OUT/analysis" \
  --num-layers "$NUM_LAYERS" \
  --title "Qwen3.5 span-1 layer sweep"

echo
echo "DONE"
echo "Results directory: $OUT"
echo "Heatmap:           $OUT/analysis/balanced_heatmap_score.png"
echo "Ranking CSV:       $OUT/analysis/top10_balanced_zdelta.csv"
echo "Ranking JSON:      $OUT/analysis/top10_balanced_zdelta.json"
echo "Summary:           $OUT/analysis/balanced_summary.json"
