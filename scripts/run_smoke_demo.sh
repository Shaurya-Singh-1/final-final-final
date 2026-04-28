#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${1:-$ROOT/results/smoke}"
MODEL_DIR="$RESULTS_DIR/tiny-llama"

mkdir -p "$RESULTS_DIR"

uv run python "$ROOT/scripts/build_tiny_llama_fixture.py" \
  --output-dir "$MODEL_DIR"

uv run python "$ROOT/scripts/init_queue.py" \
  --num-layers 6 \
  --queue-file "$RESULTS_DIR/queue.json" \
  --results-file "$RESULTS_DIR/combined_results.pkl"

uv run python "$ROOT/scripts/run_hf_math_eq_combined_worker.py" \
  --queue-file "$RESULTS_DIR/queue.json" \
  --combined-results-file "$RESULTS_DIR/combined_results.pkl" \
  --math-results-file "$RESULTS_DIR/math_results.pkl" \
  --eq-results-file "$RESULTS_DIR/eq_results.pkl" \
  --model-path "$MODEL_DIR" \
  --math-dataset-path "$ROOT/datasets/math_smoke.json" \
  --eq-dataset-path "$ROOT/datasets/eq_smoke.json" \
  --batch-size 2 \
  --math-max-new 8 \
  --eq-max-new 16 \
  --device-map cpu \
  --dtype float32 \
  --no-trust-remote-code \
  --save-responses

uv run python "$ROOT/scripts/analyze_results.py" \
  --math-scores "$RESULTS_DIR/math_results.pkl" \
  --eq-scores "$RESULTS_DIR/eq_results.pkl" \
  --out-dir "$RESULTS_DIR/analysis" \
  --num-layers 6 \
  --title "Smoke Demo"

echo "Smoke demo finished. Artifacts live in $RESULTS_DIR"
