#!/usr/bin/env python3
"""
CPU/GPU-friendly Hugging Face combined worker.

Loads a standard Hugging Face causal LM once, evaluates both math and EQ probe
sets for each queued relayer configuration, and writes:

- combined results pickle
- math-only results pickle
- eq-only results pickle

This is slower than the ExLlama path, but it is much easier to validate on
commodity hardware and in CI-like smoke tests.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.layer_config import is_baseline_layers, parse_queue_entry_layers
from src.core.layer_duplicator import build_model_with_layers
from src.core.layer_duplicator_moe import build_model_with_layers_moe
from src.workers.eq_worker import (
    PADDING_MODE_MASKED,
    pretokenize_eq_dataset,
    run_eq_test,
)
from src.workers.math_worker import pretokenize_dataset, run_math_test_batched_moe
from src.workers.model_utils import (
    is_moe_model,
    load_model_and_tokenizer,
    parse_device_map_arg,
    parse_max_memory_json,
)
from src.workers.shared_queue import SharedWorkQueue, format_eta


def parse_dtype(name: str) -> torch.dtype:
    raw = str(name).strip().lower()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if raw not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[raw]


def main() -> None:
    parser = argparse.ArgumentParser(description="Combined Hugging Face math+EQ worker")
    parser.add_argument("--queue-file", required=True, help="Path to shared queue file")
    parser.add_argument(
        "--combined-results-file",
        required=True,
        help="Path to combined results pickle",
    )
    parser.add_argument("--math-results-file", required=True, help="Path to math results pickle")
    parser.add_argument("--eq-results-file", required=True, help="Path to EQ results pickle")
    parser.add_argument("--model-path", required=True, help="HF model path or local model dir")
    parser.add_argument("--math-dataset-path", default="datasets/math_16.json")
    parser.add_argument("--eq-dataset-path", default="datasets/eq_16.json")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--math-max-new", type=int, default=64)
    parser.add_argument("--eq-max-new", type=int, default=128)
    parser.add_argument("--padding-mode", choices=[PADDING_MODE_MASKED, "inprompt_space"], default=PADDING_MODE_MASKED)
    parser.add_argument("--prompt-pad-id", type=int, default=None)
    parser.add_argument("--use-no-think-prefix", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-responses", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dtype", default="float32", help="float32, bfloat16, or float16")
    parser.add_argument("--attention-impl", default="eager", choices=["eager", "flash_attention_2", "sdpa"])
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device-map", default="cpu")
    parser.add_argument("--max-memory-json", default=None)
    parser.add_argument("--cpu-offload", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--offload-folder", default=None)
    parser.add_argument("--worker-id", default="hf-combined")
    parser.add_argument("--limit-configs", type=int, default=None, help="Optional cap for smoke runs")

    args = parser.parse_args()

    try:
        resolved_device_map = parse_device_map_arg(args.device_map)
    except Exception as exc:
        raise ValueError(f"Invalid --device-map value: {exc}") from exc
    try:
        resolved_max_memory = parse_max_memory_json(args.max_memory_json)
    except Exception as exc:
        raise ValueError(f"Invalid --max-memory-json value: {exc}") from exc

    math_dataset = json.loads(Path(args.math_dataset_path).read_text())
    eq_dataset = json.loads(Path(args.eq_dataset_path).read_text())

    print("=" * 80)
    print(f"HF combined worker [{args.worker_id}]")
    print("=" * 80)
    print(f"Queue file: {args.queue_file}")
    print(f"Combined results: {args.combined_results_file}")
    print(f"Math results: {args.math_results_file}")
    print(f"EQ results: {args.eq_results_file}")
    print(f"Model: {args.model_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Math max_new: {args.math_max_new}")
    print(f"EQ max_new: {args.eq_max_new}")
    print(f"Device map: {resolved_device_map}")

    tokenizer, model, load_meta = load_model_and_tokenizer(
        model_path=args.model_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        torch_dtype=parse_dtype(args.dtype),
        device_map=resolved_device_map,
        attn_implementation=(None if args.attention_impl == "eager" else args.attention_impl),
        max_memory=resolved_max_memory,
        cpu_offload=args.cpu_offload,
        offload_folder=args.offload_folder,
    )
    num_layers = int(load_meta["num_layers"])
    model_is_moe = is_moe_model(model)
    layer_builder = build_model_with_layers_moe if model_is_moe else build_model_with_layers
    print(
        f"Loaded {load_meta['loader']} | num_layers={num_layers} | "
        f"text_stack={load_meta['text_stack']} | model_type={'MoE' if model_is_moe else 'dense'}"
    )

    try:
        device = model.device
    except Exception:
        device = next(model.parameters()).device

    tokenized_math = pretokenize_dataset(
        math_dataset,
        tokenizer,
        device,
        use_no_think_prefix=args.use_no_think_prefix,
    )
    tokenized_eq = pretokenize_eq_dataset(
        eq_dataset,
        tokenizer,
        device,
        use_no_think_prefix=args.use_no_think_prefix,
    )

    combined_queue = SharedWorkQueue(args.queue_file, args.combined_results_file)
    math_results = SharedWorkQueue(args.queue_file, args.math_results_file)
    eq_results = SharedWorkQueue(args.queue_file, args.eq_results_file)

    started = time.time()
    processed = 0

    while True:
        if args.limit_configs is not None and processed >= args.limit_configs:
            print(f"Reached --limit-configs={args.limit_configs}; stopping early.")
            break

        entry = combined_queue.get_next_config()
        if entry is None:
            print("Queue is empty, worker is done.")
            break

        parsed = parse_queue_entry_layers(num_layers, entry)
        layer_indices = parsed["layers"]
        layer_key = parsed["layer_key"]
        legacy_key = parsed["legacy_key"]
        spec = parsed["spec"]

        run_model = model if is_baseline_layers(layer_indices, num_layers) else layer_builder(model, layer_indices)

        config_started = time.time()
        math_result = run_math_test_batched_moe(
            run_model,
            tokenized_math,
            tokenizer,
            batch_size=args.batch_size,
            max_new_tokens=args.math_max_new,
            save_responses=args.save_responses,
            padding_mode=args.padding_mode,
            prompt_pad_id=args.prompt_pad_id,
        )
        eq_result = run_eq_test(
            run_model,
            tokenized_eq,
            tokenizer,
            batch_size=args.batch_size,
            max_new_tokens=args.eq_max_new,
            save_responses=args.save_responses,
            padding_mode=args.padding_mode,
            prompt_pad_id=args.prompt_pad_id,
        )

        math_score = float(math_result["score"] if isinstance(math_result, dict) else math_result)
        eq_score = float(eq_result["score"] if isinstance(eq_result, dict) else eq_result)
        combined_payload = {
            "score": math_score + eq_score,
            "math_score": math_score,
            "eq_score": eq_score,
            "layer_indices": list(layer_indices),
            "legacy_key": list(legacy_key) if legacy_key is not None else None,
            "spec": spec,
            "elapsed_seconds": time.time() - config_started,
        }
        if args.save_responses:
            combined_payload["math"] = math_result
            combined_payload["eq"] = eq_result

        combined_queue.save_results_bulk({layer_key: combined_payload})
        math_results.save_results_bulk({layer_key: math_score})
        eq_results.save_results_bulk({layer_key: eq_score})

        processed += 1
        total_elapsed = time.time() - started
        avg_per_config = total_elapsed / processed
        remaining = combined_queue.get_remaining_count()
        eta = format_eta(avg_per_config * remaining)
        print(
            f"[{processed}] {spec} | math={math_score:.4f} eq={eq_score:.4f} "
            f"elapsed={time.time() - config_started:.1f}s remaining={remaining} eta={eta}"
        )

    print(f"Processed {processed} configs in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
