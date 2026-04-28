#!/usr/bin/env python3
"""Run baseline vs RYS on the local micro repo-repair benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent_eval.local_repo_bench import (  # noqa: E402
    LocalHFActionModel,
    condition_label,
    condition_layer_spec,
    copy_task_repo,
    load_repo_tasks,
    run_repo_repair_task,
)
from src.agent_eval.reporting import AgentRunRecord, compare_to_baseline, summarize_runs, write_summary_outputs  # noqa: E402
from src.agent_eval.experiment import load_condition_list  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local micro repo benchmark")
    parser.add_argument("--model-path", required=True, help="Hugging Face model id or local model dir")
    parser.add_argument(
        "--manifest",
        default="benchmarks/micro_repo_bench/manifest.json",
        help="Benchmark manifest JSON",
    )
    parser.add_argument("--conditions-file", required=True, help="Condition manifest JSON")
    parser.add_argument("--output-dir", required=True, help="Directory for trajectories and summary")
    parser.add_argument("--device", default="auto", help="auto, mps, cpu, or cuda")
    parser.add_argument("--dtype", default="float16", help="float16, bfloat16, or float32")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--step-limit", type=int, default=12)
    parser.add_argument("--test-timeout", type=int, default=30)
    parser.add_argument("--limit-tasks", type=int, default=0)
    parser.add_argument("--baseline", default="baseline")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conditions = load_condition_list(Path(args.conditions_file))
    tasks = load_repo_tasks(Path(args.manifest))
    if args.limit_tasks > 0:
        tasks = tasks[: args.limit_tasks]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    for condition in conditions:
        name = condition_label(condition)
        layer_spec = condition_layer_spec(condition)
        condition_dir = output_dir / name
        condition_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{name}] loading model")
        model = LocalHFActionModel(
            model_path=args.model_path,
            layer_spec=("" if name == args.baseline else layer_spec),
            device=args.device,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
        )
        try:
            for task in tasks:
                task_dir = copy_task_repo(task, condition_dir / "workspaces")
                result = run_repo_repair_task(
                    model=model,
                    task=task,
                    work_dir=task_dir,
                    step_limit=args.step_limit,
                    test_timeout=args.test_timeout,
                )
                task_out = condition_dir / "tasks" / task.task_id
                task_out.mkdir(parents=True, exist_ok=True)
                (task_out / "trajectory.json").write_text(json.dumps(result, indent=2))
                records.append(
                    {
                        "task_id": task.task_id,
                        "condition": name,
                        "success": bool(result["success"]),
                        "steps": len(result["steps"]),
                        "execution_errors": int(result["parse_errors"]) + int(result["tool_errors"]),
                        "runtime_seconds": float(result["runtime_seconds"]),
                        "extra_layers": int(condition.get("extra_layers", 0)),
                        "overhead_fraction": float(condition.get("overhead_fraction", 0.0)),
                        "evaluation_status": "resolved" if result["success"] else "unresolved",
                        "exit_status": result["exit_status"],
                    }
                )
                print(
                    f"[{name}] {task.task_id}: success={result['success']} "
                    f"steps={len(result['steps'])} runtime={result['runtime_seconds']:.1f}s"
                )
        finally:
            model.close()

    records_path = output_dir / "run_records.json"
    records_path.write_text(json.dumps(records, indent=2))

    summary = summarize_runs([AgentRunRecord.from_dict(row) for row in records])
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    write_summary_outputs(summary, out_json=summary_dir / "summary.json", out_csv=summary_dir / "summary.csv")
    (summary_dir / "baseline_deltas.json").write_text(json.dumps(compare_to_baseline(summary, args.baseline), indent=2))
    print(f"\nWrote run records to {records_path}")


if __name__ == "__main__":
    main()
