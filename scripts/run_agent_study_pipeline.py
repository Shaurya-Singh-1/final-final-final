#!/usr/bin/env python3
"""One-command entrypoint for the downstream baseline-vs-RYS agent study."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the downstream software-agent study pipeline")
    parser.add_argument(
        "--output-root",
        default="results/agent_study/pipeline_run",
        help="Root directory for conditions, manifest, runs, and summaries",
    )
    parser.add_argument("--model-routes-file", required=True, help="Route map JSON for mini-swe-agent models")
    parser.add_argument("--conditions-file", default="", help="Existing condition manifest JSON")
    parser.add_argument("--manifest", default="", help="Existing task manifest JSON")
    parser.add_argument("--num-layers", type=int, help="Layer count for build_agent_study_conditions.py")
    parser.add_argument("--base-model-id", default="", help="Base model label for the condition manifest")
    parser.add_argument(
        "--block",
        action="append",
        default=[],
        help="Single RYS block like 24,35. Repeat for multiple conditions.",
    )
    parser.add_argument("--subset", default="lite", help="SWE-bench subset alias or full dataset name")
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument("--filter", dest="filter_spec", default="", help="Regex filter over instance ids")
    parser.add_argument("--slice", dest="slice_spec", default="", help="Python slice spec like 0:25")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--instance-id",
        action="append",
        default=[],
        help="Explicit instance id. Repeat to avoid downloading the dataset when building a manifest.",
    )
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument(
        "--base-config",
        action="append",
        default=[],
        help="Repeatable mini-swe-agent config spec passed through to run_mini_swe_experiment.py",
    )
    parser.add_argument("--workers", type=int, default=1, help="Parallel agent trajectories per condition")
    parser.add_argument("--max-workers", type=int, default=1, help="Parallel SWE-bench harness workers")
    parser.add_argument("--timeout", type=int, default=1800, help="SWE-bench evaluation timeout in seconds")
    parser.add_argument("--namespace", default="auto", help="'auto', 'none', or an explicit SWE-bench namespace")
    parser.add_argument("--redo-existing", action="store_true")
    parser.add_argument("--skip-eval", action="store_true", help="Stop after producing agent trajectories")
    parser.add_argument("--dry-run", action="store_true", help="Write setup files, then print later pipeline steps")
    return parser.parse_args()


def run_checked(cmd: list[str], *, cwd: Path) -> None:
    print(f"$ {shlex.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def ensure_condition_inputs(args: argparse.Namespace) -> None:
    if args.conditions_file:
        return
    if args.num_layers is None:
        raise SystemExit("--num-layers is required when --conditions-file is not provided")
    if not args.base_model_id:
        raise SystemExit("--base-model-id is required when --conditions-file is not provided")


def build_conditions_if_needed(args: argparse.Namespace, output_root: Path) -> Path:
    if args.conditions_file:
        path = Path(args.conditions_file).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Condition manifest not found: {path}")
        return path

    path = output_root / "conditions.json"
    cmd = [
        args.python_executable,
        str(ROOT / "scripts" / "build_agent_study_conditions.py"),
        "--num-layers",
        str(args.num_layers),
        "--base-model-id",
        args.base_model_id,
    ]
    for block in args.block:
        cmd.extend(["--block", block])
    cmd.extend(["--output", str(path)])
    run_checked(cmd, cwd=ROOT)
    return path


def build_manifest_if_needed(args: argparse.Namespace, output_root: Path) -> Path:
    if args.manifest:
        path = Path(args.manifest).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Task manifest not found: {path}")
        return path

    path = output_root / "manifest.json"
    cmd = [
        args.python_executable,
        str(ROOT / "scripts" / "create_swebench_manifest.py"),
        "--subset",
        args.subset,
        "--split",
        args.split,
        "--output",
        str(path),
    ]
    if args.filter_spec:
        cmd.extend(["--filter", args.filter_spec])
    if args.slice_spec:
        cmd.extend(["--slice", args.slice_spec])
    if args.shuffle:
        cmd.append("--shuffle")
    if args.seed != 42:
        cmd.extend(["--seed", str(args.seed)])
    for instance_id in args.instance_id:
        cmd.extend(["--instance-id", instance_id])
    run_checked(cmd, cwd=ROOT)
    return path


def build_run_command(
    args: argparse.Namespace,
    *,
    manifest_path: Path,
    conditions_path: Path,
    output_root: Path,
) -> list[str]:
    cmd = [
        args.python_executable,
        str(ROOT / "scripts" / "run_mini_swe_experiment.py"),
        "--manifest",
        str(manifest_path),
        "--conditions-file",
        str(conditions_path),
        "--model-routes-file",
        str(Path(args.model_routes_file).resolve()),
        "--output-dir",
        str(output_root / "runs"),
        "--workers",
        str(args.workers),
    ]
    for spec in args.base_config:
        cmd.extend(["--base-config", spec])
    if args.redo_existing:
        cmd.append("--redo-existing")
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def build_eval_command(
    args: argparse.Namespace,
    *,
    manifest_path: Path,
    conditions_path: Path,
    output_root: Path,
) -> list[str]:
    cmd = [
        args.python_executable,
        str(ROOT / "scripts" / "evaluate_swebench_runs.py"),
        "--manifest",
        str(manifest_path),
        "--conditions-file",
        str(conditions_path),
        "--experiment-dir",
        str(output_root / "runs"),
        "--max-workers",
        str(args.max_workers),
        "--timeout",
        str(args.timeout),
        "--namespace",
        args.namespace,
    ]
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def build_records_command(
    args: argparse.Namespace,
    *,
    manifest_path: Path,
    conditions_path: Path,
    output_root: Path,
) -> list[str]:
    return [
        args.python_executable,
        str(ROOT / "scripts" / "build_agent_run_records.py"),
        "--manifest",
        str(manifest_path),
        "--conditions-file",
        str(conditions_path),
        "--experiment-dir",
        str(output_root / "runs"),
        "--output",
        str(output_root / "run_records.json"),
    ]


def build_summary_command(args: argparse.Namespace, *, output_root: Path) -> list[str]:
    return [
        args.python_executable,
        str(ROOT / "scripts" / "summarize_agent_runs.py"),
        "--runs",
        str(output_root / "run_records.json"),
        "--out-dir",
        str(output_root / "summary"),
    ]


def main() -> None:
    args = parse_args()
    ensure_condition_inputs(args)

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    conditions_path = build_conditions_if_needed(args, output_root)
    manifest_path = build_manifest_if_needed(args, output_root)

    run_cmd = build_run_command(
        args,
        manifest_path=manifest_path,
        conditions_path=conditions_path,
        output_root=output_root,
    )
    eval_cmd = build_eval_command(
        args,
        manifest_path=manifest_path,
        conditions_path=conditions_path,
        output_root=output_root,
    )
    records_cmd = build_records_command(
        args,
        manifest_path=manifest_path,
        conditions_path=conditions_path,
        output_root=output_root,
    )
    summary_cmd = build_summary_command(args, output_root=output_root)

    if args.dry_run:
        print("\nPlanned downstream steps:")
        print(f"$ {shlex.join(run_cmd)}")
        if not args.skip_eval:
            print(f"$ {shlex.join(eval_cmd)}")
            print(f"$ {shlex.join(records_cmd)}")
            print(f"$ {shlex.join(summary_cmd)}")
        return

    run_checked(run_cmd, cwd=ROOT)
    if args.skip_eval:
        return
    run_checked(eval_cmd, cwd=ROOT)
    run_checked(records_cmd, cwd=ROOT)
    run_checked(summary_cmd, cwd=ROOT)


if __name__ == "__main__":
    main()
