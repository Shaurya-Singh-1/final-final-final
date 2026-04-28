#!/usr/bin/env python3
"""Evaluate predictions for each condition with the SWE-bench harness."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent_eval.experiment import (
    build_swebench_eval_command,
    condition_output_dir,
    default_swebench_namespace,
    discover_repo_path,
    evaluation_dir_for,
    evaluation_report_path_for,
    load_condition_list,
    load_manifest,
    preds_path_for,
    prepend_pythonpath,
    swebench_repo_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SWE-bench evaluation for every condition")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--conditions-file", required=True)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--namespace", default="auto", help="'auto', 'none', or an explicit namespace")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(Path(args.manifest))
    conditions = load_condition_list(Path(args.conditions_file))
    experiment_dir = Path(args.experiment_dir)

    swebench_root = discover_repo_path(swebench_repo_candidates(ROOT))
    if swebench_root is None and not args.dry_run:
        raise FileNotFoundError(
            "SWE-bench source not found. Clone it next to this project or install it in the environment."
        )

    namespace = default_swebench_namespace() if args.namespace == "auto" else args.namespace
    base_env = prepend_pythonpath(os.environ, swebench_root)

    for condition in conditions:
        name = str(condition["name"])
        condition_dir = condition_output_dir(experiment_dir, name)
        preds_path = preds_path_for(condition_dir)
        if not preds_path.exists():
            print(f"[skip] {name}: no preds.json found")
            continue

        eval_dir = evaluation_dir_for(condition_dir)
        eval_dir.mkdir(parents=True, exist_ok=True)
        report_path = evaluation_report_path_for(condition_dir)
        run_id = f"{name}-eval"
        cmd = build_swebench_eval_command(
            python_executable=args.python_executable,
            dataset_name=manifest["dataset_name"],
            split=manifest["split"],
            predictions_path=preds_path,
            instance_ids=list(manifest["instance_ids"]),
            run_id=run_id,
            max_workers=args.max_workers,
            timeout=args.timeout,
            namespace=namespace,
        )

        if args.dry_run:
            print(" ".join(cmd))
            continue

        result = subprocess.run(
            cmd,
            cwd=eval_dir,
            env=base_env,
            text=True,
            capture_output=True,
        )
        (eval_dir / "stdout.log").write_text(result.stdout)
        (eval_dir / "stderr.log").write_text(result.stderr)
        if result.returncode != 0:
            raise RuntimeError(f"SWE-bench evaluation failed for {name}: see {eval_dir / 'stderr.log'}")

        report_candidates = [
            path
            for path in eval_dir.glob(f"*.{run_id}.json")
            if path.name not in {"run_report.json", "command.json"}
        ]
        if not report_candidates:
            raise FileNotFoundError(f"No SWE-bench report JSON produced for {name}")
        latest = max(report_candidates, key=lambda p: p.stat().st_mtime)
        report_path.write_text(latest.read_text())
        (eval_dir / "command.json").write_text(json.dumps({"command": cmd}, indent=2))
        print(f"[{name}] evaluation report saved to {report_path}")


if __name__ == "__main__":
    main()
