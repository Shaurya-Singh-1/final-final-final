#!/usr/bin/env python3
"""Run baseline/RYS conditions through mini-swe-agent on a fixed manifest."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent_eval.experiment import (
    append_jsonl,
    build_minisweagent_config_specs,
    build_miniswebench_single_command,
    condition_output_dir,
    discover_repo_path,
    instance_log_path_for,
    load_condition_list,
    load_manifest,
    load_route_map,
    merge_condition_and_route,
    metadata_path_for,
    mini_repo_src_candidates,
    parse_trajectory,
    preds_path_for,
    prepend_pythonpath,
    trajectory_path_for,
    update_predictions_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a mini-swe-agent experiment over a fixed manifest")
    parser.add_argument("--manifest", required=True, help="Manifest JSON from create_swebench_manifest.py")
    parser.add_argument("--conditions-file", required=True, help="Condition JSON")
    parser.add_argument("--model-routes-file", required=True, help="mini-swe-agent model routing JSON")
    parser.add_argument("--output-dir", required=True, help="Experiment output root")
    parser.add_argument(
        "--base-config",
        action="append",
        default=[],
        help="Base mini-swe-agent config spec. Repeatable; defaults to swebench.yaml.",
    )
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--condition-name",
        action="append",
        default=[],
        help="Run only these condition names. Repeat flag to launch conditions separately in parallel.",
    )
    parser.add_argument("--redo-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def normalize_config_specs(specs: list[str]) -> list[str]:
    """Resolve repo-local config file paths to absolute paths before subprocess launch."""
    normalized: list[str] = []
    for spec in specs:
        if "=" in spec:
            normalized.append(spec)
            continue

        path = Path(spec)
        if path.exists():
            normalized.append(str(path.resolve()))
            continue

        root_relative = ROOT / spec
        if root_relative.exists():
            normalized.append(str(root_relative.resolve()))
            continue

        normalized.append(spec)

    return normalized


def main() -> None:
    args = parse_args()
    manifest = load_manifest(Path(args.manifest))
    conditions = load_condition_list(Path(args.conditions_file))
    if args.condition_name:
        requested = set(args.condition_name)
        available = {str(condition["name"]) for condition in conditions}
        missing = sorted(requested - available)
        if missing:
            raise KeyError(
                f"Unknown condition name(s): {missing}. Available conditions: {sorted(available)}"
            )
        conditions = [condition for condition in conditions if str(condition["name"]) in requested]

    routes = load_route_map(Path(args.model_routes_file))
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    mini_src = discover_repo_path(mini_repo_src_candidates(ROOT))
    if mini_src is None and not args.dry_run:
        raise FileNotFoundError(
            "mini-swe-agent source not found. Clone it next to this project or install it in the environment."
        )

    base_specs = normalize_config_specs(list(args.base_config) if args.base_config else ["swebench.yaml"])
    env_lock = threading.Lock()

    for condition in conditions:
        bundle = merge_condition_and_route(condition, routes)
        name = bundle["name"]
        route = bundle["route"]
        condition_dir = condition_output_dir(output_root, name)
        (condition_dir / "instances").mkdir(parents=True, exist_ok=True)
        (condition_dir / "logs").mkdir(parents=True, exist_ok=True)
        (condition_dir / "condition.json").write_text(json.dumps(bundle, indent=2))

        model_name = route.get("model_name") or route.get("config", {}).get("model", {}).get("model_name", name)
        config_specs = base_specs + build_minisweagent_config_specs(route)
        base_env = prepend_pythonpath(os.environ, mini_src)
        base_env.update({str(k): str(v) for k, v in route.get("env", {}).items()})

        tasks = []
        for instance_id in manifest["instance_ids"]:
            traj_path = trajectory_path_for(condition_dir, instance_id)
            if traj_path.exists() and not args.redo_existing:
                continue
            tasks.append(instance_id)

        if args.dry_run:
            print(f"\n[{name}] {len(tasks)} instances")
            for instance_id in tasks:
                cmd = build_miniswebench_single_command(
                    python_executable=args.python_executable,
                    dataset_name=manifest["dataset_name"],
                    split=manifest["split"],
                    instance_id=instance_id,
                    output_path=trajectory_path_for(condition_dir, instance_id),
                    config_specs=config_specs,
                )
                print(" ".join(cmd))
            continue

        def run_instance(instance_id: str) -> None:
            traj_path = trajectory_path_for(condition_dir, instance_id)
            log_path = instance_log_path_for(condition_dir, instance_id)
            cmd = build_miniswebench_single_command(
                python_executable=args.python_executable,
                dataset_name=manifest["dataset_name"],
                split=manifest["split"],
                instance_id=instance_id,
                output_path=traj_path,
                config_specs=config_specs,
            )

            started = time.time()
            result = subprocess.run(
                cmd,
                cwd=condition_dir,
                env=base_env,
                text=True,
                capture_output=True,
            )
            log_path.write_text(result.stdout + ("\n[stderr]\n" + result.stderr if result.stderr else ""))
            finished = time.time()

            traj = parse_trajectory(traj_path) if traj_path.exists() else {
                "exit_status": f"subprocess_returncode_{result.returncode}",
                "submission": "",
                "api_calls": 0,
                "instance_cost": 0.0,
            }

            with env_lock:
                update_predictions_file(
                    preds_path_for(condition_dir),
                    instance_id=instance_id,
                    model_name=model_name,
                    submission=traj["submission"],
                )
                append_jsonl(
                    metadata_path_for(condition_dir),
                    {
                        "instance_id": instance_id,
                        "condition": name,
                        "runtime_seconds": finished - started,
                        "exit_status": traj["exit_status"],
                        "steps": traj["api_calls"],
                        "instance_cost": traj["instance_cost"],
                        "submission_empty": traj["submission"] in {"", None},
                        "returncode": result.returncode,
                        "extra_layers": int(condition.get("extra_layers", 0)),
                        "overhead_fraction": float(condition.get("overhead_fraction", 0.0)),
                    },
                )

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(run_instance, tasks))

        print(f"[{name}] finished {len(tasks)} new instances")


if __name__ == "__main__":
    main()
