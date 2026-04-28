from __future__ import annotations

import json
import os
import platform
from pathlib import Path
from typing import Any


DATASET_MAPPING = {
    "full": "princeton-nlp/SWE-Bench",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
    "multilingual": "swe-bench/SWE-Bench_Multilingual",
    "smith": "SWE-bench/SWE-smith",
    "_test": "klieret/swe-bench-dummy-test-dataset",
    "rebench": "nebius/SWE-rebench",
}


def resolve_dataset_name(name: str) -> str:
    return DATASET_MAPPING.get(name, name)


def flatten_config_to_specs(config: dict[str, Any], prefix: str = "") -> list[str]:
    specs: list[str] = []
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            specs.extend(flatten_config_to_specs(value, full_key))
        else:
            specs.append(f"{full_key}={json.dumps(value)}")
    return specs


def load_json_document(path: Path) -> Any:
    return json.loads(path.read_text())


def load_condition_list(path: Path) -> list[dict[str, Any]]:
    raw = load_json_document(path)
    if isinstance(raw, dict):
        conditions = raw.get("conditions", raw)
        if isinstance(conditions, dict):
            return [
                {"name": name, **value} if isinstance(value, dict) else {"name": name, "value": value}
                for name, value in conditions.items()
            ]
        if isinstance(conditions, list):
            return conditions
    if isinstance(raw, list):
        return raw
    raise ValueError(f"Unsupported conditions format in {path}")


def load_route_map(path: Path) -> dict[str, dict[str, Any]]:
    raw = load_json_document(path)
    if isinstance(raw, dict) and "routes" in raw:
        raw = raw["routes"]
    if not isinstance(raw, dict):
        raise ValueError(f"Unsupported route format in {path}")
    return raw


def load_manifest(path: Path) -> dict[str, Any]:
    raw = load_json_document(path)
    if "dataset_name" not in raw or "split" not in raw or "instance_ids" not in raw:
        raise ValueError(f"Manifest at {path} is missing required fields")
    return raw


def merge_condition_and_route(
    condition: dict[str, Any],
    route_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    name = str(condition["name"])
    if name not in route_map:
        raise KeyError(f"Missing route configuration for condition '{name}'")
    route = route_map[name]
    return {
        "name": name,
        "condition": condition,
        "route": route,
    }


def condition_output_dir(root: Path, name: str) -> Path:
    return root / name


def trajectory_path_for(condition_dir: Path, instance_id: str) -> Path:
    return condition_dir / "instances" / f"{instance_id}.traj.json"


def instance_log_path_for(condition_dir: Path, instance_id: str) -> Path:
    return condition_dir / "logs" / f"{instance_id}.log"


def metadata_path_for(condition_dir: Path) -> Path:
    return condition_dir / "run_metadata.jsonl"


def preds_path_for(condition_dir: Path) -> Path:
    return condition_dir / "preds.json"


def evaluation_dir_for(condition_dir: Path) -> Path:
    return condition_dir / "evaluation"


def evaluation_report_path_for(condition_dir: Path) -> Path:
    return evaluation_dir_for(condition_dir) / "run_report.json"


def mini_repo_src_candidates(project_root: Path) -> list[Path]:
    parent = project_root.parent
    return [
        parent / "mini-swe-agent-upstream" / "src",
        project_root / "external" / "mini-swe-agent" / "src",
    ]


def swebench_repo_candidates(project_root: Path) -> list[Path]:
    parent = project_root.parent
    return [
        parent / "SWE-bench-upstream",
        project_root / "external" / "SWE-bench",
    ]


def discover_repo_path(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def prepend_pythonpath(env: dict[str, str], new_path: Path | None) -> dict[str, str]:
    result = dict(env)
    if new_path is None:
        return result
    existing = result.get("PYTHONPATH", "")
    result["PYTHONPATH"] = (
        f"{new_path}{os.pathsep}{existing}" if existing else str(new_path)
    )
    return result


def default_swebench_namespace() -> str:
    machine = platform.machine().lower()
    system = platform.system().lower()
    if system == "darwin" or "arm" in machine or "aarch64" in machine:
        return "none"
    return "swebench"


def build_minisweagent_config_specs(
    route_entry: dict[str, Any],
    *,
    extra_specs: list[str] | None = None,
) -> list[str]:
    specs = list(route_entry.get("config_specs", []))
    if config := route_entry.get("config"):
        if not isinstance(config, dict):
            raise ValueError("route_entry.config must be a dict")
        specs.extend(flatten_config_to_specs(config))
    if extra_specs:
        specs.extend(extra_specs)
    return specs


def build_miniswebench_single_command(
    *,
    python_executable: str,
    dataset_name: str,
    split: str,
    instance_id: str,
    output_path: Path,
    config_specs: list[str],
) -> list[str]:
    cmd = [
        python_executable,
        "-m",
        "minisweagent.run.benchmarks.swebench_single",
        "--subset",
        dataset_name,
        "--split",
        split,
        "--instance",
        instance_id,
        "--agent-class",
        "default",
        "--exit-immediately",
        "--output",
        str(output_path),
    ]
    for spec in config_specs:
        cmd.extend(["-c", spec])
    return cmd


def build_swebench_eval_command(
    *,
    python_executable: str,
    dataset_name: str,
    split: str,
    predictions_path: Path,
    instance_ids: list[str],
    run_id: str,
    max_workers: int,
    timeout: int,
    namespace: str,
) -> list[str]:
    cmd = [
        python_executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset_name,
        "--split",
        split,
        "--predictions_path",
        str(predictions_path),
        "--max_workers",
        str(max_workers),
        "--timeout",
        str(timeout),
        "--run_id",
        run_id,
        "--namespace",
        namespace,
    ]
    if instance_ids:
        cmd.append("--instance_ids")
        cmd.extend(instance_ids)
    return cmd


def parse_trajectory(path: Path) -> dict[str, Any]:
    raw = load_json_document(path)
    info = raw.get("info", {})
    model_stats = info.get("model_stats", {})
    return {
        "exit_status": info.get("exit_status", ""),
        "submission": info.get("submission", ""),
        "api_calls": int(model_stats.get("api_calls", 0)),
        "instance_cost": float(model_stats.get("instance_cost", 0.0)),
    }


def update_predictions_file(
    preds_path: Path,
    *,
    instance_id: str,
    model_name: str,
    submission: str,
) -> None:
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    if preds_path.exists():
        data = load_json_document(preds_path)
    else:
        data = {}
    data[instance_id] = {
        "model_name_or_path": model_name,
        "instance_id": instance_id,
        "model_patch": submission,
    }
    preds_path.write_text(json.dumps(data, indent=2))


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows
