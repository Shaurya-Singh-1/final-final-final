#!/usr/bin/env python3
"""Build per-task run records from experiment outputs and SWE-bench evaluation reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent_eval.experiment import (
    condition_output_dir,
    evaluation_report_path_for,
    load_condition_list,
    load_manifest,
    load_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build standardized run records for agent summaries")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--conditions-file", required=True)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--output", required=True, help="Output JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(Path(args.manifest))
    conditions = load_condition_list(Path(args.conditions_file))
    experiment_dir = Path(args.experiment_dir)

    condition_map = {str(c["name"]): c for c in conditions}
    rows: list[dict] = []

    for condition_name, condition in condition_map.items():
        condition_dir = condition_output_dir(experiment_dir, condition_name)
        metadata_rows = {
            row["instance_id"]: row
            for row in load_jsonl(condition_dir / "run_metadata.jsonl")
        }
        eval_report_path = evaluation_report_path_for(condition_dir)
        resolved_ids: set[str] = set()
        unresolved_ids: set[str] = set()
        error_ids: set[str] = set()
        empty_patch_ids: set[str] = set()
        incomplete_ids: set[str] = set()
        if eval_report_path.exists():
            report = json.loads(eval_report_path.read_text())
            resolved_ids = set(report.get("resolved_ids", []))
            unresolved_ids = set(report.get("unresolved_ids", []))
            error_ids = set(report.get("error_ids", []))
            empty_patch_ids = set(report.get("empty_patch_ids", []))
            incomplete_ids = set(report.get("incomplete_ids", []))

        for instance_id in manifest["instance_ids"]:
            meta = metadata_rows.get(instance_id, {})
            status = (
                "resolved"
                if instance_id in resolved_ids
                else "unresolved"
                if instance_id in unresolved_ids
                else "empty_patch"
                if instance_id in empty_patch_ids
                else "incomplete"
                if instance_id in incomplete_ids
                else "error"
                if instance_id in error_ids
                else "not_evaluated"
            )
            rows.append(
                {
                    "task_id": instance_id,
                    "condition": condition_name,
                    "success": instance_id in resolved_ids,
                    "steps": int(meta.get("steps", 0)),
                    "execution_errors": (
                        1
                        if (
                            instance_id in error_ids
                            or instance_id in incomplete_ids
                            or str(meta.get("returncode", 0)) != "0"
                        )
                        else 0
                    ),
                    "runtime_seconds": float(meta.get("runtime_seconds", 0.0)),
                    "extra_layers": int(condition.get("extra_layers", 0)),
                    "overhead_fraction": float(condition.get("overhead_fraction", 0.0)),
                    "evaluation_status": status,
                    "submission_empty": bool(meta.get("submission_empty", True)),
                    "instance_cost": float(meta.get("instance_cost", 0.0)),
                    "exit_status": meta.get("exit_status", ""),
                }
            )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {len(rows)} run records to {out_path}")


if __name__ == "__main__":
    main()
