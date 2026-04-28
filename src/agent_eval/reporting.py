from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import csv
from typing import Any


@dataclass(frozen=True)
class AgentRunRecord:
    task_id: str
    condition: str
    success: bool
    steps: int
    execution_errors: int
    runtime_seconds: float
    extra_layers: int = 0
    overhead_fraction: float = 0.0

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "AgentRunRecord":
        return cls(
            task_id=str(row["task_id"]),
            condition=str(row["condition"]),
            success=bool(row["success"]),
            steps=int(row["steps"]),
            execution_errors=int(row["execution_errors"]),
            runtime_seconds=float(row["runtime_seconds"]),
            extra_layers=int(row.get("extra_layers", 0)),
            overhead_fraction=float(row.get("overhead_fraction", 0.0)),
        )


def load_run_records(path: Path) -> list[AgentRunRecord]:
    text = path.read_text().strip()
    if not text:
        return []

    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            rows = parsed.get("runs", [])
        else:
            rows = parsed
    return [AgentRunRecord.from_dict(row) for row in rows]


def summarize_runs(records: list[AgentRunRecord]) -> list[dict[str, Any]]:
    grouped: dict[str, list[AgentRunRecord]] = {}
    for record in records:
        grouped.setdefault(record.condition, []).append(record)

    summary: list[dict[str, Any]] = []
    for condition, items in sorted(grouped.items()):
        total = len(items)
        successes = sum(1 for item in items if item.success)
        summary.append(
            {
                "condition": condition,
                "tasks": total,
                "successes": successes,
                "success_rate": successes / total if total else 0.0,
                "avg_steps": sum(item.steps for item in items) / total if total else 0.0,
                "avg_execution_errors": (
                    sum(item.execution_errors for item in items) / total if total else 0.0
                ),
                "avg_runtime_seconds": (
                    sum(item.runtime_seconds for item in items) / total if total else 0.0
                ),
                "avg_extra_layers": (
                    sum(item.extra_layers for item in items) / total if total else 0.0
                ),
                "avg_overhead_fraction": (
                    sum(item.overhead_fraction for item in items) / total if total else 0.0
                ),
            }
        )
    return summary


def compare_to_baseline(summary: list[dict[str, Any]], baseline: str) -> list[dict[str, Any]]:
    baseline_row = next((row for row in summary if row["condition"] == baseline), None)
    if baseline_row is None:
        raise ValueError(f"Baseline condition not found: {baseline}")

    comparisons: list[dict[str, Any]] = []
    for row in summary:
        comparisons.append(
            {
                "condition": row["condition"],
                "delta_success_rate": row["success_rate"] - baseline_row["success_rate"],
                "delta_avg_steps": row["avg_steps"] - baseline_row["avg_steps"],
                "delta_avg_execution_errors": (
                    row["avg_execution_errors"] - baseline_row["avg_execution_errors"]
                ),
                "delta_avg_runtime_seconds": (
                    row["avg_runtime_seconds"] - baseline_row["avg_runtime_seconds"]
                ),
            }
        )
    return comparisons


def write_summary_outputs(
    summary: list[dict[str, Any]],
    *,
    out_json: Path,
    out_csv: Path,
) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))

    fieldnames = [
        "condition",
        "tasks",
        "successes",
        "success_rate",
        "avg_steps",
        "avg_execution_errors",
        "avg_runtime_seconds",
        "avg_extra_layers",
        "avg_overhead_fraction",
    ]
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
