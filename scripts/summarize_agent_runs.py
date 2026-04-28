#!/usr/bin/env python3
"""Aggregate baseline-vs-RYS software-agent run records into a concise report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent_eval.reporting import compare_to_baseline, load_run_records, summarize_runs, write_summary_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize agent evaluation runs")
    parser.add_argument("--runs", required=True, help="JSON or JSONL file with per-task run records")
    parser.add_argument("--out-dir", required=True, help="Directory for summary outputs")
    parser.add_argument("--baseline", default="baseline", help="Condition name used as baseline")
    args = parser.parse_args()

    records = load_run_records(Path(args.runs))
    summary = summarize_runs(records)
    comparisons = compare_to_baseline(summary, args.baseline)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_summary_outputs(
        summary,
        out_json=out_dir / "summary.json",
        out_csv=out_dir / "summary.csv",
    )
    (out_dir / "baseline_deltas.json").write_text(json.dumps(comparisons, indent=2))

    print(f"Loaded {len(records)} runs across {len(summary)} conditions")
    for row in summary:
        print(
            f"{row['condition']}: success_rate={row['success_rate']:.3f} "
            f"avg_steps={row['avg_steps']:.2f} avg_runtime={row['avg_runtime_seconds']:.2f}s"
        )


if __name__ == "__main__":
    main()
