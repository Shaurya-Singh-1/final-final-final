from pathlib import Path
import json
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_build_agent_run_records_tracks_eval_statuses(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_name": "princeton-nlp/SWE-Bench_Lite",
                "split": "test",
                "instance_ids": ["task_resolved", "task_empty", "task_incomplete", "task_error"],
            }
        )
    )

    conditions_path = tmp_path / "conditions.json"
    conditions_path.write_text(
        json.dumps(
            {
                "conditions": [
                    {"name": "baseline", "extra_layers": 0, "overhead_fraction": 0.0},
                ]
            }
        )
    )

    condition_dir = tmp_path / "runs" / "baseline"
    condition_dir.mkdir(parents=True)
    (condition_dir / "run_metadata.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "instance_id": "task_resolved",
                        "steps": 12,
                        "runtime_seconds": 100.0,
                        "submission_empty": False,
                        "instance_cost": 0.25,
                        "exit_status": "submitted",
                        "returncode": 0,
                    }
                ),
                json.dumps(
                    {
                        "instance_id": "task_empty",
                        "steps": 3,
                        "runtime_seconds": 10.0,
                        "submission_empty": True,
                        "instance_cost": 0.01,
                        "exit_status": "submitted",
                        "returncode": 0,
                    }
                ),
                json.dumps(
                    {
                        "instance_id": "task_incomplete",
                        "steps": 0,
                        "runtime_seconds": 5.0,
                        "submission_empty": False,
                        "instance_cost": 0.0,
                        "exit_status": "crashed",
                        "returncode": 1,
                    }
                ),
            ]
        )
        + "\n"
    )
    eval_dir = condition_dir / "evaluation"
    eval_dir.mkdir(parents=True)
    (eval_dir / "run_report.json").write_text(
        json.dumps(
            {
                "resolved_ids": ["task_resolved"],
                "unresolved_ids": [],
                "error_ids": ["task_error"],
                "empty_patch_ids": ["task_empty"],
                "incomplete_ids": ["task_incomplete"],
            }
        )
    )

    output_path = tmp_path / "run_records.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_agent_run_records.py"),
            "--manifest",
            str(manifest_path),
            "--conditions-file",
            str(conditions_path),
            "--experiment-dir",
            str(tmp_path / "runs"),
            "--output",
            str(output_path),
        ],
        check=True,
        cwd=ROOT,
    )

    rows = json.loads(output_path.read_text())
    status_map = {row["task_id"]: row["evaluation_status"] for row in rows}
    error_map = {row["task_id"]: row["execution_errors"] for row in rows}

    assert status_map["task_resolved"] == "resolved"
    assert status_map["task_empty"] == "empty_patch"
    assert status_map["task_incomplete"] == "incomplete"
    assert status_map["task_error"] == "error"
    assert error_map["task_resolved"] == 0
    assert error_map["task_empty"] == 0
    assert error_map["task_incomplete"] == 1
    assert error_map["task_error"] == 1


def test_run_agent_study_pipeline_dry_run_creates_setup_files(tmp_path: Path) -> None:
    output_root = tmp_path / "demo_run"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_agent_study_pipeline.py"),
            "--model-routes-file",
            str(ROOT / "configs" / "agent_eval" / "model_routes.example.json"),
            "--output-root",
            str(output_root),
            "--num-layers",
            "64",
            "--base-model-id",
            "demo/model",
            "--block",
            "24,35",
            "--instance-id",
            "astropy__astropy-12907",
            "--instance-id",
            "django__django-11019",
            "--dry-run",
        ],
        check=True,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert (output_root / "conditions.json").exists()
    assert (output_root / "manifest.json").exists()
    stdout = result.stdout
    assert "run_mini_swe_experiment.py" in stdout
    assert "evaluate_swebench_runs.py" in stdout
    assert "summarize_agent_runs.py" in stdout
