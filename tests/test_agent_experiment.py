from pathlib import Path
import json

from src.agent_eval.experiment import (
    build_minisweagent_config_specs,
    build_miniswebench_single_command,
    build_swebench_eval_command,
    flatten_config_to_specs,
    load_jsonl,
    parse_trajectory,
    update_predictions_file,
    append_jsonl,
)


def test_flatten_config_to_specs_nested_json_values() -> None:
    specs = flatten_config_to_specs(
        {
            "model": {
                "model_name": "openai/demo",
                "model_kwargs": {"temperature": 0.0, "api_base": "http://localhost:8000/v1"},
            },
            "agent": {"cost_limit": 0},
        }
    )
    assert 'model.model_name="openai/demo"' in specs
    assert 'model.model_kwargs.temperature=0.0' in specs
    assert 'agent.cost_limit=0' in specs


def test_build_minisweagent_config_specs_merges_dict_and_explicit_specs() -> None:
    specs = build_minisweagent_config_specs(
        {
            "config_specs": ["swebench.yaml"],
            "config": {"model": {"model_name": "openai/demo"}},
        },
        extra_specs=['agent.cost_limit=0'],
    )
    assert "swebench.yaml" in specs
    assert 'model.model_name="openai/demo"' in specs
    assert "agent.cost_limit=0" in specs


def test_build_commands_include_expected_arguments(tmp_path: Path) -> None:
    mini_cmd = build_miniswebench_single_command(
        python_executable="python3",
        dataset_name="princeton-nlp/SWE-Bench_Lite",
        split="test",
        instance_id="repo__issue-1",
        output_path=tmp_path / "traj.json",
        config_specs=["swebench.yaml", 'model.model_name="openai/demo"'],
    )
    assert mini_cmd[:3] == ["python3", "-m", "minisweagent.run.benchmarks.swebench_single"]
    assert "--instance" in mini_cmd
    assert "repo__issue-1" in mini_cmd

    eval_cmd = build_swebench_eval_command(
        python_executable="python3",
        dataset_name="princeton-nlp/SWE-Bench_Lite",
        split="test",
        predictions_path=tmp_path / "preds.json",
        instance_ids=["repo__issue-1", "repo__issue-2"],
        run_id="demo",
        max_workers=2,
        timeout=123,
        namespace="none",
    )
    assert eval_cmd[:3] == ["python3", "-m", "swebench.harness.run_evaluation"]
    assert "--instance_ids" in eval_cmd
    assert "repo__issue-2" in eval_cmd


def test_predictions_and_jsonl_helpers(tmp_path: Path) -> None:
    preds = tmp_path / "preds.json"
    update_predictions_file(
        preds,
        instance_id="repo__issue-1",
        model_name="openai/demo",
        submission="diff --git a/foo b/foo",
    )
    data = json.loads(preds.read_text())
    assert data["repo__issue-1"]["model_name_or_path"] == "openai/demo"

    rows_path = tmp_path / "rows.jsonl"
    append_jsonl(rows_path, {"a": 1})
    append_jsonl(rows_path, {"a": 2})
    assert [row["a"] for row in load_jsonl(rows_path)] == [1, 2]


def test_parse_trajectory_reads_summary_fields(tmp_path: Path) -> None:
    traj = tmp_path / "traj.json"
    traj.write_text(
        json.dumps(
            {
                "info": {
                    "exit_status": "Submitted",
                    "submission": "patch",
                    "model_stats": {"api_calls": 7, "instance_cost": 1.25},
                }
            }
        )
    )
    parsed = parse_trajectory(traj)
    assert parsed["exit_status"] == "Submitted"
    assert parsed["submission"] == "patch"
    assert parsed["api_calls"] == 7
    assert parsed["instance_cost"] == 1.25
