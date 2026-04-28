from pathlib import Path

from src.agent_eval.local_repo_bench import extract_first_json_object, validate_action, RepoWorkspace


def test_extract_first_json_object_handles_wrapped_text() -> None:
    payload = extract_first_json_object('Thoughts first {"thought":"ok","action":"run_tests"} trailing text')
    assert payload["action"] == "run_tests"


def test_validate_action_requires_fields() -> None:
    action = validate_action({"thought": "read it", "action": "read_file", "path": "src/app.py"})
    assert action["action"] == "read_file"


def test_workspace_replace_text_updates_file(tmp_path: Path) -> None:
    path = tmp_path / "src" / "tool.py"
    path.parent.mkdir(parents=True)
    path.write_text("value = 1\n")
    workspace = RepoWorkspace(tmp_path, "PYTHONPATH=src pytest -q")
    result = workspace.replace_text("src/tool.py", "value = 1", "value = 2")
    assert result["ok"] is True
    assert path.read_text() == "value = 2\n"
