from __future__ import annotations

import gc
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.core.layer_config import layer_spec_string, normalize_to_layers
from src.core.layer_duplicator import build_model_with_layers
from src.workers.model_utils import apply_chat_template_fallback, strip_thinking


SYSTEM_PROMPT = """You are a software repair agent working on a small Python repository.

Always respond with exactly one JSON object and no markdown.

Allowed actions:
- {"thought": "...", "action": "read_file", "path": "relative/path.py"}
- {"thought": "...", "action": "search", "query": "text to search for"}
- {"thought": "...", "action": "replace_text", "path": "relative/path.py", "old": "exact old text", "new": "replacement text"}
- {"thought": "...", "action": "write_file", "path": "relative/path.py", "content": "full file content"}
- {"thought": "...", "action": "run_tests"}
- {"thought": "...", "action": "finish", "summary": "short summary"}

Rules:
- Read tests before editing code whenever possible.
- Prefer replace_text for small fixes.
- The old field in replace_text must exactly match the current file text.
- The provided source and test files are usually enough to fix the bug. Start from them.
- Prefer editing source files instead of modifying tests.
- Keep thoughts short.
- Return valid JSON only.
"""


PATCHER_SYSTEM_PROMPT = """You are a software repair agent working on a small Python repository.

You are given the bug report, the likely source file, and the relevant tests.
Your job is to propose one concrete code edit at a time and then use the test result to improve the next edit.

Always respond with exactly one JSON object and no markdown.

Allowed actions:
- {"thought": "...", "action": "replace_text", "path": "relative/path.py", "old": "exact old text", "new": "replacement text"}
- {"thought": "...", "action": "write_file", "path": "relative/path.py", "content": "full file content"}
- {"thought": "...", "action": "finish", "summary": "short summary"}

Rules:
- Focus on the source file, not the tests.
- Prefer replace_text for small fixes.
- The old field in replace_text must exactly match the current file text.
- Keep thoughts short.
- Return valid JSON only.
"""


def extract_first_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found in model output")

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                payload = text[start : idx + 1]
                parsed = json.loads(payload)
                if not isinstance(parsed, dict):
                    raise ValueError("Model JSON payload must be an object")
                return parsed
    raise ValueError("Unterminated JSON object in model output")


def validate_action(action: dict[str, Any]) -> dict[str, Any]:
    if "action" not in action:
        raise ValueError("Missing 'action' field")
    name = str(action["action"]).strip()
    allowed = {"read_file", "search", "replace_text", "write_file", "run_tests", "finish"}
    if name not in allowed:
        raise ValueError(f"Unsupported action: {name}")

    required_fields = {
        "read_file": {"path"},
        "search": {"query"},
        "replace_text": {"path", "old", "new"},
        "write_file": {"path", "content"},
        "run_tests": set(),
        "finish": {"summary"},
    }
    missing = required_fields[name] - set(action)
    if missing:
        raise ValueError(f"Missing required field(s) for {name}: {sorted(missing)}")
    action["action"] = name
    return action


def numbered_text(text: str) -> str:
    return "\n".join(f"{idx + 1:4d}: {line}" for idx, line in enumerate(text.splitlines()))


def render_tree(root: Path) -> str:
    parts: list[str] = []
    for path in sorted(root.rglob("*")):
        if any(piece.startswith(".") for piece in path.relative_to(root).parts):
            continue
        if path.is_dir():
            continue
        parts.append(str(path.relative_to(root)))
    return "\n".join(parts)


@dataclass
class RepoTask:
    task_id: str
    repo_dir: Path
    prompt: str
    test_command: str
    likely_files: list[str]


class RepoWorkspace:
    def __init__(self, root: Path, test_command: str, *, test_timeout: int = 30):
        self.root = root
        self.test_command = test_command
        self.test_timeout = test_timeout

    def _resolve_path(self, relative_path: str) -> Path:
        path = (self.root / relative_path).resolve()
        if self.root.resolve() not in path.parents and path != self.root.resolve():
            raise ValueError(f"Path escapes workspace: {relative_path}")
        return path

    def read_file(self, relative_path: str) -> dict[str, Any]:
        path = self._resolve_path(relative_path)
        if not path.exists():
            return {"ok": False, "error": f"File not found: {relative_path}"}
        text = path.read_text()
        return {"ok": True, "path": relative_path, "content": numbered_text(text)}

    def search(self, query: str) -> dict[str, Any]:
        pattern = re.compile(re.escape(query))
        hits: list[dict[str, Any]] = []
        for path in sorted(self.root.rglob("*.py")):
            if any(piece.startswith(".") for piece in path.relative_to(self.root).parts):
                continue
            for idx, line in enumerate(path.read_text().splitlines(), start=1):
                if pattern.search(line):
                    hits.append(
                        {
                            "path": str(path.relative_to(self.root)),
                            "line": idx,
                            "text": line,
                        }
                    )
            if len(hits) >= 20:
                break
        return {"ok": True, "query": query, "hits": hits[:20]}

    def replace_text(self, relative_path: str, old: str, new: str) -> dict[str, Any]:
        path = self._resolve_path(relative_path)
        if not path.exists():
            return {"ok": False, "error": f"File not found: {relative_path}"}
        text = path.read_text()
        count = text.count(old)
        if count == 0:
            return {"ok": False, "error": "Old text was not found exactly in the file"}
        updated = text.replace(old, new, 1)
        path.write_text(updated)
        return {"ok": True, "path": relative_path, "replaced_occurrences": 1, "available_matches": count}

    def write_file(self, relative_path: str, content: str) -> dict[str, Any]:
        path = self._resolve_path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return {"ok": True, "path": relative_path, "bytes_written": len(content.encode("utf-8"))}

    def run_tests(self) -> dict[str, Any]:
        started = time.time()
        try:
            result = subprocess.run(
                self.test_command,
                shell=True,
                cwd=self.root,
                text=True,
                capture_output=True,
                timeout=self.test_timeout,
                env={**os.environ, "PYTHONPATH": str(self.root / "src")},
            )
            output = (result.stdout or "") + (("\n" + result.stderr) if result.stderr else "")
            return {
                "ok": result.returncode == 0,
                "returncode": result.returncode,
                "output": output[-8000:],
                "elapsed_seconds": time.time() - started,
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "ok": False,
                "returncode": -1,
                "output": (exc.stdout or "") + (("\n" + exc.stderr) if exc.stderr else ""),
                "elapsed_seconds": time.time() - started,
                "error": f"Timed out after {self.test_timeout}s",
            }


class LocalHFActionModel:
    def __init__(
        self,
        *,
        model_path: str,
        layer_spec: str | None = None,
        device: str = "mps",
        dtype: str = "float16",
        max_new_tokens: int = 256,
        trust_remote_code: bool = True,
        local_files_only: bool = False,
    ):
        self.model_path = model_path
        self.layer_spec = layer_spec or ""
        self.device = self._normalize_device(device)
        self.dtype = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }[dtype]
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            dtype=self.dtype,
        )
        if self.device != "cpu":
            base_model = base_model.to(self.device)
        base_model.eval()

        self.base_model = base_model
        if self.layer_spec:
            num_layers = int(getattr(base_model.config, "num_hidden_layers"))
            layer_indices = normalize_to_layers(num_layers, self.layer_spec)
            self.model = build_model_with_layers(base_model, layer_indices)
        else:
            self.model = base_model

    def _normalize_device(self, raw: str) -> str:
        if raw == "auto":
            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return raw

    def close(self) -> None:
        del self.model
        del self.base_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    def query(self, messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
        prompt = apply_chat_template_fallback(
            self.tokenizer,
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        started = time.time()
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        text = strip_thinking(text)
        meta = {
            "elapsed_seconds": time.time() - started,
            "prompt_chars": len(prompt),
            "completion_chars": len(text),
        }
        return text, meta


def format_observation(observation: dict[str, Any]) -> str:
    return "Observation:\n" + json.dumps(observation, indent=2, ensure_ascii=True)


def initial_user_prompt(task: RepoTask) -> str:
    likely = "\n".join(f"- {path}" for path in task.likely_files)
    previews: list[str] = []
    for relative_path in task.likely_files:
        path = task.repo_dir / relative_path
        if path.exists():
            previews.append(f"File: {relative_path}\n{numbered_text(path.read_text())}")
    return (
        f"{task.prompt}\n\n"
        f"Repository files:\n{render_tree(task.repo_dir)}\n\n"
        f"Likely relevant files:\n{likely}\n\n"
        f"Initial file contents:\n\n{chr(10).join(previews)}\n\n"
        f"Run tests with: {task.test_command}\n"
    )


def initial_repair_prompt(task: RepoTask) -> str:
    previews: list[str] = []
    for relative_path in task.likely_files:
        path = task.repo_dir / relative_path
        if path.exists():
            previews.append(f"File: {relative_path}\n{numbered_text(path.read_text())}")
    return (
        f"{task.prompt}\n\n"
        "Propose the smallest source-code fix that should make the tests pass.\n\n"
        f"{chr(10).join(previews)}\n"
    )


def execute_action(workspace: RepoWorkspace, action: dict[str, Any]) -> dict[str, Any]:
    name = action["action"]
    if name == "read_file":
        return workspace.read_file(str(action["path"]))
    if name == "search":
        return workspace.search(str(action["query"]))
    if name == "replace_text":
        return workspace.replace_text(str(action["path"]), str(action["old"]), str(action["new"]))
    if name == "write_file":
        return workspace.write_file(str(action["path"]), str(action["content"]))
    if name == "run_tests":
        return workspace.run_tests()
    if name == "finish":
        return {"ok": True, "finished": True, "summary": str(action["summary"])}
    raise ValueError(f"Unhandled action: {name}")


def run_repo_task(
    *,
    model: LocalHFActionModel,
    task: RepoTask,
    work_dir: Path,
    step_limit: int = 12,
    test_timeout: int = 30,
) -> dict[str, Any]:
    workspace = RepoWorkspace(work_dir, task.test_command, test_timeout=test_timeout)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": initial_user_prompt(task)},
    ]
    steps: list[dict[str, Any]] = []
    parse_errors = 0
    tool_errors = 0
    exit_status = "step_limit"
    final_summary = ""
    started = time.time()

    for step_idx in range(step_limit):
        raw_text, query_meta = model.query(messages)
        step_record: dict[str, Any] = {
            "step": step_idx + 1,
            "raw_model_output": raw_text,
            "query_meta": query_meta,
        }
        try:
            action = validate_action(extract_first_json_object(raw_text))
            step_record["action"] = action
        except Exception as exc:
            parse_errors += 1
            observation = {"ok": False, "error": f"Invalid action format: {exc}"}
            step_record["observation"] = observation
            messages.append({"role": "assistant", "content": raw_text})
            messages.append({"role": "user", "content": format_observation(observation)})
            steps.append(step_record)
            continue

        observation = execute_action(workspace, action)
        if not observation.get("ok", False):
            tool_errors += 1
        step_record["observation"] = observation
        messages.append({"role": "assistant", "content": raw_text})
        if action["action"] == "finish":
            exit_status = "finished"
            final_summary = str(action["summary"])
            steps.append(step_record)
            break
        messages.append({"role": "user", "content": format_observation(observation)})
        steps.append(step_record)

    final_test = workspace.run_tests()
    success = bool(final_test.get("ok", False))
    if success:
        exit_status = "submitted_success" if exit_status == "finished" else "step_limit_success"

    return {
        "task_id": task.task_id,
        "success": success,
        "exit_status": exit_status,
        "summary": final_summary,
        "steps": steps,
        "parse_errors": parse_errors,
        "tool_errors": tool_errors,
        "runtime_seconds": time.time() - started,
        "final_test": final_test,
    }


def run_repo_repair_task(
    *,
    model: LocalHFActionModel,
    task: RepoTask,
    work_dir: Path,
    step_limit: int = 6,
    test_timeout: int = 30,
) -> dict[str, Any]:
    workspace = RepoWorkspace(work_dir, task.test_command, test_timeout=test_timeout)
    source_path = next((path for path in task.likely_files if path.startswith("src/")), task.likely_files[0])
    messages = [
        {"role": "system", "content": PATCHER_SYSTEM_PROMPT},
        {"role": "user", "content": initial_repair_prompt(task)},
    ]
    steps: list[dict[str, Any]] = []
    parse_errors = 0
    tool_errors = 0
    exit_status = "step_limit"
    final_summary = ""
    started = time.time()

    for step_idx in range(step_limit):
        raw_text, query_meta = model.query(messages)
        step_record: dict[str, Any] = {
            "step": step_idx + 1,
            "raw_model_output": raw_text,
            "query_meta": query_meta,
        }
        try:
            action = validate_action(extract_first_json_object(raw_text))
            if action["action"] not in {"replace_text", "write_file", "finish"}:
                raise ValueError("Only replace_text, write_file, and finish are allowed in repair mode")
            step_record["action"] = action
        except Exception as exc:
            parse_errors += 1
            observation = {"ok": False, "error": f"Invalid patch format: {exc}"}
            step_record["observation"] = observation
            messages.append({"role": "assistant", "content": raw_text})
            messages.append({"role": "user", "content": format_observation(observation)})
            steps.append(step_record)
            continue

        if action["action"] == "finish":
            exit_status = "finished"
            final_summary = str(action["summary"])
            steps.append(step_record)
            break

        observation = execute_action(workspace, action)
        if not observation.get("ok", False):
            tool_errors += 1
            current_source = workspace.read_file(source_path)
            observation = {
                **observation,
                "current_source": current_source.get("content", ""),
            }
            step_record["observation"] = observation
            messages.append({"role": "assistant", "content": raw_text})
            messages.append({"role": "user", "content": format_observation(observation)})
            steps.append(step_record)
            continue

        test_result = workspace.run_tests()
        observation = {
            "edit_result": observation,
            "test_result": test_result,
            "current_source": workspace.read_file(source_path).get("content", ""),
        }
        step_record["observation"] = observation
        messages.append({"role": "assistant", "content": raw_text})
        messages.append({"role": "user", "content": format_observation(observation)})
        steps.append(step_record)
        if test_result.get("ok", False):
            exit_status = "submitted_success"
            break

    final_test = workspace.run_tests()
    success = bool(final_test.get("ok", False))
    if success and exit_status != "submitted_success":
        exit_status = "submitted_success"

    return {
        "task_id": task.task_id,
        "success": success,
        "exit_status": exit_status,
        "summary": final_summary,
        "steps": steps,
        "parse_errors": parse_errors,
        "tool_errors": tool_errors,
        "runtime_seconds": time.time() - started,
        "final_test": final_test,
    }


def load_repo_tasks(manifest_path: Path) -> list[RepoTask]:
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    tasks: list[RepoTask] = []
    for row in manifest["tasks"]:
        tasks.append(
            RepoTask(
                task_id=str(row["task_id"]),
                repo_dir=(root / row["repo_dir"]).resolve(),
                prompt=str(row["prompt"]),
                test_command=str(row.get("test_command", "PYTHONPATH=src pytest -q")),
                likely_files=[str(path) for path in row.get("likely_files", [])],
            )
        )
    return tasks


def copy_task_repo(task: RepoTask, destination_root: Path) -> Path:
    target = destination_root / task.task_id
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(task.repo_dir, target)
    return target


def condition_label(condition: dict[str, Any]) -> str:
    return str(condition["name"])


def condition_layer_spec(condition: dict[str, Any]) -> str:
    return str(condition.get("layer_spec", ""))


def condition_metadata(condition: dict[str, Any]) -> dict[str, Any]:
    return {
        "extra_layers": int(condition.get("extra_layers", 0)),
        "overhead_fraction": float(condition.get("overhead_fraction", 0.0)),
        "layer_spec": condition.get("layer_spec", layer_spec_string([])),
    }
