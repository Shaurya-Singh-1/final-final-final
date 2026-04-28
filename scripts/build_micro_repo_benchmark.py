#!/usr/bin/env python3
"""Build a lightweight local repo-repair benchmark for downstream RYS agent studies."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCH_ROOT = ROOT / "benchmarks" / "micro_repo_bench"


TASKS = [
    {
        "task_id": "chunk_list_tail",
        "prompt": (
            "The chunk_list helper is dropping the final partial chunk. "
            "For example, chunking [1, 2, 3, 4, 5] into size 2 should end with [5]. "
            "Fix the bug without changing the public API."
        ),
        "likely_files": ["src/chunker.py", "tests/test_chunker.py"],
        "files": {
            "src/chunker.py": """def chunk_list(items, size):\n    if size <= 0:\n        raise ValueError(\"size must be positive\")\n\n    chunks = []\n    for idx in range(0, len(items) - size, size):\n        chunks.append(items[idx : idx + size])\n    return chunks\n""",
            "tests/test_chunker.py": """from chunker import chunk_list\n\n\ndef test_keeps_partial_tail_chunk():\n    assert chunk_list([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]\n\n\ndef test_exact_division_still_works():\n    assert chunk_list([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]\n\n\ndef test_invalid_size_raises():\n    try:\n        chunk_list([1, 2], 0)\n    except ValueError:\n        pass\n    else:\n        raise AssertionError(\"expected ValueError\")\n""",
        },
    },
    {
        "task_id": "merge_touching_intervals",
        "prompt": (
            "The interval merger should treat touching intervals as overlapping. "
            "For example, (1, 3) followed by (3, 5) should merge into (1, 5)."
        ),
        "likely_files": ["src/intervals.py", "tests/test_intervals.py"],
        "files": {
            "src/intervals.py": """def merge_intervals(intervals):\n    if not intervals:\n        return []\n\n    ordered = sorted(intervals)\n    merged = [ordered[0]]\n    for start, end in ordered[1:]:\n        last_start, last_end = merged[-1]\n        if start < last_end:\n            merged[-1] = (last_start, max(last_end, end))\n        else:\n            merged.append((start, end))\n    return merged\n""",
            "tests/test_intervals.py": """from intervals import merge_intervals\n\n\ndef test_merges_touching_intervals():\n    assert merge_intervals([(1, 3), (3, 5), (10, 12)]) == [(1, 5), (10, 12)]\n\n\ndef test_keeps_separate_intervals():\n    assert merge_intervals([(1, 2), (4, 5)]) == [(1, 2), (4, 5)]\n""",
        },
    },
    {
        "task_id": "windowed_average_last_window",
        "prompt": (
            "The moving-average helper skips the last valid window. "
            "It should include every full window of the requested size."
        ),
        "likely_files": ["src/averages.py", "tests/test_averages.py"],
        "files": {
            "src/averages.py": """def moving_average(values, window):\n    if window <= 0:\n        raise ValueError(\"window must be positive\")\n    if window > len(values):\n        return []\n\n    result = []\n    for idx in range(0, len(values) - window):\n        chunk = values[idx : idx + window]\n        result.append(sum(chunk) / window)\n    return result\n""",
            "tests/test_averages.py": """from averages import moving_average\n\n\ndef test_includes_last_window():\n    assert moving_average([1, 2, 3, 4], 2) == [1.5, 2.5, 3.5]\n\n\ndef test_large_window_returns_empty():\n    assert moving_average([1, 2], 3) == []\n""",
        },
    },
    {
        "task_id": "parse_bool_whitespace",
        "prompt": (
            "The boolean parser should ignore surrounding whitespace and accept mixed-case inputs "
            "for true and false."
        ),
        "likely_files": ["src/boolparse.py", "tests/test_boolparse.py"],
        "files": {
            "src/boolparse.py": """def parse_bool(value):\n    cleaned = value.lower()\n    if cleaned == \"true\":\n        return True\n    if cleaned == \"false\":\n        return False\n    raise ValueError(f\"unsupported boolean value: {value}\")\n""",
            "tests/test_boolparse.py": """import pytest\n\nfrom boolparse import parse_bool\n\n\ndef test_accepts_whitespace_and_case():\n    assert parse_bool(\"  TRUE  \") is True\n    assert parse_bool(\"\\nFalse\\t\") is False\n\n\ndef test_invalid_value_raises():\n    with pytest.raises(ValueError):\n        parse_bool(\"definitely\")\n""",
        },
    },
    {
        "task_id": "median_even_case",
        "prompt": (
            "The median helper returns the lower middle value for even-length inputs. "
            "It should return the average of the two middle numbers."
        ),
        "likely_files": ["src/stats_tools.py", "tests/test_stats_tools.py"],
        "files": {
            "src/stats_tools.py": """def median(values):\n    ordered = sorted(values)\n    size = len(ordered)\n    if size == 0:\n        raise ValueError(\"median() arg is an empty sequence\")\n\n    mid = size // 2\n    if size % 2 == 1:\n        return float(ordered[mid])\n    return float(ordered[mid - 1])\n""",
            "tests/test_stats_tools.py": """from stats_tools import median\n\n\ndef test_even_length_uses_average():\n    assert median([10, 2, 8, 4]) == 6.0\n\n\ndef test_odd_length_still_works():\n    assert median([3, 1, 2]) == 2.0\n""",
        },
    },
    {
        "task_id": "slugify_collapse_dashes",
        "prompt": (
            "The slugify helper should collapse repeated dashes and strip them from the ends. "
            "Punctuation should not leave leading or trailing separators behind."
        ),
        "likely_files": ["src/text_tools.py", "tests/test_text_tools.py"],
        "files": {
            "src/text_tools.py": """import re\n\n\ndef slugify(text):\n    slug = text.lower().replace(\" \", \"-\")\n    slug = re.sub(r\"[^a-z0-9-]\", \"\", slug)\n    return slug\n""",
            "tests/test_text_tools.py": """from text_tools import slugify\n\n\ndef test_collapse_and_strip_dashes():\n    assert slugify(\"  Hello,   World!!  \") == \"hello-world\"\n\n\ndef test_internal_words_remain_separated():\n    assert slugify(\"A  B  C\") == \"a-b-c\"\n""",
        },
    },
    {
        "task_id": "roman_subtractive_pairs",
        "prompt": (
            "The Roman numeral parser fails on subtractive pairs like IV, IX, XL, and CM. "
            "Fix the implementation without changing the function signature."
        ),
        "likely_files": ["src/roman.py", "tests/test_roman.py"],
        "files": {
            "src/roman.py": """VALUES = {\n    \"I\": 1,\n    \"V\": 5,\n    \"X\": 10,\n    \"L\": 50,\n    \"C\": 100,\n    \"D\": 500,\n    \"M\": 1000,\n}\n\n\ndef roman_to_int(text):\n    total = 0\n    for char in text:\n        total += VALUES[char]\n    return total\n""",
            "tests/test_roman.py": """from roman import roman_to_int\n\n\ndef test_subtractive_pairs():\n    assert roman_to_int(\"IV\") == 4\n    assert roman_to_int(\"IX\") == 9\n    assert roman_to_int(\"XL\") == 40\n    assert roman_to_int(\"CM\") == 900\n\n\ndef test_regular_numerals_still_work():\n    assert roman_to_int(\"VIII\") == 8\n""",
        },
    },
    {
        "task_id": "version_sort_numeric_segments",
        "prompt": (
            "The version sort key sorts numeric segments lexicographically instead of numerically. "
            "For example, 1.10.0 should come after 1.2.0."
        ),
        "likely_files": ["src/versioning.py", "tests/test_versioning.py"],
        "files": {
            "src/versioning.py": """def version_sort_key(version):\n    return version.split(\".\")\n""",
            "tests/test_versioning.py": """from versioning import version_sort_key\n\n\ndef test_numeric_segments_sort_correctly():\n    ordered = sorted([\"1.2.0\", \"1.10.0\", \"1.3.0\"], key=version_sort_key)\n    assert ordered == [\"1.2.0\", \"1.3.0\", \"1.10.0\"]\n""",
        },
    },
]


def write_task(task: dict) -> dict:
    repo_dir = BENCH_ROOT / "tasks" / task["task_id"]
    if repo_dir.exists():
        for path in sorted(repo_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    repo_dir.mkdir(parents=True, exist_ok=True)
    for relative_path, content in task["files"].items():
        path = repo_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    (repo_dir / "README_issue.md").write_text(task["prompt"] + "\n")
    return {
        "task_id": task["task_id"],
        "repo_dir": f"tasks/{task['task_id']}",
        "prompt": task["prompt"],
        "test_command": "PYTHONPATH=src pytest -q",
        "likely_files": task["likely_files"],
    }


def main() -> None:
    BENCH_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {"tasks": [write_task(task) for task in TASKS]}
    manifest_path = BENCH_ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote benchmark manifest to {manifest_path}")


if __name__ == "__main__":
    main()
