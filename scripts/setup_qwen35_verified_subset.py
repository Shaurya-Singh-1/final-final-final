#!/usr/bin/env python3
"""Prepare a Qwen3.5 SWE-bench Verified subset pilot bundle."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent_eval.experiment import resolve_dataset_name
from src.core.layer_config import expand_single_block, layer_spec_string, parse_blocks_string


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set up a Qwen3.5 SWE-bench Verified subset pilot")
    parser.add_argument("--count", type=int, default=50, help="Number of SWE-bench Verified tasks")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic shuffle seed")
    parser.add_argument("--subset", default="verified", help="SWE-bench subset alias")
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before slicing")
    parser.add_argument(
        "--output-root",
        default="",
        help="Output folder for manifest, conditions, and route template",
    )
    parser.add_argument("--base-model-id", default="Qwen/Qwen3.5-27B", help="Base model label")
    parser.add_argument("--num-layers", type=int, default=64, help="Layer count for the base model")
    parser.add_argument(
        "--block",
        action="append",
        default=[],
        help="Repeatable RYS block like 16,20. Defaults to the two strongest probe blocks.",
    )
    return parser.parse_args()


def parse_single_block(spec: str) -> tuple[int, int]:
    parsed = parse_blocks_string(spec)
    if len(parsed) != 1:
        raise ValueError(f"Each --block must contain exactly one block, got: {spec}")
    return parsed[0]


def build_conditions(
    *, base_model_id: str, num_layers: int, blocks: list[tuple[int, int]]
) -> dict[str, object]:
    conditions: list[dict[str, object]] = [
        {
            "name": "baseline",
            "model_id": base_model_id,
            "layer_spec": layer_spec_string(list(range(num_layers))),
            "extra_layers": 0,
            "overhead_fraction": 0.0,
        }
    ]

    for block in blocks:
        layers = expand_single_block(num_layers, block)
        extra_layers = len(layers) - num_layers
        conditions.append(
            {
                "name": f"rys_{block[0]}_{block[1]}",
                "model_id": base_model_id,
                "block": list(block),
                "layer_spec": layer_spec_string(layers),
                "extra_layers": extra_layers,
                "overhead_fraction": extra_layers / num_layers,
            }
        )

    return {
        "base_model_id": base_model_id,
        "num_layers": num_layers,
        "conditions": conditions,
    }


def build_route_template(condition_names: list[str]) -> dict[str, object]:
    route_template: dict[str, object] = {}
    for offset, name in enumerate(condition_names):
        placeholder = f"REPLACE_WITH_{name.upper()}_MODEL_NAME".replace("-", "_")
        route_template[name] = {
            "config": {
                "model": {
                    "model_class": "litellm",
                    "model_name": placeholder,
                    "model_kwargs": {
                        "api_base": f"http://127.0.0.1:{8000 + offset}/v1",
                        "api_key": "EMPTY",
                        "temperature": 0.0,
                    },
                },
                "agent": {
                    "cost_limit": 0,
                },
            },
            "env": {
                "OPENAI_API_KEY": "EMPTY",
            },
        }
    return route_template


def build_manifest(
    *,
    subset: str,
    split: str,
    count: int,
    seed: int,
    shuffle: bool,
) -> dict[str, object]:
    from datasets import load_dataset

    dataset_name = resolve_dataset_name(subset)
    rows = list(load_dataset(dataset_name, split=split))
    instance_ids = [str(row["instance_id"]) for row in rows]
    if shuffle:
        random.Random(seed).shuffle(instance_ids)
    selected = instance_ids[:count]
    return {
        "dataset_name": dataset_name,
        "split": split,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "instance_ids": selected,
        "selection": {
            "subset_arg": subset,
            "slice_spec": f"0:{count}",
            "shuffle": shuffle,
            "seed": seed,
            "explicit_instance_ids": False,
        },
    }


def main() -> None:
    args = parse_args()
    block_specs = list(args.block) if args.block else ["16,20", "32,36"]
    blocks = [parse_single_block(spec) for spec in block_specs]
    output_root = (
        Path(args.output_root)
        if args.output_root
        else ROOT / "results" / "agent_study" / f"qwen35_verified_{args.count}"
    )
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    conditions_payload = build_conditions(
        base_model_id=args.base_model_id,
        num_layers=args.num_layers,
        blocks=blocks,
    )
    manifest_payload = build_manifest(
        subset=args.subset,
        split=args.split,
        count=args.count,
        seed=args.seed,
        shuffle=args.shuffle,
    )
    condition_names = [str(row["name"]) for row in conditions_payload["conditions"]]
    route_payload = build_route_template(condition_names)

    conditions_path = output_root / "conditions.json"
    manifest_path = output_root / "manifest.json"
    routes_path = output_root / "model_routes.template.json"

    conditions_path.write_text(json.dumps(conditions_payload, indent=2))
    manifest_path.write_text(json.dumps(manifest_payload, indent=2))
    routes_path.write_text(json.dumps(route_payload, indent=2))

    print(f"Wrote conditions to {conditions_path}")
    print(f"Wrote manifest to {manifest_path}")
    print(f"Wrote route template to {routes_path}")
    print()
    print("Next steps:")
    print(f"1. Edit {routes_path} so the model_name and api_base values match your served endpoints.")
    condition_hint = " / ".join(["baseline"] + [f"rys_{start}_{end}" for start, end in blocks])
    print(f"2. Launch one run per condition in parallel with --condition-name {condition_hint}.")
    print(
        "3. Use configs/agent_eval/swebench_singularity.yaml as an extra --base-config on Delta "
        "so mini-SWE-agent uses Apptainer instead of Docker."
    )


if __name__ == "__main__":
    main()
