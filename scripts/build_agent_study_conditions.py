#!/usr/bin/env python3
"""Build a baseline-vs-RYS condition manifest for software-agent studies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.layer_config import expand_single_block, layer_spec_string, parse_blocks_string


def parse_block_specs(specs: list[str]) -> list[tuple[int, int]]:
    blocks: list[tuple[int, int]] = []
    for spec in specs:
        parsed = parse_blocks_string(spec)
        if len(parsed) != 1:
            raise ValueError(
                f"Each --block must contain exactly one block specification, got: {spec}"
            )
        blocks.append(parsed[0])
    return blocks


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an agent-study condition manifest")
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--base-model-id", required=True, help="Model id or checkpoint label")
    parser.add_argument(
        "--block",
        action="append",
        default=[],
        help="Single block like '30,35'. Repeat flag for multiple RYS conditions.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON path for the condition manifest",
    )
    args = parser.parse_args()

    conditions = [
        {
            "name": "baseline",
            "model_id": args.base_model_id,
            "layer_spec": layer_spec_string(list(range(args.num_layers))),
            "extra_layers": 0,
            "overhead_fraction": 0.0,
        }
    ]

    for block in parse_block_specs(args.block):
        layers = expand_single_block(args.num_layers, block)
        extra_layers = len(layers) - args.num_layers
        conditions.append(
            {
                "name": f"rys_{block[0]}_{block[1]}",
                "model_id": args.base_model_id,
                "block": list(block),
                "layer_spec": layer_spec_string(layers),
                "extra_layers": extra_layers,
                "overhead_fraction": extra_layers / args.num_layers,
            }
        )

    payload = {
        "base_model_id": args.base_model_id,
        "num_layers": args.num_layers,
        "conditions": conditions,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {len(conditions)} conditions to {out_path}")


if __name__ == "__main__":
    main()
