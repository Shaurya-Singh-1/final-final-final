#!/usr/bin/env python3
"""Create a fixed SWE-bench-style manifest of exact instance IDs."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a fixed SWE-bench manifest")
    parser.add_argument("--subset", default="lite", help="Dataset alias or full dataset name")
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument("--filter", dest="filter_spec", default="", help="Regex filter over instance ids")
    parser.add_argument("--slice", dest="slice_spec", default="", help="Python slice spec like 0:25")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle deterministically before slicing")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    parser.add_argument(
        "--instance-id",
        action="append",
        default=[],
        help="Explicit instance id. Repeat to avoid downloading a dataset.",
    )
    parser.add_argument("--output", required=True, help="Output manifest path")
    return parser.parse_args()


def apply_slice(values: list[str], slice_spec: str) -> list[str]:
    if not slice_spec:
        return values
    start, stop, step = (slice_spec.split(":") + ["", ""])[:3]
    return values[slice(
        int(start) if start else None,
        int(stop) if stop else None,
        int(step) if step else None,
    )]


def main() -> None:
    args = parse_args()

    if args.instance_id:
        instance_ids = list(args.instance_id)
        dataset_name = resolve_dataset_name(args.subset)
    else:
        import re
        from datasets import load_dataset

        dataset_name = resolve_dataset_name(args.subset)
        rows = list(load_dataset(dataset_name, split=args.split))
        instance_ids = [str(row["instance_id"]) for row in rows]
        if args.filter_spec:
            instance_ids = [iid for iid in instance_ids if re.match(args.filter_spec, iid)]
        if args.shuffle:
            random.Random(args.seed).shuffle(instance_ids)
        instance_ids = apply_slice(instance_ids, args.slice_spec)

    manifest = {
        "dataset_name": dataset_name,
        "split": args.split,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "instance_ids": instance_ids,
        "selection": {
            "subset_arg": args.subset,
            "filter_spec": args.filter_spec,
            "slice_spec": args.slice_spec,
            "shuffle": args.shuffle,
            "seed": args.seed,
            "explicit_instance_ids": bool(args.instance_id),
        },
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote manifest with {len(instance_ids)} instances to {out_path}")


if __name__ == "__main__":
    main()
