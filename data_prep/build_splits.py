#!/usr/bin/env python
"""Create patient-level splits for one or more cohort sizes."""

from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dmsa706.io import load_records  # noqa: E402
from dmsa706.splits import make_cohort_splits, partition_sizes, save_splits  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, help="records.jsonl")
    parser.add_argument(
        "--n",
        default="250,500,706",
        help="comma-separated cohort sizes (default: 250,500,706)",
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--nested", action="store_true", help="smaller cohorts are subsets")
    parser.add_argument("--out", required=True, help="output splits.json")
    args = parser.parse_args()

    records = load_records(args.records)
    patient_ids = [record["patient_id"] for record in records]
    if len(set(patient_ids)) != len(patient_ids):
        raise SystemExit("duplicate patient ids in records; check the input")

    sizes = [int(value) for value in args.n.split(",") if value.strip()]

    base_order = None
    if args.nested:
        base_order = list(patient_ids)
        random.Random(args.seed).shuffle(base_order)

    all_splits = {}
    for n in sizes:
        split = make_cohort_splits(
            patient_ids, n, seed=args.seed, nested=args.nested, base_order=base_order
        )
        all_splits[str(n)] = split
        expected = partition_sizes(n)
        actual = {key: len(split[key]) for key in ("train", "val", "test")}
        print(f"N={n}: {actual} (expected {expected})")

    save_splits(args.out, all_splits)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
