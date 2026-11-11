#!/usr/bin/env python
"""Merge T2 predictions from several models."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dmsa706.io import read_jsonl  # noqa: E402
from dmsa706.labels import CATEGORY_KEYS  # noqa: E402
from dmsa706.utils.ensemble import merge_predictions  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--method", default="majority", choices=["majority", "average"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    per_model = []
    for item in args.input:
        _, path = item.split("=", 1)
        per_model.append({record["patient_id"]: record for record in read_jsonl(path)})

    merged = merge_predictions(per_model, method=args.method)
    for record in merged:
        if record.get("diagnosis") in {"0", "1", "2", "3", "4", "5", "6"}:
            record["diagnosis"] = CATEGORY_KEYS[int(record["diagnosis"])]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        for record in merged:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"merged {len(merged)} patients -> {args.out}")


if __name__ == "__main__":
    main()
