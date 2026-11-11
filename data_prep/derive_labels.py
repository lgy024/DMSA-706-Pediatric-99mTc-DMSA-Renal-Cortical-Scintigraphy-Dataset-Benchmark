#!/usr/bin/env python
"""Derive T2 labels from report conclusions and report the label distribution."""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dmsa706.io import load_records, write_jsonl  # noqa: E402
from dmsa706.labels import DIAGNOSIS_CATEGORIES, derive_diagnosis_with_reason  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--include-reason",
        action="store_true",
        help="add diagnosis_reason (matched keyword) to each record",
    )
    args = parser.parse_args()

    records = load_records(args.records)
    counts = Counter()
    for record in records:
        label, reason = derive_diagnosis_with_reason(record.get("conclusion", ""))
        record["diagnosis"] = label
        if args.include_reason:
            record["diagnosis_reason"] = reason
        counts[label or "__unlabelled__"] += 1

    write_jsonl(args.out, records)

    print(f"records: {len(records)}")
    known = [c["key"] for c in DIAGNOSIS_CATEGORIES]
    for key in known + ["__unlabelled__"]:
        if counts.get(key):
            print(f"  {key:24s} {counts[key]:5d}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
