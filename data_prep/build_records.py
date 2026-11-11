#!/usr/bin/env python
"""Build records.jsonl from an images/ + reports/ layout.

Each record contains the patient id, image paths, findings, conclusion, the
five question/answer pairs and a report-derived diagnostic label.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dmsa706.io import build_records, find_dates, redact_dates, write_jsonl  # noqa: E402


def load_labels(path: str) -> dict:
    """Read a ``patient_id,diagnosis`` CSV of clinician-verified labels."""
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        return {
            row["patient_id"].strip(): (row.get("diagnosis") or "").strip() or None
            for row in reader
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="directory of NNN_a.<ext> images")
    parser.add_argument("--reports", required=True, help="directory of NNN.docx reports")
    parser.add_argument("--out", required=True, help="output records.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="optional debug limit")
    parser.add_argument(
        "--labels",
        default=None,
        help="optional CSV (patient_id,diagnosis) with clinician-verified T2 labels; "
        "overrides the reference rules",
    )
    parser.add_argument(
        "--keep-dates",
        action="store_true",
        help="keep date-like strings in findings/conclusion (default: replace with 既往)",
    )
    args = parser.parse_args()

    records = build_records(args.images, args.reports)
    if args.limit:
        records = records[: args.limit]

    if not args.keep_dates:
        for record in records:
            record.findings = redact_dates(record.findings)
            record.conclusion = redact_dates(record.conclusion)
            for item in record.qa:
                item["answer"] = redact_dates(item["answer"])

    if args.labels:
        official = load_labels(args.labels)
        joined = 0
        for record in records:
            if record.patient_id in official:
                record.diagnosis = official[record.patient_id]
                joined += 1
        print(f"joined official labels for {joined}/{len(records)} records")
    else:
        print("using reference T2 labels; pass --labels to join an annotation file")

    write_jsonl(args.out, records)

    labels = Counter()
    empty = 0
    for record in records:
        if record.diagnosis is None:
            labels["__unlabelled__"] += 1
            empty += 1
        else:
            labels[record.diagnosis] += 1

    dated = [
        record.patient_id
        for record in records
        if find_dates(record.findings) or find_dates(record.conclusion)
    ]
    if dated:
        print(
            f"warning: {len(dated)} records contain date-like strings in the "
            "findings/conclusion (e.g. historical comparison dates): "
            + ", ".join(dated[:10])
            + (" ..." if len(dated) > 10 else "")
        )

    print(f"records: {len(records)}")
    print(f"images:  {sum(len(r.images) for r in records)}")
    print(f"unlabelled conclusions: {empty}")
    print("diagnosis distribution:")
    for key, count in labels.most_common():
        print(f"  {key:24s} {count}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
