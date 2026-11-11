#!/usr/bin/env python
"""Compute dataset statistics from records.jsonl."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dmsa706.io import load_records  # noqa: E402
from dmsa706.labels import DIAGNOSIS_CATEGORIES  # noqa: E402
from dmsa706.utils.statistics import distribution, length_stats  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    records = load_records(args.records)
    image_counts = [len(record.get("images", [])) for record in records]
    diagnoses = [record.get("diagnosis") or "__unlabelled__" for record in records]
    report_lengths = [
        sum(len(item.get("answer", "")) for item in record.get("qa", []))
        for record in records
    ]

    per_question = {}
    for qid in range(1, 6):
        answers = [
            item.get("answer", "")
            for record in records
            for item in record.get("qa", [])
            if int(item.get("qid", 0)) == qid
        ]
        per_question[str(qid)] = {
            "count": len(answers),
            "empty": sum(1 for answer in answers if not answer.strip()),
            "length": length_stats(answers),
        }

    stats = {
        "n_records": len(records),
        "n_images": sum(image_counts),
        "images_per_record": dict(Counter(image_counts)),
        "diagnosis": distribution(diagnoses),
        "known_categories": [item["key"] for item in DIAGNOSIS_CATEGORIES],
        "report_length": length_stats([str(value) for value in report_lengths]),
        "per_question": per_question,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2)[:1200])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
