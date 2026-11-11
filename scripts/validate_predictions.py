#!/usr/bin/env python
"""Validate a predictions JSONL before evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dmsa706.io import load_records  # noqa: E402
from dmsa706.labels import CATEGORY_KEYS  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refs", required=True)
    parser.add_argument("--preds", required=True)
    parser.add_argument("--task", required=True, choices=["t1", "t2"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    references = {record["patient_id"]: record for record in load_records(args.refs)}
    predictions = load_records(args.preds)

    errors = []
    seen = set()
    for prediction in predictions:
        patient_id = prediction.get("patient_id")
        if patient_id not in references:
            errors.append({"patient_id": patient_id, "error": "unknown_patient"})
            continue
        seen.add(patient_id)
        if args.task == "t1":
            qa = prediction.get("qa_pred") or []
            if qa:
                qids = sorted(int(item.get("qid", 0)) for item in qa)
                if qids != [1, 2, 3, 4, 5]:
                    errors.append(
                        {"patient_id": patient_id, "error": "qids", "found": qids}
                    )
            if not any(item.get("answer", "").strip() for item in qa) and not prediction.get("report_draft"):
                errors.append({"patient_id": patient_id, "error": "empty_prediction"})
        else:
            label = prediction.get("diagnosis")
            if label not in CATEGORY_KEYS:
                errors.append(
                    {"patient_id": patient_id, "error": "invalid_diagnosis", "found": label}
                )

    missing = sorted(set(references) - seen)
    report = {
        "n_references": len(references),
        "n_predictions": len(predictions),
        "n_missing": len(missing),
        "missing_examples": missing[:20],
        "n_errors": len(errors),
        "errors": errors[:50],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2)[:1200])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
