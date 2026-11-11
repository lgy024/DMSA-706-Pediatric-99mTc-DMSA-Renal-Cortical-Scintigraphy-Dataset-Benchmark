#!/usr/bin/env python
"""Calibration analysis and temperature scaling for T2 predictions."""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dmsa706.io import load_records  # noqa: E402
from dmsa706.labels import CATEGORY_KEYS  # noqa: E402
from dmsa706.metrics.calibration import (  # noqa: E402
    brier_score,
    expected_calibration_error,
    reliability_curve,
    temperature_scale,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refs", required=True)
    parser.add_argument("--preds", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    references = {record["patient_id"]: record for record in load_records(args.refs)}
    predictions = load_records(args.preds)
    label_to_id = {key: index for index, key in enumerate(CATEGORY_KEYS)}

    logits, labels, confidences = [], [], []
    for prediction in predictions:
        reference = references.get(prediction["patient_id"])
        if reference is None or "probs" not in prediction:
            continue
        truth = reference.get("diagnosis")
        if truth not in label_to_id:
            continue
        probs = np.asarray(prediction["probs"], dtype=float)
        logits.append(np.log(np.clip(probs, 1e-12, None)))
        labels.append(label_to_id[truth])
        confidences.append(float(probs.max()))

    if not logits:
        raise SystemExit("no predictions with probs and a valid reference label")

    logits = np.asarray(logits)
    labels = np.asarray(labels)
    correct = (logits.argmax(axis=1) == labels).astype(float)
    result = {
        "n": int(labels.size),
        "ece": expected_calibration_error(confidences, correct),
        "brier": brier_score(confidences, correct),
        "reliability": reliability_curve(confidences, correct),
        "temperature": temperature_scale(logits, labels),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2)[:1200])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
