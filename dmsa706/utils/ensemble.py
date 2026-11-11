"""Ensemble helpers for T2 predictions."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence

import numpy as np


def majority_vote(labels: Sequence[str]) -> str:
    counts = Counter(label for label in labels if label)
    if not counts:
        return ""
    return counts.most_common(1)[0][0]


def average_probabilities(probabilities: Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(probabilities, dtype=float)
    if array.ndim != 2:
        raise ValueError("probabilities must be a list of equal-length vectors")
    return array.mean(axis=0)


def merge_predictions(
    per_model: Sequence[Dict[str, dict]],
    method: str = "majority",
) -> List[dict]:
    """Merge ``patient_id -> prediction`` maps from several models."""
    patient_ids = sorted({pid for records in per_model for pid in records})
    merged = []
    for patient_id in patient_ids:
        entries = [records[patient_id] for records in per_model if patient_id in records]
        if not entries:
            continue
        if method == "average" and all("probs" in entry for entry in entries):
            probs = average_probabilities([entry["probs"] for entry in entries])
            label = str(int(probs.argmax()))
            merged.append(
                {
                    "patient_id": patient_id,
                    "images": entries[0].get("images", []),
                    "diagnosis": label,
                    "probs": probs.tolist(),
                }
            )
        else:
            labels = [entry.get("diagnosis", "") for entry in entries]
            merged.append(
                {
                    "patient_id": patient_id,
                    "images": entries[0].get("images", []),
                    "diagnosis": majority_vote(labels),
                }
            )
    return merged
