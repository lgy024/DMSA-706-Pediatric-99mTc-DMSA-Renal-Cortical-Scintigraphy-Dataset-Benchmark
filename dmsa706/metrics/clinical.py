"""Slot-level clinical attribute metrics."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

from dmsa706.clinical import ATTRIBUTE_KEYWORDS, extract_attributes, laterality_mismatch


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_clinical(
    predictions: Mapping[str, str],
    references: Mapping[str, str],
    attributes: Optional[Sequence[str]] = None,
) -> Dict:
    """Precision, recall and F1 for each clinical attribute.

    ``predictions`` and ``references`` map patient id to report text.
    """
    attributes = attributes or list(ATTRIBUTE_KEYWORDS)
    counts = {
        name: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for name in attributes
    }
    laterality_errors = 0
    n_pairs = 0

    for patient_id, reference in references.items():
        prediction = predictions.get(patient_id, "")
        if not reference or not prediction:
            continue
        n_pairs += 1
        pred_attrs = extract_attributes(prediction)
        ref_attrs = extract_attributes(reference)
        if laterality_mismatch(prediction, reference):
            laterality_errors += 1
        for name in attributes:
            predicted, truth = pred_attrs[name], ref_attrs[name]
            counts[name]["support"] += int(truth)
            counts[name]["tp"] += int(predicted and truth)
            counts[name]["fp"] += int(predicted and not truth)
            counts[name]["fn"] += int(not predicted and truth)

    per_attribute = {}
    f1_values = []
    for name, values in counts.items():
        precision = _safe_div(values["tp"], values["tp"] + values["fp"])
        recall = _safe_div(values["tp"], values["tp"] + values["fn"])
        f1 = _safe_div(2 * values["tp"], 2 * values["tp"] + values["fp"] + values["fn"])
        per_attribute[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": values["support"],
        }
        f1_values.append(f1)

    return {
        "n_patients": n_pairs,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "laterality_accuracy": 1.0 - _safe_div(laterality_errors, n_pairs),
        "laterality_errors": laterality_errors,
        "per_attribute": per_attribute,
    }
