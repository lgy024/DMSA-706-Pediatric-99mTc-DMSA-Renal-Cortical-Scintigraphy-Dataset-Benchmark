"""Diagnostic metrics: Accuracy, Macro-P, Macro-R, Macro-F1.

Macro averages are taken over categories with reference support in the current
test set. Predictions outside that set, invalid predictions and missing
predictions count as errors and as false negatives for the reference category.
A supported category that receives no predictions has precision 0.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def classification_metrics(
    y_true: Sequence[Optional[str]],
    y_pred: Sequence[Optional[str]],
    labels: Optional[Sequence[str]] = None,
) -> Dict:
    """Return accuracy, macro scores, per-class scores and a confusion matrix."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    pairs = list(zip(y_true, y_pred))
    n = len(pairs)
    accuracy = _safe_div(sum(1 for t, p in pairs if t == p), n)

    reference_counts = Counter(t for t in y_true if t is not None)
    if labels is None:
        labels = sorted(reference_counts)
    supported = [label for label in labels if reference_counts.get(label, 0) > 0]

    per_class: Dict[str, Dict[str, float]] = {}
    precision_values, recall_values, f1_values = [], [], []

    for label in supported:
        tp = sum(1 for t, p in pairs if t == label and p == label)
        fp = sum(1 for t, p in pairs if t != label and p == label)
        fn = sum(1 for t, p in pairs if t == label and p != label)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * tp, 2 * tp + fp + fn)
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        per_class[label] = {
            "support": reference_counts.get(label, 0),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    macro_p = sum(precision_values) / len(precision_values) if precision_values else 0.0
    macro_r = sum(recall_values) / len(recall_values) if recall_values else 0.0
    macro_f1 = sum(f1_values) / len(f1_values) if f1_values else 0.0

    confusion: Dict[str, Dict[str, int]] = {}
    for true_label in supported:
        row = Counter(
            (predicted if predicted is not None else "__missing__")
            for true, predicted in pairs
            if true == true_label
        )
        confusion[true_label] = dict(row)

    return {
        "accuracy": accuracy,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "n_samples": n,
        "n_supported_categories": len(supported),
        "supported_categories": supported,
        "per_class": per_class,
        "confusion": confusion,
    }
