"""Calibration metrics and temperature scaling."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def expected_calibration_error(
    probabilities: Sequence[float],
    labels: Sequence[int],
    n_bins: int = 10,
) -> float:
    """ECE for binary confidence scores."""
    probs = np.asarray(list(probabilities), dtype=float)
    labels = np.asarray(list(labels), dtype=float)
    if probs.size == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (probs > lower) & (probs <= upper)
        if mask.sum() == 0:
            continue
        confidence = probs[mask].mean()
        accuracy = labels[mask].mean()
        error += mask.mean() * abs(confidence - accuracy)
    return float(error)


def brier_score(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    probs = np.asarray(list(probabilities), dtype=float)
    labels = np.asarray(list(labels), dtype=float)
    if probs.size == 0:
        return 0.0
    return float(np.mean((probs - labels) ** 2))


def reliability_curve(
    probabilities: Sequence[float],
    labels: Sequence[int],
    n_bins: int = 10,
):
    probs = np.asarray(list(probabilities), dtype=float)
    labels = np.asarray(list(labels), dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    curve = []
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (probs > lower) & (probs <= upper)
        if mask.sum() == 0:
            continue
        curve.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "confidence": float(probs[mask].mean()),
                "accuracy": float(labels[mask].mean()),
                "count": int(mask.sum()),
            }
        )
    return curve


def temperature_scale(
    logits: Sequence[Sequence[float]],
    labels: Sequence[int],
    temperatures: Sequence[float] = tuple(np.linspace(0.5, 5.0, 46)),
) -> Dict[str, float]:
    """Grid-search a temperature that minimises negative log-likelihood."""
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels, dtype=int)
    best_temperature = 1.0
    best_nll = float("inf")
    for temperature in temperatures:
        probs = _softmax(logits / temperature)
        picked = probs[np.arange(labels.size), labels]
        nll = float(-np.log(np.clip(picked, 1e-12, None)).mean())
        if nll < best_nll:
            best_nll = nll
            best_temperature = float(temperature)
    return {"temperature": best_temperature, "nll": best_nll}
