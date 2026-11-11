"""Bootstrap confidence intervals for per-patient scores."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 2025,
) -> Dict[str, float]:
    """Percentile bootstrap interval for the mean."""
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0, "n": 0}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        sample = rng.choice(array, size=array.size, replace=True)
        means[index] = sample.mean()
    lower, upper = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "mean": float(array.mean()),
        "lower": float(lower),
        "upper": float(upper),
        "n": int(array.size),
    }


def paired_bootstrap_diff(
    values_a: Sequence[float],
    values_b: Sequence[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 2025,
) -> Dict[str, float]:
    """Bootstrap interval for the paired difference ``mean(a - b)``."""
    a = np.asarray(list(values_a), dtype=float)
    b = np.asarray(list(values_b), dtype=float)
    if a.size != b.size or a.size == 0:
        raise ValueError("paired inputs must have the same non-zero length")
    differences = a - b
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        sample = rng.choice(differences, size=differences.size, replace=True)
        means[index] = sample.mean()
    lower, upper = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "diff": float(differences.mean()),
        "lower": float(lower),
        "upper": float(upper),
        "n": int(differences.size),
    }
