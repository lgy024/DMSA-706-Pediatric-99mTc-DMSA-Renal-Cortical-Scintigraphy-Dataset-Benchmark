"""Paired significance tests for model comparisons."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def mcnemar(
    y_true: Sequence[str],
    pred_a: Sequence[str],
    pred_b: Sequence[str],
) -> Dict[str, float]:
    """McNemar test for two classifiers on the same patients."""
    if not (len(y_true) == len(pred_a) == len(pred_b)):
        raise ValueError("inputs must have the same length")
    correct_a = [a == t for a, t in zip(pred_a, y_true)]
    correct_b = [b == t for b, t in zip(pred_b, y_true)]
    b = sum(1 for ca, cb in zip(correct_a, correct_b) if ca and not cb)
    c = sum(1 for ca, cb in zip(correct_a, correct_b) if cb and not ca)
    statistic = 0.0
    p_value = 1.0
    if b + c > 0:
        statistic = (abs(b - c) - 1) ** 2 / (b + c)
        try:
            from scipy.stats import chi2

            p_value = float(chi2.sf(statistic, df=1))
        except ImportError:
            p_value = float(np.exp(-statistic / 2))
    return {
        "statistic": float(statistic),
        "p_value": p_value,
        "a_only_correct": b,
        "b_only_correct": c,
    }


def permutation_test(
    values_a: Sequence[float],
    values_b: Sequence[float],
    n_perm: int = 2000,
    seed: int = 2025,
) -> Dict[str, float]:
    """Two-sided paired permutation test on the mean difference."""
    a = np.asarray(list(values_a), dtype=float)
    b = np.asarray(list(values_b), dtype=float)
    if a.size != b.size or a.size == 0:
        raise ValueError("paired inputs must have the same non-zero length")
    observed = a.mean() - b.mean()
    differences = a - b
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=differences.size)
        if abs((differences * signs).mean()) >= abs(observed):
            count += 1
    return {
        "diff": float(observed),
        "p_value": (count + 1) / (n_perm + 1),
        "n": int(differences.size),
    }
