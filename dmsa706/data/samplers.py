"""Class-balanced sampling for the T2 classifier."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Sequence

import numpy as np
from torch.utils.data import WeightedRandomSampler


def compute_class_weights(labels: Sequence[int], power: float = 1.0) -> Dict[int, float]:
    counts = Counter(labels)
    total = sum(counts.values())
    return {
        label: (total / (len(counts) * count)) ** power
        for label, count in counts.items()
    }


def build_weighted_sampler(
    labels: Sequence[int],
    power: float = 1.0,
    num_samples: int | None = None,
    seed: int = 2025,
) -> WeightedRandomSampler:
    weights = compute_class_weights(labels, power=power)
    sample_weights = [weights[label] for label in labels]
    generator = np.random.default_rng(seed)
    # WeightedRandomSampler accepts a generator only in recent torch versions;
    # set the global seed in the caller when an older torch is used.
    return WeightedRandomSampler(
        sample_weights,
        num_samples=num_samples or len(sample_weights),
        replacement=True,
        generator=generator if _supports_generator() else None,
    )


def labels_from_dataset(dataset) -> List[int]:
    return [label for _, label in getattr(dataset, "items", [])]


def _supports_generator() -> bool:
    try:
        import inspect

        from torch.utils.data import WeightedRandomSampler as sampler

        return "generator" in inspect.signature(sampler.__init__).parameters
    except Exception:
        return False
