"""Learning-rate schedulers with linear warmup."""

from __future__ import annotations

import math


def build_scheduler(optimizer, cfg, total_steps: int):
    import torch

    name = str(cfg.train.get("scheduler", "cosine")).lower()
    warmup_ratio = float(cfg.train.get("warmup_ratio", 0.03))
    warmup_steps = max(int(total_steps * warmup_ratio), 1)

    if name == "cosine":
        def factor(step: int) -> float:
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)

    if name == "linear":
        def factor(step: int) -> float:
            if step < warmup_steps:
                return step / warmup_steps
            return max(0.0, 1.0 - (step - warmup_steps) / max(total_steps - warmup_steps, 1))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)

    if name == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)

    raise ValueError(f"unknown scheduler: {name}")
