"""Checkpoint helpers for the CNN+LSTM and ViT+Transformer baselines."""

from __future__ import annotations

import os
from typing import Any, Optional


def save_checkpoint(
    path: str,
    model: Any,
    optimizer: Optional[Any] = None,
    scheduler: Optional[Any] = None,
    epoch: Optional[int] = None,
    extra: Optional[dict] = None,
) -> None:
    import torch

    state = {"model": model.state_dict()}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if epoch is not None:
        state["epoch"] = epoch
    if extra:
        state["extra"] = extra
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, map_location: str = "cpu") -> dict:
    import torch

    return torch.load(path, map_location=map_location)
