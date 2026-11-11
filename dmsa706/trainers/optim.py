"""Optimizer factory."""

from __future__ import annotations


def build_optimizer(model, cfg):
    import torch

    parameters = [p for p in model.parameters() if p.requires_grad]
    name = str(cfg.train.get("optimizer", "adamw")).lower()
    lr = float(cfg.train.lr)
    weight_decay = float(cfg.train.weight_decay)

    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=float(cfg.train.get("momentum", 0.9)),
            weight_decay=weight_decay,
        )
    raise ValueError(f"unknown optimizer: {name}")
