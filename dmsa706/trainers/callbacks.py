"""Small training callbacks."""

from __future__ import annotations

import csv
import os
from typing import Dict, Optional

from dmsa706.utils.checkpoint import save_checkpoint


class AverageMeter:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.total += value * n
        self.count += n

    @property
    def average(self) -> float:
        return self.total / self.count if self.count else 0.0


class EarlyStopping:
    def __init__(self, patience: int = 5, mode: str = "min", min_delta: float = 0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = float("inf") if mode == "min" else -float("inf")
        self.counter = 0

    def __call__(self, score: float) -> bool:
        improved = (
            score < self.best - self.min_delta
            if self.mode == "min"
            else score > self.best + self.min_delta
        )
        if improved:
            self.best = score
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


class ModelCheckpoint:
    def __init__(self, path: str, mode: str = "min"):
        self.path = path
        self.mode = mode
        self.best = float("inf") if mode == "min" else -float("inf")

    def __call__(self, score: float, model, optimizer=None, epoch=None, extra=None) -> bool:
        improved = (
            score < self.best if self.mode == "min" else score > self.best
        )
        if improved:
            self.best = score
            save_checkpoint(self.path, model, optimizer, epoch=epoch, extra=extra)
        return improved


class CSVLogger:
    def __init__(self, path: str, fields: Optional[list] = None):
        self.path = path
        self.fields = fields or ["epoch", "train_loss", "val_loss"]
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fields)
                writer.writeheader()

    def log(self, row: Dict) -> None:
        with open(self.path, "a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fields, extrasaction="ignore")
            writer.writerow(row)
