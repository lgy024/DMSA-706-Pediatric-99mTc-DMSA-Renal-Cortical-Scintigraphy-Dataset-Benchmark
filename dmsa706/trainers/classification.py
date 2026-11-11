"""Training loop for the T2 encoder classifiers."""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dmsa706.utils.checkpoint import save_checkpoint
from dmsa706.utils.logging import get_logger


class ClassificationTrainer:
    def __init__(self, cfg, model, train_dataset, val_dataset=None, sampler=None):
        self.cfg = cfg
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.sampler = sampler
        self.logger = get_logger("dmsa706.classifier")

    def _device(self):
        requested = self.cfg.train.get("device", "cuda")
        if requested == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(requested)

    def _loader(self, dataset, shuffle: bool):
        sampler = self.sampler if shuffle else None
        return DataLoader(
            dataset,
            batch_size=int(self.cfg.train.batch_size),
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=int(self.cfg.train.get("num_workers", 0)),
            pin_memory=torch.cuda.is_available(),
        )

    def _run_epoch(self, loader, optimizer=None):
        training = optimizer is not None
        self.model.train(training)
        device = self._device()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = self.model(images)
            loss = F.cross_entropy(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), float(self.cfg.train.max_grad_norm)
                )
                optimizer.step()
            total_loss += loss.item()
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.numel()
        return total_loss / max(len(loader), 1), correct / max(total, 1)

    def train(self):
        device = self._device()
        self.model.to(device)
        train_loader = self._loader(self.train_dataset, shuffle=True)
        val_loader = self._loader(self.val_dataset, shuffle=False) if self.val_dataset else None

        from dmsa706.trainers.optim import build_optimizer

        optimizer = build_optimizer(self.model, self.cfg)
        for epoch in range(1, int(self.cfg.train.epochs) + 1):
            train_loss, train_acc = self._run_epoch(train_loader, optimizer)
            message = f"epoch {epoch} train_loss={train_loss:.4f} train_acc={train_acc:.3f}"
            if val_loader is not None:
                val_loss, val_acc = self._run_epoch(val_loader)
                message += f" val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
            self.logger.info(message)
            save_checkpoint(
                os.path.join(self.cfg.train.save_dir, f"epoch_{epoch}.pt"),
                self.model,
                optimizer,
                epoch=epoch,
            )
        save_checkpoint(
            os.path.join(self.cfg.train.save_dir, "last.pt"),
            self.model,
            optimizer,
            epoch=int(self.cfg.train.epochs),
        )
        return self.model

    @torch.no_grad()
    def predict(self, dataset):
        self.model.eval()
        device = self._device()
        loader = self._loader(dataset, shuffle=False)
        predictions = []
        for images, _ in loader:
            logits = self.model(images.to(device))
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
        return predictions
