"""Training loop for the VLM T2 classification head."""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dmsa706.data.collator import T2ImageCollator
from dmsa706.utils.logging import get_logger


class VLMClassificationTrainer:
    def __init__(self, cfg, model, train_dataset, val_dataset=None):
        self.cfg = cfg
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.logger = get_logger("dmsa706.vlm_classifier")

    def _loader(self, dataset, shuffle: bool):
        return DataLoader(
            dataset,
            batch_size=int(self.cfg.train.batch_size),
            shuffle=shuffle,
            num_workers=int(self.cfg.train.get("num_workers", 0)),
            collate_fn=T2ImageCollator(),
        )

    def _device(self):
        return next(self.model.model.parameters()).device

    def _run_epoch(self, loader, optimizer=None):
        training = optimizer is not None
        self.model.train(training)
        device = self._device()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in loader:
            labels = labels.to(device)
            logits = self.model(images)
            loss = F.cross_entropy(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    float(self.cfg.train.max_grad_norm),
                )
                optimizer.step()
            total_loss += loss.item()
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.numel()

        return total_loss / max(len(loader), 1), correct / max(total, 1)

    def train(self):
        device = self._device()
        self.model.head.to(device)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("no trainable parameters; set freeze_backbone: false")
        optimizer = torch.optim.AdamW(
            trainable,
            lr=float(self.cfg.train.lr),
            weight_decay=float(self.cfg.train.weight_decay),
        )
        train_loader = self._loader(self.train_dataset, shuffle=True)
        val_loader = self._loader(self.val_dataset, shuffle=False) if self.val_dataset else None

        for epoch in range(1, int(self.cfg.train.epochs) + 1):
            train_loss, train_acc = self._run_epoch(train_loader, optimizer)
            message = f"epoch {epoch} train_loss={train_loss:.4f} train_acc={train_acc:.3f}"
            if val_loader is not None:
                val_loss, val_acc = self._run_epoch(val_loader)
                message += f" val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
            self.logger.info(message)

        os.makedirs(self.cfg.train.save_dir, exist_ok=True)
        self.model.save(self.cfg.train.save_dir)
        self.logger.info("saved T2 head to %s", self.cfg.train.save_dir)
        return self.model

    @torch.no_grad()
    def predict(self, dataset):
        self.model.eval()
        loader = self._loader(dataset, shuffle=False)
        predictions = []
        for images, _ in loader:
            logits = self.model(images)
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
        return predictions
