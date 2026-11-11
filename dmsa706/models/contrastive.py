"""Image-text contrastive pretraining model."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import build_image_encoder, build_text_encoder


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class ImageTextContrastiveModel(nn.Module):
    def __init__(self, cfg, vocab_size: int):
        super().__init__()
        pretrained = bool(cfg.model.get("pretrained_encoder", True))
        self.image_encoder = build_image_encoder(
            cfg.model.get("image_encoder", "resnet18"), pretrained=pretrained
        )
        self.text_encoder = build_text_encoder(
            cfg.model.get("text_encoder", "gru"),
            vocab_size=vocab_size,
            embed_dim=int(cfg.model.get("text_embed_dim", 128)),
            hidden=int(cfg.model.get("text_hidden", 256)),
            num_layers=int(cfg.model.get("text_layers", 1)),
        )
        embed_dim = int(cfg.model.get("embed_dim", 256))
        self.image_proj = ProjectionHead(self.image_encoder.dim, embed_dim)
        self.text_proj = ProjectionHead(self.text_encoder.dim, embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    def forward(self, images, text_ids, attention_mask=None):
        image_features = F.normalize(self.image_proj(self.image_encoder(images)), dim=-1)
        text_features = F.normalize(
            self.text_proj(self.text_encoder(text_ids, attention_mask)), dim=-1
        )
        logits = self.logit_scale.exp() * image_features @ text_features.t()
        return logits, image_features, text_features


def info_nce_loss(logits: torch.Tensor) -> torch.Tensor:
    labels = torch.arange(logits.size(0), device=logits.device)
    image_to_text = F.cross_entropy(logits, labels)
    text_to_image = F.cross_entropy(logits.t(), labels)
    return (image_to_text + text_to_image) / 2.0
