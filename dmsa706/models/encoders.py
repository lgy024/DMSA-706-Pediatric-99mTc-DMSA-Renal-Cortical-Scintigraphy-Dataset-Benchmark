"""Generic image and text encoders used by the contrastive baseline."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class ImageEncoder(nn.Module):
    def __init__(self, backbone, dim: int, rgb: bool = False):
        super().__init__()
        self.backbone = backbone
        self.dim = dim
        self.rgb = rgb

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.rgb and images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        return self.backbone(images)


def _adapt_first_conv(module: nn.Conv2d) -> nn.Conv2d:
    new = nn.Conv2d(
        1,
        module.out_channels,
        kernel_size=module.kernel_size,
        stride=module.stride,
        padding=module.padding,
        bias=module.bias is not None,
    )
    with torch.no_grad():
        new.weight.copy_(module.weight.sum(dim=1, keepdim=True))
    return new


def build_image_encoder(name: str = "resnet18", pretrained: bool = True) -> ImageEncoder:
    name = name.lower()
    if name in {"resnet18", "resnet34", "resnet50"}:
        weights = getattr(models, f"ResNet{name[-2:]}_Weights").DEFAULT if pretrained else None
        backbone = getattr(models, name)(weights=weights)
        dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        backbone.conv1 = _adapt_first_conv(backbone.conv1)
        return ImageEncoder(backbone, dim, rgb=False)

    if name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        backbone = models.densenet121(weights=weights)
        dim = backbone.classifier.in_features
        backbone.classifier = nn.Identity()
        backbone.features.conv0 = _adapt_first_conv(backbone.features.conv0)
        return ImageEncoder(backbone, dim, rgb=False)

    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        dim = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        return ImageEncoder(backbone, dim, rgb=True)

    if name == "vit_b_16":
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        backbone = models.vit_b_16(weights=weights)
        dim = backbone.heads.head.in_features
        backbone.heads = nn.Identity()
        return ImageEncoder(backbone, dim, rgb=True)

    raise ValueError(f"unknown image encoder: {name}")


class GRUTextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden: int = 256,
        num_layers: int = 1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.gru = nn.GRU(
            embed_dim, hidden, num_layers=num_layers, batch_first=True, bidirectional=True
        )
        self.dim = hidden * 2

    def forward(self, ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        output, _ = self.gru(self.embed(ids))
        if mask is None:
            return output.mean(dim=1)
        weights = mask.unsqueeze(-1).float()
        return (output * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)


class TransformerTextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden: int = 256,
        num_layers: int = 2,
        nhead: int = 4,
        max_len: int = 256,
        pad_id: int = 0,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.project = nn.Linear(embed_dim, hidden)
        self.position = nn.Embedding(max_len, hidden)
        layer = nn.TransformerEncoderLayer(hidden, nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.dim = hidden

    def forward(self, ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        positions = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
        hidden = self.project(self.embed(ids)) + self.position(positions)
        padding_mask = None if mask is None else ~mask.bool()
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        if mask is None:
            return hidden.mean(dim=1)
        weights = mask.unsqueeze(-1).float()
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)


def build_text_encoder(
    name: str = "gru",
    vocab_size: int = 1000,
    embed_dim: int = 128,
    hidden: int = 256,
    num_layers: int = 1,
) -> nn.Module:
    name = name.lower()
    if name == "gru":
        return GRUTextEncoder(vocab_size, embed_dim, hidden, num_layers)
    if name == "transformer":
        return TransformerTextEncoder(vocab_size, embed_dim, hidden, max(num_layers, 2))
    raise ValueError(f"unknown text encoder: {name}")
