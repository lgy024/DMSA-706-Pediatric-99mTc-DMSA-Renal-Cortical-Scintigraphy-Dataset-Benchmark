"""CNN+LSTM and ViT+Transformer baselines."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ViT_B_16_Weights, vit_b_16


class CNNEncoder(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.proj = nn.Linear(128 * 4 * 4, out_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.proj(self.features(images).flatten(1))


class CNNLSTM(nn.Module):
    def __init__(self, vocab_size: int, hidden: int = 256, embed_dim: int = 128):
        super().__init__()
        self.encoder = CNNEncoder(out_dim=hidden)
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, images: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(images)
        output, _ = self.lstm(
            self.embed(target_ids),
            (hidden.unsqueeze(0), hidden.unsqueeze(0)),
        )
        return self.head(output)

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        max_len: int = 256,
        sos_id: int = 1,
        eos_id: int = 2,
    ) -> torch.Tensor:
        self.eval()
        hidden = self.encoder(images)
        h = hidden.unsqueeze(0)
        c = hidden.unsqueeze(0)
        token = torch.full((images.size(0), 1), sos_id, dtype=torch.long, device=images.device)
        generated = []
        for _ in range(max_len):
            output, (h, c) = self.lstm(self.embed(token), (h, c))
            logits = self.head(output[:, -1])
            token = logits.argmax(dim=-1, keepdim=True)
            generated.append(token.squeeze(1))
            if (token == eos_id).all():
                break
        return torch.stack(generated, dim=1)


class ViTTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 2,
        max_len: int = 512,
        pretrained: bool = False,
    ):
        super().__init__()
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        self.vit = vit_b_16(weights=weights)
        self.vit.heads = nn.Identity()
        self.memory_proj = nn.Linear(768, d_model)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.position = nn.Embedding(max_len, d_model)
        layer = nn.TransformerDecoderLayer(d_model, nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def _memory(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        return self.memory_proj(self.vit(images)).unsqueeze(1)

    def forward(self, images: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        memory = self._memory(images)
        positions = torch.arange(target_ids.shape[1], device=target_ids.device).unsqueeze(0)
        target = self.embed(target_ids) + self.position(positions)
        mask = torch.triu(
            torch.ones(target_ids.shape[1], target_ids.shape[1], device=target_ids.device),
            diagonal=1,
        ).bool()
        hidden = self.decoder(target, memory, tgt_mask=mask)
        return self.head(hidden)

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        max_len: int = 256,
        sos_id: int = 1,
        eos_id: int = 2,
    ) -> torch.Tensor:
        self.eval()
        memory = self._memory(images)
        tokens = torch.full((images.size(0), 1), sos_id, dtype=torch.long, device=images.device)
        generated = []
        for _ in range(max_len):
            positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
            target = self.embed(tokens) + self.position(positions)
            mask = torch.triu(
                torch.ones(tokens.shape[1], tokens.shape[1], device=tokens.device),
                diagonal=1,
            ).bool()
            hidden = self.decoder(target, memory, tgt_mask=mask)
            next_token = self.head(hidden[:, -1]).argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
            generated.append(next_token.squeeze(1))
            if (next_token == eos_id).all():
                break
        return torch.stack(generated, dim=1)


class CNNClassifier(nn.Module):
    def __init__(self, num_classes: int, hidden: int = 256):
        super().__init__()
        self.encoder = CNNEncoder(out_dim=hidden)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(images))


class ViTClassifier(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = False):
        super().__init__()
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        self.vit = vit_b_16(weights=weights)
        self.vit.heads = nn.Identity()
        self.head = nn.Linear(768, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        return self.head(self.vit(images))


def build_seq2seq(name: str, vocab_size: int, cfg) -> nn.Module:
    if name == "cnn_lstm":
        return CNNLSTM(
            vocab_size,
            hidden=int(cfg.model.get("hidden", 256)),
            embed_dim=int(cfg.model.get("embed_dim", 128)),
        )
    if name == "vit_transformer":
        return ViTTransformer(
            vocab_size,
            d_model=int(cfg.model.get("d_model", 256)),
            nhead=int(cfg.model.get("nhead", 8)),
            num_layers=int(cfg.model.get("num_layers", 2)),
            max_len=int(cfg.train.get("max_len", 512)),
            pretrained=bool(cfg.model.get("pretrained_vit", False)),
        )
    raise ValueError(f"unknown seq2seq model: {name}")


def build_classifier(name: str, num_classes: int, cfg) -> nn.Module:
    if name == "cnn_lstm":
        return CNNClassifier(num_classes, hidden=int(cfg.model.get("hidden", 256)))
    if name == "vit_transformer":
        return ViTClassifier(num_classes, pretrained=bool(cfg.model.get("pretrained_vit", False)))
    raise ValueError(f"no encoder classifier for: {name}")
