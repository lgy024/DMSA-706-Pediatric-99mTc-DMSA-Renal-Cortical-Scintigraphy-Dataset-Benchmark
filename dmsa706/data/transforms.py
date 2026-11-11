"""Image transforms for the encoder-decoder and contrastive baselines.

Horizontal flipping is deliberately excluded: on a renal scan it swaps left
and right, which would corrupt laterality. Only geometry-preserving
augmentations are used.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms


def _gamma(image: Image.Image, gamma: float = 0.8) -> Image.Image:
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.clip(array**gamma * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def equalize_contrast(image: Image.Image) -> Image.Image:
    """Global contrast equalisation for low-count scans."""
    return ImageOps.equalize(image)


def build_train_transform(image_size: int = 256, augment: bool = True, rgb: bool = False):
    operations = []
    if augment:
        operations.extend(
            [
                transforms.RandomAffine(
                    degrees=8,
                    translate=(0.05, 0.05),
                    scale=(0.95, 1.05),
                    fill=0,
                ),
                transforms.RandomApply(
                    [transforms.Lambda(lambda image: _gamma(image, 0.8))], p=0.3
                ),
                transforms.RandomApply(
                    [transforms.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.2
                ),
                transforms.RandomApply([transforms.Lambda(equalize_contrast)], p=0.2),
            ]
        )
    operations.extend(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5] if rgb else [0.5],
                std=[0.5, 0.5, 0.5] if rgb else [0.5],
            ),
        ]
    )
    return transforms.Compose(operations)


def build_eval_transform(image_size: int = 256, rgb: bool = False):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5] if rgb else [0.5],
                std=[0.5, 0.5, 0.5] if rgb else [0.5],
            ),
        ]
    )


def build_transform(cfg, train: bool):
    image_size = int(cfg.data.get("image_size", 256))
    augment = bool(cfg.data.get("augment", train))
    rgb = bool(cfg.data.get("rgb", False))
    if train:
        return build_train_transform(image_size, augment=augment, rgb=rgb)
    return build_eval_transform(image_size, rgb=rgb)
