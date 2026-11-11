"""Image preprocessing described in the paper.

Pipeline: intensity normalisation to [0, 1], optional mild denoising, and
aspect-ratio-preserving resizing to a maximum edge length (default 512 px).
Images are acquired at 256 x 256, so resizing is a no-op unless a caller asks
for a smaller edge.
"""

from __future__ import annotations

from typing import Union

import numpy as np
from PIL import Image, ImageFilter

PathLike = Union[str, "os.PathLike[str]"]


def load_image(path: PathLike, mode: str = "L") -> Image.Image:
    """Open an image and convert it to grayscale by default."""
    image = Image.open(path)
    return image.convert(mode) if mode else image


def normalize_intensity(image: Image.Image) -> np.ndarray:
    """Scale pixel values to [0, 1] using the per-image min/max."""
    array = np.asarray(image).astype(np.float32)
    low, high = float(array.min()), float(array.max())
    if high > low:
        array = (array - low) / (high - low)
    else:
        array = array / 255.0
    return array


def mild_denoise(image: Image.Image, size: int = 3) -> Image.Image:
    """Small median filter, applied only when counts are low."""
    return image.filter(ImageFilter.MedianFilter(size=size))


def resize_max_edge(image: Image.Image, max_edge: int = 512) -> Image.Image:
    """Resize so the longer edge equals ``max_edge``, preserving aspect ratio."""
    width, height = image.size
    longest = max(width, height)
    if longest <= max_edge:
        return image
    scale = max_edge / float(longest)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(size, Image.BILINEAR)


def preprocess_image(
    path: PathLike,
    max_edge: int = 512,
    denoise: bool = False,
    as_array: bool = False,
):
    """Run the full preprocessing chain for one image."""
    image = load_image(path)
    image = resize_max_edge(image, max_edge=max_edge)
    if denoise:
        image = mild_denoise(image)
    array = normalize_intensity(image)
    if as_array:
        return array
    # Convert back to uint8 for saving or for an image-capable endpoint.
    return Image.fromarray((array * 255.0).round().astype(np.uint8), mode="L")
