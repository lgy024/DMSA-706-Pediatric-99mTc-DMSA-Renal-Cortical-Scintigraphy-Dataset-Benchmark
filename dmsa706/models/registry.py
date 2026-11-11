"""Model registry used by the training and inference scripts."""

from __future__ import annotations

MODEL_NAMES = [
    "cnn_lstm",
    "vit_transformer",
    "llava",
    "qwen25_vl",
    "biomedgpt",
    "internvl25",
]

VLM_NAMES = {"llava", "qwen25_vl", "biomedgpt", "internvl25"}


def build_model(cfg, vocab_size: int | None = None):
    name = cfg.model.name

    if name == "cnn_lstm" or name == "vit_transformer":
        if vocab_size is None:
            raise ValueError("vocab_size is required for encoder-decoder models")
        from .seq2seq import build_seq2seq

        return build_seq2seq(name, vocab_size, cfg)

    if name == "llava":
        from .llava import LlavaModel

        return LlavaModel(cfg)

    if name == "qwen25_vl":
        from .qwen25_vl import Qwen25VLModel

        return Qwen25VLModel(cfg)

    if name == "internvl25":
        from .internvl import InternVLModel

        return InternVLModel(cfg)

    if name == "biomedgpt":
        from .biomedgpt import BioMedGPTModel

        return BioMedGPTModel(cfg)

    raise ValueError(f"unknown model: {name}")
