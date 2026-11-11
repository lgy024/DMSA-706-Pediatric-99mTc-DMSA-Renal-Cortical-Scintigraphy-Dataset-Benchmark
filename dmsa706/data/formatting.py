"""Model-specific handling of image placeholders in chat messages."""

from __future__ import annotations

from typing import Dict, List, Sequence


def format_messages(
    messages: Sequence[Dict],
    images: Sequence,
    image_style: str = "llava",
) -> List[Dict]:
    """Attach images to the first user turn in the style the processor expects.

    ``qwen`` uses content blocks; the other backends use textual placeholders.
    The dataset itself stays model-agnostic and stores plain text messages.
    """
    if not images:
        return list(messages)

    if image_style == "qwen":
        formatted = []
        attached = False
        for message in messages:
            if message["role"] == "user" and not attached:
                content = [{"type": "image", "image": image} for image in images]
                content.append({"type": "text", "text": message["content"]})
                formatted.append({"role": "user", "content": content})
                attached = True
            else:
                formatted.append(dict(message))
        return formatted

    if image_style in {"llava", "internvl", "biomedgpt"}:
        formatted = []
        attached = False
        prefix = "<image>\n" * len(images)
        for message in messages:
            if message["role"] == "user" and not attached:
                formatted.append(
                    {"role": "user", "content": prefix + message["content"]}
                )
                attached = True
            else:
                formatted.append(dict(message))
        return formatted

    return list(messages)
