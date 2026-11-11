"""Prompt builders for T1 (question-aligned report) and T2 (diagnosis)."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

from .labels import DIAGNOSIS_CATEGORIES
from .qa import QUESTION_CATEGORIES, QID_TO_QUESTION

SYSTEM_PROMPT = (
    "你是一名儿童核医学影像报告助手。请仅依据提供的平面肾皮质显像图像回答，"
    "不要编造报告中未出现的信息。回答使用简洁、规范的中文医学术语。"
)

DIAGNOSIS_INSTRUCTION = (
    "请根据提供的平面肾皮质显像图像，从下列诊断类别中选择一个最符合的类别，"
    "只输出类别英文键名，不要输出解释：\n"
    + "\n".join(f"- {c['key']}: {c['zh']}" for c in DIAGNOSIS_CATEGORIES)
)

# A neutral single-turn instruction used to assemble a full report in one go.
SINGLE_TURN_INSTRUCTION = "请根据提供的图像生成一份完整的肾皮质显像报告。"


def _text_part(text: str) -> Dict:
    return {"type": "text", "text": text}


def _image_part(data_url: str) -> Dict:
    return {"type": "image_url", "image_url": {"url": data_url}}


def build_t1_messages(
    history: Sequence[Dict[str, str]],
    qid: int,
    image_data_urls: Sequence[str] | None = None,
) -> List[Dict]:
    """Build a chat request for one question.

    ``history`` holds earlier ``{"qid", "question", "answer"}`` items. The
    images are attached to the first user message only; later turns rely on the
    dialogue context, matching the multi-turn protocol in the paper.
    """
    messages: List[Dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    for index, item in enumerate(history):
        if index == 0 and image_data_urls:
            content = [_text_part(item["question"])] + [
                _image_part(url) for url in image_data_urls
            ]
        else:
            content = item["question"]
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": item.get("answer", "")})

    question = QID_TO_QUESTION[qid]
    if not history and image_data_urls:
        content = [_text_part(question)] + [
            _image_part(url) for url in image_data_urls
        ]
    else:
        content = question
    messages.append({"role": "user", "content": content})
    return messages


def build_t2_messages(image_data_urls: Sequence[str]) -> List[Dict]:
    content = [_text_part(DIAGNOSIS_INSTRUCTION)] + [
        _image_part(url) for url in image_data_urls
    ]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def build_single_turn_messages(image_data_urls: Sequence[str]) -> List[Dict]:
    content = [_text_part(SINGLE_TURN_INSTRUCTION)] + [
        _image_part(url) for url in image_data_urls
    ]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def question_for(qid: int) -> str:
    return QID_TO_QUESTION[qid]
