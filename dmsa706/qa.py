"""Five clinical question categories (T1) and report-draft assembly.

The questions are fixed. Answers are concatenated in this order to form the
report draft; generated and reference answers are aligned by ``qid``.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

QUESTION_CATEGORIES: List[Dict] = [
    {
        "qid": 1,
        "key": "modality_anatomy",
        "question": "本次检查的影像方式/类型与涉及的解剖部位是什么？",
        "role": "examination context",
    },
    {
        "qid": 2,
        "key": "renal_morphology",
        "question": "双肾在位置、大小、外形和形态方面的表现如何？",
        "role": "morphology description",
    },
    {
        "qid": 3,
        "key": "cortical_uptake",
        "question": "皮质放射性显像剂分布模式是什么？",
        "role": "cortical findings",
    },
    {
        "qid": 4,
        "key": "collecting_system",
        "question": "集合系统与输尿管相关征象是什么？",
        "role": "collecting-system statements",
    },
    {
        "qid": 5,
        "key": "impression",
        "question": "根据检查结果，双肾功能和结构的综合结论是什么？",
        "role": "diagnostic impression",
    },
]

NUM_QUESTIONS = len(QUESTION_CATEGORIES)

QID_TO_QUESTION = {c["qid"]: c["question"] for c in QUESTION_CATEGORIES}
QID_TO_KEY = {c["qid"]: c["key"] for c in QUESTION_CATEGORIES}
QUESTION_TO_QID = {c["question"]: c["qid"] for c in QUESTION_CATEGORIES}


def questions() -> List[str]:
    return [c["question"] for c in QUESTION_CATEGORIES]


def empty_answers() -> List[Dict[str, str]]:
    return [
        {"qid": c["qid"], "question": c["question"], "answer": ""}
        for c in QUESTION_CATEGORIES
    ]


def parse_chat_messages(messages: Iterable[dict]) -> List[Dict[str, str]]:
    """Convert a user/assistant message list into qid/question/answer pairs."""
    pairs: List[Dict[str, str]] = []
    pending: Optional[str] = None
    qid = 0
    for message in messages:
        role = message.get("role")
        content = (message.get("content") or "").replace("<image>", "").strip()
        if role == "user":
            pending = content
        elif role == "assistant" and pending is not None:
            qid += 1
            pairs.append({"qid": qid, "question": pending, "answer": content})
            pending = None
    return pairs


def normalize_qa_items(qa: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    """Ensure every item has an integer ``qid``.

    Legacy files may supply only ``question``/``answer``; the question text is
    matched against the fixed templates, falling back to position.
    """
    normalized = []
    for index, item in enumerate(qa):
        qid = item.get("qid")
        if qid is None:
            qid = QUESTION_TO_QID.get(item.get("question", ""), index + 1)
        normalized.append(
            {
                "qid": int(qid),
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
            }
        )
    normalized.sort(key=lambda item: item["qid"])
    return normalized


def assemble_answers(qa: Iterable[Dict[str, str]]) -> str:
    """Concatenate non-empty answers in qid order for text scoring."""
    ordered = sorted(qa, key=lambda x: int(x.get("qid", 0)))
    return "".join((item.get("answer") or "").strip() for item in ordered)


def format_report_draft(qa: Iterable[Dict[str, str]]) -> str:
    """Human-readable draft, e.g. ``一、...\\n二、...``."""
    numerals = "一二三四五六七八九十"
    ordered = sorted(qa, key=lambda x: int(x.get("qid", 0)))
    lines = []
    for i, item in enumerate(ordered):
        answer = (item.get("answer") or "").strip()
        if not answer:
            continue
        prefix = numerals[i] if i < len(numerals) else str(i + 1)
        lines.append(f"{prefix}、{answer}")
    return "\n".join(lines)


def qa_from_sections(findings: str, conclusion: str) -> List[Dict[str, str]]:
    """Rule-based construction of the five answers from one report.

    This is a *reference* construction that mirrors the information order of
    the clinical reports. It is the default construction; an existing QA
    annotation file can be used instead when one is available.
    """
    findings = (findings or "").strip()
    conclusion = (conclusion or "").strip()
    sentences = _split_sentences(findings)

    morphology = ""
    uptake = ""
    collecting = ""
    for sentence in sentences:
        if not morphology and any(
            key in sentence for key in ("位置", "大小", "外形", "形态", "显影")
        ):
            morphology = sentence
        elif not uptake and any(
            key in sentence for key in ("分布", "稀疏", "缺损", "摄取", "皮质")
        ):
            uptake = sentence
        if any(key in sentence for key in ("集合系统", "输尿管", "肾盂", "积水")):
            collecting = sentence

    answers = [
        "本次检查为核医学肾静态显像，涉及的解剖部位为双肾。",
        morphology or "报告中未单独描述双肾位置、大小、外形及形态。",
        uptake or "报告中未单独描述皮质放射性显像剂分布模式。",
        collecting or "报告中未提及集合系统及输尿管相关异常征象。",
        conclusion or "报告中未给出明确结论。",
    ]

    return [
        {
            "qid": c["qid"],
            "question": c["question"],
            "answer": answers[i],
        }
        for i, c in enumerate(QUESTION_CATEGORIES)
    ]


def _split_sentences(text: str) -> List[str]:
    import re

    parts = re.split(r"(?<=[。；;！？!?])", text)
    return [p.strip() for p in parts if p and p.strip()]
