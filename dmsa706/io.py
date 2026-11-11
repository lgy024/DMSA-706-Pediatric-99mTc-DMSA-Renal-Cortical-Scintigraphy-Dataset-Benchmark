"""Readers and writers for DMSA-706 records and reports.

Supported inputs
----------------
* ``data/images/<NNN>_a.<ext>`` and ``<NNN>_b.<ext>`` (mixed extensions)
* ``data/reports/<NNN>.docx`` (Chinese structured reports)
* JSONL records produced by ``data_prep/build_records.py``
* The legacy JSON formats (``report.json``, ``report_qa_split.json``)

The report parser keeps only the *findings* (影像所见) and *conclusion*
(影像结论) sections. Header fields (examination number, dates of birth,
examination/report times, history, physician signatures) are dropped and
checked for, so a de-identification regression fails loudly instead of
silently leaking identifiers into a release.
"""

from __future__ import annotations

import json
import os
import re
import zipfile
from dataclasses import dataclass, field, asdict
from typing import Dict, Iterable, Iterator, List, Optional
from xml.etree import ElementTree as ET

NS_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W = "{%s}" % NS_W

HEADING_FINDINGS = "影像所见"
HEADING_CONCLUSION = "影像结论"
DISCLAIMER_PREFIX = "本报告仅供"

# Header fields that must never survive extraction.
FORBIDDEN_FIELDS = (
    "检查号",
    "出生日期",
    "门诊号",
    "住院号",
    "床号",
    "注射部位",
    "报告医师",
    "报告时间",
    "审核医师",
    "审核时间",
    "简要病史",
)


def normalize_heading(text: str) -> str:
    """Remove all whitespace so ``影 像 所 见`` matches ``影像所见``."""
    return re.sub(r"\s+", "", text or "")


def read_docx_paragraphs(path: str) -> List[str]:
    """Return every paragraph of a .docx in document order."""
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("word/document.xml"))
    return [
        "".join(t.text or "" for t in p.iter(W + "t"))
        for p in root.iter(W + "p")
    ]


def parse_report(paragraphs: Iterable[str]) -> Dict[str, List[str]]:
    """Split report paragraphs into findings and conclusion sections."""
    paragraphs = list(paragraphs)
    findings: List[str] = []
    conclusion: List[str] = []
    section: Optional[str] = None

    for raw in paragraphs:
        text = (raw or "").strip()
        cleaned = normalize_heading(text)
        if not cleaned:
            continue
        if cleaned == HEADING_FINDINGS:
            section = "findings"
            continue
        if cleaned == HEADING_CONCLUSION:
            section = "conclusion"
            continue
        if cleaned.startswith(DISCLAIMER_PREFIX):
            break
        if section == "findings":
            findings.append(text)
        elif section == "conclusion":
            conclusion.append(text)

    return {"findings": findings, "conclusion": conclusion}


def check_deidentified(text: str, source: str = "<text>") -> None:
    """Raise if any forbidden header field survived extraction."""
    for field_name in FORBIDDEN_FIELDS:
        if field_name in (text or ""):
            raise ValueError(
                f"de-identification check failed in {source}: found {field_name!r}"
            )


_DATE_RE = re.compile(
    r"\d{4}[/.\-]\d{1,2}[/.\-]\d{1,2}"
    r"|\d{4}\s*年\s*\d{1,2}\s*月(?:\s*\d{1,2}\s*日)?"
)


def find_dates(text: str) -> List[str]:
    """Return date-like strings, e.g. historical comparison dates.

    These are not patient identifiers on their own, but dates in a conclusion
    (``同2023/5/4检查相仿``) narrow the examination history. Review them before
    releasing reports.
    """
    return _DATE_RE.findall(text or "")


def redact_dates(text: str, replacement: str = "既往") -> str:
    """Replace date-like strings with a neutral phrase.

    ``同2023/5/4检查相仿`` becomes ``同既往检查相仿``, which preserves the clinical
    meaning while removing the comparison date.
    """
    return _DATE_RE.sub(replacement, text or "")


def image_index(images_dir: str) -> Dict[str, List[str]]:
    """Map patient id (``001``) to its sorted image paths."""
    index: Dict[str, List[str]] = {}
    for name in sorted(os.listdir(images_dir)):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        patient_id = stem.split("_")[0]
        index.setdefault(patient_id, []).append(os.path.join(images_dir, name))
    return index


@dataclass
class Record:
    patient_id: str
    images: List[str]
    findings: str = ""
    conclusion: str = ""
    qa: List[Dict[str, str]] = field(default_factory=list)
    diagnosis: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def build_records(images_dir: str, reports_dir: str) -> List[Record]:
    """Build one Record per patient from an images/ + reports/ layout."""
    from .qa import qa_from_sections
    from .labels import derive_diagnosis

    image_map = image_index(images_dir)
    records: List[Record] = []

    for name in sorted(os.listdir(reports_dir)):
        if not name.lower().endswith(".docx"):
            continue
        patient_id = os.path.splitext(name)[0]
        if patient_id not in image_map:
            continue
        sections = parse_report(
            read_docx_paragraphs(os.path.join(reports_dir, name))
        )
        findings = "\n".join(sections["findings"]).strip()
        conclusion = "\n".join(sections["conclusion"]).strip()
        check_deidentified(findings, f"{name}:findings")
        check_deidentified(conclusion, f"{name}:conclusion")

        records.append(
            Record(
                patient_id=patient_id,
                images=image_map[patient_id],
                findings=findings,
                conclusion=conclusion,
                qa=qa_from_sections(findings, conclusion),
                diagnosis=derive_diagnosis(conclusion),
            )
        )

    records.sort(key=lambda r: r.patient_id)
    return records


def write_jsonl(path: str, records: Iterable) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            item = record.to_dict() if isinstance(record, Record) else record
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_records(path: str) -> List[dict]:
    """Load JSONL records, or a legacy JSON list of chat-style records.

    Legacy records use ``{"messages": [...], "images": [...]}``. They are
    converted to the flat record schema with question/answer pairs parsed from
    the message list. Patient id is taken from the first image stem.
    """
    from .qa import parse_chat_messages

    if path.endswith(".jsonl"):
        return [_normalize_qa_fields(record) for record in read_jsonl(path)]

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict) and "items" in data:
        data = data["items"]

    records = []
    for item in data:
        if "qa" in item and "patient_id" in item:
            records.append(_normalize_qa_fields(item))
            continue
        images = item.get("images") or ([item["image"]] if "image" in item else [])
        patient_id = ""
        if images:
            patient_id = os.path.splitext(os.path.basename(images[0]))[0].split("_")[0]
        qa = item.get("qa") or parse_chat_messages(item.get("messages", []))
        records.append(
            _normalize_qa_fields(
                {
                    "patient_id": patient_id,
                    "images": images,
                    "findings": item.get("findings", ""),
                    "conclusion": item.get("conclusion", ""),
                    "qa": qa,
                    "diagnosis": item.get("diagnosis"),
                }
            )
        )
    return records


def _normalize_qa_fields(record: dict) -> dict:
    from .qa import normalize_qa_items

    for key in ("qa", "qa_pred"):
        if record.get(key):
            record[key] = normalize_qa_items(record[key])
    return record
