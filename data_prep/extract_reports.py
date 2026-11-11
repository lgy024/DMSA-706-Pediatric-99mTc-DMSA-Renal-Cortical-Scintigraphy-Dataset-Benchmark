#!/usr/bin/env python
"""Rewrite de-identified reports with date-like strings replaced.

Reads ``NNN.docx`` from ``--src``, keeps only the findings and conclusion
sections, replaces dates with a neutral phrase (default: 既往), and writes a
clean ``NNN.docx`` to ``--dst``.

Usage:
    python data_prep/extract_reports.py --src data/reports --dst outputs/redacted_reports
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from xml.sax.saxutils import escape

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dmsa706.io import (  # noqa: E402
    HEADING_CONCLUSION,
    HEADING_FINDINGS,
    find_dates,
    parse_report,
    read_docx_paragraphs,
    redact_dates,
)

NS_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
</Types>"""

ROOT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
</Relationships>"""

CORE = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/">
<dc:title>DMSA-706 de-identified report</dc:title>
<dc:creator>DMSA-706 dataset</dc:creator>
</cp:coreProperties>"""

SECT_PR = (
    '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
    '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" '
    'w:header="851" w:footer="992" w:gutter="0"/></w:sectPr>'
)


def paragraph_xml(text: str, is_heading: bool) -> str:
    font = (
        '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" '
        'w:eastAsia="宋体"/>'
    )
    if is_heading:
        rpr = (
            "<w:rPr>%s<w:b/><w:sz w:val=\"24\"/><w:szCs w:val=\"24\"/></w:rPr>"
            % font
        )
        spacing = '<w:spacing w:before="240" w:after="120"/>'
    else:
        rpr = '<w:rPr>%s<w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr>' % font
        spacing = '<w:spacing w:after="80" w:line="360" w:lineRule="auto"/>'
    return (
        '<w:p><w:pPr>%s<w:jc w:val="left"/></w:pPr>'
        '<w:r>%s<w:t xml:space="preserve">%s</w:t></w:r></w:p>'
        % (spacing, rpr, escape(text))
    )


def write_docx(paragraphs, out_path: str) -> None:
    body = "".join(paragraph_xml(text, heading) for text, heading in paragraphs)
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="%s"><w:body>%s%s</w:body></w:document>'
        % (NS_W, body, SECT_PR)
    )
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", CONTENT_TYPES)
        archive.writestr("_rels/.rels", ROOT_RELS)
        archive.writestr("word/document.xml", document)
        archive.writestr("docProps/core.xml", CORE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--replacement", default="既往")
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    written = 0
    changed = 0
    failed = []

    for name in sorted(os.listdir(args.src)):
        if not name.lower().endswith(".docx"):
            continue
        try:
            sections = parse_report(
                read_docx_paragraphs(os.path.join(args.src, name))
            )
            findings = sections["findings"]
            conclusion = sections["conclusion"]
            before = find_dates("\n".join(findings + conclusion))
            findings = [redact_dates(text, args.replacement) for text in findings]
            conclusion = [redact_dates(text, args.replacement) for text in conclusion]

            paragraphs = [(HEADING_FINDINGS, True)]
            paragraphs += [(text, False) for text in findings]
            paragraphs += [(HEADING_CONCLUSION, True)]
            paragraphs += [(text, False) for text in conclusion]

            write_docx(paragraphs, os.path.join(args.dst, name))
            written += 1
            if before:
                changed += 1
        except Exception as exc:  # noqa: BLE001 - batch job, report and continue
            failed.append((name, repr(exc)))

    print(f"written: {written}")
    print(f"files with dates redacted: {changed}")
    print(f"failed: {len(failed)}")
    for name, why in failed[:10]:
        print(" ", name, why)


if __name__ == "__main__":
    main()
