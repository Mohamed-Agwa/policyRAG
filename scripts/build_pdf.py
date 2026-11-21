"""Render a Markdown policy from corpus_src/ into a PDF in data/policies/.

The corpus deliberately contains one PDF so that ingestion exercises the PDF
parser. The PDF is committed to the repo; run this script only if you edit the
Markdown source:

    pip install -r requirements-dev.txt
    python scripts/build_pdf.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "corpus_src" / "POL-HR-008-benefits-and-wellbeing.md"
OUT = ROOT / "data" / "policies" / "POL-HR-008-benefits-and-wellbeing.pdf"


def inline(text: str) -> str:
    """Convert **bold** Markdown to ReportLab markup and escape XML."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)


def build(src: Path = SRC, out: Path = OUT) -> Path:
    raw = src.read_text(encoding="utf-8")
    _, fm, body = raw.split("---", 2)
    meta = yaml.safe_load(fm)

    styles = getSampleStyleSheet()
    h1, h2, h3 = styles["Heading1"], styles["Heading2"], styles["Heading3"]
    normal = ParagraphStyle("body", parent=styles["BodyText"], fontSize=10.5, leading=14)
    cell = ParagraphStyle("cell", parent=normal, fontSize=9.5, leading=12)

    story: list = [
        Paragraph(f"Nile Byte Technologies S.A.E. &middot; {meta['doc_id']} &middot; "
                  f"Version {meta['version']} &middot; Effective {meta['effective_date']}", cell),
        Spacer(1, 0.3 * cm),
    ]
    bullets: list[str] = []
    table_rows: list[list[str]] = []

    def flush_bullets() -> None:
        if bullets:
            story.append(ListFlowable(
                [ListItem(Paragraph(inline(b), normal), leftIndent=12) for b in bullets],
                bulletType="bullet", start="•", leftIndent=14))
            bullets.clear()

    def flush_table() -> None:
        if table_rows:
            data = [[Paragraph(inline(c), cell) for c in row] for row in table_rows]
            t = Table(data, repeatRows=1, hAlign="LEFT")
            t.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]))
            story.extend([t, Spacer(1, 0.3 * cm)])
            table_rows.clear()

    for line in body.strip().splitlines():
        s = line.strip()
        if s.startswith("|"):
            flush_bullets()
            cells = [c.strip() for c in s.strip("|").split("|")]
            if not all(re.fullmatch(r"-+", c) for c in cells):
                table_rows.append(cells)
            continue
        flush_table()
        if not s:
            flush_bullets()
            continue
        if s.startswith("- "):
            bullets.append(s[2:])
            continue
        flush_bullets()
        if s.startswith("### "):
            story.append(Paragraph(inline(s[4:]), h3))
        elif s.startswith("## "):
            story.append(Paragraph(inline(s[3:]), h2))
        elif s.startswith("# "):
            story.append(Paragraph(inline(s[2:]), h1))
        else:
            story.append(Paragraph(inline(s), normal))
    flush_bullets()
    flush_table()

    out.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(out), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title=meta["title"], subject=meta["doc_id"], author="Nile Byte Technologies",
        invariant=1,  # deterministic output (no timestamps/random IDs)
    )
    doc.build(story)
    return out


if __name__ == "__main__":
    path = build()
    print(f"Wrote {path.relative_to(ROOT)}", file=sys.stderr)
