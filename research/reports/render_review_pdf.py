"""OLES3D Markdown 복습서를 Korean-font PDF로 변환."""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    CondPageBreak,
    Image,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


DEFAULT_SOURCE = Path(
    "research/reports/2026-09-12_prerequisite_to_phase1_review.md"
)
DEFAULT_OUTPUT = Path(
    "research/reports/2026-09-12_prerequisite_to_phase1_review.pdf"
)
REGULAR_FONT = Path("/mnt/c/Windows/Fonts/malgun.ttf")
BOLD_FONT = Path("/mnt/c/Windows/Fonts/malgunbd.ttf")


def parse_arguments() -> argparse.Namespace:
    """Source와 output 경로 선택."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def register_fonts() -> None:
    """Windows Korean font를 PDF에 embed."""

    if not REGULAR_FONT.is_file() or not BOLD_FONT.is_file():
        raise FileNotFoundError("맑은 고딕 font를 /mnt/c/Windows/Fonts에서 찾지 못함")
    pdfmetrics.registerFont(TTFont("Malgun", REGULAR_FONT))
    pdfmetrics.registerFont(TTFont("Malgun-Bold", BOLD_FONT))
    pdfmetrics.registerFontFamily(
        "Malgun",
        normal="Malgun",
        bold="Malgun-Bold",
    )


def inline_markup(text: str) -> str:
    """Markdown inline 강조와 link를 ReportLab markup으로 변환."""

    escaped = html.escape(text, quote=True)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(
        r"`([^`]+)`",
        r'<font color="#7C2D12">\1</font>',
        escaped,
    )
    escaped = re.sub(
        r"\[([^]]+)]\(([^)]+)\)",
        r'<link href="\2" color="#1D4ED8">\1</link>',
        escaped,
    )
    return escaped


def build_styles() -> dict[str, ParagraphStyle]:
    """복습서 typography 정의."""

    sample = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleKorean",
            parent=sample["Title"],
            fontName="Malgun-Bold",
            fontSize=24,
            leading=32,
            textColor=colors.HexColor("#0F172A"),
            alignment=TA_CENTER,
            spaceAfter=14,
        ),
        "h2": ParagraphStyle(
            "H2Korean",
            parent=sample["Heading1"],
            fontName="Malgun-Bold",
            fontSize=16,
            leading=22,
            textColor=colors.HexColor("#0F3D5E"),
            spaceBefore=8,
            spaceAfter=9,
        ),
        "h3": ParagraphStyle(
            "H3Korean",
            parent=sample["Heading2"],
            fontName="Malgun-Bold",
            fontSize=12.5,
            leading=18,
            textColor=colors.HexColor("#155E75"),
            spaceBefore=10,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "BodyKorean",
            parent=sample["BodyText"],
            fontName="Malgun",
            fontSize=9.2,
            leading=15,
            textColor=colors.HexColor("#1F2937"),
            alignment=TA_LEFT,
            spaceAfter=6,
        ),
        "quote": ParagraphStyle(
            "QuoteKorean",
            parent=sample["BodyText"],
            fontName="Malgun",
            fontSize=8.8,
            leading=14,
            leftIndent=10,
            rightIndent=10,
            borderColor=colors.HexColor("#94A3B8"),
            borderWidth=0.7,
            borderPadding=7,
            backColor=colors.HexColor("#F1F5F9"),
            spaceAfter=8,
        ),
        "list": ParagraphStyle(
            "ListKorean",
            parent=sample["BodyText"],
            fontName="Malgun",
            fontSize=9,
            leading=14,
            leftIndent=14,
            firstLineIndent=-8,
            spaceAfter=3,
        ),
        "caption": ParagraphStyle(
            "CaptionKorean",
            parent=sample["BodyText"],
            fontName="Malgun",
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#475569"),
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
        "table": ParagraphStyle(
            "TableKorean",
            parent=sample["BodyText"],
            fontName="Malgun",
            fontSize=7.4,
            leading=10.2,
            textColor=colors.HexColor("#1F2937"),
        ),
    }


def is_table_separator(line: str) -> bool:
    """Markdown table separator 여부."""

    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def table_cells(line: str) -> list[str]:
    """Markdown table row 분리."""

    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def make_table(
    rows: list[list[str]],
    styles: dict[str, ParagraphStyle],
    available_width: float,
) -> Table:
    """Markdown table을 반복 header를 갖는 PDF table로 변환."""

    column_count = max(len(row) for row in rows)
    normalized = [row + [""] * (column_count - len(row)) for row in rows]
    data = [
        [Paragraph(inline_markup(cell), styles["table"]) for cell in row]
        for row in normalized
    ]
    table = Table(
        data,
        colWidths=[available_width / column_count] * column_count,
        repeatRows=1,
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCEAF3")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
                ("FONTNAME", (0, 0), (-1, 0), "Malgun-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#94A3B8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ]
        )
    )
    return table


def markdown_to_story(
    source_path: Path,
    styles: dict[str, ParagraphStyle],
    available_width: float,
) -> list[Any]:
    """사용한 Markdown subset을 ReportLab flowable로 변환."""

    lines = source_path.read_text(encoding="utf-8").splitlines()
    story: list[Any] = []
    index = 0
    h2_count = 0

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            index += 1
            continue

        if stripped.startswith("```"):
            language = stripped[3:].strip()
            index += 1
            code_lines: list[str] = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code_lines.append(lines[index])
                index += 1
            index += 1
            label = f"[{language}]\n" if language else ""
            story.append(
                Preformatted(
                    label + "\n".join(code_lines),
                    ParagraphStyle(
                        "CodeBlock",
                        fontName="Malgun",
                        fontSize=7.2,
                        leading=10.3,
                        leftIndent=6,
                        rightIndent=6,
                        borderColor=colors.HexColor("#CBD5E1"),
                        borderWidth=0.5,
                        borderPadding=7,
                        backColor=colors.HexColor("#F8FAFC"),
                        textColor=colors.HexColor("#111827"),
                        spaceBefore=3,
                        spaceAfter=8,
                    ),
                )
            )
            continue

        image_match = re.fullmatch(r"!\[([^]]*)]\(([^)]+)\)", stripped)
        if image_match:
            caption, relative_path = image_match.groups()
            image_path = (source_path.parent / relative_path).resolve()
            if image_path.is_file():
                pdf_image = Image(str(image_path))
                scale = min(
                    available_width / pdf_image.imageWidth,
                    82 * mm / pdf_image.imageHeight,
                )
                pdf_image.drawWidth = pdf_image.imageWidth * scale
                pdf_image.drawHeight = pdf_image.imageHeight * scale
                story.extend(
                    [
                        CondPageBreak(pdf_image.drawHeight + 16 * mm),
                        pdf_image,
                        Paragraph(inline_markup(caption), styles["caption"]),
                    ]
                )
            else:
                story.append(
                    Paragraph(
                        f"[Image missing: {html.escape(str(image_path))}]",
                        styles["quote"],
                    )
                )
            index += 1
            continue

        if stripped.startswith("# "):
            story.append(Spacer(1, 35 * mm))
            story.append(Paragraph(inline_markup(stripped[2:]), styles["title"]))
            index += 1
            continue

        if stripped.startswith("## "):
            h2_count += 1
            if h2_count > 1:
                story.append(PageBreak())
            story.append(Paragraph(inline_markup(stripped[3:]), styles["h2"]))
            index += 1
            continue

        if stripped.startswith("### ") or stripped.startswith("#### "):
            heading = stripped.lstrip("#").strip()
            story.extend(
                [
                    CondPageBreak(25 * mm),
                    Paragraph(inline_markup(heading), styles["h3"]),
                ]
            )
            index += 1
            continue

        if stripped.startswith("|") and index + 1 < len(lines):
            if is_table_separator(lines[index + 1]):
                rows = [table_cells(line)]
                index += 2
                while index < len(lines) and lines[index].strip().startswith("|"):
                    rows.append(table_cells(lines[index]))
                    index += 1
                story.extend(
                    [
                        CondPageBreak(22 * mm),
                        make_table(rows, styles, available_width),
                        Spacer(1, 7),
                    ]
                )
                continue

        if stripped.startswith(">"):
            quote_lines: list[str] = []
            while index < len(lines) and lines[index].strip().startswith(">"):
                quote_lines.append(lines[index].strip().lstrip(">").strip())
                index += 1
            story.append(
                Paragraph(inline_markup(" ".join(quote_lines)), styles["quote"])
            )
            continue

        list_match = re.match(r"^(?:[-*]|\d+\.)\s+(.*)$", stripped)
        if list_match:
            marker = stripped.split(maxsplit=1)[0]
            bullet = "•" if marker in {"-", "*"} else marker
            story.append(
                Paragraph(
                    inline_markup(f"{bullet} {list_match.group(1)}"),
                    styles["list"],
                )
            )
            index += 1
            continue

        paragraph_lines = [stripped.rstrip("  ")]
        index += 1
        while index < len(lines):
            candidate = lines[index].strip()
            if (
                not candidate
                or candidate.startswith("#")
                or candidate.startswith("```")
                or candidate.startswith("|")
                or candidate.startswith(">")
                or candidate.startswith("![")
                or re.match(r"^(?:[-*]|\d+\.)\s+", candidate)
            ):
                break
            paragraph_lines.append(candidate.rstrip("  "))
            index += 1
        story.append(
            Paragraph(inline_markup(" ".join(paragraph_lines)), styles["body"])
        )

    return story


def draw_page(canvas: Any, document: Any) -> None:
    """Footer와 page number 출력."""

    canvas.saveState()
    canvas.setFont("Malgun", 7.5)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawString(16 * mm, 10 * mm, "OLES3D Research Review · 2026-09-12")
    canvas.drawRightString(A4[0] - 16 * mm, 10 * mm, f"{document.page}")
    canvas.restoreState()


def main() -> None:
    """Markdown 복습서 PDF 생성."""

    arguments = parse_arguments()
    register_fonts()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)

    document = SimpleDocTemplate(
        str(arguments.output),
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=17 * mm,
        title="OLES3D 연구 출항 복습서",
        author="박용민",
        subject="Prerequisite 종료부터 Phase 1 Data Foundation까지",
    )
    available_width = A4[0] - document.leftMargin - document.rightMargin
    story = markdown_to_story(
        arguments.source,
        build_styles(),
        available_width,
    )
    document.build(story, onFirstPage=draw_page, onLaterPages=draw_page)
    print(arguments.output.resolve())


if __name__ == "__main__":
    main()
