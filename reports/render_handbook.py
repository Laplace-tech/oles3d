"""Korean 연구 Markdown을 자체 완결형 A4 복습 PDF로 렌더링."""

from __future__ import annotations

import argparse
import html
import io
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import mathtext
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    CondPageBreak,
    Flowable,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents


DEFAULT_SOURCE = Path("reports/2026-09-13_oles3d_research_handbook.md")
NAVY = colors.HexColor("#152D43")
TEAL = colors.HexColor("#077D85")
INK = colors.HexColor("#223244")
MUTED = colors.HexColor("#647487")
PALE = colors.HexColor("#F1F6F8")
RULE = colors.HexColor("#D4E1E7")
FONT_ROOT = Path("/mnt/c/Windows/Fonts")


def register_fonts(font_root: Path) -> None:
    """본문·강조·코드용 font를 PDF 내부에 포함."""

    for name, filename in (
        ("Malgun", "malgun.ttf"),
        ("Malgun-Bold", "malgunbd.ttf"),
        ("Consolas", "consola.ttf"),
    ):
        font_path = font_root / filename
        if not font_path.is_file():
            raise FileNotFoundError(f"PDF font 누락: {font_path}")
        pdfmetrics.registerFont(TTFont(name, str(font_path)))
    pdfmetrics.registerFontFamily(
        "Malgun", normal="Malgun", bold="Malgun-Bold",
        italic="Malgun", boldItalic="Malgun-Bold",
    )


def inline_markup(text: str) -> str:
    """지원 Markdown의 inline 요소를 안전한 Paragraph markup으로 변환."""

    tokens: list[str] = []

    def stash(markup: str) -> str:
        tokens.append(markup)
        return f"\x01{len(tokens) - 1}\x02"

    def code(match: re.Match[str]) -> str:
        return stash(
            '<font color="#096C75">'
            + html.escape(match.group(1))
            + "</font>"
        )

    text = re.sub(r"`([^`]+)`", code, text)

    def link(match: re.Match[str]) -> str:
        label, target = match.groups()
        if not target.startswith(("http://", "https://", "mailto:")):
            # 오프라인 PDF의 로컬 링크는 실행 링크 대신 읽을 수 있는 label 유지.
            return stash(html.escape(label))
        return stash(
            f'<link href="{html.escape(target, quote=True)}" '
            f'color="#077D85">{html.escape(label)}</link>'
        )

    text = re.sub(r"\[([^]]+)\]\(([^)]+)\)", link, text)
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
    for index in reversed(range(len(tokens))):
        text = text.replace(f"\x01{index}\x02", tokens[index])
    return text


def styles() -> dict[str, ParagraphStyle]:
    """A4 본문·표·목차의 일관된 한글 typography 정의."""

    base = ParagraphStyle(
        "Body", fontName="Malgun", fontSize=9.8, leading=15.7,
        textColor=INK, wordWrap="CJK", splitLongWords=True,
        spaceAfter=7, allowWidows=0, allowOrphans=0,
    )
    result = {"body": base}
    settings: dict[str, dict[str, Any]] = {
        "title": dict(fontName="Malgun-Bold", fontSize=27, leading=39,
                      textColor=NAVY, spaceAfter=24),
        "subtitle": dict(fontSize=12, leading=21, textColor=MUTED),
        "eyebrow": dict(fontName="Malgun-Bold", fontSize=10, leading=16,
                        textColor=TEAL, spaceAfter=14),
        "h2": dict(fontName="Malgun-Bold", fontSize=19, leading=28,
                   textColor=NAVY, spaceBefore=1, spaceAfter=14,
                   keepWithNext=True),
        "h3": dict(fontName="Malgun-Bold", fontSize=13, leading=20,
                   textColor=TEAL, spaceBefore=13, spaceAfter=7,
                   keepWithNext=True),
        "h4": dict(fontName="Malgun-Bold", fontSize=10.6, leading=17,
                   textColor=NAVY, spaceBefore=9, spaceAfter=5,
                   keepWithNext=True),
        "list": dict(leftIndent=12, firstLineIndent=-10, spaceAfter=4),
        "quote": dict(fontSize=9.4, leading=15.1, leftIndent=11,
                      rightIndent=9, borderPadding=9, borderWidth=0.5,
                      borderColor=RULE, backColor=PALE, spaceBefore=6,
                      spaceAfter=13),
        "table": dict(fontSize=8.3, leading=12.4, spaceAfter=0),
        "table_header": dict(fontName="Malgun-Bold", fontSize=8.4,
                             leading=12.7, textColor=colors.white,
                             spaceAfter=0),
        "caption": dict(fontSize=8.2, leading=12.2, textColor=MUTED,
                        alignment=TA_CENTER, spaceBefore=4, spaceAfter=10),
        "toc": dict(fontSize=10, leading=16, textColor=NAVY,
                    leftIndent=0, firstLineIndent=0, spaceBefore=5,
                    spaceAfter=5, rightIndent=24),
    }
    for name, options in settings.items():
        result[name] = ParagraphStyle(name, parent=base, **options)
    return result


def code_font(character: str) -> str:
    """ASCII 고정폭과 한글 glyph의 font 분리."""

    return "Consolas" if ord(character) < 128 else "Malgun"


def code_width(text: str, font_size: float) -> float:
    """혼합 font 코드 한 줄의 실제 출력 폭 계산."""

    return sum(
        pdfmetrics.stringWidth(character, code_font(character), font_size)
        for character in text
    )


class CodeBlock(Flowable):
    """한글 주석·긴 CLI·page split을 지원하는 고정폭 코드 블록."""

    def __init__(
        self, lines: list[str], language: str = "", *,
        continued: bool = False, prepared: bool = False,
    ) -> None:
        super().__init__()
        self.lines = [line.expandtabs(4) for line in lines]
        self.language = language
        self.continued = continued
        self.prepared = prepared
        self.font_size = 8.1
        self.leading = 11.7
        self.padding = 10.0
        self.spaceBefore = 4
        self.spaceAfter = 10
        self.rendered_lines: list[str] = []

    def _wrap_line(self, line: str, width: float) -> list[str]:
        if not line:
            return [""]
        output: list[str] = []
        remaining = line
        continuation = " " * min(len(line) - len(line.lstrip()) + 4, 12)
        while code_width(remaining, self.font_size) > width:
            running_width = 0.0
            cut = 0
            for index, character in enumerate(remaining):
                running_width += code_width(character, self.font_size)
                if running_width > width:
                    break
                cut = index + 1
            if cut <= len(continuation):
                raise ValueError("코드 블록에 필요한 최소 폭 부족")
            # 가능한 경우 공백에서 분리, 지나치게 짧은 조각은 방지.
            whitespace = remaining.rfind(" ", 0, cut + 1)
            if whitespace >= int(cut * 0.65):
                cut = whitespace
            output.append(remaining[:cut].rstrip())
            remaining = continuation + remaining[cut:].lstrip()
        output.append(remaining)
        return output

    def wrap(self, available_width: float, available_height: float) -> tuple[float, float]:
        self.width = available_width
        inner_width = self.width - self.padding * 2
        self.rendered_lines = (
            self.lines if self.prepared else [
                segment for line in self.lines
                for segment in self._wrap_line(line, inner_width)
            ]
        )
        self.height = (
            self.padding * 2 + 13 + max(1, len(self.rendered_lines)) * self.leading
        )
        return self.width, self.height

    def split(self, available_width: float, available_height: float) -> list[Flowable]:
        self.wrap(available_width, available_height)
        fitting = int((available_height - self.padding * 2 - 13) / self.leading)
        if fitting < 2:
            return []
        if fitting >= len(self.rendered_lines):
            return [self]
        return [
            CodeBlock(self.rendered_lines[:fitting], self.language,
                      continued=self.continued, prepared=True),
            CodeBlock(self.rendered_lines[fitting:], self.language,
                      continued=True, prepared=True),
        ]

    def draw(self) -> None:
        self.canv.setFillColor(PALE)
        self.canv.setStrokeColor(RULE)
        self.canv.roundRect(0, 0, self.width, self.height, 4, fill=1, stroke=1)
        self.canv.setFont("Consolas", 7.0)
        self.canv.setFillColor(TEAL)
        label = (self.language or "TEXT").upper()
        if self.continued:
            label += " / CONTINUED"
        self.canv.drawString(self.padding, self.height - self.padding - 6, label)
        y_position = self.height - self.padding - 13 - self.font_size
        for line in self.rendered_lines:
            text_object = self.canv.beginText(self.padding, y_position)
            text_object.setFillColor(INK)
            current_font: str | None = None
            for character in line:
                desired_font = code_font(character)
                if desired_font != current_font:
                    text_object.setFont(desired_font, self.font_size)
                    current_font = desired_font
                text_object.textOut(character)
            self.canv.drawText(text_object)
            y_position -= self.leading


class HandbookDocument(BaseDocTemplate):
    """동적 목차·PDF bookmark·page chrome 구성."""

    def __init__(self, path: Path, title: str, report_date: str) -> None:
        super().__init__(
            str(path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
            topMargin=21 * mm, bottomMargin=20 * mm,
            title=title, author="박용민 / OLES3D",
            subject="OLES3D 연구 근거와 재현 절차의 오프라인 학습 기록",
        )
        self.report_date = report_date
        self.chapter_title = ""
        frame = Frame(
            self.leftMargin, self.bottomMargin, self.width, self.height,
            leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
        )
        self.addPageTemplates(PageTemplate(id="handbook", frames=frame, onPage=self.chrome))

    def beforeDocument(self) -> None:
        self.chapter_title = ""

    def chrome(self, canvas: Any, document: Any) -> None:
        canvas.saveState()
        if document.page == 1:
            canvas.setFillColor(PALE)
            canvas.rect(0, A4[1] - 27 * mm, A4[0], 27 * mm, fill=1, stroke=0)
            canvas.setFillColor(TEAL)
            canvas.rect(self.leftMargin, A4[1] - 28 * mm, 28 * mm, 2 * mm, fill=1, stroke=0)
        else:
            canvas.setFont("Consolas", 8)
            canvas.setFillColor(MUTED)
            canvas.drawString(self.leftMargin, A4[1] - 12 * mm, "OLES3D / RESEARCH HANDBOOK")
            canvas.setStrokeColor(RULE)
            canvas.line(self.leftMargin, A4[1] - 15 * mm, A4[0] - self.rightMargin, A4[1] - 15 * mm)
        canvas.setStrokeColor(RULE)
        canvas.line(self.leftMargin, 15 * mm, A4[0] - self.rightMargin, 15 * mm)
        canvas.setFont("Malgun", 7.3)
        canvas.setFillColor(MUTED)
        canvas.drawString(self.leftMargin, 10.5 * mm, f"OLES3D · {self.report_date} · 연구 기록 / 오프라인 복습")
        canvas.setFont("Consolas", 8)
        canvas.drawRightString(A4[0] - self.rightMargin, 10.5 * mm, f"{document.page:02d}")
        canvas.restoreState()

    def afterFlowable(self, flowable: Flowable) -> None:
        if not isinstance(flowable, Paragraph) or not hasattr(flowable, "bookmark_key"):
            return
        key = flowable.bookmark_key
        title = flowable.getPlainText()
        level = flowable.bookmark_level
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(title, key, level=level, closed=False)
        if level == 0:
            self.notify("TOCEntry", (0, title, self.page, key))


def is_table_separator(line: str) -> bool:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def table_cells(line: str) -> list[str]:
    return [cell.strip().replace(r"\|", "|") for cell in re.split(r"(?<!\\)\|", line.strip().strip("|"))]


def make_table(rows: list[list[str]], style_map: dict[str, ParagraphStyle], width: float) -> Table:
    """반복 header와 내용 길이 기반 column 폭을 적용한 표 생성."""

    column_count = max(len(row) for row in rows)
    normalized = [row + [""] * (column_count - len(row)) for row in rows]
    weights = [
        min(3.0, max(1.0, (max(len(row[index]) for row in normalized) / 12) ** 0.55))
        for index in range(column_count)
    ]
    widths = [width * weight / sum(weights) for weight in weights]
    data = [
        [Paragraph(inline_markup(cell), style_map["table_header" if row_index == 0 else "table"])
         for cell in row]
        for row_index, row in enumerate(normalized)
    ]
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("LINEBELOW", (0, 0), (-1, 0), 0.7, TEAL),
        ("LINEBELOW", (0, 1), (-1, -1), 0.35, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def math_image(expression: str, width: float) -> Image:
    """Display math를 고해상도 bitmap으로 렌더링해 PDF에 포함."""

    buffer = io.BytesIO()
    mathtext.math_to_image(
        "$" + expression.strip() + "$", buffer, dpi=220, format="png",
        color="#152D43",
    )
    buffer.seek(0)
    result = Image(buffer)
    # mathtext default 10pt를 본문 수식 12pt 수준으로 배율 설정.
    native_scale = 72 / 220 * 1.2
    scale = min(native_scale, width / result.imageWidth)
    result.drawWidth = result.imageWidth * scale
    result.drawHeight = result.imageHeight * scale
    result.hAlign = "CENTER"
    result.spaceBefore = 8
    result.spaceAfter = 11
    return result


def make_heading(text: str, level: int, key: str, style_map: dict[str, ParagraphStyle]) -> Paragraph:
    heading = Paragraph(inline_markup(text), style_map[f"h{level}"])
    if level in (2, 3):
        heading.bookmark_key = key
        heading.bookmark_level = level - 2
    return heading


def markdown_body(source: Path, style_map: dict[str, ParagraphStyle], width: float) -> list[Flowable]:
    """명시한 Markdown subset을 페이지 분할 가능한 flowable로 변환."""

    lines = source.read_text(encoding="utf-8").splitlines()
    result: list[Flowable] = []
    index = 0
    chapter_count = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped or stripped.startswith("# "):
            index += 1
            continue
        if stripped == "<!-- pagebreak -->":
            result.append(PageBreak())
            index += 1
            continue
        if stripped.startswith("<!--"):
            while "-->" not in lines[index]:
                index += 1
                if index >= len(lines):
                    raise ValueError("닫히지 않은 HTML comment")
            index += 1
            continue
        if stripped.startswith("```"):
            language = stripped[3:].strip()
            index += 1
            code_lines: list[str] = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code_lines.append(lines[index])
                index += 1
            if index == len(lines):
                raise ValueError("닫히지 않은 code fence")
            result.append(CodeBlock(code_lines, language))
            index += 1
            continue
        if stripped.startswith("$$"):
            if len(stripped) > 4 and stripped.endswith("$$"):
                expression = stripped[2:-2]
                index += 1
            else:
                expression_lines = [stripped[2:]] if stripped[2:] else []
                index += 1
                while index < len(lines) and lines[index].strip() != "$$":
                    expression_lines.append(lines[index].strip())
                    index += 1
                if index == len(lines):
                    raise ValueError("닫히지 않은 display math")
                index += 1
                expression = " ".join(expression_lines)
            result.append(math_image(expression, width))
            continue
        heading_match = re.fullmatch(r"(#{2,4})\s+(.+)", stripped)
        if heading_match:
            markers, title = heading_match.groups()
            level = len(markers)
            if level == 2:
                if chapter_count or result:
                    result.append(PageBreak())
                chapter_count += 1
            else:
                result.append(CondPageBreak(31 * mm))
            result.append(make_heading(title, level, f"section-{index}", style_map))
            index += 1
            continue
        image_match = re.fullmatch(r"!\[([^]]*)\]\(([^)]+)\)", stripped)
        if image_match:
            caption, relative_path = image_match.groups()
            image_path = (source.parent / relative_path).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(f"보고서 이미지 누락: {image_path}")
            figure = Image(str(image_path))
            scale = min(width / figure.imageWidth, 135 * mm / figure.imageHeight)
            figure.drawWidth = figure.imageWidth * scale
            figure.drawHeight = figure.imageHeight * scale
            figure.hAlign = "CENTER"
            result.append(KeepTogether([
                figure, Paragraph(inline_markup(caption), style_map["caption"]),
            ]))
            index += 1
            continue
        if stripped.startswith("|") and index + 1 < len(lines) and is_table_separator(lines[index + 1]):
            rows = [table_cells(stripped)]
            index += 2
            while index < len(lines) and lines[index].strip().startswith("|"):
                rows.append(table_cells(lines[index]))
                index += 1
            result.extend([CondPageBreak(25 * mm), make_table(rows, style_map, width), Spacer(1, 10)])
            continue
        if stripped.startswith(">"):
            quote: list[str] = []
            while index < len(lines) and lines[index].strip().startswith(">"):
                quote.append(lines[index].strip()[1:].strip())
                index += 1
            result.append(Paragraph(inline_markup(" ".join(quote)), style_map["quote"]))
            continue
        bullet_match = re.match(r"^([-*]|\d+\.)\s+(.+)$", stripped)
        if bullet_match:
            marker, content = bullet_match.groups()
            marker = "•" if marker in ("-", "*") else marker
            result.append(Paragraph(inline_markup(f"{marker} {content}"), style_map["list"]))
            index += 1
            continue
        if stripped == "---":
            result.append(Spacer(1, 5 * mm))
            index += 1
            continue
        paragraph = [stripped]
        index += 1
        while index < len(lines):
            candidate = lines[index].strip()
            if not candidate or re.match(r"^(#|```|\| |\||>|!\[|\$\$|<!--|[-*] |\d+\. )", candidate):
                break
            paragraph.append(candidate)
            index += 1
        result.append(Paragraph(inline_markup(" ".join(paragraph)), style_map["body"]))
    return result


def main() -> None:
    """표지·자동 목차·본문을 조립하고 단일 offline PDF 생성."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--font-root", type=Path, default=FONT_ROOT)
    arguments = parser.parse_args()
    source = arguments.source.resolve()
    output = (arguments.output or source.with_suffix(".pdf")).resolve()
    register_fonts(arguments.font_root)
    source_text = source.read_text(encoding="utf-8")
    title_match = re.search(r"^# (.+)$", source_text, re.MULTILINE)
    title = title_match.group(1) if title_match else "OLES3D 연구 학습 보고서"
    date_match = re.search(r"\d{4}-\d{2}-\d{2}", source.name)
    report_date = date_match.group() if date_match else "Research snapshot"
    output.parent.mkdir(parents=True, exist_ok=True)
    style_map = styles()
    document = HandbookDocument(output, title, report_date)
    toc = TableOfContents()
    toc.levelStyles = [style_map["toc"]]
    toc.dotsMinLevel = 0
    story: list[Flowable] = [
        Spacer(1, 27 * mm),
        Paragraph("OLES3D / RESEARCH HANDBOOK", style_map["eyebrow"]),
        Paragraph(inline_markup(title), style_map["title"]),
        Paragraph("연구 질문에서 데이터 근거와 재현 가능한 판단까지", style_map["subtitle"]),
        Spacer(1, 15 * mm),
        Paragraph("질문 → 핵심 연산 → 관찰 결과 → 해석과 한계 → 재현 명령", style_map["body"]),
        Spacer(1, 34 * mm),
        Paragraph(f"박용민 · 마벨러스<br/>{report_date} 기준 연구 기록", style_map["subtitle"]),
        Spacer(1, 7 * mm),
        Paragraph("오프라인 복습용 · 완료된 실험과 계획을 구분한 역사적 snapshot", style_map["caption"]),
        PageBreak(),
        Paragraph("목차", style_map["h2"]),
        Paragraph("장 제목을 누르면 해당 페이지로 이동. PDF 북마크에서도 장·절 탐색 가능.", style_map["body"]),
        Spacer(1, 5 * mm),
        toc,
        PageBreak(),
    ]
    story.extend(markdown_body(source, style_map, document.width))
    document.multiBuild(story, maxPasses=8)
    print(f"PDF: {output}")
    print(f"Bytes: {output.stat().st_size:,}")
    print("Embedded fonts, images, equations, linked contents and PDF bookmarks: generated")


if __name__ == "__main__":
    main()
