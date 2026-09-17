"""PDF의 text·bookmark·page bounds 검사와 시각 점검용 PNG 생성.

격리 실행: uv run --no-project --with pymupdf python reports/verify_handbook.py PDF
프로젝트 Python environment 변경 없이 PyMuPDF를 일회성 도구 환경에서 사용.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pymupdf as fitz


def main() -> None:
    """자동 검사 결과와 선택 page raster를 별도 QA 경로에 저장."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expect", action="append", default=[])
    parser.add_argument("--pages", nargs="+", type=int)
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    document = fitz.open(arguments.pdf)
    pages_text = [page.get_text(sort=True) for page in document]
    full_text = "\n".join(pages_text)
    bounds_errors: list[dict[str, Any]] = []
    empty_pages: list[int] = []
    for page_index, page in enumerate(document):
        if len(pages_text[page_index].strip()) < 50:
            empty_pages.append(page_index + 1)
        for block in page.get_text("dict")["blocks"]:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    bounds = fitz.Rect(span["bbox"])
                    if not (page.rect + (-1, -1, 1, 1)).contains(bounds):
                        bounds_errors.append({
                            "page": page_index + 1, "text": span["text"],
                            "bounds": list(bounds),
                        })
    checks: dict[str, Any] = {
        "pdf": str(arguments.pdf.resolve()),
        "page_count": len(document),
        "bytes": arguments.pdf.stat().st_size,
        "hangul_characters": len(re.findall(r"[가-힣]", full_text)),
        "replacement_characters": full_text.count("\ufffd"),
        "nul_characters": full_text.count("\x00"),
        "bookmarks": len(document.get_toc()),
        "links": sum(len(page.get_links()) for page in document),
        "empty_pages": empty_pages,
        "page_bounds_errors": bounds_errors,
        "missing_expected_text": [term for term in arguments.expect if term not in full_text],
        "page_text_lengths": [len(text) for text in pages_text],
    }
    selected_pages = arguments.pages or sorted({1, 2, 3, max(1, len(document) // 2), len(document)})
    for page_number in selected_pages:
        if not 1 <= page_number <= len(document):
            raise ValueError(f"Page 범위 오류: {page_number}")
        pixmap = document[page_number - 1].get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
        pixmap.save(arguments.output_dir / f"page_{page_number:03d}.png")
    (arguments.output_dir / "extracted_text.txt").write_text(full_text, encoding="utf-8")
    (arguments.output_dir / "qa_summary.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    if (checks["replacement_characters"] or checks["nul_characters"]
            or bounds_errors or empty_pages or checks["missing_expected_text"]
            or not checks["hangul_characters"] or not checks["bookmarks"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
