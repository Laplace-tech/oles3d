from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIRECTORY / "OLES3D_Parts_1_to_4_2_Cumulative_Review.pdf"

FONT_REGULAR_PATH = Path("/mnt/c/Windows/Fonts/malgun.ttf")
FONT_BOLD_PATH = Path("/mnt/c/Windows/Fonts/malgunbd.ttf")

FONT_REGULAR = "MalgunGothic"
FONT_BOLD = "MalgunGothicBold"

PAGE_WIDTH, PAGE_HEIGHT = A4

NAVY = colors.HexColor("#14213D")
BLUE = colors.HexColor("#1D4ED8")
SKY = colors.HexColor("#EAF2FF")
CYAN = colors.HexColor("#0891B2")
GREEN = colors.HexColor("#15803D")
PALE_GREEN = colors.HexColor("#ECFDF3")
AMBER = colors.HexColor("#B45309")
PALE_AMBER = colors.HexColor("#FFF7E6")
RED = colors.HexColor("#B42318")
PALE_RED = colors.HexColor("#FFF1F0")
INK = colors.HexColor("#172033")
MUTED = colors.HexColor("#566176")
LINE = colors.HexColor("#D8DFEA")
PALE = colors.HexColor("#F6F8FB")
WHITE = colors.white


def register_fonts() -> None:
    """PDF 내부에 한국어 font embedding."""

    if not FONT_REGULAR_PATH.exists() or not FONT_BOLD_PATH.exists():
        raise FileNotFoundError(
            "맑은 고딕 font가 /mnt/c/Windows/Fonts에 필요"
        )

    pdfmetrics.registerFont(
        TTFont(FONT_REGULAR, str(FONT_REGULAR_PATH))
    )
    pdfmetrics.registerFont(
        TTFont(FONT_BOLD, str(FONT_BOLD_PATH))
    )


def make_styles() -> dict[str, ParagraphStyle]:
    """문서용 paragraph style 생성."""

    return {
        "cover_title": ParagraphStyle(
            "CoverTitle",
            fontName=FONT_BOLD,
            fontSize=27,
            leading=37,
            textColor=WHITE,
            alignment=TA_LEFT,
            spaceAfter=7 * mm,
        ),
        "cover_subtitle": ParagraphStyle(
            "CoverSubtitle",
            fontName=FONT_REGULAR,
            fontSize=12,
            leading=19,
            textColor=colors.HexColor("#DDE8FF"),
            alignment=TA_LEFT,
        ),
        "h1": ParagraphStyle(
            "Heading1",
            fontName=FONT_BOLD,
            fontSize=19,
            leading=27,
            textColor=NAVY,
            spaceAfter=5 * mm,
        ),
        "h2": ParagraphStyle(
            "Heading2",
            fontName=FONT_BOLD,
            fontSize=13,
            leading=19,
            textColor=BLUE,
            spaceBefore=3 * mm,
            spaceAfter=2 * mm,
        ),
        "h3": ParagraphStyle(
            "Heading3",
            fontName=FONT_BOLD,
            fontSize=10.5,
            leading=16,
            textColor=INK,
            spaceBefore=2 * mm,
            spaceAfter=1.2 * mm,
        ),
        "body": ParagraphStyle(
            "Body",
            fontName=FONT_REGULAR,
            fontSize=9.2,
            leading=15.2,
            textColor=INK,
            spaceAfter=2 * mm,
        ),
        "small": ParagraphStyle(
            "Small",
            fontName=FONT_REGULAR,
            fontSize=8.1,
            leading=12.5,
            textColor=MUTED,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            fontName=FONT_REGULAR,
            fontSize=9.0,
            leading=14.5,
            textColor=INK,
            leftIndent=5 * mm,
            firstLineIndent=-3.2 * mm,
            bulletIndent=0,
            spaceAfter=1.2 * mm,
        ),
        "table": ParagraphStyle(
            "Table",
            fontName=FONT_REGULAR,
            fontSize=7.8,
            leading=11.5,
            textColor=INK,
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            fontName=FONT_BOLD,
            fontSize=7.8,
            leading=11.5,
            textColor=WHITE,
            alignment=TA_CENTER,
        ),
        "formula": ParagraphStyle(
            "Formula",
            # 수학 기호 glyph 누락과 subscript clipping 방지를 위한 ASCII 전용 font
            fontName="Courier-Bold",
            fontSize=9.4,
            leading=18,
            textColor=NAVY,
            alignment=TA_CENTER,
        ),
        "code": ParagraphStyle(
            "Code",
            fontName=FONT_REGULAR,
            fontSize=7.6,
            leading=11.2,
            textColor=colors.HexColor("#12213A"),
            leftIndent=3 * mm,
            rightIndent=3 * mm,
            borderColor=LINE,
            borderWidth=0.6,
            borderPadding=3 * mm,
            backColor=PALE,
            spaceBefore=1.5 * mm,
            spaceAfter=3 * mm,
        ),
        "callout_title": ParagraphStyle(
            "CalloutTitle",
            fontName=FONT_BOLD,
            fontSize=9.2,
            leading=14,
            textColor=NAVY,
        ),
        "callout_body": ParagraphStyle(
            "CalloutBody",
            fontName=FONT_REGULAR,
            fontSize=8.6,
            leading=13.5,
            textColor=INK,
        ),
        "question": ParagraphStyle(
            "Question",
            fontName=FONT_REGULAR,
            fontSize=9.1,
            leading=14.8,
            textColor=INK,
            leftIndent=6 * mm,
            firstLineIndent=-6 * mm,
            spaceAfter=2.2 * mm,
        ),
    }


def page_chrome(canvas, document) -> None:  # type: ignore[no-untyped-def]
    """본문 page header와 footer 출력."""

    canvas.saveState()
    if document.page > 1:
        canvas.setStrokeColor(LINE)
        canvas.setLineWidth(0.5)
        canvas.line(18 * mm, PAGE_HEIGHT - 13 * mm, PAGE_WIDTH - 18 * mm, PAGE_HEIGHT - 13 * mm)
        canvas.setFont(FONT_REGULAR, 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(18 * mm, PAGE_HEIGHT - 10 * mm, "OLES3D · Cumulative Core Review · Parts 1–4.2")
        canvas.drawRightString(PAGE_WIDTH - 18 * mm, 10 * mm, f"{document.page}")
        canvas.line(18 * mm, 14 * mm, PAGE_WIDTH - 18 * mm, 14 * mm)
    canvas.restoreState()


def paragraph(story: list, styles: dict[str, ParagraphStyle], text: str, style: str = "body") -> None:
    story.append(Paragraph(text, styles[style]))


def bullets(story: list, styles: dict[str, ParagraphStyle], items: list[str]) -> None:
    for item in items:
        story.append(
            Paragraph(f"•&nbsp;&nbsp;{item}", styles["bullet"])
        )


def heading(story: list, styles: dict[str, ParagraphStyle], text: str, level: int = 1) -> None:
    story.append(Paragraph(text, styles[f"h{level}"]))


def code_block(story: list, styles: dict[str, ParagraphStyle], text: str) -> None:
    story.append(Preformatted(text.strip("\n"), styles["code"]))


def formula(story: list, styles: dict[str, ParagraphStyle], text: str) -> None:
    """Font에 독립적인 ASCII 수식 box 생성."""

    try:
        text.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError(
            f"수식은 glyph 누락 방지를 위해 ASCII만 허용: {text}"
        ) from error

    box = Table(
        [[Paragraph(text, styles["formula"])]],
        colWidths=[169 * mm],
    )
    box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), SKY),
                ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#AFC8FF")),
                ("LEFTPADDING", (0, 0), (-1, -1), 4 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 3 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3 * mm),
            ]
        )
    )
    story.extend([box, Spacer(1, 3 * mm)])


def callout(
    story: list,
    styles: dict[str, ParagraphStyle],
    title: str,
    text: str,
    background=PALE_GREEN,
    border=GREEN,
) -> None:
    box = Table(
        [[
            Paragraph(title, styles["callout_title"]),
            Paragraph(text, styles["callout_body"]),
        ]],
        colWidths=[35 * mm, 134 * mm],
    )
    box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("BOX", (0, 0), (-1, -1), 0.8, border),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 2.7 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.7 * mm),
            ]
        )
    )
    story.extend([box, Spacer(1, 3 * mm)])


def data_table(
    story: list,
    styles: dict[str, ParagraphStyle],
    rows: list[list[str]],
    widths: list[float],
) -> None:
    rendered_rows = []
    for row_index, row in enumerate(rows):
        style = styles["table_header"] if row_index == 0 else styles["table"]
        rendered_rows.append([Paragraph(value, style) for value in row])

    table = Table(
        rendered_rows,
        colWidths=[width * mm for width in widths],
        repeatRows=1,
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PALE]),
                ("GRID", (0, 0), (-1, -1), 0.45, LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2.2 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2.2 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 2 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2 * mm),
            ]
        )
    )
    story.extend([table, Spacer(1, 3 * mm)])


def new_page(story: list) -> None:
    story.append(PageBreak())


def build_pdf() -> Path:
    """검증 완료된 Parts 1–4.2 누적 복습 PDF 생성."""

    register_fonts()
    styles = make_styles()
    story: list = []

    # Cover
    cover = Table(
        [[
            Paragraph(
                "OLES3D<br/>Cumulative Core Review<br/>Parts 1–4.2",
                styles["cover_title"],
            ),
            "",
        ]],
        colWidths=[145 * mm, 24 * mm],
        rowHeights=[72 * mm],
    )
    cover.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), NAVY),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 12 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 8 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8 * mm),
            ]
        )
    )
    story.extend([cover, Spacer(1, 8 * mm)])
    paragraph(
        story,
        styles,
        "3D medical image segmentation을 위한 핵심 복습자료",
        "h1",
    )
    paragraph(
        story,
        styles,
        "Tensor contract에서 시작해 U-Net, volumetric patch inference, NIfTI geometry와 orientation까지 구현 전에 반드시 설명할 수 있어야 하는 개념을 현재 OLES3D scratch notebook 결과와 연결했다.",
    )
    data_table(
        story,
        styles,
        [
            ["범위", "핵심 질문", "완료 산출물"],
            ["Part 1", "Segmentation은 Tensor 관점에서 무엇인가?", "logits/target contract, CE, Dice/IoU"],
            ["Part 2", "U-Net은 spatial 정보를 어떻게 압축·복원하는가?", "2D U-Net, skip flow, tiny overfit"],
            ["Part 3", "3D volume을 제한된 memory로 어떻게 학습·추론하는가?", "3D memory, sampling, sliding window"],
            ["Part 4.1–4.2", "Voxel index를 환자의 physical space와 어떻게 연결하는가?", "NIfTI affine, orientation, three-plane viewer"],
        ],
        [25, 78, 66],
    )
    callout(
        story,
        styles,
        "자료 기준",
        "2026-09-06 repository snapshot · Part 1–3과 Lessons 4.1–4.2의 11개 notebook이 fresh project kernel에서 top-to-bottom 검증된 상태",
    )
    paragraph(story, styles, "박용민 · 경기대학교 AI컴퓨터공학부 컴퓨터공학전공", "small")

    # Orientation
    new_page(story)
    heading(story, styles, "0. 전체 지도를 먼저 잡기")
    paragraph(
        story,
        styles,
        "각 Part는 독립 과목이 아니다. Part 1이 <b>출력과 정답의 계약</b>, Part 2가 <b>그 출력을 만드는 network</b>, Part 3가 <b>3D volume에서 그 network를 현실적으로 운용하는 방법</b>, Part 4가 <b>voxel array를 환자의 physical space와 연결하는 방법</b>을 담당한다.",
    )
    heading(story, styles, "공통 notation", 2)
    data_table(
        story,
        styles,
        [
            ["기호", "의미", "대표 예"],
            ["B", "batch size", "한 step에 처리하는 patient patch 수"],
            ["C", "input channel", "CT는 보통 C=1"],
            ["K", "output class 수", "background 포함 10 classes라면 K=10"],
            ["D, H, W", "depth, height, width", "3D spatial axes"],
            ["pD, pH, pW", "patch spatial size", "예: 8×16×16"],
        ],
        [18, 65, 86],
    )
    heading(story, styles, "Training data flow", 2)
    code_block(
        story,
        styles,
        "CT patch [B,C,D,H,W]\n"
        "  → 3D network\n"
        "  → logits [B,K,D,H,W]\n"
        "  + target [B,D,H,W]\n"
        "  → voxel-wise loss → backward → parameter update",
    )
    heading(story, styles, "Full-volume inference data flow", 2)
    code_block(
        story,
        styles,
        "CT volume → overlapping patches → patch logits\n"
        "          → global coordinates에 누적\n"
        "          → count/weight normalization\n"
        "          → full logits [B,K,D,H,W]\n"
        "          → argmax(dim=1) → mask [B,D,H,W]",
    )
    callout(
        story,
        styles,
        "절대 혼동 금지",
        "C는 입력 modality/channel, K는 모델이 경쟁시키는 class axis다. D/H/W는 지워야 할 class axis가 아니라 보존해야 할 spatial axes다.",
        PALE_AMBER,
        AMBER,
    )

    # Part 1.1
    new_page(story)
    heading(story, styles, "Part 1. Segmentation Fundamentals")
    heading(story, styles, "1.1 Segmentation Tensor Contract", 2)
    paragraph(
        story,
        styles,
        "Multi-class segmentation은 각 voxel마다 K개 class score를 출력하는 dense classification이다. 따라서 한 volume에 대한 예측은 scalar 하나가 아니라 spatial grid 전체의 class decision이다.",
    )
    data_table(
        story,
        styles,
        [
            ["객체", "Shape", "Dtype", "의미"],
            ["CT input", "[B,C,D,H,W]", "float32/float16", "모델 입력 intensity"],
            ["Logits", "[B,K,D,H,W]", "floating", "softmax 전 class score"],
            ["Target", "[B,D,H,W]", "long", "voxel마다 하나의 정답 class ID"],
            ["Prediction", "[B,D,H,W]", "long", "가장 큰 logit의 class ID"],
        ],
        [31, 42, 34, 62],
    )
    formula(story, styles, "prediction[b,d,h,w] = argmax_k(logits[b,k,d,h,w])")
    paragraph(
        story,
        styles,
        "`argmax(dim=1)`이 K axis만 제거하므로 `[B,K,D,H,W] → [B,D,H,W]`가 된다. `dim=2`를 사용하면 class axis K가 남고 depth D가 사라져 target과 비교할 수 없는 잘못된 mask가 된다.",
    )
    heading(story, styles, "Repository에서 확인한 값", 2)
    code_block(
        story,
        styles,
        "Input       [2, 1, 32, 64, 64]\n"
        "Logits      [2,10, 32, 64, 64]\n"
        "Target      [2,   32, 64, 64]  torch.int64\n"
        "Prediction  [2,   32, 64, 64]  torch.int64",
    )
    bullets(
        story,
        styles,
        [
            "Target에 K가 없는 이유: 정답은 voxel마다 이미 선택된 class ID 하나이기 때문.",
            "Logits는 확률이 아니다. softmax 전의 제한 없는 실수 score이며 합이 1일 필요가 없다.",
            "Background도 하나의 class이므로 K에 포함. 예: 9개 장기 + background = K=10.",
            "Contract validation은 shape/dtype 오류를 training 이전에 잡는 가장 싼 test.",
        ],
    )

    # Part 1.2
    new_page(story)
    heading(story, styles, "1.2 Stable Softmax와 Voxel-wise Cross-Entropy")
    paragraph(
        story,
        styles,
        "각 voxel의 K개 logits를 class probability로 바꾸되, 큰 값의 exponential overflow를 피하기 위해 최대 logit을 먼저 뺀다. 이 이동은 모든 class에 같은 상수를 빼므로 probability ratio를 바꾸지 않는다.",
    )
    formula(
        story,
        styles,
        "p_k = exp(z_k - m) / sum_j(exp(z_j - m)),  m = max_j(z_j)",
    )
    data_table(
        story,
        styles,
        [
            ["입력 logits", "Naive softmax", "Stable softmax"],
            ["[1000,1001,1002]", "[nan,nan,nan]", "[0.0900,0.2447,0.6652]"],
        ],
        [55, 55, 59],
    )
    heading(story, styles, "Cross-Entropy의 의미", 2)
    formula(story, styles, "L_voxel = -log(p_target),  L = mean(L_voxel)")
    paragraph(
        story,
        styles,
        "정답 class의 probability가 1에 가까우면 loss가 0에 가까워지고, 정답 class에 낮은 probability를 주면 큰 penalty를 받는다. 구현에서는 class axis K에서 target ID에 해당하는 probability만 gather한 뒤 negative log를 취한다.",
    )
    code_block(
        story,
        styles,
        "logits [B,K,D,H,W]\n"
        "  → softmax(dim=1)\n"
        "  → target.unsqueeze(1) [B,1,D,H,W]\n"
        "  → gather(dim=1) → p_target [B,1,D,H,W]\n"
        "  → -log → mean → scalar loss",
    )
    callout(
        story,
        styles,
        "Scratch 검증",
        "직접 구현 loss 0.9076059461, PyTorch loss 0.9076058865, 절대 차이 약 5.96e-8. 수치 오차 범위에서 동일.",
    )
    heading(story, styles, "Multi-class와 Multi-label", 2)
    data_table(
        story,
        styles,
        [
            ["구분", "Multi-class", "Multi-label"],
            ["관계", "한 voxel에서 classes가 상호 배타적", "여러 class가 동시에 양성 가능"],
            ["출력", "softmax over K", "class별 sigmoid"],
            ["Target", "class ID [B,D,H,W]", "multi-hot [B,K,D,H,W]"],
            ["대표 loss", "CrossEntropyLoss", "BCEWithLogitsLoss"],
        ],
        [30, 69, 70],
    )

    # Part 1.3
    new_page(story)
    heading(story, styles, "1.3 Dice, IoU와 Empty-Mask Policy")
    paragraph(
        story,
        styles,
        "Accuracy는 background가 압도적으로 많을 때 장기를 하나도 찾지 못한 모델도 높게 평가할 수 있다. 의료영상 segmentation은 foreground overlap을 직접 보는 Dice와 IoU가 핵심이다.",
    )
    formula(story, styles, "Dice = 2*TP / (2*TP + FP + FN),  IoU = TP / (TP + FP + FN)")
    data_table(
        story,
        styles,
        [
            ["용어", "조건", "해석"],
            ["TP", "prediction=1, target=1", "맞게 찾은 organ voxel"],
            ["FP", "prediction=1, target=0", "없는 organ을 있다고 예측"],
            ["FN", "prediction=0, target=1", "실제 organ을 놓침"],
            ["TN", "prediction=0, target=0", "background를 맞힘; Dice 분자에는 없음"],
        ],
        [23, 62, 84],
    )
    heading(story, styles, "대표 test case", 2)
    data_table(
        story,
        styles,
        [
            ["상황", "TP/FP/FN", "Dice", "IoU"],
            ["Perfect", "모두 일치", "1.0", "1.0"],
            ["Disjoint", "TP=0", "0.0", "0.0"],
            ["Partial", "TP=1, FP=1, FN=1", "0.5", "0.3333"],
            ["실습 예", "TP=2, FP=1, FN=1", "0.6667", "0.5"],
        ],
        [40, 67, 31, 31],
    )
    heading(story, styles, "Empty mask는 수학이 아니라 reporting policy", 2)
    bullets(
        story,
        styles,
        [
            "Target empty + prediction empty: 보통 perfect 또는 평가 제외 중 하나를 protocol에서 사전 고정.",
            "Target empty + prediction non-empty: false-positive-only이므로 0점.",
            "Target non-empty + prediction empty: missed organ이므로 0점.",
            "Class별 score를 낸 뒤 patient/case macro 평균을 권장. voxel을 모두 합친 micro score는 큰 장기가 결과를 지배할 수 있음.",
        ],
    )
    callout(
        story,
        styles,
        "논문 방어 포인트",
        "Empty-reference 처리, background 포함 여부, class/case aggregation을 결과를 본 뒤 바꾸면 안 된다. Experiment protocol에 먼저 freeze.",
        PALE_RED,
        RED,
    )

    # Part 1 checkpoint
    new_page(story)
    heading(story, styles, "Part 1 Checkpoint · 반드시 말로 설명할 것")
    bullets(
        story,
        styles,
        [
            "`[B,K,D,H,W]`에서 K와 D의 의미가 왜 완전히 다른가?",
            "Target에 channel axis K가 없는 이유와 target dtype이 long인 이유는?",
            "왜 `argmax(dim=1)`은 inference decision이고 training loss 안에서는 사용하지 않는가?",
            "Maximum subtraction이 softmax 결과를 바꾸지 않으면서 overflow를 막는 이유는?",
            "Background accuracy가 99%여도 organ segmentation 실패일 수 있는 이유는?",
            "Empty-empty case의 점수는 왜 반드시 protocol에 명시해야 하는가?",
        ],
    )
    heading(story, styles, "빠른 debugging 순서", 2)
    code_block(
        story,
        styles,
        "1. input/logits/target/prediction shape 출력\n"
        "2. dtype와 class ID 범위 확인: 0 ≤ target < K\n"
        "3. softmax probability sum을 class axis에서 확인\n"
        "4. scratch CE와 framework CE 비교\n"
        "5. perfect/disjoint/partial/empty mask unit test\n"
        "6. background와 class별 metric을 분리",
    )
    callout(
        story,
        styles,
        "통과 기준",
        "Shape를 외우는 데서 끝나지 않고, 각 axis가 어떤 object를 index하는지 말할 수 있어야 한다.",
    )

    # Part 2 conv
    new_page(story)
    heading(story, styles, "Part 2. U-Net from Scratch")
    heading(story, styles, "2.1 Convolution Shape와 Receptive Field", 2)
    paragraph(
        story,
        styles,
        "Convolution layer를 조립하기 전에 output spatial size를 계산할 수 있어야 skip concatenation 오류를 예방할 수 있다.",
    )
    formula(
        story,
        styles,
        "N_out = floor((N_in + 2*P - dilation*(K - 1) - 1) / stride) + 1",
    )
    data_table(
        story,
        styles,
        [
            ["설정", "Output size", "핵심"],
            ["K=3, S=1, P=1", "N 유지", "same-size convolution"],
            ["K=3, S=2, P=1, N=32", "16", "downsampling"],
            ["K=5, S=1, P=0, N=32", "28", "valid convolution"],
        ],
        [66, 35, 68],
    )
    heading(story, styles, "Effective kernel과 receptive field", 2)
    formula(story, styles, "K_effective = dilation*(K - 1) + 1")
    formula(
        story,
        styles,
        "jump_l = jump_(l-1)*stride_l,  RF_l = RF_(l-1) + (K_eff - 1)*jump_(l-1)",
    )
    paragraph(
        story,
        styles,
        "실습 trace는 receptive field 3 → 5 → 6 → 10, jump 1 → 1 → 2 → 2였다. Dilation=2인 3×3 convolution은 5×5 범위를 보지만 실제 sampling point는 9개뿐이다. Gradient mask로 span 5×5와 sampled input 9개를 확인했다.",
    )
    callout(
        story,
        styles,
        "핵심 구분",
        "Receptive-field span이 넓다는 것과 모든 위치를 조밀하게 사용한다는 것은 다르다. Dilation은 사이를 건너뛰며 넓게 본다.",
        PALE_AMBER,
        AMBER,
    )

    # Part 2 U-Net
    new_page(story)
    heading(story, styles, "2.2 Encoder–Decoder와 Skip Connection")
    paragraph(
        story,
        styles,
        "Encoder는 context를 넓히고 memory를 줄이는 대신 fine boundary 위치를 잃는다. Decoder는 resolution을 회복하지만 pooling 이전의 정확한 위치 정보는 스스로 완벽히 되살릴 수 없다. Skip connection이 이 정보를 직접 전달한다.",
    )
    code_block(
        story,
        styles,
        "Input [B,1,64,64]\n"
        "  → DoubleConv → Skip [B,16,64,64] ───────────────┐\n"
        "  → MaxPool2d → [B,16,32,32]                     │\n"
        "  → Bottleneck [B,32,32,32]                      │\n"
        "  → UpConv [B,16,64,64]                           │\n"
        "  → concat(dim=1) [B,32,64,64]  ←─────────────────┘\n"
        "  → DoubleConv [B,16,64,64] → logits [B,K,64,64]",
    )
    data_table(
        story,
        styles,
        [
            ["Block", "Spatial 변화", "Channel 변화", "역할"],
            ["DoubleConv", "유지", "in → out", "local feature extraction"],
            ["Encoder pool", "H,W 절반", "유지", "context 확대, memory 감소"],
            ["Up-convolution", "H,W 두 배", "감소 가능", "resolution 복원"],
            ["Skip concat", "반드시 동일", "두 Tensor의 합", "fine detail + context 결합"],
            ["1×1 head", "유지", "features → K", "class logits 생성"],
        ],
        [31, 36, 36, 66],
    )
    heading(story, styles, "Concatenation 전 contract", 2)
    bullets(
        story,
        styles,
        [
            "Batch size 동일.",
            "Spatial H/W 또는 D/H/W 동일. 홀수 크기와 padding policy가 다르면 crop/pad 필요.",
            "Channel은 같을 필요 없음. `dim=1`에서 이어 붙이므로 합산됨.",
            "Skip은 max-pooled Tensor가 아니라 pooling 이전의 high-resolution features.",
        ],
    )

    # Part 2 tiny overfit
    new_page(story)
    heading(story, styles, "2.3 Minimal 2D U-Net과 Tiny Overfit")
    paragraph(
        story,
        styles,
        "Tiny overfit은 일반화 실험이 아니라 pipeline integrity test다. 1–2개 sample조차 외우지 못한다면 dataset, model, loss, optimizer 또는 metric 중 어딘가가 깨졌을 가능성이 높다.",
    )
    heading(story, styles, "실습 setup과 결과", 2)
    data_table(
        story,
        styles,
        [
            ["항목", "값"],
            ["Input", "[2,1,64,64], float32, CUDA"],
            ["Target", "[2,64,64], int64, 3 classes"],
            ["Trainable parameters", "117,107"],
            ["Loss", "1.018677 → 0.000688 over 200 steps"],
            ["Per-class Dice", "background/circle/rectangle 모두 1.0"],
            ["Pixel accuracy", "1.0"],
        ],
        [62, 107],
    )
    heading(story, styles, "무엇을 증명하고 무엇을 증명하지 않는가", 2)
    data_table(
        story,
        styles,
        [
            ["증명 가능", "증명 불가능"],
            ["forward/backward가 연결됨", "새 환자에 대한 generalization"],
            ["target와 logits contract가 맞음", "임상적 유용성"],
            ["optimizer가 해당 sample을 학습 가능", "domain shift robustness"],
            ["metric implementation이 완벽 예측에서 1.0", "test-set 성능"],
        ],
        [84.5, 84.5],
    )
    heading(story, styles, "Tiny overfit 실패 진단", 2)
    bullets(
        story,
        styles,
        [
            "Loss가 전혀 감소하지 않음: learning rate, detached graph, optimizer parameter 등록 확인.",
            "Accuracy만 높고 organ Dice가 0: background collapse와 class imbalance 확인.",
            "Loss는 감소하지만 metric 고정: argmax axis, label mapping, metric class ID 확인.",
            "출력 shape 불일치: encoder–decoder spatial 계산과 final head channel K 확인.",
        ],
    )

    # Part 2 checkpoint
    new_page(story)
    heading(story, styles, "Part 2 Checkpoint · U-Net을 그림 없이 추적하기")
    heading(story, styles, "필수 설명", 2)
    bullets(
        story,
        styles,
        [
            "왜 pooling은 receptive field와 context를 늘리지만 boundary localization을 희생하는가?",
            "Skip feature와 pooled feature는 각각 어떤 shape이며 왜 둘 다 필요한가?",
            "Decoder에서 upsample 후 skip과 `dim=1`로 concatenate하는 이유는?",
            "Final 1×1 convolution이 spatial size를 유지하면서 K logits를 만드는 원리는?",
            "Tiny overfit Dice 1.0을 얻어도 논문 성능 주장으로 사용할 수 없는 이유는?",
        ],
    )
    heading(story, styles, "One-level shape drill", 2)
    code_block(
        story,
        styles,
        "문제: [B=2,C=1,H=64,W=64]\n"
        "DoubleConv out=16 → Pool(2) → Bottleneck out=32\n"
        "→ UpConv out=16 → Skip concat → DoubleConv out=16 → Head K=3\n\n"
        "정답:\n"
        "skip [2,16,64,64], pooled [2,16,32,32]\n"
        "bottleneck [2,32,32,32], up [2,16,64,64]\n"
        "concat [2,32,64,64], logits [2,3,64,64]\n"
        "prediction [2,64,64]",
    )
    callout(
        story,
        styles,
        "연구 습관",
        "Network diagram의 화살표만 보지 말고 모든 node 옆에 Tensor shape를 적는다. 대부분의 구현 오류는 shape trace에서 먼저 드러난다.",
    )

    # Part 3 memory
    new_page(story)
    heading(story, styles, "Part 3. Volumetric Learning and Patch Mechanics")
    heading(story, styles, "3.1 Conv3D Tensor Flow와 Memory", 2)
    paragraph(
        story,
        styles,
        "Conv3D는 slice 하나가 아니라 depth까지 동시에 처리하므로 3D context를 학습하지만 activation memory가 급격히 증가한다. 8 GB GPU에서는 architecture보다 먼저 patch size와 batch size의 현실성을 계산해야 한다.",
    )
    formula(story, styles, "Raw tensor memory [bytes] = number_of_elements * bytes_per_element")
    data_table(
        story,
        styles,
        [
            ["Tensor", "FP32", "FP16", "변화"],
            ["[1,32,32,64,64]", "16 MiB", "8 MiB", "base"],
            ["Batch ×2", "32 MiB", "16 MiB", "2×"],
            ["Depth ×2", "32 MiB", "16 MiB", "2×"],
            ["H와 W ×2", "64 MiB", "32 MiB", "4×"],
            ["D,H,W 모두 ×2", "128 MiB", "64 MiB", "8×"],
        ],
        [58, 37, 37, 37],
    )
    paragraph(
        story,
        styles,
        "실습에서 `[1,8,32,64,64]` 3D output은 4.0 MiB였고 대응 2D output보다 32배 컸다. 이는 depth=32만큼 spatial element가 추가됐기 때문이다.",
    )
    heading(story, styles, "Raw output memory ≠ training peak", 2)
    data_table(
        story,
        styles,
        [
            ["항목", "실습 측정", "의미"],
            ["Returned outputs", "4.5 MiB", "skip + pooled Tensor 자체"],
            ["Additional training peak", "34.0 MiB", "forward 저장 + gradient + workspace 등"],
            ["Peak allocated", "34.5 MiB", "PyTorch가 실제 Tensor에 할당"],
            ["Peak reserved", "46.0 MiB", "CUDA allocator가 확보한 pool"],
        ],
        [55, 44, 70],
    )
    callout(
        story,
        styles,
        "용량 판단",
        "Parameter memory만 보고 학습 가능하다고 결론 내리지 말 것. Activation, backward graph, optimizer state, temporary workspace와 allocator reserve를 함께 측정.",
        PALE_RED,
        RED,
    )

    # Part 3 sampling
    new_page(story)
    heading(story, styles, "3.2 Crop, Padding과 Patch Sampling")
    paragraph(
        story,
        styles,
        "Full 3D volume을 GPU에 올릴 수 없으면 작은 patch를 학습 단위로 사용한다. 이때 crop 함수의 geometry와 patch center sampling policy가 실제 학습 분포를 결정한다.",
    )
    heading(story, styles, "Center-based crop", 2)
    code_block(
        story,
        styles,
        "volume [B,C,D,H,W]\n"
        "center = (cz,cy,cx), patch = (pD,pH,pW)\n"
        "start = center - floor(patch/2)\n"
        "end   = start + patch_size\n"
        "volume 밖 영역은 먼저 계산한 뒤 pad",
    )
    bullets(
        story,
        styles,
        [
            "중앙 patch: `[1,1,8,10,12]`에서 center `(4,5,6)`, patch `(4,6,8)` → `[1,1,4,6,8]`.",
            "경계 patch: center `(0,1,1)`에서 부족한 앞쪽 영역을 padding해 요청 shape 유지.",
            "Image padding과 label padding value는 의미가 다름. CT는 preprocessing convention, label은 background/ignore policy와 일치 필요.",
        ],
    )
    heading(story, styles, "Case selection과 patch-center selection", 2)
    data_table(
        story,
        styles,
        [
            ["단계", "질문", "예"],
            ["Case selection", "어느 patient/volume을 뽑는가?", "dataset sampler"],
            ["Center selection", "선택된 volume의 어디를 자르는가?", "uniform 또는 foreground"],
        ],
        [38, 67, 64],
    )
    heading(story, styles, "Uniform vs foreground sampling", 2)
    data_table(
        story,
        styles,
        [
            ["Sampler", "Center 선택", "Foreground hit (실습 1000회)", "효과"],
            ["Uniform", "전체 voxel에서 균일", "43/1000 = 4.3%", "대부분 background"],
            ["Foreground", "양성 voxel 중 선택", "1000/1000 = 100%", "organ signal 보장"],
        ],
        [35, 47, 51, 36],
    )
    callout(
        story,
        styles,
        "중요한 trade-off",
        "Foreground oversampling은 rare organ을 더 자주 보게 하지만 실제 prevalence를 바꾼다. 항상 foreground만 뽑는 것이 정답이 아니라 비율을 실험 변수로 통제해야 한다.",
        PALE_AMBER,
        AMBER,
    )

    # Part 3 sliding window
    new_page(story)
    heading(story, styles, "3.3 Sliding-Window Inference")
    paragraph(
        story,
        styles,
        "Training은 patch 단위여도 평가는 full volume mask가 필요하다. Sliding-window inference는 겹치는 patch를 전체 volume에 다시 배치하고 같은 voxel의 prediction을 평균한다.",
    )
    heading(story, styles, "시작 좌표와 coverage", 2)
    bullets(
        story,
        styles,
        [
            "Regular starts: `0, step, 2·step, …`.",
            "마지막 start가 `image_size − patch_size`가 아니면 해당 값을 추가해 끝 경계까지 coverage.",
            "실습: volume `(20,32,40)`, patch `(8,16,16)`, step `(4,8,8)`.",
            "Starts: Z=4개, Y=3개, X=4개 → 총 4×3×4 = 48 patches.",
        ],
    )
    heading(story, styles, "Accumulation과 normalization", 2)
    formula(
        story,
        styles,
        "final_logits(v) = sum_p(logits_p(v)) / count(v)",
    )
    code_block(
        story,
        styles,
        "accumulated [B,K,D,H,W] = 0\n"
        "count_map  [1,1,D,H,W] = 0\n\n"
        "for patch_logits, (z,y,x):\n"
        "    accumulated[..., region] += patch_logits\n"
        "    count_map[..., region] += 1\n\n"
        "full_logits = accumulated / count_map\n"
        "prediction = full_logits.argmax(dim=1)",
    )
    heading(story, styles, "Identity reconstruction test", 2)
    data_table(
        story,
        styles,
        [
            ["검사", "결과", "증명"],
            ["Minimum count", "1", "미coverage voxel 없음"],
            ["Maximum count", "8", "내부 voxel이 최대 8 patch에 포함"],
            ["13,460 × 8", "107,680", "누적 좌표가 정확"],
            ["Normalize 후 reconstruction", "max/mean error 0", "조립 pipeline 정확"],
        ],
        [61, 42, 66],
    )
    callout(
        story,
        styles,
        "해석 한계",
        "Identity patch를 완벽히 복원한 것은 inference 조립 코드가 맞다는 뜻이다. 실제 model accuracy나 새 환자 generalization을 증명하지 않는다.",
        PALE_RED,
        RED,
    )

    # Part 4.1 NIfTI and affine
    new_page(story)
    heading(story, styles, "Part 4. Medical Image Geometry and CT")
    heading(story, styles, "4.1 NIfTI Array와 Affine", 2)
    paragraph(
        story,
        styles,
        "NIfTI의 voxel array는 intensity를 저장하지만 array index 자체는 환자의 해부학적 위치가 아니다. 4×4 affine이 voxel index `(i,j,k)`를 millimeter 단위 physical coordinate `(x,y,z)`로 연결한다.",
    )
    formula(story, styles, "[x,y,z,1]^T = affine * [i,j,k,1]^T")
    heading(story, styles, "실습 geometry", 2)
    code_block(
        story,
        styles,
        "array shape   (I,J,K) = (6,8,10)\n"
        "voxel spacing (2.0,1.5,3.0) mm\n"
        "origin XYZ    (-120,-90,-60) mm\n"
        "axis codes    ('R','A','S')\n\n"
        "affine =\n"
        "[[2.0, 0.0, 0.0, -120.0],\n"
        " [0.0, 1.5, 0.0,  -90.0],\n"
        " [0.0, 0.0, 3.0,  -60.0],\n"
        " [0.0, 0.0, 0.0,    1.0]]",
    )
    heading(story, styles, "Index-to-physical 계산", 2)
    formula(story, styles, "index (2,3,4) -> (-120+2*2, -90+3*1.5, -60+4*3)")
    formula(story, styles, "physical XYZ = (-116.0, -85.5, -48.0) mm")
    data_table(
        story,
        styles,
        [
            ["객체", "좌표계", "단위", "질문"],
            ["Array index", "IJK", "voxel", "배열에서 몇 번째 원소인가?"],
            ["Physical coordinate", "XYZ", "mm", "환자 공간의 어디인가?"],
            ["Affine", "IJK → XYZ", "mm/voxel + 방향 + origin", "두 좌표계를 어떻게 연결하는가?"],
        ],
        [38, 32, 47, 52],
    )
    callout(
        story,
        styles,
        "검증 결과",
        "세 voxel의 scratch affine 계산이 nibabel 결과와 일치. Inverse affine round trip의 maximum index error도 0.0.",
    )
    heading(story, styles, "Continuous index를 버리지 말 것", 2)
    paragraph(
        story,
        styles,
        "Physical coordinate가 항상 voxel center에 놓이는 것은 아니다. 실습의 `(-115,-85.5,-48) mm`는 continuous index `(2.5,3,4)`로 복원됐다. Resampling과 interpolation에서는 반올림 전 continuous coordinate가 필요하다.",
    )
    callout(
        story,
        styles,
        "Geometry contract",
        "Shape만 같다고 image와 label이 정렬된 것은 아니다. Shape, spacing, affine, orientation을 함께 검사해야 같은 physical anatomy를 가리킨다고 말할 수 있다.",
        PALE_RED,
        RED,
    )

    # Part 4.2 orientation and three-plane viewing
    new_page(story)
    heading(story, styles, "4.2 Orientation과 Three-Plane Viewer")
    paragraph(
        story,
        styles,
        "Array axis 번호와 해부학적 방향은 동일한 개념이 아니다. `nib.aff2axcodes(affine)`가 각 voxel axis의 증가 방향을 R/L, A/P, S/I code로 알려준다. 실습 volume은 RAS orientation이었다.",
    )
    data_table(
        story,
        styles,
        [
            ["Plane", "고정 index", "추출 후 Shape", "표시 전 transpose"],
            ["Sagittal", "I", "[J,K] = [50,60]", "[K,J] = [60,50]"],
            ["Coronal", "J", "[I,K] = [40,60]", "[K,I] = [60,40]"],
            ["Axial", "K", "[I,J] = [40,50]", "[J,I] = [50,40]"],
        ],
        [34, 34, 51, 50],
    )
    code_block(
        story,
        styles,
        "volume[I,J,K]\n"
        "  sagittal = volume[i, :, :]   # [J,K]\n"
        "  coronal  = volume[:, j, :]   # [I,K]\n"
        "  axial    = volume[:, :, k]   # [I,J]\n\n"
        "display: transpose + origin='lower' + physical extent",
    )
    paragraph(
        story,
        styles,
        "Transpose는 저장된 voxel을 재배열해 화면의 세로·가로 axis에 맞추는 표시 단계다. Orientation 정보 없이 무조건 flip하거나 transpose하면 좌우가 뒤집힌 것처럼 보일 수 있으므로 axis code와 affine을 먼저 확인한다.",
    )
    heading(story, styles, "RAS와 LPS에서 index가 달라도 같은 점일 수 있음", 2)
    data_table(
        story,
        styles,
        [
            ["표현", "Landmark index", "Intensity", "Physical XYZ"],
            ["RAS", "[33,25,30]", "300", "[26,0,0] mm"],
            ["LPS", "[6,24,30]", "300", "[26,0,0] mm"],
        ],
        [38, 44, 35, 52],
    )
    formula(story, styles, "RAS index != LPS index,  but  RAS world XYZ = LPS world XYZ")
    paragraph(
        story,
        styles,
        "I/J array를 뒤집으면 landmark index는 달라진다. 동시에 affine을 갱신하면 두 index가 같은 환자 위치를 가리킨다. 따라서 index equality가 아니라 affine을 적용한 physical-coordinate equality로 geometry 보존을 검증한다.",
    )
    callout(
        story,
        styles,
        "Canonical RAS 검증",
        "LPS volume을 `nib.as_closest_canonical`로 RAS 복원한 결과 axis code RAS, array 일치, affine 일치, maximum voxel difference 0.0.",
    )
    callout(
        story,
        styles,
        "의료영상 안전 규칙",
        "좌우 laterality가 있는 organ은 viewer 모양만 보고 판단하지 않는다. Orientation code, affine, landmark physical coordinate를 함께 확인한다.",
        PALE_RED,
        RED,
    )

    # Part 4 checkpoint
    new_page(story)
    heading(story, styles, "Part 4.1–4.2 Checkpoint · Geometry를 말로 추적하기")
    bullets(
        story,
        styles,
        [
            "Voxel index `(i,j,k)`와 physical coordinate `(x,y,z)`는 무엇이 다른가?",
            "Affine의 diagonal과 마지막 column은 axis-aligned 예제에서 각각 무엇을 뜻하는가?",
            "Index `(2,3,4)`가 `(-116,-85.5,-48) mm`로 변환되는 계산을 직접 적을 수 있는가?",
            "Inverse affine 결과가 integer가 아닌 continuous index일 수 있는 이유는?",
            "Sagittal/coronal/axial slice에서 고정하는 array axis와 남는 Shape는 무엇인가?",
            "Slice display 전에 transpose가 필요한 이유와 무조건 flip하면 안 되는 이유는?",
            "RAS와 LPS landmark index가 달라도 physical point가 같을 수 있는 이유는?",
            "Canonical reorientation 뒤 무엇을 비교해야 geometry가 보존됐다고 말할 수 있는가?",
        ],
    )
    heading(story, styles, "Geometry debugging ladder", 2)
    code_block(
        story,
        styles,
        "1. array shape와 dtype 확인\n"
        "2. header spacing 확인\n"
        "3. full 4x4 affine 확인\n"
        "4. orientation axis codes 확인\n"
        "5. known landmark: index -> physical XYZ 확인\n"
        "6. image와 label의 geometry contract 비교\n"
        "7. reorientation/resampling 후 physical equality 검증\n"
        "8. axial/coronal/sagittal overlay 시각 검사",
    )
    callout(
        story,
        styles,
        "현재 경계",
        "4.1–4.2는 geometry를 읽고 방향을 검증하는 단계까지 완료. Voxel spacing을 실제로 바꾸는 interpolation policy는 다음 Lesson 4.3에서 학습.",
        PALE_AMBER,
        AMBER,
    )

    # Integration
    new_page(story)
    heading(story, styles, "Parts 1–4.2 통합 · 하나의 Geometry-Aware 3D System")
    data_table(
        story,
        styles,
        [
            ["단계", "Tensor/Data", "지켜야 할 invariant", "실패 징후"],
            ["Patch sampling", "CT/label patch", "동일 좌표, 동일 spatial shape", "image-label misalignment"],
            ["Geometry load", "array + affine", "spacing/orientation/physical alignment", "좌우 반전, shifted label"],
            ["Network", "[B,C,pD,pH,pW]", "logits channel=K", "head channel 오류"],
            ["Training loss", "logits + [B,pD,pH,pW]", "target long, class ID 범위", "CE shape/dtype error"],
            ["Patch inference", "patch logits", "global origin 보존", "shifted prediction"],
            ["Accumulation", "sum + count map", "모든 count &gt; 0", "NaN/holes"],
            ["Decision", "full logits", "argmax over K only", "spatial axis 소실"],
            ["Evaluation", "mask + target", "frozen empty/aggregation policy", "score 재현 불가"],
        ],
        [31, 44, 51, 43],
    )
    heading(story, styles, "실전 debugging ladder", 2)
    code_block(
        story,
        styles,
        "Level 1  Tensor contract unit tests\n"
        "Level 2  Stable loss + metric test cases\n"
        "Level 3  Single block forward/backward\n"
        "Level 4  Tiny overfit on 1–2 samples\n"
        "Level 5  Patch crop/padding visualization\n"
        "Level 6  Identity sliding-window reconstruction\n"
        "Level 7  Validation cohort evaluation\n"
        "Level 8  Locked test only after method freeze",
    )
    paragraph(
        story,
        styles,
        "아래 level이 실패한 상태에서 위 level의 성능을 해석하면 원인 분리가 불가능하다. OLES3D 연구는 이 ladder를 고정해 method 변화와 implementation bug를 분리한다.",
    )
    heading(story, styles, "Memory-aware design rule", 2)
    formula(
        story,
        styles,
        "(2*D)*(2*H)*(2*W) = 8*(D*H*W)  =>  activation elements ~= 8x",
    )
    bullets(
        story,
        styles,
        [
            "OOM이면 무조건 backbone부터 줄이지 말고 batch, patch, channels, precision, checkpointing 순서로 trade-off 기록.",
            "Patch 크기를 바꾸면 memory뿐 아니라 organ context와 foreground hit probability도 변함.",
            "Training patch policy와 inference sliding-window policy는 목적이 다르므로 별도 parameter로 관리.",
        ],
    )

    # Self-test
    new_page(story)
    heading(story, styles, "자가시험 · 답을 가리고 직접 설명하기")
    questions = [
        "1. `[2,10,32,64,64]` logits에서 target과 prediction shape는 무엇인가?",
        "2. `argmax(dim=2)`가 잘못된 이유를 axis 의미로 설명하라.",
        "3. Stable softmax에서 maximum logit을 빼는 이유와 결과가 보존되는 이유는?",
        "4. TP=2, FP=1, FN=1일 때 Dice와 IoU를 계산하라.",
        "5. Background-only prediction의 accuracy가 높아도 실패인 이유는?",
        "6. `[1,16,32,64,64]` FP32 Tensor의 raw memory는 몇 MiB인가?",
        "7. D, H, W를 모두 2배로 만들면 activation element 수는 몇 배인가?",
        "8. U-Net skip connection이 전달하는 정보와 concatenate 조건은?",
        "9. Tiny overfit Dice 1.0으로 새 환자 성능을 결론 낼 수 없는 이유는?",
        "10. Uniform sampling과 foreground sampling의 center 선택 차이는?",
        "11. Sliding-window에서 count map이 필요한 이유는?",
        "12. Identity reconstruction error 0이 증명하는 것과 증명하지 않는 것은?",
        "13. Training target에는 K가 없지만 model logits에는 K가 있는 이유는?",
        "14. Raw Tensor memory와 GPU peak reserved memory가 다른 이유는?",
        "15. Voxel index `(i,j,k)`와 physical coordinate `(x,y,z)`의 차이는?",
        "16. Index `(2,3,4)`, spacing `(2,1.5,3) mm`, origin `(-120,-90,-60) mm`의 physical coordinate는?",
        "17. Sagittal, coronal, axial slice에서 각각 고정하는 array axis는?",
        "18. RAS와 LPS에서 landmark index가 달라도 같은 anatomy를 가리킬 수 있는 이유는?",
        "19. Canonical RAS 변환 후 geometry 보존을 검증할 항목은?",
        "20. Image와 label의 Shape만 같아도 정렬됐다고 결론 낼 수 없는 이유는?",
    ]
    for question in questions:
        story.append(Paragraph(question, styles["question"]))
    callout(
        story,
        styles,
        "통과 방법",
        "숫자만 맞히지 말고 Tensor shape와 data flow를 그림으로 적은 뒤, 잘못된 구현이 어떤 output을 만드는지까지 설명.",
        PALE_AMBER,
        AMBER,
    )

    # Answers
    new_page(story)
    heading(story, styles, "자가시험 정답과 마지막 Cheat Sheet")
    answers = [
        "1. Target `[2,32,64,64]`, prediction `[2,32,64,64]`.",
        "2. dim=2는 D를 제거해 `[B,K,H,W]`를 만들고 class axis K를 남김. 제거 대상은 dim=1의 K.",
        "3. exp overflow 방지. 모든 logit에 같은 상수를 빼므로 softmax의 상대 비율과 최종 probability는 동일.",
        "4. Dice=4/6=0.6667, IoU=2/4=0.5.",
        "5. TN이 압도해 accuracy를 지배하지만 organ TP는 0이고 FN은 클 수 있음.",
        "6. 1×16×32×64×64×4 bytes = 8,388,608 bytes = 8 MiB.",
        "7. 2×2×2=8배.",
        "8. Pooling 이전의 high-resolution localization 정보. Batch/spatial shape가 같아야 하고 channel은 합산됨.",
        "9. Training sample 암기 test일 뿐 unseen patient distribution을 평가하지 않았기 때문.",
        "10. Uniform은 전체 voxel에서 center 선택, foreground는 양성 class voxel 중 center 선택.",
        "11. 같은 global voxel에 여러 patch logits가 더해진 횟수로 나눠 평균하기 위해 필요.",
        "12. 좌표·누적·정규화 pipeline의 정확성은 증명. Model accuracy/generalization은 증명하지 않음.",
        "13. Logits는 K개 후보 score, target은 그중 정답 class ID 하나.",
        "14. Training은 saved activation, gradient, optimizer/workspace와 CUDA allocator reserve까지 필요.",
        "15. IJK는 array element 위치이고 XYZ는 affine으로 변환한 환자 공간의 millimeter 위치.",
        "16. `(-116,-85.5,-48) mm`. 각 axis에서 `origin + index×spacing` 계산.",
        "17. Sagittal은 I, coronal은 J, axial은 K를 고정.",
        "18. Array를 flip/reorder하면서 affine도 함께 갱신하면 서로 다른 index가 동일한 physical XYZ를 가리킬 수 있음.",
        "19. Axis code와 함께 landmark physical coordinate, canonical array, affine 및 voxel difference 비교.",
        "20. 서로 다른 spacing, origin, direction/orientation의 array가 우연히 같은 Shape일 수 있기 때문.",
    ]
    for answer in answers:
        story.append(Paragraph(answer, styles["question"]))
    story.append(Spacer(1, 2 * mm))
    story.append(HRFlowable(width="100%", thickness=0.8, color=LINE))
    story.append(Spacer(1, 3 * mm))
    heading(story, styles, "최종 8줄 요약", 2)
    bullets(
        story,
        styles,
        [
            "Segmentation = voxel마다 수행하는 K-class classification.",
            "Training contract = logits `[B,K,D,H,W]` + target `[B,D,H,W]`.",
            "U-Net = context를 얻는 encoder + resolution을 복원하는 decoder + localization을 전달하는 skip.",
            "Tiny overfit = generalization test가 아니라 pipeline integrity test.",
            "3D memory는 D·H·W에 비례하므로 patch training이 현실적인 기본 단위.",
            "Full-volume inference = overlapping logits 누적 ÷ count/weight map → argmax over K.",
            "NIfTI geometry = voxel array + 4×4 affine; index와 patient-space millimeter를 분리.",
            "Orientation 변경은 index를 바꾸지만 affine을 함께 갱신하면 physical anatomy는 보존.",
        ],
    )
    callout(
        story,
        styles,
        "다음 Part",
        "다음 Lesson 4.3에서는 spacing을 바꿀 때 output Shape와 affine을 함께 갱신하고, CT image에는 trilinear, segmentation label에는 nearest-neighbor interpolation을 적용하는 이유를 검증한다.",
    )
    paragraph(
        story,
        styles,
        "Source notebooks: studies/prerequisites/part01_segmentation_fundamentals · part02_unet_from_scratch · part03_volumetric_learning · part04_medical_image_geometry_ct/01–02",
        "small",
    )

    document = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=19 * mm,
        title="OLES3D Cumulative Core Review Parts 1–4.2",
        author="Park Yongmin",
        subject="3D Medical Image Segmentation Prerequisite Cumulative Review",
    )
    document.build(
        story,
        onFirstPage=page_chrome,
        onLaterPages=page_chrome,
    )
    return OUTPUT_PATH


if __name__ == "__main__":
    generated_path = build_pdf()
    print(generated_path)
