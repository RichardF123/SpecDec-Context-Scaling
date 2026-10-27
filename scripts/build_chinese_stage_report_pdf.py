from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "specdec_chinese_stage_report_2026-10-26.md"
OUTPUT = ROOT / "docs" / "pdf" / "specdec_chinese_stage_report_2026-10-26.pdf"

PAGE_W, PAGE_H = A4
LEFT = RIGHT = 18 * mm
TOP = 18 * mm
BOTTOM = 17 * mm
CONTENT_W = PAGE_W - LEFT - RIGHT

INK = colors.HexColor("#243844")
TEAL = colors.HexColor("#218C87")
PURPLE = colors.HexColor("#8167A9")
PALE_BLUE = colors.HexColor("#EAF3F7")
PALE_TEAL = colors.HexColor("#EAF5F2")
PALE_GOLD = colors.HexColor("#F8F2E5")
GRID = colors.HexColor("#CAD5DB")
MUTED = colors.HexColor("#617681")


def register_fonts() -> None:
    font_dir = Path(r"C:\Windows\Fonts")
    pdfmetrics.registerFont(TTFont("DengXian", str(font_dir / "Deng.ttf")))
    pdfmetrics.registerFont(TTFont("DengXian-Bold", str(font_dir / "Dengb.ttf")))
    pdfmetrics.registerFont(TTFont("SimHei", str(font_dir / "simhei.ttf")))
    pdfmetrics.registerFontFamily(
        "DengXian",
        normal="DengXian",
        bold="DengXian-Bold",
        italic="DengXian",
        boldItalic="DengXian-Bold",
    )


class ReportDoc(BaseDocTemplate):
    def __init__(self, filename: str):
        super().__init__(
            filename,
            pagesize=A4,
            leftMargin=LEFT,
            rightMargin=RIGHT,
            topMargin=TOP,
            bottomMargin=BOTTOM,
            title="SpecDec 项目阶段性技术报告",
            author="SpecDec Project Team",
        )
        frame = Frame(LEFT, BOTTOM, CONTENT_W, PAGE_H - TOP - BOTTOM, id="main")
        self.addPageTemplates(PageTemplate(id="body", frames=[frame], onPage=self._page))
        self._bookmark_id = 0

    def _page(self, canvas, doc):
        canvas.saveState()
        if doc.page > 1:
            canvas.setStrokeColor(GRID)
            canvas.setLineWidth(0.5)
            canvas.line(LEFT, PAGE_H - 12 * mm, PAGE_W - RIGHT, PAGE_H - 12 * mm)
            canvas.setFont("DengXian", 8)
            canvas.setFillColor(MUTED)
            canvas.drawString(LEFT, PAGE_H - 9 * mm, "SpecDec 项目阶段性技术报告")
            canvas.drawRightString(PAGE_W - RIGHT, 9 * mm, f"{doc.page}")
        canvas.restoreState()

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name in {"H1", "H2", "H3"}:
            # The Markdown H1 is rendered on the cover and intentionally skipped
            # in the body, so H2/H3 become the two PDF outline levels.
            level = {"H1": 0, "H2": 0, "H3": 1}[flowable.style.name]
            self._bookmark_id += 1
            key = f"h{self._bookmark_id}"
            text = flowable.getPlainText()
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=level, closed=False)


def styles():
    base = getSampleStyleSheet()
    body = ParagraphStyle(
        "BodyCN",
        parent=base["BodyText"],
        fontName="DengXian",
        fontSize=9.5,
        leading=15,
        textColor=INK,
        alignment=TA_LEFT,
        wordWrap="CJK",
        spaceAfter=6,
    )
    return {
        "body": body,
        "small": ParagraphStyle("SmallCN", parent=body, fontSize=7.5, leading=10, textColor=MUTED),
        "caption": ParagraphStyle(
            "CaptionCN", parent=body, fontSize=8.2, leading=12, textColor=MUTED,
            alignment=TA_LEFT, spaceBefore=3, spaceAfter=10,
        ),
        "H1": ParagraphStyle(
            "H1", parent=body, fontName="DengXian-Bold", fontSize=17, leading=22,
            textColor=INK, spaceBefore=12, spaceAfter=9, keepWithNext=True,
        ),
        "H2": ParagraphStyle(
            "H2", parent=body, fontName="DengXian-Bold", fontSize=13, leading=18,
            textColor=TEAL, spaceBefore=12, spaceAfter=7, keepWithNext=True,
        ),
        "H3": ParagraphStyle(
            "H3", parent=body, fontName="DengXian-Bold", fontSize=10.5, leading=15,
            textColor=PURPLE, spaceBefore=8, spaceAfter=5, keepWithNext=True,
        ),
        "quote": ParagraphStyle(
            "QuoteCN", parent=body, leftIndent=10, rightIndent=8, borderColor=TEAL,
            borderWidth=1.5, borderPadding=7, backColor=PALE_TEAL, textColor=INK,
        ),
        "code": ParagraphStyle(
            "CodeCN", parent=body, fontName="Courier", fontSize=7.8, leading=11,
            leftIndent=7, rightIndent=7, borderPadding=6, backColor=colors.HexColor("#F2F4F5"),
        ),
        "table": ParagraphStyle("TableCN", parent=body, fontSize=7.2, leading=9.2, spaceAfter=0),
        "table_head": ParagraphStyle(
            "TableHeadCN", parent=body, fontName="DengXian-Bold", fontSize=7.2,
            leading=9.2, textColor=colors.white, spaceAfter=0,
        ),
    }


def inline_markup(text: str) -> str:
    text = text.strip()
    # Convert the small LaTeX subset used by the Markdown source into
    # Unicode/plain text before ReportLab escapes and lays out the paragraph.
    text = re.sub(r"\\text\{([^{}]+)\}", r"\1", text)
    for source, target in {
        r"\(": "", r"\)": "", r"\[": "", r"\]": "",
        r"\le": "≤", r"\ge": "≥", r"\delta": "δ", r"\Delta": "Δ",
        r"\in": "∈", r"\times": "×", r"\approx": "≈", r"\quad": "  ",
        r"\{": "{", r"\}": "}",
    }.items():
        text = text.replace(source, target)
    text = re.sub(r"_\{([^{}]+)\}", r"_\1", text)
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<link href="\2" color="#218C87">\1</link>', text)
    return text


def make_table(lines: list[str], st) -> Table:
    rows = []
    for i, line in enumerate(lines):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if i == 1 and all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
            continue
        style = st["table_head"] if not rows else st["table"]
        rows.append([Paragraph(inline_markup(c), style) for c in cells])
    ncols = len(rows[0])
    widths = [CONTENT_W / ncols] * ncols
    if ncols == 2:
        widths = [CONTENT_W * 0.27, CONTENT_W * 0.73]
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE_BLUE]),
        ("GRID", (0, 0), (-1, -1), 0.35, GRID),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def image_flow(path: Path) -> Image:
    img = Image(str(path))
    max_w = CONTENT_W
    max_h = 112 * mm
    scale = min(max_w / img.imageWidth, max_h / img.imageHeight)
    img.drawWidth = img.imageWidth * scale
    img.drawHeight = img.imageHeight * scale
    img.hAlign = "CENTER"
    return img


def parse_markdown(st):
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    story = []

    story.extend([
        Spacer(1, 25 * mm),
        Paragraph("SpecDec 项目阶段性技术报告", ParagraphStyle(
            "CoverTitle", fontName="DengXian-Bold", fontSize=26, leading=34,
            textColor=INK, alignment=TA_CENTER,
        )),
        Spacer(1, 8 * mm),
        Paragraph(
            "受限单卡环境下，固定模型对投机解码的上下文扩展、执行成本与接受行为分析",
            ParagraphStyle("CoverSub", fontName="DengXian", fontSize=14, leading=22,
                           textColor=TEAL, alignment=TA_CENTER, wordWrap="CJK"),
        ),
        Spacer(1, 18 * mm),
        Table([
            [Paragraph("硬件", st["table_head"]), Paragraph("RTX 4060 Laptop GPU, 8 GB", st["table_head"])],
            [Paragraph("模型对", st["table"]), Paragraph("Qwen2.5-1.5B-Instruct target + 0.5B-Instruct draft", st["table"])],
            [Paragraph("当前证据冻结", st["table"]), Paragraph("Confirmatory + Phase O + Acceptance Diagnostics", st["table"])],
            [Paragraph("日期", st["table"]), Paragraph("2026-10-26 (Asia/Hong_Kong)", st["table"])],
        ], colWidths=[42 * mm, 105 * mm], style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), INK),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE_BLUE]),
            ("GRID", (0, 0), (-1, -1), 0.5, GRID),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ])),
        Spacer(1, 28 * mm),
        Paragraph(
            "证据边界：正式性能结论来自 16 个冻结 family 的确认批次；Phase O 与 acceptance diagnostics 为开发集机制证据；relaxed verification 尚未运行。",
            ParagraphStyle("CoverNote", parent=st["quote"], alignment=TA_LEFT),
        ),
        PageBreak(),
    ])

    i = 0
    first_h1_skipped = False
    while i < len(lines):
        line = lines[i].rstrip()
        if not line:
            i += 1
            continue
        if line.startswith("# "):
            if not first_h1_skipped:
                first_h1_skipped = True
            else:
                story.append(Paragraph(inline_markup(line[2:]), st["H1"]))
            i += 1
            continue
        if line.startswith("## "):
            story.append(Paragraph(inline_markup(line[3:]), st["H2"]))
            i += 1
            continue
        if line.startswith("### "):
            story.append(Paragraph(inline_markup(line[4:]), st["H3"]))
            i += 1
            continue
        if line.startswith("**副标题") or line.startswith("**更新日期") or line.startswith("**项目成员"):
            i += 1
            continue
        if line.startswith("!["):
            m = re.match(r"!\[([^]]*)\]\(([^)]+)\)", line)
            if m:
                path = (SOURCE.parent / m.group(2)).resolve()
                if path.exists():
                    story.append(Spacer(1, 2 * mm))
                    story.append(image_flow(path))
            i += 1
            continue
        if line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i].strip())
                i += 1
            story.append(Spacer(1, 2 * mm))
            story.append(make_table(block, st))
            story.append(Spacer(1, 4 * mm))
            continue
        if line.startswith("> "):
            story.append(Paragraph(inline_markup(line[2:]), st["quote"]))
            i += 1
            continue
        if line.startswith("```"):
            code = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(html.escape(lines[i]))
                i += 1
            i += 1
            story.append(Paragraph("<br/>".join(code), st["code"]))
            continue
        if re.match(r"^[-*] ", line):
            items = []
            while i < len(lines) and re.match(r"^[-*] ", lines[i].strip()):
                items.append(ListItem(Paragraph(inline_markup(lines[i].strip()[2:]), st["body"]), leftIndent=10))
                i += 1
            story.append(ListFlowable(items, bulletType="bullet", start="circle", leftIndent=18, bulletFontName="DengXian"))
            story.append(Spacer(1, 3))
            continue
        if re.match(r"^\d+\. ", line):
            items = []
            while i < len(lines) and re.match(r"^\d+\. ", lines[i].strip()):
                content = re.sub(r"^\d+\. ", "", lines[i].strip())
                items.append(ListItem(Paragraph(inline_markup(content), st["body"]), leftIndent=10))
                i += 1
            story.append(ListFlowable(items, bulletType="1", leftIndent=22, bulletFontName="DengXian"))
            story.append(Spacer(1, 3))
            continue
        if line == "\[":
            math_lines = []
            i += 1
            while i < len(lines) and lines[i].strip() != "\]":
                math_lines.append(lines[i].strip())
                i += 1
            i += 1
            story.append(Paragraph(inline_markup(" ".join(math_lines)), st["quote"]))
            continue

        para = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(
            r"^(#|\||!\[|>|```|[-*] |\d+\. |\\\[)", lines[i].strip()
        ):
            para.append(lines[i].strip())
            i += 1
        story.append(Paragraph(inline_markup(" ".join(para)), st["body"]))

    return story


def main() -> None:
    register_fonts()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    st = styles()
    doc = ReportDoc(str(OUTPUT))
    doc.build(parse_markdown(st))
    print(OUTPUT)


if __name__ == "__main__":
    main()

