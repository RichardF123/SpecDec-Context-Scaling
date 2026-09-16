"""generate_specdec_explained_pdf.py — 把 docs/specdec_explained.md 渲染成中文 PDF。

为什么单独写一个转换器（而不是像 proposal 那样把内容写在脚本里）：
  这份文档 700+ 行、含大量代码块与表格，内容需要用可 diff、可评审的 markdown 维护；
  脚本只负责排版。输出放在 docs/pdf/（**受版本控制**，便于直接在 GitHub 上看），
  因为仓库 .gitignore 排除了 output/。

支持的 markdown 子集（够用即止，不做通用解析）：
  # / ## / ### 标题 ；普通段落 ；- 列表 ；1. 编号列表 ；``` 代码块 ；
  | 表格 |（第二行 |---| 分隔，可选 <!-- widths="20,40,40" --> 指定列宽百分比）；
  > 引用 ； --- 水平线 ；行内 **粗体** 与 `代码`
用法：python scripts/generate_specdec_explained_pdf.py
"""
from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, KeepTogether, PageBreak,
                               PageTemplate, Paragraph, Spacer, Table, TableStyle,
                               XPreformatted)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "specdec_explained.md"
OUT = ROOT / "docs" / "pdf" / "specdec_speculative_decoding_explained.pdf"

F_CN = "MSYH"
F_CN_B = "MSYHBD"
F_MONO = "CONSOLA"
FONTS = [
    (F_CN, "C:/Windows/Fonts/msyh.ttc", 0),
    (F_CN_B, "C:/Windows/Fonts/msyhbd.ttc", 0),
    (F_MONO, "C:/Windows/Fonts/consola.ttf", None),
]

ACCENT = colors.HexColor("#1f4e79")
RULE = colors.HexColor("#b8c4d0")
CODE_BG = colors.HexColor("#f4f6f8")
TBL_HEAD_BG = colors.HexColor("#e8eef4")


def register_fonts() -> None:
    for name, path, idx in FONTS:
        if idx is None:
            pdfmetrics.registerFont(TTFont(name, path))
        else:
            pdfmetrics.registerFont(TTFont(name, path, subfontIndex=idx))
    registerFontFamily(F_CN, normal=F_CN, bold=F_CN_B, italic=F_CN, boldItalic=F_CN_B)


def styles() -> dict:
    s = {}
    s["title"] = ParagraphStyle("title", fontName=F_CN_B, fontSize=17, leading=24,
                                alignment=TA_CENTER, textColor=ACCENT, spaceAfter=4)
    s["h2"] = ParagraphStyle("h2", fontName=F_CN_B, fontSize=13, leading=19,
                             textColor=ACCENT, spaceBefore=2, spaceAfter=6)
    s["h3"] = ParagraphStyle("h3", fontName=F_CN_B, fontSize=11, leading=16,
                             textColor=colors.HexColor("#333333"), spaceBefore=8,
                             spaceAfter=4)
    s["body"] = ParagraphStyle("body", fontName=F_CN, fontSize=9.6, leading=15.2,
                               alignment=TA_JUSTIFY, wordWrap="CJK", spaceAfter=5.5)
    s["bullet"] = ParagraphStyle("bullet", parent=s["body"], leftIndent=13,
                                 bulletIndent=3, spaceAfter=3.4,
                                 bulletFontName=F_CN, bulletFontSize=9.6)
    s["num"] = ParagraphStyle("num", parent=s["body"], leftIndent=13, bulletIndent=3,
                              spaceAfter=3.4, bulletFontName=F_CN, bulletFontSize=9.6)
    s["quote"] = ParagraphStyle("quote", parent=s["body"], leftIndent=12,
                                textColor=colors.HexColor("#444444"), spaceAfter=4)
    s["cell"] = ParagraphStyle("cell", fontName=F_CN, fontSize=8.3, leading=11.6,
                               wordWrap="CJK")
    s["cellh"] = ParagraphStyle("cellh", parent=s["cell"], fontName=F_CN_B)
    s["code"] = ParagraphStyle("code", fontName=F_MONO, fontSize=7.5, leading=9.9,
                               textColor=colors.HexColor("#1a1a1a"))
    return s


def inline(text: str, code_size: float = 9.0) -> str:
    """转义 + 行内 `code` 与 **bold**（先切 code，避免代码里的 * 被当格式）。"""
    out = []
    for part in re.split(r"(`[^`]+`)", text):
        if len(part) >= 2 and part.startswith("`") and part.endswith("`"):
            code = html.escape(part[1:-1])
            fam = F_MONO if code.isascii() else F_CN
            out.append(f'<font name="{fam}" size="{code_size}">{code}</font>')
        else:
            esc = html.escape(part)
            esc = re.sub(r"\*\*(.+?)\*\*", rf'<font name="{F_CN_B}">\1</font>', esc)
            out.append(esc)
    return "".join(out)


def code_flowable(lines: list[str], st: dict):
    """代码块：ASCII 行用 Consolas、含非 ASCII 的行整行用 MSYH（避免混排错位）。"""
    txt = []
    for ln in lines:
        e = html.escape(ln)
        fam = F_MONO if ln.isascii() else F_CN
        txt.append(f'<font name="{fam}">{e}</font>' if e else "")
    body = XPreformatted("\n".join(txt), st["code"])
    return Table([[body]], colWidths=[None],
                 style=TableStyle([
                     ("BACKGROUND", (0, 0), (-1, -1), CODE_BG),
                     ("BOX", (0, 0), (-1, -1), 0.4, RULE),
                     ("LEFTPADDING", (0, 0), (-1, -1), 6),
                     ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                     ("TOPPADDING", (0, 0), (-1, -1), 5),
                     ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                 ]))


def table_flowable(rows: list[list[str]], widths: list[float] | None, st: dict,
                   avail: float):
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    data = []
    for ri, row in enumerate(rows):
        style = st["cellh"] if ri == 0 else st["cell"]
        data.append([Paragraph(inline(c, 7.8), style) for c in row])
    if widths and len(widths) == ncol:
        tot = sum(widths)
        col = [avail * w / tot for w in widths]
    else:
        col = [avail / ncol] * ncol
    t = Table(data, colWidths=col, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TBL_HEAD_BG),
        ("GRID", (0, 0), (-1, -1), 0.35, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def parse(md: str, st: dict, avail: float):
    flow = []
    lines = md.splitlines()
    i = 0
    first_h2 = True
    pending_widths = None
    while i < len(lines):
        ln = lines[i]
        stripped = ln.strip()

        if not stripped:
            i += 1
            continue

        # 代码块
        if stripped.startswith("```"):
            i += 1
            buf = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            flow.append(Spacer(1, 2))
            flow.append(code_flowable(buf, st))
            flow.append(Spacer(1, 7))
            continue

        # 表格（列宽指令由上一行的 pending_widths 提供）
        if stripped.startswith("|"):
            widths = pending_widths
            pending_widths = None
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                raw = lines[i].strip().strip("|")
                cells = [c.strip() for c in raw.split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                    rows.append(cells)
                i += 1
            flow.append(Spacer(1, 2))
            flow.append(table_flowable(rows, widths, st, avail))
            flow.append(Spacer(1, 8))
            continue

        # 列宽指令
        m = re.fullmatch(r"<!--\s*widths=\"([\d,\.]+)\"\s*-->", stripped)
        if m:
            pending_widths = [float(x) for x in m.group(1).split(",")]
            i += 1
            continue

        if stripped == "---":
            flow.append(Spacer(1, 3))
            flow.append(Table([[""]], colWidths=[avail], rowHeights=[0.6],
                              style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), RULE)])))
            flow.append(Spacer(1, 8))
            i += 1
            continue

        if stripped.startswith("### "):
            flow.append(Paragraph(inline(stripped[4:]), st["h3"]))
            i += 1
            continue

        if stripped.startswith("## "):
            if not first_h2:
                flow.append(PageBreak())
            first_h2 = False
            flow.append(Paragraph(inline(stripped[3:]), st["h2"]))
            i += 1
            continue

        if stripped.startswith("# "):
            flow.append(Paragraph(inline(stripped[2:]), st["title"]))
            flow.append(Spacer(1, 10))
            i += 1
            continue

        if stripped.startswith(">"):
            buf = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip()[1:].strip())
                i += 1
            flow.append(Paragraph(inline(" ".join(buf)), st["quote"]))
            continue

        if stripped.startswith("- "):
            flow.append(Paragraph(inline(stripped[2:]), st["bullet"],
                                  bulletText="\u2022"))
            i += 1
            continue

        m = re.match(r"(\d+)\.\s+(.*)", stripped)
        if m:
            flow.append(Paragraph(inline(m.group(2)), st["num"],
                                  bulletText=f"{m.group(1)}."))
            i += 1
            continue

        flow.append(Paragraph(inline(stripped), st["body"]))
        i += 1
    return flow


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(F_CN, 7.6)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(20 * mm, 10 * mm, "SpecDec — 投机解码：原理、代码与实现细节")
    canvas.drawRightString(A4[0] - 20 * mm, 10 * mm, f"第 {doc.page} 页")
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.4)
    canvas.line(20 * mm, 14 * mm, A4[0] - 20 * mm, 14 * mm)
    canvas.restoreState()


def main() -> int:
    register_fonts()
    st = styles()
    margin = 20 * mm
    avail = A4[0] - 2 * margin
    doc = BaseDocTemplate(str(OUT), pagesize=A4, leftMargin=margin, rightMargin=margin,
                          topMargin=16 * mm, bottomMargin=20 * mm,
                          title="投机解码：原理、代码与实现细节",
                          author="SpecDec / DSAI 4207")
    frame = Frame(margin, 20 * mm, avail, A4[1] - 36 * mm, id="n")
    doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=footer)])
    story = parse(SRC.read_text(encoding="utf-8"), st, avail)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.build(story)
    size = OUT.stat().st_size
    print(f"{OUT}  ({size/1024:.1f} KiB, {doc.page} 页)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
