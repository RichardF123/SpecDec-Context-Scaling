"""md_to_pdf.py — 把 Markdown 渲染成 HTML 并打印为 PDF（**不安装任何依赖**）。

为什么这样做
  * 本机已有 Chrome / Edge，`--headless --print-to-pdf` 是**零安装**的可靠路径；
  * 中文排版直接走系统字体（Microsoft YaHei），不需要注册字体文件；
  * 报告/手册用 Markdown 维护（好改、好 diff），PDF 只是**产物** ⇒ 后期跟踪时
    改 `.md` 再跑一次本脚本即可重新生成。

只支持本仓库手册实际用到的 Markdown 子集（够用且可预测）：
  标题 `#`..`####`、表格（`|` + 分隔行）、围栏代码块 ``` ```、有序/无序列表（支持两级缩进）、
  引用 `>`、水平线 `---`、段落，以及行内 `**粗体**` 与 `` `代码` ``。

用法
  python scripts/md_to_pdf.py docs/xxx.md                       # -> docs/xxx.pdf (+ .html)
  python scripts/md_to_pdf.py docs/xxx.md --inventory           # 末尾追加"引擎代码清单"（自动扫描）
  python scripts/md_to_pdf.py docs/xxx.md --no-pdf              # 只出 HTML
"""
from __future__ import annotations

import argparse
import ast
import html
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "specdec" / "engine"
BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]

CSS = """
@page { size: A4; margin: 14mm 12mm 14mm 12mm; }
* { box-sizing: border-box; }
body { font-family: "Microsoft YaHei", "Segoe UI", "Noto Sans CJK SC", sans-serif;
       font-size: 9.5pt; line-height: 1.45; color: #111; margin: 0; }
h1 { font-size: 17pt; border-bottom: 2.2px solid #222; padding-bottom: 3px; margin: 0 0 8px 0; }
h2 { font-size: 13pt; border-bottom: 1.2px solid #999; padding-bottom: 2px;
     margin: 16px 0 6px 0; page-break-after: avoid; }
h3 { font-size: 11pt; margin: 12px 0 4px 0; page-break-after: avoid; }
h4 { font-size: 10pt; margin: 10px 0 3px 0; page-break-after: avoid; color: #333; }
p { margin: 4px 0; }
ul, ol { margin: 3px 0 3px 18px; padding: 0; }
li { margin: 1.5px 0; }
table { border-collapse: collapse; width: 100%; margin: 6px 0; font-size: 8.3pt;
        table-layout: auto; page-break-inside: auto; }
th, td { border: 0.6px solid #999; padding: 2.5px 4px; vertical-align: top;
         word-break: break-word; overflow-wrap: anywhere; }
th { background: #eee; font-weight: 600; }
tr { page-break-inside: avoid; }
code { font-family: Consolas, "Courier New", monospace; font-size: 8.6pt;
       background: #f4f4f4; padding: 0 2px; border-radius: 2px; }
pre { background: #f7f7f7; border: 0.6px solid #ccc; padding: 5px 6px; margin: 5px 0;
      font-family: Consolas, "Courier New", monospace; font-size: 8pt;
      white-space: pre-wrap; word-break: break-word; page-break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 8pt; }
blockquote { border-left: 3px solid #bbb; margin: 5px 0; padding: 2px 8px;
             background: #fafafa; color: #333; }
hr { border: 0; border-top: 0.8px solid #bbb; margin: 10px 0; }
.toc { background: #f8f8f8; border: 0.6px solid #ddd; padding: 6px 10px; margin: 8px 0 14px 0; }
.toc ul { margin: 2px 0 2px 16px; }
.toc a { color: #114; text-decoration: none; }
a { color: #114; }
.small { color: #555; font-size: 8.4pt; }
"""


def _inline(t: str) -> str:
    t = html.escape(t, quote=False)
    t = re.sub(r"`([^`]+)`", lambda m: f"<code>{m.group(1)}</code>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", lambda m: f"<b>{m.group(1)}</b>", t)
    return t


def _slug(t: str, used: dict) -> str:
    s = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "-", t).strip("-").lower() or "sec"
    n = used.get(s, 0)
    used[s] = n + 1
    return s if n == 0 else f"{s}-{n}"


def md_to_html(md: str, title: str, *, toc: bool = True) -> str:
    lines = md.split("\n")
    out: list[str] = []
    used: dict = {}
    heads: list[tuple[int, str, str]] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):                                  # 代码块
            i += 1
            buf = []
            while i < len(lines) and not lines[i].startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            out.append("<pre><code>" + html.escape("\n".join(buf)) + "</code></pre>")
            continue
        if re.match(r"^\|.*\|\s*$", ln) and i + 1 < len(lines) \
                and re.match(r"^\|[\s:|-]+\|\s*$", lines[i + 1]):
            head = [c.strip() for c in ln.strip().strip("|").split("|")]
            i += 2
            rows = []
            while i < len(lines) and re.match(r"^\|.*\|\s*$", lines[i]):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            t = ["<table><thead><tr>"] + [f"<th>{_inline(c)}</th>" for c in head] + ["</tr></thead><tbody>"]
            for r in rows:
                t.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>")
            t.append("</tbody></table>")
            out.append("".join(t))
            continue
        m = re.match(r"^(#{1,4})\s+(.*)$", ln)
        if m:
            lvl, txt = len(m.group(1)), m.group(2).strip()
            sid = _slug(txt, used)
            heads.append((lvl, txt, sid))
            out.append(f'<h{lvl} id="{sid}">{_inline(txt)}</h{lvl}>')
            i += 1
            continue
        if re.match(r"^-{3,}\s*$", ln):
            out.append("<hr/>")
            i += 1
            continue
        if ln.startswith(">"):
            buf = []
            while i < len(lines) and lines[i].startswith(">"):
                buf.append(lines[i].lstrip(">").strip())
                i += 1
            out.append("<blockquote>" + "<br/>".join(_inline(x) for x in buf) + "</blockquote>")
            continue
        if re.match(r"^\s*([-*]|\d+\.)\s+", ln):
            ordered = bool(re.match(r"^\s*\d+\.\s+", ln))
            items: list[tuple[int, str]] = []
            while i < len(lines) and re.match(r"^\s*([-*]|\d+\.)\s+", lines[i]):
                mm = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", lines[i])
                items.append((len(mm.group(1)) // 2, mm.group(3).strip()))
                i += 1
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>")
            for depth, txt in items:
                out.append("<li>" + ("&nbsp;" * 0) + _inline(txt) + "</li>")
            out.append(f"</{tag}>")
            continue
        if not ln.strip():
            i += 1
            continue
        out.append("<p>" + _inline(ln.strip()) + "</p>")
        i += 1
    body = "\n".join(out)
    nav = ""
    if toc:
        items = "".join(f'<li><a href="#{sid}">{_inline(txt)}</a></li>'
                        for lvl, txt, sid in heads if lvl <= 2)
        nav = f'<div class="toc"><b>目录</b><ul>{items}</ul></div>'
    return (f'<!doctype html><html lang="zh"><head><meta charset="utf-8">'
            f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body>"
            f"<h1>{html.escape(title)}</h1>{nav}{body}</body></html>")


def scan_inventory() -> list[dict]:
    """扫描 `specdec/engine/*.py`：文件名/行数/首行作用/入口环境变量/是否写产物。"""
    rows = []
    for p in sorted(ENGINE.glob("*.py")):
        txt = p.read_text(encoding="utf-8", errors="ignore")
        try:
            doc = ast.get_docstring(ast.parse(txt)) or ""
        except SyntaxError:
            doc = ""
        title = (doc.strip().splitlines() or [""])[0]
        title = re.sub(r"^[\w./]+\.py\s*[—-]\s*", "", title)[:150]
        envs = sorted(set(re.findall(r'os\.environ\.get\(\s*"([A-Z0-9_]+)"', txt))
                      | set(re.findall(r'S\.env_flag\(\s*"([A-Z0-9_]+)"', txt))
                      | set(re.findall(r'os\.environ\[\s*"([A-Z0-9_]+)"', txt)))
        rows.append(dict(file=p.name, lines=len(txt.splitlines()), purpose=title,
                         envs=", ".join(envs), has_main="if __name__" in txt,
                         selftest=bool(re.search(r"def _selftest|SELFTEST", txt))))
    return rows


def inventory_md() -> str:
    rows = scan_inventory()
    out = ["## 附录 A：`specdec/engine` 全量代码清单（自动扫描）", "",
           f"共 **{len(rows)}** 个 `.py`；其中带 `__main__` 的 **{sum(1 for r in rows if r['has_main'])}** 个、"
           f"含自测入口的 **{sum(1 for r in rows if r['selftest'])}** 个。"
           "本表由 `scripts/md_to_pdf.py --inventory` 在生成 PDF 时**实时扫描**，"
           "因此新增文件会自动出现在下一次生成的 PDF 里。", "",
           "| 文件 | 行数 | 作用（模块 docstring 首行） | 入口环境变量 | 自测 |",
           "| --- | --- | --- | --- | --- |"]
    for r in rows:
        out.append(f"| `{r['file']}` | {r['lines']} | {r['purpose'] or '—'} | {r['envs'] or '—'} | "
                   f"{'✔' if r['selftest'] else ''} |")
    return "\n".join(out) + "\n"


def find_browser() -> str | None:
    for b in BROWSERS:
        if Path(b).exists():
            return b
    return shutil.which("chrome") or shutil.which("msedge")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("md")
    ap.add_argument("--inventory", action="store_true", help="末尾追加自动扫描的代码清单")
    ap.add_argument("--no-pdf", action="store_true", help="只生成 HTML")
    ap.add_argument("--page-numbers", action="store_true",
                    help="保留浏览器默认页眉/页脚（含页码与来源路径；便于纸质跟踪）")
    ap.add_argument("--title", default=None)
    a = ap.parse_args()

    src = Path(a.md)
    if not src.is_absolute():
        src = (Path.cwd() / src).resolve()
    md = src.read_text(encoding="utf-8")
    if a.inventory:
        md = md.rstrip() + "\n\n---\n\n" + inventory_md()
    title = a.title or src.stem
    out_html = src.with_suffix(".html")
    out_html.write_text(md_to_html(md, title), encoding="utf-8")
    print(f"[md2pdf] HTML -> {out_html}  ({out_html.stat().st_size / 1024:.0f} KB)")
    if a.no_pdf:
        return 0
    br = find_browser()
    if not br:
        print("[md2pdf] !! 未找到 Chrome/Edge，无法生成 PDF（HTML 已生成）")
        return 3
    out_pdf = src.with_suffix(".pdf")
    log = src.with_suffix(".pdf.log")
    with tempfile.TemporaryDirectory() as tmp, open(log, "w", encoding="utf-8") as lf:
        cmd = [br, "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
               f"--user-data-dir={tmp}", "--virtual-time-budget=15000",
               f"--print-to-pdf={out_pdf}", out_html.as_uri()]
        if not a.page_numbers:
            cmd.insert(5, "--no-pdf-header-footer")
        r = subprocess.run(cmd, stdout=lf, stderr=lf, timeout=300)
    if out_pdf.exists() and out_pdf.stat().st_size > 5000:
        print(f"[md2pdf] PDF  -> {out_pdf}  ({out_pdf.stat().st_size / 1024:.0f} KB)  "
              f"浏览器={Path(br).name} rc={r.returncode}")
        return 0
    print(f"[md2pdf] !! PDF 未生成或过小（rc={r.returncode}）；见日志 {log}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
