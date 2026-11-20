"""보고서 모델과 HTML·PDF·DOCX 출력.

반출 기준상 보고서는 PDF(또는 한글) 형태, 그림은 PNG가 권장되고 텍스트(csv)·스프레드시트는 원칙적으로 반출 불가.
그래서 결과물은 PDF·DOCX·PNG로만 만든다. DOCX는 사업계획서에 옮겨 쓰기 위한 편집용(센터 안에서만 사용).
"""
from __future__ import annotations

import base64
import html
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from dataolle.analysis.engine import AnalysisResult
from dataolle.sdc.engine import MASK_FOOTNOTE, MASK_TOKEN, CheckResult


@dataclass
class Paragraph:
    text: str


@dataclass
class Bullets:
    items: list[str]


@dataclass
class Subhead:
    """절 안의 소제목(개선안의 Ⅰ·Ⅱ… 등)."""
    text: str


@dataclass
class Figure:
    path: Path
    caption: str = ""
    result: AnalysisResult | None = None


@dataclass
class Table:
    df: pd.DataFrame
    caption: str = ""
    result: AnalysisResult | None = None
    style: str = "num"            # num: 숫자표(오른쪽 정렬) | text: 글이 긴 표(왼쪽 정렬, 줄바꿈)
    show_index: bool = True
    widths: list[int] | None = None   # 열 너비(%)


@dataclass
class Callout:
    title: str
    items: list[str]
    tone: str = "insight"         # insight | caution | note


@dataclass
class Sheet:
    """그대로 인쇄해 쓰는 한 장짜리 글(안내문·현수막·문자 등). lines의 '■ '는 소제목, '· '는 항목."""
    title: str
    lines: list[str] = field(default_factory=list)
    kind: str = "notice"          # notice | big(어르신용 큰 글씨) | banner(현수막) | sms(문자) | sns
    sign: str = ""
    note: str = ""


@dataclass
class Section:
    heading: str
    blocks: list = field(default_factory=list)
    new_page: bool = False          # 실행물 묶음: 문서마다 새 쪽에서 시작


@dataclass
class Report:
    title: str
    subtitle: str = ""
    sections: list[Section] = field(default_factory=list)
    results: list[AnalysisResult] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)
    synthetic: bool = False
    created: datetime = field(default_factory=datetime.now)
    ai_note: str = ""
    extra: dict = field(default_factory=dict)      # 코스별 부가 정보(예: 사업계획서 붙여넣기용 문단)

    def sources(self) -> list[tuple[str, str | None, str | None, list[str]]]:
        """데이터셋별로 합친 (이름, 가장 이른 월, 가장 늦은 월, 사용 파일)."""
        merged: dict[str, list] = {}
        for r in self.results:
            for n, a, b, files in r.source_parts():
                m = merged.setdefault(n, [a, b, []])
                if a and (m[0] is None or a < m[0]):
                    m[0] = a
                if b and (m[1] is None or b > m[1]):
                    m[1] = b
                for f in files:
                    if f not in m[2]:
                        m[2].append(f)
        return [(n, a, b, sorted(files)) for n, (a, b, files) in merged.items()]

    def citations(self) -> list[str]:
        from dataolle.analysis.engine import cite_label
        return [cite_label(n, a, b) for n, a, b, _ in self.sources()]

    def checks(self) -> list[CheckResult]:
        return [r.check for r in self.results]

    def figures(self) -> list[Figure]:
        return [b for s in self.sections for b in s.blocks if isinstance(b, Figure)]

    def all_notes(self) -> list[str]:
        seen, out = set(), []
        for r in self.results:
            for n in r.notes:
                if n not in seen:
                    seen.add(n)
                    out.append(n)
        for n in self.limitations:
            if n not in seen:
                seen.add(n)
                out.append(n)
        return out


# ---------------------------------------------------------------------------
# HTML
# 가상데이터로 만든 문서는 PDF 쪽마다 오른쪽 위에 표시한다(낱장으로 돌아다녀도 실제 자료로 오해하지 않게).
SYNTH_PAGE_CSS = """
@page { @top-right { content: "가상데이터(시연용) — 실제 자료 아님"; font-size: 8pt; color: #8A4B00; } }
"""

# ---------------------------------------------------------------------------
CSS = """
@page { size: A4; margin: 15mm 14mm 16mm 14mm;
  @bottom-center { content: counter(page) " / " counter(pages); font-size: 8pt; color: #898781; } }
* { box-sizing: border-box; }
body { font-family: "Malgun Gothic", "맑은 고딕", "Noto Sans CJK KR", "Apple SD Gothic Neo", sans-serif;
  color: #1b1b1a; font-size: 10pt; line-height: 1.62; margin: 0; background: #ffffff;
  word-break: keep-all; overflow-wrap: anywhere; }
.wrap { max-width: 182mm; margin: 0 auto; padding: 0 2mm; }
.cover { border-bottom: 2px solid #1c5cab; padding: 4mm 0 5mm; margin-bottom: 5mm; }
.kicker { color: #1c5cab; font-weight: 700; font-size: 9pt; letter-spacing: .02em; }
h1 { font-size: 19pt; line-height: 1.3; margin: 1.5mm 0 1mm; color: #0b0b0b; }
.subtitle { color: #52514e; font-size: 10.5pt; }
.synth { margin: 3mm 0 0; padding: 2.2mm 3mm; border-radius: 2mm; background: #fff4e5; color: #8a4b00;
  font-size: 8.8pt; border: 1px solid #f3c58a; }
table.meta { border-collapse: collapse; width: 100%; margin-top: 3.5mm; font-size: 8.8pt; }
table.meta th { text-align: left; color: #52514e; font-weight: 600; width: 24mm; padding: 1.1mm 2mm 1.1mm 0; vertical-align: top; }
table.meta td { padding: 1.1mm 0; color: #1b1b1a; }
h2 { font-size: 13pt; color: #0d366b; margin: 7mm 0 2mm; padding-bottom: 1mm; border-bottom: 1px solid #e1e0d9;
  break-after: avoid; }
p { margin: 1.6mm 0; }
ul { margin: 1.2mm 0 1.2mm 5mm; padding: 0; }
li { margin: .6mm 0; }
figure { margin: 3mm 0 2mm; break-inside: avoid; }
figure img { width: 100%; border: 1px solid #ecebe6; border-radius: 1.5mm; }
figcaption, .tcap { font-size: 8.5pt; color: #52514e; margin-top: 1mm; }
table.data { border-collapse: collapse; width: 100%; font-size: 8.4pt; margin: 2mm 0 1mm; break-inside: avoid;
  font-variant-numeric: tabular-nums; }
table.data th, table.data td { border-bottom: 1px solid #e1e0d9; padding: 1.1mm 1.6mm; text-align: right; white-space: nowrap; }
table.data th { background: #f4f6fa; color: #33322f; font-weight: 600; }
table.data td:first-child, table.data th:first-child { text-align: left; }
table.data tr.total td { font-weight: 700; border-top: 1px solid #c3c2b7; }
table.data.text td, table.data.text th { white-space: normal; text-align: left; vertical-align: top; font-size: 7.9pt; }
table.data.text td.c, table.data.text th.c { text-align: center; }
.callout { border-left: 3px solid #2a78d6; background: #f3f7fd; padding: 2.4mm 3.5mm; margin: 3mm 0; border-radius: 0 1.5mm 1.5mm 0;
  break-inside: avoid; }
.callout.caution { border-left-color: #eda100; background: #fff8e8; }
.callout.note { border-left-color: #898781; background: #f6f6f4; }
.callout .ct { font-weight: 700; margin-bottom: 1mm; }
.small { font-size: 8.3pt; color: #52514e; }
.appendix h2 { font-size: 11.5pt; }
.newpage { break-before: page; }
h3 { font-size: 10.8pt; color: #0b0b0b; margin: 4mm 0 1.2mm; break-after: avoid; }
h3.doc-title { font-size: 13.5pt; text-align: center; margin: 2mm 0 4mm; color: #0d366b; }
.sheet { border: 1.6px solid #1c5cab; border-radius: 3mm; padding: 6mm 7mm 5mm; margin: 4mm 0; break-inside: avoid; background: #fff; }
.sheet .sh-title { font-size: 15pt; font-weight: 800; color: #0d366b; line-height: 1.35; margin-bottom: 3mm; }
.sheet p { margin: 1.4mm 0; }
.sheet .sh-h { font-weight: 700; color: #0d366b; margin: 3.2mm 0 1mm; }
.sheet ul { margin: .6mm 0 .6mm 5mm; }
.sheet .sh-sign { margin-top: 6mm; text-align: right; font-weight: 700; }
.sheet .sh-note { margin-top: 2.5mm; font-size: 8pt; color: #898781; }
.sheet.big { font-size: 15.5pt; line-height: 1.75; border-width: 2.2px; }
.sheet.big .sh-title { font-size: 22pt; color: #0b0b0b; }
.sheet.big .sh-h { font-size: 16.5pt; }
.sheet.banner { border: 3px solid #2a78d6; background: #f3f7fd; text-align: center; padding: 6mm 6mm 5mm; }
.sheet.banner .sh-title { font-size: 8.8pt; color: #52514e; font-weight: 600; margin-bottom: 1.5mm; }
.sheet.banner .sh-main { font-size: 21pt; font-weight: 800; color: #0b0b0b; line-height: 1.3; }
.sheet.sms { max-width: 120mm; border-radius: 5mm; background: #f4f6fa; border: 1px solid #c9d3e2; }
.sheet.sms .sh-title, .sheet.sns .sh-title { font-size: 9pt; color: #52514e; font-weight: 600; }
.sheet.sns { border: 1px solid #c9d3e2; border-radius: 3mm; }
.chip { display: inline-block; font-size: 7.6pt; padding: .2mm 1.6mm; border-radius: 3mm; background: #e9f1fc; color: #1c5cab;
  margin-right: 1mm; }
"""


def _img_src(path: Path) -> str:
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def _table_html(df: pd.DataFrame, style: str = "num", show_index: bool = True, widths: list[int] | None = None) -> str:
    esc = html.escape
    cols = ([df.index.name or ""] if show_index else []) + list(df.columns)
    colgroup = ("<colgroup>" + "".join(f"<col style='width:{w}%'>" for w in widths) + "</colgroup>") if widths else ""
    head = "".join(f"<th>{esc(str(c))}</th>" for c in cols)
    rows = []
    for idx, row in df.iterrows():
        label = str(idx)
        if len(label) == 6 and label.isdigit():
            label = f"{label[:4]}.{label[4:]}"
        cls = ' class="total"' if str(idx) == "합계" else ""
        cells = "".join(f"<td>{esc(str(v))}</td>" for v in row.tolist())
        first = f"<td>{esc(label)}</td>" if show_index else ""
        rows.append(f"<tr{cls}>{first}{cells}</tr>")
    klass = "data text" if style == "text" else "data"
    fixed = " style='table-layout:fixed'" if widths else ""
    return f'<table class="{klass}"{fixed}>{colgroup}<thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def _sheet_html(sh: Sheet) -> str:
    esc = html.escape
    out = [f"<div class='sheet {esc(sh.kind)}'><div class='sh-title'>{esc(sh.title)}</div>"]
    items: list[str] = []

    def flush():
        if items:
            out.append("<ul>" + "".join(f"<li>{esc(i)}</li>" for i in items) + "</ul>")
            items.clear()
    for i, line in enumerate(sh.lines):
        if line.startswith("· "):
            items.append(line[2:])
            continue
        flush()
        if line.startswith("■ "):
            out.append(f"<div class='sh-h'>{esc(line[2:])}</div>")
        elif sh.kind == "banner" and i == 0:
            out.append(f"<div class='sh-main'>{esc(line)}</div>")
        else:
            out.append(f"<p>{esc(line)}</p>")
    flush()
    if sh.sign:
        out.append(f"<div class='sh-sign'>{esc(sh.sign)}</div>")
    if sh.note:
        out.append(f"<div class='sh-note'>{esc(sh.note)}</div>")
    out.append("</div>")
    return "".join(out)


def to_html(report: Report, path: str | Path, appendix: bool = True) -> Path:
    esc = html.escape
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_rows = "".join(f"<tr><th>{esc(k)}</th><td>{esc(str(v))}</td></tr>" for k, v in report.meta.items())
    parts = [f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><title>{esc(report.title)}</title>"
             f"<style>{CSS}{SYNTH_PAGE_CSS if report.synthetic else ''}</style></head><body><div class='wrap'>"]
    kicker = report.extra.get("kicker", "데이터올레 · 제주빅데이터센터 데이터 분석 보고서")
    parts.append(f"<div class='cover'><div class='kicker'>{esc(kicker)}</div>"
                 f"<h1>{esc(report.title)}</h1><div class='subtitle'>{esc(report.subtitle)}</div>")
    if report.synthetic:
        parts.append("<div class='synth'>이 보고서의 센터 데이터 수치는 <b>가상데이터(시연용)</b>로 만든 것입니다. "
                     "실제 제주빅데이터센터 데이터가 아니므로 의사결정에 사용하지 마세요.</div>")
    parts.append(f"<table class='meta'>{meta_rows}</table></div>")
    for sec in report.sections:
        parts.append(f"<h2{' class=\"newpage\"' if sec.new_page else ''}>{esc(sec.heading)}</h2>")
        for b in sec.blocks:
            if isinstance(b, Paragraph):
                parts.append(f"<p>{esc(b.text)}</p>")
            elif isinstance(b, Subhead):
                cls = " class='doc-title'" if b.text.startswith("■") else ""
                parts.append(f"<h3{cls}>{esc(b.text.lstrip('■ '))}</h3>")
            elif isinstance(b, Bullets):
                parts.append("<ul>" + "".join(f"<li>{esc(i)}</li>" for i in b.items) + "</ul>")
            elif isinstance(b, Figure):
                parts.append(f"<figure><img src='{_img_src(b.path)}' alt='{esc(b.caption)}'>"
                             f"<figcaption>{esc(b.caption)}</figcaption></figure>")
            elif isinstance(b, Table):
                parts.append(_table_html(b.df, b.style, b.show_index, b.widths))
                cap = b.caption
                if b.style == "num" and (b.df.to_numpy() == MASK_TOKEN).any():
                    cap = (cap + " · " if cap else "") + MASK_FOOTNOTE
                if cap:
                    parts.append(f"<div class='tcap'>{esc(cap)}</div>")
            elif isinstance(b, Callout):
                parts.append(f"<div class='callout {esc(b.tone)}'><div class='ct'>{esc(b.title)}</div><ul>"
                             + "".join(f"<li>{esc(i)}</li>" for i in b.items) + "</ul></div>")
            elif isinstance(b, Sheet):
                parts.append(_sheet_html(b))
    if appendix:
        parts.append("<div class='appendix'>")
        parts.append("<h2>부록 1. 데이터 출처</h2><ul>" + "".join(f"<li>{esc(c)}</li>" for c in report.citations()) + "</ul>")
        parts.append("<h2>부록 2. 반출 기준 검수 내역</h2><table class='data' style='table-layout:fixed'>"
                     "<colgroup><col style='width:38%'><col style='width:12%'><col style='width:8%'><col style='width:34%'>"
                     "<col style='width:8%'></colgroup><thead><tr><th>표·그림</th><th>집계 유형</th>"
                     "<th>비공개 칸</th><th>처리 내용</th><th>결과</th></tr></thead><tbody>")
        wrap = "style='white-space:normal;text-align:left;font-size:7.8pt'"
        for r in report.results:
            acts = "; ".join(dict.fromkeys(a.detail for a in r.check.actions)) or "해당 없음(10 미만 수치 없음)"
            parts.append(f"<tr><td {wrap}>{esc(r.title)}</td><td>{esc(r.kind.value)}</td><td>{r.check.masked_cells}</td>"
                         f"<td {wrap}>{esc(acts)}</td>"
                         f"<td>{'통과' if r.check.passed else '보완 필요'}</td></tr>")
        parts.append("</tbody></table>")
        parts.append(f"<p class='small'>{esc(MASK_FOOTNOTE)}. 여러 달을 합친 칸의 바탕 인원은 '월별 최댓값'(실제 인원의 하한)으로 판정했고, "
                     "가린 칸이 합계·다른 칸으로 다시 계산되지 않는지 선형대수(행렬 계수) 방식으로 확인했습니다. "
                     "최종 반출 여부는 제주빅데이터센터의 반출 심사 결과를 따릅니다.</p>")
        notes = report.all_notes()
        if notes:
            parts.append("<h2>부록 3. 해석 시 유의사항(한계)</h2><ul>" + "".join(f"<li>{esc(n)}</li>" for n in notes) + "</ul>")
        if report.ai_note:
            parts.append(f"<h2>부록 4. AI 활용 범위</h2><p class='small'>{esc(report.ai_note)}</p>")
        parts.append("</div>")
    parts.append("</div></body></html>")
    path.write_text("".join(parts), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# PDF: Windows에 기본 설치된 Edge(또는 Chrome)의 headless 인쇄 기능 사용
# ---------------------------------------------------------------------------
def find_browser() -> str | None:
    """PDF 인쇄에 쓸 브라우저. Windows는 기본 설치된 Edge → Chrome 순서."""
    cands = [
        os.environ.get("DATAOLLE_BROWSER", ""),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
    ]
    for c in cands:
        if c and os.path.exists(c):
            return c
    if os.name != "nt":   # 개발 환경(Linux): 인쇄 전용 headless shell이 있으면 우선
        roots = [os.environ.get("PLAYWRIGHT_BROWSERS_PATH", ""), "/opt/pw-browsers", os.path.expanduser("~/.cache/ms-playwright")]
        for root in filter(None, roots):
            hits = sorted(Path(root).glob("chromium_headless_shell-*/chrome-headless-shell-*/chrome-headless-shell"))
            if hits:
                return str(hits[-1])
    for name in ("msedge", "chrome", "google-chrome", "chromium", "chromium-browser"):
        p = shutil.which(name)
        if p:
            return p
    return None


_QUIET_FLAGS = ["--disable-gpu", "--no-pdf-header-footer", "--no-first-run", "--no-default-browser-check",
                "--disable-background-networking", "--disable-component-update", "--disable-sync",
                "--disable-extensions", "--disable-default-apps", "--mute-audio"]


def to_pdf(html_path: str | Path, pdf_path: str | Path, timeout: int = 60) -> Path | None:
    """HTML → PDF. 브라우저가 PDF를 다 쓰면(크기가 더 이상 변하지 않으면) 바로 끝낸다. 실패하면 None."""
    import time
    html_path, pdf_path = Path(html_path).resolve(), Path(pdf_path).resolve()
    exe = find_browser()
    if not exe:
        return None
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    if pdf_path.exists():
        pdf_path.unlink()
    shell = "headless-shell" in Path(exe).name
    with tempfile.TemporaryDirectory(prefix="dataolle_pdf_", ignore_cleanup_errors=True) as profile:   # Windows: Edge 종료 직후 잠긴 파일
        cmd = [exe] + ([] if shell else ["--headless=new"]) + _QUIET_FLAGS + [
            f"--user-data-dir={profile}", f"--print-to-pdf={pdf_path}", html_path.as_uri()]
        if os.name != "nt":
            cmd.insert(1, "--no-sandbox")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
        except Exception:
            return None
        t0, last, stable = time.time(), -1, 0
        while time.time() - t0 < timeout:
            if proc.poll() is not None and not pdf_path.exists():
                break
            if pdf_path.exists():
                size = pdf_path.stat().st_size
                stable = stable + 1 if (size == last and size > 1000) else 0
                last = size
                if stable >= 3 or (proc.poll() is not None and size > 1000):
                    break
            time.sleep(0.3)
        if proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
    return pdf_path if pdf_path.exists() and pdf_path.stat().st_size > 1000 else None


# ---------------------------------------------------------------------------
# DOCX(편집용)
# ---------------------------------------------------------------------------
def to_docx(report: Report, path: str | Path) -> Path | None:
    try:
        from docx import Document  # noqa: WPS433
        from docx.enum.text import WD_ALIGN_PARAGRAPH  # noqa: WPS433
        from docx.oxml.ns import qn  # noqa: WPS433
        from docx.shared import Cm, Pt, RGBColor  # noqa: WPS433
    except ImportError:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Malgun Gothic"
    st.font.size = Pt(10)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "맑은 고딕")
    for sec in doc.sections:
        sec.left_margin = sec.right_margin = Cm(2.0)
        sec.top_margin = sec.bottom_margin = Cm(1.8)

    def para(text, size=10, bold=False, color=None, align=None):
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.font.size = Pt(size)
        run.bold = bold
        run.font.name = "Malgun Gothic"
        run._element.rPr.rFonts.set(qn("w:eastAsia"), "맑은 고딕")
        if color:
            run.font.color.rgb = RGBColor.from_string(color)
        if align:
            p.alignment = align
        return p

    para(report.extra.get("kicker", "데이터올레 · 제주빅데이터센터 데이터 분석 보고서"), 9, True, "1C5CAB")
    para(report.title, 18, True)
    if report.subtitle:
        para(report.subtitle, 10.5, False, "52514E")
    if report.synthetic:
        para("※ 센터 데이터 수치는 가상데이터(시연용)입니다. 실제 데이터가 아니므로 의사결정에 사용하지 마세요.", 9, True, "8A4B00")
    if report.meta:
        t = doc.add_table(rows=0, cols=2)
        t.style = "Table Grid"
        for k, v in report.meta.items():
            row = t.add_row().cells
            row[0].text, row[1].text = str(k), str(v)

    def add_df(df: pd.DataFrame):
        cols = [df.index.name or ""] + [str(c) for c in df.columns]
        t = doc.add_table(rows=1, cols=len(cols))
        t.style = "Table Grid"
        for i, c in enumerate(cols):
            t.rows[0].cells[i].text = c
        for idx, row in df.iterrows():
            cells = t.add_row().cells
            label = str(idx)
            cells[0].text = f"{label[:4]}.{label[4:]}" if len(label) == 6 and label.isdigit() else label
            for i, v in enumerate(row.tolist(), start=1):
                cells[i].text = str(v)
                cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT

    from docx.enum.text import WD_BREAK  # noqa: WPS433

    def add_sheet(sh: Sheet):
        big = sh.kind == "big"
        t = doc.add_table(rows=1, cols=1)
        t.style = "Table Grid"
        cell = t.rows[0].cells[0]
        cell.paragraphs[0].text = ""

        def cpara(text, size, bold=False, color=None, align=None):
            p = cell.add_paragraph()
            run = p.add_run(text)
            run.font.size = Pt(size)
            run.bold = bold
            run.font.name = "Malgun Gothic"
            run._element.rPr.rFonts.set(qn("w:eastAsia"), "맑은 고딕")
            if color:
                run.font.color.rgb = RGBColor.from_string(color)
            if align:
                p.alignment = align
            return p
        center = WD_ALIGN_PARAGRAPH.CENTER if sh.kind == "banner" else None
        cpara(sh.title, 20 if big else (9 if sh.kind in ("banner", "sms", "sns") else 14), True, "0D366B", center)
        body = 15 if big else 10
        for i, line in enumerate(sh.lines):
            if line.startswith("■ "):
                cpara(line[2:], body + 1, True, "0D366B")
            elif line.startswith("· "):
                cpara("• " + line[2:], body)
            elif sh.kind == "banner" and i == 0:
                cpara(line, 22, True, None, center)
            else:
                cpara(line, body, False, None, center)
        if sh.sign:
            cpara(sh.sign, body, True, None, WD_ALIGN_PARAGRAPH.RIGHT)
        if sh.note:
            cpara(sh.note, 8, False, "898781")
        doc.add_paragraph()

    for sec in report.sections:
        if sec.new_page:
            doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
        doc.add_heading(sec.heading, level=2)
        for b in sec.blocks:
            if isinstance(b, Paragraph):
                para(b.text)
            elif isinstance(b, Subhead):
                if b.text.startswith("■"):
                    para(b.text.lstrip("■ "), 13, True, "0D366B", WD_ALIGN_PARAGRAPH.CENTER)
                else:
                    doc.add_heading(b.text, level=3)
            elif isinstance(b, Bullets):
                for i in b.items:
                    doc.add_paragraph(i, style="List Bullet")
            elif isinstance(b, Figure):
                doc.add_picture(str(b.path), width=Cm(16.5))
                if b.caption:
                    para(b.caption, 8.5, False, "52514E")
            elif isinstance(b, Table):
                add_df(b.df)
                if b.caption:
                    para(b.caption, 8.5, False, "52514E")
            elif isinstance(b, Callout):
                para(b.title, 10, True, "0D366B")
                for i in b.items:
                    doc.add_paragraph(i, style="List Bullet")
            elif isinstance(b, Sheet):
                add_sheet(b)
    if not report.extra.get("no_docx_appendix"):
        _docx_appendix(doc, report)
    doc.save(str(path))
    return path


def _docx_appendix(doc, report: Report):
    doc.add_heading("데이터 출처", level=2)
    for c in report.citations():
        doc.add_paragraph(c, style="List Bullet")
    notes = report.all_notes()
    if notes:
        doc.add_heading("해석 시 유의사항(한계)", level=2)
        for n in notes:
            doc.add_paragraph(n, style="List Bullet")
