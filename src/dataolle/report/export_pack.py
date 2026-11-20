"""반출 신청 도우미.

센터 반출 신청이 반려되는 주된 이유(게시판 답변): ① 10 미만 수치 등 반출 기준 미충족 ② 활용 사유가 부족함
③ 파일이 깨짐. 그래서 이 모듈은
  - 반출할 파일만(PDF·PNG) 새 폴더에 모으고, 허용되지 않는 형식(csv·엑셀·txt·코드 등)은 막고,
  - 신청서에 붙여 넣을 '활용 사유'를 센터가 안내한 틀로 만들고
    ("(툴 혹은 분석 방법)으로 분석 후 (어떠한 가공처리)로 가공하여 (내용)의 보고서 작성 및 (어디)에 활용"),
  - 표·그림별 출처·집계 유형·비공개 처리·자동 점검 결과를 담은 '반출 신청 설명자료' PDF를 만든다
    (설명란 글자 수 제한 없음, 설명 PDF 첨부 가능 — 설명란에 파일 이름을 적어 달라는 답변 반영).
최종 승인은 센터 심사로 결정된다. 이 도구는 신청 준비를 돕는 것이지 승인을 보장하지 않는다.
"""
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from dataolle.analysis.engine import AnalysisResult
from dataolle.report.document import Bullets, Callout, Paragraph, Report, Section, Table, to_html, to_pdf
from dataolle.analysis.fmt import fmt_period
from dataolle.sdc.engine import AggKind, audit_table

TOOL = "Python·DuckDB 기반 분석 도구 '데이터올레'"
ALLOWED = {".pdf", ".png", ".jpg", ".jpeg", ".hwp", ".hwpx"}
BLOCKED = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".xlsm", ".json", ".parquet", ".py", ".ipynb", ".r",
           ".duckdb", ".db", ".sqlite", ".html", ".htm", ".zip"}
MAX_BYTES = 3 * 1024 ** 3        # 첨부 최대 3GB

USE_PRESETS = {
    "business": "창업 사업계획서의 시장분석(상권 규모·고객 특성·영업시간 판단) 근거 자료",
    "status": "기관 내부 현황자료(읍면동 소비·유동인구 현황 보고)",
    "paper": "학술 논문의 실증분석 표·그림 자료",
    "policy": "정책 협의 자료(제주도·관계기관 개선안 검토, 예: 탐나는전 적립 방식 변경)",
}

TRANSFORM_PHRASE = {"share": "구성비", "yoy": "전년 동기 대비 증감률", "index": "지수화", "rank": "순위"}


@dataclass
class PackFile:
    path: Path
    role: str          # 보고서 | 그림 | 설명자료
    desc: str
    pages: int | None = None


@dataclass
class ExportPack:
    folder: Path
    files: list[PackFile]
    purpose: str
    sources_text: str
    description: str
    checklist: pd.DataFrame
    problems: list[str] = field(default_factory=list)      # 반출 신청 전에 반드시 고칠 것
    warnings: list[str] = field(default_factory=list)      # 확인 권장

    @property
    def ok(self) -> bool:
        return not self.problems


# ---------------------------------------------------------------------------
# 문구 만들기
# ---------------------------------------------------------------------------
def safe_name(text: str, limit: int = 40) -> str:
    t = re.sub(r'[\\/:*?"<>|\r\n\t]+', " ", str(text)).strip()
    t = re.sub(r"\s+", "_", t)
    return t[:limit].rstrip("_.") or "결과"


def _uniq(items) -> list:
    seen, out = set(), []
    for i in items:
        if i and i not in seen:
            seen.add(i)
            out.append(i)
    return out


def method_phrases(results: list[AnalysisResult]) -> list[str]:
    out = []
    for r in results:
        rd, cd = r.rows_dim, r.cols_dim
        if rd is None:
            out.append("기간 합계")
        elif rd.key == "area":
            out.append("후보지·권역·제주 전체 비교")
        elif rd.kind == "time":
            out.append(f"{'월' if r.spec.rows == 'month' else rd.label}별 추이")
        elif rd.kind == "geo":
            out.append(f"{rd.label} 비교")
        else:
            out.append(f"{rd.label}별 분포")
        if cd is not None and rd is not None and cd.key not in ("series",) and rd.key != "area":
            out.append(f"{rd.label}×{cd.label} 교차집계")
        if "특화지수" in r.title:
            out.append("업종 특화지수(입지계수) 산출")
        if r.spec.transform != "none":
            out.append(TRANSFORM_PHRASE[r.spec.transform])
        if r.kind == AggKind.AVERAGE:
            out.append("평균값(" + r.value_label.split("(")[0] + ") 산출")
    return _uniq(out)


def processing_phrases(results: list[AnalysisResult]) -> list[str]:
    out = ["10 미만 수치 비공개(x) 처리"]
    if any(a.kind == "secondary" for r in results for a in r.check.actions):
        out.append("합계로 역산되지 않도록 추가 비공개")
    if any(a.kind == "rollup" for r in results for a in r.check.actions):
        out.append("상위 단위로 묶는 범주처리")
    if any(r.kind == AggKind.APPLIED for r in results):
        out.append("비율·증감률 등 응용집계 변환")
    if any(r.kind == AggKind.AVERAGE for r in results):
        out.append("바탕 인원 10 이상 평균값만 표시")
    return out


def purpose_text(report: Report, use: str, n_pdf: int, n_png: int, tool: str = TOOL, n_extra: int = 0) -> str:
    """센터 안내 틀: (툴/분석 방법)으로 분석 후 (가공처리)로 가공하여 (내용)의 보고서 작성 및 (어디)에 활용."""
    from dataolle.analysis.engine import is_public
    allnames = _uniq(n for r in report.results for n in r.dataset_names)
    names = [n for n in allnames if not is_public(n)] or allnames          # 반출 심사는 센터 자료 기준
    pub = [n for n in allnames if is_public(n) and n not in names]
    methods = method_phrases(report.results)
    proc = processing_phrases(report.results)
    files = f"PDF {max(n_pdf, 1)}개" + (f"와 그림 PNG {n_png}개" if n_png else "")
    data = "·".join(names[:4]) + (f" 외 {len(names) - 4}종" if len(names) > 4 else "") \
        + (f"(센터 밖 공공데이터 {'·'.join(pub)} 함께 사용)" if pub else "")
    use = USE_PRESETS.get(use, use)
    from dataolle.analysis.josa import josa
    proc_txt = ", ".join(proc[:-1] + [josa(proc[-1], "으로")]) if proc else "가공 없이"
    return (f"{tool}로 {data} 데이터를 {', '.join(methods[:8])} 방식으로 분석 후, "
            f"{proc_txt} 가공하여 '{report.title}'의 보고서({files})"
            + (f"와 실행물 묶음(개선안 초안·안내문·홍보 문구, PDF {n_extra}개)" if n_extra else "") + f" 작성 및 {use}에 활용")


def sources_block(report: Report) -> str:
    """출처별 사용 파일(센터에서 내려받은 Results_*.csv) — 심사자가 다운로드 기록과 맞춰 볼 수 있게."""
    from dataolle.analysis.fmt import fmt_period
    lines = []
    from dataolle.analysis.engine import cite_label
    for n, a, b, files in report.sources():
        c = cite_label(n, a, b)
        if files:
            lines.append(f"{c} — 사용 파일 {len(files)}개: {files[0]}" + (f" 외 {len(files) - 1}개" if len(files) > 1 else ""))
        else:
            lines.append(c)
    return "\n".join(lines)


def _labels_by_result(report: Report) -> dict[int, list[str]]:
    from dataolle.report.document import Figure, Table as TableBlock
    out: dict[int, list[str]] = {}
    nf = nt = 0
    for sec in report.sections:
        for b in sec.blocks:
            if isinstance(b, Figure):
                nf += 1
                if b.result is not None:
                    out.setdefault(id(b.result), []).append(f"그림 {nf}")
            elif isinstance(b, TableBlock) and b.style == "num":
                nt += 1
                if b.result is not None:
                    out.setdefault(id(b.result), []).append(f"표 {nt}")
    return out


def _auto_check(r: AnalysisResult) -> str:
    if r.kind == AggKind.SIMPLE:
        left = audit_table(r.table, AggKind.SIMPLE)
        return "10 미만 수치 0건 · 역산 불가 확인" if (not left and r.check.passed) else "보완 필요: " + "; ".join(left + r.check.violations)
    if r.kind == AggKind.AVERAGE:
        return "바탕 인원·건수 10 미만 평균값 비공개 · 확인 완료" if r.check.passed else "보완 필요"
    return "응용집계(수치 포함 가능) · 바탕 집계 검수 후 변환"


def checklist_frame(report: Report) -> pd.DataFrame:
    labels = _labels_by_result(report)
    rows = []
    for i, r in enumerate(report.results, start=1):
        acts = [a.detail for a in r.check.actions if a.kind != "info"] or ["10 미만 수치 없음"]
        info = [a.detail for a in r.check.actions if a.kind == "info"]
        kind = r.kind.value + (f"·{TRANSFORM_PHRASE[r.spec.transform]}" if r.spec.transform != "none" else "")
        rows.append({
            "위치": "·".join(labels.get(id(r), [])) or "본문 수치",
            "표·그림 제목": r.title,
            "원본 데이터(기간)": "; ".join(f"{n}({fmt_period(a, b)})" for n, a, b, _ in r.source_parts()).replace("~", "~\u200b"),
            "집계 유형": kind,
            "x칸": r.check.masked_cells,
            "처리 내용 → 자동 점검": "; ".join(acts + info) + " → " + _auto_check(r),
        })
    return pd.DataFrame(rows)


CHECK_WIDTHS = [9, 21, 23, 11, 5, 31]


# ---------------------------------------------------------------------------
# 점검
# ---------------------------------------------------------------------------
def pdf_pages(path: Path) -> int | None:
    try:
        from pypdf import PdfReader  # noqa: WPS433
        return len(PdfReader(str(path)).pages)
    except Exception:
        return None


def preflight(files: list[PackFile], report: Report) -> tuple[list[str], list[str]]:
    problems, warnings = [], []
    total = 0
    for f in files:
        ext = f.path.suffix.lower()
        if ext in BLOCKED:
            problems.append(f"{f.path.name}: 텍스트·스프레드시트·코드 파일은 원칙적으로 반출할 수 없습니다(PDF·그림으로 바꾸세요).")
        elif ext not in ALLOWED:
            warnings.append(f"{f.path.name}: 반출 권장 형식(PDF·HWP·PNG·JPG)이 아닙니다.")
        if not f.path.exists() or f.path.stat().st_size == 0:
            problems.append(f"{f.path.name}: 파일이 없거나 비어 있습니다.")
            continue
        total += f.path.stat().st_size
        if ext == ".pdf":
            head = f.path.read_bytes()[:5]
            if head != b"%PDF-":
                problems.append(f"{f.path.name}: PDF 파일이 손상된 것 같습니다.")
        if ext == ".png" and f.path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
            problems.append(f"{f.path.name}: PNG 파일이 손상된 것 같습니다.")
    if total > MAX_BYTES:
        problems.append("첨부 용량이 3GB를 넘습니다. 파일을 나눠 신청하세요.")
    for r in report.results:
        if not r.check.passed:
            problems.append(f"'{r.title}': 반출 기준 검수 미통과 — " + "; ".join(r.check.violations))
        left = audit_table(r.table, r.kind)
        if left:
            problems.append(f"'{r.title}': " + "; ".join(left))
        if r.kind == AggKind.AVERAGE and any(a.kind == "info" for a in r.check.actions):
            warnings.append(f"'{r.title}': 바탕 표본 수를 알 수 없는 평균값입니다(예: 점포당 메뉴 매출). 심사에서 추가 설명을 요구할 수 있습니다.")
        if r.synthetic:
            warnings.append("가상데이터로 만든 결과입니다. 실제 반출 신청에는 센터 실데이터로 다시 만든 파일을 쓰세요.")
            break
    if not any(f.role == "보고서" for f in files):
        warnings.append("보고서 PDF가 없습니다. 브라우저에서 보고서를 열어 'PDF로 저장'한 뒤 함께 신청하세요.")
    return _uniq(problems), _uniq(warnings)


# ---------------------------------------------------------------------------
# 묶음 만들기
# ---------------------------------------------------------------------------
def build_pack(report: Report, out_dir: str | Path, use: str = "business", report_pdf: Path | None = None,
               figures: list[Path] | None = None, include_figures: bool = True,
               extra_pdfs: list[tuple[Path, str]] | None = None) -> ExportPack:
    """extra_pdfs: 보고서와 함께 낼 PDF(예: 실행물 묶음) — (경로, 설명)."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    folder = Path(out_dir) / f"반출신청_{stamp}"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    files: list[PackFile] = []
    if report_pdf and Path(report_pdf).exists():
        dst = folder / f"01_보고서_{safe_name(report.title)}.pdf"
        shutil.copy2(report_pdf, dst)
        files.append(PackFile(dst, "보고서", f"분석 결과 보고서 '{report.title}'", pdf_pages(dst)))
    n_extra = 0
    for pth, desc in extra_pdfs or []:
        if pth and Path(pth).exists():
            n_extra += 1
            dst = folder / f"{1 + n_extra:02d}_실행물_{safe_name(Path(pth).stem, 40)}.pdf"
            shutil.copy2(pth, dst)
            files.append(PackFile(dst, "실행물", desc, pdf_pages(dst)))
    figs = figures if figures is not None else [b.path for b in report.figures()]
    if include_figures:
        caps = {str(b.path): re.sub(r"^그림\s*\d+\.\s*", "", b.caption) for b in report.figures()}
        for i, fp in enumerate(figs, start=1):
            dst = folder / f"{i + 1 + n_extra:02d}_그림{i:02d}_{safe_name(caps.get(str(fp), Path(fp).stem), 30)}.png"
            shutil.copy2(fp, dst)
            files.append(PackFile(dst, "그림", caps.get(str(fp), "그림")))

    n_pdf = sum(1 for f in files if f.role == "보고서")
    n_png = sum(1 for f in files if f.role == "그림")
    purpose = purpose_text(report, use, n_pdf, n_png, n_extra=n_extra)
    sources = sources_block(report)
    check = checklist_frame(report)
    explain_name = "00_반출신청_설명자료.pdf"
    pages = next((f.pages for f in files if f.role == "보고서"), None)
    desc_lines = [
        f"[반출 파일] 보고서 PDF {sum(1 for f in files if f.role == '보고서')}개"
        + (f"({pages}쪽)" if pages else "") + (f", 실행물 묶음 PDF {n_extra}개" if n_extra else "")
        + (f", 그림 PNG {n_png}개" if n_png else "") + f", 설명자료 PDF 1개({explain_name})",
        f"[내용] '{report.title}' — " + ", ".join(
            s.heading.split(". ", 1)[-1] for s in report.sections if not s.heading.startswith(("부록", "참고"))),
        "[원본 데이터]",
        *[f"  - {l}" for l in sources.splitlines()],
        "[가공 처리] " + ", ".join(processing_phrases(report.results))
        + ". 보고서·그림의 'x'는 반출 기준에 따라 비공개 처리한 칸입니다.",
        f"[자동 점검] 표·그림 {len(report.results)}개: 10 미만 단순집계 수치 0건, 합계로 역산 가능한 비공개 칸 0건(행렬 계수 검사).",
        "[미포함] 원본·발췌 자료, CSV·엑셀 등 텍스트 파일, 소스코드는 포함하지 않았습니다.",
        f"[상세 설명] 첨부한 '{explain_name}' 참고",
    ]
    description = "\n".join(desc_lines)

    problems, warnings = preflight(files, report)
    explain = _explain_report(report, purpose, sources, description, check, files, explain_name, problems, warnings)
    html_path = folder.parent / f".설명자료_{stamp}.html"
    to_html(explain, html_path, appendix=False)
    pdf = to_pdf(html_path, folder / explain_name)
    if pdf:
        files.insert(0, PackFile(pdf, "설명자료", "반출 신청 설명자료(목적·출처·점검표)", pdf_pages(pdf)))
        html_path.unlink(missing_ok=True)
    else:
        warnings.append("설명자료 PDF를 만들지 못했습니다(브라우저 인쇄 실패). 같은 폴더의 HTML을 브라우저에서 열어 PDF로 저장하세요.")
    return ExportPack(folder, files, purpose, sources, description, check, problems, warnings)


def _explain_report(report, purpose, sources, description, check, files, explain_name, problems, warnings) -> Report:
    flist = pd.DataFrame([{"파일": explain_name, "구분": "설명자료", "내용": "반출 신청 설명자료(이 문서)"}]
                         + [{"파일": f.path.name, "구분": f.role, "내용": f.desc + (f" · {f.pages}쪽" if f.pages else "")} for f in files])
    ex = Report(
        title="반출 신청 설명자료",
        subtitle=f"'{report.title}' 반출 신청용 · 심사 담당자 확인용",
        synthetic=report.synthetic,
        meta={"신청 파일": f"{len(files) + 1}개(PDF·PNG)", "작성일": datetime.now().strftime("%Y-%m-%d %H:%M"),
              "작성 도구": "데이터올레(제주빅데이터센터 분석 도우미)"},
    )
    ex.sections.append(Section("1. 반출 목적(신청서 '활용 사유'란)", [Paragraph(purpose)]))
    ex.sections.append(Section("2. 반출 파일 목록", [Table(flist, style="text", show_index=False, widths=[46, 10, 44])]))
    ex.sections.append(Section("3. 원본 데이터(출처)", [Bullets(sources.splitlines()),
                               Paragraph("출처 표기는 반출 기준의 형식(제주빅데이터센터, 데이터셋(이름, 기간))을 따랐고, "
                                         "보고서의 모든 그림·표 아래에도 같은 출처를 적었습니다.")]))
    ex.sections.append(Section("4. 표·그림별 반출 기준 점검표", [Table(check, style="text", show_index=False, widths=CHECK_WIDTHS)]))
    ex.sections.append(Section("5. 가공(비공개) 처리 방법", [Bullets([
        "단순집계(합계·건수·인원)는 10 미만(0 초과) 칸을 'x'로 비공개 처리했습니다. 0(자료 없음)은 '-'로 표시했습니다.",
        "합계·소계와 다른 칸으로 비공개 칸이 다시 계산되지 않도록, 표 전체를 연립방정식으로 보고 비공개 칸 각각이 "
        "풀리지 않는지(행렬 계수 비교) 확인했으며, 풀리는 경우 가장 작은 칸을 추가로 비공개 처리했습니다.",
        "1인당·건당 금액 같은 평균값은 바탕 인원·건수가 10 미만인 칸을 비공개 처리했습니다. 여러 달을 합친 경우 바탕 인원은 "
        "'월별 최댓값'(실제 인원의 하한)으로 판정했습니다.",
        "구성비·증감률·지수·순위는 응용집계로 표시했으며, 바탕이 되는 집계값을 먼저 검수한 뒤 변환했습니다.",
        "비공개 칸이 많은 표는 상위 단위(월→분기, 법정동→권역, 연령 세분→묶음)로 다시 집계하는 범주처리를 적용했습니다.",
    ])]))
    ex.sections.append(Section("6. 신청서 '설명'란 입력 문구", [Paragraph(l) for l in description.splitlines()]))
    items = [f"고칠 것: {p}" for p in problems] + [f"확인: {w}" for w in warnings]
    if items:
        ex.sections.append(Section("7. 신청 전 자동 확인 결과", [Callout("확인 사항", items, "caution")]))
    else:
        ex.sections.append(Section("7. 신청 전 자동 확인 결과", [Callout("확인 사항", ["형식·용량·검수 항목에서 문제를 찾지 못했습니다."], "note")]))
    ex.sections.append(Section("참고", [Paragraph("이 설명자료는 반출 신청 준비를 돕기 위해 자동 작성되었습니다. "
                                                "반출 여부는 제주빅데이터센터의 심사 결과에 따릅니다.")]))
    return ex
