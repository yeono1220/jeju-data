"""파생표: 검수를 통과한 결과들을 나란히 붙이거나(비교 지역별), 비율·지수(입지계수 등)로 다시 계산한다.

입력은 모두 반출 검수를 거친 결과이고, 새 합계를 만들지 않으므로 역산 경로가 새로 생기지 않는다.
가린 칸(x)은 파생 계산에서도 값이 없는 칸(NaN)으로 남아 'x'로 표시된다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dataolle.analysis.engine import AnalysisResult
from dataolle.analysis.fmt import fmt_period, fmt_value
from dataolle.analysis.semantic import Dim
from dataolle.sdc.engine import MASK_TOKEN, Action, AggKind, CheckResult


def _body(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(index=[i for i in df.index if i == "합계"], columns=[c for c in df.columns if c == "합계"])


def _cite_parts(results: list[AnalysisResult]) -> list[tuple[str, str | None, str | None]]:
    out = []
    for r in results:
        for n, a, b, _ in r.source_parts():
            if (n, a, b) not in out:
                out.append((n, a, b))
    return out


def _period(results: list[AnalysisResult]) -> tuple[str | None, str | None]:
    starts = [r.period[0] for r in results if r.period[0]]
    ends = [r.period[1] for r in results if r.period[1]]
    return (min(starts) if starts else None, max(ends) if ends else None)


def make_derived(base: list[AnalysisResult], table: pd.DataFrame, display: pd.DataFrame, title: str, value_label: str,
                 unit: str, kind: AggKind, rows_dim: Dim | None, cols_dim: Dim | None = None, method: str = "",
                 subject: str | None = None, notes: list[str] | None = None) -> AnalysisResult:
    first = base[0]
    masked = int((display.to_numpy() == MASK_TOKEN).sum())
    actions, seen = [], set()
    for r in base:
        for a in r.check.actions:
            # 바탕 표의 비공개·범주처리 내역만 물려받고, 바탕 표 자신의 계산 방법(info)은 반복하지 않음(점검표가 길어짐)
            if a.kind == "info" and (r.method or len(base) > 1):
                continue
            if a.detail not in seen:
                seen.add(a.detail)
                actions.append(a)
    if method:
        actions.append(Action("info", method))
    violations = [v for r in base for v in r.check.violations]
    check = CheckResult(title, first.check.kind if kind != AggKind.APPLIED else AggKind.APPLIED, display.copy(), masked,
                        actions, violations, [])
    all_notes = []
    for r in base:
        for n in r.notes:
            if n not in all_notes:
                all_notes.append(n)
    for n in notes or []:
        if n not in all_notes:
            all_notes.append(n)
    spec = first.spec.model_copy(update={"title": title})
    sources = []
    for r in base:
        for f in r.sources:
            if f not in sources:
                sources.append(f)
    part_files: dict[str, list[str]] = {}
    for r in base:
        for n, _a, _b, files in r.source_parts():
            lst = part_files.setdefault(n, [])
            lst += [f for f in files if f not in lst]
    return AnalysisResult(spec, first.model, title, subject or first.subject, value_label, unit, kind, table, display, check,
                          first.sql, first.params, sources, _period(base), all_notes, synthetic=any(r.synthetic for r in base),
                          dims_override=(rows_dim, cols_dim), cite_parts=_cite_parts(base), method=method,
                          part_files=part_files)


def stack_rows(results: list[AnalysisResult], labels: list[str], title: str, row_dim: Dim | None = None,
               method: str = "") -> AnalysisResult:
    """1차원 결과 여러 개(같은 항목 구성)를 행으로 쌓는다. 예: 애월읍·권역·제주 전체의 관광구분 구성비."""
    row_dim = row_dim or Dim("area", "", "비교 대상", "geo")
    cols = []
    for r in results:
        for i in r.table.index:
            if i not in cols and i != "기타":
                cols.append(i)
    table = pd.DataFrame([[r.table.iloc[:, 0].get(c, np.nan) for c in cols] for r in results], index=labels, columns=cols, dtype=float)
    disp_rows = []
    for r in results:
        d = _body(r.display).iloc[:, 0]
        disp_rows.append([d.get(c, "-") for c in cols])
    display = pd.DataFrame(disp_rows, index=labels, columns=cols)
    table.index.name = display.index.name = row_dim.label
    first = results[0]
    return make_derived(results, table, display, title, first.value_label, first.unit, first.kind, row_dim, first.rows_dim,
                        method or "비교 대상별 결과를 행으로 나란히 배치(새 합계 없음)")


def combine_cols(results: list[AnalysisResult], labels: list[str], title: str, value_label: str | None = None,
                 method: str = "") -> AnalysisResult:
    """같은 행 구성(예: 월)을 가진 1차원 결과를 열로 붙인다. 예: 애월읍·제주 전체의 월별 지수."""
    idx = []
    for r in results:
        for i in r.table.index:
            if i not in idx:
                idx.append(i)
    first = results[0]
    if first.rows_dim is not None and first.rows_dim.kind == "time":
        idx = sorted(idx, key=str)
    table = pd.DataFrame({lab: r.table.iloc[:, 0].reindex(idx) for lab, r in zip(labels, results)}, dtype=float)
    display = pd.DataFrame({lab: _body(r.display).iloc[:, 0].reindex(idx).fillna("-") for lab, r in zip(labels, results)})
    table.index.name = display.index.name = first.table.index.name
    return make_derived(results, table, display, title, value_label or first.value_label, first.unit, first.kind,
                        first.rows_dim, Dim("series", "", "비교 대상", "category"),
                        method or "같은 기준의 결과를 열로 나란히 배치(새 합계 없음)")


def ratio_index(num: AnalysisResult, den: AnalysisResult, title: str, value_label: str, scale: float = 1.0,
                ref: float | None = None, unit: str = "지수", method: str = "") -> AnalysisResult:
    """두 결과의 칸별 비율(응용집계). ref를 주면 ref로 나눈 뒤 scale을 곱한다(예: 입지계수 = 지역 비중 / 전체 비중)."""
    n = num.table.iloc[:, 0]
    d = den.table.iloc[:, 0].reindex(n.index)
    val = (n / d.replace(0, np.nan)) * scale
    if ref:
        val = val / ref
    nd = _body(num.display).iloc[:, 0].reindex(n.index)
    dd = _body(den.display).iloc[:, 0].reindex(n.index)
    disp = []
    for i in n.index:
        if nd.get(i) == MASK_TOKEN or dd.get(i) == MASK_TOKEN:
            disp.append(MASK_TOKEN)
        elif pd.isna(val.get(i)):
            disp.append("-")
        else:
            disp.append(fmt_value(val[i], unit))
    table = val.to_frame(value_label)
    display = pd.DataFrame({value_label: disp}, index=n.index)
    table.index.name = display.index.name = num.table.index.name
    table = table.where(display != MASK_TOKEN)
    return make_derived([num, den], table, display, title, value_label, unit, AggKind.APPLIED, num.rows_dim, None,
                        method or f"응용집계: {num.value_label} ÷ {den.value_label}")


__all__ = ["make_derived", "stack_rows", "combine_cols", "ratio_index", "fmt_period"]
