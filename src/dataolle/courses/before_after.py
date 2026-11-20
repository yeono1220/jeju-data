"""코스 3. 논문용 전후 비교(이중차분, DiD-lite).

질문 예: "2025년 7월 이후 한림읍 카페 이용이 늘었나?" → 처리 지역(한림읍)과 비교 지역(사건 전 추이가 가장 비슷한 법정동, 자동 매칭)의
월별 이용금액을 사건 전 평균=100으로 맞춘 뒤, (처리 지역 변화) − (비교 지역 변화)를 로그로 계산한다.

강건성
  - 공간 위약 검정: 비교 지역 하나하나를 '가짜 처리 지역'으로 놓고 같은 계산을 반복 → 실제 효과의 순위로 p값 근사
  - 사전 추세: 사건 전 기간의 (처리 − 비교) 로그 차이의 월별 기울기
결과는 지수·증감률·검정값(응용집계)이라 반출 기준상 수치 포함 반출이 가능하다. 단순집계(금액)는 보고서에 넣지 않는다.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from dataolle.analysis.derive import make_derived
from dataolle.analysis.engine import AnalysisResult, Catalog, run_analysis
from dataolle.analysis.facts import FactStore
from dataolle.analysis.fmt import fmt_month, fmt_value
from dataolle.analysis.josa import josa
from dataolle.analysis.semantic import MODELS, Dim
from dataolle.analysis.spec import AnalysisSpec
from dataolle.courses.market import BIZ_LINK, CourseOutput, _Builder
from dataolle.report.document import Bullets, Callout, Paragraph, Report, Section, Table, to_docx, to_html, to_pdf
from dataolle.sdc.engine import MASK_TOKEN, AggKind, Level


@dataclass
class BeforeAfterInput:
    treat: list[str]                       # 처리 지역(법정동)
    event: str                             # 사건 시작 월(YYYYMM, 이 달부터 '사건 후')
    biz: str | None = None                 # 업종(없으면 전체 업종)
    control: list[str] | None = None       # 비교 지역(없으면 사건 전 추이가 가장 비슷한 법정동을 자동 매칭)
    measure: str = "amt"                   # amt(이용금액) | cnt(이용건수)
    event_name: str = ""                   # 예: '○○ 개장'
    title: str | None = None


def did(t: pd.Series, c: pd.Series, event: str) -> float:
    """로그 이중차분: [평균 log T(후) − 평균 log T(전)] − [평균 log C(후) − 평균 log C(전)]."""
    lt, lc = np.log(t.astype(float)), np.log(c.astype(float))
    pre = [i for i in t.index if str(i) < event]
    post = [i for i in t.index if str(i) >= event]
    return float((lt[post].mean() - lt[pre].mean()) - (lc[post].mean() - lc[pre].mean()))


def pre_trend(t: pd.Series, c: pd.Series, event: str) -> float:
    """사건 전 (log T − log C)의 월별 기울기(%/월)."""
    pre = [i for i in t.index if str(i) < event]
    y = np.log(t[pre].astype(float)) - np.log(c[pre].astype(float))
    if len(y) < 3:
        return float("nan")
    x = np.arange(len(y))
    return float(np.polyfit(x, y.to_numpy(), 1)[0] * 100)


class BeforeAfterCourse:
    def __init__(self, catalog: Catalog, out_dir: str | Path, level: Level = Level.CONSERVATIVE):
        self.cat, self.out, self.level = catalog, Path(out_dir), level

    def run(self, inp: BeforeAfterInput, progress=None, make_pdf: bool = True, make_docx: bool = True,
            facts=None) -> CourseOutput:
        t0 = time.time()
        b = _Builder(self.cat, self.out, self.level, progress, facts)
        P = lambda fid, attr="": "{{" + fid + (f".{attr}" if attr else "") + "}}"
        months = self.cat.months("card")
        pre_m = [m for m in months if m < inp.event]
        post_m = [m for m in months if m >= inp.event]
        if len(pre_m) < 3 or len(post_m) < 3:
            raise ValueError(f"사건 전후로 최소 3개월씩 필요합니다(현재 전 {len(pre_m)}개월, 후 {len(post_m)}개월).")
        period = [months[0], months[-1]]
        label = BIZ_LINK.get(inp.biz, {}).get("label", inp.biz) if inp.biz else "전체 업종"
        tname = "·".join(inp.treat)
        mlabel = MODELS["card"].measure(inp.measure).label
        ev_text = fmt_month(inp.event) + (f" {inp.event_name}" if inp.event_name else "")
        bf = {"biz": [inp.biz]} if inp.biz else {}

        b.step("1/4 처리 지역과 사건 전 추이가 비슷한 비교 지역을 찾는 중")
        rt = run_analysis(AnalysisSpec(dataset="card", measure=inp.measure, rows="month", filters={**bf, "bjd": inp.treat},
                                       period=period), self.cat, FactStore(), self.level)
        st_ = rt.table.iloc[:, 0]
        if st_.isna().any() or (st_ <= 0).any():
            raise ValueError("처리 지역에 비공개(x)이거나 0인 달이 있어 전후 비교를 할 수 없습니다. 지역·업종을 넓혀 주세요.")
        piv = run_analysis(AnalysisSpec(dataset="card", measure=inp.measure, rows="month", cols="bjd", filters=bf,
                                        period=period, totals=False), self.cat, FactStore(), self.level)
        T = piv.table.reindex(st_.index)
        pool = [c for c in T.columns if c not in inp.treat and T[c].notna().all() and (T[c] > 0).all()]
        if len(pool) < 3:
            raise ValueError("비교할 수 있는 법정동이 3곳 미만입니다(작은 지역은 비공개 처리됨). 업종을 넓혀 주세요.")
        k = max(2, min(5, len(pool) // 3))
        ind = lambda sr: sr / sr[pre_m].mean() * 100

        def match(target: pd.Series, cands: list[str]) -> list[str]:
            ti = ind(target)[pre_m]
            score = {c: float(np.sqrt(((ind(T[c])[pre_m] - ti) ** 2).mean())) for c in cands}
            return sorted(cands, key=score.get)[:k]

        control = [c for c in (inp.control or []) if c in pool] or match(st_, pool)
        matched = not inp.control
        sc_idx = pd.concat([ind(T[c]) for c in control], axis=1).mean(axis=1)
        t_idx = ind(st_)
        idx = pd.DataFrame({tname: t_idx, "비교 지역": sc_idx})
        idx.index.name = "연월"
        disp = idx.apply(lambda col: col.map(lambda v: fmt_value(v, "지수")))
        series = make_derived([rt, piv], idx, disp, f"{label} {mlabel} 지수(사건 전 평균=100)", f"{mlabel} 지수", "지수",
                              AggKind.APPLIED, MODELS["card"].dim("month"), Dim("series", "", "비교 대상", "category"),
                              method="응용집계: 지역별 월 값 ÷ 사건 전 기간 평균 × 100(비교 지역은 지수의 단순 평균)")

        b.step("2/4 이중차분 효과를 추정하는 중")
        d = did(t_idx, sc_idx, inp.event)
        eff = (np.exp(d) - 1) * 100
        pt = pre_trend(t_idx, sc_idx, inp.event)
        pre_t, post_t = t_idx[pre_m].mean(), t_idx[post_m].mean()
        pre_c, post_c = sc_idx[pre_m].mean(), sc_idx[post_m].mean()

        b.step("3/4 공간 위약 검정(다른 법정동을 가짜 처리 지역으로)을 하는 중")
        placebo = {}
        for u in pool:
            cu = match(T[u], [c for c in pool if c != u])
            cu_idx = pd.concat([ind(T[c]) for c in cu], axis=1).mean(axis=1)
            placebo[u] = (np.exp(did(ind(T[u]), cu_idx, inp.event)) - 1) * 100
        n_pl = len(placebo)
        p_val = (sum(1 for v in placebo.values() if abs(v) >= abs(eff)) + 1) / (n_pl + 1) if n_pl else float("nan")
        top = sorted(placebo.items(), key=lambda kv: abs(kv[1]), reverse=True)[:11]
        pl = pd.DataFrame({"효과(%)": pd.Series({**dict(top), f"{tname}(실제)": eff})}).sort_values("효과(%)", ascending=False)
        pl.index.name = "지역"
        pl_disp = pl.apply(lambda col: col.map(lambda v: fmt_value(v, "%")))
        plr = make_derived([rt, piv], pl, pl_disp, "공간 위약 검정: 가짜 효과가 큰 지역과 실제 효과", "추정 효과", "%",
                           AggKind.APPLIED, Dim("area", "", "지역", "geo"), None,
                           method="응용집계(검정값): 다른 법정동 각각을 처리 지역으로 가정한 이중차분 효과")
        f_eff = b.fact(f"{tname} {label} {mlabel} 이중차분 효과", eff, "%", AggKind.APPLIED, series.title, change=True)
        f_pt = b.fact(f"{tname} 사건 전 상대 추세(월별 기울기)", pt, "%", AggKind.APPLIED, series.title, change=True)
        f_p = b.fact("공간 위약 검정 p값(근사)", p_val, "p", AggKind.APPLIED, plr.title)
        f_npl = b.fact("위약 검정에 쓴 비교 지역 수", n_pl, "개", AggKind.APPLIED, plr.title)
        f_ctl = b.fact("비교 지역 법정동 수", len(control), "곳", AggKind.APPLIED, series.title)
        f_tc = [b.fact(f"{tname} 사건 전 평균 지수", pre_t, "지수", AggKind.APPLIED, series.title),
                b.fact(f"{tname} 사건 후 평균 지수", post_t, "지수", AggKind.APPLIED, series.title),
                b.fact("비교 지역 사건 전 평균 지수", pre_c, "지수", AggKind.APPLIED, series.title),
                b.fact("비교 지역 사건 후 평균 지수", post_c, "지수", AggKind.APPLIED, series.title)]
        sig = (not np.isnan(p_val)) and p_val <= 0.1
        trend_ok = np.isnan(pt) or abs(pt) < 1.0

        b.step("4/4 보고서를 정리하는 중")
        sections = []
        design = pd.DataFrame({"항목": ["처리 지역", "비교 지역", "사건 시작", "사건 전", "사건 후", "업종", "지표", "추정 방법"],
                               "내용": [tname, f"{', '.join(control[:8])}{' 외' if len(control) > 8 else ''}",
                                      ev_text, f"{fmt_month(pre_m[0])}~{fmt_month(pre_m[-1])}({len(pre_m)}개월)",
                                      f"{fmt_month(post_m[0])}~{fmt_month(post_m[-1])}({len(post_m)}개월)", label, mlabel,
                                      "로그 이중차분(DiD), 공간 위약 검정, 사전 추세 점검"]})
        how = (f"비교 지역은 사건 전 {len(pre_m)}개월의 지수 흐름이 처리 지역과 가장 비슷한 법정동 {len(control)}곳(매칭)입니다."
               if matched else "비교 지역은 이용자가 지정한 법정동입니다.")
        sections.append(Section("1. 분석 설계", [Table(design, style="text", show_index=False, widths=[22, 78]),
                                              Paragraph(how + " 두 지역의 계절·경기 변화가 같다고 가정하고, 사건 후 처리 지역만의 "
                                                        "추가 변화를 효과로 봅니다.")]))
        fig = b.figure(series, series.title, kind="line", event=inp.event, event_label=f"사건 시작({fmt_month(inp.event)})",
                       ref_line=100)
        blocks = [Paragraph(b.render(
            f"사건 전 평균을 100으로 맞추면 {josa(tname, '은')} 사건 후 평균 {P(f_tc[1])}, 비교 지역은 {P(f_tc[3])}입니다."))]
        if fig:
            blocks.append(fig)
        sections.append(Section("2. 추이 비교", blocks))
        res_tab = pd.DataFrame({"사건 전(평균 지수)": [fmt_value(pre_t, "지수"), fmt_value(pre_c, "지수")],
                                "사건 후(평균 지수)": [fmt_value(post_t, "지수"), fmt_value(post_c, "지수")]},
                               index=[tname, "비교 지역"])
        res_tab.index.name = "구분"
        b.keep(series)
        verdict = ("통계적으로 의미 있는 차이로 볼 수 있습니다" if sig else "우연한 변동과 구별하기 어렵습니다")
        blocks = [Table(res_tab, "표 1. 사건 전후 평균 지수"),
                  Callout("추정 결과", [b.render(f"이중차분 효과: {P(f_eff)} (사건 후 {tname}의 {mlabel}이 비교 지역 대비 "
                                               f"{P(f_eff, 'abs')} {P(f_eff, 'dir')})"),
                                      b.render(f"공간 위약 검정 p값 ≈ {P(f_p)} (다른 법정동 {P(f_npl)}를 가짜 처리 지역으로 반복) → {verdict}"),
                                      b.render(f"사건 전 상대 추세: 월 {P(f_pt)} "
                                               + ("(평행 추세 가정에 큰 문제 없음)" if trend_ok else "(사전 추세 차이가 있어 해석 주의)"))],
                          "insight")]
        figp = b.figure(plr, plr.title, kind="hbar", highlight=[f"{tname}(실제)"])
        if figp:
            blocks.append(figp)
        sections.append(Section("3. 추정 결과와 강건성", blocks))
        methods = (f"분석에는 제주빅데이터센터의 카드 이용 현황(읍면동) 자료(비씨카드 결제분, {fmt_month(months[0])}~{fmt_month(months[-1])}, "
                   f"월 단위)를 사용하였다. 처리 지역은 {tname}, 비교 지역은 사건 전 추이가 가장 비슷한 법정동 {P(f_ctl)}이며, "
                   f"{fmt_month(inp.event)}을 사건 시점으로 두고 {label}의 {mlabel}을 비교하였다. 효과는 로그 이중차분 "
                   "[(처리 지역 사건 후 평균 − 사건 전 평균) − (비교 지역 사건 후 평균 − 사건 전 평균)]으로 추정하고 백분율로 환산하였다. "
                   "강건성 확인을 위해 다른 법정동 각각을 처리 지역으로 가정하고 같은 절차(매칭·이중차분)를 반복하는 공간 위약 검정을 수행하여 "
                   "실제 효과의 순위로 p값을 근사하였고, "
                   "사건 전 기간의 상대 추세 기울기로 평행 추세 가정을 점검하였다. 반출 기준에 따라 10 미만 수치는 비공개 처리하였으며, "
                   "보고하는 수치는 모두 지수·증감률·검정값(응용집계)이다.")
        results_p = (f"추정 결과, 사건 후 {tname}의 {mlabel}은 비교 지역 대비 {P(f_eff, 'abs')} {P(f_eff, 'dir')}하였다"
                     f"(이중차분 효과 {P(f_eff)}). 공간 위약 검정의 근사 p값은 {P(f_p)}로 {'유의수준 10%에서 유의하였다' if sig else '유의하지 않았다'}. "
                     f"사건 전 상대 추세는 월 {P(f_pt)}로 " + ("평행 추세 가정을 크게 벗어나지 않았다." if trend_ok else "다소 차이가 있어 해석에 주의가 필요하다."))
        sections.append(Section("4. 논문용 문단", [Callout("데이터 및 방법", [b.render(methods)], "note"),
                                                Callout("결과", [b.render(results_p)], "note")]))
        sections.append(Section("5. 해석 시 유의사항", [Bullets([
            "월 단위 자료라 사건 직후의 짧은 효과나 다른 동시 사건의 영향을 분리하기 어렵습니다.",
            "사건 전 기간이 짧으면 계절 진폭(성수기 차이)이 다른 지역이 비교 지역으로 뽑혀 효과가 부풀거나 줄 수 있습니다. "
            "전년 같은 달 자료를 함께 받으면 전년 대비로 보정할 수 있습니다.",
            "비씨카드 결제분 표본이므로 시장 전체가 아니라 상대적 변화로 해석해야 합니다.",
            "위약 검정에 쓸 수 있는 법정동이 적으면 p값의 최소값이 커집니다(n곳이면 최소 1/(n+1)). 작은 지역은 비공개 처리되어 빠집니다.",
            "인과 해석은 '다른 조건이 같았다면'이라는 가정에 의존합니다. 사건과 같은 시기의 다른 변화(개발·축제 등)를 함께 검토하세요.",
        ])]))
        b.keep(plr)
        title = inp.title or f"{tname} {label} {mlabel} 전후 비교({ev_text})"
        report = Report(title, "논문·연구용 · 이중차분 전후 비교", sections, list(b.results),
                        {"처리 지역": tname, "비교 지역": f"{', '.join(control)}" + (" (사건 전 추이 매칭)" if matched else ""), "사건 시작": ev_text,
                         "분석 기간": f"{fmt_month(months[0])}~{fmt_month(months[-1])}", "작성일": datetime.now().strftime("%Y-%m-%d"),
                         "작성 도구": "데이터올레"},
                        synthetic=self.cat.synthetic,
                        ai_note="문장은 규칙 기반 템플릿으로 작성했고 모든 수치는 계산 결과에서 채웠습니다(AI 미사용).")
        report.extra["methods_text"] = b.render(methods)
        report.extra["results_text"] = b.render(results_p)
        report.extra["effect"] = {"did_pct": eff, "p": p_val, "placebos": n_pl, "pre_trend": pt}
        report.extra["drafts"] = {"summary": [results_p], "rx": [], "plan": results_p}
        self.out.mkdir(parents=True, exist_ok=True)
        stem = f"전후비교_{tname}_{label}_{inp.event}".replace("/", "·").replace(" ", "")
        html = to_html(report, self.out / f"{stem}.html")
        pdf = to_pdf(html, self.out / f"{stem}.pdf") if make_pdf else None
        docx = to_docx(report, self.out / f"{stem}.docx") if make_docx else None
        return CourseOutput(report, b.facts, html, pdf, docx, b.figures, b.log, b.skipped, time.time() - t0)
