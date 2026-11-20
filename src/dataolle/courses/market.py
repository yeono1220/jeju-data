"""코스 1. 사업계획서 시장분석.

입력: 업종(카드 업종명) + 후보 법정동. 출력: 사업계획서 '시장분석' 절에 옮겨 쓸 수 있는 보고서(PDF·DOCX)와 그림(PNG).

구성
  1. 시장 규모와 추세 ........ 카드 1-08: 월별 이용금액, 계절 패턴(제주 전체 대비 지수)
  2. 입지 비교 ................ 카드 1-08: 같은 권역 법정동 비교, 업종 특화지수(입지계수, 제주 평균=100)
  3. 고객 구성 ................ 카드 1-08: 도민·관광객 구성비, 연령대 구성비 (후보지·권역·제주 전체)
  4. 요일·시간대 .............. 카드 1-12·1-14: 요일·시간대 구성비 → 핵심 영업시간
  5. 객단가 .................... 카드 1-08: 건당 이용금액 비교
  6. 외식업 POS ............... POS 1-19·1-24: 표본 점포당 매출 추이, 인기 메뉴·판매단가 (외식 업종일 때)
  7. 유동인구 ................. 유동인구 4-11: 시각별 방문인구, 거주·근무·방문 구성
  8. 시사점 · 사업계획서 문단 · 한계 · 부록(출처·반출 검수)

모든 수치는 반출 검수를 통과한 결과(FactStore)에서만 문장으로 들어간다.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from dataolle.analysis.derive import combine_cols, make_derived, ratio_index, stack_rows
from dataolle.analysis.engine import AnalysisResult, Catalog, mask_ratio, run_analysis, run_safe
from dataolle.analysis.facts import FactStore
from dataolle.analysis.fmt import fmt_month, fmt_value
from dataolle.analysis.josa import josa
from dataolle.analysis.semantic import MODELS, Dim
from dataolle.analysis.spec import AnalysisSpec
from dataolle.report.charts import render_chart
from dataolle.report.document import Bullets, Callout, Figure, Paragraph, Report, Section, Table, to_docx, to_html, to_pdf
from dataolle.sdc.engine import AggKind, Level

BIZ_LINK = {
    "비알콜 음료점업": dict(label="카페·음료점", pos_mid="커피/음료", pos_biz=["커피전문점"], menu_large=["음료류"]),
    "한식 음식점업": dict(label="한식 음식점", pos_mid="한식", pos_biz=None, menu_large=["밥류", "국/탕/찌개/전골류"]),
    "중식 음식점업": dict(label="중식 음식점", pos_mid="중식", pos_biz=None, menu_large=["면류", "볶음류"]),
    "일식 음식점업": dict(label="일식 음식점", pos_mid="일식/수산물", pos_biz=None, menu_large=["회류", "밥류"]),
    "서양식 음식점업": dict(label="양식 음식점", pos_mid="양식", pos_biz=None, menu_large=["면류", "구이류"]),
    "피자, 햄버거, 샌드위치 및 유사 음식점업": dict(label="패스트푸드점", pos_mid="패스트푸드", pos_biz=None, menu_large=["빵/과자류"]),
    "기타 주점업": dict(label="주점", pos_mid="간이주점", pos_biz=None, menu_large=["주류"]),
    "일반유흥 주점업": dict(label="유흥주점", pos_mid="유흥주점", pos_biz=None, menu_large=["주류"]),
    "빵 및 과자류 소매업": dict(label="베이커리", pos_mid="제과/제빵/떡/케익", pos_biz=None, menu_large=["빵/과자류"]),
}

AREA = Dim("area", "", "비교 대상", "geo")
TOURIST = ("개별관광", "단체관광")


@dataclass
class MarketInput:
    biz: str                                   # 카드 업종명(예: 비알콜 음료점업)
    bjd: str                                   # 후보 법정동(예: 애월읍, 월평동은 '월평동(제주시)')
    period: tuple[str, str] | None = None      # (YYYYMM, YYYYMM), 없으면 최근 12개월
    persona: str = ""                          # 예: 애월읍 카페 창업 준비자
    title: str | None = None


@dataclass
class CourseOutput:
    report: Report
    facts: FactStore
    html: Path
    pdf: Path | None
    docx: Path | None
    figures: list[Path]
    log: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    seconds: float = 0.0


class _Builder:
    """결과·그림·사실을 모으는 작은 도우미."""

    def __init__(self, catalog: Catalog, out_dir: Path, level: Level, progress: Callable[[str], None] | None,
                 facts: FactStore | None = None):
        self.cat, self.out, self.level = catalog, out_dir, level
        self.facts = facts if facts is not None else FactStore()      # 에이전트는 여러 코스에 한 저장소를 넘겨 번호가 겹치지 않게 함
        self.results: list[AnalysisResult] = []
        self.figures: list[Path] = []
        self.log: list[str] = []
        self.skipped: list[str] = []
        self.progress = progress
        self.nfig = 0
        self.ntab = 0

    def step(self, msg: str):
        self.log.append(msg)
        if self.progress:
            self.progress(msg)

    def run(self, safe: bool = True, **kw) -> AnalysisResult:
        spec = AnalysisSpec(**kw)
        r = run_safe(spec, self.cat, self.facts, self.level) if safe else run_analysis(spec, self.cat, self.facts, self.level)
        return r

    def quiet(self, **kw) -> AnalysisResult:
        """사실을 만들지 않는 보조 집계(파생 계산의 재료)."""
        return run_analysis(AnalysisSpec(**kw), self.cat, FactStore(), self.level)

    def keep(self, r: AnalysisResult) -> AnalysisResult:
        if r not in self.results:
            self.results.append(r)
        return r

    def fact(self, label: str, value, unit: str, kind: AggKind, source: str = "", change: bool = False) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            f = self.facts.add(label, None, unit, kind, source, masked=True, is_change=change)
        else:
            f = self.facts.add(label, float(value), unit, kind, source, is_change=change)
        return f.id

    def figure(self, r: AnalysisResult, caption: str, **kw) -> Figure | None:
        self.keep(r)
        self.nfig += 1
        path = self.out / "figures" / f"그림{self.nfig:02d}.png"
        p = render_chart(r, path, **kw)
        if p is None:
            self.nfig -= 1
            return None
        self.figures.append(p)
        return Figure(p, f"그림 {self.nfig}. {caption}", r)

    def table(self, r: AnalysisResult, caption: str, max_rows: int = 14) -> Table:
        self.keep(r)
        self.ntab += 1
        d = r.display
        if len(d) > max_rows:
            d = pd.concat([d.iloc[: max_rows - 1], d.iloc[[-1]]]) if "합계" in d.index else d.iloc[:max_rows]
        return Table(d, f"표 {self.ntab}. {caption}", r)

    def render(self, text: str) -> str:
        return self.facts.render(text)


def _short(v) -> str:
    s = str(v)
    return fmt_month(s) if (len(s) == 6 and s.isdigit()) else s


def _val(r: AnalysisResult, row=None, col=None):
    t = r.table
    if t.empty:
        return np.nan
    try:
        if row is None:
            return float(t.iloc[0, 0] if col is None else t[col].iloc[0])
        if row not in t.index:
            return np.nan
        return float(t.loc[row].iloc[0] if col is None else t.loc[row, col])
    except (KeyError, ValueError, TypeError):
        return np.nan


class MarketCourse:
    def __init__(self, catalog: Catalog, out_dir: str | Path, level: Level = Level.CONSERVATIVE):
        self.cat = catalog
        self.out = Path(out_dir)
        self.level = level

    # -- 준비 -----------------------------------------------------------------
    def _period(self, inp: MarketInput) -> tuple[str, str]:
        if inp.period:
            return inp.period
        months = self.cat.months("card")
        if not months:
            raise ValueError("카드 이용 현황(읍면동) 데이터가 없습니다.")
        end = months[-1]
        start = months[max(0, len(months) - 12)]
        return (start, end)

    def _region_of(self, bjd: str) -> tuple[str | None, str | None]:
        m = MODELS["card"]
        row = self.cat.con.execute(
            f"SELECT dstrct_se, city, count(*) c FROM {self.cat.tables['card']} WHERE ({m.dim('bjd').expr}) = ? "
            "GROUP BY 1, 2 ORDER BY c DESC LIMIT 1", [bjd]).fetchone()
        return (row[0], row[1]) if row else (None, None)

    def _hdong_of(self, bjd: str, city: str | None) -> list[str]:
        import json
        from importlib import resources
        data = json.loads(resources.files("dataolle.spec").joinpath("data/codes.json").read_text(encoding="utf-8"))
        name = bjd.split("(")[0]
        approx = data.get("hdong_approx", {})
        for c in ([city] if city else []) + ["제주시", "서귀포시"]:
            hit = approx.get(c, {}).get(name)
            if hit:
                return list(hit)
        return []

    # -- 실행 -----------------------------------------------------------------
    def run(self, inp: MarketInput, progress: Callable[[str], None] | None = None, make_pdf: bool = True,
            make_docx: bool = True, gateway=None, facts: FactStore | None = None) -> CourseOutput:
        t0 = time.time()
        b = _Builder(self.cat, self.out, self.level, progress, facts)
        period = list(self._period(inp))
        region, city = self._region_of(inp.bjd)
        if region is None:
            raise ValueError(f"'{inp.bjd}' 법정동을 카드 데이터에서 찾지 못했습니다.")
        link = BIZ_LINK.get(inp.biz, {})
        label = link.get("label", inp.biz)
        place = f"{inp.bjd} {label}"
        P = lambda fid, attr="": "{{" + fid + (f".{attr}" if attr else "") + "}}"   # 자리표시자
        base_f = {"bjd": [inp.bjd], "biz": [inp.biz]}
        reg_f = {"region": [region], "biz": [inp.biz]}
        all_f = {"biz": [inp.biz]}
        areas = [inp.bjd, f"{region}(권역)", "제주 전체"]
        filters3 = [base_f, reg_f, all_f]
        sections: list[Section] = []
        summary: list[str] = []         # 핵심 요약(합니다체)
        insights: list[str] = []        # 시사점
        plan: list[str] = []            # 사업계획서 문단(-다체)

        # 1. 시장 규모와 추세 ---------------------------------------------------
        b.step("1/7 시장 규모와 추세를 계산하는 중")
        tot = b.keep(b.run(dataset="card", measure="amt", filters=base_f, period=period, subject=place,
                           title=f"{place} 기간 합계 이용금액", safe=False))
        f_total = b.fact(f"{place} 이용금액 합계({tot.period_text})", _val(tot), "원", AggKind.SIMPLE, tot.title)
        trend = b.run(dataset="card", measure="amt", rows="month", filters=base_f, period=period, subject=place,
                      title=f"{place} 월별 이용금액")
        s = trend.table.iloc[:, 0].dropna()
        blocks = [Paragraph(b.render(f"{tot.period_text} {place}의 비씨카드 이용금액은 모두 {P(f_total)}입니다. "
                                     + " ".join(trend.sentences)))]
        fig = b.figure(trend, trend.title)
        if fig:
            blocks.append(fig)
        amp_word = None
        if len(s) >= 3:
            idx_t = b.quiet(dataset="card", measure="amt", rows="month", filters=base_f, period=period, transform="index")
            idx_j = b.quiet(dataset="card", measure="amt", rows="month", filters=all_f, period=period, transform="index")
            comb = combine_cols([idx_t, idx_j], [inp.bjd, "제주 전체"], f"{label} 월별 이용금액 지수(첫 달=100)",
                                value_label="이용금액 지수")
            sj = comb.table["제주 전체"].dropna()
            amp_t = s.max() / s.min() if s.min() else np.nan
            amp_j = sj.max() / sj.min() if len(sj) and sj.min() else np.nan
            f_amp_t = b.fact(f"{place} 최고 달 ÷ 최저 달", amp_t, "배", AggKind.APPLIED, comb.title)
            f_amp_j = b.fact(f"제주 전체 {label} 최고 달 ÷ 최저 달", amp_j, "배", AggKind.APPLIED, comb.title)
            amp_word = "크" if amp_t > amp_j * 1.05 else ("작" if amp_t < amp_j * 0.95 else "비슷하")
            tail = {"크": "계절에 따른 차이가 더 큽니다. 성수기·비수기 대비가 필요합니다.",
                    "작": "계절에 따른 차이가 더 작아 연중 매출이 비교적 고른 편입니다.",
                    "비슷하": "계절에 따른 차이가 비슷한 수준입니다."}[amp_word]
            jl = josa(f"제주 전체 {label}", "과")[len(f"제주 전체 {label}"):]
            blocks.append(Paragraph(b.render(
                f"가장 많은 달은 가장 적은 달의 {P(f_amp_t)}(으)로, 제주 전체 {label}({P(f_amp_j)}){jl} 견주면 {tail} "
                "아래 그림은 두 지역의 첫 달을 100으로 맞춰 계절 흐름만 비교한 것입니다.")))
            fig2 = b.figure(comb, comb.title)
            if fig2:
                blocks.append(fig2)
            if amp_word == "크":
                insights.append(f"월별 차이가 제주 평균보다 큽니다(최고·최저 달 {P(f_amp_t)}). 비수기 운영자금과 도민 단골 확보 계획을 함께 쓰세요.")
            peak, low = s.idxmax(), s.idxmin()
            summary.append(f"{place} 이용금액은 {tot.period_text} 동안 {P(f_total)}이며, {_short(peak)}에 가장 많았습니다.")
            plan.append(f"월별로는 {_short(peak)}에 가장 많고 {_short(low)}에 가장 적어 최고·최저 달 차이가 {P(f_amp_t)}"
                        f"(제주 전체 같은 업종 {P(f_amp_j)})이다.")
        blocks.append(b.table(trend, trend.title))
        sections.append(Section("1. 시장 규모와 추세", blocks))

        # 2. 입지 비교 ------------------------------------------------------------
        b.step("2/7 같은 권역 법정동과 비교하는 중")
        comp = b.run(dataset="card", measure="amt", rows="bjd", filters=reg_f, period=period, subject=f"{region} {label}",
                     title=f"{region} 법정동별 {label} 이용금액", top_n=15)
        blocks = []
        rank_s = comp.table.iloc[:, 0].drop(index=[i for i in comp.table.index if i == "기타"]).dropna().sort_values(ascending=False)
        rank_clause = ""
        if inp.bjd in rank_s.index:
            rk = list(rank_s.index).index(inp.bjd) + 1
            full = comp.check.table.loc["합계"].iloc[0] if "합계" in comp.check.table.index else comp.table.iloc[:, 0].sum()
            share = rank_s[inp.bjd] / float(full) * 100 if not isinstance(full, str) and full else np.nan
            f_rk = b.fact(f"{region} 안 {inp.bjd} {label} 이용금액 순위", rk, "위", AggKind.APPLIED, comp.title)
            f_nb = b.fact(f"{region} 권역의 법정동 수", len(rank_s), "개", AggKind.APPLIED, comp.title)
            f_sh = b.fact(f"{region} {label} 이용금액 중 {inp.bjd} 비중", share, "%", AggKind.APPLIED, comp.title)
            top3 = ", ".join(str(x) for x in rank_s.index[:3])
            blocks.append(Paragraph(b.render(
                f"같은 권역({region})의 법정동 {P(f_nb)}와 비교하면 {josa(inp.bjd, '은')} {label} 이용금액 {P(f_rk)}이며 "
                f"권역 전체의 {P(f_sh)}(을)를 차지합니다." + (f" 상위 3곳은 {top3}입니다." if len(rank_s) > 3 else ""))))
            rank_clause = f"{region} 권역 법정동 {P(f_nb)} 중 {P(f_rk)}(권역 비중 {P(f_sh)})"
        fig = b.figure(comp, comp.title, highlight=[inp.bjd])
        if fig:
            blocks.append(fig)
        num = b.quiet(dataset="card", measure="amt", rows="bjd", filters=reg_f, period=period)
        den = b.quiet(dataset="card", measure="amt", rows="bjd", filters={"region": [region]}, period=period)
        j_biz = b.quiet(dataset="card", measure="amt", filters=all_f, period=period)
        j_all = b.quiet(dataset="card", measure="amt", period=period)
        ref = _val(j_biz) / _val(j_all) if _val(j_all) else np.nan
        lq = ratio_index(num, den, f"{region} 법정동별 {label} 특화지수(제주 평균=100)", "특화지수", scale=100.0, ref=ref,
                         method=f"응용집계: (법정동의 {label} 이용금액 ÷ 법정동 전체 업종 이용금액) ÷ (제주 전체 같은 비율) × 100")
        lq.table = lq.table.sort_values(lq.table.columns[0], ascending=False)
        lq.display = lq.display.reindex(lq.table.index)
        v_lq = _val(lq, inp.bjd)
        f_lq = b.fact(f"{inp.bjd} {label} 특화지수(제주 평균=100)", v_lq, "지수", AggKind.APPLIED, lq.title)
        lq_word = "높" if v_lq > 110 else ("낮" if v_lq < 90 else "비슷")
        lq_text = {"높": "제주 평균보다 높습니다. 수요가 검증된 상권이지만 경쟁도 치열할 수 있습니다.",
                   "낮": "제주 평균보다 낮습니다. 덜 붐비는 시장이지만 수요 근거를 따로 확인해야 합니다.",
                   "비슷": "제주 평균과 비슷한 수준입니다."}[lq_word]
        blocks.append(Paragraph(b.render(
            f"특화지수는 그 지역 전체 카드 이용금액 가운데 {label}의 비중을 제주 전체의 같은 비중과 견준 값으로, "
            f"100보다 크면 그 업종이 몰린 곳입니다. {inp.bjd}의 특화지수는 {P(f_lq)}(으)로 {lq_text}")))
        figlq = b.figure(lq, lq.title, highlight=[inp.bjd], ref_line=100, ref_label="제주 평균 100")
        if figlq:
            blocks.append(figlq)
        insights.append({"높": f"특화지수 {P(f_lq)}: {label}가 이미 몰린 상권입니다. 차별화 포인트(메뉴·공간·영업시간)를 분명히 쓰세요.",
                         "낮": f"특화지수 {P(f_lq)}: {label} 비중이 낮은 상권입니다. 유동인구·관광객 흐름으로 수요 근거를 보강하세요.",
                         "비슷": f"특화지수 {P(f_lq)}: 제주 평균 수준의 {label} 상권입니다. 입지보다 상품·운영 경쟁력이 관건입니다."}[lq_word])
        summary.append(f"{label} 특화지수는 {P(f_lq)}(제주 평균=100)로 " + {"높": "업종이 몰린 상권입니다.", "낮": "업종 비중이 낮은 상권입니다.",
                                                                       "비슷": "제주 평균 수준입니다."}[lq_word])
        sections.append(Section("2. 입지 비교", blocks))

        # 3. 고객 구성 ------------------------------------------------------------
        b.step("3/7 고객 구성(도민·관광객, 연령대)을 계산하는 중")
        blocks = []
        tour = self._compare_share(b, "tour_type", filters3, areas, period, f"{label} 이용금액의 관광구분 구성비")
        if tour is not None:
            tv = [np.nansum([_val(tour, a, c) for c in TOURIST if c in tour.table.columns]) for a in areas]
            f_tt = b.fact(f"{inp.bjd} {label} 관광객(개별+단체) 비중", tv[0], "%", AggKind.APPLIED, tour.title)
            f_tj = b.fact(f"제주 전체 {label} 관광객 비중", tv[2], "%", AggKind.APPLIED, tour.title)
            f_td = b.fact(f"{inp.bjd} 관광객 비중 − 제주 전체", tv[0] - tv[2], "%p", AggKind.APPLIED, tour.title, change=True)
            tw = "높" if tv[0] > tv[2] + 2 else ("낮" if tv[0] < tv[2] - 2 else "비슷")
            blocks.append(Paragraph(b.render(
                f"{place} 이용금액 가운데 관광객(개별+단체)의 몫은 {P(f_tt)}(으)로 제주 전체({P(f_tj)})"
                + {"높": f"보다 높습니다(차이 {P(f_td)}).", "낮": f"보다 낮습니다(차이 {P(f_td)}).", "비슷": "와 비슷합니다."}[tw])))
            fig = b.figure(tour, tour.title)
            if fig:
                blocks.append(fig)
            summary.append(f"이용금액의 {P(f_tt)}(이)가 관광객(개별+단체) 몫으로 제주 전체({P(f_tj)})"
                           + {"높": "보다 높습니다.", "낮": "보다 낮습니다.", "비슷": "와 비슷합니다."}[tw])
            insights.append({"높": f"관광객 비중이 높은 상권입니다({P(f_tt)}). 관광 동선 노출(지도·SNS), 포장·기념 상품을 강조하세요.",
                             "낮": f"도민 비중이 높은 상권입니다(관광객 {P(f_tt)}). 재방문(단골)·평일 수요 중심으로 계획하세요.",
                             "비슷": f"관광객 비중이 제주 평균 수준입니다({P(f_tt)}). 도민·관광객을 함께 겨냥한 구성이 필요합니다."}[tw])
            plan.append(f"이용금액의 {P(f_tt)}(이)가 관광객(개별·단체)에게서 발생해 제주 전체 평균({P(f_tj)})"
                        + {"높": "보다 관광객 의존도가 높다.", "낮": "보다 도민 비중이 높다.", "비슷": "과 비슷하다."}[tw])
        age = self._compare_share(b, "age", filters3, areas, period, f"{label} 이용금액의 연령대 구성비")
        if age is not None:
            cols = [c for c in age.table.columns if c != "알수없음"]
            top_age = max(cols, key=lambda c: _val(age, inp.bjd, c) if not np.isnan(_val(age, inp.bjd, c)) else -1)
            f_ag = b.fact(f"{inp.bjd} {label} 이용금액 1위 연령대({top_age}) 비중", _val(age, inp.bjd, top_age), "%",
                          AggKind.APPLIED, age.title)
            f_agj = b.fact(f"제주 전체 {label} {top_age} 비중", _val(age, "제주 전체", top_age), "%", AggKind.APPLIED, age.title)
            blocks.append(Paragraph(b.render(f"연령대로는 {josa(top_age, '이')} 가장 큰 고객층으로 이용금액의 {P(f_ag)}"
                                             f"(제주 전체 {P(f_agj)})를 차지합니다.")))
            fig = b.figure(age, age.title)
            if fig:
                blocks.append(fig)
        # 핵심 고객층(연령대×성별) — 최근 1개월 이용자수. 작은 칸은 반출 기준에 따라 x로 가려진다.
        last = period[1]
        xs = b.run(dataset="card", measure="users", rows="age", cols="sex", filters=base_f, period=[last, last],
                   subject=place, title=f"{place} 연령대×성별 이용자수({_short(last)})", safe=False)
        if not xs.table.empty:
            stacked = xs.table.drop(index=[i for i in xs.table.index if i == "알수없음"],
                                    columns=[c for c in xs.table.columns if c == "알수없음"], errors="ignore").stack().dropna()
            if len(stacked):
                (ra, ca) = stacked.idxmax()
                f_x = b.fact(f"{place} 이용자수가 가장 많은 연령대×성별({ra} {ca}, {_short(last)})", stacked.max(), "명",
                             AggKind.SIMPLE, xs.title)
                note = (f" 표의 x {xs.check.masked_cells}칸은 10명 미만이거나 합계로 역산될 수 있어 반출 기준에 따라 가린 칸입니다."
                        if xs.check.masked_cells else "")
                blocks.append(Paragraph(b.render(
                    f"{_short(last)} 한 달 기준으로 이용자가 가장 많은 고객층은 {ra} {ca}입니다({P(f_x)}). "
                    "사업계획서의 '목표 고객' 설정에 활용할 수 있습니다." + note)))
                blocks.append(b.table(xs, xs.title))
                insights.append(f"핵심 고객층은 {ra} {ca}입니다. 메뉴·공간·홍보 채널을 이 고객층에 맞춰 설계하세요.")
        if blocks:
            sections.append(Section("3. 고객 구성", blocks))
        else:
            b.skipped.append("고객 구성")

        # 4. 요일·시간대 ----------------------------------------------------------
        b.step("4/7 요일·시간대 패턴을 계산하는 중")
        blocks = []
        hours = {"오전": "07~11시", "점심": "11~14시", "오후": "14~18시", "저녁": "18~22시", "밤": "22~07시"}
        if self.cat.has("card_week"):
            wk = self._compare_share(b, "week", filters3, areas, period, f"{label} 이용건수의 요일 구성비", dataset="card_week",
                                     measure="cnt")
            if wk is not None:
                we = [np.nansum([_val(wk, a, c) for c in ("토요일", "일요일") if c in wk.table.columns]) for a in areas]
                f_we = b.fact(f"{inp.bjd} {label} 주말(토·일) 이용건수 비중", we[0], "%", AggKind.APPLIED, wk.title)
                f_wj = b.fact(f"제주 전체 {label} 주말 비중", we[2], "%", AggKind.APPLIED, wk.title)
                ww = "높" if we[0] > we[2] + 2 else ("낮" if we[0] < we[2] - 2 else "비슷")
                blocks.append(Paragraph(b.render(
                    f"주말(토·일) 이용건수 비중은 {P(f_we)}(으)로 제주 전체({P(f_wj)})"
                    + {"높": "보다 높아 주말 장사 비중이 큰 편입니다.", "낮": "보다 낮아 평일 수요가 탄탄한 편입니다.",
                       "비슷": "와 비슷합니다."}[ww]
                    + " (요일별 표의 '공휴일' 행은 다른 요일에도 중복 집계되어 있어 제외했습니다.)")))
                fig = b.figure(wk, wk.title)
                if fig:
                    blocks.append(fig)
                if ww == "높":
                    insights.append(f"주말 비중이 높습니다({P(f_we)}). 주말 인력·재고를 넉넉히, 평일은 단축 운영도 검토하세요.")
        else:
            b.skipped.append("요일 패턴(1-12 데이터 없음)")
        if self.cat.has("card_time"):
            tm = self._compare_share(b, "time", filters3, areas, period, f"{label} 이용건수의 시간대 구성비", dataset="card_time",
                                     measure="cnt")
            if tm is not None:
                row = tm.table.loc[inp.bjd].dropna().sort_values(ascending=False)
                core, acc = [], 0.0
                for k, v in row.items():
                    core.append(k)
                    acc += v
                    if acc >= 70:
                        break
                core_txt = "·".join(f"{c}({hours.get(c, c)})" for c in core)
                f_core = b.fact(f"{inp.bjd} {label} 핵심 시간대({'·'.join(core)}) 이용건수 비중", acc, "%", AggKind.APPLIED, tm.title)
                f_top = b.fact(f"{inp.bjd} {label} 1위 시간대({row.index[0]}) 비중", row.iloc[0], "%", AggKind.APPLIED, tm.title)
                blocks.append(Paragraph(b.render(
                    f"시간대로는 {josa(row.index[0] + '(' + hours.get(row.index[0], '') + ')', '이')} 가장 많고({P(f_top)}), "
                    f"{core_txt}에 이용건수의 {P(f_core)}(이)가 몰려 있습니다. 인력 배치와 영업시간을 이 시간대에 맞추는 것이 효율적입니다.")))
                fig = b.figure(tm, tm.title)
                if fig:
                    blocks.append(fig)
                insights.append(f"핵심 영업시간은 {core_txt}입니다(이용건수의 {P(f_core)}).")
                summary.append(f"이용건수의 {P(f_core)}(이)가 {'·'.join(core)} 시간대에 몰립니다.")
                plan.append(f"이용건수는 {'·'.join(core)} 시간대에 {P(f_core)}(이)가 집중된다.")
        else:
            b.skipped.append("시간대 패턴(1-14 데이터 없음)")
        if blocks:
            sections.append(Section("4. 요일·시간대 패턴", blocks))

        # 5. 객단가 ---------------------------------------------------------------
        b.step("5/7 객단가를 비교하는 중")
        aps = [b.quiet(dataset="card", measure="amt_per_cnt", filters=f, period=period) for f in filters3]
        apc = stack_rows(aps, areas, f"{label} 건당 이용금액(객단가) 비교", AREA,
                         method="평균값: 이용금액 합계 ÷ 이용건수 합계(바탕 이용자 10명 미만이면 비공개)")
        apc.table.columns = ["건당 이용금액"]
        apc.display.columns = ["건당 이용금액"]
        v_ap = [_val(apc, a) for a in areas]
        f_ap = b.fact(f"{place} 건당 이용금액", v_ap[0], "원", AggKind.AVERAGE, apc.title)
        f_apj = b.fact(f"제주 전체 {label} 건당 이용금액", v_ap[2], "원", AggKind.AVERAGE, apc.title)
        f_apr = b.fact(f"{inp.bjd} 객단가 ÷ 제주 전체 − 1", (v_ap[0] / v_ap[2] - 1) * 100 if v_ap[2] else np.nan, "%",
                       AggKind.APPLIED, apc.title, change=True)
        hi = v_ap[0] >= v_ap[2]
        blocks = [Paragraph(b.render(f"건당 이용금액(객단가)은 {P(f_ap)}(으)로 제주 전체({P(f_apj)})보다 {P(f_apr, 'abs')} "
                                     f"{'높습니다' if hi else '낮습니다'}. 가격 책정과 예상 매출(객단가 × 예상 건수) 계산의 기준으로 쓸 수 있습니다."))]
        fig = b.figure(apc, apc.title, kind="bar", highlight=[inp.bjd])
        if fig:
            blocks.append(fig)
        summary.append(f"건당 이용금액은 {P(f_ap)}(으)로 제주 전체보다 {P(f_apr, 'abs')} {'높습니다' if hi else '낮습니다'}.")
        insights.append(f"객단가 기준점은 {P(f_ap)}입니다. " + ("제주 평균보다 높은 상권이라 프리미엄 메뉴 구성이 가능합니다."
                                                           if hi else "제주 평균보다 낮아 가격 경쟁력·세트 구성이 중요합니다."))
        plan.append(f"건당 결제금액은 {P(f_ap)}(으)로 제주 평균({P(f_apj)})보다 {P(f_apr, 'abs')} {'높다' if hi else '낮다'}.")
        sections.append(Section("5. 객단가", blocks))

        # 6. 외식업 POS -----------------------------------------------------------
        if link and self.cat.has("pos"):
            b.step("6/7 외식업 POS(표본 점포 매출·메뉴)를 계산하는 중")
            blocks = []
            pos_name = (link.get("pos_biz") or [link["pos_mid"]])[0]
            pf = {"pos_biz": link["pos_biz"]} if link.get("pos_biz") else {"pos_mid": [link["pos_mid"]]}
            pos_r = b.run(dataset="pos", measure="sales_per_store", rows="month", filters={**pf, "bjd": [inp.bjd]}, period=period,
                          subject=f"{inp.bjd} {pos_name}", title=f"{inp.bjd} {pos_name} 표본 점포당 월 매출액")
            scope = inp.bjd
            if mask_ratio(pos_r) > 0.5 and city:
                pos_r = b.run(dataset="pos", measure="sales_per_store", rows="month", filters={**pf, "city": [city]},
                              period=period, subject=f"{city} {pos_name}", title=f"{city} {pos_name} 표본 점포당 월 매출액")
                scope = city
                blocks.append(Paragraph(f"{josa(inp.bjd, '은')} {pos_name} 표본 점포가 적어(10곳 미만이면 비공개) "
                                        f"{city} 단위로 넓혀 보았습니다."))
            ps = pos_r.table.iloc[:, 0].dropna()
            if len(ps):
                f_pm = b.fact(f"{scope} {pos_name} 표본 점포당 월 매출액(기간 평균)", ps.mean(), "원", AggKind.AVERAGE, pos_r.title)
                blocks.append(Paragraph(b.render(
                    f"외식업 POS 표본 점포 기준 {scope} {pos_name}의 점포당 월 매출액은 평균 {P(f_pm)}입니다. "
                    + " ".join(pos_r.sentences) + " 표본 점포는 전체 점포가 아니므로 규모보다 추세·비교에 활용하세요.")))
                fig = b.figure(pos_r, pos_r.title)
                if fig:
                    blocks.append(fig)
                plan.append(f"외식업 POS 표본 기준 {scope} {pos_name}의 점포당 월 매출은 평균 {P(f_pm)}이다.")
            if self.cat.has("pos_menu") and link.get("menu_large"):
                months = self.cat.months("pos_menu")
                mperiod = [months[-3:][0], months[-1]] if months else None
                mf = {"menu_large": link["menu_large"], "city": [city]} if city else {"menu_large": link["menu_large"]}
                where = city or "제주"
                menu = b.run(dataset="pos_menu", measure="avg_cnt", rows="menu", filters=mf, period=mperiod, top_n=10,
                             subject=f"{where} {'·'.join(link['menu_large'])}",
                             title=f"{where} {'·'.join(link['menu_large'])} 인기 메뉴(점포당 월 판매건수)", safe=False)
                price = b.quiet(dataset="pos_menu", measure="avg_price", rows="menu", filters=mf, period=mperiod)
                if not menu.table.empty:
                    tops = [i for i in menu.table.index if i != "기타"][:3]
                    pf_ids = [b.fact(f"{m} 평균 판매단가", _val(price, m), "원", AggKind.AVERAGE, menu.title) for m in tops]
                    blocks.append(Paragraph(b.render(
                        f"최근 메뉴 자료({menu.period_text})에서 점포당 판매건수가 많은 메뉴는 "
                        + ", ".join(f"{m}(평균 단가 {P(fid)})" for m, fid in zip(tops, pf_ids)) + " 순입니다. "
                        "메뉴 구성과 가격표를 정할 때 참고하세요.")))
                    fig = b.figure(menu, menu.title)
                    if fig:
                        blocks.append(fig)
                    pt = price.table.reindex([i for i in menu.table.index if i in price.table.index])
                    if not pt.empty:
                        price.table = pt
                        price.display = price.display.reindex(pt.index)
                        price.title = f"{where} 인기 메뉴 평균 판매단가"
                        blocks.append(b.table(price, price.title))
                    if tops:
                        insights.append(f"가장 많이 팔리는 메뉴는 {tops[0]}(평균 단가 {P(pf_ids[0])})입니다. "
                                        "대표 메뉴 가격을 지역 평균 단가와 비교해 정하세요.")
            if blocks:
                sections.append(Section("6. 외식업 POS로 본 매출·메뉴", blocks))
        else:
            b.skipped.append("외식업 POS(외식 업종이 아니거나 POS 데이터 없음)")

        # 7. 유동인구 -------------------------------------------------------------
        hd = self._hdong_of(inp.bjd, city)
        if self.cat.has("pop") and hd:
            b.step("7/7 유동인구(시각별 방문인구)를 계산하는 중")
            blocks = []
            hname = "·".join(hd)
            popf = {"emd": hd}
            pr = b.run(dataset="pop", measure="visit_hourly", rows="hour", filters=popf, subject=hname,
                       title=f"{hname} 시각별 시간당 평균 방문인구")
            hs = pr.table.iloc[:, 0].dropna()
            if len(hs):
                pk = hs.idxmax()
                f_pk = b.fact(f"{hname} 방문인구가 가장 많은 시각({pk})의 시간당 평균 방문인구", hs.max(), "명", AggKind.SIMPLE, pr.title)
                approx = "" if hd == [inp.bjd.split("(")[0]] else f"(법정동 {inp.bjd} → 행정동 {hname} 근사 대응)"
                blocks.append(Paragraph(b.render(
                    f"유동인구 자료{approx}로 보면 방문인구는 {pk}에 가장 많습니다(시간당 평균 {P(f_pk)}, {pr.period_text}). "
                    "방문인구는 그 지역에 살거나 일하지 않는 사람으로, 외부 고객의 흐름을 보여 줍니다.")))
                fig = b.figure(pr, pr.title)
                if fig:
                    blocks.append(fig)
            comp_res = [[b.quiet(dataset="pop", measure=m, filters=scope_f) for m in ("resd_pop", "work", "visit")]
                        for scope_f in (popf, {})]
            tab = pd.DataFrame([[_val(p) for p in parts] for parts in comp_res], index=[hname, "제주 전체"],
                               columns=["거주", "근무", "방문"], dtype=float)
            share = tab.div(tab.sum(axis=1), axis=0) * 100
            disp = share.apply(lambda c: c.map(lambda v: fmt_value(v, "%") if not np.isnan(v) else "x"))
            share.index.name = disp.index.name = "비교 대상"
            mix = make_derived([p for parts in comp_res for p in parts], share, disp, "유동인구 구성(거주·근무·방문, 연인원 기준)",
                               "구성비", "%", AggKind.APPLIED, AREA, Dim("ptype", "", "인구 구분", "category"),
                               method="응용집계: 거주·근무·방문 인구(연인원) 합계의 구성비")
            mix.spec = mix.spec.model_copy(update={"transform": "share"})
            vs, vj = share.iloc[0]["방문"], share.iloc[1]["방문"]
            f_vs = b.fact(f"{hname} 유동인구 중 방문인구 비중", vs, "%", AggKind.APPLIED, mix.title)
            f_vj = b.fact("제주 전체 유동인구 중 방문인구 비중", vj, "%", AggKind.APPLIED, mix.title)
            vw = "높" if vs > vj + 2 else ("낮" if vs < vj - 2 else "비슷")
            blocks.append(Paragraph(b.render(
                f"유동인구 가운데 방문인구 비중은 {P(f_vs)}(으)로 제주 전체({P(f_vj)})"
                + {"높": "보다 높아 외부 유입이 많은 곳입니다.", "낮": "보다 낮아 주민·직장인 중심 지역입니다.", "비슷": "와 비슷합니다."}[vw])))
            fig = b.figure(mix, mix.title, kind="stacked")
            if fig:
                blocks.append(fig)
            if len(hs):
                insights.append(f"방문인구는 {hs.idxmax()}에 가장 많습니다. 이 시간대 전후로 매장 노출·프로모션을 집중하세요.")
            sections.append(Section("7. 유동인구", blocks))
        else:
            b.skipped.append("유동인구(4-11 데이터 없음 또는 행정동 대응 없음)")

        # 요약 · 시사점 · 사업계획서 문단 -------------------------------------------
        b.step("보고서를 정리하는 중")
        from dataolle.ai.narrator import narrate
        cites = list(dict.fromkeys(c for r in b.results for c in r.citations()))
        lead = (f"제주빅데이터센터의 비씨카드 이용 데이터({tot.period_text})를 분석한 결과, {place}의 이용금액은 {P(f_total)}"
                + (f"(으)로 {rank_clause}이다." if rank_clause else "이다."))
        draft_plan = " ".join([lead] + plan)
        if gateway is not None and getattr(gateway, "enabled", False):
            b.step("AI가 문장을 다듬는 중(숫자는 자리표시자로만 전달)")
        nar = narrate(gateway, b.facts, summary, insights, draft_plan,
                      {"제목": f"{place} 시장분석", "대상": inp.persona or f"{place} 창업 준비자",
                       "절": [sec.heading for sec in sections], "업종": label, "지역": f"{inp.bjd}({city}, {region})"})
        plan_text = b.render(nar.plan)
        sections.insert(0, Section("요약", [
            Callout("핵심 요약", [b.render(x) for x in nar.summary], "insight"),
            Paragraph(f"이 보고서는 '{inp.persona or place + ' 창업 준비자'}'를 위해 제주빅데이터센터 데이터로 {place}의 시장 규모, "
                      "입지, 고객, 시간대, 객단가를 분석한 것입니다. 모든 수치는 반출 기준(10 미만 비공개·역산 방지) 검수를 통과한 값입니다.")]))
        sections.append(Section("8. 시사점", [Bullets([b.render(i) for i in nar.insights])]))
        sections.append(Section("9. 사업계획서 '시장분석' 붙여넣기용 문단", [
            Callout("그대로 옮겨 쓸 수 있는 문단", [plan_text], "note"),
            Paragraph("출처: " + "; ".join(cites))]))
        if nar.used_ai:
            lvl = gateway.effective_level()
            ai_note = (f"요약·시사점·사업계획서 문단은 AI({nar.model})가 규칙 기반 초안을 다듬었습니다. AI에는 숫자 대신 자리표시자와 "
                       f"데이터 이름만 보냈고(전송 레벨 {lvl}), 돌려받은 문장의 모든 수치는 이 PC에서 계산 결과로 채웠습니다. "
                       "계산 근거가 없는 숫자가 들어 있으면 AI 문장을 버리도록 자동 검증했습니다.")
        else:
            ai_note = "이 보고서의 문장은 규칙 기반 템플릿으로 작성되었습니다(AI 미사용). 수치는 모두 계산 결과에서 자동으로 채웠습니다."
            if nar.problems:
                ai_note += f" (AI 문장 다듬기는 사용하지 않음: {nar.problems[0]})"

        title = inp.title or f"{place} 시장분석 보고서"
        meta = {"분석 대상": f"{inp.bjd} ({city}, {region})", "업종": f"{label} (카드 업종: {inp.biz})",
                "비교 기준": f"{region} 권역, 제주 전체 같은 업종", "분석 기간": f"{tot.period_text} (자료별 제공 기간은 그림·표에 표기)",
                "작성일": datetime.now().strftime("%Y-%m-%d"), "작성 도구": "데이터올레"}
        report = Report(title, "사업계획서 시장분석용 · 제주빅데이터센터 데이터 기반", sections, list(b.results), meta,
                        limitations=[f"이번 보고서에서 다루지 못한 항목: {', '.join(b.skipped)}"] if b.skipped else [],
                        synthetic=self.cat.synthetic, ai_note=ai_note)
        report.extra["plan_text"] = plan_text
        report.extra["drafts"] = {"summary": list(summary), "rx": list(insights), "plan": draft_plan}
        self.out.mkdir(parents=True, exist_ok=True)
        stem = f"시장분석_{inp.bjd}_{label}".replace("/", "·").replace(" ", "")
        html = to_html(report, self.out / f"{stem}.html")
        pdf = to_pdf(html, self.out / f"{stem}.pdf") if make_pdf else None
        docx = to_docx(report, self.out / f"{stem}.docx") if make_docx else None
        return CourseOutput(report, b.facts, html, pdf, docx, b.figures, b.log, b.skipped, time.time() - t0)

    # -- 비교 지역 구성비 -------------------------------------------------------
    def _compare_share(self, b: _Builder, dim: str, filters3: list[dict], areas: list[str], period, title: str,
                       dataset: str = "card", measure: str = "amt") -> AnalysisResult | None:
        return compare_share(b, dim, filters3, areas, period, title, dataset, measure)


def compare_share(b: _Builder, dim: str, filters3: list[dict], areas: list[str], period, title: str,
                  dataset: str = "card", measure: str = "amt") -> AnalysisResult | None:
    """비교 대상(후보지·권역·제주 전체 등)의 구성비를 한 표로. 첫 대상이 범주처리되면 같은 단위로 맞춘다."""
    first = run_safe(AnalysisSpec(dataset=dataset, measure=measure, rows=dim, filters=filters3[0], period=period,
                                  transform="share"), b.cat, FactStore(), b.level)
    if first.table.empty or first.table.iloc[:, 0].isna().all():
        return None
    dim_used = first.spec.rows
    rest = [run_analysis(AnalysisSpec(dataset=dataset, measure=measure, rows=dim_used, filters=f, period=period,
                                      transform="share"), b.cat, FactStore(), b.level) for f in filters3[1:]]
    return stack_rows([first] + rest, areas, title, AREA,
                      method="응용집계: 비교 대상별 구성비(바탕 집계 검수 후 변환), 새 합계 없음")
