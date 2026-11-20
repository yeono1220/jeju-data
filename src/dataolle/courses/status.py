"""코스 2. 읍면동 현황자료(기관 보고용 한 지역 프로필).

구성: 소비 규모·추세(제주 전체 대비 지수) → 업종 구성과 특화 업종(입지계수) → 이용자 구성(도민·관광객·외국인)
      → 요일·시간대 → 유동인구 → 요약.
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from dataolle.analysis.derive import combine_cols, make_derived, ratio_index
from dataolle.analysis.engine import Catalog
from dataolle.analysis.fmt import fmt_value
from dataolle.analysis.josa import josa
from dataolle.analysis.semantic import Dim
from dataolle.courses.market import AREA, CourseOutput, MarketCourse, _Builder, _short, _val, compare_share
from dataolle.report.document import Bullets, Callout, Paragraph, Report, Section, to_docx, to_html, to_pdf
from dataolle.sdc.engine import AggKind, Level

HOURS = {"오전": "07~11시", "점심": "11~14시", "오후": "14~18시", "저녁": "18~22시", "밤": "22~07시"}


class StatusCourse(MarketCourse):
    def __init__(self, catalog: Catalog, out_dir: str | Path, level: Level = Level.CONSERVATIVE):
        super().__init__(catalog, out_dir, level)

    def run_status(self, bjd: str, period: tuple[str, str] | None = None, title: str | None = None, progress=None,
                   make_pdf: bool = True, make_docx: bool = True, facts=None) -> CourseOutput:
        from dataolle.courses.market import MarketInput
        t0 = time.time()
        b = _Builder(self.cat, self.out, self.level, progress, facts)
        P = lambda fid, attr="": "{{" + fid + (f".{attr}" if attr else "") + "}}"
        per = list(self._period(MarketInput(biz="", bjd=bjd, period=period)))
        region, city = self._region_of(bjd)
        if region is None:
            raise ValueError(f"'{bjd}' 법정동을 카드 데이터에서 찾지 못했습니다.")
        f1, fr = {"bjd": [bjd]}, {"region": [region]}
        areas = [bjd, f"{region}(권역)", "제주 전체"]
        sections, summary = [], []

        b.step("1/5 소비 규모와 추세")
        tot = b.keep(b.run(dataset="card", measure="amt", filters=f1, period=per, subject=bjd, title=f"{bjd} 기간 합계 카드 이용금액",
                           safe=False))
        f_tot = b.fact(f"{bjd} 카드 이용금액 합계({tot.period_text})", _val(tot), "원", AggKind.SIMPLE, tot.title)
        jt = b.quiet(dataset="card", measure="amt", period=per)
        f_js = b.fact(f"제주 전체 카드 이용금액 중 {bjd} 비중", _val(tot) / _val(jt) * 100 if _val(jt) else np.nan, "%",
                      AggKind.APPLIED, tot.title)
        trend = b.run(dataset="card", measure="amt", rows="month", filters=f1, period=per, subject=bjd, title=f"{bjd} 월별 카드 이용금액")
        blocks = [Paragraph(b.render(f"{tot.period_text} {bjd}의 비씨카드 이용금액은 {P(f_tot)}(으)로 제주 전체의 {P(f_js)}입니다. "
                                     + " ".join(trend.sentences)))]
        fig = b.figure(trend, trend.title)
        if fig:
            blocks.append(fig)
        if len(trend.table) >= 3:
            it = b.quiet(dataset="card", measure="amt", rows="month", filters=f1, period=per, transform="index")
            ij = b.quiet(dataset="card", measure="amt", rows="month", period=per, transform="index")
            comb = combine_cols([it, ij], [bjd, "제주 전체"], "월별 카드 이용금액 지수(첫 달=100)", value_label="이용금액 지수")
            fig2 = b.figure(comb, comb.title)
            if fig2:
                blocks.append(fig2)
        summary.append(f"{tot.period_text} 카드 이용금액 {P(f_tot)}, 제주 전체의 {P(f_js)}입니다.")
        sections.append(Section("1. 소비 규모와 추세", blocks))

        b.step("2/5 업종 구성과 특화 업종")
        top = b.run(dataset="card", measure="amt", rows="biz", filters=f1, period=per, top_n=10, subject=bjd,
                    title=f"{bjd} 업종별 카드 이용금액(상위 10)")
        blocks = [Paragraph(b.render(" ".join(top.sentences)))]
        fig = b.figure(top, top.title)
        if fig:
            blocks.append(fig)
        num = b.quiet(dataset="card", measure="amt", rows="biz", filters=f1, period=per)
        den = b.quiet(dataset="card", measure="amt", rows="biz", period=per)
        ref = _val(tot) / _val(jt) if _val(jt) else np.nan
        lq = ratio_index(num, den, f"{bjd} 업종별 특화지수(제주 평균=100, 비중 1% 이상 업종)", "특화지수", scale=100.0, ref=ref,
                         method="응용집계: (업종의 지역 내 비중) ÷ (제주 전체 같은 비중) × 100")
        share = num.table.iloc[:, 0] / _val(tot) * 100 if _val(tot) else num.table.iloc[:, 0] * np.nan
        keep = [i for i in lq.table.index if share.get(i, 0) >= 1 and not np.isnan(lq.table.loc[i].iloc[0])]
        lq.table = lq.table.loc[keep].sort_values(lq.table.columns[0], ascending=False).head(8)
        lq.display = lq.display.reindex(lq.table.index)
        if len(lq.table):
            lead = lq.table.index[0]
            f_lq = b.fact(f"{bjd} 최고 특화 업종({lead}) 특화지수", lq.table.iloc[0, 0], "지수", AggKind.APPLIED, lq.title)
            blocks.append(Paragraph(b.render(f"제주 전체와 비교해 이 지역에 특히 몰린 업종은 {josa(lead, '이')}며 특화지수는 {P(f_lq)}입니다"
                                             "(100보다 크면 제주 평균보다 비중이 높은 업종).")))
            figlq = b.figure(lq, lq.title, kind="hbar", ref_line=100, ref_label="제주 평균 100")
            if figlq:
                blocks.append(figlq)
            summary.append(f"가장 특화된 업종은 {lead}(특화지수 {P(f_lq)})입니다.")
        sections.append(Section("2. 업종 구성과 특화 업종", blocks))

        b.step("3/5 이용자 구성")
        blocks = []
        f3 = [f1, fr, {}]
        tour = compare_share(b, "tour_type", f3, areas, per, "카드 이용금액의 관광구분 구성비")
        if tour is not None:
            tv = [np.nansum([_val(tour, a, c) for c in ("개별관광", "단체관광") if c in tour.table.columns]) for a in areas]
            f_t = b.fact(f"{bjd} 관광객 이용금액 비중", tv[0], "%", AggKind.APPLIED, tour.title)
            f_tj = b.fact("제주 전체 관광객 이용금액 비중", tv[2], "%", AggKind.APPLIED, tour.title)
            blocks.append(Paragraph(b.render(f"카드 이용금액 가운데 관광객(개별+단체)의 몫은 {P(f_t)}(으)로 제주 전체({P(f_tj)})"
                                             + ("보다 높습니다." if tv[0] > tv[2] + 2 else ("보다 낮습니다." if tv[0] < tv[2] - 2 else "와 비슷합니다.")))))
            fig = b.figure(tour, tour.title)
            if fig:
                blocks.append(fig)
            summary.append(f"관광객 이용금액 비중 {P(f_t)}(제주 전체 {P(f_tj)}).")
        ug = compare_share(b, "user_group", f3, areas, per, "카드 이용금액의 이용자구분 구성비(외국인 묶음)")
        if ug is not None and "외국인" in ug.table.columns:
            f_fg = b.fact(f"{bjd} 외국인 이용금액 비중", _val(ug, bjd, "외국인"), "%", AggKind.APPLIED, ug.title)
            f_fgj = b.fact("제주 전체 외국인 이용금액 비중", _val(ug, "제주 전체", "외국인"), "%", AggKind.APPLIED, ug.title)
            blocks.append(Paragraph(b.render(f"외국인 카드 이용금액 비중은 {P(f_fg)}(제주 전체 {P(f_fgj)})입니다.")))
            fig = b.figure(ug, ug.title)
            if fig:
                blocks.append(fig)
        if blocks:
            sections.append(Section("3. 이용자 구성", blocks))

        b.step("4/5 요일·시간대")
        blocks = []
        f2 = [f1, {}]
        if self.cat.has("card_time"):
            tm = compare_share(b, "time", f2, [bjd, "제주 전체"], per, "카드 이용건수의 시간대 구성비", "card_time", "cnt")
            if tm is not None:
                row = tm.table.loc[bjd].dropna().sort_values(ascending=False)
                f_tt = b.fact(f"{bjd} 1위 시간대({row.index[0]}) 이용건수 비중", row.iloc[0], "%", AggKind.APPLIED, tm.title)
                blocks.append(Paragraph(b.render(f"이용건수는 {row.index[0]}({HOURS.get(row.index[0], '')})에 가장 많습니다({P(f_tt)}).")))
                fig = b.figure(tm, tm.title)
                if fig:
                    blocks.append(fig)
        if self.cat.has("card_week"):
            wk = compare_share(b, "daytype", f2, [bjd, "제주 전체"], per, "카드 이용건수의 평일·주말 구성비", "card_week", "cnt")
            if wk is not None and "주말" in wk.table.columns:
                f_we = b.fact(f"{bjd} 주말 이용건수 비중", _val(wk, bjd, "주말"), "%", AggKind.APPLIED, wk.title)
                f_wj = b.fact("제주 전체 주말 이용건수 비중", _val(wk, "제주 전체", "주말"), "%", AggKind.APPLIED, wk.title)
                blocks.append(Paragraph(b.render(f"주말(토·일) 이용건수 비중은 {P(f_we)}(제주 전체 {P(f_wj)})입니다. "
                                                 "요일별 표의 '공휴일' 행은 중복이라 제외했습니다.")))
        if blocks:
            sections.append(Section("4. 요일·시간대", blocks))

        b.step("5/5 유동인구")
        hd = self._hdong_of(bjd, city)
        if self.cat.has("pop") and hd:
            hname = "·".join(hd)
            pr = b.run(dataset="pop", measure="visit_hourly", rows="hour", filters={"emd": hd}, subject=hname,
                       title=f"{hname} 시각별 시간당 평균 방문인구")
            blocks = []
            hs = pr.table.iloc[:, 0].dropna()
            if len(hs):
                f_pk = b.fact(f"{hname} 방문인구 최다 시각({hs.idxmax()})의 시간당 평균 방문인구", hs.max(), "명", AggKind.SIMPLE, pr.title)
                blocks.append(Paragraph(b.render(f"방문인구는 {hs.idxmax()}에 가장 많습니다(시간당 평균 {P(f_pk)}, {pr.period_text}).")))
                fig = b.figure(pr, pr.title)
                if fig:
                    blocks.append(fig)
            parts = [[b.quiet(dataset="pop", measure=m, filters=sf) for m in ("resd_pop", "work", "visit")]
                     for sf in ({"emd": hd}, {})]
            tab = pd.DataFrame([[_val(x) for x in row] for row in parts], index=[hname, "제주 전체"], columns=["거주", "근무", "방문"])
            shr = tab.div(tab.sum(axis=1), axis=0) * 100
            disp = shr.apply(lambda c: c.map(lambda v: fmt_value(v, "%") if not np.isnan(v) else "x"))
            shr.index.name = disp.index.name = "비교 대상"
            mix = make_derived([x for row in parts for x in row], shr, disp, "유동인구 구성(거주·근무·방문, 연인원 기준)", "구성비", "%",
                               AggKind.APPLIED, AREA, Dim("ptype", "", "인구 구분", "category"),
                               method="응용집계: 거주·근무·방문 인구(연인원) 합계의 구성비")
            fig = b.figure(mix, mix.title, kind="stacked")
            if fig:
                blocks.append(fig)
            f_v = b.fact(f"{hname} 방문인구 비중", shr.iloc[0]["방문"], "%", AggKind.APPLIED, mix.title)
            summary.append(f"유동인구 중 방문인구 비중 {P(f_v)}.")
            sections.append(Section("5. 유동인구", blocks))
        else:
            b.skipped.append("유동인구")

        sections.insert(0, Section("요약", [Callout("한눈에 보기", [b.render(x) for x in summary], "insight"),
                                           Paragraph(f"{bjd}({city}, {region})의 소비·이용자·유동인구 현황을 제주빅데이터센터 데이터로 정리했습니다. "
                                                     "모든 수치는 반출 기준(10 미만 비공개·역산 방지) 검수를 통과한 값입니다.")]))
        report = Report(title or f"{bjd} 현황자료", "기관 내부 현황 보고용 · 제주빅데이터센터 데이터 기반", sections, list(b.results),
                        {"대상 지역": f"{bjd} ({city}, {region})", "분석 기간": f"{tot.period_text} (자료별 제공 기간은 그림·표에 표기)",
                         "비교 기준": f"{region} 권역, 제주 전체", "작성일": datetime.now().strftime("%Y-%m-%d"), "작성 도구": "데이터올레"},
                        limitations=[f"다루지 못한 항목: {', '.join(b.skipped)}"] if b.skipped else [],
                        synthetic=self.cat.synthetic, ai_note="문장은 규칙 기반 템플릿으로 작성했고 수치는 계산 결과에서 채웠습니다(AI 미사용).")
        report.extra["drafts"] = {"summary": list(summary), "rx": [], "plan": ""}
        self.out.mkdir(parents=True, exist_ok=True)
        stem = f"현황자료_{bjd}".replace(" ", "")
        html = to_html(report, self.out / f"{stem}.html")
        pdf = to_pdf(html, self.out / f"{stem}.pdf") if make_pdf else None
        docx = to_docx(report, self.out / f"{stem}.docx") if make_docx else None
        return CourseOutput(report, b.facts, html, pdf, docx, b.figures, b.log, b.skipped, time.time() - t0)
