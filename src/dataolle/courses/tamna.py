"""코스 4. 탐나는전 캐시백 설계(정책 담당자·주민센터·상인회·연구자용) — 진단 → 효과 추정 → 적립 설계 → 실행물 → 효과 확인.

탐나는전 거래 자료(1-32, 일 단위·읍면동·업종·전통시장)는 제주데이터허브에 없고 센터에만 있다.
→ "센터에 와야만 할 수 있는 분석"을 사람(유동인구)·돈(카드·탐나는전)·날씨와 묶어 한 번에:
  · 사용 규모·추세, 전통시장·업종(오일장 장날 효과)
  · 침투지수·사각지대(도민 카드 소비 대비, 제주 평균=100) + 관광객 활용 여지
  · 원도심·신도심 상권, 인구 구조와 동네 유형(고령화·인구 변화), 학령인구 감소 경보  ← courses/dongne.py
  · 요일·날씨(비 오는 날 효과), 정책 전후 효과(주간 이중차분 + 시간 위약 검정)
  · 윈윈 처방, 동네 유형별 처방, 활용 극대화 처방, 기관 보고용 문단
  · 처방 설계: 적립 예산 배분 시뮬레이션(현행·균등·동네 살리기 집중·사각지대 보강, 효과 확인 뒤 조정안)  ← courses/policy.py
  · 읍면동 하나를 고르면 맨 앞에 '동네 진단 카드'
절 번호는 데이터가 있는 절만 모아 마지막에 매긴다.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from dataolle.analysis.derive import make_derived, ratio_index
from dataolle.analysis.engine import Catalog, run_analysis
from dataolle.analysis.facts import FactStore
from dataolle.analysis.fmt import fmt_month, fmt_value
from dataolle.analysis.josa import josa
from dataolle.analysis.semantic import MODELS, Dim
from dataolle.analysis.spec import AnalysisSpec
from dataolle.courses.before_after import did, pre_trend
from dataolle.courses.market import AREA, CourseOutput, _Builder, _val, compare_share
from dataolle.report.document import Bullets, Callout, Paragraph, Report, Section, Table, to_docx, to_html, to_pdf
from dataolle.sdc.engine import AggKind, Level
from dataolle.synth import world as W


@dataclass
class TamnaInput:
    scope: str = "all"                 # all | city | emd
    value: str | None = None           # 시 이름 또는 읍면동 이름
    event: str | None = None           # 정책 시작일(YYYYMMDD) 또는 월(YYYYMM) — 있으면 전후 효과 분석
    event_name: str = ""
    title: str | None = None
    policy: object | None = None       # courses.policy.PolicyInput(예산·기간·목표) — 없으면 기본값


def _holiday_dates() -> set[str]:
    return {y + md for y, lst in W.HOLIDAYS.items() for md in lst}


def rain_effect(daily: pd.Series, rain: pd.Series, holidays: set[str], wet_mm: float = 5.0) -> tuple[float, int, int]:
    """비 오는 날 효과(%): 같은 달·같은 요일 안에서 (비 온 날 평균 log − 안 온 날 평균 log)을 가중평균 → 백분율.
    공휴일·명절은 제외. 반환: (효과 %, 비 온 날 수, 맑은 날 수)."""
    df = pd.DataFrame({"v": daily.astype(float)}).dropna()
    df = df[(df.v > 0) & (~df.index.isin(holidays))]
    df["rain"] = rain.reindex(df.index)
    df = df.dropna()
    if df.empty:
        return float("nan"), 0, 0
    dts = pd.to_datetime(df.index, format="%Y%m%d")
    df["m"], df["wd"], df["wet"], df["lv"] = dts.month, dts.weekday, df["rain"] >= wet_mm, np.log(df["v"])
    num = den = 0.0
    for _, g in df.groupby(["m", "wd"]):
        a, b_ = g[g.wet], g[~g.wet]
        if len(a) and len(b_):
            w = min(len(a), len(b_))
            num += w * (a.lv.mean() - b_.lv.mean())
            den += w
    if den == 0:
        return float("nan"), int(df.wet.sum()), int((~df.wet).sum())
    return float((np.exp(num / den) - 1) * 100), int(df.wet.sum()), int((~df.wet).sum())


def market_day_ratio(series: pd.Series) -> tuple[tuple[int, int] | None, float]:
    """일별 시장 사용액에서 오일장 주기(날짜 끝자리 두 개, 5일 간격)를 찾아 '장날 ÷ 그 밖의 날' 배수를 계산."""
    s = series.dropna()
    s = s[s > 0]
    if len(s) < 20:
        return None, float("nan")
    day = pd.to_datetime(s.index, format="%Y%m%d").day % 10
    by = s.groupby(day % 5).mean()
    if by.empty:
        return None, float("nan")
    r = int(by.idxmax())
    on = s[(day % 5) == r]
    off = s[(day % 5) != r]
    if off.empty or by.max() < 1.8 * by.drop(r).mean():
        return None, float("nan")
    return (r, (r + 5) % 10), float(on.mean() / off.mean())


class TamnaCourse:
    def __init__(self, catalog: Catalog, out_dir: str | Path, level: Level = Level.CONSERVATIVE):
        self.cat, self.out, self.level = catalog, Path(out_dir), level

    def _q(self, **kw):
        return run_analysis(AnalysisSpec(**kw), self.cat, FactStore(), self.level)

    def run(self, inp: TamnaInput, progress=None, make_pdf: bool = True, make_docx: bool = True, gateway=None,
            facts: FactStore | None = None) -> CourseOutput:
        if not self.cat.has("tamna"):
            raise ValueError("탐나는전 이용 현황(1-32) 데이터가 없습니다. ② 데이터 탭에서 먼저 불러오세요.")
        t0 = time.time()
        b = _Builder(self.cat, self.out, self.level, progress, facts)
        P = lambda fid, attr="": "{{" + fid + (f".{attr}" if attr else "") + "}}"
        months = self.cat.months("tamna")
        per = [months[0], months[-1]]
        sf = self._scope_filter(inp)
        where = "제주 전체" if not sf else inp.value
        fm, fg = {**sf, "mflag": ["전통시장"]}, {**sf, "mflag": ["일반 가맹점"]}
        sections, summary, rx, plan = [], [], [], []
        hol = _holiday_dates()
        win: dict = {}                     # 윈윈 처방 재료(지역·시기·근거 사실 번호)
        cycles: list = []                  # 오일장 장날 주기(시장, 날짜 끝자리, 배수) — 처방 설계에도 씀
        f_rm = f_md = None

        # 1. 사용 규모·추세 --------------------------------------------------------
        b.step("1/8 탐나는전 사용 규모와 추세")
        tot = b.keep(b.run(dataset="tamna", measure="amt", filters=sf, period=per, subject=where, safe=False,
                           title=f"{where} 탐나는전 기간 합계 사용금액"))
        f_tot = b.fact(f"{where} 탐나는전 사용금액 합계({tot.period_text})", _val(tot), "원", AggKind.SIMPLE, tot.title)
        apc = b.keep(b.run(dataset="tamna", measure="amt_per_cnt", filters=sf, period=per, subject=where, safe=False,
                           title=f"{where} 탐나는전 건당 사용금액"))
        f_apc = b.fact(f"{where} 탐나는전 건당 사용금액", _val(apc), "원", AggKind.AVERAGE, apc.title)
        trend = b.run(dataset="tamna", measure="amt", rows="month", filters=sf, period=per, subject=where,
                      title=f"{where} 월별 탐나는전 사용금액")
        blocks = [Paragraph(b.render(f"{tot.period_text} {where}의 탐나는전 사용금액은 {P(f_tot)}, 건당 {P(f_apc)}입니다. "
                                     + " ".join(trend.sentences)))]
        fig = b.figure(trend, trend.title)
        if fig:
            blocks.append(fig)
        summary.append(f"{tot.period_text} 사용금액 {P(f_tot)}(건당 {P(f_apc)}).")
        plan.append(f"제주빅데이터센터 탐나는전 이용 자료({tot.period_text})에 따르면 {where}의 탐나는전 사용금액은 {P(f_tot)}이다.")
        sections.append(Section("사용 규모와 추세", blocks))

        # 2. 전통시장·업종 ----------------------------------------------------------
        b.step("2/8 전통시장·업종·오일장 장날")
        blocks = []
        mshare = self._q(dataset="tamna", measure="amt", rows="mflag", filters=sf, period=per, transform="share")
        b.keep(mshare)
        if "전통시장" in mshare.table.index:
            v_ms = _val(mshare, "전통시장")
            f_ms = b.fact(f"{where} 탐나는전 사용금액 중 전통시장 비중", v_ms, "%", AggKind.APPLIED, f"{where} 가맹점 구분별 구성비")
            blocks.append(Paragraph(b.render(f"사용금액 가운데 전통시장 몫은 {P(f_ms)}입니다.")))
        else:
            blocks.append(Paragraph(f"{josa(where, '은')} 탐나는전 전통시장 가맹점 사용이 없는 지역입니다(시장이 없거나 자료에 시장 이름이 없음)."))
        mk = b.run(dataset="tamna", measure="amt", rows="market", filters=fm, period=per, top_n=10, subject=f"{where} 전통시장",
                   title=f"{where} 전통시장별 탐나는전 사용금액")
        if not mk.table.empty and mk.table.iloc[:, 0].notna().any():
            fig = b.figure(mk, mk.title)
            if fig:
                blocks.append(fig)
        bz = b.run(dataset="tamna", measure="amt", rows="biz", filters=sf, period=per, top_n=10, subject=where,
                   title=f"{where} 업종별 탐나는전 사용금액")
        blocks.append(Paragraph(b.render(" ".join(bz.sentences))))
        fig = b.figure(bz, bz.title)
        if fig:
            blocks.append(fig)
        # 오일장 장날 효과
        dm = self._q(dataset="tamna", measure="amt", rows="date", cols="market", filters=fm, period=per, totals=False)
        for m in dm.table.columns:
            cyc, ratio = market_day_ratio(dm.table[m])
            if cyc is not None:
                cycles.append((m, cyc, ratio))
        if cycles:
            cycles.sort(key=lambda x: -x[2])
            def _days_label(cyc):
                a, c = sorted(x if x else 10 for x in cyc)
                return f"{a}·{c}일"
            tabv = pd.DataFrame({"장날(날짜 끝자리)": [_days_label(cyc) for _, cyc, _ in cycles],
                                 "장날 사용액 ÷ 그 밖의 날": [fmt_value(r, "배") for *_, r in cycles]},
                                index=[m for m, _, _ in cycles])
            tabv.index.name = "오일시장"
            num = pd.DataFrame({"배수": [r for *_, r in cycles]}, index=tabv.index)
            mdr = make_derived([dm], num, tabv[["장날 사용액 ÷ 그 밖의 날"]].rename(columns={"장날 사용액 ÷ 그 밖의 날": "배수"}),
                               "오일장 장날 효과(장날 사용액 ÷ 그 밖의 날)", "배수", "배", AggKind.APPLIED,
                               Dim("market", "", "오일시장", "category"), None,
                               method="응용집계: 일별 시장 사용액에서 5일 주기(장날)를 찾아 평균을 비교")
            b.keep(mdr)
            avg_r = float(np.mean([r for *_, r in cycles]))
            f_md = b.fact("오일장 장날 사용액 ÷ 그 밖의 날(평균)", avg_r, "배", AggKind.APPLIED, mdr.title)
            win["md"] = (f_md, [m for m, _, _ in cycles[:3]])
            blocks.append(Table(tabv, f"표 {b.ntab + 1}. 오일장 장날 효과(자료에서 5일 주기를 자동으로 찾음)", mdr, style="text"))
            b.ntab += 1
            blocks.append(Paragraph(b.render(f"오일시장 {len(cycles)}곳은 장날 사용액이 그 밖의 날의 평균 {P(f_md)}입니다. "
                                             "장날에 맞춘 탐나는전 추가 할인·홍보가 효율적입니다.")))
            rx.append(f"오일장 장날 사용액이 평소의 {P(f_md)}입니다. 장날 연계 추가 할인·홍보를 우선 검토하세요.")
        sections.append(Section("전통시장·업종", blocks))

        # 3. 침투지수·사각지대 --------------------------------------------------------
        if self.cat.has("card"):
            b.step("3/8 카드 소비 대비 탐나는전 침투지수와 사각지대")
            cm = set(self.cat.months("card"))
            ov = [m for m in months if m in cm]
            if ov:
                po = [ov[0], ov[-1]]
                cf = {"city": [inp.value]} if inp.scope == "city" and inp.value else {}
                tf = {"city": [inp.value]} if inp.scope == "city" and inp.value else {}
                num = self._q(dataset="tamna", measure="amt", rows="emd", filters=tf, period=po)
                # 탐나는전은 만 14세 이상 누구나 살 수 있지만(단체관광객 지급 등), 사각지대는 '도민 카드 소비'와 견줘 찾는다.
                # 전체 카드 소비(관광객 포함)와 견주면 관광지가 사각지대로 잘못 뽑히기 때문 → 두 지수를 함께 보여 준다.
                den_r = self._q(dataset="card", measure="amt", rows="bjd", filters={**cf, "tour_type": ["도민"]}, period=po)
                den_a = self._q(dataset="card", measure="amt", rows="bjd", filters=cf, period=po)
                t_all = _val(self._q(dataset="tamna", measure="amt", period=po))
                ref_r = t_all / _val(self._q(dataset="card", measure="amt", filters={"tour_type": ["도민"]}, period=po))
                ref_a = t_all / _val(self._q(dataset="card", measure="amt", period=po))
                pen = ratio_index(num, den_r, f"읍면동별 탐나는전 침투지수(도민 카드 소비 대비, 제주 평균=100)", "침투지수",
                                  scale=100.0, ref=ref_r,
                                  method="응용집계: (탐나는전 사용금액 ÷ 도민 카드 이용금액) ÷ (제주 전체 같은 비율) × 100")
                pen_a = ratio_index(num, den_a, "읍면동별 탐나는전 침투지수(전체 카드 소비 대비)", "침투지수", scale=100.0, ref=ref_a,
                                    method="응용집계: (탐나는전 사용금액 ÷ 전체 카드 이용금액) ÷ (제주 전체 같은 비율) × 100")
                big = den_r.table.iloc[:, 0]
                big = big[big >= big.median()].index
                t = pen.table.loc[[i for i in pen.table.index if i in big]].dropna().sort_values(pen.table.columns[0])
                if len(t) >= 3:
                    low = t.head(10)
                    full_r, full_a = pen.table.iloc[:, 0], pen_a.table.iloc[:, 0]
                    pen.table, pen.display = low, pen.display.reindex(low.index)
                    pen.title = "도민 소비가 많은데 탐나는전 침투지수가 낮은 지역(사각지대 상위 10)"
                    worst = low.index[0]
                    f_w = b.fact(f"사각지대 1위 {worst} 침투지수(도민 소비 대비)", low.iloc[0, 0], "지수", AggKind.APPLIED, pen.title)
                    n80 = int((t.iloc[:, 0] < 80).sum())
                    f_n = b.fact("도민 소비 상위 지역 중 침투지수 80 미만 지역 수", n80, "곳", AggKind.APPLIED, pen.title)
                    blocks = [Paragraph(b.render(
                        "탐나는전은 만 14세 이상이면 관광객도 살 수 있지만(제주도는 단체관광객에게 탐나는전을 지급하기도 합니다), "
                        "가맹점 확대·도민 홍보가 필요한 '사각지대'는 도민 카드 소비와 견줘 찾았습니다. "
                        f"도민 소비가 많은 지역 중 {josa(worst, '이')} 가장 낮고(침투지수 {P(f_w)}), 80 미만인 지역이 {P(f_n)}입니다.")),
                    ]
                    fig = b.figure(pen, pen.title, kind="hbar", ref_line=100, ref_label="제주 평균 100", unit_note="제주 평균=100")
                    if fig:
                        blocks.append(fig)
                    # 관광객 활용 여지: 전체 소비 기준 지수가 도민 기준보다 크게 낮은 곳 = 관광객 소비가 탐나는전 밖에 있음
                    gap = (full_r - full_a.reindex(full_r.index)).dropna()
                    gap = gap[[i for i in gap.index if i in big]].sort_values(ascending=False)
                    if len(gap) >= 3 and gap.iloc[0] > 10:
                        tg = gap.head(5)
                        tab = pd.DataFrame({"도민 소비 대비": [fmt_value(full_r[i], "지수") for i in tg.index],
                                            "전체 소비 대비": [fmt_value(full_a[i], "지수") for i in tg.index],
                                            "차이": [fmt_value(v, "지수") for v in tg.values]}, index=tg.index)
                        tab.index.name = "읍면동"
                        num_t = pd.DataFrame({"차이": tg.values}, index=tg.index)
                        tr = make_derived([pen, pen_a], num_t, tab[["차이"]], "관광객 활용 여지(전체 소비 기준 지수가 도민 기준보다 낮은 지역)",
                                          "지수 차이", "지수", AggKind.APPLIED, Dim("emd", "", "읍면동", "geo"), None,
                                          method="응용집계: 도민 소비 대비 지수 − 전체 소비 대비 지수")
                        b.keep(tr)
                        b.ntab += 1
                        blocks.append(Table(tab, f"표 {b.ntab}. 관광객 소비가 탐나는전 밖에 머무는 지역(지수 차이 상위 5)", tr, style="text"))
                        first = tg.index[0]
                        f_g = b.fact(f"{first} 관광객 활용 여지(지수 차이)", tg.iloc[0], "지수", AggKind.APPLIED, tr.title)
                        blocks.append(Paragraph(b.render(
                            f"전체 카드 소비(관광객 포함)와 견주면 {josa(first, '은')} 지수가 도민 기준보다 {P(f_g)}만큼 낮습니다. "
                            "관광객 소비가 탐나는전 밖에서 이뤄진다는 뜻으로, 단체관광 인센티브·관광객 홍보로 넓힐 여지가 큰 곳입니다.")))
                        rx.append(f"관광객 활용 여지가 큰 지역: {', '.join(str(i) for i in tg.index[:3])}. 관광객 대상 탐나는전 안내·인센티브를 이 지역 가맹점과 연계하세요.")
                        win["tour"] = ([str(i) for i in tg.index[:3]], f_g)
                    blocks.append(Paragraph("카드 자료는 비씨카드 결제분 표본이고 탐나는전에는 관광객 사용분도 섞여 있으므로, 침투지수는 실제 점유율이 아니라 "
                                            "지역 간 상대 비교 값입니다."))
                    sections.append(Section("침투지수와 사각지대", blocks))
                    top3 = ", ".join(str(i) for i in low.index[:3])
                    rx.insert(0, f"사각지대 우선 지역: {top3}(도민 소비 대비 침투지수 하위). 가맹점 모집·도민 홍보를 이 지역부터 집중하세요.")
                    summary.append(f"도민 소비 대비 탐나는전이 가장 덜 쓰이는 지역은 {worst}(침투지수 {P(f_w)}).")
                    ro = josa(str(worst), "으로")[len(str(worst)):]
                    plan.append(f"도민 카드 소비 대비 탐나는전 침투지수가 가장 낮은 지역은 {worst}({P(f_w)}, 제주 평균=100){ro} 가맹점 확대가 우선 필요하다.")
        else:
            b.skipped.append("침투지수(카드 1-08 데이터 없음)")

        # 원도심·신도심 상권, 인구 구조와 동네 유형, 학령인구 감소 경보, 동네 유형별 처방 ---------------------------
        dn = None
        if self.cat.has("card") or self.cat.has("pop"):
            b.step("4/8 동네 살리기: 원도심·신도심 상권, 인구 구조, 학령인구")
            from dataolle.courses.dongne import Dongne
            dn = Dongne(b, inp, where, P, per).build()
            sections += [s for k, s in dn.sections if k != "types"]
            summary += dn.summary
            rx += dn.rx
            plan += dn.plan
            if not self.cat.has("pop"):
                b.skipped.append("인구 구조·학령인구(유동인구 4-10·4-11 데이터 없음)")
        else:
            b.skipped.append("동네 살리기 분석(카드 1-08·유동인구 4-10·4-11 데이터 없음)")

        # 4. 요일·날씨 ---------------------------------------------------------------
        b.step("5/8 요일 패턴과 비 오는 날 효과")
        blocks = []
        wk = compare_share(b, "weekday", [fm, fg], ["전통시장", "일반 가맹점"], per, "탐나는전 사용금액의 요일 구성비",
                           dataset="tamna", measure="amt")
        if wk is not None and "전통시장" in wk.table.index:
            row = wk.table.loc["전통시장"].dropna()
            if len(row):
                f_wl = b.fact(f"전통시장 사용금액이 가장 적은 요일({row.idxmin()}) 비중", row.min(), "%", AggKind.APPLIED, wk.title)
                win["wday"] = (row.idxmin(), f_wl)
        if wk is not None:
            fig = b.figure(wk, wk.title, kind="stacked")
            if fig:
                blocks.append(fig)
        daily = self._q(dataset="tamna", measure="amt", rows="date", cols="mflag", filters=sf, period=per, totals=False)
        rain = pd.Series(dtype=float)
        effects: dict = {}
        if self.cat.has("weather"):
            rdf = self.cat.con.execute(f"SELECT base_date, AVG(daily_rainfall) r FROM {self.cat.tables['weather']} "
                                       "GROUP BY 1").df()
            rain = rdf.set_index("base_date")["r"]
            for g in ("전통시장", "일반 가맹점"):
                if g in daily.table.columns:
                    effects[g] = rain_effect(daily.table[g], rain, hol)
            if effects:
                tab = pd.DataFrame({"비 오는 날 효과(%)": [v[0] for v in effects.values()]}, index=list(effects))
                tab.index.name = "가맹점 구분"
                disp = tab.apply(lambda c: c.map(lambda v: fmt_value(v, "%")))
                wr = make_derived([daily], tab, disp, "비 오는 날(5mm 이상) 탐나는전 사용금액 변화(같은 달·요일 비교)", "변화율", "%",
                                  AggKind.APPLIED, Dim("mflag", "", "가맹점 구분", "category"), None,
                                  method="응용집계: 같은 달·같은 요일 안에서 비 온 날과 안 온 날의 로그 평균 차이(공휴일 제외)")
                from dataolle.analysis.engine import add_actual_sources
                add_actual_sources(wr, self.cat, "weather", (per[0], per[1]))     # 실제 날씨 파일의 달만 적는다
                fig = b.figure(wr, wr.title, kind="hbar")
                if fig:
                    blocks.append(fig)
                em, eg = effects.get("전통시장", (np.nan, 0, 0)), effects.get("일반 가맹점", (np.nan, 0, 0))
                f_rm = b.fact("비 오는 날 전통시장 탐나는전 사용금액 변화", em[0], "%", AggKind.APPLIED, wr.title, change=True)
                f_rg = b.fact("비 오는 날 일반 가맹점 탐나는전 사용금액 변화", eg[0], "%", AggKind.APPLIED, wr.title, change=True)
                f_wd = b.fact("분석에 쓴 비 온 날 수", em[1], "일", AggKind.APPLIED, wr.title)
                blocks.insert(0, Paragraph(b.render(
                    f"날씨 자료와 맞춰 보면, 비가 5mm 이상 온 날({P(f_wd)}) 전통시장 탐나는전 사용금액은 같은 달·같은 요일의 맑은 날보다 "
                    f"{P(f_rm, 'abs')} {P(f_rm, 'dir')}했고, 일반 가맹점은 {P(f_rg, 'abs')} {P(f_rg, 'dir')}했습니다.")))
                if em[0] < -5:
                    win["rain"] = f_rm
                    rx.append(f"비 오는 날 전통시장 사용이 {P(f_rm, 'abs')} 줄어듭니다. 우천일 한정 추가 적립·배달·포장 연계를 검토하세요.")
                    plan.append(f"비 오는 날 전통시장 사용금액은 맑은 날보다 {P(f_rm, 'abs')} 감소하여 우천 대응책이 필요하다.")
                summary.append(f"비 오는 날 전통시장 사용 {P(f_rm)}, 일반 가맹점 {P(f_rg)}.")
        else:
            b.skipped.append("날씨 효과(5-02 일일 날씨 데이터 없음)")
        if blocks:
            sections.append(Section("요일과 날씨", blocks))

        # 5. 정책 전후 효과 ------------------------------------------------------------
        effect = None
        if inp.event:
            b.step("6/8 정책 전후 효과(전통시장 vs 일반 가맹점)")
            effect = self._event(b, daily, inp, sections, rx, plan, summary, P, rain, effects)
        else:
            b.skipped.append("정책 전후 효과(정책 시작일을 넣으면 분석)")

        # 윈윈 처방: 관광객·소상공인·제주도가 함께 이득을 보는 '관광객 탐나는전' 타깃팅 ----------------------
        if win.get("tour") or win.get("rain"):
            rows = []
            if win.get("tour"):
                areas, f_g = win["tour"]
                rows.append(("어디서", f"관광객 소비가 탐나는전 밖에 머무는 지역: {', '.join(areas)}",
                             f"도민 기준보다 전체 소비 기준 지수가 {P(f_g)} 낮음(침투지수와 사각지대)"))
            when, why = [], []
            if win.get("rain"):
                when.append("비 오는 날")
                why.append(f"비 오는 날 전통시장 {P(win['rain'])}(요일과 날씨)")
            if win.get("wday"):
                when.append(f"{win['wday'][0]}")
                why.append(f"전통시장 가장 한산한 요일 비중 {P(win['wday'][1])}(요일과 날씨)")
            if win.get("md"):
                when.append("오일장 장날")
                why.append(f"장날 사용액 평소의 {P(win['md'][0])}(전통시장·업종)")
            if when:
                rows.append(("언제", " · ".join(when), " / ".join(why)))
            rows.append(("누구에게·어디에", "제주를 찾은 관광객(단체관광 인센티브·관광 도민증 연계) → 전통시장·소상공인 가맹점",
                         "탐나는전은 만 14세 이상 누구나 사용 가능"))
            rows.append(("어떻게", "위 지역·시기에 관광객 탐나는전 추가 적립, 한산한 날 쿠폰 집중", "수요가 비는 시간을 채워 같은 예산으로 효과 극대화"))
            rows.append(("어떻게 확인", "시행 후 이 코스의 '정책 전후 효과'로 대상 지역·시기와 나머지를 비교", "주간 이중차분 + 위약 검정"))
            tab = pd.DataFrame([{"항목": a, "처방": b_, "근거": c} for a, b_, c in rows]).set_index("항목")
            tab = tab.apply(lambda col: col.map(b.render))
            sections.append(Section("윈윈 처방 — 관광객·소상공인·제주가 함께 이득", [
                Paragraph("사각지대를 '탐나는전이 안 쓰이는 곳'으로만 보지 않고, 관광객 소비가 탐나는전 밖에 머무는 곳과 상인에게 한산한 시간을 "
                          "겹쳐 처방을 만들었습니다. 같은 예산이라도 수요가 비는 곳·때에 쓰면 관광객은 혜택을, 소상공인은 매출을, "
                          "제주는 외부 소비의 지역 내 순환을 얻습니다."),
                Table(tab, "", None, style="text", widths=[16, 44, 40]),
                Callout("누가 무엇을 얻나", ["관광객: 비 오는 날·한산한 날에도 쓸 곳과 추가 혜택",
                                          "소상공인·전통시장: 매출이 비는 날의 손님",
                                          "제주도: 같은 인센티브 예산으로 더 큰 효과, 외부 소비의 지역 내 순환",
                                          "지역 경제: 대형 업체 대신 골목·시장으로 소비 이동"], "insight")]))
            summary.append("관광객 소비가 탐나는전 밖에 머무는 지역·한산한 시기를 겹친 '윈윈 처방' 절에 제시했습니다.")

        if dn is not None:
            sections += [s for k, s in dn.sections if k == "types"]
            if dn.diag:
                plan.insert(1, dn.diag["plan"])                   # 동네 하나를 골랐으면 보고 문단 둘째 문장에 진단

        # 처방 설계: 적립 예산 배분 시뮬레이션 ----------------------------------------------------------------
        b.step("7/8 처방 설계: 탐나는전 적립 예산 배분 시뮬레이션")
        from dataolle.courses import policy as PL
        pin = inp.policy if isinstance(inp.policy, PL.PolicyInput) else PL.PolicyInput()
        bases = [trend] + ([dm] if cycles else []) + ([daily] if self.cat.has("weather") else []) \
            + (list(dn.bases) if dn is not None else [])
        from dataolle.courses import response as RS
        rr = RS.estimate(self.cat)
        rsec = RS.build_section(b, P, self.cat, rr, bases)
        sections.append(rsec["section"])
        summary += rsec["summary"]
        rx += rsec["rx"]
        plan += rsec["plan"]
        pol = PL.build_section(b, P, self.cat, inp, where, pin, cycles, dn, bases, effect, f_rain=f_rm, f_md=f_md, rr=rr)
        if pol:
            sections.append(pol["section"])
            summary += pol["summary"]
            rx.insert(0, pol["rx"][0])
            plan += pol["plan"]
        else:
            b.skipped.append("처방 설계(탐나는전 사용액이 없음)")

        # 처방·문단 ---------------------------------------------------------------------
        b.step("8/8 요약·처방·보고 문단")
        from dataolle.ai.narrator import narrate
        nar = narrate(gateway, b.facts, summary, rx, " ".join(plan),
                      {"제목": f"{where} 탐나는전 캐시백 설계", "대상": "제주도·행정시 정책 담당자, 주민센터, 상인회",
                       "절": [s.heading for s in sections]})
        sections.append(Section("활용 극대화 처방", [Bullets([b.render(x) for x in nar.insights])]))
        sections.append(Section("기관 보고용 문단", [Callout("현황 보고·보도자료에 옮겨 쓸 수 있는 문단", [b.render(nar.plan)], "note")]))
        if dn is not None and dn.card is not None:
            sections.insert(0, dn.card)
        for i, s in enumerate(sections, 1):                       # 데이터가 있는 절만 모아 번호를 매김
            s.heading = f"{i}. {s.heading}"
        used = ["1-32 탐나는전"] + (["1-08 카드"] if self.cat.has("card") else []) \
            + (["4-10·4-11 유동인구(거주인구)"] if self.cat.has("pop") else []) + (["5-02 일일 날씨"] if self.cat.has("weather") else [])
        sections.insert(0, Section("요약", [Callout("핵심 요약", [b.render(x) for x in nar.summary], "insight"),
                                           Paragraph("탐나는전 거래 자료(일 단위·읍면동·업종·전통시장)는 제주빅데이터센터에서만 제공됩니다. "
                                                     f"이 보고서는 {', '.join(used)} 자료를 동네 단위로 묶어 진단하고, "
                                                     "같은 예산으로 적립을 끝까지 이어 가면서 원도심·전통시장에 몫을 늘리는 배분안(처방 설계)까지 계산했습니다. "
                                                     "모든 수치는 반출 기준(10 미만 비공개·역산 방지) 검수를 통과한 값입니다.")]))
        ai_note = ("요약·처방·문단은 AI가 규칙 기반 초안을 다듬었습니다(숫자는 자리표시자로만 전달, 이 PC에서 채움)." if nar.used_ai
                   else "문장은 규칙 기반 템플릿으로 작성했고 모든 수치는 계산 결과에서 채웠습니다(AI 미사용).")
        title = inp.title or f"{where} 탐나는전 캐시백 설계"
        report = Report(title, "정책 담당자·주민센터·상인회용 · 제주빅데이터센터 탐나는전·카드·유동인구·날씨 데이터 기반", sections,
                        list(b.results),
                        {"대상": where, "분석 기간": f"{fmt_month(per[0])}~{fmt_month(per[1])}(탐나는전)", "데이터": ", ".join(used),
                         "작성일": datetime.now().strftime("%Y-%m-%d"), "작성 도구": "데이터올레"},
                        limitations=[f"다루지 못한 항목: {', '.join(b.skipped)}"] if b.skipped else [],
                        synthetic=self.cat.synthetic, ai_note=ai_note)
        report.extra["plan_text"] = b.render(nar.plan)
        report.extra["plan_label"] = "기관 보고용 문단(현황 보고·보도자료)"
        report.extra["drafts"] = {"summary": list(summary), "rx": list(rx), "plan": " ".join(plan)}   # 자리표시자 초안(에이전트용)
        report.extra["response"] = {"result": rr, "facts": rsec["facts"]}
        if effect:
            report.extra["effect"] = effect
        if pol:
            report.extra["policy"] = pol["pack"]
        if dn is not None:
            report.extra["dongne_types"] = dn.types
            if dn.diag:
                report.extra["diag"] = dn.diag
            if dn.card is not None:
                report.extra["diag_unit"] = dn.card.heading
        self.out.mkdir(parents=True, exist_ok=True)
        stem = f"탐나는전_{where}".replace(" ", "")
        html = to_html(report, self.out / f"{stem}.html")
        pdf = to_pdf(html, self.out / f"{stem}.pdf") if make_pdf else None
        docx = to_docx(report, self.out / f"{stem}.docx") if make_docx else None
        return CourseOutput(report, b.facts, html, pdf, docx, b.figures, b.log, b.skipped, time.time() - t0)

    def _scope_filter(self, inp: TamnaInput) -> dict:
        """분석 범위 → 탐나는전 필터. 읍면동 이름이 탐나는전 자료에 없으면(행정동 이름 등) 동네(행정동) 단위로 바꿔 거른다."""
        if inp.scope == "all" or not inp.value:
            return {}
        if inp.scope in ("city", "hd", "agrp"):
            return {inp.scope: [inp.value]}
        if inp.value in set(self.cat.values("tamna", "emd")):
            return {"emd": [inp.value]}
        from dataolle.courses.dongne import unit_of_name
        u = unit_of_name(inp.value)
        return {"hd": [u]} if u else {"emd": [inp.value]}

    def cat_name(self, ds: str) -> str:
        from dataolle.spec import default_registry
        return default_registry().get(ds).name_ko

    # -- 정책 전후 효과 ---------------------------------------------------------------
    def _event(self, b: _Builder, daily, inp: TamnaInput, sections, rx, plan, summary, P, rain=None, effects=None):
        ev = inp.event if len(inp.event) == 8 else inp.event + "01"
        evd = datetime.strptime(ev, "%Y%m%d").date()
        t = daily.table
        if not {"전통시장", "일반 가맹점"} <= set(t.columns):
            return None
        d = t[["전통시장", "일반 가맹점"]].copy()
        adjusted = False
        if rain is not None and len(rain) and effects:
            wet = rain.reindex(d.index).fillna(0) >= 5                     # 날씨 보정: 비 온 날 값을 맑은 날 수준으로
            for g in ("전통시장", "일반 가맹점"):
                e = effects.get(g, (np.nan,))[0]
                if not np.isnan(e):
                    d.loc[wet, g] = d.loc[wet, g] / (1 + e / 100)
                    adjusted = True
        dt = pd.to_datetime(d.index, format="%Y%m%d")
        d["wk"] = (dt - pd.to_timedelta(dt.weekday, unit="D")).strftime("%Y%m%d")        # 월요일 시작 주
        g = d.groupby("wk")
        cnt = g.count()
        ws = g.sum(min_count=1)[(cnt["전통시장"] == 7) & (cnt["일반 가맹점"] == 7)].dropna()   # 7일이 모두 공개된 주만
        hol = sorted(_holiday_dates())
        hol_wk = pd.Series(1, index=pd.to_datetime([h for h in hol], format="%Y%m%d"))
        hw = (hol_wk.index - pd.to_timedelta(hol_wk.index.weekday, unit="D")).strftime("%Y%m%d")
        counts = pd.Series(1, index=hw).groupby(level=0).sum()
        drop = set(counts.index)                                                  # 공휴일이 낀 주
        for w, n in counts.items():
            if n >= 2:                                                            # 명절 주의 앞 주(대목)도 제외
                drop.add((datetime.strptime(w, "%Y%m%d") - timedelta(days=7)).strftime("%Y%m%d"))
        ws = ws.drop(index=[w for w in ws.index if w in drop])
        evw = (evd - timedelta(days=evd.weekday())).strftime("%Y%m%d")
        if evd.weekday() != 0:
            ws = ws.drop(index=[evw], errors="ignore")                                      # 정책이 걸친 주는 제외
            evw = (evd + timedelta(days=7 - evd.weekday())).strftime("%Y%m%d")
        pre = [i for i in ws.index if i < evw]
        post = [i for i in ws.index if i >= evw]
        if len(pre) < 8 or len(post) < 4:
            sections.append(Section("정책 전후 효과", [Paragraph(f"정책 시작({fmt_month(ev[:6])}) 전후 주 수가 부족해 분석하지 않았습니다"
                                                                 f"(전 {len(pre)}주, 후 {len(post)}주 — 각각 8주·4주 이상 필요).")]))
            return None
        tm, tg = ws["전통시장"], ws["일반 가맹점"]
        eff = (np.exp(did(tm, tg, evw)) - 1) * 100
        pt = pre_trend(tm, tg, evw) if len(pre) >= 3 else float("nan")
        placebo = {}
        for k in range(4, len(pre) - 4):
            fake = pre[k]
            sub = pre
            placebo[fake] = (np.exp(did(tm[sub], tg[sub], fake)) - 1) * 100
        n_pl = len(placebo)
        p = (sum(1 for v in placebo.values() if abs(v) >= abs(eff)) + 1) / (n_pl + 1) if n_pl else float("nan")
        idx = pd.DataFrame({"전통시장": tm / tm[pre].mean() * 100, "일반 가맹점": tg / tg[pre].mean() * 100})
        idx.index.name = "주(월요일 시작)"
        disp = idx.apply(lambda c: c.map(lambda v: fmt_value(v, "지수")))
        ser = make_derived([daily], idx, disp, "주간 탐나는전 사용금액 지수(정책 전 평균=100)", "사용금액 지수", "지수", AggKind.APPLIED,
                           MODELS["tamna"].dim("week"), Dim("series", "", "가맹점 구분", "category"),
                           method="응용집계: 주간 합계 ÷ 정책 전 평균 × 100(7일이 모두 있는 주만)")
        f_e = b.fact(f"{inp.event_name or '정책'} 이후 전통시장 탐나는전 효과(이중차분)", eff, "%", AggKind.APPLIED, ser.title, change=True)
        f_p = b.fact("시간 위약 검정 p값(근사)", p, "p", AggKind.APPLIED, ser.title)
        f_np = b.fact("위약 검정에 쓴 가짜 시작 주 수", n_pl, "개", AggKind.APPLIED, ser.title)
        f_pt = b.fact("정책 전 상대 추세(주별 기울기)", pt, "%", AggKind.APPLIED, ser.title, change=True)
        sig = (not np.isnan(p)) and p <= 0.1
        at_floor = n_pl > 0 and p <= 1 / (n_pl + 1) + 1e-9
        verdict = ("우연으로 보기 어려운 변화입니다." if sig else
                   ("실제 효과가 모든 가짜 효과보다 컸지만, 정책 전 기간이 짧아 가짜 시작 주가 적어서 p값을 더 낮출 수 없습니다"
                    "(기간을 늘리면 판단이 분명해집니다)." if at_floor else "우연한 변동과 구별하기 어렵습니다."))
        blocks = [Paragraph(b.render(
            f"{fmt_month(ev[:6])}{(' ' + inp.event_name) if inp.event_name else ''} 이후 전통시장의 탐나는전 사용금액은 일반 가맹점 대비 "
            f"{P(f_e, 'abs')} {P(f_e, 'dir')}했습니다(주간 이중차분 {P(f_e)}). 정책 전 기간에 가짜 시작 주 {P(f_np)}를 놓고 같은 계산을 반복한 "
            f"시간 위약 검정의 p값은 {P(f_p)}로 " + verdict))]
        fig = b.figure(ser, ser.title, kind="line", event=evw if evw in idx.index else None, event_label="정책 시작", ref_line=100)
        if fig:
            blocks.append(fig)
        blocks.append(Paragraph(b.render(
            f"정책 전 상대 추세는 주당 {P(f_pt)}입니다. 공휴일이 낀 주와 명절 대목 주는 제외했고"
            + (", 비 오는 날 값은 '요일과 날씨' 절의 날씨 효과만큼 보정했습니다." if adjusted else ".")
            + " 같은 시기 다른 변화(지원금 지급 등)가 있었는지 함께 확인하세요.")))
        sections.append(Section("정책 전후 효과", blocks))
        rx.append(f"전후 비교 결과 전통시장 효과 {P(f_e)}(p≈{P(f_p)}): " + ("효과가 확인되어 유지·확대의 근거가 됩니다." if sig else
                  ("방향은 뚜렷하지만 기간을 늘려 다시 확인하세요." if at_floor else "효과가 뚜렷하지 않아 설계 보완이 필요합니다.")))
        summary.append(f"정책 이후 전통시장 효과 {P(f_e)}, 위약 검정 p≈{P(f_p)}.")
        plan.append(f"{fmt_month(ev[:6])} 이후 전통시장 탐나는전 사용금액은 일반 가맹점 대비 {P(f_e, 'abs')} {P(f_e, 'dir')}하였다"
                    f"(이중차분, 시간 위약 검정 p≈{P(f_p)}).")
        return {"did_pct": float(eff), "p": float(p), "placebos": n_pl, "pre_trend": float(pt) if not np.isnan(pt) else None,
                "f_e": f_e, "f_p": f_p, "event": ev[:6], "event_name": inp.event_name}
