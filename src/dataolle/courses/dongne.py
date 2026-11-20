"""코스 4 확장 — 탐나는전으로 동네 살리기.

사람(유동인구 4-10·4-11의 야간 거주인구)과 돈(카드 1-08 도민 소비, 탐나는전 1-32)을 같은 '동네(행정동)' 단위로 맞춰
  · 원도심·신도심 상권: 도민 카드 소비의 상대 흐름(제주 전체 대비), 동네 유형별 탐나는전 침투지수·전통시장 몫
  · 인구 구조와 동네 유형: 고령화율·아동 비율·거주인구 변화 → 동네 지도(산점도)·동네 진단표
  · 학령인구 감소 경보: 0~14세 거주인구 변화(학교별 학생 수 자료가 없어 대리지표로 사용)
  · 동네 유형별 처방: 고령·상권 위축 / 아이 감소 / 고령 / 사각지대 / 관광 의존 / 성장
  · 동네 진단 카드(읍면동 하나를 고른 경우): 이 동네 · 같은 유형 · 제주 전체를 한 표로
을 만든다. 표·그림의 값은 모두 반출 검수를 통과한 집계표에서 계산한 응용집계(비율·변화율·지수)다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from dataolle.analysis.derive import make_derived, ratio_index
from dataolle.analysis.engine import AnalysisResult
from dataolle.analysis.fmt import fmt_month, fmt_value
from dataolle.analysis.josa import josa
from dataolle.analysis.semantic import AREA_GROUPS, BJD_UNIT, HD_UNIT, MODELS, UNIT_GROUP, Dim
from dataolle.report.document import Callout, Paragraph, Section, Table
from dataolle.sdc.engine import MASK_TOKEN, AggKind

UNIT_DIM = Dim("hd", "", "동네(행정동)", "geo")
GROUP_DIM = Dim("agrp", "", "동네 유형", "geo", AREA_GROUPS)
METRIC_DIM = Dim("metric", "", "지표", "category")

TYPE_ORDER = ["고령·상권 위축", "아이 감소", "고령 동네", "사각지대", "관광 의존", "성장", "안정"]
TYPE_RX = {
    "고령·상권 위축": ("전통시장·장날 추가 적립을 이 동네 가맹점에 우선 배정, 주민센터에서 카드형 탐나는전 충전·사용 안내, "
                   "장보기 배달 가맹점 연계", "시행 전후 도민 소비 상대 지수와 탐나는전 사용액(이 코스 전후 효과)"),
    "아이 감소": ("전입·출산 지원금을 탐나는전으로 지급해 동네 안에서 쓰이게 하는 방안 검토, 학원·소아과·문구 등 아이 관련 업종 가맹 확대, "
              "학교별 학생 수 자료와 함께 소규모 학교 지원 대상 검토", "다음 해 같은 달 0~14세 거주인구 변화"),
    "고령 동네": ("고령층 결제 지원(카드형 탐나는전·창구 충전), 의원·약국·식료품 등 생활 업종 가맹 유지", "고령층 생활 업종 탐나는전 사용액 추이"),
    "사각지대": ("가맹점 모집·도민 홍보를 먼저 집중(도민 소비에 비해 탐나는전이 적게 쓰임)", "침투지수 재계산(제주 평균=100)"),
    "관광 의존": ("관광객 탐나는전 안내·인센티브를 이 동네 가맹점과 연계(윈윈 처방)", "관광 성수기·비수기 사용액 비교"),
    "성장": ("새 상가의 가맹 등록과 QR 결제 보급으로 침투 유지", "가맹점 수·침투지수 점검"),
    "안정": ("현재 정책 유지, 분기마다 지표 점검", "분기별 진단 카드"),
}


def unit_of_name(name: str | None) -> str | None:
    """탐나는전·카드의 읍면동 이름(법정동 또는 행정동, '월평동(제주시)' 형식 포함) → 동네(행정동 단위)."""
    if not name:
        return None
    base, city = name, None
    if name.endswith(")") and "(" in name:
        base, city = name[:-1].split("(", 1)
    for c in ([city] if city else []) + ["제주시", "서귀포시"]:
        if (c, base) in BJD_UNIT:
            return BJD_UNIT[(c, base)]
        if (c, base) in HD_UNIT:
            return HD_UNIT[(c, base)]
    return base if base in UNIT_GROUP else None


def _first_last(months: list[str]) -> tuple[str | None, str | None]:
    """인구 비교 시점: 가장 최근 달과, 그와 같은 달(계절)인 가장 이른 해. 없으면 가장 이른 달."""
    if not months:
        return None, None
    last = months[-1]
    same = [m for m in months if m[4:] == last[4:] and m < last]
    first = same[0] if same else (months[0] if months[0] < last else None)
    return first, last


def _chg(a, b) -> float:
    return float((b / a - 1) * 100) if (a and not np.isnan(a) and not np.isnan(b) and a > 0) else float("nan")


def _disp(v, unit: str, masked: bool = False) -> str:
    if masked:
        return MASK_TOKEN
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else fmt_value(v, unit)


@dataclass
class DongneResult:
    sections: list = field(default_factory=list)          # (key, Section)
    summary: list = field(default_factory=list)
    rx: list = field(default_factory=list)
    plan: list = field(default_factory=list)
    card: Section | None = None                            # 동네 진단 카드(읍면동 하나를 고른 경우)
    metrics: pd.DataFrame | None = None
    types: dict = field(default_factory=dict)
    diag: dict = field(default_factory=dict)              # 동네 진단 요약(에이전트·보고 문단용, 자리표시자 판)
    bases: list = field(default_factory=list)             # 유형 판정에 쓴 바탕 결과(처방 설계의 출처용)


class Dongne:
    """동네 단위 지표 계산과 보고서 절 만들기. b는 courses.market._Builder."""

    def __init__(self, b, inp, where: str, P, tamna_period: list[str]):
        self.b, self.inp, self.P = b, inp, P
        self.where = where if inp.scope in ("all", "city") else "제주 전체"     # 비교 기준(동네 하나를 고르면 제주 전체와 비교)
        self.cat = b.cat
        self.per_t = tamna_period
        city = inp.value if inp.scope == "city" and inp.value else None
        self.cf = {"city": [city]} if city else {}                   # 카드·탐나는전·유동인구 모두 'city' 차원이 있음
        self.focus = unit_of_name(inp.value) if inp.scope in ("emd", "hd") else None
        self.focus_group = UNIT_GROUP.get(self.focus) if self.focus else None
        self.out = DongneResult()
        self.m = pd.DataFrame()                                      # 동네별 지표
        self.masked: dict[str, set] = {}                              # 지표 → 가려져 값이 없는 동네
        self.g = pd.DataFrame(index=list(AREA_GROUPS))                # 동네 유형별 지표
        self.j: dict[str, float] = {}                                 # 범위 전체(제주 전체 또는 시) 지표
        self.res: dict[str, AnalysisResult] = {}
        self.pop_first = self.pop_last = None
        self.card_months: list[str] = []

    # ------------------------------------------------------------------ 계산
    def _col(self, name: str, s: pd.Series, base: AnalysisResult | None = None):
        self.m[name] = s.reindex(self.m.index) if len(self.m.index) else s
        if base is not None:
            d = base.display
            col0 = d.columns[0]
            self.masked[name] = {i for i in d.index if (d.loc[i] == MASK_TOKEN).any()} if d.shape[1] > 1 else \
                {i for i in d.index if d.loc[i, col0] == MASK_TOKEN}

    def compute(self):
        b, cf = self.b, self.cf
        units = sorted({u for u in UNIT_GROUP if not cf or self._unit_city(u) == cf["city"][0]})
        self.m = pd.DataFrame(index=units)
        self.m["group"] = [UNIT_GROUP[u] for u in units]
        if self.cat.has("pop"):
            self._compute_pop()
        if self.cat.has("card"):
            self._compute_card()
            if self.cat.has("tamna"):
                self._compute_pen()

    @staticmethod
    def _unit_city(u: str) -> str:
        from dataolle.analysis.semantic import UNIT_CITY
        return UNIT_CITY.get(u, "제주시")

    def _compute_pop(self):
        b, cf = self.b, self.cf
        months = self.cat.months("pop")
        first, last = _first_last(months)
        self.pop_first, self.pop_last = first, last
        age = b.quiet(dataset="pop", measure="resd_night", rows="hd", cols="age3", filters=cf, period=[last, last],
                      transform="share", totals=False)
        age_g = b.quiet(dataset="pop", measure="resd_night", rows="agrp", cols="age3", filters=cf, period=[last, last],
                        transform="share", totals=False)
        age_j = b.quiet(dataset="pop", measure="resd_night", rows="age3", filters=cf, period=[last, last], transform="share")
        self.res.update(age=age, age_g=age_g, age_j=age_j)
        t = age.table
        self._col("aging", t.get("65세 이상", pd.Series(dtype=float)), age)
        self._col("child", t.get("0~14세", pd.Series(dtype=float)), age)
        for col, key in (("65세 이상", "aging"), ("0~14세", "child")):
            if col in age_g.table.columns:
                self.g[key] = age_g.table[col].reindex(self.g.index)
            self.j[key] = float(age_j.table.iloc[:, 0].get(col, np.nan))
        if first:
            mf = {**cf, "month": [first, last]}
            for key, extra in (("pop_chg", {}), ("child_chg", {"age3": ["0~14세"]})):
                ru = b.quiet(dataset="pop", measure="resd_night", rows="hd", cols="month", filters={**mf, **extra}, totals=False)
                rg = b.quiet(dataset="pop", measure="resd_night", rows="agrp", cols="month", filters={**mf, **extra}, totals=False)
                rj = b.quiet(dataset="pop", measure="resd_night", rows="month", filters={**mf, **extra}, totals=False)
                self.res[key] = ru
                self.res[key + "_g"] = rg
                if {first, last} <= set(ru.table.columns):
                    self._col(key, (ru.table[last] / ru.table[first] - 1) * 100, ru)
                if {first, last} <= set(rg.table.columns):
                    self.g[key] = ((rg.table[last] / rg.table[first] - 1) * 100).reindex(self.g.index)
                s = rj.table.iloc[:, 0]
                self.j[key] = _chg(s.get(first, np.nan), s.get(last, np.nan))

    def _compute_card(self):
        b, cf = self.b, self.cf
        months = self.cat.months("card")
        if len(months) < 6:
            return
        self.card_months = months
        f3, l3 = months[:3], months[-3:]
        res_f = {**cf, "tour_type": ["도민"]}
        ru = b.quiet(dataset="card", measure="amt", rows="hd", cols="month", filters=res_f, totals=False)
        rg = b.quiet(dataset="card", measure="amt", rows="month", cols="agrp", filters=res_f, totals=False)
        rj = b.quiet(dataset="card", measure="amt", rows="month", filters=res_f, totals=False)
        js = rj.table.iloc[:, 0]
        jr = js.reindex(l3).sum(min_count=3) / js.reindex(f3).sum(min_count=3)
        tu = ru.table
        ur = tu.reindex(columns=l3).sum(axis=1, min_count=3) / tu.reindex(columns=f3).sum(axis=1, min_count=3)
        self._col("card_rel", (ur / jr - 1) * 100, ru)
        gr = rg.table.reindex(l3).sum(min_count=3) / rg.table.reindex(f3).sum(min_count=3)
        self.g["card_rel"] = ((gr / jr - 1) * 100).reindex(self.g.index)
        # 동네 유형별 월별 상대 지수(제주 전체 대비, 첫 달=100): 계절 변화는 지우고 상대 흐름만
        idx = rg.table.div(rg.table.iloc[0]).div(js / js.iloc[0], axis=0) * 100
        idx = idx[[c for c in AREA_GROUPS if c in idx.columns]]
        idx.index.name = "연월"
        disp = idx.apply(lambda c: c.map(lambda v: _disp(v, "지수")))
        rel = make_derived([rg, rj], idx, disp, "동네 유형별 도민 카드 소비 상대 지수(제주 전체 대비, 첫 달=100)", "상대 지수", "지수",
                           AggKind.APPLIED, MODELS["card"].dim("month"), GROUP_DIM,
                           method="응용집계: (동네 유형 월별 도민 카드 이용금액 ÷ 첫 달) ÷ (제주 전체 같은 비율) × 100")
        self.res.update(card_u=ru, card_rel=rel)
        # 관광객 소비 비중(관광 의존 판정용)
        tour = b.quiet(dataset="card", measure="amt", rows="hd", cols="tour_type", filters=cf, transform="share", totals=False)
        ts = sum(tour.table[c] for c in ("개별관광", "단체관광") if c in tour.table.columns)
        if isinstance(ts, pd.Series):
            self._col("tour_share", ts, tour)

    def _compute_pen(self):
        b, cf = self.b, self.cf
        cm = set(self.card_months or self.cat.months("card"))
        ov = [m for m in self.cat.months("tamna") if m in cm]
        if not ov:
            return
        po = [ov[0], ov[-1]]
        res_f = {**cf, "tour_type": ["도민"]}
        from dataolle.analysis.engine import run_analysis  # noqa: F401  (b.quiet 사용)
        t_all = b.quiet(dataset="tamna", measure="amt", filters=cf, period=po).table.iloc[0, 0]
        c_all = b.quiet(dataset="card", measure="amt", filters=res_f, period=po).table.iloc[0, 0]
        ref = float(t_all) / float(c_all) if c_all else None
        for dim, store in (("hd", "u"), ("agrp", "g")):
            num = b.quiet(dataset="tamna", measure="amt", rows=dim, filters=cf, period=po)
            den = b.quiet(dataset="card", measure="amt", rows=dim, filters=res_f, period=po)
            label = "동네별" if dim == "hd" else "동네 유형별"
            pen = ratio_index(num, den, f"{label} 탐나는전 침투지수(도민 카드 소비 대비, 제주 평균=100)", "침투지수", scale=100.0, ref=ref,
                              method="응용집계: (탐나는전 사용금액 ÷ 도민 카드 이용금액) ÷ (범위 전체 같은 비율) × 100")
            self.res["pen_" + store] = pen
            if dim == "hd":
                self._col("pen", pen.table.iloc[:, 0], pen)
            else:
                self.g["pen"] = pen.table.iloc[:, 0].reindex(self.g.index)
        ms = b.quiet(dataset="tamna", measure="amt", rows="agrp", cols="mflag", filters=cf, period=self.per_t, transform="share",
                     totals=False)
        self.res["ms_g"] = ms
        if "전통시장" in ms.table.columns:
            self.g["market"] = ms.table["전통시장"].reindex(self.g.index)
        self.pen_period = po

    # ------------------------------------------------------------------ 유형 판정
    def classify(self) -> pd.Series:
        m, j = self.m, self.j
        out = {}
        child_q = m["child_chg"].quantile(0.25) if "child_chg" in m and m["child_chg"].notna().sum() >= 4 else np.nan
        for u, r in m.iterrows():
            g = lambda k: r.get(k, np.nan)
            aging_hi = not np.isnan(g("aging")) and not np.isnan(j.get("aging", np.nan)) and g("aging") >= j["aging"] + 5
            spend_dn = not np.isnan(g("card_rel")) and g("card_rel") <= -3
            child_dn = (not np.isnan(g("child_chg")) and not np.isnan(child_q) and g("child_chg") <= child_q
                        and g("child_chg") < j.get("child_chg", 0))
            low_pen = not np.isnan(g("pen")) and g("pen") < 80
            tour_dep = not np.isnan(g("tour_share")) and g("tour_share") >= 50
            growth = (not np.isnan(g("pop_chg")) and not np.isnan(j.get("pop_chg", np.nan)) and g("pop_chg") >= j["pop_chg"] + 5
                      and (np.isnan(g("aging")) or g("aging") < j.get("aging", 100)))
            if aging_hi and spend_dn:
                t = "고령·상권 위축"
            elif child_dn:
                t = "아이 감소"
            elif aging_hi:
                t = "고령 동네"
            elif low_pen:
                t = "사각지대"
            elif tour_dep:
                t = "관광 의존"
            elif growth:
                t = "성장"
            else:
                t = "안정"
            out[u] = t
        return pd.Series(out)

    # ------------------------------------------------------------------ 보고서 절
    def build(self) -> DongneResult:
        self.compute()
        o = self.out
        if self.cat.has("card") and "card_rel" in self.res:
            self._sec_commercial()
        if "age" in self.res:
            self._sec_population()
        if "child_chg" in self.m:
            self._sec_school()
        if len(self.m) and any(c in self.m for c in ("aging", "card_rel", "pen")):
            self.m["type"] = self.classify()
            self._sec_types()
            if self.focus:
                self._diag_card()
        o.metrics = self.m
        o.bases = [r for k, r in self.res.items() if k in ("age", "pop_chg", "child_chg", "card_u", "pen_u")]
        return o

    def _whole(self) -> str:
        return self.where if self.where.endswith("전체") else f"{self.where} 전체"

    def _fact(self, label, value, unit, kind=AggKind.APPLIED, source="", change=False):
        return self.b.fact(label, value, unit, kind, source, change)

    def _sec_commercial(self):
        b, P, g = self.b, self.P, self.g
        rel = self.res["card_rel"]
        blocks = []
        last = rel.table.index[-1]
        f = {}
        for grp in ("원도심", "신도심", "읍면"):
            if grp in rel.table.columns and not np.isnan(rel.table.loc[last, grp]):
                f[grp] = self._fact(f"{grp} 도민 카드 소비 상대 변화(제주 전체 대비, {fmt_month(rel.table.index[0])}→{fmt_month(last)})",
                                    rel.table.loc[last, grp] - 100, "%", source=rel.title, change=True)
        if "원도심" in f and "신도심" in f:
            blocks.append(Paragraph(b.render(
                f"계절에 따른 오르내림을 지우려고 도민 카드 소비를 제주 전체 흐름과 견줬습니다(첫 달=100). "
                f"{fmt_month(rel.table.index[0])}부터 {fmt_month(last)}까지 원도심은 제주 전체보다 {P(f['원도심'], 'abs')} "
                f"{P(f['원도심'], 'dir')}했고, 신도심은 {P(f['신도심'], 'abs')} {P(f['신도심'], 'dir')}했습니다.")))
            self.out.summary.append(f"원도심 도민 소비는 제주 전체 흐름 대비 {P(f['원도심'], 'abs')} {P(f['원도심'], 'dir')}"
                                    f"(신도심 {P(f['신도심'], 'abs')} {P(f['신도심'], 'dir')}).")
        fig = b.figure(rel, rel.title, kind="line", ref_line=100, unit_note="제주 전체 대비 상대 지수(첫 달=100)")
        if fig:
            blocks.append(fig)
        pen_g = self.res.get("pen_g")
        if pen_g is not None and "pen" in g:
            fp = {grp: self._fact(f"{grp} 탐나는전 침투지수(도민 카드 소비 대비, 제주 평균=100)", g.loc[grp, "pen"], "지수",
                                  source=pen_g.title) for grp in AREA_GROUPS if not np.isnan(g.loc[grp, "pen"])}
            if "원도심" in fp:
                old = g.loc["원도심", "pen"]
                txt = (f"탐나는전 침투지수(도민 카드 소비 대비 탐나는전 사용액, 제주 평균=100)는 원도심 {P(fp['원도심'])}"
                       + "".join(f", {k} {P(v)}" for k, v in fp.items() if k != "원도심") + "입니다. ")
                if old >= 110:
                    txt += ("원도심에서는 탐나는전이 도민 소비를 떠받치는 몫이 커서, 캐시백 축소·중단의 충격도 원도심 상인에게 더 크게 갑니다. "
                            "예산이 줄 때에도 원도심·전통시장 몫은 지키는 설계가 필요합니다.")
                    self.out.rx.append(f"원도심 침투지수 {P(fp['원도심'])}: 캐시백 예산이 줄 때에도 원도심·전통시장 적립은 우선 유지하세요"
                                       "(중단 충격이 가장 큰 곳).")
                    self.out.plan.append(f"원도심의 탐나는전 침투지수는 {P(fp['원도심'])}(제주 평균=100)로 탐나는전 의존도가 높아 "
                                         "캐시백 변동의 영향을 크게 받는다.")
                elif old < 90:
                    txt += "원도심은 탐나는전 침투가 낮아 가맹점 모집과 도민 홍보로 넓힐 여지가 큽니다."
                    self.out.rx.append(f"원도심 침투지수 {P(fp['원도심'])}: 원도심 가맹점 모집·홍보를 먼저 하세요.")
                blocks.append(Paragraph(b.render(txt)))
            fig = b.figure(pen_g, pen_g.title, kind="hbar", ref_line=100, ref_label="제주 평균 100", unit_note="제주 평균=100")
            if fig:
                blocks.append(fig)
        if "market" in g and not np.isnan(g.loc["원도심", "market"]):
            fm = self._fact("원도심 탐나는전 사용금액 중 전통시장 비중", g.loc["원도심", "market"], "%", source=self.res["ms_g"].title)
            b.keep(self.res["ms_g"])
            blocks.append(Paragraph(b.render(f"원도심 탐나는전 사용액 가운데 전통시장 몫은 {P(fm)}입니다. "
                                             "원도심 상권 대책과 전통시장 대책을 따로 떼지 말고 함께 설계해야 하는 이유입니다.")))
        blocks.append(Paragraph("동네 유형: 원도심(제주시 일도1·이도1·삼도1·삼도2·용담1·용담2·건입동, 서귀포시 송산·정방·중앙·천지동), "
                                "신도심(연동·노형동·아라동·이도2동·외도동·삼양동·동홍동·대륜동), 그 밖의 동, 읍면. "
                                "제주연구원(2024) 인구감소위험 분석의 구도심 동과 서귀포 원도심 상권 활성화 사업 대상 동을 바탕으로 한 근사 구분입니다. "
                                "법정동 자료(카드·탐나는전)는 근사 대응표로 행정동에 붙였습니다."))
        self.out.sections.append(("commercial", Section("원도심·신도심 상권", blocks)))

    def _sec_population(self):
        b, P, g, j = self.b, self.P, self.g, self.j
        age_g, age_j = self.res["age_g"], self.res["age_j"]
        last = self.pop_last
        blocks = []
        fa = {grp: self._fact(f"{grp} 고령화율(65세 이상 비중, {fmt_month(last)})", g.loc[grp, "aging"], "%", source=age_g.title)
              for grp in AREA_GROUPS if "aging" in g and not np.isnan(g.loc[grp, "aging"])}
        fj = self._fact(f"{self.where} 고령화율(65세 이상 비중, {fmt_month(last)})", j.get("aging"), "%", source=age_j.title)
        fc = self._fact(f"{self.where} 0~14세 비중({fmt_month(last)})", j.get("child"), "%", source=age_j.title)
        if fa:
            blocks.append(Paragraph(b.render(
                f"{fmt_month(last)} 새벽(02~05시) 거주인구로 본 65세 이상 비중(고령화율)은 "
                + ", ".join(f"{k} {P(v)}" for k, v in fa.items()) + f"입니다({self.where} {P(fj)}, 0~14세 {P(fc)}).")))
        # 연령 구성: 동네 유형 + 범위 전체
        t = age_g.table.copy()
        tj = age_j.table.iloc[:, 0]
        t.loc[self.where] = [tj.get(c, np.nan) for c in t.columns]
        d = age_g.display.copy()
        d.loc[self.where] = [_disp(tj.get(c, np.nan), "%") for c in d.columns]
        comp = make_derived([age_g, age_j], t, d, f"동네 유형별 연령 구성({fmt_month(last)}, 야간 거주인구 기준)", "구성비", "%",
                            AggKind.APPLIED, GROUP_DIM, MODELS["pop"].dim("age3"),
                            method="응용집계: 연령 3구분별 야간 거주인구 ÷ 전체 × 100")
        comp.spec = comp.spec.model_copy(update={"transform": "share"})
        fig = b.figure(comp, comp.title, kind="stacked")
        if fig:
            blocks.append(fig)
        if "pop_chg" in self.m and "aging" in self.m:
            sc = self.m[["aging", "pop_chg"]].dropna()
            masked = self.masked.get("aging", set()) | self.masked.get("pop_chg", set())
            sc = sc.drop(index=[u for u in sc.index if u in masked], errors="ignore")
            if len(sc) >= 5:
                disp = pd.DataFrame({"고령화율": [fmt_value(v, "%") for v in sc["aging"]],
                                     "거주인구 변화율": [fmt_value(v, "%") for v in sc["pop_chg"]]}, index=sc.index)
                tab = sc.rename(columns={"aging": "고령화율", "pop_chg": "거주인구 변화율"})
                tab.index.name = disp.index.name = "동네(행정동)"
                first = self.pop_first
                smap = make_derived([self.res["age"], self.res["pop_chg"]], tab, disp,
                                    f"동네 지도: 고령화율({fmt_month(last)})과 거주인구 변화({fmt_month(first)}→{fmt_month(last)})",
                                    "고령화율", "%", AggKind.APPLIED, UNIT_DIM, Dim("metric", "", "지표", "category"),
                                    method="응용집계: 동네별 65세 이상 비중, 같은 달 기준 거주인구 변화율(야간 거주인구)")
                fig = b.figure(smap, smap.title, kind="scatter", groups={u: UNIT_GROUP.get(u) for u in sc.index},
                               ref_x=j.get("aging"), ref_y=0.0, x_label="고령화율(65세 이상 비중)",
                               y_label="거주인구 변화율", y_unit="%", unit_note="% (가로: 고령화율, 세로: 거주인구 변화율)")
                if fig:
                    blocks.append(fig)
                risk = sc[(sc["aging"] > j.get("aging", 0)) & (sc["pop_chg"] < 0)]
                risk = risk.sort_values("aging", ascending=False)
                if len(risk):
                    names = ", ".join(str(u) for u in risk.index[:5])
                    fp = self._fact(f"{self.where} 거주인구 변화율({fmt_month(first)}→{fmt_month(last)})", j.get("pop_chg"), "%",
                                    source=smap.title, change=True)
                    fn = self._fact(f"고령화율이 {self.where}보다 높고 인구가 줄어든 동네 수", len(risk), "곳", source=smap.title)
                    blocks.append(Paragraph(b.render(
                        f"같은 달 기준으로 {fmt_month(first)}부터 {fmt_month(last)}까지 {self.where} 거주인구는 {P(fp, 'abs')} {P(fp, 'dir')}했습니다. "
                        f"고령화율은 {self.where}보다 높은데 인구가 실제로 줄어든 동네(그림의 오른쪽 아래 칸)는 {P(fn)}(으)로 {names} 순입니다. "
                        "이 동네들은 생활 소비가 동네 밖으로 새기 쉬워 탐나는전 생활 업종 지원이 먼저 필요합니다.")))
                    self.out.summary.append(f"고령화율이 높고 인구가 줄어든 동네 {P(fn)}: {names}.")
        blocks.append(Paragraph("거주인구는 통신 자료로 추정한 새벽(02~05시) 거주인구 평균이며 주민등록 인구와 다릅니다. "
                                "0~9세는 휴대전화 미보유가 많아 통신사 추정 방식의 영향을 크게 받습니다."))
        self.out.sections.append(("population", Section("인구 구조와 동네 유형", blocks)))

    def _sec_school(self):
        b, P, j = self.b, self.P, self.j
        first, last = self.pop_first, self.pop_last
        s = self.m["child_chg"].dropna()
        s = s.drop(index=[u for u in s.index if u in self.masked.get("child_chg", set())], errors="ignore")
        if len(s) < 3:
            return
        low = s.sort_values().head(10)
        disp = pd.DataFrame({"0~14세 변화율": [fmt_value(v, "%") for v in low]}, index=low.index)
        tab = low.to_frame("0~14세 변화율")
        tab.index.name = disp.index.name = "동네(행정동)"
        r = make_derived([self.res["child_chg"]], tab, disp,
                         f"0~14세 거주인구가 가장 많이 줄어든 동네({fmt_month(first)}→{fmt_month(last)})", "0~14세 변화율", "%",
                         AggKind.APPLIED, UNIT_DIM, None,
                         method="응용집계: 같은 달 기준 0~14세 야간 거주인구 변화율(동네별)")
        fj = self._fact(f"{self.where} 0~14세 거주인구 변화율({fmt_month(first)}→{fmt_month(last)})", j.get("child_chg"), "%",
                        source=r.title, change=True)
        fw = self._fact(f"0~14세 감소 1위 {low.index[0]} 변화율", low.iloc[0], "%", source=r.title, change=True)
        blocks = [Paragraph(b.render(
            f"같은 달 기준 {fmt_month(first)}부터 {fmt_month(last)}까지 0~14세 거주인구는 {self._whole()} {P(fj, 'abs')} "
            f"{P(fj, 'dir')}했고, {josa(str(low.index[0]), '은')} {P(fw, 'abs')} {P(fw, 'dir')}해 변화 폭이 가장 컸습니다."))]
        fig = b.figure(r, r.title, kind="hbar", ref_line=j.get("child_chg"), ref_label=f"{self.where}")
        if fig:
            blocks.append(fig)
        g = self.g
        if "child_chg" in g and not np.isnan(g.loc["원도심", "child_chg"]):
            fo = self._fact("원도심 0~14세 거주인구 변화율", g.loc["원도심", "child_chg"], "%", source=r.title, change=True)
            blocks.append(Paragraph(b.render(f"동네 유형으로 보면 원도심의 0~14세 거주인구가 {P(fo, 'abs')} {P(fo, 'dir')}했습니다.")))
        blocks.append(Callout("이 지표를 쓰는 법", [
            "학교별 학생 수는 제주빅데이터센터·제주데이터허브에 없습니다(허브의 중·고교 현황은 2020년에서 끝남). "
            "그래서 동네의 0~14세 거주인구 변화를 '학령인구 감소 경보'로 씁니다.",
            "감소 폭이 큰 동네는 교육청 학생 수·학급 수 자료와 맞춰 소규모 학교 지원·통폐합 논의의 우선 확인 대상으로 삼으세요.",
            "학교 급식 잔반 자료도 공개 데이터에 없어 이번 분석에서 다루지 않았습니다(데이터 확보 시 확장).",
        ], "note"))
        names = ", ".join(str(u) for u in low.index[:3])
        self.out.summary.append(f"0~14세 거주인구 {self.where} {P(fj, 'abs')} {P(fj, 'dir')}, 감소 폭이 큰 동네: {names}.")
        self.out.plan.append(f"{fmt_month(first)} 대비 {fmt_month(last)} 0~14세 거주인구는 {P(fj, 'abs')} {P(fj, 'dir')}하였으며, "
                             f"감소 폭이 큰 동네({names})는 학령인구 감소에 대비한 지원 검토가 필요하다.")
        self.out.rx.append(f"학령인구 감소 경보 동네({names}): 전입·출산 지원금의 탐나는전 지급과 아이 관련 업종 가맹 확대를 검토하세요.")
        self.out.sections.append(("school", Section("학령인구 감소 경보", blocks)))

    def _sec_types(self):
        b, P, m, j = self.b, self.P, self.m, self.j
        cols = [("group", "유형 구분", None), ("aging", "고령화율", "%"), ("child_chg", "0~14세 변화", "%"),
                ("pop_chg", "인구 변화", "%"), ("card_rel", "도민 소비(상대)", "%"), ("pen", "탐나는전 침투", "지수")]
        cols = [c for c in cols if c[0] in m.columns]
        rank = {t: i for i, t in enumerate(TYPE_ORDER)}
        by, asc = (["_r", "aging"], [True, False]) if "aging" in m else (["_r"], [True])
        sel = m.assign(_r=m["type"].map(rank)).sort_values(by, ascending=asc)
        sel = sel[sel["type"] != "안정"].head(14)
        if sel.empty:
            return
        num = sel[[c for c, _, u in cols if u]].astype(float)
        num.columns = [lab for c, lab, u in cols if u]
        disp = pd.DataFrame(index=sel.index)
        disp["동네 진단"] = sel["type"]
        for c, lab, u in cols:
            disp[lab] = sel[c] if u is None else [_disp(sel.loc[i, c], u, i in self.masked.get(c, set())) for i in sel.index]
        disp.index.name = num.index.name = "동네(행정동)"
        bases = [r for k, r in self.res.items() if k in ("age", "pop_chg", "child_chg", "card_u", "pen_u")]
        dt = make_derived(bases, num, disp, "동네 진단표(먼저 챙겨야 할 동네)", "진단 지표", "%", AggKind.APPLIED, UNIT_DIM, METRIC_DIM,
                          method="응용집계: 동네별 고령화율·0~14세 변화율·거주인구 변화율·도민 카드 소비 상대 변화·탐나는전 침투지수를 모아 유형 판정")
        b.keep(dt)
        b.ntab += 1
        blocks = [Paragraph("사람(거주인구)과 돈(도민 카드 소비·탐나는전)을 동네마다 한 줄로 모아, 먼저 챙겨야 할 동네와 처방을 고릅니다. "
                            f"기준: 고령화율이 {self.where}보다 5%p 이상 높으면 '고령', 도민 소비가 {self.where} 흐름보다 3% 넘게 뒤처지면 '상권 위축', "
                            "0~14세 감소 폭이 하위 4분의 1이면 '아이 감소', 침투지수 80 미만이면 '사각지대', 관광객 소비가 절반 이상이면 '관광 의존'."),
                  Table(disp, f"표 {b.ntab}. 동네 진단표(먼저 챙겨야 할 동네)", dt, style="text")]
        # 유형별 처방
        rows = []
        for t in TYPE_ORDER[:-1]:
            sub = m[m["type"] == t]
            us = list(sub.sort_values("aging", ascending=False).index) if "aging" in sub else list(sub.index)
            if not us:
                continue
            rx, check = TYPE_RX[t]
            rows.append({"동네 유형": t, "해당 동네": ", ".join(us[:5]) + (f" 외 {len(us) - 5}곳" if len(us) > 5 else ""),
                         "탐나는전 처방": rx, "확인 방법": check})
            self.out.types[t] = us
        if rows:
            tab = pd.DataFrame(rows).set_index("동네 유형")
            blocks.append(Table(tab, "동네 유형별 탐나는전 처방(검토 제안)", None, style="text", widths=[14, 24, 40, 22]))
            n_types = self._fact("처방이 필요한 동네 유형 수", len(rows), "가지", source=dt.title)
            top = rows[0]
            blocks.append(Paragraph(b.render(
                f"동네를 {P(n_types)} 유형으로 나눴습니다. 가장 먼저 챙길 유형은 '{top['동네 유형']}'({top['해당 동네']})입니다. "
                "같은 탐나는전 예산이라도 동네 유형에 맞춰 적립·홍보·가맹 확대를 나눠 쓰면 효과가 커집니다.")))
            lead_units = ", ".join(self.out.types[top["동네 유형"]][:3])
            self.out.rx.append(f"동네 유형별 처방: 먼저 '{top['동네 유형']}' 동네({lead_units} 등)에 {TYPE_RX[top['동네 유형']][0]}.")
        self.out.sections.append(("types", Section("동네 유형별 처방", blocks)))

    def _diag_card(self):
        """읍면동 하나를 고른 경우: 이 동네 · 같은 유형 · 범위 전체를 한 표로."""
        b, P, m, g, j = self.b, self.P, self.m, self.g, self.j
        u, grp = self.focus, self.focus_group
        if u not in m.index:
            return
        items = [("aging", "고령화율(65세 이상 비중)", "%"), ("child", "0~14세 비중", "%"), ("pop_chg", "거주인구 변화", "%"),
                 ("child_chg", "0~14세 거주인구 변화", "%"), ("card_rel", "도민 카드 소비(제주 흐름 대비)", "%"),
                 ("pen", "탐나는전 침투지수(제주 평균=100)", "지수")]
        items = [it for it in items if it[0] in m.columns]
        if not items:
            return
        num = pd.DataFrame({"이 동네": [m.loc[u, k] for k, _, _ in items],
                            f"{grp} 평균": [g.loc[grp, k] if k in g.columns else np.nan for k, _, _ in items],
                            self.where: [100.0 if k == "pen" else (0.0 if k == "card_rel" else j.get(k, np.nan)) for k, _, _ in items]},
                           index=[lab for _, lab, _ in items])
        disp = pd.DataFrame({c: [_disp(num.loc[lab, c], unit, c == "이 동네" and u in self.masked.get(k, set()))
                                 for k, lab, unit in items] for c in num.columns}, index=num.index)
        num.index.name = disp.index.name = "지표"
        bases = [r for k, r in self.res.items() if k in ("age", "age_g", "age_j", "pop_chg", "pop_chg_g", "child_chg", "child_chg_g",
                                                         "card_u", "card_rel", "pen_u", "pen_g")]
        dc = make_derived(bases, num, disp, f"{u} 동네 진단 카드", "진단 지표", "%", AggKind.APPLIED, METRIC_DIM,
                          Dim("cmp", "", "비교", "category"),
                          method="응용집계: 동네·같은 유형·범위 전체의 고령화율·아동 비율·인구 변화·도민 소비 상대 변화·침투지수")
        b.keep(dc)
        b.ntab += 1
        t = m.loc[u, "type"]
        rx, check = TYPE_RX[t]
        ft = []
        for k, lab, unit in items[:3]:
            v = m.loc[u, k]
            if not np.isnan(v) and u not in self.masked.get(k, set()):
                ft.append((lab, self._fact(f"{u} {lab}", v, unit, source=dc.title, change=(k.endswith("_chg")))))
        lead = ", ".join(f"{lab} {P(fid)}" for lab, fid in ft)
        src = self.inp.value
        intro = (f"{josa(src, '이')} 속한 행정동 {u}(동네 유형: {grp})" if src and src.split("(")[0] != u
                 else f"{u}(동네 유형: {grp})")
        self.out.card = Section("동네 진단 카드", [
            Paragraph(b.render(f"{intro}의 주요 지표는 {lead}입니다.")) if ft else Paragraph(intro),
            Table(disp, f"표 {b.ntab}. {u} 동네 진단 카드", dc, style="text"),
            Callout(f"진단: {t}", [f"처방: {rx}", f"확인 방법: {check}"], "insight"),
        ])
        self.out.summary.insert(0, f"{u} 동네 진단: '{t}' — {rx.split(',')[0]}.")
        lead_draft = ", ".join(f"{lab} {P(fid)}" for lab, fid in ft)
        ro = josa(t, "으로")[len(t):]
        self.out.diag = {"unit": u, "group": grp, "type": t, "rx": rx, "check": check, "lead": lead_draft,
                         "plan": (f"{u}({grp})은(는) " + (f"{lead_draft}(으)로 " if lead_draft else "")
                                  + f"'{t}' 유형에 해당한다(우선 처방: {rx.split(',')[0]}).").replace("은(는)", josa(u, "은")[len(u):])}
