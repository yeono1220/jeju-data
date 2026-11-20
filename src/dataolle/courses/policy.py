"""처방 설계 — 탐나는전 적립(캐시백) 예산 배분 시뮬레이터.

문제: 탐나는전 적립은 한 해 예산이 정해져 있는데, 적립률을 높게 시작하면 한 해 중간에 멈춘다.
2026년에는 캐시백 예산 425억 원 중 2월(20% 적립)에만 154억 원이 쓰여 7월 13일 적립이 중단됐다(제이누리, 2026.7.13).
2022년 4월과 2023년 5월·9월에도 같은 일이 있었다.

이 모듈은 센터의 탐나는전 거래(1-32: 날짜·읍면동·전통시장)로 '기준 1년'을 만들고, 적립 방식별로 다음을 계산한다.
  · 월별 적립 비용, 예산이 바닥나는 날(소진 시점), 예산이 버티는 개월 수
  · 적립 혜택의 행선지: 원도심·전통시장·동네 살리기 대상 동네·읍면·사각지대·질문한 동네
적립 방식(모두 더하기 방식, 상한 20%):
  A  현행형            기본 10%, 설·추석이 있는 달 20%(2026년 방식)
  B  균등 유지형        예산이 기간 끝까지 가는 한 가지 적립률(자동 계산)
  C  동네 살리기 집중형  기본률(자동 계산)에 가산: 원도심·살리기 대상 동네 +3%p, 전통시장 +5%p,
                       오일장 장날 시장 +3%p, 비 오는 날 시장 +3%p
  D  사각지대 보강형     기본률(자동 계산)에 가산: 도민 소비 대비 탐나는전이 덜 쓰이는 동네(침투지수 80 미만) +5%p
  E  쏠림 방지형        C안 + 건당 적립 상한(1만 원) + 학원·교육 업종은 기본률만(선결제·고액 결제로 혜택이 쏠리는 것을 막음)
  C′ 효과 반영 조정안    '정책 전후 효과' 결과에 따라 C안을 고친 다음 배분
                       효과 확인 → 유지(늘어난 시장 사용만큼 비용 반영), 불분명 → 장날·비 오는 날로 좁힘, 역효과 → 시장 가산 중단
사용 반응: courses/response.py가 자료에서 추정한 반응도 b(적립률 1%p당 log 사용액, 앞당긴 소비 제외)를 쓴다.
  기준 1년의 사용액은 먼저 '적립률 10%였다면'으로 되돌리고(앞당긴 소비 업종은 걷어 냄), 적립 방식별로 exp(b × (적립률 − 10))을 곱한다.
  추정할 수 없으면 반응 0(사용 패턴 그대로)으로 계산하고 그렇다고 밝힌다. 민감도로 반응 0과 신뢰구간 위쪽 값을 함께 계산한다.
입력은 결제 10건 미만 칸을 뺀 집계이고, 결과(합계·비중)는 응용집계다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from dataolle.analysis.semantic import MODELS, UNIT_CITY, UNIT_GROUP

# 설·추석이 있는 달(양력)
FESTIVE = {2024: (2, 9), 2025: (1, 10), 2026: (2, 9), 2027: (2, 9), 2028: (1, 10), 2029: (2, 9), 2030: (2, 9)}
REAL_BUDGET_2026 = 425e8            # 2026년 캐시백 예산(제이누리 2026.7.13)
REAL_USAGE_2025 = 6800e8            # 2025년 탐나는전 사용액(뉴스서울 2026.3.15)
DEFAULT_RATIO = REAL_BUDGET_2026 / REAL_USAGE_2025     # 사용액 대비 예산 6.25%
CAP = 20.0                          # 2026년 최대 적립률
CURRENT_RATE = 10.0                 # 기준 1년의 적립률(가정)
TXCAP = 10_000                      # E안: 결제 1건당 적립 상한(원)
REVIVE_TYPES = ("고령·상권 위축", "고령 동네", "아이 감소")
C_BONUS = {"dong": 3.0, "mkt": 5.0, "mday": 3.0, "rain": 3.0}
D_BONUS = {"blind": 5.0}
WET_MM = 5.0
ORDER = ["A", "B", "C", "D", "E", "C′"]
NAMES = {"A": "현행형", "B": "균등 유지형", "C": "동네 살리기 집중형", "D": "사각지대 보강형", "E": "쏠림 방지형",
         "C′": "효과 반영 조정안"}
GOALS = {"revive": "C", "blind": "D", "even": "B", "current": "A", "fair": "E"}
GOAL_WORDS = [(("쏠림", "역진", "고소득", "선결제", "고액 결제", "학원", "몰아"), "fair"),
              (("사각지대", "보급", "가맹점 확대", "덜 쓰이"), "blind"),
              (("고르게", "균등", "똑같이", "형평", "일률"), "even"),
              (("지금처럼", "현행대로", "지금 방식 그대로"), "current"),
              (("원도심", "구도심", "전통시장", "오일장", "장날", "동네 살리", "동네를 살리", "살릴"), "revive")]
_BUDGET = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(조|억)\s*(?:원)?")


def festive_months(year: int) -> tuple[int, ...]:
    return FESTIVE.get(year, (2, 9))


@dataclass
class PolicyInput:
    budget: float | None = None        # 원. 없으면 2026년 예산 비율(사용액의 6.25%)을 이 자료에 적용
    months: int = 12                   # 적립을 이어 갈 기간(1~12개월)
    start: int = 1                     # 시작 월
    year: int | None = None            # 계획 연도(없으면 자료 마지막 해 + 1)
    goal: str = "auto"                 # auto | revive | blind | even | current | fair
    response: float | None = None      # 적립률 1%p당 사용액 변화(%). None이면 자료에서 추정(추정 못 하면 0)


def parse_policy(q: str, effect_question: bool = False) -> dict:
    """질문에서 예산·기간·시작 월·연도·목표를 읽는다(이 PC 안에서만 — 숫자는 AI로 보내지 않음)."""
    out: dict = {}
    m = _BUDGET.search(q)
    if m and any(w in q for w in ("예산", "캐시백", "적립", "재원", "인센티브", "할인")):
        out["budget"] = float(m.group(1).replace(",", "")) * (1e12 if m.group(2) == "조" else 1e8)
    if not effect_question:
        m = re.search(r"(?<!\d)(1[0-2]|[1-9])\s*월\s*부터", q)
        if m:
            out["start"] = int(m.group(1))
        # 계획 연도는 '2027년 예산', '2027년 적립', '2027년부터'처럼 계획을 가리킬 때만(‘2025년 2분기 15% 때’ 같은 과거 언급은 제외)
        m = re.search(r"(20\d{2})\s*년\s*(?:도\s*)?(?:예산|적립|캐시백|부터|계획|1월부터)", q)
        if m:
            out["year"] = int(m.group(1))
    m = re.search(r"(?<!\d)(1[0-2]|[1-9])\s*개월", q)
    if m:
        out["months"] = int(m.group(1))
    elif "연말까지" in q and out.get("start"):
        out["months"] = 13 - out["start"]
    elif any(w in q for w in ("1년", "일 년", "한 해", "연중", "일년")):
        out["months"] = 12
    for words, g in GOAL_WORDS:
        if any(w in q for w in words):
            out["goal"] = g
            break
    return out


# ---------------------------------------------------------------------------------------------- 자료 준비
def load_segments(cat) -> tuple[pd.DataFrame, float]:
    """날짜 × 동네 × 가맹점 구분 × 시장별 사용액. 결제 10건 미만 칸은 0으로(반출 기준과 같은 처리). 반환: (표, 뺀 사용액 비율 %)."""
    T, d = cat.tables["tamna"], MODELS["tamna"].dims
    sql = (f"SELECT base_ymd AS d, {d['hd'].expr} AS hd, {d['mflag'].expr} AS mflag, {d['market'].expr} AS market, "
           f"{d['biz'].expr} AS biz, SUM(amount) AS amt, SUM(cnt) AS cnt FROM {T} GROUP BY ALL")
    seg = cat.con.execute(sql).df()
    seg["amt"] = seg["amt"].astype(float)
    seg["cnt"] = seg["cnt"].astype(float)
    seg["biz"] = seg["biz"].fillna("(업종 정보 없음)")
    seg["d"] = seg["d"].astype(str)
    small = seg["cnt"] < 10
    # 업종까지 나누면 10건 미만 칸이 늘어나므로, 뺀 비율은 업종을 합친 칸 기준으로 따로 계산해 보여 준다
    total = float(seg["amt"].sum()) or 1.0
    dropped = float(seg.loc[small, "amt"].sum()) / total * 100
    seg.loc[small, "amt"] = 0.0
    seg["hd"] = seg["hd"].fillna("(동네 정보 없음)")
    return seg, dropped


def rain_days(cat) -> pd.Series:
    if not cat.has("weather"):
        return pd.Series(dtype=float)
    df = cat.con.execute(f"SELECT base_date, AVG(daily_rainfall) r FROM {cat.tables['weather']} GROUP BY 1").df()
    return df.set_index("base_date")["r"]


@dataclass
class Frame:
    """기준 1년을 계획 기간 순서로 늘어놓은 표와 꼬리표."""
    df: pd.DataFrame
    year: int
    start: int
    months: int
    picks: dict                 # 계획 월 → (자료 월, 추정 여부)
    usage_window: float         # 계획 기간 사용액
    usage_year: float           # 기준 1년 사용액
    has_rain: bool
    has_cycles: bool
    has_types: bool
    has_blind: bool
    pop_old: float = 0.0        # 고령 동네(고령화율 중앙값 이상) 거주인구
    pop_young: float = 0.0      # 그 밖의 동네 거주인구

    def month_label(self, o: int) -> str:
        return f"{(self.start - 1 + o) % 12 + 1}월"

    @property
    def est_months(self) -> list[int]:
        return [c for c, (_, est) in self.picks.items() if est]


def prepare(seg: pd.DataFrame, *, cycles=None, rain: pd.Series | None = None, types: pd.Series | None = None,
            pen: pd.Series | None = None, year: int | None = None, start: int = 1, months: int = 12,
            focus_units: set | None = None, aging: pd.Series | None = None, residents: pd.Series | None = None) -> Frame:
    f = seg.copy()
    f["ym"] = f["d"].str[:6]
    avail = sorted(f["ym"].unique())
    if not avail:
        raise ValueError("탐나는전 자료가 비어 있습니다.")
    year = year or int(avail[-1][:4]) + 1
    start = max(1, min(12, int(start or 1)))
    months = max(1, min(12, int(months or 12)))
    by_cal: dict[int, list[str]] = {}
    for m in avail:
        by_cal.setdefault(int(m[4:]), []).append(m)
    picks = {}
    for c in range(1, 13):
        if c in by_cal:
            picks[c] = (by_cal[c][-1], False)
        else:                                   # 자료에 없는 달은 가장 가까운 달의 모양으로(추정)
            best = min(by_cal, key=lambda k: (min((k - c) % 12, (c - k) % 12), k))
            picks[c] = (by_cal[best][-1], True)
    parts = []
    for c in range(1, 13):
        ym, est = picks[c]
        p = f[f["ym"] == ym].copy()
        p["pm"], p["o"], p["est"] = c, (c - start) % 12, est
        parts.append(p)
    fr = pd.concat(parts, ignore_index=True)
    usage_year = float(fr["amt"].sum())
    fr = fr[fr["o"] < months].reset_index(drop=True)
    fr["dd"] = fr["d"].str[6:8].astype(int)
    fr["grp"] = fr["hd"].map(UNIT_GROUP).fillna("그 밖의 동")
    fr["type"] = fr["hd"].map(types) if types is not None and len(types) else None
    fr["mkt"] = fr["mflag"].eq("전통시장")
    cyc = {m: int(c[0]) for m, c, _r in (cycles or [])}
    res = fr["market"].map(cyc)
    fr["mday"] = fr["mkt"] & res.notna() & (fr["dd"] % 5 == res.fillna(-1))
    has_rain = rain is not None and len(rain) > 0
    fr["rain"] = (fr["d"].map(rain).fillna(0) >= WET_MM) if has_rain else False
    fr["fest"] = fr["pm"].isin(festive_months(year))
    fr["rev"] = fr["grp"].eq("원도심") | fr["type"].isin(REVIVE_TYPES)
    has_blind = pen is not None and len(pen.dropna()) > 0
    fr["blind"] = (fr["hd"].map(pen) < 80).fillna(False) if has_blind else False
    fr["focus"] = fr["hd"].isin(focus_units) if focus_units else False
    fr["key"] = fr["o"] * 32 + fr["dd"]
    from dataolle.courses.response import is_edu
    fr["edu"] = fr["biz"].map(is_edu) if "biz" in fr.columns else False
    if "cnt" not in fr.columns:
        fr["cnt"] = 0.0
    pop_old = pop_young = 0.0
    fr["old"] = False
    if aging is not None and residents is not None and len(aging.dropna()) >= 4 and len(residents.dropna()):
        med = float(aging.dropna().median())
        old_units = set(aging[aging >= med].index)
        fr["old"] = fr["hd"].isin(old_units)
        r = residents.dropna()
        pop_old = float(r[r.index.isin(old_units)].sum())
        pop_young = float(r[~r.index.isin(old_units)].sum())
    for c in ("mday", "rain", "fest", "rev", "blind", "focus", "mkt", "edu", "old"):
        fr[c] = fr[c].astype(bool)
    return Frame(fr, year, start, months, picks, float(fr["amt"].sum()), usage_year, has_rain, bool(cyc),
                 types is not None and len(types) > 0, has_blind, pop_old, pop_young)


# ---------------------------------------------------------------------------------------------- 시뮬레이션
@dataclass
class Scenario:
    key: str
    name: str
    params: dict                       # base, fest, dong, mkt, mday, rain, blind
    solved: bool = False
    status: str = ""                   # over: 가산만으로 예산 초과 | under: 상한까지 올려도 남음
    lasting: float = 0.0               # 예산이 버티는 개월 수
    exhaust: tuple[int, int] | None = None     # (월, 일)
    exhaust_x: float | None = None     # 그림용 위치(개월)
    need: float = 0.0                  # 기간 내내 적립하려면 드는 돈
    spent: float = 0.0
    monthly_paid: list = field(default_factory=list)
    monthly_need: list = field(default_factory=list)
    shares: dict = field(default_factory=dict)
    focus_amt: float | None = None
    focus_share: float | None = None
    unpaid: float = 0.0                # 예산 소진 뒤 적립 없이 지나간 사용액 비중(%)
    eff: dict = field(default_factory=dict)    # 실효 적립률(%) — 전체·전통시장·원도심·질문한 동네
    use_chg: float = 0.0               # 사용 반응을 넣었을 때 탐나는전 사용액 변화(%) — 반응 0이면 0
    percap_ratio: float | None = None  # 고령 동네 1인당 적립액 ÷ 그 밖의 동네 1인당 적립액(1보다 작으면 젊은 동네 쏠림)
    note: str = ""

    @property
    def peak(self) -> float:
        p = self.params
        if p.get("fest"):
            return min(CAP, p["base"] + p["fest"])
        extra = p.get("dong", 0) + p.get("mkt", 0) + p.get("mday", 0) + p.get("rain", 0)
        extra = max(extra, p.get("blind", 0))
        return min(CAP, p["base"] + extra)

    @property
    def label(self) -> str:
        return f"{self.key}안 {self.name}"

    def exhaust_text(self, start: int = 1, year: int | None = None) -> str:
        if not self.exhaust:
            return ""
        m, d = self.exhaust
        return f"{m}월 {d}일께"

    def rate_of(self, *, rev=False, mkt=False, mday=False, rain=False, blind=False, fest=False) -> float:
        p = self.params
        r = p["base"] + p.get("fest", 0) * fest + p.get("dong", 0) * rev + p.get("mkt", 0) * mkt \
            + p.get("mday", 0) * (mday and mkt) + p.get("rain", 0) * (rain and mkt) + p.get("blind", 0) * blind
        return float(min(CAP, r))


def rule_text(p: dict, year: int) -> str:
    """적립 방식을 사람이 읽는 문장으로(숫자 포함 — 이 PC 화면·보고서용)."""
    fmt = lambda v: f"{v:.0f}%" if abs(v - round(v)) < 1e-9 else f"{v:.1f}%"
    if p.get("fest"):
        fm = "·".join(str(m) for m in festive_months(year))
        return f"기본 {fmt(p['base'])}, 설·추석이 있는 달({fm}월) {fmt(p['base'] + p['fest'])}"
    parts = []
    if p.get("dong"):
        parts.append(f"원도심·살리기 대상 동네 +{p['dong']:g}%p")
    if p.get("mkt"):
        parts.append(f"전통시장 +{p['mkt']:g}%p")
    if p.get("mday"):
        parts.append(f"오일장 장날 시장 +{p['mday']:g}%p")
    if p.get("rain"):
        parts.append(f"비 오는 날 시장 +{p['rain']:g}%p")
    if p.get("blind"):
        parts.append(f"사각지대 동네 +{p['blind']:g}%p")
    tail = ""
    if p.get("txcap"):
        tail += f", 건당 적립 {p['txcap'] / 10000:g}만 원 상한"
    if p.get("edu_base_only"):
        tail += ", 학원·교육 업종은 기본률만"
    if not parts:
        return f"모든 결제 {fmt(p['base'])}" + tail
    return f"기본 {fmt(p['base'])} + " + ", ".join(parts) + f"(상한 {CAP:.0f}%)" + tail


class Simulator:
    def __init__(self, frame: Frame, budget: float, response: float = 0.0):
        """response: 반응도 b(적립률 1%p당 log 사용액). 사용액 = 기준 사용액(10% 기준으로 되돌린 값) × exp(b × (적립률 − 10))."""
        self.F, self.B, self.resp = frame, float(budget), float(response)
        df = frame.df
        self.amt = df["amt"].to_numpy(dtype=float)
        self.cnt = df["cnt"].to_numpy(dtype=float) if "cnt" in df.columns else np.zeros(len(df))
        self.flags = {c: df[c].to_numpy(dtype=bool) for c in ("fest", "rev", "mkt", "mday", "rain", "blind", "focus", "edu", "old")
                      if c in df.columns}
        self.key = df["key"].to_numpy()
        self.o = df["o"].to_numpy()
        self.dd = df["dd"].to_numpy()
        self.pm = df["pm"].to_numpy()
        self.grp = df["grp"].to_numpy()

    def rates(self, p: dict) -> np.ndarray:
        g = self.flags
        bonus = (p.get("dong", 0) * g["rev"] + p.get("mkt", 0) * g["mkt"] + p.get("mday", 0) * g["mday"]
                 + p.get("rain", 0) * (g["mkt"] & g["rain"]) + p.get("blind", 0) * g["blind"])
        if p.get("edu_base_only") and "edu" in g:
            bonus = np.where(g["edu"], 0.0, bonus)
        r = p["base"] + p.get("fest", 0) * g["fest"] + bonus
        return np.minimum(CAP, np.maximum(0.0, r))

    def _use(self, rate: np.ndarray, amt: np.ndarray | None = None) -> np.ndarray:
        a = self.amt if amt is None else amt
        if not self.resp:
            return a
        return a * np.exp(self.resp * (rate - CURRENT_RATE))

    def _cost(self, p: dict, rate: np.ndarray, use: np.ndarray, amt: np.ndarray | None = None) -> np.ndarray:
        cost = rate / 100 * use
        if p.get("txcap"):
            a = self.amt if amt is None else amt
            n = np.where(a > 0, self.cnt * use / np.where(a > 0, a, 1.0), 0.0)      # 사용이 늘면 결제 건수도 같은 비율로
            cost = np.where(n > 0, np.minimum(cost, p["txcap"] * n), cost)
        return cost

    def need(self, p: dict, amt=None) -> float:
        rate = self.rates(p)
        return float(np.sum(self._cost(p, rate, self._use(rate, amt), amt)))

    def solve(self, p: dict, amt=None) -> tuple[float, str]:
        """예산을 기간 끝까지 쓰는 기본 적립률(이분법)."""
        lo, hi = 0.0, CAP
        if self.need({**p, "base": 0.0}, amt) >= self.B:
            return 0.0, "over"
        if self.need({**p, "base": CAP}, amt) <= self.B:
            return CAP, "under"
        for _ in range(60):
            mid = (lo + hi) / 2
            if self.need({**p, "base": mid}, amt) < self.B:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2, ""

    def run(self, key: str, params: dict, solve: bool = False, amt=None, name: str | None = None) -> Scenario:
        p = dict(params)
        status = ""
        if solve:
            p["base"], status = self.solve(p, amt)
        sc = Scenario(key, name or NAMES.get(key, key), p, solve, status)
        rate = self.rates(p)
        use = self._use(rate, amt)
        cost = self._cost(p, rate, use, amt)
        order = np.argsort(self.key, kind="stable")
        daily = pd.Series(cost).groupby(self.key).sum().sort_index()
        cum = daily.cumsum()
        sc.need = float(daily.sum())
        months = self.F.months
        need_m = pd.Series(cost).groupby(self.o).sum().reindex(range(months), fill_value=0.0)
        sc.monthly_need = [float(v) for v in need_m]
        paid = cost.copy()
        unpaid_use = 0.0
        if sc.need > self.B * (1 + 1e-9):
            k = int(cum.index[np.argmax(cum.to_numpy() > self.B)])
            prev = float(cum[cum.index < k].iloc[-1]) if (cum.index < k).any() else 0.0
            frac = (self.B - prev) / float(daily[k]) if daily[k] > 0 else 0.0
            paid = np.where(self.key < k, cost, np.where(self.key == k, cost * frac, 0.0))
            after = np.where(self.key > k, use, np.where(self.key == k, use * (1 - frac), 0.0))
            unpaid_use = float(after.sum()) / float(use.sum()) * 100 if use.sum() else 0.0
            o_k, d_k = k // 32, k % 32
            m_k = (self.F.start - 1 + o_k) % 12 + 1
            ndays = int(self.dd[self.o == o_k].max())
            sc.exhaust = (m_k, d_k)
            sc.lasting = o_k + (d_k - 1 + frac) / ndays
            sc.exhaust_x = sc.lasting
        else:
            sc.lasting = float(months)
        del order
        sc.unpaid = unpaid_use
        sc.use_chg = (float(use.sum()) / float(self.amt.sum()) - 1) * 100 if self.amt.sum() else 0.0
        sc.spent = float(paid.sum())
        sc.monthly_paid = [float(v) for v in pd.Series(paid).groupby(self.o).sum().reindex(range(months), fill_value=0.0)]
        tot = sc.spent or 1.0
        g = self.flags
        masks = {"원도심": self.grp == "원도심", "전통시장": g["mkt"], "살리기 대상 동네": g["rev"], "읍면": self.grp == "읍면",
                 "사각지대 동네": g["blind"], "학원·교육": g.get("edu", np.zeros(len(paid), dtype=bool))}
        sc.shares = {k: float(paid[m].sum()) / tot * 100 for k, m in masks.items()}
        if self.F.pop_old > 0 and self.F.pop_young > 0 and "old" in g:
            po, py = float(paid[g["old"]].sum()) / self.F.pop_old, float(paid[~g["old"]].sum()) / self.F.pop_young
            sc.percap_ratio = po / py if py > 0 else None
        if g["focus"].any():
            sc.focus_amt = float(paid[g["focus"]].sum())
            sc.focus_share = sc.focus_amt / tot * 100

        def eff(m):
            w = use[m]
            return float(np.sum(rate[m] * w) / np.sum(w)) if w.sum() else float("nan")
        sc.eff = {"전체": eff(np.ones_like(self.amt, dtype=bool)), "전통시장": eff(g["mkt"]),
                  "원도심": eff(self.grp == "원도심"), **({"질문한 동네": eff(g["focus"])} if g["focus"].any() else {})}
        if status == "over":
            sc.note = "가산만으로 예산을 넘어 기본 적립률 0%에서도 기간 중 소진"
        elif status == "under":
            sc.note = f"상한 {CAP:.0f}%까지 올려도 예산이 남음"
        return sc


def design(frame: Frame, budget: float, response: float = 0.0, effect: dict | None = None) -> dict[str, Scenario]:
    """적립 방식 A·B·C·D(+C′)를 같은 예산으로 계산."""
    sim = Simulator(frame, budget, response)
    out = {"A": sim.run("A", {"base": CURRENT_RATE, "fest": CURRENT_RATE}),
           "B": sim.run("B", {"base": 0.0}, solve=True),
           "C": sim.run("C", {"base": 0.0, **C_BONUS}, solve=True)}
    if frame.has_blind:
        out["D"] = sim.run("D", {"base": 0.0, **D_BONUS}, solve=True)
    out["E"] = sim.run("E", {"base": 0.0, **C_BONUS, "txcap": TXCAP, "edu_base_only": True}, solve=True)
    if effect and effect.get("p") is not None and not np.isnan(effect.get("did_pct", np.nan)):
        dec = effect_decision(effect)
        if dec == "keep":
            amt = sim.amt * (1 + effect["did_pct"] / 100 * sim.flags["mkt"])
            out["C′"] = sim.run("C′", {"base": 0.0, **C_BONUS}, solve=True, amt=amt)
        elif dec == "narrow":
            out["C′"] = sim.run("C′", {"base": 0.0, "dong": C_BONUS["dong"], "mday": 5.0, "rain": 5.0, "mkt": 0.0}, solve=True)
        else:
            out["C′"] = sim.run("C′", {"base": 0.0, "dong": C_BONUS["dong"]}, solve=True)
        out["C′"].note = DECISION_TEXT[dec]
    return out


DECISION_TEXT = {"keep": "효과 확인 → 시장 가산 유지, 늘어난 시장 사용만큼 비용을 반영해 기본률 재계산",
                 "narrow": "효과 불분명 → 시장 상시 가산을 없애고 장날·비 오는 날 가산(+5%p)으로 좁힘",
                 "stop": "역효과 → 시장 가산 중단, 동네 가산만 유지"}


def effect_decision(effect: dict) -> str:
    p, e = effect.get("p"), effect.get("did_pct")
    if p is not None and p <= 0.1 and e is not None and e > 0:
        return "keep"
    if p is not None and p <= 0.1 and e is not None and e < 0:
        return "stop"
    return "narrow"


def recommend(scen: dict[str, Scenario], goal: str = "auto", focus_type: str | None = None) -> tuple[str, str]:
    """추천 적립 방식과 이유(숫자 없는 문장)."""
    if "C′" in scen:
        return "C′", "정책 전후 효과를 확인한 결과를 반영한 다음 배분안입니다"
    want = GOALS.get(goal)
    if want and want in scen:
        why = {"A": "질문에서 지금 방식을 유지하길 원했습니다",
               "B": "질문에서 모든 가맹점에 고르게 적립하길 원했습니다",
               "C": "질문의 우선순위가 원도심·전통시장 살리기입니다",
               "D": "질문의 우선순위가 탐나는전 사각지대 보강입니다",
               "E": "질문의 우선순위가 혜택 쏠림(선결제·고액 결제) 막기입니다"}[want]
        return want, why
    if focus_type == "사각지대" and "D" in scen:
        return "D", "질문한 동네가 '사각지대'(도민 소비에 비해 탐나는전이 덜 쓰이는 동네)입니다"
    return "C", "예산을 기간 끝까지 유지하면서 원도심·전통시장·살리기 대상 동네의 몫을 가장 많이 늘립니다"


def focus_units_of(scope: str, value: str | None) -> tuple[set | None, str | None]:
    """분석 범위 → 질문한 동네(행정동 집합)와 이름."""
    if scope in ("emd", "hd") and value:
        from dataolle.courses.dongne import unit_of_name
        u = unit_of_name(value) or value
        return {u}, u
    if scope == "city" and value:
        return {u for u, c in UNIT_CITY.items() if c == value}, value
    if scope == "agrp" and value:
        return {u for u, g in UNIT_GROUP.items() if g == value}, value
    return None, None


def market_table(frame: Frame, cycles, focus_units: set | None, top: int = 3) -> list[dict]:
    """안내문·홍보 문구에 쓸 시장: 질문한 동네의 시장(없으면 장날이 있는 큰 시장)."""
    df = frame.df[frame.df["mkt"]]
    if df.empty:
        return []
    cyc = {m: c for m, c, _r in (cycles or [])}
    g = df.groupby("market").agg(amt=("amt", "sum"), hd=("hd", lambda s: s.mode().iloc[0]),
                                 rev=("rev", "max"), blind=("blind", "max"))
    g = g[g.index != "시장 외"]
    in_focus = g[g["hd"].isin(focus_units)] if focus_units else g.iloc[0:0]
    pick = in_focus.sort_values("amt", ascending=False)
    if pick.empty:
        pick = g.assign(c=[m in cyc for m in g.index]).sort_values(["c", "amt"], ascending=False).drop(columns="c")
    rows = []
    for m, r in pick.head(top).iterrows():
        c = cyc.get(m)
        days = None
        if c is not None:
            a, b2 = sorted(x if x else 10 for x in c)
            days = f"{a}·{b2}일"
        rows.append({"name": str(m), "hd": r["hd"], "rev": bool(r["rev"]), "blind": bool(r["blind"]), "days": days,
                     "focus": bool(focus_units and r["hd"] in focus_units)})
    return rows


def effect_question(year: int, start: int, focus: str | None) -> str:
    where = f"{focus} 전통시장 기준으로" if focus else "전통시장 기준으로"
    return f"{year}년 {start}월부터 바꾼 탐나는전 적립 방식의 효과를 {where} 확인해 줘"


NEEDED_DATA = ["1-32 탐나는전 이용 현황: 시행 전 12개월과 시행 후 3개월 이상(최소 시행 전 8주·후 4주)",
               "5-02 일일 날씨: 같은 기간(비 오는 날 보정)",
               "1-08 카드 이용 현황(선택): 같은 기간(도민 소비 흐름과 비교)"]


# ---------------------------------------------------------------------------------------------- 보고서 절
_BONUS_LABEL = (("dong", "원도심·살리기 대상 동네"), ("mkt", "전통시장"), ("mday", "오일장 장날 시장"), ("rain", "비 오는 날 시장"),
                ("blind", "사각지대 동네"))


def _figure(b, r, caption: str, fn, **kw):
    from dataolle.report.document import Figure
    b.keep(r)
    b.nfig += 1
    path = b.out / "figures" / f"그림{b.nfig:02d}.png"
    p = fn(r, path, **kw)
    if p is None:
        b.nfig -= 1
        return None
    b.figures.append(p)
    return Figure(p, f"그림 {b.nfig}. {caption}", r)


def _add_cites(r, cat, names: list[tuple[str, str]], period: tuple[str, str]):
    """파생 결과에 바탕 자료(날씨·카드·인구)를 출처로 더한다."""
    from dataolle.analysis.engine import add_actual_sources
    for _ds, key in names:
        if not cat.has(key):
            continue
        # 날씨는 기준 1년의 날짜에만 쓰므로 그 기간과 겹치는 달, 카드·인구(동네 유형·침투지수)는 불러온 달 전부.
        # 기간은 분석 기간이 아니라 실제 파일의 달로 적는다(반출 신청서의 원본 데이터 기간과 같아야 함).
        add_actual_sources(r, cat, key, period if key == "weather" else None)


def residents_by_unit(cat) -> pd.Series | None:
    """동네(행정동)별 밤 시간 거주인구(가장 최근 달). 1인당 적립액 계산용."""
    if not cat.has("pop"):
        return None
    m = MODELS["pop"]
    T = cat.tables["pop"]
    last = cat.months("pop")[-1]
    try:
        df = cat.con.execute(f"SELECT {m.dims['hd'].expr} AS hd, {m.measures['resd_night'].expr} AS v FROM {T} "
                             f"WHERE {m.dims['month'].expr} = ? GROUP BY 1", [last]).df()
    except Exception:
        return None
    return df.dropna().set_index("hd")["v"].astype(float)


def build_section(b, P, cat, inp, where: str, pin: PolicyInput, cycles, dn, bases: list, effect: dict | None = None,
                  f_rain: str | None = None, f_md: str | None = None, rr=None) -> dict | None:
    """탐나는전 보고서의 '처방 설계' 절. 반환: {section, summary, rx, plan, pack} — 문장 초안은 자리표시자만 씀."""
    from dataolle.analysis.derive import make_derived
    from dataolle.analysis.fmt import fmt_month, fmt_value
    from dataolle.analysis.semantic import Dim
    from dataolle.report.charts import render_burn_chart, render_pair_bars
    from dataolle.report.document import Callout, Paragraph, Section, Table
    from dataolle.sdc.engine import AggKind

    from dataolle.courses.response import estimate, normalize_amounts
    seg, dropped = load_segments(cat)
    if seg.empty or seg["amt"].sum() <= 0:
        return None
    resp = rr if rr is not None else estimate(cat)
    # 사용 반응: 질문·설정 값 > 자료 추정 > 0
    if pin.response is not None:
        b_use, b_src = float(np.log(1 + pin.response / 100)), "set"
    elif resp is not None and resp.ok:
        b_use, b_src = resp.b, "est"
        seg = normalize_amounts(seg, resp, CURRENT_RATE)          # 기준 1년을 '10%였다면'으로(앞당긴 소비 걷어 냄)
    else:
        b_use, b_src = 0.0, "none"
    rain = rain_days(cat)
    types = pen = aging = None
    if dn is not None and dn.metrics is not None and len(dn.metrics):
        m = dn.metrics
        types = m["type"] if "type" in m.columns else None
        pen = m["pen"] if "pen" in m.columns else None
        aging = m["aging"] if "aging" in m.columns else None
    focus_units, focus = focus_units_of(inp.scope, inp.value)
    # 고령 동네 판정·1인당 분모: 공식 주민등록 인구(공개 실데이터)가 있으면 그것을, 없으면 센터 거주인구 추정을 쓴다
    from dataolle.courses.public_pop import for_units
    residents = residents_by_unit(cat)
    pub = for_units(sorted(seg["hd"].dropna().unique()))
    pop_src = "센터 유동인구(야간 거주 추정)"
    if pub is not None:
        aging, residents, pop_src = pub.table["aging"], pub.table["total"], pub.label
    fr = prepare(seg, cycles=cycles, rain=rain, types=types, pen=pen, year=pin.year, start=pin.start,
                 months=pin.months, focus_units=focus_units, aging=aging, residents=residents)
    user_budget = bool(pin.budget)
    B = float(pin.budget) if pin.budget else DEFAULT_RATIO * fr.usage_window
    scen = design(fr, B, b_use, effect)
    sens = design(fr, B, 0.0 if b_use else float(np.log(1.05)))          # 반응 0(또는 반응이 없다고 놓았으면 5%/%p)
    sens_hi = design(fr, B, resp.b_hi) if b_src == "est" and resp.b_hi == resp.b_hi else None
    focus_type = dn.diag.get("type") if (dn is not None and dn.diag) else None
    rec, why = recommend(scen, pin.goal, focus_type)
    keys = [k for k in ORDER if k in scen]
    R = scen[rec]
    used_months = sorted({ym for ym, _ in fr.picks.values()})
    period = (used_months[0], used_months[-1])
    ptxt = f"{fmt_month(period[0])}~{fmt_month(period[1])}"
    title = "적립 방식별 예산 유지와 혜택 배분(시뮬레이션)"
    F = lambda label, v, unit, change=False: b.fact(label, v, unit, AggKind.APPLIED, title, change)

    # 사실(자리표시자) ----------------------------------------------------------------------------------
    f_budget = F("적립 예산", B, "원")
    f_months = F("적립을 이어 갈 목표 기간", fr.months, "개월")
    f_ratio = F("적립 예산 ÷ 계획 기간 탐나는전 사용액", B / fr.usage_window * 100, "%")
    f_cap = F("적립률 상한", CAP, "%")
    fbv: dict = {}

    def fb(key, v):
        if (key, v) not in fbv:
            lab = dict(_BONUS_LABEL)[key]
            fbv[(key, v)] = F(f"가산 적립률: {lab}", v, "%p")
        return fbv[(key, v)]

    sf: dict[str, dict] = {}
    for k in keys:
        s = scen[k]
        d = {"base": F(f"{k}안 기본 적립률", s.params["base"], "%"),
             "peak": F(f"{k}안 최대 적립률", s.peak, "%"),
             "last": F(f"{k}안 예산 유지 기간", s.lasting, "개월"),
             "mkt": F(f"{k}안 적립 혜택 중 전통시장 몫", s.shares["전통시장"], "%"),
             "old": F(f"{k}안 적립 혜택 중 원도심 몫", s.shares["원도심"], "%"),
             "rev": F(f"{k}안 적립 혜택 중 살리기 대상 동네 몫", s.shares["살리기 대상 동네"], "%"),
             "rural": F(f"{k}안 적립 혜택 중 읍면 몫", s.shares["읍면"], "%"),
             "eff_mkt": F(f"{k}안 전통시장 실효 적립률", s.eff["전통시장"], "%"),
             "edu": F(f"{k}안 적립 혜택 중 학원·교육 업종 몫", s.shares["학원·교육"], "%")}
        if s.percap_ratio is not None:
            d["percap"] = F(f"{k}안 고령 동네 1인당 적립액 ÷ 그 밖의 동네 1인당 적립액", s.percap_ratio, "배")
        if s.exhaust:
            d["exm"] = F(f"{k}안 예산 소진 월", s.exhaust[0], "월")
            d["unpaid"] = F(f"{k}안 예산 소진 뒤 적립 없이 쓰인 사용액 비중", s.unpaid, "%")
        if fr.has_blind:
            d["blind"] = F(f"{k}안 적립 혜택 중 사각지대 동네 몫", s.shares["사각지대 동네"], "%")
        if s.focus_amt is not None:
            d["focus_amt"] = F(f"{k}안 {focus}에 돌아가는 적립액", s.focus_amt, "원")
            d["focus_sh"] = F(f"{k}안 적립 혜택 중 {focus} 몫", s.focus_share, "%")
        sf[k] = d

    f_txcap = F("E안 결제 한 건당 적립 상한", TXCAP, "원")

    def rule_ph(k: str) -> str:
        p = scen[k].params
        if p.get("fest"):
            return f"기본 {P(sf[k]['base'])}, 설·추석이 있는 달 {P(sf[k]['peak'])}"
        parts = [f"{lab} {P(fb(key, p[key]))}" for key, lab in _BONUS_LABEL if p.get(key)]
        tail = (f", 건당 적립 상한 {P(f_txcap)}" if p.get("txcap") else "") + (", 학원·교육 업종은 기본률만" if p.get("edu_base_only") else "")
        if not parts:
            return f"모든 결제 {P(sf[k]['base'])}" + tail
        return f"기본 {P(sf[k]['base'])}에 " + ", ".join(parts) + f" 가산(상한 {P(f_cap)})" + tail

    # 대안 비교표 ---------------------------------------------------------------------------------------
    show_rev = fr.has_types
    show_pc = all(scen[k].percap_ratio is not None for k in keys)
    cols = ["원도심", "전통시장"] + (["대상 동네"] if show_rev else []) + (["사각지대"] if fr.has_blind else []) \
        + ["학원·교육"] + ([focus] if focus else [])
    rows, num_rows = [], []
    for k in keys:
        s = scen[k]
        vals = [s.shares["원도심"], s.shares["전통시장"]] + ([s.shares["살리기 대상 동네"]] if show_rev else []) \
            + ([s.shares["사각지대 동네"]] if fr.has_blind else []) + [s.shares["학원·교육"]] + ([s.focus_share] if focus else [])
        rows.append({"적립 방식": s.label + (" ★추천" if k == rec else ""), "적립률": rule_text(s.params, fr.year),
                     "예산 유지": (f"{fr.months}개월 끝까지" if not s.exhaust
                                else f"{fmt_value(s.lasting, '개월')}({s.exhaust_text().replace('께', '')} 소진)"),
                     **{c: fmt_value(v, "%") for c, v in zip(cols, vals)},
                     **({"고령 동네 1인당": fmt_value(s.percap_ratio, "배")} if show_pc else {})})
        num_rows.append({"기본 적립률": s.params["base"], "최대 적립률": s.peak, "예산 유지(개월)": s.lasting,
                         **dict(zip(cols, vals)), **({"고령 동네 1인당": s.percap_ratio} if show_pc else {})})
    disp = pd.DataFrame(rows).set_index("적립 방식")
    num = pd.DataFrame(num_rows, index=disp.index)
    sdim = Dim("scenario", "", "적립 방식", "category")
    mdim = Dim("metric", "", "지표", "category")
    method = ("응용집계: 기준 1년의 날짜·동네·가맹점 구분·시장별 탐나는전 사용금액(결제 10건 미만 칸 제외)에 적립 방식별 적립률을 곱해 "
              "날짜 순서로 더하고, 예산을 넘는 날 이후는 적립 없음으로 계산")
    cmp_r = make_derived(bases, num, disp, title, "시뮬레이션 지표", "%", AggKind.APPLIED, sdim, mdim, method=method)
    cmp_r.period = period                                   # 기간 표기는 기준 1년(유형 판정에 쓴 인구 자료 기간이 아님)
    cites = ([("5-02", "weather")] if fr.has_rain else []) + ([("1-08", "card")] if fr.has_blind or fr.has_types else []) \
        + ([("4-11", "pop")] if fr.has_types else [])
    _add_cites(cmp_r, cat, cites, period)
    b.keep(cmp_r)
    b.ntab += 1
    ncol = len(cols) + (1 if show_pc else 0)
    widths = [12, 25, 12] + [max(6, int(51 / max(1, ncol)))] * ncol
    blocks = [
        Paragraph("탐나는전 적립(캐시백)은 한 해 예산이 정해져 있어, 적립률을 높게 시작하면 한 해 중간에 멈춥니다. 2026년에는 캐시백 예산 "
                  "425억 원 가운데 2월(적립률 20%)에만 154억 원이 쓰여 7월 13일 적립이 중단됐고, 2022년 4월과 2023년 5월·9월에도 "
                  "예산 문제로 할인·적립이 멈췄습니다(제이누리, 2026.7.13). 멈췄다 다시 시작하는 적립은 상인과 손님 모두에게 혼란을 줍니다."),
        Paragraph(b.render(
            f"이 PC에서 기준 1년({ptxt})의 날짜·동네·전통시장별 탐나는전 사용액에 적립 방식 {len(keys)}가지를 적용해, 예산 {P(f_budget)}(으)로 "
            f"{fr.year}년 {fr.start}월부터 {P(f_months)} 동안 적립을 이어 갈 수 있는지와 적립 혜택이 어디로 가는지 계산했습니다. "
            + (f"예산은 질문에서 받은 값입니다(계획 기간 사용액의 {P(f_ratio)})." if user_budget else
               "예산은 2026년 비율(425억 원 ÷ 2025년 사용액 6,800억 원 = 6.25%)을 이 자료의 사용액에 적용한 값입니다. "
               "실제 예산을 질문에 넣으면(예: '예산 400억 원으로 1년') 그 값으로 다시 계산합니다."))),
        Table(disp, f"표 {b.ntab}. 적립 방식별 예산 유지와 적립 혜택의 행선지(같은 예산). 오른쪽 칸은 지급된 적립액 중 몫"
              + (" — 대상 동네: 원도심과 고령·상권 위축·고령·아이 감소 동네, 사각지대: 침투지수 80 미만 동네" if show_rev else "")
              + (f". 고령 동네 1인당: 고령화율이 중앙값 이상인 동네 주민 1인당 적립액 ÷ 그 밖의 동네(1보다 작으면 젊은 동네로 쏠림). "
                 f"인구 기준: {pop_src}" if show_pc else ""),
              cmp_r, style="text", widths=widths),
    ]

    # 핵심 문장 ------------------------------------------------------------------------------------------
    A = scen["A"]
    s1 = []
    if A.exhaust:
        s1.append(f"지금 방식(A안: {rule_ph('A')})은 {A.exhaust_text()} 예산이 바닥나 {P(sf['A']['last'])}만 적립되고, "
                  f"그 뒤 사용액의 {P(sf['A']['unpaid'])}는 적립 없이 지나갑니다.")
    else:
        s1.append(f"이 예산이면 지금 방식(A안)도 {P(f_months)}을 버팁니다.")
    s1.append(f"같은 예산으로 기간 내내 고르게 적립하려면 모든 결제를 {P(sf['B']['base'])}(으)로 낮춰야 합니다(B안).")
    s1.append(f"C안은 기본 적립률을 {P(sf['C']['base'])}(으)로 두고 원도심·살리기 대상 동네, 전통시장, 장날, 비 오는 날에 가산해 "
              f"{P(sf['C']['last'])} 동안 예산을 유지하면서 전통시장 몫을 {P(sf['A']['mkt'])}에서 {P(sf['C']['mkt'])}(으)로, "
              f"원도심 몫을 {P(sf['A']['old'])}에서 {P(sf['C']['old'])}(으)로 늘립니다.")
    if "D" in scen:
        s1.append(f"D안은 사각지대 동네 몫을 {P(sf['A']['blind'])}에서 {P(sf['D']['blind'])}(으)로 늘립니다.")
    if "E" in scen:
        s1.append(f"E안은 C안에 건당 적립 상한 {P(f_txcap)}과 '학원·교육 업종은 기본률만'을 더해 학원·교육 업종 몫을 "
                  f"{P(sf['A']['edu'])}에서 {P(sf['E']['edu'])}(으)로 줄입니다"
                  + (f"(고령 동네 1인당 적립액은 그 밖의 동네의 {P(sf['A']['percap'])} → {P(sf['E']['percap'])})."
                     if "percap" in sf["E"] and "percap" in sf["A"] else "."))
    if focus and "focus_amt" in sf[rec]:
        s1.append(f"{focus}에 돌아가는 적립액은 A안 {P(sf['A']['focus_amt'])}, {rec}안 {P(sf[rec]['focus_amt'])}입니다.")
    blocks.append(Paragraph(b.render(" ".join(s1))))

    # 그림: 예산 소진 ------------------------------------------------------------------------------------
    show = ["A", rec] if rec != "A" else ["A", "C"]
    if rec == "C′":
        show = ["A", "C", "C′"]
    lab0 = ["시작"] + [fr.month_label(o) for o in range(fr.months)]
    burn = pd.DataFrame({scen[k].label: np.concatenate([[0.0], np.cumsum(scen[k].monthly_need)]) for k in show}, index=lab0)
    burn.index.name = "계획 월(말)"
    bdisp = burn.apply(lambda c: c.map(lambda v: fmt_value(v, "원")))
    burn_r = make_derived([cmp_r], burn, bdisp, "적립 방식별 누적 적립 필요액과 예산", "누적 적립액", "원", AggKind.APPLIED,
                          Dim("pm", "", "계획 월", "time"), sdim, method="응용집계: 적립 방식별 월별 적립 필요액의 누계")
    ex = {scen[k].label: (scen[k].exhaust_x, f"{k}안 소진 {scen[k].exhaust_text()}") for k in show if scen[k].exhaust}
    fig = _figure(b, burn_r, f"적립 방식별 누적 적립액과 예산({fr.year}년, 선이 예산선을 넘는 곳이 소진)", render_burn_chart,
                  budget=B, exhaust=ex, budget_label=f"예산 {fmt_value(B, '원')}")
    if fig:
        blocks.append(fig)

    # 그림: 혜택의 행선지 --------------------------------------------------------------------------------
    targets = ["전통시장", "원도심"] + (["살리기 대상 동네"] if show_rev else []) + ["읍면"] + (["사각지대 동네"] if fr.has_blind else [])
    sh = pd.DataFrame({scen[k].label: [scen[k].shares[t] for t in targets] for k in show}, index=targets)
    if focus and all(scen[k].focus_share is not None for k in show):
        sh.loc[focus] = [scen[k].focus_share for k in show]
    sh.index.name = "대상"
    sh_r = make_derived([cmp_r], sh, sh.apply(lambda c: c.map(lambda v: fmt_value(v, "%"))),
                        "적립 혜택의 행선지(지급된 적립액 중 비중)", "적립액 비중", "%", AggKind.APPLIED,
                        Dim("target", "", "대상", "category"), sdim, method="응용집계: 지급된 적립액 중 대상별 비중")
    fig = _figure(b, sh_r, "적립 혜택이 어디로 가나(지급된 적립액 중 비중)", render_pair_bars, unit_note="%(지급된 적립액 중 비중)")
    if fig:
        blocks.append(fig)

    # 질문한 동네의 시장 적립률 -----------------------------------------------------------------------------
    markets = market_table(fr, cycles, focus_units)
    mk_lines = []
    for mkd in markets[:2]:
        rr = {"평일": R.rate_of(rev=mkd["rev"], mkt=True, blind=mkd["blind"]),
              "비 오는 날": R.rate_of(rev=mkd["rev"], mkt=True, rain=True, blind=mkd["blind"])}
        if mkd["days"]:
            rr["장날"] = R.rate_of(rev=mkd["rev"], mkt=True, mday=True, blind=mkd["blind"])
            rr["비 오는 장날"] = R.rate_of(rev=mkd["rev"], mkt=True, mday=True, rain=True, blind=mkd["blind"])
        ids = {kk: F(f"{rec}안 {mkd['name']} {kk} 적립률", v, "%") for kk, v in rr.items()}
        mkd["rates"] = {rec: ids}
        mk_lines.append(f"{mkd['name']}({mkd['hd']}{', 장날 ' + mkd['days'] if mkd['days'] else ''}): "
                        + ", ".join(f"{kk} {P(v)}" for kk, v in ids.items()))
    items = [f"{R.label}: {rule_ph(rec)}.", f"고른 이유: {why}."]
    if R.exhaust:
        items.append(f"이 안도 {R.exhaust_text()} 소진됩니다. 예산을 늘리거나 가산 폭을 줄여야 합니다.")
    else:
        items.append(f"예산이 {P(f_months)} 끝까지 유지됩니다(적립 중단 없음).")
    items += [f"{rec}안 시장 적립률 — {x}" for x in mk_lines]
    items.append("이 안으로 도청 개선안 초안·상인회 안내문·홍보 문구·어르신 안내문·일정표를 '실행물 묶음'으로 바로 만들 수 있습니다.")
    blocks.append(Callout(f"추천 배분안: {R.label}", [b.render(x) for x in items], "insight"))

    # 효과 반영(C′) ---------------------------------------------------------------------------------------
    if "C′" in scen and effect:
        dec = effect_decision(effect)
        fe, fp = effect.get("f_e"), effect.get("f_p")
        lead = (f"정책 전후 효과({P(fe)}, 위약 검정 p≈{P(fp)})" if fe and fp else "정책 전후 효과")
        blocks.insert(2, Paragraph(b.render(
            f"효과 확인 후 다음 배분: {lead}를 반영해 '{DECISION_TEXT[dec]}'(으)로 C′안을 만들었습니다. "
            f"C′안 기본 적립률은 {P(sf['C′']['base'])}, 전통시장 몫은 {P(sf['C′']['mkt'])}입니다.")))

    # 가정과 한계 -----------------------------------------------------------------------------------------
    SA, SR = sens["A"], sens.get(rec, sens["C"])
    pct = lambda x: (np.exp(x) - 1) * 100
    if b_src == "est":
        lead = (f"사용 반응은 '효과 추정' 절에서 이 자료로 구한 값(적립률 1%p당 사용액 {pct(b_use):+.1f}%, 앞당긴 소비 제외)을 썼습니다. "
                f"기준 1년({ptxt})의 사용액은 먼저 '적립률 10%였다면'으로 되돌렸습니다"
                + (f"(앞당긴 소비 업종 {', '.join(resp.flagged)}은 기간 중 부풀림과 뒤 빈자리를 걷어 냄)." if resp.flagged else "."))
        lab_sens = "반응이 없다고(사용 패턴 그대로) 놓으면"
    elif b_src == "set":
        lead = f"사용 반응은 설정 값(적립률 1%p당 사용액 {pin.response:+.1f}%)을 썼습니다."
        lab_sens = "반응이 없다고 놓으면"
    else:
        lead = (f"자료로 사용 반응을 추정하지 못해({resp.reason if resp is not None else '탐나는전 자료 부족'}) 적립률이 바뀌어도 사용 패턴이 "
                f"기준 1년({ptxt})과 같다고 가정했습니다. 실제로는 적립률이 오르면 사용도 늘어 예산이 더 빨리 줄어듭니다.")
        lab_sens = "적립률 1%p당 사용액이 5% 늘어난다고 놓으면(가정값, 도 발표 7→10% +21.6%에 가까운 크기)"
    pop_note = []
    if pub is not None:
        t = pub.table
        pop_note = [f"고령 동네 판정과 주민 1인당 계산의 인구는 공개 실데이터인 {pub.label}을 썼습니다(제주 읍면동 43곳 주민 "
                    f"{t['total'].sum():,.0f}명, 65세 이상 {t['age65'].sum() / t['total'].sum() * 100:.1f}%, 읍면동 고령화율 중앙값 {t['aging'].median():.1f}%). "
                    "적립액은 센터 탐나는전 자료에서, 인구는 이 공개 통계에서 가져옵니다."]
    cau = [lead,
           f"{lab_sens} A안 소진은 {SA.exhaust_text() or '없음'}(유지 {fmt_value(SA.lasting, '개월')}), "
           f"{rec}안 기본 적립률은 {fmt_value(SR.params['base'], '%')}입니다."
           + (f" 반응이 신뢰구간 위쪽(1%p당 {pct(resp.b_hi):+.1f}%)이면 A안 소진은 {sens_hi['A'].exhaust_text() or '없음'}, "
              f"{rec}안 기본 적립률은 {fmt_value(sens_hi.get(rec, sens_hi['C']).params['base'], '%')}입니다." if sens_hi else "")
           + " 실제 반응은 시행 뒤 '효과 확인'으로 다시 잽니다.",
           *pop_note,
           "캐시백은 연매출 10억 원 이하 가맹점 결제에만 적립됩니다. 탐나는전 자료(1-32)에는 가맹점 구분이 없어 모든 결제에 적립된다고 계산했으므로 "
           "필요 예산은 실제보다 조금 크게 나옵니다(공공데이터포털 탐나는전 가맹점 데이터 2026.3.31 기준, 가맹점 수로는 93.2%가 캐시백 대상).",
           f"A안의 명절 달은 {fr.year}년 설·추석이 있는 {'·'.join(str(x) for x in festive_months(fr.year))}월입니다. "
           "비 오는 날·장날은 기준 1년의 날짜를 그대로 썼습니다(실제 날씨는 해마다 다름).",
           "1인 월 적립 한도, 가맹점 연매출 기준, 충전·결제 방식 차이는 반영하지 않았습니다. 결과는 배분 방식의 비교용입니다."]
    if fr.est_months:
        cau.append(f"자료에 없는 {', '.join(f'{m}월' for m in fr.est_months)}은 가장 가까운 달의 사용 모양으로 채웠습니다"
                   "(연간 판단은 12개월 자료로 다시 확인하세요).")
    if not fr.has_rain:
        cau.append("날씨 자료(5-02)가 없어 비 오는 날 가산은 계산에 넣지 못했습니다.")
    if dropped > 0.05:
        cau.append(f"결제 10건 미만 칸(사용액의 {dropped:.1f}%)은 계산에서 뺐습니다.")
    ratio = B / fr.usage_window * 100
    if user_budget and not (0.5 <= ratio <= 25):
        cau.append(f"예산이 계획 기간 사용액의 {ratio:.1f}%입니다. 예산 단위(억 원)와 자료 범위가 맞는지 확인하세요.")
    for k in keys:
        if scen[k].note and k != "C′":
            cau.append(f"{k}안: {scen[k].note}.")
    blocks.append(Callout("가정과 한계", cau, "caution"))

    # 효과 확인 -------------------------------------------------------------------------------------------
    q = effect_question(fr.year, fr.start, focus)
    blocks.append(Callout("실행 후 효과 확인(다음 방문 때 그대로 붙여 넣기)", [
        f"질문: “{q}”",
        "필요한 자료: " + " / ".join(NEEDED_DATA),
        "시점: 시행 4주 뒤 예산 집행 점검(월 적립액이 계획보다 10% 넘게 많으면 기본률 조정), 12주 뒤 효과 확인 — 전통시장과 일반 가맹점의 "
        "주간 이중차분·위약 검정으로 판정하고, 결과에 따라 다음 배분(C′안)을 다시 계산합니다."], "note"))

    # 초안(자리표시자) ---------------------------------------------------------------------------------------
    summary, rx, plan = [], [], []
    if A.exhaust:
        summary.append(f"예산 {P(f_budget)}(으)로 지금 방식(A안)은 {P(sf['A']['exm'])}께 적립이 멈추고, {rec}안({R.name})은 "
                       f"{P(sf[rec]['last'])} 동안 이어지며 전통시장 몫이 {P(sf['A']['mkt'])}에서 {P(sf[rec]['mkt'])}(으)로 바뀝니다.")
    else:
        summary.append(f"예산 {P(f_budget)}(으)로 지금 방식도 {P(f_months)}을 버팁니다. {rec}안({R.name})이면 전통시장 몫이 "
                       f"{P(sf['A']['mkt'])}에서 {P(sf[rec]['mkt'])}(으)로 바뀝니다.")
    rx.append(f"적립 예산 배분: {rec}안({R.name}) — {rule_ph(rec)}. 예산을 {P(sf[rec]['last'])} 동안 유지하면서 "
              f"원도심 몫 {P(sf[rec]['old'])}, 전통시장 몫 {P(sf[rec]['mkt'])}입니다.")
    if A.exhaust:
        plan.append(f"현행 방식(기본 {P(sf['A']['base'])}, 명절 달 {P(sf['A']['peak'])})으로는 적립 예산 {P(f_budget)}이 "
                    f"{P(sf['A']['exm'])}께 소진되므로, {rule_ph(rec)}하는 배분({rec}안)으로 {P(sf[rec]['last'])} 동안 적립을 유지하면서 "
                    f"전통시장 몫을 {P(sf[rec]['mkt'])}(으)로 높일 수 있다.")
    else:
        plan.append(f"적립 예산 {P(f_budget)}을 {rule_ph(rec)}하는 배분({rec}안)으로 쓰면 전통시장 몫을 {P(sf[rec]['mkt'])}(으)로 높일 수 있다.")

    sec = Section("처방 설계 — 탐나는전 적립 예산 배분", blocks)
    pack = {"where": where, "focus": focus, "focus_units": sorted(focus_units) if focus_units else [], "scope": inp.scope,
            "year": fr.year, "start": fr.start, "months": fr.months, "budget": B, "user_budget": user_budget,
            "period": period, "period_text": ptxt, "f_budget": f_budget, "f_months": f_months, "f_cap": f_cap,
            "scen": scen, "sf": sf, "rule_ph": {k: rule_ph(k) for k in keys}, "rec": rec, "why": why, "order": keys,
            "markets": markets, "f_rain": f_rain, "f_md": f_md, "diag": (dn.diag if dn is not None else None),
            "types": (dn.types if dn is not None else {}), "table": disp, "results": [cmp_r, burn_r, sh_r],
            "question": q, "needs": list(NEEDED_DATA),
            "effect": effect, "decision": (effect_decision(effect) if effect and "C′" in scen else None),
            "festive": festive_months(fr.year), "has_rain": fr.has_rain, "has_cycles": fr.has_cycles,
            "frame": fr, "sens": {"A_exhaust": SA.exhaust_text(), "A_last": SA.lasting, "rec_base": SR.params["base"],
                                  "rec_use": SR.use_chg, "A_use": SA.use_chg},
            "response": resp, "b_use": b_use, "b_src": b_src, "f_txcap": f_txcap, "pop_src": pop_src,
            "pop_public": pub}
    burn_fig = [blk for blk in blocks if getattr(blk, "result", None) is burn_r]
    pack["figures"] = {"burn": burn_fig[0].path if burn_fig else None}
    return {"section": sec, "summary": summary, "rx": rx, "plan": plan, "pack": pack}
