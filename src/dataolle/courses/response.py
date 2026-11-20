"""효과 추정 — 적립률을 올리면 탐나는전 사용이 얼마나 느나(반응도), 그중 '앞당긴 소비'는 얼마인가.

자연실험: 적립률이 해마다 다르게 바뀐 달(예: 2025년 4~6월 15%)을 작년 같은 달과 비교한다.
  Δ(업종, 달) = log(올해 사용액) − log(작년 같은 달 사용액) = 업종 성장(g) + b × 적립률 차이(%p) + 앞당긴 소비

  1) 성장 g: 같은 해에서 적립률 차이가 흔한 값인 달(예: 2025년 1~3월, 10% vs 7%)로 구한다.
  2) 반응 b: '높은 적립률 기간 + 바로 뒤 두 달'을 한 덩어리로 맞춘다. 선결제처럼 앞당긴 소비는 기간 안에서 늘고 바로 뒤에서
     줄어 덩어리 합계에서는 상쇄되므로, b는 '새로 생긴 소비'만 잰다. 높은 기간만 보면(흔한 방식) 앞당긴 소비까지 반응으로 잡는다
     → b_naive(착시 포함)를 함께 보여 준다.
  3) 업종별로 기간 중 초과분과 뒤 두 달의 빈자리, 건당 금액 변화를 보고 '앞당긴 소비 의심' 업종을 찾는다(학원 선결제 등).
  4) 백테스트: 추정에 쓰지 않은 달(예: 2025년 9월 13% vs 2024년 9월 14%)의 사용액을 예측해 오차를 잰다(b와 b_naive 둘 다).
  5) 외부 근거: 도가 발표한 과거 반응(7→10% +21.6%, 7→14% +39.2%)과 견준다.

모든 계산은 이 PC 안에서 하고, 결과(비율·지수·추정값)는 응용집계다. 적립률 이력은 courses/rates.py.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from dataolle.analysis.semantic import MODELS
from dataolle.courses.rates import OFFICIAL_RESPONSES, rate_of

# 공개된 탐나는전 가맹점 현황(2026.3.31)의 실제 업종 이름: 입시/교습학원, 기술/기능 교육, 외국어/언어, 예술 교육, 기타교육기관
EDU_WORDS = ("학원", "교습", "교육", "외국어", "어학")
GRID = np.arange(-0.05, 0.2001, 0.0005)
MIN_DR = 3.0                      # 자연실험으로 쓸 최소 적립률 차이(%p)
WIN_FRAC = 0.75                   # 가장 큰 차이의 이만큼 이상이면 같은 높은 적립률 기간으로 묶음


def is_edu(name: str) -> bool:
    return any(w in str(name) for w in EDU_WORDS)


def _prev(ym: str) -> str:
    return f"{int(ym[:4]) - 1}{ym[4:]}"


def _next(ym: str, k: int = 1) -> str:
    y, m = int(ym[:4]), int(ym[4:]) + k
    y += (m - 1) // 12
    m = (m - 1) % 12 + 1
    return f"{y}{m:02d}"


@dataclass
class ResponseResult:
    ok: bool
    reason: str = ""
    b: float = float("nan")                  # 순수 반응: log 사용액 / 적립률 %p
    b_naive: float = float("nan")            # 높은 기간만 본 반응(앞당긴 소비 포함)
    b_lo: float = float("nan")
    b_hi: float = float("nan")
    window: tuple[str, str] | None = None
    post: list[str] = field(default_factory=list)
    dr_window: float = 0.0
    per_biz: pd.DataFrame | None = None
    backtest: list[dict] = field(default_factory=list)
    monthly: pd.DataFrame | None = None      # 달별 실제 vs 예측(합계)
    adj: dict = field(default_factory=dict)  # (업종, 연월) → 앞당긴 소비를 걷어 낸 배수(시뮬레이션 기준 1년 보정용)
    official: list[dict] = field(default_factory=list)
    months: list[str] = field(default_factory=list)
    edu_share_prev: float = float("nan")     # 학원·교육 업종 사용액 비중(작년 같은 기간)
    edu_share_win: float = float("nan")      # 같은 비중(높은 적립률 기간)
    flagged: list[str] = field(default_factory=list)

    @staticmethod
    def pct(b: float) -> float:
        return (np.exp(b) - 1) * 100 if b == b else float("nan")

    @property
    def pct_net(self) -> float:
        return self.pct(self.b)

    @property
    def pct_naive(self) -> float:
        return self.pct(self.b_naive)

    @property
    def mape(self) -> float:
        e = [abs(x["err"]) for x in self.backtest if x.get("err") == x.get("err")]
        return float(np.mean(e)) if e else float("nan")

    @property
    def mape_naive(self) -> float:
        e = [abs(x["err_naive"]) for x in self.backtest if x.get("err_naive") == x.get("err_naive")]
        return float(np.mean(e)) if e else float("nan")


def monthly_by_biz(cat) -> pd.DataFrame:
    T, d = cat.tables["tamna"], MODELS["tamna"].dims
    sql = (f"SELECT {d['month'].expr} AS ym, {d['biz'].expr} AS biz, SUM(amount) AS amt, SUM(cnt) AS cnt "
           f"FROM {T} GROUP BY ALL")
    q = cat.con.execute(sql).df()
    q["ym"] = q["ym"].astype(str).str[:6]
    q["amt"] = q["amt"].astype(float)
    q["cnt"] = q["cnt"].astype(float)
    q["biz"] = q["biz"].fillna("(업종 정보 없음)")
    return q


def estimate(cat) -> ResponseResult:
    if cat is None or not cat.has("tamna"):
        return ResponseResult(False, "탐나는전(1-32) 자료가 없습니다.")
    q = monthly_by_biz(cat)
    return estimate_from(q)


def estimate_from(q: pd.DataFrame) -> ResponseResult:
    months = sorted(q["ym"].unique())
    y = q.pivot_table(index="biz", columns="ym", values="amt", aggfunc="sum").fillna(0.0)
    # 하루 평균으로(2024년 2월 29일처럼 달의 날수가 해마다 다르면 작년 대비가 틀어짐)
    import calendar
    y = y / np.array([calendar.monthrange(int(m[:4]), int(m[4:]))[1] for m in y.columns])
    c = q.pivot_table(index="biz", columns="ym", values="cnt", aggfunc="sum").fillna(0.0)
    res = ResponseResult(False, months=months)
    res.official = [{"from": a, "to": b_, "pct": p, "source": s, "pct_per_pp": (np.exp(np.log(1 + p / 100) / (b_ - a)) - 1) * 100}
                    for a, b_, p, s in OFFICIAL_RESPONSES]
    pairs = [m for m in months if _prev(m) in y.columns and rate_of(m) is not None and rate_of(_prev(m)) is not None]
    if len(pairs) < 4:
        res.reason = "작년 같은 달과 비교할 수 있는 달이 4개 미만입니다(최소 13개월 이상 자료 필요)."
        return res
    dr = {m: rate_of(m) - rate_of(_prev(m)) for m in pairs}
    top = max(dr.values())
    if top < MIN_DR:
        res.reason = f"작년 같은 달보다 적립률이 {MIN_DR:g}%p 이상 높았던 달이 자료 기간 안에 없습니다."
        return res
    win = [m for m in pairs if dr[m] >= top * WIN_FRAC]              # 가장 큰 차이와 비슷하게 높았던 달까지 한 기간으로
    # 가장 긴 연속 구간
    runs, cur = [], [win[0]]
    for m in win[1:]:
        if m == _next(cur[-1]):
            cur.append(m)
        else:
            runs.append(cur)
            cur = [m]
    runs.append(cur)
    runs = [r for r in runs if any(dr[m] == top for m in r)] or runs
    win = max(runs, key=len)
    post = [m for m in (_next(win[-1], 1), _next(win[-1], 2)) if m in pairs]
    wy = win[0][:4]
    clean = [m for m in pairs if m not in win and m not in post]
    # 해마다 '흔한 적립률 차이'인 달로 성장을 구하고, 나머지 깨끗한 달은 백테스트로 남긴다
    grow: dict[str, list[str]] = {}
    hold: list[str] = []
    for yr in sorted({m[:4] for m in clean}):
        ms = [m for m in clean if m[:4] == yr]
        mode = Counter(dr[m] for m in ms).most_common(1)[0][0]
        grow[yr] = [m for m in ms if dr[m] == mode]
        hold += [m for m in ms if dr[m] != mode]
    if not grow.get(wy):
        res.reason = f"{wy}년에 성장률을 잴 달(적립률 차이가 같은 달)이 없습니다."
        return res

    tot = y[[m for m in months]].sum(axis=1)
    keep = tot[tot >= tot.sum() * 0.005].index                       # 아주 작은 업종은 반응 추정에서 뺌
    Y, Yp = y.loc[keep], y.loc[keep]
    lg = lambda a: np.log(np.maximum(a, 1.0))

    A = {m: Y[m].to_numpy() for m in pairs}                      # 올해 하루 평균(업종 순)
    PV = {m: Yp[_prev(m)].to_numpy() for m in pairs}             # 작년 같은 달

    def growth(b, yr):
        # 달별 log 비의 평균이 아니라 합계의 비(설 연휴가 1월↔2월로 옮겨 다니는 영향을 상쇄)
        ms = grow[yr]
        act = np.sum([A[m] for m in ms], axis=0)
        base = np.sum([PV[m] * np.exp(b * dr[m]) for m in ms], axis=0)
        return lg(act) - lg(base)

    block = win + post
    w = Y[block].sum(axis=1).to_numpy()
    w = w / w.sum()

    def loss_vec(b, ms):
        g = growth(b, wy)
        act = lg(np.sum([A[m] for m in ms], axis=0))
        pred = lg(np.sum([PV[m] * np.exp(g + b * dr[m]) for m in ms], axis=0))
        return act - pred

    Lb = np.array([loss_vec(x, block) for x in GRID])              # 격자 × 업종
    Lw = np.array([loss_vec(x, win) for x in GRID])
    b = float(GRID[int(np.argmin((Lb ** 2) @ w))])
    ww = Y[win].sum(axis=1).to_numpy()
    b_naive = float(GRID[int(np.argmin((Lw ** 2) @ (ww / ww.sum())))])
    # 업종별 b → 퍼짐으로 신뢰구간
    bb = GRID[np.argmin(Lb ** 2, axis=0)]
    n_eff = 1 / np.sum(w ** 2)
    se = float(np.sqrt(np.sum(w * (bb - b) ** 2) / max(1.0, n_eff - 1)))
    res.b, res.b_naive, res.b_lo, res.b_hi = b, b_naive, b - 1.96 * se, b + 1.96 * se
    res.window, res.post, res.dr_window = (win[0], win[-1]), post, float(np.mean([dr[m] for m in win]))

    # 업종별 표: 기간 중 증가, 순수 반응 몫, 앞당긴 소비 몫, 건당 금액 변화
    g = growth(b, wy)
    rows, adj, flagged = [], {}, []
    for i, biz in enumerate(keep):
        base = {m: float(Yp.loc[biz, _prev(m)]) * float(np.exp(g[i])) for m in block}      # 적립률이 작년과 같았다면
        net = {m: base[m] * float(np.exp(b * dr[m])) for m in block}                       # 순수 반응만 있었다면
        act = {m: float(Y.loc[biz, m]) for m in block}
        inc = sum(act[m] - base[m] for m in win)
        excess = sum(act[m] - net[m] for m in win)
        deficit = sum(net[m] - act[m] for m in post) if post else float("nan")
        pf = (min(1.0, max(0.0, deficit / inc)) if inc > 0 and deficit == deficit else float("nan"))
        cw = float(c.loc[biz, win].sum())
        cp = float(c.loc[biz, [_prev(m) for m in win]].sum())
        tw = float(Y.loc[biz, win].sum()) / cw if cw else float("nan")
        tp = float(Yp.loc[biz, [_prev(m) for m in win]].sum()) / cp if cp else float("nan")
        tr = tw / tp if tp == tp and tp else float("nan")
        flag = bool(pf == pf and pf >= 0.3 and deficit > 0 and excess > 0 and tr == tr and tr >= 1.2)
        if flag:
            flagged.append(biz)
            for m in block:
                adj[(biz, m)] = net[m] / act[m] if act[m] > 0 else 1.0
        rows.append({"업종": biz, "사용액 비중": float(tot[biz] / tot.sum() * 100),
                     "기간 중 증가": (sum(act[m] for m in win) / sum(base[m] for m in win) - 1) * 100,
                     "순수 반응 몫": (sum(net[m] for m in win) / sum(base[m] for m in win) - 1) * 100,
                     "뒤 두 달 빈자리": (deficit / sum(net[m] for m in post) * 100 if post and deficit == deficit else float("nan")),
                     "앞당긴 소비 비율": (pf * 100 if pf == pf else float("nan")),
                     "건당 금액 변화": ((tr - 1) * 100 if tr == tr else float("nan")),
                     "업종별 반응": ResponseResult.pct(float(bb[i])),
                     "판정": "앞당긴 소비 의심" if flag else ""})
    res.per_biz = pd.DataFrame(rows).set_index("업종").sort_values("사용액 비중", ascending=False)
    res.adj, res.flagged = adj, flagged
    edu = [bz for bz in y.index if is_edu(bz)]
    if edu:
        prev_w = [_prev(m) for m in win]
        res.edu_share_prev = float(y.loc[edu, prev_w].sum().sum() / y[prev_w].sum().sum() * 100)
        res.edu_share_win = float(y.loc[edu, win].sum().sum() / y[win].sum().sum() * 100)

    # 백테스트: 추정에 쓰지 않은 달
    for m in hold:
        yr = m[:4]
        if not grow.get(yr):
            continue
        out = {"ym": m, "rate": rate_of(m), "prev_rate": rate_of(_prev(m)), "actual": float(Y[m].sum())}   # 하루 평균 사용액
        for key, bx in (("pred", b), ("pred_naive", b_naive)):
            gg = growth(bx, yr)
            out[key] = float(np.sum(Yp[_prev(m)].to_numpy() * np.exp(gg + bx * dr[m])))
        out["err"] = (out["pred"] / out["actual"] - 1) * 100
        out["err_naive"] = (out["pred_naive"] / out["actual"] - 1) * 100
        res.backtest.append(out)
    # 달별 실제 vs 예측(그림용)
    mon = []
    for m in pairs:
        yr = m[:4]
        if not grow.get(yr):
            continue
        gg = growth(b, yr)
        mon.append({"ym": m, "실제": float(Y[m].sum()),            # 하루 평균
                    "예측(순수 반응)": float(np.sum(Yp[_prev(m)].to_numpy() * np.exp(gg + b * dr[m]))),
                    "적립률": rate_of(m)})
    res.monthly = pd.DataFrame(mon).set_index("ym") if mon else None
    res.ok = True
    return res


def normalize_amounts(seg: pd.DataFrame, rr: ResponseResult, ref_rate: float) -> pd.DataFrame:
    """기준 1년 사용액을 '적립률 ref_rate%였다면'으로 되돌린다(앞당긴 소비 업종은 먼저 걷어 냄).
    seg: d(YYYYMMDD), biz, amt 열이 있는 표."""
    if not rr.ok:
        return seg
    s = seg.copy()
    ym = s["d"].str[:6]
    if rr.adj and "biz" in s.columns:
        for (bz, m), f in rr.adj.items():
            sel = (s["biz"] == bz) & (ym == m)
            s.loc[sel, "amt"] = s.loc[sel, "amt"] * f
    rmap = {m: (rate_of(m) if rate_of(m) is not None else ref_rate) for m in ym.unique()}
    hist = ym.map(rmap).astype(float)
    s["amt"] = s["amt"] * np.exp(-rr.b * (hist - ref_rate))
    if "cnt" in s.columns:
        s["cnt"] = s["cnt"] * np.exp(-rr.b * (hist - ref_rate))
    return s


__all__ = ["estimate", "estimate_from", "ResponseResult", "normalize_amounts", "is_edu", "EDU_WORDS"]


# ---------------------------------------------------------------------------------------------- 보고서 절
def build_section(b, P, cat, rr: ResponseResult, bases: list) -> dict | None:
    """'효과 추정' 절. 반환: {section, summary, rx, plan, facts} — 문장 초안은 자리표시자만 씀."""
    from dataolle.analysis.derive import make_derived
    from dataolle.analysis.fmt import fmt_value
    from dataolle.analysis.semantic import Dim
    from dataolle.courses.policy import _figure
    from dataolle.courses.rates import history_rows
    from dataolle.report.charts import render_response_chart
    from dataolle.report.document import Callout, Paragraph, Section, Table
    from dataolle.sdc.engine import AggKind

    title = "적립률 반응도 추정(작년 같은 달 비교 · 앞당긴 소비 분리)"
    if not rr.ok:
        sec = Section("효과 추정 — 적립률을 올리면 얼마나 더 쓰나", [
            Paragraph(f"이 자료로는 반응도를 추정하지 못했습니다: {rr.reason}"),
            Callout("어떻게 하면 되나", ["탐나는전(1-32)을 적립률이 바뀐 기간과 그 작년 같은 달이 모두 들어가게(예: 2024년 1월~2025년 9월) "
                                     "받으면 자동으로 추정합니다. 운영자에게 여러 달 치를 미리 요청하세요.",
                                     "추정 전까지 처방 설계는 '사용 패턴 그대로'로 계산하고, 도 발표(7→10% 때 +21.6%)를 참고로 민감도를 함께 보여 줍니다."],
                    "note")])
        return {"section": sec, "summary": [], "rx": [], "plan": [], "facts": {}}
    F = lambda label, v, unit, change=False: b.fact(label, v, unit, AggKind.APPLIED, title, change)
    w0, w1 = rr.window
    wtxt = f"{w0[:4]}년 {int(w0[4:])}~{int(w1[4:])}월" if w0[:4] == w1[:4] else f"{w0[:4]}.{w0[4:]}~{w1[:4]}.{w1[4:]}"
    rate_w = rate_of(w0)
    f = {"net": F("적립률을 한 %p 올릴 때 탐나는전 사용액 변화(순수 반응, 앞당긴 소비 제외)", rr.pct_net, "%", True),
         "naive": F("적립률을 한 %p 올릴 때 탐나는전 사용액 변화(높은 기간만 본 단순 비교)", rr.pct_naive, "%", True),
         "lo": F("순수 반응 신뢰구간 하한", rr.pct(rr.b_lo), "%", True),
         "hi": F("순수 반응 신뢰구간 상한", rr.pct(rr.b_hi), "%", True),
         "ill": F("단순 비교가 반응을 부풀린 정도", (rr.b_naive / rr.b - 1) * 100 if rr.b > 0 else float("nan"), "%", True),
         "dr": F("높은 적립률 기간의 작년 대비 적립률 차이", rr.dr_window, "%p")}
    if rr.backtest:
        f["bt"] = F("백테스트 예측 오차 평균(순수 반응)", rr.mape, "%")
        f["btn"] = F("백테스트 예측 오차 평균(단순 비교)", rr.mape_naive, "%")
    for i, o in enumerate(rr.official, 1):
        f[f"off{i}"] = F(f"도 발표 과거 적립률 인상 사례 {i}: 한 %p당 사용액 변화(공개 보도)", o["pct_per_pp"], "%", True)
    pb = rr.per_biz
    fl = rr.flagged[0] if rr.flagged else None
    if fl is not None:
        f["pf"] = F(f"{fl} 업종: 높은 기간 증가분 중 앞당긴 소비 비율", pb.loc[fl, "앞당긴 소비 비율"], "%")
        f["tr"] = F(f"{fl} 업종: 높은 기간 건당 금액 변화(작년 같은 달 대비)", pb.loc[fl, "건당 금액 변화"], "%", True)
        f["gap"] = F(f"{fl} 업종: 기간 바로 뒤 두 달 사용액 빈자리", pb.loc[fl, "뒤 두 달 빈자리"], "%")
    if rr.edu_share_prev == rr.edu_share_prev:
        f["edu0"] = F("학원·교육 업종 사용액 비중(작년 같은 기간)", rr.edu_share_prev, "%")
        f["edu1"] = F("학원·교육 업종 사용액 비중(높은 적립률 기간)", rr.edu_share_win, "%")

    # 표 1: 추정 요약 ------------------------------------------------------------------------------------
    rows = [{"항목": "순수 반응(앞당긴 소비 제외)", "적립률 1%p당 사용액": fmt_value(rr.pct_net, "%"),
             "설명": f"{wtxt}({rate_w:g}%)와 바로 뒤 두 달을 한 덩어리로 작년 같은 달과 비교"},
            {"항목": "95% 신뢰구간", "적립률 1%p당 사용액": f"{fmt_value(rr.pct(rr.b_lo), '%')} ~ {fmt_value(rr.pct(rr.b_hi), '%')}",
             "설명": "업종별 추정값의 퍼짐으로 계산"},
            {"항목": "단순 비교(높은 기간만)", "적립률 1%p당 사용액": fmt_value(rr.pct_naive, "%"),
             "설명": "흔한 방식 — 미리 결제한 몫까지 반응으로 잡음"}]
    for x in rr.backtest:
        rows.append({"항목": f"백테스트 {x['ym'][:4]}.{x['ym'][4:]}({x['prev_rate']:g}%→{x['rate']:g}%)",
                     "적립률 1%p당 사용액": f"오차 {fmt_value(x['err'], '%')}",
                     "설명": f"추정에 쓰지 않은 달의 사용액 예측. 단순 비교로 예측하면 오차 {fmt_value(x['err_naive'], '%')}"})
    for o in rr.official:
        rows.append({"항목": f"외부 근거: {o['from']:g}→{o['to']:g}%", "적립률 1%p당 사용액": fmt_value(o["pct_per_pp"], "%"),
                     "설명": f"월평균 사용액 +{o['pct']:g}% — {o['source']}"})
    disp = pd.DataFrame(rows).set_index("항목")
    num = pd.DataFrame({"값": [rr.pct_net, rr.pct(rr.b_lo), rr.pct_naive] + [x["err"] for x in rr.backtest]
                        + [o["pct_per_pp"] for o in rr.official]}, index=disp.index)
    idim, mdim = Dim("item", "", "항목", "category"), Dim("metric", "", "지표", "category")
    method = ("응용집계: 업종×달 하루 평균 사용액의 작년 같은 달 대비 log 비를 업종 성장과 적립률 차이로 나눔(높은 기간+뒤 두 달을 한 덩어리로 맞춰 "
              "앞당긴 소비를 상쇄). 적립률 이력은 보도 기준(courses/rates.py)")
    r1 = make_derived(bases[:1], num, disp, title, "반응도", "%", AggKind.APPLIED, idim, mdim, method=method)
    r1.period = (rr.months[0], rr.months[-1])
    b.keep(r1)
    b.ntab += 1
    t1 = Table(disp, f"표 {b.ntab}. 적립률 반응도 추정과 검증", r1, style="text", widths=[30, 22, 48])

    # 표 2: 업종별 -----------------------------------------------------------------------------------------
    show = pb[(pb["사용액 비중"] >= 2) | (pb["판정"] != "")].head(10)
    cols = ["사용액 비중", "기간 중 증가", "순수 반응 몫", "뒤 두 달 빈자리", "앞당긴 소비 비율", "건당 금액 변화"]
    d2 = show[cols].apply(lambda c: c.map(lambda v: fmt_value(v, "%") if v == v else "-"))
    d2["판정"] = show["판정"].replace("", "-")
    r2 = make_derived([r1], show[cols], d2, "업종별 높은 적립률 기간의 증가와 앞당긴 소비", "증감률", "%", AggKind.APPLIED,
                      Dim("biz", "", "업종", "category"), mdim,
                      method="응용집계: 업종별 작년 같은 달 대비 증감(성장 제외)과 기간 뒤 두 달의 빈자리, 건당 금액 비")
    r2.period = r1.period
    b.keep(r2)
    b.ntab += 1
    t2 = Table(d2, f"표 {b.ntab}. 업종별 {wtxt} 증가와 앞당긴 소비(기간 중 증가·순수 반응 몫은 '적립률이 작년과 같았다면' 대비)",
               r2, style="text", widths=[16, 11, 12, 12, 13, 13, 12, 11])

    # 그림 --------------------------------------------------------------------------------------------------
    fig = None
    if rr.monthly is not None and len(rr.monthly):
        mon = rr.monthly.copy()
        mon.index = [f"{m[:4]}.{m[4:]}" for m in mon.index]
        mon.index.name = "연월"
        mdisp = mon.apply(lambda c: c.map(lambda v: fmt_value(v, "%" if c.name == "적립률" else "원")))
        r3 = make_derived([r1], mon, mdisp, "달별 하루 평균 사용액: 실제와 반응도로 그린 예측", "사용액", "원", AggKind.APPLIED,
                          Dim("ym", "", "연월", "time"), Dim("series", "", "계열", "category"),
                          method="응용집계: 작년 같은 달 × 업종 성장 × exp(반응도 × 적립률 차이)")
        r3.period = r1.period
        fig = _figure(b, r3, "적립률이 오른 달, 사용은 얼마나 늘었나(실제 vs 순수 반응 예측)", render_response_chart,
                      window=(f"{w0[:4]}.{w0[4:]}", f"{w1[:4]}.{w1[4:]}"))

    # 문단 ---------------------------------------------------------------------------------------------------
    blocks = [Paragraph("예산 소진일을 맞히려면 '적립률을 올리면 사용이 얼마나 늘어나는가'를 알아야 합니다. 같은 사람에게 두 적립률을 줄 수는 없으므로, "
                        "적립률이 해마다 다르게 바뀐 달을 작년 같은 달과 비교하는 자연실험으로 잽니다. 이때 학원비처럼 몇 달치를 미리 결제할 수 "
                        "있는 업종은 높은 적립률 기간에 크게 늘었다가 바로 뒤에 줄어듭니다. 이 '앞당긴 소비'를 반응으로 착각하면 예산이 실제보다 "
                        "빨리 늘어날 것처럼 계산하게 됩니다. 그래서 높은 기간과 바로 뒤 두 달을 한 덩어리로 맞춰 새로 생긴 소비만 잽니다."),
              Paragraph(b.render(
                  f"이 자료에서 적립률 1%p당 탐나는전 사용액은 {P(f['net'])} 변했습니다(95% 신뢰구간 {P(f['lo'])}~{P(f['hi'])}). "
                  f"높은 기간만 작년과 비교하는 흔한 방식이면 {P(f['naive'])}(으)로, 반응을 {P(f['ill'])} 부풀립니다."
                  + (f" 추정에 쓰지 않은 달로 검증하면 예측 오차는 평균 {P(f['bt'])}로, 단순 비교 방식({P(f['btn'])})보다 작습니다." if "bt" in f else "")
                  + (" 도가 발표한 과거 반응(" + ", ".join(f"{o['from']:g}→{o['to']:g}% 때 1%p당 {P(f[f'off{i}'])}" for i, o in enumerate(rr.official, 1))
                     + ")과도 견줄 수 있습니다." if rr.official else ""))),
              t1]
    if fig:
        blocks.append(fig)
    blocks.append(t2)
    if fl is not None:
        items = [b.render(f"{fl}: {wtxt} 증가분의 {P(f['pf'])}는 바로 뒤 두 달에서 빠진 '앞당긴 소비'였습니다. 같은 기간 건당 금액은 "
                          f"작년보다 {P(f['tr'])} 달라졌고, 뒤 두 달 사용액은 순수 반응 예측보다 {P(f['gap'])} 적었습니다."),
                 "이 몫은 새로 생긴 소비가 아니라 시기만 당긴 것이라, 예산만 먼저 빠지고 지역 소비는 늘지 않습니다.",
                 "학원·교육비는 소득이 높은 가구가 더 많이 쓰는 지출이라(2025년 초중고 사교육비조사: 학생 1인당 월평균 사교육비가 월소득 800만 원 이상 가구 66만 2천 원, "
                 "300만 원 미만 가구 19만 2천 원 — 교육부·국가데이터처, 2026.3.12 발표), 적립률을 올릴수록 혜택이 그쪽으로 쏠릴 수 있습니다(역진성)."]
        if "edu0" in f:
            items.append(b.render(f"학원·교육 업종 사용액 비중: 작년 같은 기간 {P(f['edu0'])} → 높은 적립률 기간 {P(f['edu1'])}."))
        items.append("처방: 적립률을 올릴 때는 학원·교육 업종을 가산에서 빼거나 건당 적립 상한을 둡니다(처방 설계 E안 '쏠림 방지형').")
        blocks.append(Callout("앞당긴 소비(선결제) 착시", items, "insight"))
    hist = pd.DataFrame(history_rows([m for m in rr.months])).drop_duplicates(subset=["적립률", "월 한도", "근거"]).set_index("연월")
    blocks.append(Callout("가정과 한계", [
        "적립률 이력은 보도 원문으로 확인한 값입니다(아래 표, 2024년 1월 이후). 2023년은 현장 할인 방식이라 추정에서 뺐고, 달 중간에 바뀐 달은 날수 평균이며, '가정'으로 표시한 구간은 달 안에서 적립이 멈춘 날이 있거나 끝난 날을 확인하지 못한 구간입니다. 담당 부서 공식 이력이 있으면 설치 폴더 data/rates.csv로 바꿔 넣으면 됩니다.",
        "누가 썼는지(나이·소득·도민 여부)와 1인 월 적립 한도 효과는 자료에 없어 직접 볼 수 없습니다. 역진성은 업종(학원·교육)과 동네(고령 비율) 단위로 대신 봅니다.",
        "적립률 외에 같은 시기 다른 변화(정부 지원 확대, 경기, 가맹점 수)가 섞일 수 있습니다. 업종 성장은 같은 해 다른 달로 걷어 냈지만 완전하지 않습니다.",
        "높은 기간 바로 뒤 두 달 안에 다 빠지지 않는 선결제(3개월 이상)는 순수 반응에 일부 남을 수 있습니다."], "caution"))
    blocks.append(Table(hist, "적립률 이력(이 자료 기간, 바뀐 구간만)", None, style="text", widths=[14, 12, 14, 60]))
    sec = Section("효과 추정 — 적립률을 올리면 얼마나 더 쓰나(앞당긴 소비 분리)", blocks)

    summary = [f"적립률 1%p당 탐나는전 사용액 변화는 {P(f['net'])}(앞당긴 소비 제외)이고, 높은 기간만 본 단순 비교({P(f['naive'])})는 반응을 "
               f"{P(f['ill'])} 부풀립니다." + (f" 쓰지 않은 달 예측 오차 {P(f['bt'])}." if "bt" in f else "")]
    rx = []
    if fl is not None:
        summary.append(f"{fl} 업종은 높은 적립률 기간 증가분의 {P(f['pf'])}가 앞당긴 소비(건당 금액 {P(f['tr'])})였습니다.")
        rx.append(f"선결제가 몰리는 업종({fl})은 적립률 인상 때 가산에서 빼거나 건당 적립 상한을 두세요(E안 쏠림 방지형).")
    plan = [f"탐나는전 자료의 자연실험({wtxt} 적립률 상향)으로 추정한 적립률 1%p당 사용액 변화는 {P(f['net'])}이며, "
            f"선결제 등 앞당긴 소비를 걷어 내지 않으면 {P(f['naive'])}(으)로 과대 추정된다."]
    return {"section": sec, "summary": summary, "rx": rx, "plan": plan, "facts": f}
