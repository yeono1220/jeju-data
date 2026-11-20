"""반출 검수 엔진 — 제주빅데이터센터 데이터 반출 기준(2026.09.01 개정)을 코드로 구현.

반출 불가 데이터(카드·POS·온라인카드·탐나는전·기업·유동인구)의 분석산출물 규칙
- 단순집계(합계·평균·최솟값·최댓값·중앙값 등): 10 미만 수치가 없을 때만 반출 가능
  · 처리방법 1 범주처리(상위 단위로 묶기) / 2 삭제처리 / 3 비식별처리
  · 상위 단위로 다시 집계해 모든 칸을 10 이상으로 만드는 것도 범주처리로 인정(게시판 답변)
- 역산할 수 있으면 반출 제한(게시판 답변) → 합계에서 빼서 숨긴 값을 알 수 없게 2차 억제
- 응용집계(차이·증감률·비율·순위·범주·지수·모델링 결과·검정값): 수치 포함 반출 가능
- 시각화: 그림 안의 수치는 위 기준을 따름

이 모듈은 '표 한 장' 단위로 동작한다. 보고서에 들어가는 모든 표와 차트 데이터가 이 검수를 통과해야 한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np
import pandas as pd

THRESHOLD = 10
MASK_TOKEN = "x"
MASK_FOOTNOTE = "x: 제주빅데이터센터 반출 기준(10 미만 수치 비공개·역산 방지)에 따라 비공개 처리"


class AggKind(str, Enum):
    SIMPLE = "단순집계"
    AVERAGE = "단순집계(평균)"   # 건당·1인당 금액 등 합계÷합계. 값 자체보다 바탕 인원·건수로 판정
    APPLIED = "응용집계"


class Level(str, Enum):
    LITERAL = "literal"            # 표시되는 수치가 10 미만인 칸만 가림(반출 기준 문언 그대로)
    CONSERVATIVE = "conservative"  # + 기반 인원/건수가 10 미만인 칸의 단순집계도 가림(기본값)
    STRICT = "strict"              # + 기반 인원/건수가 10 미만이면 응용집계(비율 등)도 가림


@dataclass
class Action:
    kind: str          # primary | secondary | rollup | info
    detail: str
    cells: int = 0


# ---------------------------------------------------------------------------
# 1차 억제
# ---------------------------------------------------------------------------
def primary_mask(values: pd.DataFrame, kinds: dict[str, AggKind], support: pd.Series | pd.DataFrame | None = None,
                 level: Level = Level.CONSERVATIVE, threshold: int = THRESHOLD) -> pd.DataFrame:
    """칸별 1차 억제 여부(True=가림).

    values  : 표에 표시될 수치(행=칸, 열=지표)
    kinds   : 지표별 단순/응용 구분
    support : 칸별 기반 인원·건수(예: 이용자수). Series면 모든 지표에 공통, DataFrame이면 지표별
    """
    mask = pd.DataFrame(False, index=values.index, columns=values.columns)
    for col in values.columns:
        kind = kinds.get(col, AggKind.SIMPLE)
        v = pd.to_numeric(values[col], errors="coerce")
        value_rule = kind == AggKind.SIMPLE or (kind == AggKind.AVERAGE and level in (Level.LITERAL, Level.STRICT))
        m = (v.notna() & (v.abs() > 0) & (v.abs() < threshold)) if value_rule else pd.Series(False, index=values.index)  # 0(빈 칸)은 '-'로 표시하는 알려진 값
        support_rule = support is not None and (
            (kind == AggKind.SIMPLE and level != Level.LITERAL)
            or kind == AggKind.AVERAGE                       # 평균은 문언 모드에서도 바탕 인원 확인
            or (kind == AggKind.APPLIED and level == Level.STRICT))
        if support_rule:
            s = support[col] if isinstance(support, pd.DataFrame) else support
            s = pd.to_numeric(s, errors="coerce").reindex(values.index)
            m = m | (s.notna() & (s > 0) & (s < threshold))
        mask[col] = m.fillna(False)
    return mask


def protect_average(table: pd.DataFrame, support: pd.DataFrame | None, title: str = "",
                    level: Level = Level.CONSERVATIVE, threshold: int = THRESHOLD) -> "CheckResult":
    """평균값 표(건당·1인당 금액 등): 바탕 인원·건수가 10 미만인 칸을 가림. 더해서 합계가 되지 않으므로 2차 억제 없음."""
    num = table.apply(pd.to_numeric, errors="coerce")
    sup = support.reindex(index=table.index, columns=table.columns) if support is not None else None
    m = primary_mask(num, {c: AggKind.AVERAGE for c in table.columns}, sup, level, threshold)
    out = table.astype(object).copy()
    out[m] = MASK_TOKEN
    n = int(m.to_numpy().sum())
    actions = [Action("primary", f"바탕 인원·건수 10 미만인 평균값 {n}칸 비공개", n)] if n else []
    if sup is None:
        actions.append(Action("info", "바탕 인원·건수 정보가 없는 평균값(예: 점포당 평균) — 값 기준으로만 점검"))
    return CheckResult(title, AggKind.AVERAGE, out, n, actions, [], [MASK_FOOTNOTE] if n else [])


# ---------------------------------------------------------------------------
# 역산 감사(합계가 보이는 2차원 표)
# ---------------------------------------------------------------------------
@dataclass
class PivotMask:
    """2차원 표(행×열)와 합계 칸의 상태.

    row_totals/col_totals/grand_total: 공격자가 모르는 값(가렸거나 표시하지 않음)이면 True.
    *_sensitive: 그중 '지켜야 하는' 값(10 미만이라 가린 합계)이면 True. 표시만 안 했을 뿐 민감하지 않은
    합계는 역산되어도 문제가 아니므로 감사 대상에서 뺀다.
    """
    cells: np.ndarray                  # (R, C) bool  가린 칸(항상 민감)
    row_totals: np.ndarray             # (R,) bool
    col_totals: np.ndarray             # (C,) bool
    grand_total: bool = False
    row_sensitive: np.ndarray | None = None
    col_sensitive: np.ndarray | None = None
    grand_sensitive: bool | None = None

    def __post_init__(self):
        if self.row_sensitive is None:
            self.row_sensitive = self.row_totals.copy()
        if self.col_sensitive is None:
            self.col_sensitive = self.col_totals.copy()
        if self.grand_sensitive is None:
            self.grand_sensitive = bool(self.grand_total)

    def copy(self) -> "PivotMask":
        return PivotMask(self.cells.copy(), self.row_totals.copy(), self.col_totals.copy(), self.grand_total,
                         self.row_sensitive.copy(), self.col_sensitive.copy(), self.grand_sensitive)

    def is_sensitive(self, u: tuple[str, int, int]) -> bool:
        kind, r, c = u
        if kind == "cell":
            return True
        if kind == "row":
            return bool(self.row_sensitive[r])
        if kind == "col":
            return bool(self.col_sensitive[c])
        return bool(self.grand_sensitive)


def _unknown_index(pm: PivotMask) -> list[tuple[str, int, int]]:
    unk = [("cell", r, c) for r, c in zip(*np.nonzero(pm.cells))]
    unk += [("row", r, -1) for r in np.nonzero(pm.row_totals)[0]]
    unk += [("col", -1, c) for c in np.nonzero(pm.col_totals)[0]]
    if pm.grand_total:
        unk.append(("grand", -1, -1))
    return unk


def _equations(pm: PivotMask, unk: list[tuple[str, int, int]]) -> np.ndarray:
    """가산 관계(행합·열합·총합)를 미지수 계수 행렬로. 이미 아는 값은 우변으로 넘어가므로 계수만 필요."""
    pos = {u: i for i, u in enumerate(unk)}
    R, C = pm.cells.shape
    rows = []
    for r in range(R):                                   # Σ_c x[r,c] − RowTotal[r] = 0
        eq = np.zeros(len(unk))
        for c in range(C):
            if ("cell", r, c) in pos:
                eq[pos[("cell", r, c)]] = 1
        if ("row", r, -1) in pos:
            eq[pos[("row", r, -1)]] = -1
        rows.append(eq)
    for c in range(C):                                   # Σ_r x[r,c] − ColTotal[c] = 0
        eq = np.zeros(len(unk))
        for r in range(R):
            if ("cell", r, c) in pos:
                eq[pos[("cell", r, c)]] = 1
        if ("col", -1, c) in pos:
            eq[pos[("col", -1, c)]] = -1
        rows.append(eq)
    for kind, n in (("row", R), ("col", C)):             # Σ RowTotal − Grand = 0, Σ ColTotal − Grand = 0
        eq = np.zeros(len(unk))
        for i in range(n):
            key = (kind, i, -1) if kind == "row" else (kind, -1, i)
            if key in pos:
                eq[pos[key]] = 1
        if ("grand", -1, -1) in pos:
            eq[pos[("grand", -1, -1)]] = -1
        rows.append(eq)
    A = np.array([e for e in rows if e.any()]) if rows else np.zeros((0, len(unk)))
    return A


def derivable(pm: PivotMask) -> list[tuple[str, int, int]]:
    """가려진 칸 중 다른 칸과 합계로 정확히 역산되는 칸 목록.

    미지수 j가 유일하게 정해짐 ⇔ e_j가 방정식 행렬 A의 행공간에 있음 ⇔ A의 영공간(null space) 기저 벡터들의
    j번째 성분이 모두 0. 특이값 분해 한 번으로 모든 미지수를 판정한다(rank(A) == rank([A; e_j])와 동치).
    """
    unk = _unknown_index(pm)
    if not unk:
        return []
    A = _equations(pm, unk)
    if A.size == 0:
        return []
    _, sv, vt = np.linalg.svd(A, full_matrices=True)
    tol = max(A.shape) * np.finfo(float).eps * (sv[0] if sv.size else 1.0)
    rank = int((sv > tol).sum())
    null = vt[rank:].T                                    # (미지수 수, 영공간 차원)
    out = []
    for j, u in enumerate(unk):
        if not pm.is_sensitive(u):
            continue
        if null.shape[1] == 0 or np.abs(null[j]).max() < 1e-9:
            out.append(u)
    return out


def _candidates_for(bad: list[tuple[str, int, int]], pm: PivotMask, values: np.ndarray) -> set[tuple[int, int]]:
    """역산되는 칸들과 같은 행·열에 있는, 아직 안 가린 0이 아닌 칸(0은 '-'로 공개되는 값이라 제외)."""
    R, C = values.shape
    out: set[tuple[int, int]] = set()
    for kind, r, c in bad:
        rows = [r] if kind in ("cell", "row") and r >= 0 else []
        cols = [c] if kind in ("cell", "col") and c >= 0 else []
        for rr in rows:
            out |= {(rr, cc) for cc in range(C) if not pm.cells[rr, cc] and values[rr, cc] != 0}
        for cc in cols:
            out |= {(rr, cc) for rr in range(R) if not pm.cells[rr, cc] and values[rr, cc] != 0}
    return out


def _pair_options(bad: list[tuple[str, int, int]], pm: PivotMask, values: np.ndarray) -> list[tuple[tuple[int, int], ...]]:
    """사각형 완성 후보: 역산되는 칸 (r,c)와 같은 열(행)에 이미 가린 칸 (r2,c)가 있으면
    다른 열 c2의 (r,c2)·(r2,c2)를 함께 가리는 두 칸 묶음. 한 칸씩 고르는 탐욕법이 놓치는 최소 해(2×2 사각형)를 찾는다."""
    R, C = values.shape
    opts: set[tuple[tuple[int, int], ...]] = set()

    def ok(rr, cc):
        return not pm.cells[rr, cc] and values[rr, cc] != 0

    for kind, r, c in bad:
        if kind != "cell":
            continue
        for r2 in range(R):                       # 같은 열의 가린 칸과 짝
            if r2 != r and pm.cells[r2, c]:
                for c2 in range(C):
                    if c2 != c and ok(r, c2) and ok(r2, c2):
                        opts.add(tuple(sorted(((r, c2), (r2, c2)))))
        for c2 in range(C):                       # 같은 행의 가린 칸과 짝
            if c2 != c and pm.cells[r, c2]:
                for r2 in range(R):
                    if r2 != r and ok(r2, c) and ok(r2, c2):
                        opts.add(tuple(sorted(((r2, c), (r2, c2)))))
    return sorted(opts, key=lambda o: sum(values[x] for x in o))


def secondary_suppress(values: np.ndarray, pm: PivotMask, max_rounds: int = 500,
                       max_candidates: int = 40, max_pairs: int = 80) -> tuple[PivotMask, list[Action]]:
    """역산되는 칸이 없어질 때까지 칸을 추가로 가린다.

    매 단계 후보(같은 행·열의 한 칸, 또는 사각형을 완성하는 두 칸)를 가려 보고
    (남는 역산 칸 수, 추가로 가리는 칸 수, 가리는 값의 합)이 가장 작은 것을 고른다 = 정보 손실 최소.
    """
    pm = pm.copy()
    actions: list[Action] = []
    R, C = values.shape
    added = 0
    for _ in range(max_rounds):
        bad = derivable(pm)
        if not bad:
            break
        singles = [((rc),) for rc in sorted(_candidates_for(bad, pm, values), key=lambda rc: values[rc])[:max_candidates]]
        options = singles + _pair_options(bad, pm, values)[:max_pairs]
        if options:
            best = None
            for opt in options:
                trial = pm.copy()
                for rr, cc in opt:
                    trial.cells[rr, cc] = True
                score = (len(derivable(trial)), len(opt), float(sum(values[x] for x in opt)))
                if best is None or score < best[0]:
                    best = (score, opt)
                    if score[0] == 0 and len(opt) == 1:
                        break
            for rr, cc in best[1]:
                pm.cells[rr, cc] = True
                added += 1
            continue
        kind, r, c = bad[0]
        # 10 미만이라 가린 행(열) 합계가 '총합 − 다른 행 합계'로 역산되면, 보이는 다른 행(열) 합계 중 가장 작은 것을 숨김
        if kind == "row":
            others = [(values[i, :].sum(), i) for i in range(R) if i != r and not pm.row_totals[i]]
            if others:
                pm.row_totals[min(others)[1]] = True
                continue
        if kind == "col":
            others = [(values[:, j].sum(), j) for j in range(C) if j != c and not pm.col_totals[j]]
            if others:
                pm.col_totals[min(others)[1]] = True
                continue
        if kind == "grand":
            # 총합이 역산되면 보이는 행·열 합계 중 가장 작은 것을 가림
            totals = [(values[i, :].sum(), "row", i) for i in range(R) if not pm.row_totals[i]]
            totals += [(values[:, j].sum(), "col", j) for j in range(C) if not pm.col_totals[j]]
            if totals:
                _, which, i = min(totals)
                if which == "row":
                    pm.row_totals[i] = True
                else:
                    pm.col_totals[i] = True
                continue
        # 더 가릴 칸이 없으면 합계를 숨긴다
        if kind in ("cell", "row") and r >= 0 and not pm.row_totals[r]:
            pm.row_totals[r] = True
        elif c >= 0 and not pm.col_totals[c]:
            pm.col_totals[c] = True
        elif not pm.grand_total:
            pm.grand_total = True
        else:
            break
    if added:
        actions.append(Action("secondary", f"합계로 역산되지 않도록 {added}칸을 추가로 비공개(2차 억제)", added))
    return pm, actions


# ---------------------------------------------------------------------------
# 표 단위 검수 결과
# ---------------------------------------------------------------------------
@dataclass
class CheckResult:
    title: str
    kind: AggKind
    table: pd.DataFrame                  # 표시용(가린 칸은 'x')
    masked_cells: int
    actions: list[Action] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.violations

    def summary(self) -> str:
        head = f"[{'통과' if self.passed else '보완 필요'}] {self.title} · {self.kind.value} · 비공개 {self.masked_cells}칸"
        lines = [head] + [f"  - {a.detail}" for a in self.actions] + [f"  ! {v}" for v in self.violations] + [f"  · {n}" for n in self.notes]
        return "\n".join(lines)


def protect_pivot(pivot: pd.DataFrame, title: str = "", kind: AggKind = AggKind.SIMPLE,
                  support: pd.DataFrame | None = None, level: Level = Level.CONSERVATIVE,
                  show_totals: bool = True, totals_known: tuple[bool, bool] = (True, True),
                  threshold: int = THRESHOLD) -> CheckResult:
    """2차원 표(행×열, 합계 제외)를 반출 기준에 맞게 보호.

    totals_known=(행합계, 열합계): 표에 합계를 안 보여도 다른 곳(보고서·공개자료)에서 알 수 있다고 보면 True(보수적 기본값).
    """
    num = pivot.apply(pd.to_numeric, errors="coerce")
    vals = num.fillna(0).to_numpy(dtype=float)
    R, C = vals.shape
    actions: list[Action] = []
    sup_df = support.reindex(index=pivot.index, columns=pivot.columns) if support is not None else None

    if kind == AggKind.APPLIED:
        # 비율·증감률 등은 더해서 합계가 되지 않으므로 역산(가산 관계) 감사 대상이 아님
        table = pivot.astype(object).copy()
        n = 0
        if level == Level.STRICT and sup_df is not None:
            m = primary_mask(num, {c: kind for c in pivot.columns}, sup_df, level, threshold).to_numpy()
            for r, c in zip(*np.nonzero(m)):
                table.iat[r, c] = MASK_TOKEN
            n = int(m.sum())
            if n:
                actions.append(Action("primary", f"기반 인원 10 미만인 비율 {n}칸 비공개(엄격 모드)", n))
        actions.append(Action("info", "응용집계(비율·증감률·순위·지수 등): 반출 기준상 수치 포함 반출 가능"))
        return CheckResult(title, kind, table, n, actions, [], [MASK_FOOTNOTE] if n else [])

    pm_cells = primary_mask(num, {c: kind for c in pivot.columns}, sup_df, level, threshold).to_numpy()
    row_tot, col_tot, grand = vals.sum(axis=1), vals.sum(axis=0), vals.sum()
    rt_mask = (np.abs(row_tot) > 0) & (np.abs(row_tot) < threshold)   # 합계 칸 자체가 10 미만이면 합계도 가림
    ct_mask = (np.abs(col_tot) > 0) & (np.abs(col_tot) < threshold)
    if sup_df is not None and level != Level.LITERAL:
        rs, cs = sup_df.sum(axis=1).to_numpy(), sup_df.sum(axis=0).to_numpy()
        rt_mask |= (rs > 0) & (rs < threshold)
        ct_mask |= (cs > 0) & (cs < threshold)
    rows_known = show_totals or totals_known[0]
    cols_known = show_totals or totals_known[1]
    grand_small = bool(0 < abs(grand) < threshold)
    pm = PivotMask(
        cells=pm_cells.astype(bool),
        row_totals=(rt_mask if rows_known else np.ones(R, bool)),
        col_totals=(ct_mask if cols_known else np.ones(C, bool)),
        grand_total=grand_small if (rows_known or cols_known) else True,
        row_sensitive=rt_mask.copy(), col_sensitive=ct_mask.copy(), grand_sensitive=grand_small,
    )
    n_primary = int(pm.cells.sum())
    if n_primary:
        actions.append(Action("primary", f"10 미만 수치(또는 기반 인원 10 미만) {n_primary}칸 비공개(1차 억제)", n_primary))
    pm, sec = secondary_suppress(vals, pm)
    actions += sec

    table = pivot.astype(object).copy()
    for r, c in zip(*np.nonzero(pm.cells)):
        table.iat[r, c] = MASK_TOKEN
    if show_totals:
        table["합계"] = [MASK_TOKEN if pm.row_totals[r] else row_tot[r] for r in range(R)]
        total_row = [MASK_TOKEN if pm.col_totals[c] else col_tot[c] for c in range(C)]
        total_row.append(MASK_TOKEN if pm.grand_total else grand)
        table.loc["합계"] = total_row

    violations = audit_table(table, kind, threshold)
    if derivable(pm):
        violations.append("가려진 칸이 합계로 역산됩니다.")
    notes = [MASK_FOOTNOTE] if pm.cells.any() else []
    return CheckResult(title, kind, table, int(pm.cells.sum()), actions, violations, notes)


def protect_series(series: pd.Series, title: str = "", kind: AggKind = AggKind.SIMPLE,
                   support: pd.Series | None = None, level: Level = Level.CONSERVATIVE,
                   show_total: bool = True, threshold: int = THRESHOLD) -> CheckResult:
    """1차원 목록(범주별 값 + 합계)을 보호. 전체 합계는 어딘가에서 알 수 있다고 보고 감사(보수적)."""
    pivot = series.to_frame(name=series.name or "값")
    sup = support.to_frame(name=pivot.columns[0]) if support is not None else None
    res = protect_pivot(pivot, title, kind, sup, level, show_totals=False, totals_known=(False, True), threshold=threshold)
    if show_total and kind == AggKind.SIMPLE:
        col = pivot.columns[0]
        total = pd.to_numeric(series, errors="coerce").fillna(0).sum()
        res.table.loc["합계", col] = total if abs(total) >= threshold else MASK_TOKEN
    return res


def audit_table(table: pd.DataFrame, kind: AggKind, threshold: int = THRESHOLD) -> list[str]:
    """표시용 표에 10 미만 수치가 남아 있는지 최종 확인(단순집계만)."""
    if kind != AggKind.SIMPLE:
        return []
    bad = 0
    for v in table.to_numpy().ravel():
        if isinstance(v, str):
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if not np.isnan(fv) and 0 < abs(fv) < threshold:   # 0은 빈 칸('-')
            bad += 1
    return [f"10 미만 수치 {bad}칸이 그대로 남아 있습니다."] if bad else []


# ---------------------------------------------------------------------------
# 범주처리: 상위 단위로 묶어 10 미만 칸을 없애기
# ---------------------------------------------------------------------------
@dataclass
class Rollup:
    dim: str
    label: str                                   # 예: "연령대 통합"
    mapper: Callable[[pd.Series], pd.Series]     # 값 → 상위 범주


def rollup_until_safe(df: pd.DataFrame, dims: list[str], measures: list[str], rollups: list[Rollup],
                      check_measure: str, threshold: int = THRESHOLD, max_steps: int = 6,
                      ) -> tuple[pd.DataFrame, list[Action], int]:
    """10 미만 칸이 사라질 때까지 위반을 가장 많이 줄이는 묶기를 한 단계씩 적용(탐욕적).

    df: 차원(dims) + 가산 측정값(measures, 합계로 다시 집계 가능한 값)으로 된 긴 표
    check_measure: 10 미만 여부를 볼 측정값(예: 이용자수)
    반환: (묶은 표, 적용 내역, 남은 위반 칸 수)
    """
    cur = df.groupby(dims, dropna=False, as_index=False)[measures].sum()
    actions: list[Action] = []
    remaining = list(rollups)

    def violations(t: pd.DataFrame) -> int:
        return int((pd.to_numeric(t[check_measure], errors="coerce") < threshold).sum())

    v = violations(cur)
    steps = 0
    while v > 0 and remaining and steps < max_steps:
        best = None
        for ru in remaining:
            if ru.dim not in cur.columns:
                continue
            trial = cur.copy()
            trial[ru.dim] = ru.mapper(trial[ru.dim])
            trial = trial.groupby(dims, dropna=False, as_index=False)[measures].sum()
            tv = violations(trial)
            if best is None or tv < best[0] or (tv == best[0] and len(trial) > len(best[2])):
                best = (tv, ru, trial)
        if best is None or best[0] >= v:
            break
        tv, ru, trial = best
        actions.append(Action("rollup", f"{ru.label}(범주처리): 10 미만 칸 {v}→{tv}", v - tv))
        cur, v = trial, tv
        remaining.remove(ru)
        steps += 1
    return cur, actions, v


# 자주 쓰는 상위 범주 정의 -------------------------------------------------
def month_to_quarter(s: pd.Series) -> pd.Series:
    s = s.astype(str)
    return s.str.slice(0, 4) + "년 " + ((s.str.slice(4, 6).astype(int) - 1) // 3 + 1).astype(str) + "분기"


def card_age_band(s: pd.Series) -> pd.Series:
    m = {"10": "20대 이하", "20": "20대 이하", "30": "30·40대", "40": "30·40대",
         "50": "50·60대", "60": "50·60대", "70": "70세 이상", "NA": "알수없음"}
    return s.astype(str).map(lambda x: m.get(x, x))


def card_age_band_wide(s: pd.Series) -> pd.Series:
    """연령대 2단계 통합(원래 코드와 1단계 묶음 모두 받음)."""
    m = {"10": "39세 이하", "20": "39세 이하", "30": "39세 이하", "40": "40~59세", "50": "40~59세",
         "60": "60세 이상", "70": "60세 이상", "20대 이하": "39세 이하", "30·40대": "30~59세",
         "50·60대": "50세 이상", "70세 이상": "50세 이상", "NA": "알수없음", "알수없음": "알수없음"}
    return s.astype(str).map(lambda x: m.get(x, x))


def foreign_merge(s: pd.Series) -> pd.Series:
    return s.astype(str).map(lambda x: "외국인" if x in {"중국", "일본", "동남아", "기타외국"} else x)


def make_region_mapper(bjd_to_region: dict[str, str]) -> Callable[[pd.Series], pd.Series]:
    return lambda s: s.astype(str).map(lambda x: bjd_to_region.get(x, x))


def classify_metric(name: str) -> AggKind:
    """지표 이름으로 단순/응용 구분(보고서 생성기가 지표를 만들 때 함께 지정하는 것이 원칙).

    1인당·건당 금액(객단가)은 반출 기준의 '평균값'에 해당하므로 단순집계로 본다(보수적 해석).
    """
    applied_words = ("비율", "비중", "구성비", "증감", "증가율", "감소율", "변화율", "순위", "지수", "점유율", "%")
    return AggKind.APPLIED if any(w in name for w in applied_words) else AggKind.SIMPLE
