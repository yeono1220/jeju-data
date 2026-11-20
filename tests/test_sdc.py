import numpy as np
import pandas as pd
import pytest

from dataolle.sdc.engine import (MASK_TOKEN, AggKind, Level, PivotMask, Rollup, card_age_band, card_age_band_wide,
                                 classify_metric,
                                 derivable, foreign_merge, month_to_quarter, primary_mask, protect_pivot,
                                 protect_series, rollup_until_safe, secondary_suppress, _equations, _unknown_index)


def nullspace_determined(pm: PivotMask) -> list:
    """독립 검증: 해공간(null space)의 모든 기저가 0인 미지수는 값이 유일하게 정해짐."""
    unk = _unknown_index(pm)
    A = _equations(pm, unk)
    if A.size == 0:
        return []
    u, s, vt = np.linalg.svd(A)
    rank = int((s > 1e-9).sum())
    null = vt[rank:]
    if null.shape[0] == 0:
        return list(unk)
    return [unk[j] for j in range(len(unk)) if np.all(np.abs(null[:, j]) < 1e-9)]


def test_primary_mask_value_and_support_rules():
    vals = pd.DataFrame({"이용건수": [5, 50, 120], "이용금액": [40000, 300000, 900000]})
    support = pd.Series([3, 8, 40])
    kinds = {"이용건수": AggKind.SIMPLE, "이용금액": AggKind.SIMPLE}
    lit = primary_mask(vals, kinds, support, Level.LITERAL)
    assert lit["이용건수"].tolist() == [True, False, False]
    assert lit["이용금액"].tolist() == [False, False, False]
    cons = primary_mask(vals, kinds, support, Level.CONSERVATIVE)
    assert cons["이용건수"].tolist() == [True, True, False]
    assert cons["이용금액"].tolist() == [True, True, False]   # 이용자 10명 미만 칸의 금액도 가림


def test_applied_metrics_not_masked_by_value():
    piv = pd.DataFrame({"2024": [3.2, -1.5], "2025": [12.0, 4.4]}, index=["애월읍", "한림읍"])
    res = protect_pivot(piv, "매출 증감률(%)", AggKind.APPLIED)
    assert res.passed and res.masked_cells == 0
    assert (res.table == piv).all().all()


def test_series_single_small_cell_gets_secondary():
    s = pd.Series({"20대": 120, "30대": 340, "40대": 7, "50대": 210}, name="이용자수")
    res = protect_series(s, "연령대별 이용자수")
    assert res.passed
    masked = [k for k, v in res.table["이용자수"].items() if v == MASK_TOKEN]
    assert "40대" in masked and len(masked) >= 2            # 합계에서 빼는 역산 방지
    assert res.table.loc["합계", "이용자수"] == 677
    assert "20대" in masked                                   # 가장 작은 값을 추가로 가림


def test_pivot_with_totals_is_not_derivable():
    piv = pd.DataFrame([[120, 45, 8], [300, 210, 90], [50, 30, 25]],
                       index=["오전", "점심", "저녁"], columns=["도민", "개별", "단체"])
    res = protect_pivot(piv, "시간대×관광구분 이용건수")
    assert res.passed, res.summary()
    assert res.table.loc["오전", "단체"] == MASK_TOKEN
    flat = res.table.drop(index="합계", columns="합계").to_numpy().ravel()
    assert (flat == MASK_TOKEN).sum() >= 4                    # 사각형(행 2 × 열 2) 이상으로 가려야 안전


def test_small_total_is_masked():
    piv = pd.DataFrame([[3, 2], [400, 500]], index=["추자면", "애월읍"], columns=["남성", "여성"])
    res = protect_pivot(piv, "성별 이용자수")
    assert res.table.loc["추자면", "합계"] == MASK_TOKEN
    assert res.passed


def test_bridge_pattern_detected_and_fixed():
    # 두 사이클(2x2)이 다리 칸 (1,2)로 이어진 패턴: 모든 행·열에 가린 칸이 2개 이상이어도 다리 칸은 역산됨
    cells = np.zeros((4, 4), bool)
    for r, c in [(0, 0), (0, 1), (1, 0), (1, 1), (2, 2), (2, 3), (3, 2), (3, 3), (1, 2)]:
        cells[r, c] = True
    pm = PivotMask(cells, np.zeros(4, bool), np.zeros(4, bool), False)
    assert ("cell", 1, 2) in derivable(pm)
    assert set(derivable(pm)) == set(nullspace_determined(pm))
    vals = np.arange(16, dtype=float).reshape(4, 4) + 20
    fixed, actions = secondary_suppress(vals, pm)
    assert derivable(fixed) == [] and nullspace_determined(fixed) == []
    assert actions and actions[0].kind == "secondary"


@pytest.mark.parametrize("seed", range(25))
def test_random_tables_never_leak(seed):
    rng = np.random.default_rng(seed)
    R, C = rng.integers(2, 7), rng.integers(2, 7)
    vals = rng.integers(10, 400, size=(R, C)).astype(float)
    small = rng.random((R, C)) < 0.2
    vals[small] = rng.integers(1, 10, size=small.sum())
    piv = pd.DataFrame(vals, index=[f"r{i}" for i in range(R)], columns=[f"c{j}" for j in range(C)])
    res = protect_pivot(piv, "랜덤 표")
    assert res.passed, res.summary()
    body = res.table.drop(index="합계", columns="합계")
    # 10 미만 원래 값은 모두 가려짐
    for (i, j) in zip(*np.nonzero(vals < 10)):
        assert body.iat[i, j] == MASK_TOKEN
    # 독립 검증: 표시된 정보(보이는 칸·합계)로 가린 칸을 하나도 확정할 수 없음
    pm = PivotMask((body == MASK_TOKEN).to_numpy(),
                   (res.table["합계"].iloc[:-1] == MASK_TOKEN).to_numpy(),
                   (res.table.loc["합계"].iloc[:-1] == MASK_TOKEN).to_numpy(),
                   res.table.loc["합계", "합계"] == MASK_TOKEN)
    assert nullspace_determined(pm) == []


def test_rollup_until_safe_reduces_small_cells():
    df = pd.DataFrame({
        "base_year_month": ["202501", "202502", "202503"] * 4,
        "age": ["20"] * 3 + ["30"] * 3 + ["60"] * 3 + ["70"] * 3,
        "user_cnt": [4, 5, 3, 40, 42, 39, 6, 7, 5, 2, 3, 2],
        "use_amt": [1, 1, 1, 9, 9, 9, 2, 2, 2, 1, 1, 1],
    })
    # 1단계 묶음만으로는 70세 이상(7명)이 남음 → 남은 칸 수를 정직하게 돌려줘야 함
    one_level = [Rollup("base_year_month", "월→분기 통합", month_to_quarter), Rollup("age", "연령대 통합", card_age_band)]
    _, _, left1 = rollup_until_safe(df, ["base_year_month", "age"], ["user_cnt", "use_amt"], one_level, "user_cnt")
    assert left1 == 1
    # 2단계 묶음까지 허용하면 모두 10 이상
    two_level = one_level + [Rollup("age", "연령대 넓게 통합", card_age_band_wide)]
    out, actions, left = rollup_until_safe(df, ["base_year_month", "age"], ["user_cnt", "use_amt"], two_level, "user_cnt")
    assert left == 0
    assert all(a.kind == "rollup" for a in actions)
    assert (out["user_cnt"] >= 10).all()


def test_helpers():
    assert month_to_quarter(pd.Series(["202501", "202512"])).tolist() == ["2025년 1분기", "2025년 4분기"]
    assert foreign_merge(pd.Series(["중국", "제주도민"])).tolist() == ["외국인", "제주도민"]
    assert classify_metric("매출 증감률(%)") == AggKind.APPLIED
    assert classify_metric("이용금액 합계") == AggKind.SIMPLE
    assert classify_metric("1인당 이용금액") == AggKind.SIMPLE   # 평균값 → 단순집계(보수적)


def test_series_without_small_cells_is_untouched():
    s = pd.Series({"202501": 120.0, "202502": 340.0, "202503": 210.0}, name="이용금액")
    res = protect_series(s, "월별 이용금액")
    assert res.masked_cells == 0 and res.passed
    assert not res.actions


def test_zero_cells_are_known_and_not_masked():
    # 법인(미상) 열은 '알수없음' 연령에만 값이 있음 → 나머지 0은 '-'로 공개되는 값
    piv = pd.DataFrame([[120, 300, 0], [5, 40, 0], [30, 25, 146]],
                       index=["20대", "30대", "알수없음"], columns=["도민", "개별", "미상"])
    res = protect_pivot(piv, "연령×관광구분")
    body = res.table.drop(index="합계", columns="합계")
    assert body.loc["20대", "미상"] == 0 and body.loc["30대", "미상"] == 0   # 0은 가리지 않음
    assert body.loc["30대", "도민"] == MASK_TOKEN
    pm = PivotMask((body == MASK_TOKEN).to_numpy(),
                   (res.table["합계"].iloc[:-1] == MASK_TOKEN).to_numpy(),
                   (res.table.loc["합계"].iloc[:-1] == MASK_TOKEN).to_numpy(),
                   res.table.loc["합계", "합계"] == MASK_TOKEN)
    assert nullspace_determined(pm) == [] and res.passed


def test_two_small_cells_in_one_column_need_only_a_rectangle():
    """같은 열의 작은 칸 두 개는 다른 열 두 칸만 더 가리면 된다(2×2 사각형). 예전 탐욕법은 10칸을 더 가렸다."""
    df = pd.DataFrame({"남성": [93, 1666, 1584, 1220, 922, 484, 136, 199],
                       "여성": [144, 2051, 2663, 1930, 1451, 709, 240, 229],
                       "알수없음": [4, 81, 115, 76, 61, 22, 9, 515]},
                      index=["19세 이하", "20대", "30대", "40대", "50대", "60대", "70세 이상", "알수없음"])
    res = protect_pivot(df, "연령×성별")
    assert res.passed
    assert res.masked_cells == 4
    body = res.table.drop(index="합계", columns="합계")
    assert body.loc["19세 이하", "알수없음"] == MASK_TOKEN and body.loc["70세 이상", "알수없음"] == MASK_TOKEN
    assert body.loc["19세 이하", "남성"] == MASK_TOKEN and body.loc["70세 이상", "남성"] == MASK_TOKEN
