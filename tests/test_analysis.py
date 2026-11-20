import csv

import pytest

from dataolle.analysis import AnalysisSpec, Catalog, FactStore, run_analysis
from dataolle.analysis.josa import josa
from dataolle.sdc.engine import MASK_TOKEN, AggKind
from dataolle.spec import default_registry
from dataolle.synth.generator import generate

REG = default_registry()


@pytest.fixture(scope="module")
def cat(tmp_path_factory):
    out = tmp_path_factory.mktemp("요청파일")
    generate(out, profile="small", pop_days=1, pop_emd=["한림읍", "애월읍", "연동"], verbose=False)
    return Catalog.from_folder(out)


def test_spec_validation():
    bad = AnalysisSpec(dataset="card", measure="nope", rows="weird")
    assert any("지표" in p for p in bad.problems()) and any("차원" in p for p in bad.problems())
    assert AnalysisSpec(dataset="card", measure="amt", rows="month").problems() == []
    assert AnalysisSpec(dataset="card", measure="amt", rows="bjd", transform="yoy").problems()


def test_monthly_series_and_facts(cat):
    fs = FactStore()
    r = run_analysis(AnalysisSpec(dataset="card", measure="amt", rows="month",
                                  filters={"bjd": ["애월읍"], "biz": ["비알콜 음료점업"]}), cat, fs)
    assert list(r.table.index) == ["202510", "202511", "202512"]
    assert r.check.passed and r.check.masked_cells == 0
    assert r.sources and r.period == ("202510", "202512") and r.synthetic
    text = " ".join(fs.render(s) for s in r.sentences)
    assert "억 원" in text and "{{" not in text
    assert r.citation().startswith("제주빅데이터센터, 카드 이용 현황(읍면동)_v02(")


def test_wolpyeong_dong_is_split_by_city(cat):
    vals = cat.values("card", "bjd")
    assert "월평동(제주시)" in vals and "월평동(서귀포시)" in vals and "월평동" not in vals


def test_week_table_excludes_holiday_rows(cat):
    f = {"bjd": ["애월읍"], "biz": ["비알콜 음료점업"]}
    total = run_analysis(AnalysisSpec(dataset="card", measure="cnt", filters=f, period=["202512", "202512"]), cat).table.iloc[0, 0]
    wk = run_analysis(AnalysisSpec(dataset="card_week", measure="cnt", rows="week", filters=f, period=["202512", "202512"]), cat)
    assert "공휴일" not in wk.table.index
    assert wk.table.iloc[:, 0].sum() == pytest.approx(total)
    assert any("공휴일" in n for n in wk.notes)


def test_share_transform_sums_to_100(cat):
    r = run_analysis(AnalysisSpec(dataset="card_time", measure="cnt", rows="time", transform="share",
                                  filters={"bjd": ["애월읍"], "biz": ["비알콜 음료점업"]}), cat)
    assert r.kind == AggKind.APPLIED and r.unit == "%"
    assert r.table.iloc[:, 0].sum() == pytest.approx(100, abs=0.01)
    assert list(r.table.index) == ["오전", "점심", "오후", "저녁", "밤"]


def test_top_n_groups_rest_as_other(cat):
    r = run_analysis(AnalysisSpec(dataset="card", measure="amt", rows="bjd", top_n=5, filters={"biz": ["비알콜 음료점업"]}), cat)
    assert list(r.table.index)[-1] == "기타" and len(r.table) == 6
    fs = FactStore()
    r2 = run_analysis(AnalysisSpec(dataset="card", measure="amt", rows="bjd", top_n=5, filters={"biz": ["비알콜 음료점업"]}), cat, fs)
    assert all("기타" not in fs.render(s).split("가장")[0] for s in r2.sentences)


def test_support_uses_per_month_maximum(tmp_path):
    """매달 같은 5명이 쓰면 3개월 합은 15명이지만 실제 이용자는 5명일 수 있음 → 평균값은 가려야 함."""
    spec = REG.get("1-08")
    path = tmp_path / "Results_20261113_100000.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(spec.csv_columns)
        for ym in ("202510", "202511", "202512"):
            w.writerow([ym, "제주시 서부", "제주시", "읍면", "추자면", "I56220", "비알콜 음료점업", "제주도민", "도민", "40", "여성", "5", "9", "90000"])
            w.writerow([ym, "제주시 서부", "제주시", "읍면", "한림읍", "I56220", "비알콜 음료점업", "제주도민", "도민", "40", "여성", "60", "90", "900000"])
    c = Catalog.from_folder(tmp_path)
    r = run_analysis(AnalysisSpec(dataset="card", measure="amt_per_cnt", rows="bjd"), c)
    assert r.display.loc["추자면"].iloc[0] == MASK_TOKEN     # 합계 15명이어도 월 최대 5명 → 비공개
    assert r.display.loc["한림읍"].iloc[0] != MASK_TOKEN


def test_josa():
    assert josa("인구", "은") == "인구는" and josa("이용금액", "은") == "이용금액은"
    assert josa("'오후'", "이") == "'오후'가" and josa("2025년 1월", "과") == "2025년 1월과"
