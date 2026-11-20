import csv
from pathlib import Path

import pytest

from dataolle.ingest.inventory import (check_plan, make_plan, month_range, operator_request_text,
                                       plan_progress, scan_folder)
from dataolle.ingest.recognizer import match_header, recognize
from dataolle.spec import default_registry

REG = default_registry()


def write_center_csv(path: Path, header: list[str], rows: list[list[str]], encoding="utf-8", quote_all=True):
    with open(path, "w", newline="", encoding=encoding) as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL if quote_all else csv.QUOTE_MINIMAL)
        w.writerow(header)
        w.writerows(rows)
    return path


def card_row(ym="202501", bjd="애월읍"):
    return [ym, "제주시 서부", "제주시", "읍면", bjd, "I56220", "비알콜 음료점업", "제주도민", "도민", "30", "여성", "12", "20", "130000"]


def test_exact_header_1_08(tmp_path):
    h = list(REG.get("1-08").csv_columns)
    p = write_center_csv(tmp_path / "Results_20261109_111652.csv", h, [card_row(), card_row(bjd="한림읍")])
    rec = recognize(p)
    assert rec.status == "ok"
    assert rec.dataset_id == "1-08"
    assert rec.best.exact
    assert rec.months == ["202501"]
    assert rec.row_count == 2
    assert rec.fingerprint.center_filename and rec.fingerprint.all_quoted
    assert rec.fingerprint.downloaded_at.year == 2026
    assert not rec.warnings


def test_week_vs_time_tables(tmp_path):
    for ds in ("1-12", "1-14"):
        h = list(REG.get(ds).csv_columns)
        row = ["202501", "제주시 서부", "애월읍", "I56220", "비알콜 음료점업", "제주도민", "도민",
               "월요일" if ds == "1-12" else "오후", "30", "여성", "3", "3", "21000"]
        p = write_center_csv(tmp_path / f"{ds}.csv", h, [row])
        assert recognize(p).dataset_id == ds


def test_partition_columns_present_still_exact(tmp_path):
    h = list(REG.get("1-08").columns)  # pt_year, pt_month 포함
    row = card_row()
    row = row[:1] + ["2025", "1"] + row[1:]
    p = write_center_csv(tmp_path / "with_pt.csv", h, [row])
    rec = recognize(p)
    assert rec.dataset_id == "1-08" and rec.best.exact


def test_partial_columns_flagged(tmp_path):
    h = ["base_year_month", "bjd", "biz_nm", "use_amt"]
    p = write_center_csv(tmp_path / "partial.csv", h, [["202501", "애월읍", "비알콜 음료점업", "1000"]])
    rec = recognize(p)
    assert rec.status == "ambiguous"
    assert "1-08" in [c.dataset_id for c in rec.candidates] or "1-07" in [c.dataset_id for c in rec.candidates]


def test_excel_resaved_warnings(tmp_path):
    h = list(REG.get("1-08").csv_columns)
    p = write_center_csv(tmp_path / "excel.csv", h, [card_row()], encoding="cp949", quote_all=False)
    rec = recognize(p)
    assert rec.dataset_id == "1-08"
    joined = " ".join(rec.warnings)
    assert "CP949" in joined and "따옴표" in joined


def test_identical_headers_disambiguated_by_values(tmp_path):
    h = list(REG.get("1-02").csv_columns)
    p = write_center_csv(tmp_path / "cn.csv", h, [["201701", "제주", "면세점", "중국", "500000", "250000"]])
    assert recognize(p).dataset_id == "1-05"
    p2 = write_center_csv(tmp_path / "kr.csv", h, [["201701", "제주", "면세점", "내국인", "500000", "250000"]])
    assert recognize(p2).dataset_id == "1-02"


def test_identical_headers_unresolvable(tmp_path):
    h = list(REG.get("5-05").csv_columns)
    p = write_center_csv(tmp_path / "food.csv", h, [["20250101", "A", "T1", "1200", "N", "100", "카드", "D1"]])
    rec = recognize(p)
    assert rec.status == "ambiguous" and rec.dataset_id is None
    assert {c.dataset_id for c in rec.candidates} >= {"5-05", "5-06"}


def test_multi_month_file_warning(tmp_path):
    h = list(REG.get("1-08").csv_columns)
    p = write_center_csv(tmp_path / "Results_20261109_120000.csv", h, [card_row("202501"), card_row("202502")])
    rec = recognize(p)
    assert rec.months == ["202501", "202502"]
    assert any("개월" in w for w in rec.warnings)


def test_unknown_file(tmp_path):
    p = write_center_csv(tmp_path / "x.csv", ["foo", "bar"], [["1", "2"]])
    assert recognize(p).status == "unknown"


def test_match_header_ranks_exact_first():
    cands = match_header(list(REG.get("4-11").csv_columns))
    assert cands[0].dataset_id == "4-11" and cands[0].exact


def test_inventory_plan_and_operator_request(tmp_path):
    h = list(REG.get("1-08").csv_columns)
    for i, ym in enumerate(["202501", "202502", "202504"]):
        write_center_csv(tmp_path / f"Results_20261109_1100{i:02d}.csv", h, [card_row(ym)])
    write_center_csv(tmp_path / "Results_20261109_110099.csv", h, [card_row("202502")])  # 중복
    write_center_csv(tmp_path / "memo.csv", ["a", "b"], [["1", "2"]])                       # 무관 파일
    inv = scan_folder(tmp_path)
    assert inv.months("1-08") == ["202501", "202502", "202504"]
    assert inv.missing("1-08", "202501", "202504") == ["202503"]
    dups = inv.duplicates()
    assert len(dups) == 1 and dups[0][1] == "202502"
    assert len(inv.unknown) == 1
    assert len(inv.files_for("1-08")) == 3

    plan = check_plan(make_plan(["1-08"], "202501", "202504", {"bjd": "애월읍"}), inv)
    assert plan_progress(plan) == (3, 4)
    text = operator_request_text(plan)
    assert "2025.03" in text and "법정동=애월읍" in text


def test_month_range_crosses_year():
    assert month_range("202511", "202602") == ["202511", "202512", "202601", "202602"]
