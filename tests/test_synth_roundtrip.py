"""가상데이터 생성 → 자동 인식 → 적재 → 반출 검수까지 한 번에 도는지 확인하는 통합 테스트."""
import json

import duckdb
import pandas as pd
import pytest

from dataolle.ingest.inventory import scan_folder
from dataolle.ingest.loader import load_dataset
from dataolle.sdc.engine import AggKind, protect_pivot
from dataolle.spec import default_registry
from dataolle.synth.generator import generate

REG = default_registry()


@pytest.fixture(scope="module")
def demo_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("요청파일")
    generate(out, profile="small", pop_days=1, pop_emd=["한림읍", "애월읍", "연동"], verbose=False)
    return out


def test_files_look_like_center_csv(demo_dir):
    manifest = json.loads((demo_dir / "_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        raw = (demo_dir / item["file"]).read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf")                     # BOM 없음
        header, first = raw.decode("utf-8").splitlines()[:2]
        cols = [c.strip('"') for c in header.split(",")]
        assert "pt_year" not in cols and "pt_month" not in cols        # 파티션 컬럼 없음
        assert cols == list(REG.get(item["dataset_id"]).csv_columns)   # 명세서 컬럼 순서 그대로
        assert first.startswith('"') and first.endswith('"')           # 모든 값 큰따옴표


def test_recognizer_identifies_every_generated_file(demo_dir):
    manifest = json.loads((demo_dir / "_manifest.json").read_text(encoding="utf-8"))
    truth = {m["file"]: (m["dataset_id"], m["month"]) for m in manifest["files"]}
    inv = scan_folder(demo_dir)
    assert not inv.unknown and not inv.ambiguous
    for rec in inv.files:
        ds, ym = truth[rec.path.name]
        assert rec.dataset_id == ds, rec.summary()
        assert rec.months == [ym]
        assert rec.fingerprint.all_quoted and rec.fingerprint.center_filename
        assert not rec.warnings, rec.warnings


def test_load_and_protect_end_to_end(demo_dir):
    inv = scan_folder(demo_dir)
    con = duckdb.connect()
    t = load_dataset(con, "1-08", inv.files_for("1-08"))
    types = dict(con.execute(f"SELECT column_name, column_type FROM (DESCRIBE {t})").fetchall())
    assert types["use_amt"] == "BIGINT" and types["user_cnt"] == "BIGINT" and types["bjd"] == "VARCHAR"
    # NA → NULL (법인 카드의 연령)
    assert con.execute(f"SELECT count(*) FROM {t} WHERE user_type='법인' AND age IS NULL").fetchone()[0] > 0
    # 애월읍 카페: 연령대 × 관광구분 이용자수 표를 만들고 반출 검수
    df = con.execute(f"""
        SELECT coalesce(age,'NA') AS age, tour_type, sum(user_cnt) AS users
        FROM {t} WHERE bjd='애월읍' AND biz_cd='I56220' AND base_year_month='202512'
        GROUP BY 1, 2""").df()
    piv = df.pivot_table(index="age", columns="tour_type", values="users", aggfunc="sum", fill_value=0)
    res = protect_pivot(piv, "애월읍 비알콜 음료점업 연령대×관광구분 이용자수", AggKind.SIMPLE)
    assert res.passed, res.summary()

    p = load_dataset(con, "4-11", inv.files_for("4-11"))
    types = dict(con.execute(f"SELECT column_name, column_type FROM (DESCRIBE {p})").fetchall())
    assert types["visit_pop_cnt"] == "DOUBLE" and types["age"] == "VARCHAR"
    assert con.execute(f"SELECT count(DISTINCT emd_nm) FROM {p}").fetchone()[0] == 3


def test_week_table_holiday_double_counting(demo_dir):
    """요일별 표(1-12)의 '공휴일'은 원래 요일에도 들어 있어 합치면 1-08보다 커짐 → 함정 재현 확인."""
    inv = scan_folder(demo_dir)
    con = duckdb.connect()
    t8 = load_dataset(con, "1-08", inv.files_for("1-08", ["202512"]))
    t12 = load_dataset(con, "1-12", inv.files_for("1-12", ["202512"]))
    base = con.execute(f"SELECT sum(use_cnt) FROM {t8}").fetchone()[0]
    weekdays = con.execute(f"SELECT sum(use_cnt) FROM {t12} WHERE week <> '공휴일'").fetchone()[0]
    with_hol = con.execute(f"SELECT sum(use_cnt) FROM {t12}").fetchone()[0]
    assert weekdays == base                    # 월~일 합 = 월 합계
    assert with_hol > base                     # 공휴일까지 더하면 중복
