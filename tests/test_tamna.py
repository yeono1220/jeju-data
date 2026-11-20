import pandas as pd
import pytest

from dataolle.ai.planner import rule_plan
from dataolle.analysis import Catalog
from dataolle.courses.tamna import TamnaCourse, TamnaInput, market_day_ratio
from dataolle.synth.generator import generate


@pytest.fixture(scope="module")
def cat(tamna_cat):
    return tamna_cat


def test_market_day_cycle_detection():
    idx = [f"202503{d:02d}" for d in range(1, 32)]
    vals = [300 if d % 10 in (2, 7) else 100 for d in range(1, 32)]
    cyc, ratio = market_day_ratio(pd.Series(vals, index=idx, dtype=float))
    assert set(cyc) == {2, 7} and ratio == pytest.approx(3.0)
    flat = pd.Series([100.0] * 31, index=idx)
    assert market_day_ratio(flat)[0] is None


def test_tamna_course_finds_rain_and_policy_effects(cat, tmp_path):
    out = TamnaCourse(cat, tmp_path).run(TamnaInput(event="20250601", event_name="가상 정책"), make_pdf=False, make_docx=False)
    heads = [s.heading for s in out.report.sections]
    assert any("사각지대" in h for h in heads) and any("날씨" in h for h in heads) and any("전후" in h for h in heads)
    rain = [f for f in out.facts if f.label == "비 오는 날 전통시장 탐나는전 사용금액 변화"][0]
    assert -32 < rain.value < -14                       # 심어 둔 −22%
    e = out.report.extra["effect"]
    assert 4 < e["did_pct"] < 20                         # 심어 둔 +12%(날씨 보정·명절 주 제외 후)
    assert all(r.check.passed for r in out.report.results)
    assert "{{" not in out.report.extra["plan_text"]


def test_tamna_planner_routing():
    p = rule_plan("2025년 6월 이후 전통시장 탐나는전 효과가 있었나요?")
    assert p.course == "tamna" and p.market["event"] == "202506" and p.market["scope"] == "all"
    assert rule_plan("애월읍 탐나는전 분석").market == {"scope": "emd", "value": "애월읍", "event": None}


def test_dongne_sections_and_types(cat, tmp_path):
    out = TamnaCourse(cat, tmp_path).run(TamnaInput(), make_pdf=False, make_docx=False)
    heads = [s.heading for s in out.report.sections]
    for key in ("원도심·신도심 상권", "인구 구조와 동네 유형", "학령인구 감소 경보", "동네 유형별 처방"):
        assert any(key in h for h in heads), key
    # 절 번호는 1부터 빠짐없이
    nums = [int(h.split(".")[0]) for h in heads if h[0].isdigit()]
    assert nums == list(range(1, len(nums) + 1))
    types = out.report.extra["dongne_types"]
    # 가상 세계에 심어 둔 경향: 원도심 고령·상권 위축, 외곽 섬 지역 아이 감소, 신도심 성장
    assert "일도1동" in types.get("고령·상권 위축", [])
    assert {"추자면", "우도면"} & set(types.get("고령·상권 위축", []) + types.get("아이 감소", []))
    assert set(types.get("성장", [])) & {"연동", "외도동", "삼양동", "대륜동", "동홍동"}
    f = {x.label: x for x in out.facts}
    old_pen = [v for k, v in f.items() if k.startswith("원도심 탐나는전 침투지수")][0]
    assert old_pen.value > 105                                 # 원도심 침투 1.2배로 심어 둠
    rel = [v for k, v in f.items() if k.startswith("원도심 도민 카드 소비 상대 변화")][0]
    assert rel.value < -2                                      # 원도심 월 −0.5% 추세
    assert all(r.check.passed for r in out.report.results)
    text = " ".join(getattr(b_, "text", "") for s in out.report.sections for b_ in s.blocks)
    assert "{{" not in text and "제주 전체 전체" not in text


def test_dongne_diag_card_for_one_emd(cat, tmp_path):
    out = TamnaCourse(cat, tmp_path).run(TamnaInput(scope="emd", value="일도일동"), make_pdf=False, make_docx=False)
    first = out.report.sections[1]
    assert first.heading == "1. 동네 진단 카드"
    tbl = [b_ for b_ in first.blocks if hasattr(b_, "df")][0].df
    assert list(tbl.columns) == ["이 동네", "원도심 평균", "제주 전체"]
    assert "일도1동" in out.report.extra["diag_unit"] or "진단" in out.report.extra["diag_unit"]


def test_unit_mapping_and_pop_union(cat):
    from dataolle.analysis.semantic import BJD_UNIT, UNIT_GROUP
    from dataolle.courses.dongne import unit_of_name
    assert unit_of_name("월평동(제주시)") == "아라동" and unit_of_name("월평동(서귀포시)") == "대천동"
    assert unit_of_name("서귀동") == "송산·정방·중앙·천지동" and UNIT_GROUP["송산·정방·중앙·천지동"] == "원도심"
    assert unit_of_name("일도1동") == "일도1동" and BJD_UNIT[("제주시", "해안동")] == "노형동"
    # 유동인구 v01(4-10)과 v02(4-11)를 한 표로 잇고, 출처는 데이터셋별로 따로 적는다
    from dataolle.analysis.engine import run_analysis
    from dataolle.analysis.spec import AnalysisSpec
    r = run_analysis(AnalysisSpec(dataset="pop", measure="resd_night", rows="year", totals=False), cat)
    names = [n for n, *_ in r.source_parts()]
    assert any("v01" in n for n in names) and any("v02" in n for n in names)
    assert all(files for *_, files in r.source_parts())
    assert any("통신사" in n for n in r.notes)


def test_report_sources_use_actual_months_and_files(cat, tmp_path):
    """출처 줄의 기간·파일은 분석 기간이 아니라 데이터셋마다 실제로 불러온 달·파일이어야 한다(반출 신청서의 다운로드 기록 대조용)."""
    from pathlib import Path

    from dataolle.report.export_pack import sources_block
    from dataolle.spec import default_registry

    out = TamnaCourse(cat, tmp_path).run(TamnaInput(), make_pdf=False, make_docx=False)
    inv, reg = cat.inventory, default_registry()
    by_name = {reg.get(ds).name_ko: ds for ds in inv.by_dataset}
    srcs = out.report.sources()
    seen = set()
    for name, a, b, files in srcs:
        ds = by_name.get(name)
        if ds is None:                                   # 센터 밖 공개 자료
            continue
        seen.add(ds)
        months = [m[:6] for m in inv.months(ds)]
        assert months[0] <= a <= b <= months[-1], (name, a, b, months)          # 없는 달을 적지 않는다
        own = {Path(f).name for f in inv.files_for(ds)}
        assert files and set(files) <= own, (name, files)                        # 다른 판(v01·v02)의 파일이 섞이지 않는다
    assert {"1-32", "5-02", "1-08"} <= seen
    assert len(srcs) == len({n for n, *_ in srcs})                               # 데이터셋마다 한 줄
    assert "사용 파일" in sources_block(out.report)
