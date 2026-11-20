import pytest

from dataolle.analysis import Catalog
from dataolle.courses.before_after import BeforeAfterCourse, BeforeAfterInput, did
from dataolle.courses.status import StatusCourse
from dataolle.synth.generator import generate
from dataolle.synth.world import DEMO_EVENTS


@pytest.fixture(scope="module")
def card12(tmp_path_factory):
    out = tmp_path_factory.mktemp("card12")
    generate(out, profile="card12", verbose=False)
    return Catalog.from_folder(out)


def test_did_basic():
    import pandas as pd
    idx = ["202501", "202502", "202503", "202504", "202505", "202506"]
    t = pd.Series([100, 100, 100, 120, 120, 120], index=idx)
    c = pd.Series([50, 50, 50, 50, 50, 50], index=idx)
    assert did(t, c, "202504") == pytest.approx(0.18232, abs=1e-4)      # log(1.2)


def test_before_after_recovers_planted_effect(card12, tmp_path):
    ev = DEMO_EVENTS[0]
    out = BeforeAfterCourse(card12, tmp_path).run(
        BeforeAfterInput(treat=[ev["bjd"]], event=ev["from"], biz="비알콜 음료점업"), make_pdf=False, make_docx=False)
    e = out.report.extra["effect"]
    assert e["did_pct"] > 5 and e["p"] <= 0.1 and e["placebos"] >= 20
    assert "{{" not in out.report.extra["methods_text"] and "이중차분" in out.report.extra["methods_text"]
    assert all(r.check.passed for r in out.report.results)


def test_before_after_needs_enough_months(card12, tmp_path):
    with pytest.raises(ValueError):
        BeforeAfterCourse(card12, tmp_path).run(BeforeAfterInput(treat=["한림읍"], event="202502"), make_pdf=False, make_docx=False)


def test_status_course_runs_with_card_only(card12, tmp_path):
    out = StatusCourse(card12, tmp_path).run_status("한림읍", make_pdf=False, make_docx=False)
    heads = [s.heading for s in out.report.sections]
    assert heads[0] == "요약" and any("업종" in h for h in heads)
    assert "유동인구" in out.skipped
    assert all(r.check.passed for r in out.report.results)
