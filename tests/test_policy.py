"""처방 설계(적립 예산 배분 시뮬레이터)와 실행물 묶음."""
import json

import pandas as pd
import pytest

from dataolle.courses import policy as PL


def _seg():
    """2025년 12개월 × 28일, 원도심(일도1동) 일반·전통시장 + 신도심(노형동) 일반. 하루 사용액 고정."""
    rows = []
    for m in range(1, 13):
        for d in range(1, 29):
            day = f"2025{m:02d}{d:02d}"
            rows += [(day, "노형동", "일반 가맹점", "시장 외", 100.0, 50),
                     (day, "일도1동", "일반 가맹점", "시장 외", 50.0, 50),
                     (day, "일도1동", "전통시장", "동문공설시장", 10.0, 50)]
    return pd.DataFrame(rows, columns=["d", "hd", "mflag", "market", "amt", "cnt"])


def _frame(**kw):
    seg = _seg()
    rain = pd.Series({f"2025{m:02d}03": 12.0 for m in range(1, 13)})
    return PL.prepare(seg, cycles=[("동문공설시장", (2, 7), 2.0)], rain=rain, focus_units={"일도1동"}, **kw)


def test_parse_policy_reads_budget_period_goal_locally():
    q = PL.parse_policy("예산 400억 원으로 탐나는전 적립을 1년 내내 이어 가려면? 원도심 전통시장 위주로")
    assert q["budget"] == pytest.approx(4e10) and q["months"] == 12 and q["goal"] == "revive"
    q = PL.parse_policy("3월부터 연말까지 캐시백 예산 1.2조를 사각지대 중심으로")
    assert q["start"] == 3 and q["months"] == 10 and q["budget"] == pytest.approx(1.2e12) and q["goal"] == "blind"
    q = PL.parse_policy("2025년 6월부터 한 전통시장 추가 적립 효과를 확인해 줘", effect_question=True)
    assert "start" not in q and "year" not in q and "budget" not in q
    assert PL.parse_policy("탐나는전 현황 알려 줘") == {}


def test_frame_flags_and_typical_year():
    fr = _frame()
    df = fr.df
    assert fr.year == 2026 and fr.months == 12 and not fr.est_months
    assert df["mday"].sum() == 12 * 6                     # 2·7·12·17·22·27일(28일까지) × 12개월, 시장 행만
    assert df.loc[df["mday"], "dd"].mod(5).eq(2).all() and df.loc[df["mday"], "mkt"].all()
    assert df.loc[df["rain"], "dd"].eq(3).all()
    assert df.loc[df["fest"], "pm"].isin([2, 9]).all()  # 2026년 설·추석
    assert df.loc[df["hd"] == "일도1동", "rev"].all() and not df.loc[df["hd"] == "노형동", "rev"].any()


def test_budget_math_exhaustion_and_solved_rates():
    fr = _frame()
    U = fr.usage_window                                    # 160 × 28 × 12
    assert U == pytest.approx(160 * 28 * 12)
    B = 3370.0
    sc = PL.design(fr, B)
    A, Bs, C = sc["A"], sc["B"], sc["C"]
    # A: 하루 16(10%), 2·9월은 32 → 6월 말 누적 3,136, 7월 15일에 3,370 도달
    assert A.exhaust == (7, 15) and A.lasting == pytest.approx(6 + (14 + 0.625) / 28, abs=1e-6)
    assert A.spent == pytest.approx(B) and A.unpaid > 0 and A.need > B
    # B: 균등 적립률 = 예산 ÷ 사용액
    assert Bs.params["base"] == pytest.approx(B / U * 100, rel=1e-6) and Bs.exhaust is None
    # C: 기본률을 풀어 예산을 정확히 씀, 가산 덕에 전통시장·원도심 몫이 늘고 상한을 넘지 않음
    assert C.need == pytest.approx(B, rel=1e-6) and C.exhaust is None
    assert C.shares["전통시장"] > A.shares["전통시장"] and C.shares["원도심"] > A.shares["원도심"]
    assert C.params["base"] < Bs.params["base"] and C.peak <= PL.CAP
    assert C.focus_amt > A.focus_amt                      # 질문한 동네(일도1동)로 가는 적립액 증가
    assert sum(C.monthly_paid) == pytest.approx(B, rel=1e-6)
    sim = PL.Simulator(fr, B)
    assert sim.rates({"base": 18.0, **PL.C_BONUS}).max() == PL.CAP


def test_response_sensitivity_and_effect_adjustment():
    fr = _frame()
    B = 3370.0
    base0 = PL.design(fr, B)["B"].params["base"]
    base2 = PL.design(fr, B, response=2.0)["B"].params["base"]
    assert base2 > base0                                   # 적립률을 낮추면 사용도 줄어 같은 예산으로 조금 더 줄 수 있음
    assert PL.effect_decision({"did_pct": 12.0, "p": 0.05}) == "keep"
    assert PL.effect_decision({"did_pct": 3.0, "p": 0.4}) == "narrow"
    assert PL.effect_decision({"did_pct": -9.0, "p": 0.05}) == "stop"
    for eff, key in (({"did_pct": 12.0, "p": 0.05}, "keep"), ({"did_pct": 3.0, "p": 0.4}, "narrow"),
                     ({"did_pct": -9.0, "p": 0.05}, "stop")):
        sc = PL.design(fr, B, effect=eff)
        assert "C′" in sc and sc["C′"].note == PL.DECISION_TEXT[key] and sc["C′"].need == pytest.approx(B, rel=1e-6)
        assert PL.recommend(sc)[0] == "C′"
    keep = PL.design(fr, B, effect={"did_pct": 12.0, "p": 0.05})
    assert keep["C′"].params["base"] < keep["C"].params["base"]     # 늘어난 시장 사용만큼 비용 반영
    narrow = PL.design(fr, B, effect={"did_pct": 3.0, "p": 0.4})["C′"].params
    assert narrow["mkt"] == 0 and narrow["mday"] == 5.0 and narrow["rain"] == 5.0


def test_recommendation_follows_goal_and_diagnosis():
    fr = _frame()
    sc = PL.design(fr, 3370.0)
    assert PL.recommend(sc)[0] == "C"
    assert PL.recommend(sc, "even")[0] == "B" and PL.recommend(sc, "current")[0] == "A"


def test_tamna_report_has_policy_section_and_action_pack(tamna_cat, tmp_path):
    from dataolle.courses.action_pack import make_action_pack
    from dataolle.courses.tamna import TamnaCourse, TamnaInput
    out = TamnaCourse(tamna_cat, tmp_path).run(TamnaInput(scope="hd", value="일도1동",
                                                          policy=PL.PolicyInput(budget=None, goal="auto")),
                                               make_pdf=False, make_docx=False)
    rep = out.report
    heads = [s.heading for s in rep.sections]
    assert any("처방 설계" in h for h in heads)
    pol = rep.extra["policy"]
    assert pol["rec"] in pol["scen"] and {"A", "B", "C"} <= set(pol["scen"])
    assert pol["scen"]["A"].exhaust is not None            # 기본 예산(사용액의 6.25%)이면 10% 적립은 중간에 소진
    assert all(r.check.passed for r in rep.results)
    assert "2026년" not in pol["question"] or "효과" in pol["question"]
    text = " ".join(str(b.text) for s in rep.sections for b in s.blocks if hasattr(b, "text"))
    assert "확인 불가" not in text and "{{" not in text
    # 실행물 묶음
    pk = make_action_pack(out, tmp_path / "pack", make_pdf=False)
    assert pk.scenario == pol["rec"] and len(pk.titles) == 9
    html = pk.html.read_text(encoding="utf-8")
    for w in ("개선안 초안", "상인회 안내문", "현수막", "어르신 안내문", "추진 일정", "효과 확인 질문", pol["question"]):
        assert w in html
    assert "확인 불가" not in html and "{{" not in html
    assert len(pk.copy["sms_short"]) <= 45 and all("{{" not in v for v in pk.copy.values())
    assert pk.docx is not None and pk.docx.exists()
    # 다른 안으로도 만들 수 있음(시장 적립률 사실을 새로 만듦)
    pk_b = make_action_pack(out, tmp_path / "pack_b", "B", make_pdf=False, make_docx=False)
    assert pk_b.scenario == "B" and "B안" in pk_b.report.subtitle


class _FakeCopywriter:
    """홍보 문구 다듬기용 가짜 AI(외부 공급자로 취급): 초안의 자리표시자를 지켜 다듬되, 한 문구에는 근거 없는 숫자를 섞는다."""
    provider = "anthropic"
    model = "fake-copy"
    label = "fake"

    def __init__(self):
        self.users = []

    @property
    def available(self):
        return True

    def complete(self, system, user, max_tokens=1200, temperature=0.2):
        from dataolle.ai.clients import LLMResponse
        self.users.append(user)
        drafts = json.loads(user)["초안"]
        out = {k: v.replace("탐나는전", "탐나는전(제주 지역화폐)") if k == "sns" else v for k, v in drafts.items()}
        out["banner3"] = "우리 동네 가게, 탐나는전 적립 30% 더!"          # 근거 없는 숫자 → 이 문구만 버려야 함
        return LLMResponse(json.dumps(out, ensure_ascii=False), "fake", "fake-copy", 10, 10, 0.01, {})


def test_copy_polish_sends_no_values_and_rejects_stray_numbers(tamna_cat, tmp_path):
    from dataolle.ai.gateway import AuditLog, Gateway
    from dataolle.courses.action_pack import make_action_pack
    from dataolle.courses.tamna import TamnaCourse, TamnaInput
    out = TamnaCourse(tamna_cat, tmp_path).run(TamnaInput(scope="hd", value="일도1동"), make_pdf=False, make_docx=False)
    fake = _FakeCopywriter()
    gw = Gateway(fake, 0, AuditLog())
    pk = make_action_pack(out, tmp_path / "pack", gateway=gw, make_pdf=False, make_docx=False)
    assert pk.polished and "(제주 지역화폐)" in pk.copy["sns"]
    assert "30%" not in pk.copy["banner3"] and any("숫자" in p for p in pk.problems)
    calls = [r for r in gw.audit.records if r.get("event") == "llm_call"]
    assert len(calls) == 1 and calls[0]["values_sent"] == 0
    sent = json.loads(fake.users[0])
    assert all(row["값"] == "(로컬에서 채움)" for row in sent["사실 목록"])     # 레벨 0: 값 0건
    from dataolle.ai.narrator import stray_numbers
    assert not [v for v in sent["초안"].values() if stray_numbers(v)]          # 초안에도 자리표시자 밖 숫자 0건
    # 레벨 0: 데이터로 고른 시장 이름·배분안도 가려서 보내고, 돌아온 문구는 이 PC에서 되돌림
    mname = out.report.extra["policy"]["markets"][0]["name"]
    assert mname not in fake.users[0] and "{{t" in fake.users[0]
    assert mname in pk.copy["sns"] and "{{" not in pk.copy["sns"]


def test_export_bundle_includes_action_pack_pdf(tamna_cat, tmp_path):
    from dataolle.courses.action_pack import make_action_pack
    from dataolle.courses.tamna import TamnaCourse, TamnaInput
    from dataolle.report.document import find_browser
    from dataolle.report.export_pack import build_pack
    if not find_browser():
        pytest.skip("PDF 인쇄용 브라우저 없음")
    out = TamnaCourse(tamna_cat, tmp_path).run(TamnaInput(scope="all"), make_pdf=True, make_docx=False)
    pk = make_action_pack(out, tmp_path / "pack", make_docx=False)
    assert pk.pdf and pk.pdf.read_bytes()[:5] == b"%PDF-"
    ex = build_pack(out.report, tmp_path / "exp", use="status", report_pdf=out.pdf, extra_pdfs=[(pk.pdf, "실행물 묶음")])
    roles = [f.role for f in ex.files]
    assert "실행물" in roles and not ex.problems
    assert "실행물 묶음 PDF 1개" in ex.description
    names = sorted(f.path.name for f in ex.files)
    assert len(names) == len(set(names))


def test_response_recovers_planted_truth_and_flags_prepay(tamna_cat):
    """가상데이터에 심은 정답(1%p당 log 0.045, 학원 선결제)을 에이전트 도구가 되찾는다.
    단순 비교(높은 기간만)는 앞당긴 소비 때문에 부풀고, 백테스트 오차도 더 크다."""
    from dataolle.courses.response import estimate
    from dataolle.synth.tamna import TRUE_B
    rr = estimate(tamna_cat)
    assert rr.ok and rr.window == ("202504", "202506") and rr.post == ["202507", "202508"]
    assert rr.b_lo <= TRUE_B + 0.002 and abs(rr.b - TRUE_B) < 0.004              # 순수 반응 ≈ 정답
    assert rr.b_naive > rr.b * 1.1                                                # 착시: 10% 넘게 부풀림
    assert rr.flagged == ["학원"]
    pb = rr.per_biz.loc["학원"]
    assert pb["앞당긴 소비 비율"] > 30 and pb["건당 금액 변화"] > 40
    assert rr.backtest and rr.mape < rr.mape_naive and rr.mape < 3
    assert rr.edu_share_win > rr.edu_share_prev


def test_policy_uses_estimated_response_and_fair_scenario(tamna_cat, tmp_path):
    from dataolle.courses.policy import PolicyInput, parse_policy
    from dataolle.courses.tamna import TamnaCourse, TamnaInput
    out = TamnaCourse(tamna_cat, tmp_path).run(TamnaInput(policy=PolicyInput(goal="fair")), make_pdf=False, make_docx=False)
    pol = out.report.extra["policy"]
    assert pol["b_src"] == "est" and pol["rec"] == "E"
    S = pol["scen"]
    assert S["E"].shares["학원·교육"] < S["A"].shares["학원·교육"] and S["E"].shares["학원·교육"] < S["C"].shares["학원·교육"]
    assert not S["E"].exhaust and S["A"].exhaust                                   # 같은 예산: E는 끝까지, 현행은 소진
    assert any("효과 추정" in s.heading for s in out.report.sections)
    assert parse_policy("학원 선결제로 혜택이 쏠리지 않게 나눠 줘")["goal"] == "fair"


def test_plan_year_not_taken_from_past_mention():
    from dataolle.courses.policy import parse_policy
    assert "year" not in parse_policy("2025년 2분기 15% 적립 때 늘어난 사용액 중 학원 선결제는 얼마나 돼요?")
    assert parse_policy("2027년 예산 400억 원으로 적립을 짜 줘")["year"] == 2027


def test_public_population_is_used_for_aging_and_per_capita(tamna_cat, tmp_path):
    """공개 실데이터(행안부 주민등록 인구)가 고령 동네 판정·1인당 분모로 쓰인다. 묶인 단위는 합쳐지고, 일부만 있으면 쓰지 않는다."""
    from dataolle.courses import public_pop as PP
    from dataolle.courses.tamna import TamnaCourse, TamnaInput
    pub = PP.for_units(["일도1동", "송산·정방·중앙·천지동", "추자면", "애월읍"])
    assert pub is not None and not pub.missing and pub.ym >= "202610"
    raw = {u["name"]: u for u in PP._raw()["units"]}
    assert len(raw) == 43 and {u["si"] for u in raw.values()} == {"제주시", "서귀포시"}
    merged = pub.table.loc["송산·정방·중앙·천지동"]
    assert merged["total"] == sum(raw[n]["total"] for n in ("송산동", "정방동", "중앙동", "천지동"))
    assert abs(pub.table.loc["일도1동", "aging"] - raw["일도1동"]["age65"] / raw["일도1동"]["total"] * 100) < 1e-9
    assert PP.for_units(["일도1동", "없는동", "가짜읍"]) is None                       # 90% 미만이면 센터 자료로
    out = TamnaCourse(tamna_cat, tmp_path).run(TamnaInput(), make_pdf=False, make_docx=False)
    pol = out.report.extra["policy"]
    assert "주민등록" in pol["pop_src"] and pol["scen"]["A"].percap_ratio is not None
    assert pol["frame"].pop_old + pol["frame"].pop_young == sum(u["total"] for u in raw.values())


def test_public_usage_check_range():
    """공개 월별 사용액(도청 누리집)으로 잰 반응은 비교 기간에 따라 크게 달라진다 — 문서에 적은 범위와 같아야 한다."""
    import importlib.util
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "tools" / "public_usage_check.py"
    spec = importlib.util.spec_from_file_location("public_usage_check", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    v = [r["pct_per_pp"] for r in mod.rows()]
    assert round(min(v), 1) == 2.9 and round(max(v), 1) == 9.1
    used = mod.load()
    assert sum(val for k, val in used.items() if k[:4] == "2024") == 295584
