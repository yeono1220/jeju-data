import json

import pytest

from dataolle.ai.clients import LLMClient, LLMError, LLMResponse
from dataolle.ai.gateway import AuditLog, Gateway, extract_json, sanitize_question
from dataolle.ai.narrator import narrate, stray_numbers, verify
from dataolle.ai.planner import ai_plan, find_period, rule_plan
from dataolle.analysis.facts import FactStore
from dataolle.sdc.engine import AggKind


class FakeClient(LLMClient):
    provider = "anthropic"          # 외부 전송으로 취급

    def __init__(self, replies):
        super().__init__(model="fake-model", api_key="test")
        self.replies = list(replies)
        self.calls = []

    def complete(self, system, user, max_tokens=1200, temperature=0.2):
        self.calls.append((system, user))
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return LLMResponse(r, self.provider, self.model, 100, 50, 0.1)


def _facts():
    fs = FactStore()
    fs.add("애월읍 카페 이용금액 합계", 2_960_000_000, "원", AggKind.SIMPLE)       # f1
    fs.add("관광객 비중", 63.3, "%", AggKind.APPLIED)                              # f2
    fs.add("객단가 차이", 2.8, "%", AggKind.APPLIED, is_change=True)              # f3
    return fs


def test_sanitize_question_masks_values_but_keeps_dates():
    q, n = sanitize_question("월 매출 1,200만원 카페, 2025년 8월 기준 상위 10개 동, 손님 300명, 202512, 12345678")
    assert "1,200" not in q and "300명" not in q and "12345678" not in q
    assert "2025년 8월" in q and "상위 10개" in q and "202512" in q
    assert n == 3


def test_level0_sends_no_values_and_logs(tmp_path):
    fs = _facts()
    gw = Gateway(FakeClient(["{}"]), level=0, audit=AuditLog(tmp_path / "audit.jsonl"))
    rows, sent = gw.facts_for_prompt(fs)
    assert sent == 0 and all(r["값"] == "(로컬에서 채움)" for r in rows)
    gw1 = Gateway(FakeClient(["{}"]), level=1)
    rows1, sent1 = gw1.facts_for_prompt(fs)
    assert sent1 == 2                                   # 응용집계(%) 두 개만
    gw.call("테스트", "sys", "user")
    rec = json.loads((tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert rec["ok"] and rec["level"] == 0 and rec["sent_user"] == "user"


def test_stray_number_detection():
    assert stray_numbers("이용금액은 {{f1}}(으)로 2025년 8월에 가장 많았다") == []
    assert stray_numbers("오후(14~18시)에 몰린다") == []
    assert stray_numbers("이용금액은 29.6억 원이다") == ["29.6"]
    assert verify(["{{f9}} 증가"], _facts())                 # 없는 사실 번호


def test_narration_accepts_placeholders_and_rejects_numbers():
    fs = _facts()
    good = json.dumps({"summary": ["이용금액은 {{f1}}입니다.", "관광객 몫이 {{f2}}(으)로 큽니다."],
                       "insights": ["객단가가 {{f3.abs}} {{f3.dir}}했습니다."],
                       "plan_paragraph": "제주빅데이터센터 데이터에 따르면 이용금액은 {{f1}}이다."}, ensure_ascii=False)
    bad = json.dumps({"summary": ["이용금액은 30억 원입니다."], "insights": ["좋습니다."], "plan_paragraph": "약 30억이다."},
                     ensure_ascii=False)
    # 레벨 0(기본): 초안 문장의 방향·순위 단어도 센터 데이터에서 나온 정보 → 외부 AI로 보내지 않고 초안 그대로
    gw0 = Gateway(FakeClient([good]), level=0)
    r0 = narrate(gw0, fs, ["초안"], ["초안"], "초안", {})
    assert not r0.used_ai and not gw0.client.calls and "레벨 0" in r0.problems[0]
    # 레벨 1 이상(센터 반출 승인 범위를 넓힌 경우)에서만 다듬기
    gw = Gateway(FakeClient([good]), level=1)
    r = narrate(gw, fs, ["초안"], ["초안"], "초안", {})
    assert r.used_ai and fs.render(r.summary[1]) == "관광객 몫이 63.3%로 큽니다."
    gw2 = Gateway(FakeClient([bad]), level=1)
    r2 = narrate(gw2, fs, ["초안 요약"], ["초안 시사점"], "초안 문단", {})
    assert not r2.used_ai and r2.summary == ["초안 요약"] and r2.problems
    gw3 = Gateway(FakeClient([LLMError("연결 실패")]), level=1)
    r3 = narrate(gw3, fs, ["초안"], ["초안"], "초안", {})
    assert not r3.used_ai and "연결 실패" in r3.problems[0]


def test_planner_rules_and_ai_validation():
    p = rule_plan("애월읍에 카페 창업하려고 해요")
    assert p.course == "market" and p.market == {"biz": "비알콜 음료점업", "bjd": "애월읍"}
    p2 = rule_plan("노형동 카페 시간대별 이용건수 비중")
    assert p2.course == "free" and p2.analyses[0].dataset == "card_time" and p2.analyses[0].transform == "share"
    assert find_period("2025년 3~5월") == ["202503", "202505"]
    ok = '{"course": "free", "analyses": [{"dataset": "card", "measure": "amt", "rows": "month", "filters": {"bjd": ["연동"]}}]}'
    gw = Gateway(FakeClient([ok]), level=0)
    p3 = ai_plan("연동 월별 매출 1,000만원?", gw)
    assert p3.by_ai and p3.analyses[0].rows == "month"
    sent_user = gw.client.calls[0][1]
    assert "1,000만원" not in sent_user and "[수치]" in sent_user
    broken = '{"course": "free", "analyses": [{"dataset": "card", "measure": "nope"}]}'
    p4 = ai_plan("연동 카페 월별 매출", Gateway(FakeClient([broken]), level=0))
    assert not p4.by_ai and p4.analyses and p4.problems


def test_extract_json_variants():
    assert extract_json('설명\n```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('앞말 {"a": 2} 뒷말') == {"a": 2}
    assert extract_json("JSON 없음") is None


def test_market_course_drafts_have_no_raw_numbers(tmp_path):
    """코스 1의 규칙 기반 초안이 외부 AI로 나갈 때 숫자가 하나도 없어야 한다(자리표시자만)."""
    from dataolle.analysis import Catalog
    from dataolle.courses.market import MarketCourse, MarketInput
    from dataolle.synth.generator import generate

    class Echo(LLMClient):
        provider = "anthropic"

        def __init__(self):
            super().__init__(model="echo", api_key="x")
            self.users = []

        def complete(self, system, user, max_tokens=1200, temperature=0.2):
            self.users.append(user)
            return LLMResponse(json.dumps(json.loads(user)["초안"], ensure_ascii=False), self.provider, self.model)

    data = tmp_path / "요청파일"
    generate(data, profile="small", pop_days=1, pop_emd=["애월읍"], verbose=False)
    cat = Catalog.from_folder(data)
    client = Echo()
    gw0 = Gateway(Echo(), level=0, audit=AuditLog())
    out0 = MarketCourse(cat, tmp_path / "out0").run(MarketInput(biz="비알콜 음료점업", bjd="애월읍"), gateway=gw0,
                                                     make_pdf=False, make_docx=False)
    assert "AI(" not in out0.report.ai_note and not any(r.get("event") == "llm_call" for r in gw0.audit.records)
    gw = Gateway(client, level=1, audit=AuditLog(tmp_path / "audit.jsonl"))
    out = MarketCourse(cat, tmp_path / "out").run(MarketInput(biz="비알콜 음료점업", bjd="애월읍"), gateway=gw,
                                                   make_pdf=False, make_docx=False)
    assert "AI(" in out.report.ai_note, out.report.ai_note
    sent = json.loads(client.users[0])
    # 레벨 1: 응용집계(비율·지수)만 값이 가고, 단순집계(합계·인원)는 여전히 이 PC에서 채움
    assert all(r["값"] == "(로컬에서 채움)" for r in sent["사실 목록"] if r["유형"] != "응용집계")
    draft = sent["초안"]
    assert not verify(draft["summary"] + draft["insights"] + [draft["plan_paragraph"]], out.facts)
