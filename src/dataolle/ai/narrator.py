"""AI 문장 다듬기 + 숫자 검증.

AI에는 규칙 기반 초안(숫자 대신 {{f3}} 같은 자리표시자)과 사실 목록을 준다. 레벨 0이면 사실의 '값'은 주지 않는다.
AI가 돌려준 문장에 자리표시자가 아닌 숫자가 하나라도 있으면(날짜·시각 표현 제외) 그 결과를 버리고 초안을 쓴다.
→ 보고서에 '계산하지 않은 숫자'가 들어갈 수 없다.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from dataolle.ai.gateway import Gateway, extract_json
from dataolle.ai.redact import TOKEN
from dataolle.analysis.facts import PLACEHOLDER, FactStore

_ALLOWED_NUMERIC = re.compile(
    r"(19|20)\d{2}년|\d{1,2}월|\d{1,2}일|\d{1,2}~\d{1,2}시|\d{1,2}시|\d분기|(19|20)\d{2}\.\d{2}|=100|10\s*미만|10명\s*미만|"
    r"\d{1,2}~\d{1,2}|\d{2}~\d{2}|\d0대|\d{2}세\s*(이상|이하)|v\d{2}|\d-\d{2}|[가-힣]\d동|\d{1,2}·\d{1,2}일|"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}")
# 날짜(영문 'November 20' 포함)·시간대·연령대·데이터 번호·행정동 이름(일도1동)·오일장 장날(2·7일)


def stray_numbers(text: str) -> list[str]:
    """자리표시자·허용된 날짜 표현을 뺀 뒤 남은 숫자."""
    t = TOKEN.sub("", PLACEHOLDER.sub("", text))          # {{t3}}: 레벨 0에서 가린 이름 토큰
    t = _ALLOWED_NUMERIC.sub("", t)
    return re.findall(r"\d[\d,.]*", t)


def verify(texts: list[str], facts: FactStore) -> list[str]:
    problems = []
    for t in texts:
        for m in PLACEHOLDER.finditer(t):
            if m.group(1) not in facts.facts:
                problems.append(f"없는 사실 번호 {m.group(1)}")
        nums = stray_numbers(t)
        if nums:
            problems.append(f"계산 근거 없는 숫자 {', '.join(nums[:3])}: \"{t[:40]}…\"")
    return problems


SYSTEM = """당신은 제주빅데이터센터 데이터 분석 보고서의 문장을 다듬는 편집자입니다.
규칙:
1. 숫자를 직접 쓰지 마세요. 모든 수치는 사실 목록의 자리표시자({{f1}}, {{f1.abs}}, {{f1.dir}})로만 씁니다.
   날짜·시간대 표현(2025년, 8월, 14~18시)은 써도 됩니다.
2. 초안에 있는 자리표시자의 의미를 바꾸지 마세요. 사실 목록에 없는 번호를 만들지 마세요.
3. 자리표시자 바로 뒤 조사는 '(으)로', '(이)가', '(을)를', '(은)는', '(과)와'처럼 쓰세요(값이 채워질 때 자동 선택).
4. 과장·단정 금지. 데이터는 비씨카드 결제분 등 표본이며 인과를 주장하지 마세요.
5. 결과는 JSON 하나로만: {"summary": [...], "insights": [...], "plan_paragraph": "..."}
   summary: 핵심 요약 3~5개(합니다체), insights: 사업 시사점 3~6개(합니다체, 실행 제안 포함),
   plan_paragraph: 사업계획서 '시장분석'에 붙여 넣을 한 문단(-다체, 4~7문장, 첫 문장에 데이터 출처 명시)."""


@dataclass
class NarrationResult:
    used_ai: bool
    summary: list[str]
    insights: list[str]
    plan: str
    problems: list[str] = field(default_factory=list)
    model: str = ""


def narrate(gateway: Gateway | None, facts: FactStore, draft_summary: list[str], draft_insights: list[str],
            draft_plan: str, context: dict) -> NarrationResult:
    """초안(자리표시자 포함)을 AI로 다듬는다. 실패·검증 탈락 시 초안 그대로."""
    fallback = NarrationResult(False, draft_summary, draft_insights, draft_plan)
    if gateway is None or not gateway.enabled:
        return fallback
    if getattr(gateway, "strict", False):
        # 레벨 0: 초안 문장의 방향·순위 단어도 센터 데이터에서 나온 정보이므로 외부 AI로 보내지 않는다(규칙 기반 초안 사용)
        gateway.audit.write({"event": "narration_skipped_level0"})
        fallback.problems.append("레벨 0: 데이터에서 나온 문장은 외부 AI로 보내지 않음(규칙 기반 초안 사용)")
        return fallback
    leaked = verify(draft_summary + draft_insights + [draft_plan], facts)
    if leaked and gateway.external:
        # 초안에 자리표시자가 아닌 숫자가 있으면 외부로 보내지 않는다(템플릿 실수까지 막는 마지막 관문)
        gateway.audit.write({"event": "draft_blocked", "problems": leaked})
        fallback.problems.append("초안에 수치가 직접 들어 있어 외부 AI로 보내지 않음")
        return fallback
    fact_rows, sent = gateway.facts_for_prompt(facts)
    user = json.dumps({"보고서": context, "사실 목록": fact_rows,
                       "초안": {"summary": draft_summary, "insights": draft_insights, "plan_paragraph": draft_plan}},
                      ensure_ascii=False, indent=1)
    resp = gateway.call("문장 다듬기", SYSTEM, user, max_tokens=1800, values_sent=sent)
    if resp is None:
        fallback.problems.append(gateway.last_error or "AI 호출 실패")
        return fallback
    data = extract_json(resp.text)
    if not isinstance(data, dict):
        fallback.problems.append("AI 응답이 JSON 형식이 아님")
        return fallback
    summary = [str(x) for x in data.get("summary", []) if str(x).strip()][:6]
    insights = [str(x) for x in data.get("insights", []) if str(x).strip()][:7]
    plan = str(data.get("plan_paragraph", "")).strip()
    if not summary or not insights or not plan:
        fallback.problems.append("AI 응답에 빠진 항목이 있음")
        return fallback
    problems = verify(summary + insights + [plan], facts)
    if problems:
        gateway.audit.write({"event": "narration_rejected", "problems": problems})
        fallback.problems.extend(problems)
        return fallback
    return NarrationResult(True, summary, insights, plan, [], gateway.client.label)
