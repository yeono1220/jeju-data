"""전송 게이트웨이(Policy-as-Code): 외부 AI로 나가는 모든 내용은 여기를 거친다.

공식 답변(2026.11): 반출 불가 데이터의 원본·조회 결과는 외부 LLM 전송 불가, 집계 결과의 외부 전송은 사전 승인 불가,
질문에 센터 데이터 수치를 입력하는 것도 반출로 본다. 그래서 기본값은 '레벨 0'이다.

  레벨 0  구조만: 질문(수치 가림) + 데이터 목록·지표·차원 이름(공개 명세서 정보). 센터 값은 0건.
          → 문장은 자리표시자({{f3}})로만 쓰게 하고, 숫자는 이 PC에서 채운다.
  레벨 1  + 응용집계 값(비율·증감률·지수·순위). 관리자가 켤 때만.
  레벨 2  + 반출 기준을 통과한 단순집계 값(10 이상, 역산 불가). 관리자가 켤 때만.

Ollama(로컬 모델)는 이 PC 밖으로 나가지 않으므로 레벨과 무관하게 사실 값을 줄 수 있다.
모든 호출은 감사 로그(JSONL)에 '무엇을 보냈는지' 그대로 남는다.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from dataolle.ai.clients import LLMClient, LLMError, LLMResponse, NullClient, is_external
from dataolle.ai.redact import Redactor
from dataolle.analysis.facts import FactStore

LEVEL_TEXT = {
    0: "레벨 0 · 구조만 전송(센터 수치 0건)",
    1: "레벨 1 · 응용집계(비율·증감률·지수) 값까지 전송",
    2: "레벨 2 · 반출 기준을 통과한 집계값까지 전송",
}

# 단위가 붙은 수치(센터 데이터 값일 수 있음) — 날짜·기간·순번 표현은 남긴다.
_NUM = r"\d[\d,]*(?:\.\d+)?"
_VALUE_UNITS = r"(?:억\s*원|만\s*원|천\s*원|원|억|만|천|명|건|%|퍼센트|프로|배|회|개소|곳|가구|대|편|석|m/s|㎧|미터|mm|km|m(?![a-zA-Z]))"
_QUANTITY = re.compile(rf"(?<![\w.]){_NUM}\s*{_VALUE_UNITS}")
_BIG_NUMBER = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?![\w.])|(?<![\w.])\d{5,}(?![\w.])")
_DATE_LIKE = re.compile(r"^(19|20)\d{2}(0[1-9]|1[0-2])?$")
_AGE_BAND = re.compile(r"^[1-9]0\s*대$")


def sanitize_question(text: str) -> tuple[str, int]:
    """질문 속 수치를 [수치]로 가린다. 날짜(2025년, 8월, 202512)·개수 조건(상위 10개, 3개월)은 남긴다."""
    count = 0

    def mask(m):
        nonlocal count
        if _AGE_BAND.match(m.group(0)):          # '20대'·'30대'는 연령대 이름(수치 아님)
            return m.group(0)
        count += 1
        return "[수치]"

    out = _QUANTITY.sub(mask, text)

    def mask_big(m):
        nonlocal count
        raw = m.group(0).replace(",", "")
        if _DATE_LIKE.match(raw):
            return m.group(0)
        count += 1
        return "[수치]"

    out = _BIG_NUMBER.sub(mask_big, out)
    return out, count


@dataclass
class AuditLog:
    path: Path | None = None
    records: list[dict] = field(default_factory=list)

    def write(self, rec: dict):
        rec = {"time": datetime.now().isoformat(timespec="seconds"), **rec}
        self.records.append(rec)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def totals(self) -> dict:
        calls = [r for r in self.records if r.get("event") == "llm_call"]
        return {"calls": len(calls), "ok": sum(1 for r in calls if r.get("ok")),
                "input_tokens": sum(r.get("input_tokens", 0) for r in calls),
                "output_tokens": sum(r.get("output_tokens", 0) for r in calls),
                "values_sent": sum(r.get("values_sent", 0) for r in calls),
                "masked_in_questions": sum(r.get("masked_numbers", 0) for r in calls)}


@dataclass
class Gateway:
    client: LLMClient = field(default_factory=NullClient)
    level: int = 0
    audit: AuditLog = field(default_factory=AuditLog)
    redactor: Redactor = field(default_factory=Redactor)

    @property
    def strict(self) -> bool:
        """외부 AI + 레벨 0: 센터 수치는 물론 센터 데이터에서 나온 이름·문장도 토큰으로 가려서 보낸다."""
        return self.external and self.level == 0

    @property
    def external(self) -> bool:
        return is_external(self.client.provider)

    @property
    def enabled(self) -> bool:
        return self.client.available

    def effective_level(self) -> int:
        """로컬 모델은 밖으로 나가지 않으므로 모든 검수 통과 값을 줄 수 있다."""
        return 2 if not self.external else self.level

    def facts_for_prompt(self, facts: FactStore) -> tuple[list[dict], int]:
        rows = facts.table_for_prompt(self.effective_level())
        sent = sum(1 for r in rows if r.get("값") != "(로컬에서 채움)")
        return rows, sent

    def call(self, purpose: str, system: str, user: str, max_tokens: int = 1200, temperature: float = 0.2,
             values_sent: int = 0, masked_numbers: int = 0) -> LLMResponse | None:
        """AI 호출. 실패하면 None(상위 계층은 규칙 기반으로 계속 진행)."""
        if not self.enabled:
            return None
        n0 = self.redactor.count
        if self.strict:
            user = self.redactor.redact(user)
        rec = {"event": "llm_call", "purpose": purpose, "provider": self.client.provider, "model": self.client.model,
               "external": self.external, "level": self.effective_level(), "values_sent": values_sent,
               "masked_numbers": masked_numbers, "chars_sent": len(system) + len(user),
               "terms_redacted": self.redactor.count - n0, "sent_system": system, "sent_user": user}
        t0 = time.time()
        try:
            resp = self.client.complete(system, user, max_tokens=max_tokens, temperature=temperature)
        except LLMError as e:
            self.audit.write({**rec, "ok": False, "error": str(e), "seconds": round(time.time() - t0, 2)})
            self.last_error = str(e)
            return None
        self.audit.write({**rec, "ok": True, "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens,
                          "seconds": round(resp.seconds, 2), "received": resp.text[:4000]})
        if self.strict:
            resp.text = self.redactor.unredact(resp.text)
        return resp

    @property
    def supports_tools(self) -> bool:
        return self.enabled and hasattr(self.client, "chat")

    def chat(self, purpose: str, system: str, messages: list, tools: list, new_content: str, max_tokens: int = 4096,
             masked_numbers: int = 0) -> LLMResponse | None:
        """도구 사용 대화의 한 걸음. 감사 로그에는 이번에 새로 보낸 내용(질문 또는 도구 결과)을 그대로 남긴다.
        도구 결과는 에이전트가 만들 때 이미 숫자를 뺀 '구조'만 담는다(values_sent=0)."""
        if not self.supports_tools:
            return None
        rec = {"event": "llm_call", "purpose": purpose, "provider": self.client.provider, "model": self.client.model,
               "external": self.external, "level": self.effective_level(), "values_sent": 0,
               "masked_numbers": masked_numbers, "chars_sent": len(new_content), "sent_system": system if len(messages) <= 1 else "",
               "sent_user": new_content}
        t0 = time.time()
        try:
            resp = self.client.chat(system, messages, tools, max_tokens=max_tokens)
        except LLMError as e:
            self.audit.write({**rec, "ok": False, "error": str(e), "seconds": round(time.time() - t0, 2)})
            self.last_error = str(e)
            return None
        calls = [b.get("name") for b in resp.raw.get("content", []) if b.get("type") == "tool_use"]
        self.audit.write({**rec, "ok": True, "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens,
                          "seconds": round(resp.seconds, 2), "received": (resp.text or "")[:4000], "tool_calls": calls})
        return resp

    last_error: str = ""


def extract_json(text: str):
    """응답에서 JSON 객체를 꺼낸다(```json 블록·앞뒤 설명문 허용)."""
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    cand = m.group(1) if m else None
    if cand is None:
        start, end = text.find("{"), text.rfind("}")
        cand = text[start:end + 1] if start >= 0 and end > start else None
    if cand is None:
        return None
    try:
        return json.loads(cand)
    except json.JSONDecodeError:
        return None
