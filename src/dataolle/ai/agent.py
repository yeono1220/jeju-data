"""데이터올레 에이전트 — 질문을 받으면 스스로 도구를 골라 분석하고, 숫자를 보지 않은 채 답을 설계한다.

  AI가 보는 것  질문(수치 가림), 불러온 데이터 목록·기간·코드값, 코스 실행 결과의 '구조'(절 제목, 사실 번호와 지표 이름).
                레벨 0(기본)에서는 센터 데이터에서 나온 이름·순위·판정·초안 문장도 {{t3}} 토큰으로 가린다(ai/redact.py)
                → 센터 수치 0건, 센터 데이터에서 나온 정보 0건
  AI가 하는 것  도구 고르기(데이터 확인 → 코스·자유 분석 실행 → 필요하면 한 번 더) → 자리표시자({{f12}})로 답 쓰기
  이 PC가 하는 것  실제 계산·반출 검수·보고서 작성, 자리표시자 채우기, 근거 없는 숫자 검사(있으면 답을 버림)

일의 흐름(탐나는전): 진단(동네 진단 카드·유형) → 처방 설계(적립 예산 배분 A~D안 시뮬레이션) → 실행물(개선안·안내문·홍보 문구·
일정표 묶음) → 효과 확인(다음 방문 때 붙여 넣을 질문 → 전후 효과로 다음 배분 C′안). 예산·기간 같은 숫자는 질문에서 이 PC가 읽는다.
AI가 없거나 실패하면 같은 도구를 규칙 기반 순서로 호출한다(결과물·작업 기록 형식 동일).
Claude는 Messages API의 도구 사용(tool use)으로, 그 밖의 공급자는 JSON 한 줄 행동 방식으로 같은 도구를 쓴다.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from dataolle.ai.gateway import Gateway, extract_json, sanitize_question
from dataolle.ai.narrator import verify
from dataolle.ai.planner import Plan, catalog_text, rule_plan
from dataolle.analysis.facts import PLACEHOLDER, FactStore
from dataolle.sdc.engine import Level

MAX_STEPS = 6
COURSES = ("tamna", "market", "status", "before_after")
COURSE_LABEL = {"tamna": "탐나는전 캐시백 설계", "market": "창업 시장분석",
                "status": "동네 현황자료", "before_after": "사건 전후 효과"}
# 코스별 꼭 필요한 데이터 / 있으면 더 좋아지는 데이터
NEEDS = {"tamna": (["1-32"], ["1-08", "4-11", "4-10", "5-02"]),
         "market": (["1-08"], ["1-12", "1-14", "1-19", "1-24", "4-11"]),
         "status": (["1-08"], ["1-12", "1-14", "4-11"]),
         "before_after": (["1-08"], [])}

TOOLS = [
    {"name": "check_data",
     "description": "이 PC에 불러온 제주빅데이터센터 데이터(데이터셋 이름·기간)와 코스별로 모자란 데이터를 알려 줍니다. "
                    "자유 분석에 쓸 데이터셋·지표·차원 키 안내도 들어 있습니다. 숫자 값은 없습니다.",
     "input_schema": {"type": "object", "properties": {
         "course": {"type": "string", "enum": list(COURSES), "description": "모자란 데이터를 확인할 코스(선택)"}}}},
    {"name": "run_course",
     "description": "분석 코스를 실행해 반출 기준 검수를 통과한 보고서(PDF)를 만듭니다. 결과로 절 제목, 사실 번호와 설명(값 없음), "
                    "동네 유형별 동네 목록, 자리표시자로 쓴 초안 문장을 돌려줍니다. "
                    "tamna=탐나는전 캐시백 설계(탐나는전·지역화폐·전통시장·오일장·원도심·고령화·인구·학령인구·동네 진단. "
                    "scope=all|city|emd|hd, value=시 또는 동네 이름, 정책 효과를 물으면 event=YYYYMM. "
                    "보고서에는 적립률 반응도 추정(2025년 2분기 15% 자연실험, 학원 선결제 같은 앞당긴 소비 분리, 백테스트)과 "
                    "적립 예산 배분 시뮬레이션(처방 설계: A 현행형·B 균등 유지형·C 동네 살리기 집중형·D 사각지대 보강형·E 쏠림 방지형, "
                    "효과를 확인했으면 C′ 조정안)이 들어가며 goal로 우선순위를 줍니다. 예산·기간은 질문에서 이 PC가 읽습니다), "
                    "market=창업 시장분석(biz 업종, bjd 법정동), status=동네 현황자료(bjd), before_after=사건 전후 효과(bjd, event, biz 선택).",
     "input_schema": {"type": "object", "properties": {
         "course": {"type": "string", "enum": list(COURSES)},
         "scope": {"type": "string", "enum": ["all", "city", "emd", "hd"],
                   "description": "tamna 범위: 제주 전체 / 시 / 읍면동(법정동 이름) / 행정동(예: 일도1동, 아라동)"},
         "value": {"type": "string", "description": "시 이름 또는 동네 이름"},
         "event": {"type": "string", "description": "정책·사건 시작 월 YYYYMM(선택)"},
         "event_name": {"type": "string", "description": "정책·사건 이름(선택)"},
         "biz": {"type": "string", "description": "카드 업종 이름(예: 비알콜 음료점업)"},
         "bjd": {"type": "string", "description": "법정동 이름(예: 애월읍, 월평동(제주시))"},
         "goal": {"type": "string", "enum": ["auto", "revive", "blind", "even", "current", "fair"],
                  "description": "처방 설계의 우선순위. tamna: revive=원도심·전통시장 살리기, blind=사각지대 보강, even=고르게, "
                                 "current=지금 방식 유지, fair=혜택 쏠림(선결제·고액 결제·역진성) 막기, auto=진단에 맞춰 자동"}},
         "required": ["course"]}},
    {"name": "make_action_pack",
     "description": "마지막 보고서의 처방 설계에서 고른 안으로 바로 쓰는 문서 묶음(PDF·DOCX)을 만듭니다. tamna: 한 장 요약, "
                    "도청 개선안 초안, 대안 비교표, 상인회 안내문, 홍보 문구(현수막·SNS·문자·관광객), 어르신 안내문, 추진 일정·체크리스트, "
                    "효과 확인 질문. 안내 문구는 이 PC가 자리표시자로 AI에 보내 다듬습니다.",
     "input_schema": {"type": "object", "properties": {
         "scenario": {"type": "string", "enum": ["auto", "A", "B", "C", "D", "E", "C′"],
                      "description": "auto면 보고서의 추천안(A~D, 효과를 확인했으면 C′)"},
         "emphasis": {"type": "string", "description": "홍보 문구에서 강조할 점(숫자 없이, 예: 어르신이 읽기 쉽게·비 오는 날 강조)"}}}},
    {"name": "run_analysis",
     "description": "코스로 답하기 어려운 짧은 질문을 위한 자유 집계 1건(표·그림)을 만듭니다. 키는 check_data의 안내를 따르세요.",
     "input_schema": {"type": "object", "properties": {
         "dataset": {"type": "string"}, "measure": {"type": "string"}, "rows": {"type": "string"}, "cols": {"type": "string"},
         "filters": {"type": "object", "description": "차원 키 → 값 목록"},
         "transform": {"type": "string", "enum": ["none", "share", "yoy", "index", "rank"]},
         "top_n": {"type": "integer"}, "period": {"type": "array", "items": {"type": "string"}}},
         "required": ["dataset", "measure"]}},
    {"name": "finish",
     "description": "사용자에게 답을 마칩니다. answer에는 숫자를 직접 쓰지 말고 사실 번호 자리표시자({{f12}}, 증감은 {{f12.abs}} {{f12.dir}})만 쓰세요.",
     "input_schema": {"type": "object", "properties": {
         "answer": {"type": "string", "description": "3~6문장 한국어 답(합니다체)"},
         "next_steps": {"type": "array", "items": {"type": "string"}, "description": "사용자가 다음에 할 수 있는 일 1~3개"}},
         "required": ["answer"]}},
]

SYSTEM = """당신은 제주빅데이터센터 분석 PC에서 동작하는 '데이터올레' 분석 에이전트입니다.
목표: 사용자의 질문에 답하고, 반출 가능한 보고서까지 만들어 주는 것. 주된 주제는 '탐나는전 캐시백 설계'
(정해진 적립 예산으로 몇 %를 어디에 줄지 — 연말까지 끊기지 않는지, 전통시장·원도심 몫, 혜택 쏠림·역진성, 앞당긴 소비 착시)이며,
창업 시장분석·동네 현황·사건 전후 효과도 다룹니다.

일하는 순서:
1. check_data로 불러온 데이터를 확인합니다.
2. 질문에 맞는 코스를 run_course로 실행합니다. 코스로 안 되는 짧은 질문만 run_analysis를 씁니다.
   - 특정 동네를 물으면 tamna의 scope=emd(법정동 이름) 또는 hd(행정동 이름, 예: 일도1동·아라동)로 동네 진단 카드를 만듭니다.
   - 정책·캐시백·지원 '이후 효과'를 물으면 event(YYYYMM)를 넣습니다(효과를 반영한 다음 배분 C′안이 함께 나옵니다).
   - 질문의 우선순위에 맞춰 goal을 고릅니다(원도심·시장 살리기=revive, 사각지대=blind, 고르게=even,
     혜택 쏠림·역진성·선결제·학원=fair).
3. 문제 확인에서 멈추지 않습니다. 결과에 '처방 설계'가 있으면 make_action_pack으로 실행물 묶음을 만듭니다
   (보통 추천안, 질문이 다른 우선순위를 말하면 그에 맞는 안). 꼭 필요할 때만 코스를 한 번 더 실행합니다.
   도구 호출은 모두 합쳐 5번 이하.
4. finish로 답을 마칩니다. 답은 진단(어떤 동네·어떤 사용 패턴인지) → 처방(어떤 안, 무엇이 바뀌는지) → 실행물(무엇을 만들었는지)
   → 효과 확인(다음에 넣을 질문) 순서로 씁니다. 각 단계 앞에 [진단] [처방] [실행물] [효과 확인]을 붙입니다.

반드시 지킬 것:
- 당신은 숫자 값을 볼 수 없습니다. 도구 결과의 사실 번호(f12)와 설명만 보고, 답에는 {{f12}} 같은 자리표시자로 수치를 넣습니다.
  증감은 {{f12.abs}} {{f12.dir}}(예: '{{f12.abs}} {{f12.dir}}했습니다'). 사실 목록에 없는 숫자를 쓰면 답이 버려집니다.
  날짜(2025년 9월)와 동네 이름(일도1동)은 써도 됩니다.
- '높다·낮다·가장' 같은 비교·방향은 직접 쓰지 않습니다. 도구 결과의 초안 문장 토큰({{t3}})을 그대로 옮겨 쓰면 이 PC가 문장을 채웁니다.
- {{t12}} 같은 토큰은 센터 데이터에서 나온 이름·판정·문장을 가린 것입니다. 내용을 짐작하지 말고 필요한 자리에 그대로 옮겨 씁니다.
- 자리표시자 바로 뒤 조사는 '(으)로', '(이)가', '(은)는', '(을)를'처럼 씁니다.
- 데이터가 모자라면 무엇을 받아야 하는지 알려 주고 멈춥니다(운영자 요청 문안은 화면에 따로 나옵니다).
- 가상데이터라는 표시가 있으면 답 끝에 '가상데이터 결과'라고 밝힙니다."""

PACK_LINE = "[실행물] {k}안으로 도청 개선안 초안, 상인회 안내문, 홍보 문구, 어르신 안내문, 추진 일정·체크리스트를 묶어 만들었습니다."

JSON_PROTOCOL = """한 번에 행동 하나를 JSON 한 개로만 답하세요. 형식:
{"tool": "check_data" | "run_course" | "make_action_pack" | "run_analysis" | "finish", "input": {...}}
도구 설명:
""" + "\n".join(f"- {t['name']}: {t['description']} 입력: {json.dumps(t['input_schema']['properties'], ensure_ascii=False)}"
                for t in TOOLS)


@dataclass
class AgentStep:
    kind: str                  # plan | tool | result | answer | fallback | error
    title: str
    detail: str = ""


@dataclass
class AgentResult:
    question: str
    answer: str = ""
    answer_draft: str = ""
    next_steps: list[str] = field(default_factory=list)
    steps: list[AgentStep] = field(default_factory=list)
    outputs: list = field(default_factory=list)             # CourseOutput
    free: list = field(default_factory=list)                # (AnalysisResult, 그림 경로, FactStore)
    plan: Plan | None = None
    by_ai: bool = False
    request_text: str = ""
    facts: FactStore = field(default_factory=FactStore)
    problems: list[str] = field(default_factory=list)
    packs: list = field(default_factory=list)               # 실행물 묶음(PackOutput)


class Agent:
    def __init__(self, catalog, out_dir: str | Path, gateway: Gateway | None = None, level: Level = Level.CONSERVATIVE,
                 progress: Callable[[AgentStep], None] | None = None, make_pdf: bool = True, make_docx: bool = True):
        self.cat, self.out, self.gw, self.level = catalog, Path(out_dir), gateway, level
        self.progress = progress
        self.make_pdf, self.make_docx = make_pdf, make_docx
        self.res: AgentResult | None = None
        self.n_runs = 0

    # ------------------------------------------------------------------ 레벨 0 가림막
    @property
    def _strict(self) -> bool:
        return self.gw is not None and self.gw.strict

    def _tok(self, value, label: str):
        """레벨 0이면 값을 토큰으로(센터 데이터에서 나온 이름·판정·문장). 아니면 그대로."""
        if not self._strict:
            return value
        if isinstance(value, (list, tuple)):
            value = ", ".join(map(str, value)) if value else "해당 없음"
        return self.gw.redactor.token(str(value), label)

    def _sentences(self, items, label: str) -> list:
        if not self._strict:
            return list(items)
        return [self.gw.redactor.opaque(x, f"{label} {i + 1}(이 PC가 채움)") for i, x in enumerate(items)]

    _PLAN_ONLY = re.compile(r"^[A-E]′?안 (?=.*(?:평일|비 오는 날|장날) 적립률$)")
    _EXHAUST_ONLY = re.compile(r"^[A-E]′?안 예산 소진 (?:월|뒤)")

    def _fact_hidden(self, label: str) -> bool:
        """레벨 0: 소진되는 안에만 생기는 사실은 목록에서 뺀다(있다는 것 자체가 판정을 알려 줌)."""
        return self._strict and bool(self._EXHAUST_ONLY.match(label))

    def _fact_label(self, label: str, rep) -> str:
        """레벨 0: 사실 설명에 섞인 판정을 가린다 — 추천안에만 있는 시장 적립률의 안 글자, 앞당긴 소비 의심 업종 이름
        (업종 이름이 질문에 있으면 낱말 가림에서 면제되므로 여기서 따로 가린다)."""
        if not self._strict:
            return label
        label = self._PLAN_ONLY.sub("추천안 ", label)
        resp = rep.extra.get("response")
        for biz in (resp["result"].flagged if resp and resp["result"].ok else []):
            if label.startswith(f"{biz} 업종:"):
                label = "앞당긴 소비 의심 업종:" + label[len(biz) + 4:]
        return label

    # ------------------------------------------------------------------ 기록
    def _log(self, kind: str, title: str, detail: str = ""):
        st = AgentStep(kind, title, detail)
        self.res.steps.append(st)
        if self.progress:
            self.progress(st)

    # ------------------------------------------------------------------ 진입점
    def run(self, question: str) -> AgentResult:
        q = re.sub(r"\s+", " ", question).strip()
        self.res = AgentResult(q)
        gw = self.gw
        done = False
        # 입력 가드레일: AI를 부르기 전에 이 PC에서 먼저 거른다(원본 요구·개인 식별·지시 무시·범위 밖)
        from dataolle.ai.guardrails import check as guard_check
        from dataolle.ai.redact import center_vocab as _vocab
        g = guard_check(q, rule_plan(q).course, _vocab(self.cat))
        if g is not None:
            self._log("guardrail", f"입력 가드레일({g.code})로 분석을 시작하지 않았습니다", g.message)
            self.res.answer = self.res.answer_draft = g.message
            self.res.problems.append(f"guardrail:{g.code}")
            self.res.next_steps = ["예시 질문 버튼을 눌러 시작하기", "동네·업종·기간을 넣어 다시 묻기"]
            if gw is not None:
                gw.audit.write({"event": "guardrail_block", "code": g.code})
            return self.res
        if gw is not None and gw.strict:
            from dataolle.ai.redact import center_vocab
            gw.redactor.reset(q, center_vocab(self.cat))
        if gw is not None and gw.enabled:
            try:
                done = self._loop_tools(q) if gw.supports_tools else self._loop_json(q)
            except Exception as e:                        # 어떤 실패도 규칙 기반으로 이어 감
                self.res.problems.append(f"에이전트 오류: {e}")
                done = False
            if not done:
                self._log("fallback", "AI 에이전트를 끝까지 쓰지 못해 규칙 기반으로 이어 갑니다", gw.last_error or "")
        if not done:
            self._rule(q)
        else:
            self._ensure_pack()
        self._add_notices(q)
        return self.res

    # ------------------------------------------------------------------ 알림: 말없이 바꿔 계산하지 않기
    _NOT_PLACE = {"유동", "이동", "변동", "활동", "행동", "노동", "가동", "작동", "자동", "수동", "연동", "공동", "읍면", "읍면동", "행정동", "법정동",
                  "측면", "방면", "화면", "장면", "반면", "정면", "전면", "대면", "비대면", "서면", "표면", "라면", "하면", "그러면", "이라면",
                  "운동", "감동", "충동", "아동", "소동", "파동", "원동", "기동", "출동", "거동", "협동", "제동", "구동", "변동", "상면"}

    def _add_notices(self, q: str):
        """질문에 적힌 값·동네를 읽지 못했거나 계산을 바꿨으면 답 맨 앞에 알린다(규칙·AI 경로 공통, 이 PC에서 만듦)."""
        res = self.res
        if not res.outputs or not res.answer:
            return
        rep = res.outputs[-1].report
        pol = rep.extra.get("policy")
        notes: list[str] = []
        if pol is not None:
            from dataolle.courses.policy import parse_policy
            kw = parse_policy(q)
            if re.search(r"예산[^.?!]{0,12}?\d", q) and "budget" not in kw:
                notes.append("질문의 예산 금액을 읽지 못해 기본값(자료 사용액의 6.25%)으로 계산했습니다. '400억 원'처럼 억·만 단위로 적어 주세요.")
            m = re.search(r"(\d+)\s*개월", q)
            if m and not 1 <= int(m.group(1)) <= 12:
                notes.append(f"기간은 1~12개월만 계산합니다. '{m.group(1)}개월' 대신 {pol['frame'].months}개월로 계산했습니다.")
            if pol.get("b_src") == "none":
                why = getattr(pol.get("response"), "reason", "") or "탐나는전 자료 부족"
                notes.append(f"적립률 반응은 추정하지 못했습니다({why}) 그래서 사용 패턴이 지금과 같다고 놓고 계산했습니다.")
            if not pol.get("focus") and self.cat is not None:
                from dataolle.ai.redact import center_vocab
                known = set(center_vocab(self.cat))
                for tok in dict.fromkeys(re.findall(r"[가-힣]{2,4}\d동|[가-힣]{2,4}(?:동|읍)(?=[\s,.?!]|에|의|은|는|을|를|이|가|도|으로|$)", q)):
                    if tok in known or tok in self._NOT_PLACE or len(tok) < 3 or any(tok in k or k in tok for k in known if len(k) >= 3):
                        continue
                    notes.append(f"'{tok}'은(는) 불러온 자료에서 찾지 못해 제주 전체로 분석했습니다. 동네 이름을 확인해 주세요.")
                    break
        if notes:
            head = "[알림] " + " ".join(notes) + "\n\n"
            res.answer = head + res.answer
            if res.answer_draft:
                res.answer_draft = head + res.answer_draft
            self._log("notice", "질문에서 읽지 못했거나 바꿔 계산한 것을 알림", " ".join(notes))

    def _ensure_pack(self):
        """AI가 처방 설계까지만 하고 실행물을 빠뜨렸으면 추천안으로 만들어 붙인다(문제 확인에서 멈추지 않게)."""
        if self.res.packs or not any("policy" in o.report.extra for o in self.res.outputs):
            return
        r = self.t_make_action_pack({"scenario": "auto"})
        if not r.get("ok"):
            return
        pk = self.res.packs[-1]
        line = PACK_LINE.format(k=pk.scenario)
        tail = " (가상데이터 결과)"
        for attr in ("answer", "answer_draft"):
            v = getattr(self.res, attr) or ""
            if line in v:
                continue
            v = (v[: -len(tail)] + " " + line + tail) if v.endswith(tail) else (v + " " + line).strip()
            setattr(self.res, attr, v)

    # ------------------------------------------------------------------ 도구 실행(공통)
    def _exec(self, name: str, inp: dict) -> dict:
        inp = inp or {}
        if name == "check_data":
            return self.t_check_data(inp.get("course"))
        if name == "run_course":
            return self.t_run_course(inp)
        if name == "run_analysis":
            return self.t_run_analysis(inp)
        if name == "make_action_pack":
            return self.t_make_action_pack(inp)
        return {"error": f"알 수 없는 도구: {name}"}

    def _missing(self, course: str) -> dict:
        inv = self.cat.inventory if self.cat is not None else None
        have = set(inv.by_dataset) if inv is not None else set()
        req, opt = NEEDS[course]
        return {"꼭 필요한데 없음": [d for d in req if d not in have], "있으면 좋은데 없음": [d for d in opt if d not in have]}

    def t_check_data(self, course: str | None = None) -> dict:
        from dataolle.spec import default_registry
        reg = default_registry()
        loaded = []
        if self.cat is not None and self.cat.inventory is not None:
            for ds in sorted(self.cat.inventory.by_dataset):
                months = self.cat.inventory.months(ds)
                loaded.append({"데이터": ds, "이름": reg.get(ds).name_ko,
                               "기간": f"{months[0][:4]}.{months[0][4:]}~{months[-1][:4]}.{months[-1][4:]}" if months else "-",
                               "개월 수": len(months)})
        if self.cat is not None:
            for key, pd_ in getattr(self.cat, "public", {}).items():
                a, b = pd_.period
                loaded.append({"데이터": f"공공데이터({key})", "이름": pd_.name,
                               "기간": f"{a[:4]}.{a[4:]}~{b[:4]}.{b[4:]}" if a and b else "-", "개월 수": None})
        out = {"불러온 데이터": loaded, "가상데이터": bool(self.cat is not None and self.cat.synthetic),
               "코스별 데이터": {c: self._missing(c) for c in ([course] if course in COURSES else COURSES)}}
        if course is None:
            out["자유 분석 안내"] = catalog_text()[:6000]
        self._log("tool", "데이터 확인", f"{len(loaded)}종 불러옴" + (" · 가상데이터" if out["가상데이터"] else ""))
        return out

    def _request_text(self, course: str) -> str:
        from dataolle.ingest.inventory import check_plan, make_plan, month_range, operator_request_text
        req, opt = NEEDS[course]
        end = "202512"
        if self.cat is not None and self.cat.inventory is not None:
            ms = sorted({m for d in self.cat.inventory.by_dataset for m in self.cat.inventory.months(d)})
            end = ms[-1] if ms else end
        items = []
        for d in req + opt:
            n = {"1-32": 12, "1-08": 12, "5-02": 12, "4-11": 1, "4-10": 0}.get(d, 3)
            if d == "4-10":
                items += make_plan([d], f"2017{end[4:]}", f"2017{end[4:]}")
            else:
                items += make_plan([d], month_range("200001", end)[-n], end)
        if self.cat is not None and self.cat.inventory is not None:
            items = check_plan(items, self.cat.inventory)
        return operator_request_text(items)

    def t_run_course(self, inp: dict) -> dict:
        course = inp.get("course")
        if course not in COURSES:
            return {"error": f"course는 {', '.join(COURSES)} 중 하나여야 합니다."}
        miss = self._missing(course)
        if miss["꼭 필요한데 없음"]:
            self.res.request_text = self._request_text(course)
            self._log("tool", f"{COURSE_LABEL[course]} — 데이터 부족", ", ".join(miss["꼭 필요한데 없음"]))
            return {"error": f"꼭 필요한 데이터가 없습니다: {', '.join(miss['꼭 필요한데 없음'])}. "
                             "운영자 사전 다운로드 요청 문안을 화면에 준비했습니다. 사용자에게 데이터를 먼저 받아야 한다고 알리세요."}
        self.n_runs += 1
        out_dir = self.out / f"{self.n_runs:02d}_{course}"
        before = set(self.res.facts.facts)
        desc = COURSE_LABEL[course]
        try:
            if course == "tamna":
                from dataolle.courses.tamna import TamnaCourse, TamnaInput
                scope = inp.get("scope") or ("all" if not inp.get("value") else "emd")
                value = inp.get("value") if scope != "all" else None
                ev = str(inp.get("event") or "")[:6] or None
                pin = self._policy_input(ev, inp.get("goal"))
                desc += f"({value or '제주 전체'}" + (f", {ev[:4]}년 {int(ev[4:])}월 전후" if ev else "") + ")"
                self._log("tool", f"코스 실행: {desc}", self._policy_note(pin))
                out = TamnaCourse(self.cat, out_dir, self.level).run(
                    TamnaInput(scope=scope, value=value, event=ev, event_name=inp.get("event_name") or "", policy=pin),
                    make_pdf=self.make_pdf, make_docx=self.make_docx, gateway=None, facts=self.res.facts)
            elif course == "market":
                from dataolle.courses.market import MarketCourse, MarketInput
                if not inp.get("biz") or not inp.get("bjd"):
                    return {"error": "market 코스에는 biz(업종)와 bjd(법정동)가 필요합니다."}
                desc += f"({inp['bjd']} {inp['biz']})"
                self._log("tool", f"코스 실행: {desc}")
                out = MarketCourse(self.cat, out_dir, self.level).run(
                    MarketInput(biz=inp["biz"], bjd=inp["bjd"]), make_pdf=self.make_pdf, make_docx=self.make_docx,
                    gateway=None, facts=self.res.facts)
            elif course == "status":
                from dataolle.courses.status import StatusCourse
                if not inp.get("bjd"):
                    return {"error": "status 코스에는 bjd(법정동)가 필요합니다."}
                desc += f"({inp['bjd']})"
                self._log("tool", f"코스 실행: {desc}")
                out = StatusCourse(self.cat, out_dir, self.level).run_status(
                    inp["bjd"], make_pdf=self.make_pdf, make_docx=self.make_docx, facts=self.res.facts)
            else:
                from dataolle.courses.before_after import BeforeAfterCourse, BeforeAfterInput
                if not inp.get("bjd") or not inp.get("event"):
                    return {"error": "before_after 코스에는 bjd(법정동)와 event(YYYYMM)가 필요합니다."}
                desc += f"({inp['bjd']}, {inp['event']})"
                self._log("tool", f"코스 실행: {desc}")
                out = BeforeAfterCourse(self.cat, out_dir, self.level).run(
                    BeforeAfterInput(treat=[inp["bjd"]], event=str(inp["event"])[:6], biz=inp.get("biz"),
                                     event_name=inp.get("event_name") or ""),
                    make_pdf=self.make_pdf, make_docx=self.make_docx, facts=self.res.facts)
        except (ValueError, KeyError) as e:
            self._log("error", f"{desc} 실행 못 함", str(e))
            return {"error": f"코스를 실행하지 못했습니다: {e}"}
        self.res.outputs.append(out)
        rep = out.report
        new = [f for fid, f in self.res.facts.facts.items() if fid not in before]
        passed = sum(1 for r in rep.results if r.check.passed)
        masked = sum(r.check.masked_cells for r in rep.results)
        self._log("result", f"보고서 완성: {rep.title}",
                  f"절 {len(rep.sections)}개 · 그림 {len(out.figures)}개 · 표·그림 검수 {passed}/{len(rep.results)} 통과 · 비공개 {masked}칸")
        drafts = rep.extra.get("drafts", {})
        result = {"ok": True, "보고서": rep.title, "PDF": out.pdf.name if out.pdf else None,
                  "절": [s.heading for s in rep.sections],
                  "사실": [{"id": f.id, "설명": self._fact_label(f.label, rep), "단위": f.unit, **({"증감": True} if f.is_change else {}),
                           **({"비공개": True} if f.masked else {})} for f in new if not self._fact_hidden(f.label)],
                  "초안_요약": self._sentences(drafts.get("summary", []), "요약 문장"),
                  "초안_처방": self._sentences(drafts.get("rx", []), "처방 문장"),
                  "검수": self._tok(f"표·그림 {len(rep.results)}개 중 {passed}개 통과, 비공개 칸 {masked}개", "검수 결과"),
                  "다루지 못한 항목": out.skipped}
        if rep.extra.get("dongne_types"):
            types = rep.extra["dongne_types"]
            if self._strict:                               # 어느 동네가 어느 유형인지·몇 곳인지도 센터 데이터에서 나온 정보
                from dataolle.courses.dongne import TYPE_ORDER
                types = {t: self._tok(types.get(t, []), f"'{t}' 유형에 든 동네 목록") for t in TYPE_ORDER}
            result["동네 유형"] = types
        if rep.extra.get("effect"):
            e = rep.extra["effect"]
            p = e.get("p")
            result["전후 효과 판정"] = self._tok("우연으로 보기 어려움" if p is not None and p <= 0.1
                                           else "우연과 구별하기 어렵거나 기간이 짧음", "전후 효과 판정")
        if rep.extra.get("response"):
            rf, rres = rep.extra["response"]["facts"], rep.extra["response"]["result"]
            ph = lambda k: "{{" + rf[k] + "}}" if k in rf else None
            result["효과 추정"] = ({"적립률을 한 %p 올릴 때 사용액 변화(순수 반응)": ph("net"), "단순 비교(앞당긴 소비 포함)": ph("naive"),
                                 "부풀린 정도": ph("ill"), "백테스트 오차": ph("bt"),
                                 "앞당긴 소비 의심 업종": self._tok(rres.flagged, "앞당긴 소비 의심 업종"),
                                 "앞당긴 소비 비율": ph("pf")} if rres.ok else {"추정 못 함": rres.reason})
        if rep.extra.get("policy"):
            result["처방 설계"] = self._policy_brief(rep.extra["policy"], self._tok, strict=self._strict)
            pol = rep.extra["policy"]
            self._log("result", f"처방 설계: 추천 {pol['rec']}안 {pol['scen'][pol['rec']].name}",
                      " · ".join(f"{k}안 " + ("끝까지 유지" if not pol["scen"][k].exhaust else f"{pol['scen'][k].exhaust_text()} 소진")
                                 for k in pol["order"]))
        return result

    # ------------------------------------------------------------------ 처방 설계·실행물
    def _policy_input(self, ev: str | None, goal: str | None):
        """예산·기간·시작 월은 질문 원문에서 이 PC가 읽는다(AI는 숫자를 보지 않음). goal은 AI가 고른 값이 우선."""
        from dataolle.courses.policy import PolicyInput, parse_policy
        kw = parse_policy(self.res.question if self.res else "", effect_question=bool(ev))
        if goal and goal != "auto":
            kw["goal"] = goal
        return PolicyInput(**kw)

    @staticmethod
    def _policy_note(pin) -> str:
        from dataolle.analysis.fmt import fmt_value
        bits = [f"예산 {fmt_value(pin.budget, '원')}(질문에서 읽음)" if pin.budget else "예산: 2026년 비율 적용"]
        if pin.months != 12 or pin.start != 1:
            bits.append(f"{pin.start}월부터 {pin.months}개월")
        if pin.goal != "auto":
            bits.append({"revive": "우선순위: 원도심·전통시장", "blind": "우선순위: 사각지대", "even": "우선순위: 고르게",
                         "current": "우선순위: 현행 유지"}.get(pin.goal, ""))
        return " · ".join(b for b in bits if b)

    @staticmethod
    def _policy_brief(pol: dict, tok=lambda v, label: v, strict: bool = False) -> dict:
        """AI에 줄 처방 설계 구조(값 없음: 사실 번호와 안 이름만). 레벨 0이면 추천안·추천 이유·유지 여부·시장 이름도 토큰."""
        sf = pol["sf"]
        lab = {"base": "기본 적립률", "peak": "최대 적립률", "last": "예산 유지 기간", "mkt": "전통시장 몫", "old": "원도심 몫",
               "rev": "살리기 대상 동네 몫", "exm": "예산 소진 월", "unpaid": "소진 뒤 적립 없이 쓰인 사용액 비중",
               "focus_amt": "질문한 동네에 돌아가는 적립액", "focus_sh": "질문한 동네 몫", "blind": "사각지대 동네 몫",
               "edu": "학원·교육 업종 몫", "percap": "고령 동네 1인당 적립액 ÷ 그 밖의 동네 1인당"}
        return {"적립 방식": [{"안": k, "이름": pol["scen"][k].name,
                            "기간 끝까지 유지": tok("예" if not pol["scen"][k].exhaust else "아니오", f"{k}안이 기간 끝까지 유지되는지"),
                            # 레벨 0: '소진 월'·'소진 뒤 비중'은 소진되는 안에만 있어, 있다는 것만으로 어느 안이 끊기는지 드러난다 → 뺀다
                            "사실": {lab[x]: f"{{{{{fid}}}}}" for x, fid in sf[k].items()
                                    if x in lab and not (strict and x in ("exm", "unpaid"))}} for k in pol["order"]],
                "추천": tok(f"{pol['rec']}안 {pol['scen'][pol['rec']].name}", "추천안"),
                "추천 이유": tok(pol["why"], "추천 이유"), "적립 방식 설명(자리표시자)": pol["rule_ph"],
                "예산": f"{{{{{pol['f_budget']}}}}}" + (" (질문에서 읽음)" if pol["user_budget"] else " (2026년 예산 비율을 자료에 적용)"),
                "질문한 동네": pol["focus"], "안내문에 쓸 시장": tok([m["name"] for m in pol["markets"]], "안내문에 쓸 시장"),
                "효과 확인 질문": pol["question"]}

    def _last_rx(self):
        outs = [o for o in self.res.outputs if "policy" in o.report.extra]
        return outs[-1] if outs else None

    def t_make_action_pack(self, inp: dict) -> dict:
        from dataolle.courses.action_pack import make_action_pack
        co = self._last_rx()
        if co is None:
            return {"error": "처방 설계가 들어간 보고서(tamna)가 아직 없습니다. 먼저 run_course를 실행하세요."}
        pol = co.report.extra["policy"]
        sc = inp.get("scenario")
        sc = sc if sc in pol["scen"] else pol["rec"]
        self.n_runs += 1
        self._log("tool", f"실행물 묶음 만들기: {sc}안 {pol['scen'][sc].name}")
        try:
            pk = make_action_pack(co, self.out / f"{self.n_runs:02d}_실행물", sc, gateway=self.gw,
                                  emphasis=str(inp.get("emphasis") or "")[:200], facts=self.res.facts,
                                  make_pdf=self.make_pdf, make_docx=self.make_docx)
        except (ValueError, KeyError) as e:
            self._log("error", "실행물 묶음을 만들지 못함", str(e))
            return {"error": f"실행물 묶음을 만들지 못했습니다: {e}"}
        self.res.packs.append(pk)
        self._log("result", f"실행물 묶음 완성: {len(pk.titles) - 1}종 문서",
                  ("홍보 문구 AI 다듬기 통과" if pk.polished else "홍보 문구 규칙 기반") + f" · {', '.join(pk.titles[:-1])}")
        chosen = f"{sc}안 {pol['scen'][sc].name}"
        return {"ok": True, "고른 안": chosen if inp.get("scenario") in pol["scen"] else self._tok(chosen, "고른 안(추천안)"),
                "문서": pk.titles, "PDF": pk.pdf.name if pk.pdf else None,
                "홍보 문구(자리표시자)": {k: self._sentences([v], f"홍보 문구 {k}")[0] for k, v in pk.drafts.items()}, "AI 문구 다듬기": "통과" if pk.polished else "규칙 기반 초안 사용",
                "효과 확인 질문": pol["question"]}

    def t_run_analysis(self, inp: dict) -> dict:
        from pydantic import ValidationError

        from dataolle.analysis.engine import run_safe
        from dataolle.analysis.spec import AnalysisSpec
        from dataolle.report.charts import render_chart
        try:
            spec = AnalysisSpec(**{k: v for k, v in inp.items() if v is not None})
        except ValidationError as e:
            return {"error": f"분석 설계가 형식에 맞지 않습니다: {str(e).splitlines()[0]}"}
        probs = spec.problems()
        if probs:
            return {"error": "; ".join(probs)}
        if self.cat is None or not self.cat.has(spec.dataset):
            return {"error": f"'{spec.dataset}' 데이터가 불러와지지 않았습니다."}
        before = set(self.res.facts.facts)
        r = run_safe(spec, self.cat, self.res.facts, self.level)
        self.n_runs += 1
        img = render_chart(r, self.out / f"{self.n_runs:02d}_자유분석.png")
        self.res.free.append((r, img, self.res.facts))
        new = [f for fid, f in self.res.facts.facts.items() if fid not in before]
        self._log("result", f"자유 분석: {r.title}", f"검수 {'통과' if r.check.passed else '보완 필요'} · 비공개 {r.check.masked_cells}칸")
        return {"ok": True, "제목": r.title, "사실": [{"id": f.id, "설명": (re.sub(r"^[A-E]′?안 ", "고른 안 ", f.label) if self._strict else f.label), "단위": f.unit} for f in new],
                "초안": self._sentences(r.sentences, "초안 문장"), "검수": "통과" if r.check.passed else "보완 필요",
                "유의사항": r.notes[:3]}

    # ------------------------------------------------------------------ 밖으로 보내기 전 마지막 점검
    def _outbound(self, obj, tool: str = "") -> tuple[str, int]:
        text = json.dumps(obj, ensure_ascii=False)
        if self._strict and tool != "check_data":
            # 데이터 목록·코드값 안내(check_data)는 공개 코드값이라 그대로, 분석 결과는 센터 데이터의 이름을 토큰으로
            text = self.gw.redactor.redact(text)
            if self.gw.redactor.tokens:
                text = text[:-1] + ', "토큰 안내": ' + json.dumps(
                    "{{t1}} 같은 토큰은 센터 데이터에서 나온 말이라 가렸습니다. 답에 그대로 옮겨 쓰면 이 PC가 채웁니다. "
                    "토큰의 내용을 짐작해 비교·순위·방향(높다·늘었다)을 덧붙이지 마세요.", ensure_ascii=False) + "}"
        clean, n = sanitize_question(text)          # 단위가 붙은 수치·큰 수가 섞였다면 가림(정상이면 0건)
        if n and self.gw is not None:
            self.gw.audit.write({"event": "agent_guard_masked", "count": n})
        return clean, n

    # ------------------------------------------------------------------ Claude 도구 사용 루프
    def _loop_tools(self, q: str) -> bool:
        clean, masked = sanitize_question(q)
        first = f"[질문]\n{clean}"
        messages = [{"role": "user", "content": first}]
        self._log("plan", "AI 에이전트가 질문을 읽고 도구를 고릅니다", self.gw.client.label)
        new_content, n_mask = first, masked
        for _ in range(MAX_STEPS):
            resp = self.gw.chat("에이전트", SYSTEM, messages, TOOLS, new_content, masked_numbers=n_mask)
            if resp is None:
                return False
            content = resp.raw.get("content", [])
            messages.append({"role": "assistant", "content": content})
            uses = [b for b in content if b.get("type") == "tool_use"]
            if not uses:
                return self._finish(resp.text, [])
            results, finish = [], None
            for u in uses:
                if u.get("name") == "finish":
                    finish = u.get("input") or {}
                    continue
                inp_ = u.get("input") or {}
                if self._strict:
                    inp_ = self.gw.redactor.unredact_obj(inp_)      # AI가 토큰으로 고른 동네 이름 등을 되돌림
                out = self._exec(u.get("name"), inp_)
                text, n = self._outbound(out, u.get("name") or "")
                results.append({"type": "tool_result", "tool_use_id": u["id"], "content": text,
                                **({"is_error": True} if out.get("error") else {})})
            if finish is not None:
                return self._finish(str(finish.get("answer", "")), [str(x) for x in finish.get("next_steps", [])][:3])
            messages.append({"role": "user", "content": results})
            new_content = json.dumps([r["content"] for r in results], ensure_ascii=False)
            n_mask = 0
        self.res.problems.append("도구 호출 횟수 한도에 도달")
        return False

    # ------------------------------------------------------------------ JSON 행동 루프(도구 사용 API가 없는 공급자)
    def _loop_json(self, q: str) -> bool:
        clean, masked = sanitize_question(q)
        self._log("plan", "AI 에이전트가 질문을 읽고 도구를 고릅니다", self.gw.client.label)
        transcript = [f"[질문]\n{clean}"]
        for _ in range(MAX_STEPS):
            user = "\n\n".join(transcript) + "\n\n다음 행동을 JSON 하나로만 답하세요."
            resp = self.gw.call("에이전트", SYSTEM + "\n\n" + JSON_PROTOCOL, user, max_tokens=1500, masked_numbers=masked)
            masked = 0
            act = extract_json(resp.text) if resp else None
            if not isinstance(act, dict) or "tool" not in act:
                return False
            if act["tool"] == "finish":
                inp = act.get("input") or {}
                return self._finish(str(inp.get("answer", "")), [str(x) for x in inp.get("next_steps", [])][:3])
            out = self._exec(act["tool"], act.get("input") or {})
            text, _ = self._outbound(out, act["tool"])
            transcript.append(f"[행동] {json.dumps(act, ensure_ascii=False)}\n[결과] {text}")
        return False

    def _finish(self, draft: str, next_steps: list[str]) -> bool:
        if self._strict:                                     # 토큰을 이 PC에서 원래 말·문장으로 되돌린 뒤 검증
            draft, next_steps = self.gw.redactor.unredact(draft), [self.gw.redactor.unredact(x) for x in next_steps]
        draft = draft.strip()
        if not draft:
            return False
        problems = verify([draft] + next_steps, self.res.facts)
        if problems:
            self.gw.audit.write({"event": "agent_answer_rejected", "problems": problems})
            self._log("fallback", "AI 답에 근거 없는 숫자가 있어 버리고 규칙 기반 답을 씁니다", "; ".join(problems[:2]))
            self.res.problems += problems
            if self.res.outputs or self.res.free:
                self._rule_answer()
                return True
            return False
        self.res.answer_draft = draft
        self.res.answer = self.res.facts.render(draft)
        self.res.next_steps = next_steps or self._default_next()
        self.res.by_ai = True
        self._log("answer", "답변 작성(숫자는 이 PC에서 채움·검증 통과)")
        return True

    # ------------------------------------------------------------------ 규칙 기반(AI 없음·실패)
    def _rule(self, q: str):
        plan = rule_plan(q)
        self.res.plan = plan
        self._log("plan", "규칙 기반 계획", plan.notes)
        if self.cat is None:
            course = plan.course if plan.course in COURSES else "tamna"
            self.res.request_text = self._request_text(course)
            self.res.answer = ("아직 불러온 데이터가 없습니다. ② 데이터 탭에서 센터 CSV 폴더(또는 시연용 가상데이터)를 불러오세요. "
                               "센터에서 받을 파일은 아래 요청 문안에 정리했습니다.")
            return
        self.t_check_data(plan.course if plan.course in COURSES else None)
        if plan.course in COURSES:
            mk = dict(plan.market)
            if plan.course == "market" and not mk.get("bjd"):
                self.res.answer = "후보 지역(법정동)을 알려 주시면 시장분석을 시작합니다. 예) '애월읍에 카페 창업'"
                return
            out = self.t_run_course({"course": plan.course, **mk})
            if out.get("error"):
                self.res.answer = out["error"].split(" 사용자에게")[0]
                return
            if out.get("처방 설계"):                        # 문제 확인에서 멈추지 않고 실행물까지
                self.t_make_action_pack({"scenario": "auto"})
        else:
            for spec in plan.analyses:
                self.t_run_analysis(spec.model_dump(exclude_none=True))
        self._rule_answer()

    def _rule_answer(self):
        """규칙 기반 답: 진단 → 처방 설계 → 실행물 → 효과 확인 순서(숫자는 자리표시자, 이 PC에서 채움)."""
        parts = []
        if self.res.outputs:
            out = self.res.outputs[-1]
            d = out.report.extra.get("drafts", {})
            summ, rx = d.get("summary", []), d.get("rx", [])
            dg = out.report.extra.get("diag")
            pol = out.report.extra.get("policy")
            if dg:                                         # 동네 하나를 물었으면 그 동네 진단을 먼저
                u, t = dg["unit"], dg["type"]
                parts.append(f"[진단] {u}({dg['group']})은(는) '{t}' 동네입니다.".replace("은(는)", "은" if _has_batchim(u) else "는"))
                if dg.get("lead"):
                    parts.append(f"주요 지표는 {dg['lead']}입니다.")
                parts += [x for x in summ if x.startswith(("원도심 도민 소비", "0~14세 거주인구"))][:1]
            elif pol:
                parts.append("[진단] " + " ".join(x for x in summ[:2]))
            else:
                parts += summ[:3] + rx[:2]
            if pol:
                # 효과 추정(반응도·앞당긴 소비)은 설계의 근거이므로 처방 앞에 붙인다
                parts += [x for x in summ if x.startswith("적립률 1%p당")][:1]
                parts += [x for x in summ if "앞당긴 소비(" in x][:1]
                parts += self._policy_answer(pol, out.report.extra.get("effect"))
            elif dg:
                parts.append(f"먼저 할 일: {dg['rx']}. 확인 방법: {dg['check']}.")
        for r, _img, _fs in self.res.free:
            parts += r.sentences[:2]
        draft = " ".join(p.strip() for p in parts if p.strip())
        if self.cat is not None and self.cat.synthetic:
            draft += " (가상데이터 결과)"
        self.res.answer_draft = draft
        self.res.answer = self.res.facts.render(draft) if draft else "분석 결과를 ③ 분석 결과 탭에서 확인하세요."
        self.res.next_steps = self._default_next()
        self._log("answer", "답변 작성(규칙 기반 요약)")

    def _policy_answer(self, pol: dict, effect: dict | None) -> list[str]:
        P = lambda fid: "{{" + fid + "}}"
        k, sf, A = pol["rec"], pol["sf"], pol["scen"]["A"]
        S = pol["scen"][k]
        out = [f"[처방] {k}안 {S.name}: {pol['rule_ph'][k]}."]
        if A.exhaust:
            out.append(f"지금 방식이면 적립 예산 {P(pol['f_budget'])}이 {P(sf['A']['exm'])}께 바닥나지만, 이 안은 "
                       f"{P(sf[k]['last'])} 동안 이어지고 전통시장 몫이 {P(sf['A']['mkt'])}에서 {P(sf[k]['mkt'])}(으)로 바뀝니다.")
        else:
            out.append(f"이 안이면 전통시장 몫이 {P(sf['A']['mkt'])}에서 {P(sf[k]['mkt'])}(으)로 바뀝니다.")
        if pol.get("focus") and "focus_amt" in sf[k]:
            out.append(f"{pol['focus']}에 돌아가는 적립액은 {P(sf['A']['focus_amt'])}에서 {P(sf[k]['focus_amt'])}(으)로 바뀝니다.")
        if effect and k == "C′" and effect.get("f_e"):
            out.append(f"전후 효과({P(effect['f_e'])}, p≈{P(effect['f_p'])})를 반영해 다음 배분(C′안)을 다시 계산했습니다.")
        pk = next((x for x in reversed(self.res.packs)), None)
        if pk is not None:
            out.append(f"[실행물] {pk.scenario}안으로 도청 개선안 초안, 상인회 안내문, 홍보 문구(현수막·SNS·문자), 어르신 안내문, "
                       "추진 일정·체크리스트를 묶어 만들었습니다.")
        out.append(f"[효과 확인] 시행 12주 뒤 “{pol['question']}”라고 물으면 효과를 판정하고 다음 배분을 고칩니다.")
        return out

    def _default_next(self) -> list[str]:
        nxt = []
        if self.res.outputs:
            rep = self.res.outputs[-1].report
            nxt.append("④ 반출 신청 탭에서 보고서·실행물 묶음을 반출 묶음으로 만들기" if self.res.packs
                       else "④ 반출 신청 탭에서 보고서·그림을 반출 묶음으로 만들기")
            pol = rep.extra.get("policy")
            if pol and not rep.extra.get("effect"):
                nxt.append(f"시행 뒤 '{pol['question']}'로 효과 확인")
            elif "탐나는전" in rep.title and not rep.extra.get("effect"):
                nxt.append("캐시백·지원 정책 시작 월을 알려 주면 전통시장 전후 효과까지 분석")
            if pol and not pol.get("user_budget"):
                nxt.append("실제 예산을 넣어 다시 계산(예: '예산 400억 원으로 1년 동안')")
            types = rep.extra.get("dongne_types") or {}
            first = next((us[0] for t, us in types.items() if us), None)
            if first and not rep.extra.get("diag_unit"):
                nxt.append(f"'{first} 동네 진단해 줘'처럼 동네 하나를 골라 진단 카드 만들기")
        return nxt[:3]


def _has_batchim(word: str) -> bool:
    ch = word.strip()[-1:]
    return bool(ch) and "가" <= ch <= "힣" and (ord(ch) - 0xAC00) % 28 != 0


def placeholders_in(text: str) -> list[str]:
    return [m.group(1) for m in PLACEHOLDER.finditer(text)]
