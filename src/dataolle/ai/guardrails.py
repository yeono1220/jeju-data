"""입력 가드레일 — 질문을 분석에 넘기기 전에 이 PC에서 먼저 거른다(AI 호출 전, 규칙 기반).

  막는 것  ① 원본을 그대로 달라는 요청(반출 불가 데이터의 원본·발췌는 금지)
           ② 개인·개별 가게를 가려 달라는 요청(자료에 없고, 있어도 반출 기준에 어긋남)
           ③ 지시를 무시하라는 식의 요청(프롬프트 주입)
           ④ 센터 데이터로 답할 수 없는 범위 밖 질문(엉뚱한 표를 내놓지 않게)
  막지 않는 것  예산·기간·동네·업종을 묻는 분석 질문 전부(tests/data/questions.json 50문항으로 확인)
출력 쪽 가드레일은 따로 있다: 반출 검수(sdc), 레벨 0 가림막(redact), AI 답의 숫자 검증(narrator·agent).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

RAW = re.compile(r"원본|원자료|raw|csv\s*(를|을)?\s*(그대로|보여|출력|내보)|\d+\s*(줄|행)\s*(만|을|를)?\s*(그대로|보여|출력)|행\s*그대로|전체\s*(데이터|자료)\s*(를|을)?\s*(보여|출력|내려)", re.I)
PERSON = re.compile(r"사람\s*이름|이름\s*(을|를)?\s*알려|개인\s*별|개인\s*정보|누구인지|명단|전화\s*번호|카드\s*번호|주민\s*번호|"
                    r"(가게|가맹점|업체|상호)\s*(이름|명)|어느\s*(가게|가맹점|업체)가")
INJECT = re.compile(r"(이전|앞|위)\s*(의)?\s*(지시|명령|규칙)|지시\s*(는|를|을)?\s*무시|시스템\s*프롬프트|ignore\s+(all\s+)?previous|system\s+prompt", re.I)
SCOPE = ("탐나는전", "지역화폐", "캐시백", "적립", "예산", "소비", "카드", "매출", "이용", "사용", "결제", "인구", "유동", "거주", "방문",
         "관광", "업종", "동네", "상권", "시장", "가맹", "창업", "현황", "효과", "전후", "비중", "추이", "추세", "통계", "분석", "데이터",
         "자료", "날씨", "비 오는", "강수", "시간대", "요일", "건수", "금액", "고령", "어르신", "아이", "학령", "읍", "면", "동 ", "제주",
         "서귀포", "카페", "음식점", "학원", "선결제", "쏠림", "사각지대", "원도심", "보고서", "혜택", "역진", "지원", "정책",
         "전통", "상인", "소상공인", "축제", "버스", "와이파이")

HELP = ("이 프로그램은 제주빅데이터센터 데이터로 탐나는전 캐시백 설계(예산 배분·소진일·전통시장 몫·쏠림), 정책 전후 효과, "
        "동네 현황, 창업 시장분석을 합니다. 예: \"예산 400억 원으로 탐나는전 적립을 1년 내내 이어 가려면 어떻게 나눠야 해요?\"")


@dataclass
class Guard:
    code: str            # raw | person | inject | scope
    message: str


def check(question: str, plan_course: str | None = None, vocab: list[str] | None = None) -> Guard | None:
    """막아야 하면 Guard, 아니면 None. plan_course: 규칙 계획기가 고른 코스(자유 질문 'free'일 때만 범위 검사)."""
    q = (question or "").strip()
    if not q:
        return Guard("empty", "질문을 한 줄 적어 주세요. " + HELP)
    if INJECT.search(q):
        return Guard("inject", "지시를 바꾸라는 요청은 따르지 않습니다. 분석할 질문을 다시 적어 주세요. " + HELP)
    if RAW.search(q):
        return Guard("raw", "원본 데이터를 그대로 보여 주거나 내보낼 수 없습니다(센터 반출 기준: 원본·발췌 금지). "
                            "대신 동네·업종·기간으로 묶은 집계와 그림, 보고서를 만들어 드립니다. " + HELP)
    if PERSON.search(q):
        return Guard("person", "개인이나 개별 가게를 가려내는 분석은 하지 않습니다. 자료에 그런 정보가 없고, 반출 기준에도 어긋납니다. "
                               "동네·업종·전통시장 단위로는 볼 수 있습니다. " + HELP)
    if plan_course in (None, "free"):
        words = SCOPE + tuple(vocab or ())
        if not any(w and w in q for w in words):
            return Guard("scope", "센터 데이터로 답할 수 있는 질문이 아닙니다. " + HELP)
    return None


__all__ = ["check", "Guard", "HELP"]
