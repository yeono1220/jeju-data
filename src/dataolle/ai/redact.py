"""레벨 0 반출 가림막: 센터 데이터에서 나온 '이름·문장'을 외부 AI에 보내지 않는다.

숫자만 가리면 충분하지 않다. '사각지대 1위 신효동', '가장 높은 연월(2025년 8월)', '비 오는 날 전통시장 사용이 줄어듭니다'처럼
이름·순위·방향도 센터 데이터를 조회해서 나온 정보다(11/10 게시판 답변: 질문 속 센터 값도 외부 전송으로 본다).

  가리는 것  ① 센터 데이터의 범주 값(동네·법정동·전통시장·업종·매입사·메뉴·요일·연령대·시간대 이름)
             ② 순위·최댓값을 설명하는 괄호 안 값('1위 업종(일반음식점)', '가장 적은 요일(일요일)')
             ③ 규칙 기반 초안 문장 통째(방향 단어 '늘었다·줄었다·가장'이 섞여 있으므로) — opaque()
             ④ 그때그때 더하는 값(추천안, 추천 이유, 동네 유형별 목록 등) — token()
  가리지 않는 것  사용자 질문에 이미 있는 말(예: 질문한 '일도1동'), 데이터셋 이름·기간, 절 제목, 지표 이름

가린 자리는 {{t12}}로 보낸다. AI는 이 토큰을 그대로 옮겨 쓰고, 돌아온 글은 unredact()로 이 PC에서 원래 말로 되돌린다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

TOKEN = re.compile(r"\{\{\s*(t\d+)\s*\}\}")

# 센터 범주 값을 모을 차원(동네·시장·업종처럼 '어디·무엇'을 가리키는 것). 시·도심 구분·동네 유형 같은 고정 묶음은 제외.
VOCAB_DIMS = ("bjd", "hd", "emd", "market", "biz", "induty", "card", "pos_mid", "pos_biz",
              "menu_large", "menu_mid", "menu_small", "menu", "age", "time", "week", "weekday")
_SKIP = {"전통시장", "일반 가맹점", "기타", "합계", "전체", "제주시", "서귀포시", "제주", "그 밖의 동", "원도심", "신도심", "읍면",
         "평일", "주말", "NA"}
_WEEKDAYS = ("월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일")
# 순위·최댓값 설명 뒤 괄호: '가장 높은 연월(2025년 8월)', '1위 시간대(11~14시)', '최다 시각(18시)'
_RANK_PAREN = re.compile(r"((?:가장\s(?:높은|낮은|적은|많은|큰|작은)|1위|최다|최고|최저|핵심)\s?[가-힣×·\s]{0,12}?)\(([^(){}]{1,40})\)")


def center_vocab(catalog) -> list[str]:
    """불러온 센터 데이터의 범주 값 목록(가릴 말). 카탈로그가 없으면 요일만."""
    words: set[str] = set(_WEEKDAYS)
    if catalog is not None:
        from dataolle.analysis.semantic import MODELS
        for key in list(getattr(catalog, "tables", {})):
            model = MODELS.get(key)
            if model is None:
                continue
            for dim in VOCAB_DIMS:
                if dim not in model.dims:
                    continue
                try:
                    words.update(catalog.values(key, dim, limit=2000))
                except Exception:                       # 차원 식이 이 판 파일에 없을 때
                    continue
    return sorted({w.strip() for w in words if w and len(w.strip()) >= 2 and w.strip() not in _SKIP
                   and not re.fullmatch(r"[\d.,\s]+", w.strip())}, key=len, reverse=True)


@dataclass
class Redactor:
    vocab: list[str] = field(default_factory=list)
    exempt: set[str] = field(default_factory=set)
    tokens: dict[str, str] = field(default_factory=dict)      # t1 → 원래 말
    labels: dict[str, str] = field(default_factory=dict)      # t1 → 무엇인지(AI에 보여 줄 설명)
    _rev: dict[str, str] = field(default_factory=dict)
    _rx: re.Pattern | None = None
    count: int = 0                                            # 가린 횟수(감사 로그용)

    def reset(self, question: str = "", vocab: list[str] | None = None):
        """새 질문마다: 질문에 있는 말은 가리지 않는다."""
        if vocab is not None:
            self.vocab = list(vocab)
        q = question or ""
        self.exempt = {w for w in self.vocab if w in q}
        self.tokens.clear(), self.labels.clear(), self._rev.clear()
        self.count = 0
        self._compile()

    def add_terms(self, terms):
        new = [t for t in terms if t and len(str(t)) >= 2 and str(t) not in self.vocab and str(t) not in _SKIP]
        if new:
            self.vocab = sorted(set(self.vocab) | {str(t) for t in new}, key=len, reverse=True)
            self._compile()

    def _compile(self):
        words = [w for w in self.vocab if w not in self.exempt]
        self._rx = re.compile("|".join(re.escape(w) for w in words)) if words else None

    # ------------------------------------------------------------------ 토큰 만들기
    def token(self, text: str, label: str = "", dedupe: bool = False) -> str:
        """값 하나를 토큰으로. dedupe=True(이름 가리기)면 같은 말은 같은 토큰.
        판정·예/아니오처럼 값이 같다는 사실 자체가 정보인 것은 매번 새 토큰(dedupe=False)."""
        text = str(text)
        if dedupe and text in self._rev:
            return "{{" + self._rev[text] + "}}"
        tid = f"t{len(self.tokens) + 1}"
        self.tokens[tid] = text
        if dedupe:
            self._rev[text] = tid
        if label:
            self.labels[tid] = label
        self.count += 1
        return "{{" + tid + "}}"

    def opaque(self, sentence: str, label: str = "초안 문장") -> str:
        """문장 통째를 토큰으로(방향 단어·순위가 섞인 규칙 기반 초안)."""
        return self.token(sentence, label)

    # ------------------------------------------------------------------ 가리기·되돌리기
    def redact(self, text: str) -> str:
        if not text:
            return text
        out = text
        if self._rx is not None:
            out = self._rx.sub(lambda m: self.token(m.group(0), "센터 데이터의 항목 이름", dedupe=True), out)
        # 이름을 먼저 가린 뒤 순위 괄호를 본다('1위 {{t3}} 침투지수(도민 소비 대비)'는 사이에 토큰이 있어 걸리지 않음)
        return _RANK_PAREN.sub(lambda m: f"{m.group(1)}({self.token(m.group(2), '순위·최댓값의 항목 이름', dedupe=True)})", out)

    def unredact(self, text: str) -> str:
        """AI가 돌려준 글의 {{t12}}를 원래 말로. 모르는 토큰은 지운다."""
        if not text or "{{" not in text:
            return text
        prev = None
        while prev != text and "{{t" in text.replace(" ", ""):
            prev = text
            text = TOKEN.sub(lambda m: self.tokens.get(m.group(1), ""), text)
        return text

    def unredact_obj(self, obj):
        if isinstance(obj, str):
            return self.unredact(obj)
        if isinstance(obj, list):
            return [self.unredact_obj(x) for x in obj]
        if isinstance(obj, dict):
            return {k: self.unredact_obj(v) for k, v in obj.items()}
        return obj

    def legend(self) -> list[dict]:
        """AI에 줄 토큰 설명(값 없음)."""
        return [{"토큰": "{{" + t + "}}", "뜻": self.labels.get(t, "센터 데이터에서 나온 말")} for t in self.tokens]


__all__ = ["Redactor", "center_vocab", "TOKEN", "VOCAB_DIMS"]
