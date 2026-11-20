"""질문 → 분석 계획.

1) 규칙 기반(항상 동작, AI 없이도): 법정동·업종·지표·나눌 기준 낱말을 찾아 코스 또는 분석 스펙을 만든다.
2) AI(선택): 데이터 목록·지표·차원 '이름'(공개 명세서 정보)만 주고 JSON 계획을 받는다. 센터 값은 보내지 않는다.
   AI 계획은 pydantic 스키마와 시맨틱 레이어로 검증하고, 통과하지 못하면 규칙 기반 계획을 쓴다.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources

from pydantic import ValidationError

from dataolle.ai.gateway import Gateway, extract_json, sanitize_question
from dataolle.analysis.semantic import MODELS
from dataolle.analysis.spec import AnalysisSpec

BIZ_KEYWORDS = [
    (("카페", "커피", "음료", "디저트카페", "찻집"), "비알콜 음료점업"),
    (("한식", "식당", "음식점", "밥집", "국밥", "고깃집", "흑돼지"), "한식 음식점업"),
    (("중식", "중국집", "짜장"), "중식 음식점업"),
    (("일식", "초밥", "횟집", "스시"), "일식 음식점업"),
    (("양식", "레스토랑", "파스타", "브런치"), "서양식 음식점업"),
    (("피자", "햄버거", "샌드위치", "패스트푸드"), "피자, 햄버거, 샌드위치 및 유사 음식점업"),
    (("술집", "주점", "호프", "펍", "이자카야"), "기타 주점업"),
    (("빵", "베이커리", "제과"), "빵 및 과자류 소매업"),
    (("편의점",), "체인화 편의점"),
    (("슈퍼", "마트"), "슈퍼마켓"),
    (("호텔",), "호텔업"),
    (("펜션", "모텔", "여관", "게스트하우스", "숙박"), "여관업"),
    (("렌터카", "렌트카", "자동차 임대"), "자동차 임대업"),
    (("기념품", "선물"), "관광 민예품 및 선물용품 소매업"),
    (("화장품",), "화장품 및 방향제 소매업"),
    (("골프",), "골프장 운영업"),
    (("마사지",), "마사지업"),
    (("주유",), "차량용 주유소 운영업"),
]
COURSE_WORDS = ("창업", "사업계획", "시장분석", "상권", "입점", "오픈", "개업", "가게를 열", "차리")
STATUS_WORDS = ("현황", "프로필", "지역 정보", "동네 정보", "현황자료", "한눈에", "개요")
EFFECT_WORDS = ("전후", "이후", "효과", "영향", "늘었", "증가했", "줄었", "감소했", "달라졌", "변했")
TAMNA_WORDS = ("탐나는전", "지역화폐", "지역 화폐", "제주사랑상품권", "캐시백", "적립", "인센티브율",
               "선결제", "앞당긴 소비")
# 동네 살리기(원도심·고령화·인구·학령인구·전통시장) → 탐나는전 동네 살리기 코스
DONGNE_WORDS = ("원도심", "구도심", "신도심", "동네 살리", "동네를 살리", "살릴 수", "고령화", "노인", "어르신", "인구 감소", "인구감소",
                "인구 불균형", "인구불균형", "학령인구", "폐교", "작은 학교", "소규모 학교", "아이가 줄", "아이들이 줄", "아이 줄",
                "전통시장", "오일장", "장날", "급식", "공실", "빈 가게", "빈 점포", "상권 쇠퇴", "동네 진단", "진단해")
GROUP_WORDS = (("원도심", "원도심"), ("구도심", "원도심"), ("신도심", "신도심"))
MEASURE_WORDS = [(("객단가", "건당"), "amt_per_cnt"), (("1인당",), "amt_per_user"), (("건수", "몇 건", "결제 횟수"), "cnt"),
                 (("이용자", "손님 수", "고객 수"), "users"), (("매출", "이용금액", "소비", "결제액", "금액"), "amt")]
ROW_WORDS = [(("월별", "추이", "추세", "변화", "달마다", "계절"), "month"), (("분기",), "quarter"), (("요일",), "week"),
             (("시간대", "몇 시"), "time"), (("연령", "나이", "세대"), "age"), (("성별", "남녀"), "sex"),
             (("관광객", "도민", "내국인"), "tour_type"), (("외국인", "국적"), "user_type"),
             (("동별", "지역별", "읍면동", "어디", "법정동별", "동 비교", "지역 비교", "동네별", "상위"), "bjd"),
             (("권역",), "region"), (("업종별",), "biz")]
TRANSFORM_WORDS = [(("비중", "구성비", "점유율", "몇 %", "비율"), "share"), (("전년", "작년 대비", "증가율", "증감"), "yoy"),
                   (("순위", "랭킹"), "rank")]


@lru_cache(maxsize=1)
def _codes() -> dict:
    return json.loads(resources.files("dataolle.spec").joinpath("data/codes.json").read_text(encoding="utf-8"))


def bjd_names() -> list[str]:
    names = []
    for city, groups in _codes()["bjd"].items():
        for lst in (groups.values() if isinstance(groups, dict) else [groups]):
            for n in lst:
                n = n if isinstance(n, str) else n[-1]
                if n == "월평동":
                    n = f"월평동({city})"
                if n not in names:
                    names.append(n)
    return names


def biz_names() -> list[str]:
    return [b[1] for b in _codes()["biz42"]]


def find_bjd(q: str) -> str | None:
    hits = []
    for n in bjd_names():
        base = n.split("(")[0]
        variants = {base, base.replace("일동", "1동").replace("이동", "2동").replace("삼동", "3동")}
        for v in variants:
            if v and v in q:
                hits.append((len(v), n))
    if not hits:
        return None
    hits.sort(reverse=True)
    best = hits[0][1]
    if best.startswith("월평동"):
        return "월평동(서귀포시)" if "서귀포" in q else "월평동(제주시)"
    return best


def unit_names() -> list[str]:
    """행정동(동네) 이름: '아라동', '삼양동'처럼 법정동 목록에 없는 이름도 찾기 위해."""
    from dataolle.analysis.semantic import UNIT_GROUP
    out = []
    for u in UNIT_GROUP:
        out += [u] if "·" not in u else [u] + u.replace("동", "").split("·")
    return sorted(set(out), key=len, reverse=True)


def find_unit(q: str) -> str | None:
    from dataolle.courses.dongne import unit_of_name
    for n in unit_names():
        if n.endswith(("동", "읍", "면")) and n in q:
            return unit_of_name(n) or n
        if not n.endswith(("동", "읍", "면")) and (n + "동") in q:          # '송산동' 등 합쳐진 원도심 동네의 한 동
            return unit_of_name(n + "동")
    return None


def find_biz(q: str) -> str | None:
    for name in biz_names():
        if name in q:
            return name
    for words, name in BIZ_KEYWORDS:
        if any(w in q for w in words):
            return name
    return None


def _pick(q: str, table) -> str | None:
    for words, key in table:
        if any(w in q for w in words):
            return key
    return None


@dataclass
class Plan:
    course: str                                  # market | free
    question: str
    market: dict = field(default_factory=dict)   # {"biz":..., "bjd":...}
    analyses: list[AnalysisSpec] = field(default_factory=list)
    notes: str = ""
    by_ai: bool = False
    problems: list[str] = field(default_factory=list)

    @property
    def datasets(self) -> list[str]:
        if self.course == "market":
            return ["1-08", "1-12", "1-14", "1-19", "1-24", "4-11"]
        if self.course == "status":
            return ["1-08", "1-12", "1-14", "4-11"]
        if self.course == "before_after":
            return ["1-08"]
        if self.course == "tamna":
            return ["1-32", "1-08", "4-11", "4-10", "5-02"]
        out = []
        for a in self.analyses:
            for ds in MODELS[a.dataset].dataset_ids:
                if ds not in out:
                    out.append(ds)
        return out


def find_period(q: str) -> list[str] | None:
    """'2025년 8월' → [202508, 202508], '2025년 3~5월' → [202503, 202505], '2025년' → [202501, 202512]."""
    m = re.search(r"((?:19|20)\d{2})\s*년\s*(\d{1,2})\s*[~\-]\s*(\d{1,2})\s*월", q)
    if m:
        y, a, b = m.group(1), int(m.group(2)), int(m.group(3))
        return [f"{y}{min(a, b):02d}", f"{y}{max(a, b):02d}"]
    m = re.search(r"((?:19|20)\d{2})\s*년\s*(\d{1,2})\s*월", q)
    if m:
        ym = f"{m.group(1)}{int(m.group(2)):02d}"
        return [ym, ym]
    m = re.search(r"((?:19|20)\d{2})(0[1-9]|1[0-2])(?!\d)", q)
    if m:
        return [m.group(0), m.group(0)]
    m = re.search(r"((?:19|20)\d{2})\s*년", q)
    if m:
        return [f"{m.group(1)}01", f"{m.group(1)}12"]
    return None


def find_top_n(q: str) -> int | None:
    m = re.search(r"(?:상위|top|TOP|Top)\s*(\d{1,2})", q)
    return max(1, min(30, int(m.group(1)))) if m else None


def rule_plan(question: str) -> Plan:
    q = question.strip()
    bjd, biz = find_bjd(q), find_biz(q)
    specific = any(_pick(q, t) for t in (MEASURE_WORDS[:4], ROW_WORDS, TRANSFORM_WORDS))
    per = find_period(q)
    tamna = any(w in q for w in TAMNA_WORDS)
    dongne = any(w in q for w in DONGNE_WORDS) and not (biz and any(w in q for w in COURSE_WORDS))
    if tamna or dongne:
        unit = find_unit(q)
        grp = next((g for w, g in GROUP_WORDS if w in q), None)
        if bjd and bjd.split("(")[0] in q:                 # 법정동 이름 그대로 물음(예: 애월읍, 일도일동)
            scope, value = "emd", bjd
        elif unit:                                          # 행정동 이름으로 물음(예: 일도1동, 아라동)
            scope, value = "hd", unit
        elif bjd:
            scope, value = "emd", bjd
        elif grp and not tamna:
            scope, value = "all", None                  # '원도심' 전체 질문은 제주 전체 분석 안에서 동네 유형별로 비교
        else:
            scope, value = (("city", "서귀포시") if "서귀포" in q else (("city", "제주시") if "제주시" in q else ("all", None)))
        ev = per[0] if (per and per[0] == per[1] and any(w in q for w in EFFECT_WORDS)) else None
        focus = f"{value} 동네 진단 카드와 " if scope in ("emd", "hd") else ""
        return Plan("tamna", q, {"scope": scope, "value": value, "event": ev},
                    notes=f"탐나는전(센터 전용)·카드·거주인구·날씨 자료를 동네 단위로 묶는 '탐나는전 캐시백 설계' 코스로 진행합니다. "
                          f"{focus}원도심·신도심 상권, 고령화·학령인구, 동네 유형별 처방까지 만듭니다"
                          + (f" — {ev[:4]}년 {int(ev[4:])}월 전후 효과도 분석합니다." if ev else "."))
    if bjd and per and per[0] == per[1] and any(w in q for w in EFFECT_WORDS):
        return Plan("before_after", q, {"bjd": bjd, "biz": biz, "event": per[0]},
                    notes=f"'{per[0][:4]}년 {int(per[0][4:])}월' 전후를 비교하는 '논문용 전후 비교' 코스로 진행합니다(이중차분).")
    if bjd and not biz and any(w in q for w in STATUS_WORDS):
        return Plan("status", q, {"bjd": bjd}, notes="지역 하나의 소비·이용자·유동인구를 정리하는 '읍면동 현황자료' 코스로 진행합니다.")
    if biz and bjd and (any(w in q for w in COURSE_WORDS) or (not specific and len(q) < 40)):
        return Plan("market", q, {"biz": biz, "bjd": bjd}, notes="업종과 후보 지역이 있어 '사업계획서 시장분석' 코스로 진행합니다.")
    if biz and any(w in q for w in COURSE_WORDS) and not bjd:
        return Plan("market", q, {"biz": biz, "bjd": None}, notes="후보 지역(법정동)을 알려 주시면 시장분석을 시작합니다.",
                    problems=["후보 법정동을 찾지 못했습니다."])
    dataset = "card"
    if any(w in q for w in ("유동인구", "방문인구", "거주인구", "사람이 많")):
        dataset = "pop"
    measure = _pick(q, MEASURE_WORDS) or "amt"
    rows = _pick(q, ROW_WORDS)
    transform = _pick(q, TRANSFORM_WORDS) or "none"
    if dataset == "pop":
        measure = "visit"
        rows = rows if rows in ("month", "age", "sex") else ("hour" if any(w in q for w in ("시간", "몇 시")) else "month")
        filters = {}
    else:
        if rows == "week":
            dataset = "card_week"
        elif rows == "time":
            dataset = "card_time"
        filters = {}
        if bjd:
            filters["bjd"] = [bjd]
        if biz:
            filters["biz"] = [biz]
        if rows is None:
            rows = "month" if (bjd or biz) else "bjd"
        if rows == "bjd" and bjd:
            filters.pop("bjd", None)
    if transform == "yoy" and rows not in ("month", "quarter", "year"):
        transform = "none"
    top_n = find_top_n(q) or (15 if rows in ("bjd", "biz") else None)
    spec = AnalysisSpec(dataset=dataset, measure=measure, rows=rows, filters=filters, transform=transform,
                        top_n=top_n if rows in ("bjd", "biz", "region") else None, period=find_period(q))
    return Plan("free", q, analyses=[spec], notes="질문의 낱말로 분석 기준을 정했습니다(규칙 기반).")


def catalog_text() -> str:
    """AI에 줄 데이터 안내(공개 명세서 정보만: 이름·지표·차원·코드값)."""
    lines = []
    for m in MODELS.values():
        meas = ", ".join(f"{k}={v.label}" for k, v in m.measures.items())
        dims = ", ".join(f"{k}={d.label}" + (f"[{'/'.join(d.order)}]" if d.order else "") for k, d in m.dims.items())
        lines.append(f"- {m.key} ({'/'.join(m.dataset_ids)} {m.title})\n  지표: {meas}\n  차원: {dims}")
    lines.append("- 업종(biz) 값: " + ", ".join(biz_names()))
    lines.append("- 법정동(bjd) 값: " + ", ".join(bjd_names()))
    return "\n".join(lines)


PLAN_SYSTEM = """당신은 제주빅데이터센터 데이터 분석 계획을 세우는 도우미입니다. 아래 데이터 안내에 있는 키만 사용하세요.
결과는 JSON 하나로만 답합니다:
{"course": "tamna" | "market" | "status" | "before_after" | "free",
 "market": {"biz": 업종 값, "bjd": 법정동 값, "event": "YYYYMM", "scope": "all|city|emd", "value": 시 또는 법정동},
   // tamna: 탐나는전·지역화폐·전통시장·원도심·고령화·학령인구·동네 살리기(scope·value, 정책 효과면 event)
   // market: 창업·사업계획서·상권 분석(biz·bjd 필요) / status: 한 지역 현황(bjd) / before_after: 특정 달 전후 효과 비교(bjd·event, biz 선택)
 "analyses": [{"dataset":..,"measure":..,"rows":..,"cols":..,"filters":{차원:[값]},"transform":"none|share|yoy|index|rank","top_n":..}],  // free일 때 1~4개
 "notes": "계획 설명 1~2문장(한국어)"}
질문 속 수치는 [수치]로 가려져 있을 수 있습니다. 값을 추측하지 말고 분석 계획만 세우세요."""


def ai_plan(question: str, gateway: Gateway | None) -> Plan:
    """AI 계획(실패하면 규칙 기반)."""
    base = rule_plan(question)
    if gateway is None or not gateway.enabled:
        return base
    clean, masked = sanitize_question(question)
    user = f"[데이터 안내]\n{catalog_text()}\n\n[질문]\n{clean}"
    resp = gateway.call("분석 계획", PLAN_SYSTEM, user, max_tokens=900, masked_numbers=masked)
    data = extract_json(resp.text) if resp else None
    if not isinstance(data, dict):
        base.problems.append("AI 계획을 받지 못해 규칙 기반 계획을 사용합니다.")
        return base
    try:
        course = data.get("course")
        if course == "tamna":
            mk = data.get("market") or {}
            scope, value, ev = mk.get("scope") or "all", mk.get("value"), str(mk.get("event") or "")
            if scope not in ("all", "city", "emd") or (scope == "city" and value not in ("제주시", "서귀포시")) \
                    or (scope == "emd" and value not in bjd_names()):
                raise ValueError("탐나는전 코스 범위 값이 목록에 없음")
            ev = ev if re.fullmatch(r"(19|20)\d{2}(0[1-9]|1[0-2])", ev) else None
            return Plan("tamna", question, {"scope": scope, "value": value if scope != "all" else None, "event": ev},
                        notes=str(data.get("notes", "")), by_ai=True)
        if course in ("market", "status", "before_after"):
            mk = data.get("market") or {}
            biz, bjd, ev = mk.get("biz"), mk.get("bjd"), str(mk.get("event") or "")
            if bjd not in bjd_names() or (biz and biz not in biz_names()):
                raise ValueError("업종·법정동 값이 목록에 없음")
            if course == "market" and biz:
                return Plan("market", question, {"biz": biz, "bjd": bjd}, notes=str(data.get("notes", "")), by_ai=True)
            if course == "status":
                return Plan("status", question, {"bjd": bjd}, notes=str(data.get("notes", "")), by_ai=True)
            if course == "before_after" and re.fullmatch(r"(19|20)\d{2}(0[1-9]|1[0-2])", ev):
                return Plan("before_after", question, {"bjd": bjd, "biz": biz, "event": ev}, notes=str(data.get("notes", "")),
                            by_ai=True)
            raise ValueError("코스에 필요한 값이 부족함")
        specs, probs = [], []
        for a in (data.get("analyses") or [])[:4]:
            try:
                s = AnalysisSpec(**{k: v for k, v in a.items() if v is not None})
            except ValidationError as e:
                probs.append(str(e).splitlines()[0])
                continue
            p = s.problems()
            if p:
                probs += p
                continue
            specs.append(s)
        if not specs:
            raise ValueError("; ".join(probs) or "분석 없음")
        return Plan("free", question, analyses=specs, notes=str(data.get("notes", "")), by_ai=True, problems=probs)
    except (ValueError, TypeError, AttributeError) as e:
        base.problems.append(f"AI 계획 검증 실패({e}) → 규칙 기반 계획을 사용합니다.")
        return base


def clean_question(q: str) -> str:
    return re.sub(r"\s+", " ", q).strip()
