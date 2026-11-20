"""실행물 묶음 — 처방 설계(courses/policy.py)에서 고른 적립 배분안을 바로 쓸 수 있는 문서로.

진단 → 처방 설계 → **실행물** → 효과 확인 중 세 번째 단계. 탐나는전 보고서의 '처방 설계' 결과(report.extra["policy"])로
  1. 한 장 요약                 문제·처방·기대 효과·할 일·확인 방법
  2. 개선안 초안(도청 내부 검토) 추진 배경 / 현황과 문제 / 개선안 / 기대 효과 / 소요 예산·월별 적립 계획 / 일정 / 효과 확인 / 검토 사항
  3. 대안 비교표                A~D(C′)안 표·그림, 안별 장단점
  4. 상인회 안내문              무엇이·왜 바뀌는지, 상인회가 할 일
  5. 손님 홍보 문구             현수막 3안, SNS 글, 문자(단문·장문), 관광객 안내(국문·영문)
  6. 어르신 안내문(주민센터)     큰 글씨, '1만 원 쓰면 얼마 돌려받나'
  7. 추진 일정·체크리스트        D-60 ~ D+84, 담당, 확인 칸
  8. 효과 확인 질문             다음 방문 때 그대로 붙여 넣을 질문과 미리 요청할 자료
를 PDF(반출용)·DOCX(센터 안 편집용)로 만든다.

모든 수치는 사실 저장소의 자리표시자로 쓰고 이 PC에서 채운다. AI를 켜면 홍보 문구만 다듬는다(자리표시자만 보냄,
숫자가 섞이거나 길이를 넘으면 그 문구는 규칙 기반 초안으로 되돌림).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from dataolle.analysis.facts import PLACEHOLDER, FactStore
from dataolle.analysis.fmt import fmt_value
from dataolle.analysis.josa import josa
from dataolle.courses.policy import CAP, DECISION_TEXT, NAMES, rule_text
from dataolle.report.document import (Bullets, Callout, Figure, Paragraph, Report, Section, Sheet, Subhead, Table, to_docx,
                                      to_html, to_pdf)
from dataolle.sdc.engine import AggKind

COPY_KEYS = ("banner1", "banner2", "banner3", "sns", "sms_short", "sms_long", "tour_ko", "tour_en")
LIMITS = {"banner1": 30, "banner2": 30, "banner3": 30, "sms_short": 45, "sms_long": 300, "sns": 400, "tour_ko": 200,
          "tour_en": 300}

COPY_SYSTEM = """당신은 제주특별자치도의 홍보 문구를 다듬는 카피라이터입니다. 탐나는전(제주 지역화폐) 적립 안내 문구 초안을 받습니다.
규칙:
1. 숫자를 직접 쓰지 마세요. 적립률·금액은 초안에 있는 자리표시자({{f12}})를 그대로 옮겨 씁니다. 새 자리표시자를 만들지 마세요.
   요일·날짜 표현(장날, 비 오는 날, 1월)은 써도 됩니다.
   {{t3}} 같은 토큰은 시장 이름 등 센터 데이터에서 나온 말을 가린 것입니다. 지우거나 바꾸지 말고 그대로 옮겨 씁니다.
2. 사실을 바꾸지 마세요(어느 시장, 언제, 누구에게). 과장(최고·무조건·평생)과 이모지는 쓰지 마세요.
3. 길이: banner는 한 줄 짧게(값을 채워 30자 이내), sms_short는 값을 채워 45자 이내(단문 문자), 나머지는 초안 길이 안팎.
4. banner는 서로 다른 느낌 세 가지(날씨·장날·동네 살리기), sns는 친근한 해요체, 어르신이 읽어도 쉬운 말.
5. tour_en은 자연스러운 영어로, 나머지는 한국어로.
6. 결과는 JSON 하나로만: {"banner1": "...", "banner2": "...", "banner3": "...", "sns": "...", "sms_short": "...",
   "sms_long": "...", "tour_ko": "...", "tour_en": "..."}"""


@dataclass
class PackOutput:
    report: Report
    html: Path
    pdf: Path | None
    docx: Path | None
    scenario: str
    copy: dict = field(default_factory=dict)        # 채운 홍보 문구
    drafts: dict = field(default_factory=dict)      # 자리표시자 초안(에이전트·AI용)
    polished: bool = False
    problems: list[str] = field(default_factory=list)
    titles: list[str] = field(default_factory=list)


PLAN_PREFIX = re.compile(r"^[A-E]′?안 ")


def _d(dt: date) -> str:
    return f"{dt.year}.{dt.month}.{dt.day}"


def _placeholders(text: str) -> set[str]:
    return {m.group(1) for m in PLACEHOLDER.finditer(text)}


def polish_copy(gateway, facts: FactStore, drafts: dict, context: dict, emphasis: str = "") -> tuple[dict, list[str]]:
    """홍보 문구를 AI로 다듬는다(자리표시자만 보냄). 통과한 문구만 돌려준다."""
    from dataolle.ai.gateway import extract_json
    from dataolle.ai.narrator import verify
    if gateway is None or not gateway.enabled:
        return {}, []
    used = set().union(*[_placeholders(t) for t in drafts.values()])
    sub = FactStore({fid: facts.facts[fid] for fid in used if fid in facts.facts})
    leaked = verify(list(drafts.values()), sub)
    if leaked and gateway.external:
        gateway.audit.write({"event": "copy_draft_blocked", "problems": leaked})
        return {}, ["초안에 수치가 직접 들어 있어 외부 AI로 보내지 않음"]
    if getattr(gateway, "strict", False):
        # 레벨 0: 데이터로 고른 시장 이름·장날·배분안도 토큰으로(돌아온 문구는 gateway가 이 PC에서 되돌림)
        rd = gateway.redactor
        rd.add_terms([context.get("시장"), context.get("장날") if context.get("장날") not in (None, "없음") else None])
        context = {**context, "배분안": rd.token(context.get("배분안", ""), "고른 배분안")}
    rows, sent = gateway.facts_for_prompt(sub)
    if getattr(gateway, "strict", False):
        # 사실 설명 맨 앞의 'C안 ' 같은 안 글자도 데이터로 고른 결과 → '고른 안'으로
        rows = [{k: (PLAN_PREFIX.sub("고른 안 ", v) if isinstance(v, str) else v) for k, v in r.items()} if isinstance(r, dict) else r
                for r in rows]
    user = json.dumps({"상황": context, "강조할 점": emphasis or "(없음)", "사실 목록": rows, "초안": drafts}, ensure_ascii=False, indent=1)
    resp = gateway.call("실행물 문구 다듬기", COPY_SYSTEM, user, max_tokens=1600, values_sent=sent)
    if resp is None:
        return {}, [gateway.last_error or "AI 호출 실패"]
    data = extract_json(resp.text)
    if not isinstance(data, dict):
        return {}, ["AI 응답이 JSON 형식이 아님"]
    ok, probs = {}, []
    for k, draft in drafts.items():
        v = str(data.get(k, "")).strip()
        if not v:
            continue
        p = verify([v], facts)
        if p:
            probs += p
            continue
        if not _placeholders(v) <= _placeholders(draft):
            probs.append(f"{k}: 초안에 없는 사실 번호를 씀")
            continue
        if len(facts.render(v)) > LIMITS.get(k, 400):
            probs.append(f"{k}: 길이 초과")
            continue
        ok[k] = v
    if probs:
        gateway.audit.write({"event": "copy_partly_rejected", "problems": probs[:6]})
    return ok, probs


class ActionPack:
    def __init__(self, course_out, out_dir: str | Path, facts: FactStore | None = None):
        rep = course_out.report
        if "policy" not in rep.extra:
            raise ValueError("이 보고서에는 처방 설계 결과가 없습니다(탐나는전 코스로 먼저 분석하세요).")
        self.co, self.rep, self.pol = course_out, rep, rep.extra["policy"]
        self.facts = facts if facts is not None else course_out.facts
        self.out = Path(out_dir)

    # ------------------------------------------------------------------------------------------ 도우미
    def F(self, label, v, unit) -> str:
        return self.facts.add(label, None if v is None else float(v), unit, AggKind.APPLIED, "실행물 묶음(처방 설계)").id

    @staticmethod
    def P(fid: str | None, attr: str = "") -> str:
        return "{{" + fid + (f".{attr}" if attr else "") + "}}" if fid else "확인 불가"

    def r(self, text: str) -> str:
        return self.facts.render(text)

    def market_rates(self, k: str, mkd: dict) -> dict:
        if k in mkd.get("rates", {}):
            return mkd["rates"][k]
        S = self.pol["scen"][k]
        rr = {"평일": S.rate_of(rev=mkd["rev"], mkt=True, blind=mkd["blind"]),
              "비 오는 날": S.rate_of(rev=mkd["rev"], mkt=True, rain=True, blind=mkd["blind"])}
        if mkd["days"]:
            rr["장날"] = S.rate_of(rev=mkd["rev"], mkt=True, mday=True, blind=mkd["blind"])
            rr["비 오는 장날"] = S.rate_of(rev=mkd["rev"], mkt=True, mday=True, rain=True, blind=mkd["blind"])
        ids = {kk: self.F(f"{k}안 {mkd['name']} {kk} 적립률", v, "%") for kk, v in rr.items()}
        mkd.setdefault("rates", {})[k] = ids
        return ids

    # ------------------------------------------------------------------------------------------ 만들기
    def build(self, scenario: str | None = None, gateway=None, emphasis: str = "", make_pdf: bool = True,
              make_docx: bool = True) -> PackOutput:
        pol, P, r = self.pol, self.P, self.r
        k = scenario if scenario in pol["scen"] else pol["rec"]
        S, sf, sA = pol["scen"][k], pol["sf"][k], pol["sf"]["A"]
        A = pol["scen"]["A"]
        year, start, months = pol["year"], pol["start"], pol["months"]
        focus, where = pol["focus"], pol["where"]
        area = focus or where
        rule = pol["rule_ph"][k]
        start_day = date(year, start, 1)
        mk = pol["markets"][0] if pol["markets"] else None
        mr = self.market_rates(k, mk) if mk else {}
        mname = mk["name"] if mk else "우리 동네 전통시장"
        days = mk["days"] if mk else None
        base_id = sf["base"]
        dg = pol.get("diag") or {}
        rain_on = pol["has_rain"] and bool(S.params.get("rain"))
        mday_on = bool(days) and bool(S.params.get("mday"))
        problems: list[str] = []

        # 홍보 문구 초안(자리표시자) ------------------------------------------------------------------------
        wk = mr.get("평일")
        drafts = {
            "banner1": (f"비 오는 날엔 {mname}! 탐나는전 {P(mr.get('비 오는 날'))} 적립" if rain_on and wk else
                        f"{mname}에서 탐나는전 쓰면 {P(wk)} 적립" if wk else f"탐나는전 적립 {P(base_id)}, 끊기지 않고 1년 내내"),
            "banner2": (f"장날({days})엔 탐나는전 적립 {P(mr.get('장날'))}" if mday_on else
                        f"{year}년 탐나는전 적립, 끝까지 이어집니다"),
            "banner3": "우리 동네 가게에서 쓰면 적립이 더 큽니다 — 탐나는전",
            "sns": (f"[{area} 탐나는전 소식] {year}년 {start}월부터 {mname}에서 탐나는전으로 결제하면 {P(wk)} 적립돼요. "
                    + ((" ".join(x for x in [f"장날({days})엔 {P(mr.get('장날'))}" if mday_on else "",
                                            f"비 오는 날엔 {P(mr.get('비 오는 날'))}" if rain_on else ""] if x)
                        .replace(f"{P(mr.get('장날'))} 비", f"{P(mr.get('장날'))}, 비") + "까지 적립됩니다. ") if (mday_on or rain_on) else "")
                    + "장 보고, 적립 받고, 우리 동네도 살려요. #탐나는전 #" + mname.replace(" ", "") + " #" + area.replace(" ", "")
                    + " #제주전통시장") if wk else
                   (f"[{area} 탐나는전 소식] {year}년 {start}월부터 탐나는전 적립이 {P(base_id)}(으)로 바뀝니다. "
                    f"예산이 끊기지 않도록 {months}개월 내내 이어지는 방식이에요. #탐나는전 #우리동네살리기"),
            "sms_short": (f"[제주도] {mname} 탐나는전 {P(wk)} 적립" + (", 비 오는 날 더!" if rain_on else "!")) if wk else
                         f"[제주도] 탐나는전 적립 {P(base_id)}, {months}개월 내내!",
            "sms_long": (f"[제주특별자치도] {year}년 {start}월부터 탐나는전 적립 방식이 바뀝니다. {mname}에서 탐나는전으로 결제하면 "
                         f"평일 {P(wk)}" + (f", 장날({days}) {P(mr.get('장날'))}" if mday_on else "")
                         + (f", 비 오는 날 {P(mr.get('비 오는 날'))}" if rain_on else "")
                         + f" 적립됩니다. 그 밖의 가맹점은 기본 {P(base_id)}. 적립 예산이 한 해 끝까지 이어지도록 바꾼 방식입니다.")
                        if wk else (f"[제주특별자치도] {year}년 {start}월부터 탐나는전 적립률이 {P(base_id)}입니다. "
                                    f"적립 예산이 {months}개월 내내 끊기지 않도록 바꾼 방식입니다."),
            "tour_ko": (f"제주 전통시장에서 탐나는전으로 결제하고 적립 받으세요. {mname}: 평일 {P(wk)}"
                        + (f", 비 오는 날 {P(mr.get('비 오는 날'))}" if rain_on else "") + ". 탐나는전은 만 14세 이상 누구나 앱으로 쓸 수 있습니다.")
                       if wk else "제주 가게에서 탐나는전으로 결제하고 적립 받으세요. 만 14세 이상 누구나 앱으로 쓸 수 있습니다.",
            "tour_en": (f"Shop at this traditional market ({mname}) with Tamnanjeon, Jeju's local currency, and get {P(wk)} back"
                        + (f" ({P(mr.get('비 오는 날'))} on rainy days)" if rain_on else "") + ". Join easily with the Tamnanjeon app.")
                       if wk else "Pay with Tamnanjeon, Jeju's local currency, at local shops and get cashback. Join easily with the Tamnanjeon app.",
        }
        drafts = {kk: re.sub(r"\s+", " ", v).strip() for kk, v in drafts.items()}
        ctx = {"동네": area, "시장": mname, "시행": f"{year}년 {start}월", "배분안": f"{k}안 {NAMES.get(k, '')}",
               "장날": days or "없음", "비 오는 날 가산": "있음" if rain_on else "없음"}
        polished, probs = polish_copy(gateway, self.facts, drafts, ctx, emphasis)
        problems += probs
        copy_ph = {kk: polished.get(kk, v) for kk, v in drafts.items()}
        copy = {kk: r(v) for kk, v in copy_ph.items()}

        # 1. 한 장 요약 ----------------------------------------------------------------------------------------
        sections = []
        prob = []
        if dg:
            prob.append(f"{josa(dg['unit'], '은')} '{dg['type']}' 동네입니다" + (f"({dg['lead']})." if dg.get("lead") else "."))
        if A.exhaust:
            prob.append(f"지금 적립 방식(기본 {P(sA['base'])}, 명절 달 {P(sA['peak'])})이면 적립 예산 {P(pol['f_budget'])}이 "
                        f"{P(sA['exm'])}께 바닥나 그 뒤 사용액의 {P(sA['unpaid'])}는 적립 없이 지나갑니다.")
        prob.append(f"지금은 적립 혜택이 쓰는 만큼 나뉘어 전통시장 몫이 {P(sA['mkt'])}, 원도심 몫이 {P(sA['old'])}입니다.")
        eff = [f"적립을 {P(sf['last'])} 동안 유지" + ("(중단 없음)" if not S.exhaust else f"({S.exhaust_text()} 소진)"),
               f"전통시장 몫 {P(sA['mkt'])} → {P(sf['mkt'])}, 원도심 몫 {P(sA['old'])} → {P(sf['old'])}"]
        if focus and "focus_amt" in sf:
            eff.append(f"{focus}에 돌아가는 적립액 {P(sA['focus_amt'])} → {P(sf['focus_amt'])}")
        if mk and wk:
            eff.append(f"{mname} 적립률: " + ", ".join(f"{kk} {P(v)}" for kk, v in mr.items()))
        one = Table(pd.DataFrame({"내용": [r(" ".join(prob)), r(f"{k}안 {S.name}: {rule}."), r(" / ".join(eff)),
                                         f"D-60 배분안 확정 → D-45 상인회 설명 → D-30 홍보 → {_d(start_day)} 시행 → 4주 뒤 집행 점검 → 12주 뒤 효과 확인",
                                         f"“{pol['question']}”"]},
                                index=pd.Index(["문제", "처방", "기대 효과", "할 일", "효과 확인"], name="항목")),
                    "", None, style="text", widths=[14, 86])
        sections.append(Section("한 장 요약", [one, Paragraph(
            "이 묶음의 수치는 데이터올레가 제주빅데이터센터 데이터로 계산하고 반출 기준 검수를 통과한 분석 보고서에서 옮겼습니다. "
            "안내문·홍보 문구는 고른 배분안 기준이며, 확정 전 담당 부서가 적립률·연락처를 확인해 고쳐 쓰세요.")]))

        # 2. 개선안 초안 ----------------------------------------------------------------------------------------
        L = [Subhead(f"■ 탐나는전 적립 예산 배분 개선안({year}년)"),
             Subhead("Ⅰ. 추진 배경"),
             Bullets(["탐나는전 할인·적립이 예산 문제로 멈췄다 다시 시작하는 '멈췄다 달리기'가 반복됨(2022년 4월, 2023년 5월·9월, "
                      "2026년 7월 13일). 2026년에는 예산 425억 원 중 2월(적립률 20%)에만 154억 원이 쓰임(제이누리, 2026.7.13).",
                      "원도심 상권 위축·고령화·학령인구 감소로 '동네 살리기'에 적립 예산을 더 쓰자는 요구가 큼. 제주도는 업종·지역·상권별 "
                      "결제 패턴 빅데이터 분석을 계획(뉴스서울, 2026.3.15)."]),
             Subhead("Ⅱ. 현황과 문제(제주빅데이터센터 분석)")]
        cur = []
        if A.exhaust:
            cur.append(r(f"현행 방식(기본 {P(sA['base'])}, 명절 달 {P(sA['peak'])})으로 예산 {P(pol['f_budget'])}을 쓰면 {P(sA['exm'])}께 소진, "
                         f"그 뒤 사용액의 {P(sA['unpaid'])}는 적립 없음"))
        cur.append(r(f"적립 혜택이 사용액에 비례해 전통시장 몫 {P(sA['mkt'])}, 원도심 몫 {P(sA['old'])}에 그침"))
        if pol.get("f_rain"):
            cur.append(r(f"비 오는 날 전통시장 탐나는전 사용은 맑은 날보다 {P(pol['f_rain'], 'abs')} {P(pol['f_rain'], 'dir')}(날씨에 약한 시장)"))
        if pol.get("f_md"):
            cur.append(r(f"오일장 장날 사용액은 그 밖의 날의 {P(pol['f_md'])}(장날에 손님이 몰림)"))
        if dg:
            cur.append(r(f"{dg['unit']} 동네 진단: '{dg['type']}'" + (f" — {dg['lead']}" if dg.get("lead") else "")))
        types = pol.get("types") or {}
        L += [Bullets(cur), Subhead(f"Ⅲ. 개선안: {k}안 {S.name}")]
        imp = [r(f"적립 방식: {rule}")]
        if S.params.get("dong") and types:
            tgt = [u for t in ("고령·상권 위축", "고령 동네", "아이 감소") for u in types.get(t, [])]
            imp.append("동네 가산 대상: 원도심 동네와 동네 진단 유형이 고령·상권 위축·고령·아이 감소인 동네"
                       + (f"({', '.join(tgt[:10])}{' 등' if len(tgt) > 10 else ''})" if tgt else ""))
        if S.params.get("blind") and types.get("사각지대"):
            imp.append(f"사각지대 가산 대상: 도민 소비 대비 탐나는전 침투지수 80 미만 동네({', '.join(types['사각지대'][:8])} 등)")
        if mday_on or (S.params.get("mday") and pol["markets"]):
            cyc = [m for m in pol["markets"] if m["days"]]
            if cyc:
                imp.append("장날 가산: 오일장 장날(자료에서 5일 주기를 찾음) — " + ", ".join(f"{m['name']} {m['days']}" for m in cyc))
        if S.params.get("rain"):
            imp.append("비 오는 날 판정(검토): 기상청 제주 지점 일강수량 5mm 이상 예보·관측일. 운영사 적립 규칙에 날짜 목록으로 넣는 방식이 간단함")
        imp.append(r(f"적립률 상한 {P(pol['f_cap'])}, 기존 예산 {P(pol['f_budget'])} 안에서 재배분(추가 예산 없음)"))
        L.append(Bullets(imp))
        L += [Subhead("Ⅳ. 기대 효과"), Bullets([r(x) for x in eff])]
        # 월별 적립 계획
        lab = [f"{(start - 1 + o) % 12 + 1}월" for o in range(months)]
        paid = list(S.monthly_paid)
        cum = pd.Series(paid).cumsum()
        mp = pd.DataFrame({"계획 적립액": paid, "누적": cum.values, "예산 대비 누적(%)": (cum / pol["budget"] * 100).values}, index=lab)
        mp.index.name = "월"
        mpd = pd.DataFrame({"계획 적립액": [fmt_value(v, "원") for v in mp["계획 적립액"]],
                            "누적": [fmt_value(v, "원") for v in mp["누적"]],
                            "예산 대비 누적": [fmt_value(v, "%") for v in mp["예산 대비 누적(%)"]]}, index=mp.index)
        from dataolle.analysis.derive import make_derived
        from dataolle.analysis.semantic import Dim
        mp_r = make_derived(pol["results"][:1], mp, mpd, f"{k}안 월별 적립 계획(예산 집행 점검용)", "적립액", "원", AggKind.APPLIED,
                            Dim("pm", "", "계획 월", "time"), Dim("metric", "", "지표", "category"),
                            method="응용집계: 기준 1년의 월별 사용금액 × 적립 방식별 적립률(예산 소진 뒤는 0)")
        mp_r.period = pol["period"]
        L += [Subhead("Ⅴ. 소요 예산과 월별 적립 계획"),
              Bullets([r(f"총 {P(pol['f_budget'])}(" + ("질문에서 받은 예산" if pol["user_budget"] else
                                                    "2026년 예산 비율 6.25%를 자료 사용액에 적용한 값 — 실제 예산으로 다시 계산 권장") + ")"),
                       "아래 표의 월별 계획보다 실제 적립액이 10% 넘게 많은 달이 나오면 기본 적립률을 조정(집행 점검)"]),
              Table(mpd, f"월별 적립 계획 — {k}안({year}년)", mp_r, style="num"),
              Subhead("Ⅵ. 추진 일정"),
              Paragraph("'추진 일정·체크리스트' 쪽 참고(D-60 배분안 확정 ~ D+84 효과 확인)."),
              Subhead("Ⅶ. 효과 확인과 환류"),
              Bullets([f"시행 12주 뒤 데이터올레에 효과 확인 질문을 넣어 전통시장과 일반 가맹점의 주간 이중차분·위약 검정으로 판정: “{pol['question']}”",
                       "판정에 따라 다음 배분을 자동으로 다시 계산(C′안): " + " / ".join(DECISION_TEXT.values())]),
              Subhead("Ⅷ. 검토 필요 사항"),
              Bullets(["적립률이 바뀌어도 사용 패턴이 같다고 가정한 계산임(민감도 — 적립률 1%p당 사용액 2% 변화 가정 시 "
                       + (f"현행 방식 소진 {pol['sens']['A_exhaust']}" if pol['sens'].get("A_exhaust") else "현행 방식 소진 없음")
                       + f", {pol['rec']}안 기본 적립률 {fmt_value(pol['sens']['rec_base'], '%')}, 사용액 {pol['sens']['rec_use']:+.1f}% — "
                       "보고서 '가정과 한계' 참고). 실제 반응은 시행 뒤 효과 확인으로 측정",
                       "1인 월 한도·가맹점 연매출 기준·국비 매칭 조건·운영사 시스템 반영 가능 여부 확인",
                       "신도심 등 일반 가맹점 기본 적립률 하향에 대한 설명 자료 필요(예산이 끝까지 이어진다는 점을 함께 안내)"])]
        sections.append(Section("개선안 초안 — 도 담당 부서 내부 검토용", L, new_page=True))

        # 3. 대안 비교표 ---------------------------------------------------------------------------------------
        sfB = pol["sf"]["B"]
        pros = {"A": ("익숙한 방식, 명절 소비 진작", r(f"{P(sA['exm'])}께 예산 소진(적립 중단)") if A.exhaust else "예산이 남으면 문제 없음",
                      "예산이 넉넉할 때"),
                "B": ("단순하고 끝까지 유지", r(f"적립률이 {P(sfB['base'])}(으)로 낮아지고 원도심·시장 몫은 그대로"), "형평·단순함이 우선일 때"),
                "C": ("끝까지 유지 + 원도심·전통시장·장날·비 오는 날에 몫 집중", r(f"그 밖의 가맹점 기본 적립률이 {P(pol['sf']['C']['base'])}(으)로 낮아짐, 조건 설정 필요"),
                      "원도심·전통시장 살리기가 우선일 때"),
                "D": ("탐나는전이 덜 쓰이는 동네의 보급 확대", "원도심·시장 몫은 크게 늘지 않음", "사각지대 보강이 우선일 때"),
                "E": ("C안 효과 + 선결제·고액 결제로 혜택이 쏠리는 것을 막음", "학원·교육 업종·고액 결제 이용자의 혜택이 줄어 설명 필요",
                      "혜택 쏠림(역진성)을 줄이는 게 우선일 때"),
                "C′": ("효과 확인 결과를 반영한 다음 배분", "효과 판정이 정확하려면 시행 전후 자료가 충분해야 함", "시행 뒤 효과를 확인했을 때")}
        pc = pd.DataFrame([{"안": f"{kk}안 {NAMES[kk]}", "장점": pros[kk][0], "단점": pros[kk][1], "이런 목표라면": pros[kk][2]}
                           for kk in pol["order"]]).set_index("안")
        blocks = [Table(pol["table"], "적립 방식별 예산 유지와 적립 혜택의 행선지(같은 예산, 오른쪽 칸은 지급된 적립액 중 몫)",
                        pol["results"][0], style="text", widths=[12, 25, 12] + [max(6, int(51 / max(1, pol["table"].shape[1] - 2)))]
                        * (pol["table"].shape[1] - 2))]
        if pol["figures"].get("burn"):
            blocks.append(Figure(pol["figures"]["burn"], "적립 방식별 누적 적립액과 예산(선이 예산선을 넘는 곳이 소진)", pol["results"][1]))
        blocks.append(Table(pc, "안별 장단점", None, style="text", widths=[18, 30, 32, 20]))
        sections.append(Section("대안 비교표", blocks, new_page=True))

        # 4. 상인회 안내문 -------------------------------------------------------------------------------------
        to = f"{mname} 상인회 회원 여러분께" if mk else "전통시장 상인회 회원 여러분께"
        lines = [to, "손님이 우리 시장에서 탐나는전으로 결제하면 적립이 더 많아집니다." if (mk and S.params.get("mkt", 0) + S.params.get("dong", 0) + S.params.get("mday", 0) + S.params.get("rain", 0))
                 else "탐나는전 적립이 한 해 끝까지 끊기지 않도록 적립 방식이 바뀝니다.", "■ 무엇이 바뀌나요"]
        if wk:
            lines.append(r(f"· 평일: 결제 금액의 {P(wk)} 적립(그 밖의 가맹점 기본 {P(base_id)})"))
            if mday_on:
                lines.append(r(f"· 장날({days}): {P(mr['장날'])} 적립"))
            if rain_on:
                lines.append(r(f"· 비 오는 날(일강수량 5mm 이상 예보·관측일): {P(mr['비 오는 날'])} 적립"))
        else:
            lines.append(r(f"· 모든 가맹점 {P(base_id)} 적립"))
        lines.append(r(f"· 적립 예산이 {P(sf['last'])} 동안 이어집니다" + ("(중간에 멈추지 않음)" if not S.exhaust else "")))
        lines += ["■ 왜 바뀌나요", "· 적립 예산이 한 해 중간에 떨어지지 않게 하면서, 시장·원도심에 혜택을 더 드리기 위해서입니다."]
        why = []
        if pol.get("f_md") and mday_on:
            why.append(f"장날 사용액이 평소의 {P(pol['f_md'])}")
        if pol.get("f_rain") and rain_on:
            why.append(f"비 오는 날 시장 사용은 {P(pol['f_rain'], 'abs')} {P(pol['f_rain'], 'dir')}")
        if why:
            lines.append(r("· 제주빅데이터센터 자료로 보면 " + ", ".join(why) + "했습니다. 손님이 적은 날에 혜택을 더 드립니다."))
        lines += ["■ 상인회에서 해 주실 일", "· ① 가게의 탐나는전 가맹·QR 결제 가능 여부를 확인해 주세요(단말기 점검).",
                  "· ② 함께 드리는 안내문·포스터를 가게 앞에 붙여 주세요.",
                  "· ③ 장날·비 오는 날 덤·시식 행사를 함께 하면 효과가 더 커집니다.",
                  "· ④ 손님이 잘 모르시면 '탐나는전으로 결제하면 적립이 더 많아요'라고 알려 주세요.",
                  "■ 문의", "· 담당 부서 ________ ☎ ________ / 상인회 사무실 ☎ ________"]
        sections.append(Section("상인회 안내문", [Sheet(f"[안내] {year}년 {start}월부터 탐나는전 적립이 이렇게 바뀝니다", lines,
                                                    "notice", sign=f"{_d(start_day - timedelta(days=45))}  제주특별자치도 · {mname if mk else '전통시장'} 상인회",
                                                    note="빈칸(담당 부서·연락처)을 채워 인쇄하세요. 적립률은 확정 전 검토안입니다.")],
                                new_page=True))

        # 5. 손님 홍보 문구 ------------------------------------------------------------------------------------
        def ln(kk):
            return f"{len(copy[kk])}자"
        blocks = [Paragraph("고른 배분안 기준의 문구입니다." + (" AI가 초안을 다듬었습니다(숫자는 이 PC에서 채움)." if polished else
                                                       " 규칙 기반 초안입니다(AI를 켜면 문구를 다듬습니다).")),
                  Sheet("현수막 1안 — 날씨", [copy["banner1"]], "banner", note=ln("banner1")),
                  Sheet("현수막 2안 — 장날", [copy["banner2"]], "banner", note=ln("banner2")),
                  Sheet("현수막 3안 — 동네 살리기", [copy["banner3"]], "banner", note=ln("banner3")),
                  Sheet("SNS 게시글", [copy["sns"]], "sns", note=ln("sns")),
                  Sheet("문자 — 단문(45자 이내)", [copy["sms_short"]], "sms", note=ln("sms_short") + " · 단문 기준 45자(90바이트)"),
                  Sheet("문자 — 장문", [copy["sms_long"]], "sms", note=ln("sms_long")),
                  Sheet("관광객 안내(국문·영문)", [copy["tour_ko"], copy["tour_en"]], "notice",
                        note="탐나는전은 만 14세 이상이면 관광객도 앱으로 쓸 수 있습니다.")]
        sections.append(Section("손님 홍보 문구", blocks, new_page=True))

        # 6. 어르신 안내문 -------------------------------------------------------------------------------------
        if wk:
            v = self.facts.get(wk).value or 0
            f10 = self.F(f"{mname} 1만 원 결제 때 돌려받는 돈(평일)", round(100 * v, -1), "원")
            big = [r(f"{mname}에서 탐나는전으로 1만 원어치 장을 보시면 {P(f10)}을 돌려드립니다.")]
            if mday_on:
                f10m = self.F(f"{mname} 1만 원 결제 때 돌려받는 돈(장날)", round(100 * (self.facts.get(mr['장날']).value or 0), -1), "원")
                big.append(r(f"장날({days})에는 {P(f10m)}을 돌려드립니다."))
            if rain_on:
                f10r = self.F(f"{mname} 1만 원 결제 때 돌려받는 돈(비 오는 날)", round(100 * (self.facts.get(mr['비 오는 날']).value or 0), -1), "원")
                big.append(r(f"비 오는 날에는 {P(f10r)}을 돌려드립니다."))
        else:
            fb10 = self.F("1만 원 결제 때 돌려받는 돈", round(100 * (self.facts.get(base_id).value or 0), -1), "원")
            big = [r(f"탐나는전으로 1만 원을 쓰시면 {P(fb10)}을 돌려드립니다.")]
        big += ["■ 이렇게 쓰세요", "· 탐나는전 카드나 휴대전화 앱으로 내시면 됩니다.",
                "· 카드 만들기·충전이 어려우시면 주민센터나 가까운 금융기관 창구에 물어보세요.",
                "■ 문의", "· ________ 주민센터  ☎ ________"]
        sections.append(Section("어르신 안내문(주민센터 비치용)", [Sheet("시장에서 탐나는전으로 장 보시면 돈을 더 돌려드립니다", big, "big",
                                                                    sign=f"{year}년 {start}월  ________ 주민센터")], new_page=True))

        # 7. 추진 일정·체크리스트 --------------------------------------------------------------------------------
        sched = [(-60, "배분안 확정(재정·국비 매칭 협의), 운영사에 적립 규칙(동네·시장·장날·비 오는 날) 설정 요청", "도 담당 부서"),
                 (-45, "상인회·주민센터 설명회, 상인회 안내문 배포", "도·행정시, 상인회"),
                 (-30, "현수막·SNS·문자 홍보물 제작, 어르신 안내문 주민센터 비치", "도·행정시·주민센터"),
                 (-14, "가맹점 QR·단말기 점검, 장날 일정 공지", "상인회·운영사"),
                 (-7, "운영사 적립 규칙 시험(장날·비 오는 날 조건이 제대로 붙는지)", "운영사·도 담당 부서"),
                 (0, "시행, 첫 장날 현장 안내", "모두"),
                 (28, "집행 점검: 월 적립액이 월별 계획보다 10% 넘게 많으면 기본 적립률 조정", "도 담당 부서"),
                 (84, "효과 확인: 데이터올레에 효과 확인 질문 실행 → 다음 배분(C′안) 조정", "데이터 담당(센터 이용)")]
        sc = pd.DataFrame([{"시점": ("D-day" if dd == 0 else (f"D{dd:+d}")), "날짜": _d(start_day + timedelta(days=dd)), "할 일": what,
                            "담당": who, "확인": "□"} for dd, what, who in sched]).set_index("시점")
        sections.append(Section("추진 일정·체크리스트", [
            Table(sc, f"추진 일정(시행일 {_d(start_day)} 기준)", None, style="text", widths=[9, 11, 52, 20, 8]),
            Bullets(["매월: 월별 적립 계획(개선안 Ⅴ) 대비 집행률을 상인회·주민센터와 공유",
                     "날짜는 시행일 기준 자동 계산입니다. 실제 일정에 맞춰 고쳐 쓰세요."])], new_page=True))

        # 8. 효과 확인 질문 ------------------------------------------------------------------------------------
        from dataolle.ai.planner import rule_plan
        plan = rule_plan(pol["question"])
        ev = plan.market.get("event") if plan.course == "tamna" else None
        sections.append(Section("효과 확인 질문(다음 방문용)", [
            Callout("다음 방문 때 데이터올레 첫 화면에 그대로 붙여 넣으세요", [f"“{pol['question']}”"], "insight"),
            Bullets(["센터에 미리 요청할 자료: " + " / ".join(pol["needs"]),
                     "시점: 시행 4주 뒤 집행 점검, 12주 뒤 효과 확인(시행 후 자료가 4주 이상 있어야 판정)",
                     "판정과 다음 배분: " + " / ".join(DECISION_TEXT.values()) + ". 결과 보고서의 'C′안'이 다음 배분입니다."]
                    + ([f"이 질문은 '{ev[:4]}년 {int(ev[4:])}월 전후 효과'로 읽힙니다(확인 완료)."] if ev else []))]))

        # 근거와 출처 -------------------------------------------------------------------------------------------
        results = list(pol["results"]) + [mp_r]
        tmp = Report("", results=results)
        sections.append(Section("근거와 출처", [Bullets(tmp.citations() + [
            f"근거 보고서: {self.rep.title}" + (f"({self.co.pdf.name})" if getattr(self.co, "pdf", None) else ""),
            "수치는 반출 기준 검수(10 미만 비공개·역산 방지)를 통과한 분석 결과에서 옮겼습니다. 최종 반출 여부는 센터 심사를 따릅니다.",
            "외부 사실(2026년 예산·중단 시점 등)은 언론 보도(제이누리 2026.7.13, 뉴스서울 2026.3.15)를 인용했습니다."])]))

        title = f"{area} 탐나는전 캐시백 설계 실행물 묶음"
        ai_note = ("홍보 문구는 AI가 규칙 기반 초안을 다듬었습니다(자리표시자만 전달, 숫자는 이 PC에서 채움)." if polished
                   else "모든 문장은 규칙 기반 템플릿으로 썼고 수치는 계산 결과에서 채웠습니다(AI 미사용).")
        report = Report(title, f"{k}안 {S.name} · {year}년 {start}월 시행 기준 · 도청·행정시·주민센터·상인회용", sections, results,
                        {"대상": area, "배분안": f"{k}안 {S.name} — {rule_text(S.params, year)}",
                         "적립 예산": r(P(pol["f_budget"])) + (" (질문에서 받은 값)" if pol["user_budget"] else " (2026년 예산 비율 적용)"),
                         "근거 보고서": self.rep.title, "작성일": datetime.now().strftime("%Y-%m-%d"), "작성 도구": "데이터올레"},
                        synthetic=self.rep.synthetic, ai_note=ai_note)
        report.extra["kicker"] = "데이터올레 · 실행물 묶음(바로 쓰는 문서)"
        report.extra["no_docx_appendix"] = True
        self.out.mkdir(parents=True, exist_ok=True)
        stem = f"실행물묶음_{area}_{k}안".replace(" ", "").replace("′", "2")
        html = to_html(report, self.out / f"{stem}.html", appendix=False)
        pdf = to_pdf(html, self.out / f"{stem}.pdf") if make_pdf else None
        docx = to_docx(report, self.out / f"{stem}.docx") if make_docx else None
        return PackOutput(report, html, pdf, docx, k, copy, copy_ph, bool(polished), problems, [s.heading for s in sections])


def make_action_pack(course_out, out_dir: str | Path, scenario: str | None = None, gateway=None, emphasis: str = "",
                     facts: FactStore | None = None, make_pdf: bool = True, make_docx: bool = True) -> PackOutput:
    return ActionPack(course_out, out_dir, facts).build(scenario, gateway, emphasis, make_pdf, make_docx)


__all__ = ["make_action_pack", "ActionPack", "PackOutput", "polish_copy", "CAP"]
