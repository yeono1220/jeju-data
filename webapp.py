"""데이터올레 — 웹 시연판(Streamlit Community Cloud용).

설치형 본체와 다른 점은 하나뿐입니다.
- 본체: 센터 CSV(또는 가상데이터 693MB)를 이 PC에 올려 DuckDB로 계산 → 사실(fact) 생성 → AI가 문장 작성
- 시연판: 그 계산을 미리 돌려 **사실만** 저장해 두고(result_*.json), AI 문장 작성만 실제로 호출

숫자는 저장된 사실에서 나오므로 기획서·보고서와 한 값도 다르지 않습니다.
AI는 여전히 숫자를 보지 못하고 {{f12}} 자리표시자만 쓰며, 값은 이 서버가 채웁니다(본체와 같은 규칙).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))   # 본체 코드를 그대로 쓴다

from dataolle.ai.agent import SYSTEM
from dataolle.ai.clients import make_client
from dataolle.ai.gateway import AuditLog, Gateway, sanitize_question
from dataolle.ai.narrator import verify
from dataolle.analysis.facts import Fact, FactStore
from dataolle.sdc.engine import AggKind

HERE = Path(__file__).parent
WD = HERE / "webdata"
DS = json.loads((WD / "datasets.json").read_text(encoding="utf-8"))
QUESTIONS = ["budget", "dongne"]      # 코스가 다른 것만. 같은 결과가 나오는 질문은 넣지 않는다.
MAX_PER_USER, MAX_PER_DAY = 3, 300
COUNTER = Path("/tmp/dataolle_calls.json")

st.set_page_config(page_title="데이터올레 — 웹 시연판", page_icon="🍊", layout="wide")

CSS = """<style>
[data-testid="stMainBlockContainer"]{padding-top:2.6rem;max-width:1140px}
[data-testid="stMain"] [data-testid="stMarkdownContainer"]>p,
[data-testid="stMain"] [data-testid="stMarkdownContainer"]>ul{max-width:780px;line-height:1.62}
[data-testid="stDataFrame"]{font-variant-numeric:tabular-nums}
.do-ribbon{background:#fff4df;color:#8a4b00;border:1px solid #f0d8ae;border-radius:6px;padding:5px 12px;
 font-size:.86rem;margin-bottom:14px}
.do-web{background:#1b1b1a;color:#fff;border-radius:6px;padding:8px 13px;font-size:.9rem;margin-bottom:10px}
.do-web a{color:#9ec3f0}
.do-concl{display:grid;grid-template-columns:1.5fr 1fr 1fr;gap:8px;margin:.4rem 0 1.2rem;max-width:1000px}
.do-cell{border:1px solid #e3e1da;border-radius:6px;padding:14px 16px;background:#fff}
.do-cell.lead{background:#f6f5f2;border-color:#d9d7cf}
.do-k{color:#5f5e5a;font-size:.84rem;margin-bottom:4px}
.do-v{font-size:1.42rem;font-weight:700;line-height:1.25;font-variant-numeric:tabular-nums}
.do-v2{font-size:1.18rem;font-weight:700;font-variant-numeric:tabular-nums}
.do-d{color:#5f5e5a;font-size:.86rem;margin-top:6px;line-height:1.5}
.do-tag{color:#5f5e5a;font-size:.86rem;border:1px solid #e3e1da;border-radius:6px;padding:2px 8px}
.do-tag.ai{color:#1c5cab;border-color:#c6d8ef}
.do-tl{border-left:2px solid #e3e1da;margin:.2rem 0 .2rem 6px;padding:2px 0 2px 16px}
.do-tl .row{position:relative;padding:5px 0}
.do-tl .row:before{content:"";position:absolute;left:-22px;top:12px;width:8px;height:8px;border-radius:50%;background:#c9c7c0}
.do-tl .row.plan:before,.do-tl .row.answer:before{background:#1c5cab}
.do-tl .k{color:#5f5e5a;font-size:.78rem}
.do-tl .t{font-weight:600}
.do-tl .d{color:#5f5e5a;font-size:.9rem}
.do-saw{display:grid;grid-template-columns:1fr 1fr;gap:8px;max-width:860px;margin:.6rem 0}
.do-saw div{border:1px solid #e3e1da;border-radius:6px;padding:10px 14px;font-size:.88rem;color:#5f5e5a}
.do-saw b{color:#1b1b1a;display:block;margin-bottom:3px;font-size:.9rem}
.do-hr{border:0;border-top:1px solid #e3e1da;margin:1.8rem 0 1.2rem;max-width:1000px}
.do-why{max-width:900px;font-size:.88rem;color:#5f5e5a;margin:6px 0 0;line-height:1.62}
.do-why b{color:#1b1b1a}
.do-why .h{color:#1b1b1a;font-weight:700;font-size:.96rem;margin:0 0 8px}
.do-why table{border-collapse:collapse;width:100%;margin:0 0 12px;font-size:.88rem}
.do-why th,.do-why td{border:1px solid #e3e1da;padding:7px 12px;text-align:left;vertical-align:top}
.do-why th{background:#f6f5f2;color:#5f5e5a;font-weight:600}
.do-why td:first-child{color:#5f5e5a;width:116px}
.do-why td:nth-child(3){color:#1b1b1a}
.do-why p{margin:0 0 8px;max-width:900px}
.do-why code{background:#f6f5f2;border:1px solid #e3e1da;border-radius:4px;padding:1px 5px;font-size:.92em}
/* 제목 블록 */
.do-head{margin:2px 0 4px}
.do-head h1{font-size:1.75rem;font-weight:700;margin:0 0 4px;letter-spacing:-.01em}
.do-head p{color:#5f5e5a;font-size:.98rem;margin:0;max-width:820px}
/* 질문 고르기 — 고른 쪽만 테두리로 표시(주 버튼은 아래 하나뿐) */
.st-key-pick_budget button,.st-key-pick_dongne button{height:auto;min-height:52px;white-space:normal;
 line-height:1.45;padding:11px 16px;font-weight:600}
.st-key-pick_budget button:has(div p:first-child),.st-key-pick_dongne button:has(div p:first-child){}
/* 고른 질문 전문 */
.do-qtext{border-left:3px solid #1c5cab;background:#f6f5f2;border-radius:0 6px 6px 0;padding:11px 16px;
 margin:12px 0 16px;max-width:900px;color:#1b1b1a;font-size:1rem;line-height:1.55}
/* 하는 일 세 칸 */
.do-g3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;max-width:1000px;margin:6px 0 2px}
</style>"""

STAGE_NOTE = {"진단": "지금 어떤 상태인가", "처방": "그래서 어떻게 나눌까",
              "실행물": "바로 쓸 서류", "효과 확인": "시행 뒤 다시 재기"}


@st.cache_data
def load(slug: str) -> dict:
    return json.loads((WD / f"result_{slug}.json").read_text(encoding="utf-8"))


def secret(name: str) -> str:
    """Secrets 파일이 아예 없으면 예외가 나므로 감싼다."""
    try:
        return str(st.secrets.get(name, "") or "")
    except Exception:
        return ""


def store(d: dict) -> FactStore:
    fs = FactStore()
    for x in d["facts"]:
        fs.facts[x["id"]] = Fact(x["id"], x["label"], x["value"], x["unit"], AggKind(x["kind"]),
                                 "", x.get("masked", False), x.get("is_change", False))
    return fs


def quota() -> tuple[int, bool]:
    try:
        c = json.loads(COUNTER.read_text())
    except Exception:
        c = {}
    n = c.get(time.strftime("%Y-%m-%d"), 0)
    return n, n < MAX_PER_DAY


def bump() -> None:
    today = time.strftime("%Y-%m-%d")
    try:
        c = json.loads(COUNTER.read_text())
    except Exception:
        c = {}
    c[today] = c.get(today, 0) + 1
    try:
        COUNTER.write_text(json.dumps(c))
    except Exception:
        pass


def ai_answer(d: dict, fs: FactStore):
    """본체와 같은 계약으로 AI에게 답 문장만 맡긴다 — 숫자는 보여 주지 않고, 값은 여기서 채운다."""
    key = secret("ANTHROPIC_API_KEY")
    if not key:
        return None, "키가 없어 규칙 기반으로 답했습니다.", {}
    _, ok = quota()
    if not ok:
        return None, f"오늘 AI 호출 상한({MAX_PER_DAY}회)에 닿아 규칙 기반으로 답했습니다.", {}
    if st.session_state.get("calls", 0) >= MAX_PER_USER:
        return None, f"한 사람당 {MAX_PER_USER}번까지만 AI로 돌립니다. 더 보시려면 내려받아 설치해 주세요.", {}

    gw = Gateway(make_client("anthropic", api_key=key), 0, AuditLog())   # 레벨 0 — 센터 수치 0건
    rows, sent = gw.facts_for_prompt(fs)
    clean, masked = sanitize_question(d["question"])
    rec = d.get("rec")
    user = (f"[질문]\n{clean}\n\n"
            f"[이미 실행한 도구]\ncheck_data → 데이터 {len(DS['loaded'])}종 확인. "
            f"run_course({d.get('course', 'tamna')}) → 진단·효과 추정·처방 설계 완료. "
            + (f"make_action_pack({rec}안) → 실행물 묶음 생성 완료.\n\n" if rec else "\n\n")
            + f"[사실 목록]\n{json.dumps(rows, ensure_ascii=False)}\n\n"
            + (f"[추천안]\n{rec}안 {d.get('scen_names', {}).get(rec, '')}\n\n" if rec else "")
            + "이제 finish 단계입니다. [진단] [처방] [실행물] [효과 확인] 네 단계로 한국어 답을 쓰세요. "
              "숫자는 반드시 사실 번호 자리표시자로만 쓰고, 사실 목록에 없는 숫자는 쓰지 마세요.")
    t0 = time.time()
    resp = gw.call("웹 시연", SYSTEM, user, max_tokens=1100, values_sent=sent, masked_numbers=masked)
    if resp is None:
        return None, f"AI 호출이 실패해 규칙 기반으로 답했습니다. ({gw.last_error or '원인 미상'})", {}
    draft = (resp.text or "").strip()
    meta = {"걸린 시간": f"{time.time() - t0:.1f}초", "모델": gw.client.model,
            "보낸 사실": f"{len(rows)}개", "값까지 보낸 사실": f"{sent}개",
            "보낸 글자": f"{len(SYSTEM) + len(user):,}자", "질문에서 가린 숫자": f"{masked}개",
            "받은 글자": f"{len(draft):,}자"}
    problems = verify([draft], fs)
    if problems:
        return None, ("AI 답에 근거 없는 숫자가 있어 버리고 규칙 기반 답을 씁니다 — 본체와 같은 검증입니다. "
                      + "; ".join(problems[:2])), meta
    bump()
    st.session_state.calls = st.session_state.get("calls", 0) + 1
    return fs.render(draft), "", meta


def answer_md(text: str) -> str:
    import re
    parts = re.split(r"\s*\[(진단|처방|실행물|효과 확인)\]\s*", text or "")
    if len(parts) == 1:
        return text
    out = [parts[0].strip()] if parts[0].strip() else []
    for tag, body in zip(parts[1::2], parts[2::2]):
        out.append(f"**{tag}** · {STAGE_NOTE.get(tag, '')}\n\n{body.strip()}")
    return "\n\n".join(out)


def cards(d: dict) -> str:
    if not d.get("scen_rows"):
        return ""
    i = {c: n for n, c in enumerate(d["scen_cols"])}
    rec = next((r for r in d["scen_rows"] if r[0].startswith(d["rec"] + "안")), d["scen_rows"][0])
    base = next((r for r in d["scen_rows"] if r[0].startswith("A안")), d["scen_rows"][0])
    out = [f"<div class='do-cell lead'><div class='do-k'>추천하는 배분안</div>"
           f"<div class='do-v'>{d['rec']}안 {d['scen_names'].get(d['rec'], '')}</div>"
           f"<div class='do-d'>{d['why']}</div></div>"]
    for col, lab in (("예산 유지", "예산이 버티는 기간"), ("전통시장", "전통시장에 가는 몫")):
        if col in i:
            out.append(f"<div class='do-cell'><div class='do-k'>{lab}</div>"
                       f"<div class='do-v2'>{rec[i[col]]}</div>"
                       f"<div class='do-d'>지금 방식(A안) {base[i[col]]}</div></div>")
    return "<div class='do-concl'>" + "".join(out) + "</div>"


def timeline(steps) -> str:
    L = {"plan": "계획", "tool": "도구 실행", "result": "결과", "answer": "답", "fallback": "되돌림", "error": "문제"}
    r = "".join(f"<div class='row {s['kind']}'><div class='k'>{L.get(s['kind'], '진행')}</div>"
                f"<div class='t'>{s['title']}</div>"
                + (f"<div class='d'>{s['detail']}</div>" if s.get("detail") else "") + "</div>" for s in steps)
    return f"<div class='do-tl'>{r}</div>"


def scen_table(d: dict) -> None:
    import pandas as pd
    df = pd.DataFrame(d["scen_rows"], columns=d["scen_cols"])
    rules = None
    if "적립률" in df.columns:
        rules = list(zip(df.iloc[:, 0], df["적립률"]))
        df = df.drop(columns=["적립률"])
    st.dataframe(df.style.apply(lambda r: ["background-color:#eef2f8;font-weight:700"
                                           if str(r.iloc[0]).startswith(d["rec"] + "안") else "" for _ in r], axis=1),
                 hide_index=True, width="stretch", height=36 * (len(df) + 1) + 3)
    if rules:
        st.markdown("**적립 방식별 규칙**\n" + "\n".join(f"- **{k}** — {v}" for k, v in rules))


# ---------------------------------------------------------------------------- 화면
st.markdown(CSS, unsafe_allow_html=True)
slug = st.session_state.get("slug", QUESTIONS[0])
D = load(slug)

with st.sidebar:
    st.markdown("### 데이터올레")
    st.caption("탐나는전 캐시백 설계 에이전트 — 웹 시연판")
    st.markdown("---")
    st.markdown(f"**데이터** {len(DS['loaded'])}종 불러옴  \n<span class='do-tag' style='color:#8a4b00;"
                f"border-color:#f0d8ae;background:#fff4df'>가상데이터</span>", unsafe_allow_html=True)
    has_key = bool(secret("ANTHROPIC_API_KEY"))
    st.markdown(f"**AI** {'Claude Haiku 5.5' if has_key else '사용 안 함(규칙 기반)'}")
    if has_key:
        used, _ = quota()
        st.caption(f"오늘 AI 호출 {used}/{MAX_PER_DAY}회 · 한 사람당 {MAX_PER_USER}회")
    st.caption("전송 레벨 0 — 구조만 전송(센터 수치 0건)")
    st.markdown("---")
    st.link_button("내려받아 직접 돌려 보기",
                   "https://github.com/yeono1220/jeju-data/archive/refs/heads/main.zip", width="stretch")
    st.caption("본체는 센터 PC에서 센터 CSV로 직접 계산합니다.")

st.markdown("<div class='do-ribbon'><b>가상데이터</b> — 실제와 같은 형식으로 만든 가짜 자료입니다. "
            "화면의 모든 숫자는 시연용이며 실제 의사결정에 쓰지 마세요.</div>", unsafe_allow_html=True)

t1, t2, t3, t4 = st.tabs(["① 질문", "② 데이터", "③ 분석 결과", "④ 전송 기록"])

with t1:
    if st.session_state.get("ans"):
        badge = "AI 에이전트" if st.session_state.get("by_ai") else "규칙 기반 에이전트"
        st.markdown(f"### 답 <span class='do-tag ai'>{badge}</span>", unsafe_allow_html=True)
        st.caption(f"질문: {D['question']}")
        st.markdown(cards(D), unsafe_allow_html=True)
        if st.session_state.get("note"):
            st.info(st.session_state["note"])
        st.markdown(answer_md(st.session_state["ans"]))
        seen = ("질문(숫자는 가림) · 사실 번호와 설명 · 데이터 목록" if st.session_state.get("by_ai")
                else "없음 — AI를 쓰지 않고 규칙 기반으로 답했습니다.")
        st.markdown(f"<div class='do-saw'><div><b>AI가 본 것</b>{seen}</div>"
                    "<div><b>AI가 보지 못한 것</b>센터 원본 자료 · 실제 수치 · 개인 정보</div></div>",
                    unsafe_allow_html=True)
        st.caption("표·그림의 숫자는 모두 저장된 계산 결과에서 나옵니다. AI가 쓴 문장은 자리표시자뿐이고 값은 서버가 채웠습니다.")
        with st.expander("에이전트가 한 일 — 계획 › 도구 실행 › 결과"):
            st.markdown(timeline(D["steps"]), unsafe_allow_html=True)
        if st.button("다른 질문 해 보기"):
            for k in ("ans", "by_ai", "note", "meta"):
                st.session_state.pop(k, None)
            st.rerun()
    else:
        st.markdown("<div class='do-head'><h1>데이터올레</h1>"
                    "<p>탐나는전 캐시백을 감이 아니라 데이터로 설계합니다. "
                    "질문 하나로 진단 → 배분안 비교 → 보고서·실행물까지 만듭니다.</p></div>",
                    unsafe_allow_html=True)
        st.markdown("#### 무엇을 알고 싶으세요?")
        cols = st.columns(len(QUESTIONS))
        for col, q in zip(cols, QUESTIONS):
            qd = load(q)
            mark = "선택됨 · " if q == slug else ""
            if col.button(f"{mark}{qd['label']}", key=f"pick_{q}", width="stretch"):
                if q != slug:
                    st.session_state.slug = q
                    st.rerun()
        st.markdown(f"<div class='do-qtext'>{D['question']}</div>", unsafe_allow_html=True)
        if st.button("에이전트에게 맡기기", type="primary"):
            box = st.status("에이전트가 일하는 중입니다 — 보통 20~40초 걸립니다.", expanded=True)
            rail = box.empty()
            seen = []
            for s in D["steps"]:
                seen.append(s)
                rail.markdown(timeline(seen), unsafe_allow_html=True)
                time.sleep(0.45)
            ans, note, meta = ai_answer(D, store(D))
            st.session_state["by_ai"] = ans is not None
            st.session_state["ans"] = ans or D["rule_answer"]
            st.session_state["note"] = note
            st.session_state["meta"] = meta
            box.update(label=f"에이전트가 일을 마쳤습니다 — {len(D['steps'])}단계", state="complete", expanded=False)
            st.rerun()
        st.caption("보통 20~40초 걸립니다.")
        st.markdown("<hr class='do-hr' style='margin:1.6rem 0 1.1rem'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='do-g3'>"
            "<div class='do-cell'><div class='do-k'>1. 진단</div><div class='do-v2'>어디에 얼마나 쓰이나</div>"
            "<div class='do-d'>동네·업종·요일·날씨별 사용 흐름과 적립이 닿지 않는 곳을 찾습니다.</div></div>"
            "<div class='do-cell'><div class='do-k'>2. 처방</div><div class='do-v2'>예산 안에서 어떻게 나눌까</div>"
            "<div class='do-d'>적립률 반응도에서 앞당긴 소비를 걸러 내고, 배분안 다섯 개를 비교합니다.</div></div>"
            "<div class='do-cell'><div class='do-k'>3. 실행물·반출</div><div class='do-v2'>바로 쓸 서류까지</div>"
            "<div class='do-d'>보고서 PDF, 개선안·안내문, 반출 신청 문구와 점검표를 만듭니다.</div></div>"
            "</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='do-why'>"
            "<div class='h'>이 화면과 설치판의 차이</div>"
            "<table>"
            "<tr><th></th><th>이 화면 (무료 서버)</th><th>설치판 (내 PC)</th></tr>"
            "<tr><td>질문</td><td>미리 돌려 둔 두 가지</td><td>자유롭게</td></tr>"
            "<tr><td>숫자 계산</td><td>미리 돌린 결과를 읽어 옵니다</td><td>그 자리에서 계산합니다</td></tr>"
            "<tr><td>AI 답 작성</td><td><b>실제로 호출합니다</b></td><td><b>실제로 호출합니다</b></td></tr>"
            "<tr><td>데이터</td><td>가상데이터</td><td>센터 CSV 또는 가상데이터</td></tr>"
            "</table>"
            "<p>무료 서버는 메모리를 1GB까지만 주는데 자료를 올리는 데만 1.6GB가 듭니다. "
            "그래서 이 화면은 계산을 다시 하지 않고 미리 돌려 둔 결과를 읽어 옵니다 — "
            "<b>화면의 숫자는 그 계산에서 실제로 나온 값이고, 답 문장은 지금 AI가 새로 쓴 것입니다.</b> "
            "사무용 PC는 보통 8~16GB라 설치하면 전부 그대로 돌아갑니다.</p>"
            "<p><b>설치</b> — 파일을 내려받아 압축을 풀고 <code>설치.bat</code> → <code>실행.bat</code>, "
            "두 번 누르면 끝입니다. 센터 CSV가 없어도 첫 화면의 '가상데이터로 바로 체험하기'로 "
            "질문부터 반출 서류까지 전 과정을 볼 수 있습니다.</p>"
            "</div>", unsafe_allow_html=True)

with t2:
    import pandas as pd
    st.markdown("### 데이터")
    st.markdown(f"지금은 **가상데이터 {len(DS['loaded'])}종**을 불러왔습니다. 본체에서는 센터에서 받은 CSV "
                "폴더를 지정하면 됩니다 — 파일 이름이 아니라 **헤더(컬럼 이름)를 명세서와 대조해서** "
                "무슨 데이터인지, 몇 월치인지 스스로 알아냅니다.")
    st.caption(f"알고 있는 센터 데이터셋 {DS['known']}종 · {DS['source']}. "
               "모르는 파일은 추측해서 읽지 않고 '확인 필요'로 따로 표시합니다.")
    st.dataframe(pd.DataFrame([{"데이터": f"{x['id']} {x['name']}", "기간": x["period"],
                                "개월": x["months"], "파일": x["files"]} for x in DS["loaded"]]),
                 hide_index=True, width="stretch", height=36 * (len(DS["loaded"]) + 1) + 3)
    st.markdown("#### 센터에서 무엇을 받아 어디에 두나")
    st.markdown("1. 아래 문안을 센터 운영자에게 그대로 전달합니다(본체가 질문에 맞춰 자동으로 만들어 줍니다).\n"
                "2. 받은 CSV를 **한 폴더에** 모아 둡니다. 파일 이름은 센터가 준 그대로 두세요.\n"
                "3. 본체 **② 데이터** 탭에서 그 폴더를 지정하면 끝입니다. 하위 폴더까지 훑습니다.")
    st.code(DS["request_text"], language=None, wrap_lines=True)
    st.caption("센터 CSV 형식: 헤더는 명세서 영문 컬럼명, 쉼표 구분, 모든 값 큰따옴표, 결측 NA, UTF-8(BOM 없음).")

with t3:
    st.markdown(f"### {D['title']}")
    st.caption("가상데이터(시연용)로 만든 결과입니다. 실제 의사결정에 쓰지 마세요.")
    st.markdown(cards(D), unsafe_allow_html=True)
    ok, tot = D["checks"]
    st.caption(f"그림 {D['n_fig']}개 · 표·그림 반출 검수 {ok}/{tot} 통과 · "
               f"비공개(x) 처리한 칸 {D['masked_cells']}개 · 만드는 데 {D['seconds']:.0f}초")
    if D.get("paragraph"):
        st.markdown("#### 기관 보고용 문단(현황 보고·보도자료)")
        st.code(D["paragraph"], language=None, wrap_lines=True)
    if D.get("scen_rows"):
        st.markdown("#### 처방 설계 — 탐나는전 적립 예산 배분")
        scen_table(D)
    st.markdown("<hr class='do-hr'>", unsafe_allow_html=True)
    st.markdown("#### 보고서에 들어간 그림")
    figs = sorted((WD / f"figs_{slug}").glob("*.png"))
    for row in [figs[i:i + 2] for i in range(0, len(figs), 2)]:
        for col, f in zip(st.columns(len(row)), row):
            col.image(str(f), width="stretch")

with t4:
    st.markdown("### 전송 기록")
    st.caption("AI에게 실제로 무엇을 보냈는지 그대로 보여 줍니다. 본체의 관리자 탭과 같은 내용입니다.")
    meta = st.session_state.get("meta") or {}
    if not meta:
        st.info("① 질문 탭에서 '에이전트에게 맡기기'를 누르면 여기에 기록이 남습니다.")
    else:
        st.table({"항목": list(meta.keys()), "값": [str(v) for v in meta.values()]})
    st.markdown("#### 규칙")
    st.markdown("- 이 시연판은 **레벨 0**으로 돌립니다 — AI에게 사실의 **설명과 번호만** 보내고 "
                "**값은 한 건도 보내지 않습니다**. 센터 데이터에서 나온 이름·문장도 토큰으로 가려서 보냅니다.\n"
                "- 질문 속 금액·수치는 보내기 전에 `[수치]`로 가립니다.\n"
                "- AI가 쓴 답은 `{{f12}}` 자리표시자뿐이고, 값은 이 서버가 채웁니다.\n"
                "- 사실 목록에 없는 숫자가 답에 있으면 **그 답을 버리고** 규칙 기반 답으로 바꿉니다.\n"
                "- 센터 원본 자료와 개인 정보는 어떤 경우에도 나가지 않습니다.")
