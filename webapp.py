"""데이터올레 — 웹 시연판(Streamlit Community Cloud용).

설치형 본체와 다른 점은 하나뿐입니다.
- 본체: 센터 CSV(또는 가상데이터 693MB)를 이 PC에 올려 DuckDB로 계산 → 사실(fact) 생성 → AI가 문장 작성
- 시연판: 그 계산을 미리 한 번 돌려 **사실만** 저장해 두고(result.json), AI 문장 작성만 실제로 호출

숫자는 저장된 사실에서 나오므로 기획서·보고서와 한 값도 다르지 않습니다.
AI는 여전히 숫자를 보지 못하고 {{f12}} 자리표시자만 쓰며, 값은 이 서버가 채웁니다(본체와 같은 규칙).
무거운 적재·계산을 하지 않으므로 무료 티어 메모리 안에서 돕니다.
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
DATA = json.loads((HERE / "webdata" / "result.json").read_text(encoding="utf-8"))
FIGS = HERE / "webdata" / "figs"
MAX_PER_USER = 3                      # 한 사람이 쓸 수 있는 AI 호출 수
MAX_PER_DAY = 300                     # 하루 전체 상한(키 보호)
COUNTER = Path("/tmp/dataolle_calls.json")

st.set_page_config(page_title="데이터올레 — 웹 시연판", page_icon="🍊", layout="wide")

CSS = """<style>
[data-testid="stMainBlockContainer"]{padding-top:2.6rem;max-width:1140px}
[data-testid="stMain"] [data-testid="stMarkdownContainer"]>p,
[data-testid="stMain"] [data-testid="stMarkdownContainer"]>ul{max-width:760px;line-height:1.62}
[data-testid="stDataFrame"]{font-variant-numeric:tabular-nums}
.do-ribbon{background:#fff4df;color:#8a4b00;border:1px solid #f0d8ae;border-radius:6px;padding:6px 12px;
 font-size:.92rem;margin-bottom:10px}
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
</style>"""

EX = [("동네 진단 — 일도1동", "일도1동, 탐나는전으로 동네를 살릴 수 있을까요?"),
      ("예산 400억 원 배분", DATA["question"]),
      ("전통시장 추가 적립 효과", "전통시장에 적립률을 더 얹으면 매출이 얼마나 늘까요?")]


def secret(name: str) -> str:
    """Secrets 파일이 아예 없으면 예외가 나므로 감싼다(로컬 실행·키 없는 배포 모두 동작)."""
    try:
        return str(st.secrets.get(name, "") or "")
    except Exception:
        return ""


def store() -> FactStore:
    fs = FactStore()
    for d in DATA["facts"]:
        fs.facts[d["id"]] = Fact(d["id"], d["label"], d["value"], d["unit"], AggKind(d["kind"]),
                                 d.get("source", ""), d.get("masked", False), d.get("is_change", False))
    return fs


def quota() -> tuple[int, bool]:
    """하루 전체 호출 수를 세어 키를 지킨다."""
    today = time.strftime("%Y-%m-%d")
    try:
        c = json.loads(COUNTER.read_text())
    except Exception:
        c = {}
    n = c.get(today, 0)
    return n, n < MAX_PER_DAY


def bump():
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


def ai_answer(q: str, fs: FactStore):
    """본체와 같은 계약으로 AI에게 답 문장만 맡긴다 — 숫자는 보여 주지 않고, 값은 여기서 채운다."""
    key = secret("ANTHROPIC_API_KEY")
    if not key:
        return None, "키가 없어 규칙 기반으로 답했습니다.", {}
    used, ok = quota()
    if not ok:
        return None, f"오늘 AI 호출 상한({MAX_PER_DAY}회)에 닿아 규칙 기반으로 답했습니다. 내일 다시 눌러 보세요.", {}
    if st.session_state.get("calls", 0) >= MAX_PER_USER:
        return None, f"한 사람당 {MAX_PER_USER}번까지만 AI로 돌립니다. 더 보시려면 내려받아 설치해 주세요.", {}

    gw = Gateway(make_client("anthropic", api_key=key), 0, AuditLog())   # 레벨 0 — 센터 수치는 0건만 내보낸다
    rows, sent = gw.facts_for_prompt(fs)
    clean, masked = sanitize_question(q)
    user = (f"[질문]\n{clean}\n\n"
            f"[이미 실행한 도구]\ncheck_data → 데이터 11종 확인. "
            f"run_course(tamna, 제주 전체, 적립 예산 400억 원, goal=revive) → 진단·효과 추정·처방 설계 완료. "
            f"make_action_pack({DATA['rec']}안) → 실행물 묶음 생성 완료.\n\n"
            f"[사실 목록]\n{json.dumps(rows, ensure_ascii=False)}\n\n"
            f"[추천안]\n{DATA['rec']}안 {DATA['scen_names'].get(DATA['rec'], '')}\n\n"
            "이제 finish 단계입니다. [진단] [처방] [실행물] [효과 확인] 네 단계로 한국어 답을 쓰세요. "
            "숫자는 반드시 사실 번호 자리표시자로만 쓰고, 사실 목록에 없는 숫자는 쓰지 마세요.")
    t0 = time.time()
    resp = gw.call("웹 시연", SYSTEM, user, max_tokens=1100, values_sent=sent, masked_numbers=masked)
    if resp is None:
        return None, f"AI 호출이 실패해 규칙 기반으로 답했습니다. ({gw.last_error or '원인 미상'})", {}
    draft = (resp.text or "").strip()
    problems = verify([draft], fs)
    meta = {"초": round(time.time() - t0, 1), "모델": gw.client.model, "보낸 사실": f"{len(rows)}개",
            "값까지 보낸 사실": f"{sent}개", "보낸 글자": len(SYSTEM) + len(user),
            "가린 숫자": masked, "받은 글자": len(draft)}
    if problems:
        return None, "AI 답에 근거 없는 숫자가 있어 버리고 규칙 기반 답을 씁니다 — 본체와 같은 검증입니다. " \
                     + "; ".join(problems[:2]), meta
    bump()
    st.session_state.calls = st.session_state.get("calls", 0) + 1
    return fs.render(draft), "", meta


STAGE_NOTE = {"진단": "지금 어떤 상태인가", "처방": "그래서 어떻게 나눌까",
              "실행물": "바로 쓸 서류", "효과 확인": "시행 뒤 다시 재기"}


def answer_md(text: str) -> str:
    """'[진단] … [처방] …'을 본체와 같은 단계별 문단으로."""
    import re
    parts = re.split(r"\s*\[(진단|처방|실행물|효과 확인)\]\s*", text or "")
    if len(parts) == 1:
        return text
    out = [parts[0].strip()] if parts[0].strip() else []
    for tag, body in zip(parts[1::2], parts[2::2]):
        out.append(f"**{tag}** · {STAGE_NOTE.get(tag, '')}\n\n{body.strip()}")
    return "\n\n".join(out)


def cards() -> str:
    cols, rows = DATA["scen_cols"], DATA["scen_rows"]
    i = {c: n for n, c in enumerate(cols)}
    rec = next((r for r in rows if r[0].startswith(DATA["rec"] + "안")), rows[0])
    base = next((r for r in rows if r[0].startswith("A안")), rows[0])
    out = [f"<div class='do-cell lead'><div class='do-k'>추천하는 배분안</div>"
           f"<div class='do-v'>{DATA['rec']}안 {DATA['scen_names'].get(DATA['rec'], '')}</div>"
           f"<div class='do-d'>{DATA['why']}</div></div>"]
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


# ---------------------------------------------------------------------------- 화면
st.markdown(CSS, unsafe_allow_html=True)
with st.sidebar:
    st.markdown("### 데이터올레")
    st.caption("탐나는전 캐시백 설계 에이전트 — 웹 시연판")
    st.markdown("---")
    st.markdown(f"**데이터** 11종 불러옴  \n<span class='do-tag' style='color:#8a4b00;border-color:#f0d8ae;"
                f"background:#fff4df'>가상데이터</span>", unsafe_allow_html=True)
    has_key = bool(secret("ANTHROPIC_API_KEY"))
    st.markdown(f"**AI** {'Claude Haiku 5.5' if has_key else '사용 안 함(규칙 기반)'}")
    if has_key:
        used, _ = quota()
        st.caption(f"오늘 AI 호출 {used}/{MAX_PER_DAY}회 · 한 사람당 {MAX_PER_USER}회")
    st.caption("전송 레벨 0 — 구조만 전송(센터 수치 0건)")
    st.markdown("---")
    st.link_button("내려받아 직접 돌려 보기", "https://github.com/yeono1220/jeju-data/archive/refs/heads/main.zip",
                   width="stretch")
    st.caption("설치형 본체는 센터 PC에서 센터 CSV로 계산합니다.")

st.markdown("<div class='do-web'><b>웹 시연판입니다.</b> 무거운 데이터 적재·계산은 미리 끝내 두고, "
            "<b>AI 답 작성만 실제로 호출</b>합니다. 숫자는 저장된 계산 결과에서 나오므로 보고서·기획서와 같습니다. "
            "전체 기능은 <a href='https://github.com/yeono1220/jeju-data'>내려받아 설치</a>하면 동작합니다.</div>",
            unsafe_allow_html=True)
st.markdown("<div class='do-ribbon'><b>가상데이터</b> — 실제와 같은 형식으로 만든 가짜 자료입니다. "
            "화면과 보고서의 모든 숫자는 시연용이며, 실제 의사결정에 쓰지 마세요.</div>", unsafe_allow_html=True)

t1, t2, t3 = st.tabs(["① 질문", "② 분석 결과", "③ 전송 기록"])

with t1:
    if st.session_state.get("ans"):
        badge = "AI 에이전트" if st.session_state.get("by_ai") else "규칙 기반 에이전트"
        st.markdown(f"### 답 <span class='do-tag ai'>{badge}</span>", unsafe_allow_html=True)
        st.markdown(cards(), unsafe_allow_html=True)
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
            st.markdown(timeline(DATA["steps"]), unsafe_allow_html=True)
        if st.button("다시 묻기"):
            for k in ("ans", "by_ai", "note", "meta"):
                st.session_state.pop(k, None)
            st.rerun()
    else:
        st.markdown("### 무엇을 알고 싶으세요?")
        st.caption("이 시연판은 아래 질문 하나로 미리 계산해 둔 결과를 씁니다. AI는 그 결과를 읽고 답 문장을 새로 씁니다.")
        c = st.columns(3)
        for n, (lab, full) in enumerate(EX):
            c[n].button(lab, width="stretch", disabled=n != 1, help=full,
                        key=f"ex{n}")
        st.text_area("질문", value=DATA["question"], height=80, label_visibility="collapsed", disabled=True)
        st.caption("다른 질문은 설치형 본체에서 쓸 수 있습니다(시연판은 미리 계산한 한 가지만).")
        if st.button("에이전트에게 맡기기", type="primary"):
            box = st.status("에이전트가 일하는 중입니다 — 보통 20~40초 걸립니다.", expanded=True)
            rail = box.empty()
            seen = []
            for s in DATA["steps"]:
                seen.append(s)
                rail.markdown(timeline(seen), unsafe_allow_html=True)
                time.sleep(0.45)
            fs = store()
            ans, note, meta = ai_answer(DATA["question"], fs)
            st.session_state["by_ai"] = ans is not None
            st.session_state["ans"] = ans or DATA["rule_answer"]
            st.session_state["note"] = note
            st.session_state["meta"] = meta
            box.update(label=f"에이전트가 일을 마쳤습니다 — {len(DATA['steps'])}단계", state="complete", expanded=False)
            st.rerun()

with t2:
    st.markdown(f"### {DATA['title']}")
    st.caption("가상데이터(시연용)로 만든 결과입니다. 실제 의사결정에 쓰지 마세요.")
    st.markdown(cards(), unsafe_allow_html=True)
    ok, tot = DATA["checks"]
    st.caption(f"그림 {DATA['n_fig']}개 · 표·그림 반출 검수 {ok}/{tot} 통과 · "
               f"비공개(x) 처리한 칸 {DATA['masked_cells']}개 · 만드는 데 {DATA['seconds']:.0f}초")
    st.markdown("#### 기관 보고용 문단(현황 보고·보도자료)")
    st.code(DATA["paragraph"], language=None, wrap_lines=True)
    st.markdown("#### 처방 설계 — 탐나는전 적립 예산 배분")
    import pandas as pd
    df = pd.DataFrame(DATA["scen_rows"], columns=DATA["scen_cols"])
    rules = None
    if "적립률" in df.columns:
        rules = list(zip(df.iloc[:, 0], df["적립률"]))
        df = df.drop(columns=["적립률"])
    rec = DATA["rec"]
    st.dataframe(df.style.apply(lambda r: ["background-color:#eef2f8;font-weight:700"
                                           if str(r.iloc[0]).startswith(rec + "안") else "" for _ in r], axis=1),
                 hide_index=True, width="stretch", height=36 * (len(df) + 1) + 3)
    if rules:
        st.markdown("**적립 방식별 규칙**\n" + "\n".join(f"- **{k}** — {v}" for k, v in rules))
    st.markdown("<hr class='do-hr'>", unsafe_allow_html=True)
    st.markdown("#### 보고서에 들어간 그림")
    figs = sorted(FIGS.glob("*.png"))
    for row in [figs[i:i + 2] for i in range(0, len(figs), 2)]:
        cc = st.columns(len(row))
        for col, f in zip(cc, row):
            col.image(str(f), width="stretch")

with t3:
    st.markdown("### 전송 기록")
    st.caption("AI에게 실제로 무엇을 보냈는지 그대로 보여 줍니다. 본체의 관리자 탭과 같은 내용입니다.")
    meta = st.session_state.get("meta") or {}
    if not meta:
        st.info("① 질문 탭에서 '에이전트에게 맡기기'를 누르면 여기에 기록이 남습니다.")
    else:
        st.table({"항목": list(meta.keys()), "값": [str(v) for v in meta.values()]})
    st.markdown("#### 규칙")
    st.markdown("- 이 시연판은 **레벨 0**으로 돌립니다 — AI에게 사실의 **설명과 번호만** 보내고 **값은 한 건도 보내지 않습니다**. 센터 데이터에서 나온 이름·문장도 토큰으로 가려서 보냅니다.\n"
                "- 질문 속 금액·수치는 보내기 전에 `[수치]`로 가립니다.\n"
                "- AI가 쓴 답은 `{{f12}}` 자리표시자뿐이고, 값은 이 서버가 채웁니다.\n"
                "- 사실 목록에 없는 숫자가 답에 있으면 **그 답을 버리고** 규칙 기반 답으로 바꿉니다.\n"
                "- 센터 원본 자료와 개인 정보는 어떤 경우에도 나가지 않습니다.")
