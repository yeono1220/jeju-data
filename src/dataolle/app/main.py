"""데이터올레 화면(Streamlit, 이 PC 안 127.0.0.1에서만 열림).

실행: 실행.bat  (= python -m streamlit run src/dataolle/app/main.py --server.address 127.0.0.1)
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))           # src 폴더(설치 없이 실행해도 동작)

from dataolle.ai.agent import Agent  # noqa: E402
from dataolle.ai.clients import DEFAULT_MODELS, ENV_KEYS, PROVIDER_LABEL, is_external, make_client  # noqa: E402
from dataolle.ai.gateway import LEVEL_TEXT, AuditLog, Gateway  # noqa: E402
from dataolle.ai.planner import ai_plan, biz_names, bjd_names  # noqa: E402
from dataolle.analysis import Catalog, FactStore, run_safe  # noqa: E402
from dataolle.analysis.semantic import MODELS, UNIT_GROUP  # noqa: E402
from dataolle.config import (Settings, Workspace, app_home, cleanup_sessions, list_sessions, load_env, mask_key,  # noqa: E402
                             save_env_key)
from dataolle.courses.action_pack import make_action_pack  # noqa: E402
from dataolle.courses.before_after import BeforeAfterCourse, BeforeAfterInput  # noqa: E402
from dataolle.courses.market import BIZ_LINK, MarketCourse, MarketInput  # noqa: E402
from dataolle.courses.policy import PolicyInput, parse_policy  # noqa: E402
from dataolle.courses.status import StatusCourse  # noqa: E402
from dataolle.courses.tamna import TamnaCourse, TamnaInput  # noqa: E402
from dataolle.ingest.inventory import check_plan, make_plan, month_range, operator_request_text  # noqa: E402
from dataolle.report.charts import render_chart  # noqa: E402
from dataolle.report.export_pack import USE_PRESETS, build_pack  # noqa: E402
from dataolle.sdc.engine import Level  # noqa: E402
from dataolle.spec import default_registry  # noqa: E402

st.set_page_config(page_title="데이터올레", page_icon="🍊", layout="wide")

MARKET_NEEDS = {"1-08": 12, "1-12": 3, "1-14": 3, "1-19": 12, "1-24": 3, "4-11": 1}
# 4-10(유동인구 v01)은 '최근 N개월'이 아니라 비교용 1개월(2017년의 같은 달)만 받으면 된다 → 0으로 표시
COURSE_NEEDS = {"market": MARKET_NEEDS, "status": {"1-08": 12, "1-12": 3, "1-14": 3, "4-11": 1}, "before_after": {"1-08": 12},
                "tamna": {"1-32": 12, "1-08": 9, "4-11": 1, "4-10": 0, "5-02": 12}}
COURSE_NAME = {"tamna": "탐나는전 캐시백 설계", "before_after": "사건·정책 전후 효과",
               "status": "동네 현황자료", "market": "창업 시장분석"}
EXAMPLES = ["일도1동, 탐나는전으로 동네를 살릴 수 있을까요?",
            "예산 400억 원으로 탐나는전 적립을 1년 내내 이어 가려면 어떻게 나눠야 해요?",
            "2025년 6월부터 한 전통시장 추가 적립 효과를 확인하고 다음 배분을 짜 줘",
            "2025년 2분기 15% 적립 때 늘어난 사용액 중 앞당긴 소비(학원 선결제 등)는 얼마나 돼요?",
            "적립 혜택이 고령·읍면 동네보다 신도심에 쏠리지 않게 하려면 어떻게 나눠야 해요?",
            "애월읍에 카페를 창업하려고 해요. 사업계획서용 시장분석을 해 주세요."]
GOAL_LABEL = {"auto": "자동(동네 진단에 맞춤)", "revive": "원도심·전통시장 살리기", "blind": "사각지대 보강", "even": "모든 가맹점 고르게",
              "current": "지금 방식 유지", "fair": "혜택 쏠림 막기(선결제·고액 결제)"}
STEP_LABEL = {"plan": "계획", "tool": "도구 실행", "result": "결과", "answer": "답", "fallback": "되돌림", "error": "문제"}
STAGE_NOTE = {"진단": "지금 어떤 상태인가", "처방": "그래서 어떻게 나눌까", "실행물": "바로 쓸 서류",
              "효과 확인": "시행 뒤 다시 재기"}


def md(text: str) -> str:
    """화면 마크다운에서 '0~14세 … 2017~2025'의 물결표 두 개가 취소선으로 바뀌지 않게 이스케이프."""
    return str(text).replace("~", "\\~")


CSS = """
<style>
/* 차분한 공공 업무 문서 — 모양은 .streamlit/config.toml [theme]이 맡고, 여기는 모자란 것만 채운다.
   선택자는 data-testid 와 key= 로 생기는 st-key-… 만 쓴다(자동 생성 클래스 이름은 안 씀). */

[data-testid="stMainBlockContainer"] {padding-top: 3.1rem; padding-bottom: 5rem; max-width: 1140px;}
[data-testid="stHeader"] {background: transparent;}
[data-testid="stTabs"] button[data-baseweb="tab"] p {font-size: 1rem; font-weight: 600;}

/* 읽는 글은 760px 안에서 끊는다. 표·그림·입력칸은 그대로 넓게. */
[data-testid="stMain"] [data-testid="stMarkdownContainer"] > p,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] > ul,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] > ol {max-width: 760px; line-height: 1.62;}
[data-testid="stMain"] [data-testid="stMarkdownContainer"] > h3 {margin: 0.2rem 0 0.1rem;}
[data-testid="stMain"] [data-testid="stMarkdownContainer"] > h4 {margin: 1.6rem 0 0.3rem;}
[data-testid="stCaptionContainer"] {max-width: 760px;}

/* 숫자는 폭이 같은 숫자로 */
[data-testid="stDataFrame"], [data-testid="stMetricValue"], .do-num {font-variant-numeric: tabular-nums;}

/* 붙여넣기용 긴 글: 고정폭·양끝맞춤을 풀어 본문 글꼴로 읽히게(복사 단추는 그대로) */
[data-testid="stMain"] [data-testid="stCode"] pre,
[data-testid="stMain"] [data-testid="stCode"] code {font-family: inherit; font-size: 0.95rem; line-height: 1.62;
  text-align: left; word-spacing: normal; white-space: pre-wrap;}
[data-testid="stMain"] [data-testid="stCode"] {max-width: 860px;}

/* 맨 위 '가상데이터' 띠 — 내려도 따라온다 */
.do-ribbon {position: sticky; top: 0; z-index: 6; background: #fff4df; color: #8a4b00;
  border: 1px solid #f0d8ae; border-radius: 6px; padding: 6px 12px; font-size: 0.92rem; margin-bottom: 10px;}
.do-ribbon b {font-weight: 700;}

/* 순서 줄 */
.do-step {display: flex; gap: 6px; flex-wrap: wrap; align-items: center; margin: 0.1rem 0 1.2rem;}
.do-step span {color: #5f5e5a; font-size: 0.9rem;}
.do-step span.on {color: #1c5cab; font-weight: 700;}
.do-step span.sep {color: #c9c7c0;}

/* 조용한 표시(알약 대신 글자) */
.do-tag {color: #5f5e5a; font-size: 0.86rem; border: 1px solid #e3e1da; border-radius: 6px; padding: 2px 8px;}
.do-tag.ai {color: #1c5cab; border-color: #c6d8ef;}
.do-tag.syn {color: #8a4b00; border-color: #f0d8ae; background: #fff4df;}

/* 결론 칸(벤토) — 가장 중요한 한 칸만 크게 */
.do-concl {display: grid; grid-template-columns: 1.5fr 1fr 1fr; gap: 8px; margin: 0.4rem 0 1.4rem; max-width: 1000px;}
.do-cell {border: 1px solid #e3e1da; border-radius: 6px; padding: 14px 16px; background: #ffffff;}
.do-cell.lead {background: #f6f5f2; border-color: #d9d7cf;}
.do-k {color: #5f5e5a; font-size: 0.84rem; margin-bottom: 4px;}
.do-v {color: #1b1b1a; font-size: 1.45rem; font-weight: 700; line-height: 1.25; font-variant-numeric: tabular-nums;}
.do-v2 {color: #1b1b1a; font-size: 1.2rem; font-weight: 700; font-variant-numeric: tabular-nums;}
.do-d {color: #5f5e5a; font-size: 0.86rem; margin-top: 6px; line-height: 1.5;}

/* 에이전트 진행 — 세로 타임라인 */
.do-tl {border-left: 2px solid #e3e1da; margin: 0.2rem 0 0.2rem 6px; padding: 2px 0 2px 16px;}
.do-tl .row {position: relative; padding: 5px 0;}
.do-tl .row:before {content: ""; position: absolute; left: -22px; top: 12px; width: 8px; height: 8px;
  border-radius: 50%; background: #c9c7c0;}
.do-tl .row.plan:before, .do-tl .row.answer:before {background: #1c5cab;}
.do-tl .row.error:before {background: #8a4b00;}
.do-tl .k {color: #5f5e5a; font-size: 0.78rem; letter-spacing: 0.02em;}
.do-tl .t {color: #1b1b1a; font-weight: 600;}
.do-tl .d {color: #5f5e5a; font-size: 0.9rem; line-height: 1.55;}

/* AI가 본 것 / 보지 못한 것 */
.do-saw {display: grid; grid-template-columns: 1fr 1fr; gap: 8px; max-width: 860px; margin: 0.6rem 0 0.2rem;}
.do-saw div {border: 1px solid #e3e1da; border-radius: 6px; padding: 10px 14px; font-size: 0.88rem; color: #5f5e5a;}
.do-saw b {color: #1b1b1a; display: block; margin-bottom: 3px; font-size: 0.9rem;}

.do-sub {color: #5f5e5a; font-size: 0.95rem; max-width: 760px; line-height: 1.6; margin: -0.2rem 0 1rem;}
.do-hr {border: 0; border-top: 1px solid #e3e1da; margin: 2rem 0 1.4rem; max-width: 1000px;}

/* 보조 단추도 또렷하게(흐려 보여 못 누르는 줄 알지 않게) */
[data-testid="stMain"] [data-testid="stBaseButton-secondary"] {color: #1b1b1a;}
[data-testid="stMain"] [data-testid="stBaseButton-secondary"]:hover {color: #1c5cab; border-color: #1c5cab;}

/* 사이드바 — 상태는 조용히, 삭제는 보조 단추로 */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {font-size: 0.92rem;}
.st-key-finish button {color: #8a4b00; border-color: #e3e1da;}
.st-key-finish button:hover {color: #8a4b00; border-color: #c89b63; background: #fff4df;}

/* 관리자 PIN 같은 짧은 입력칸은 짧게 */
.st-key-pin_new input, .st-key-pin_chk input, .st-key-pin_in input {max-width: 260px;}
</style>
"""


# ---------------------------------------------------------------------------
# 세션 준비
# ---------------------------------------------------------------------------
def _init():
    ss = st.session_state
    if "ws" in ss:
        return
    home = app_home()
    load_env(home)
    ss.settings = Settings.load(home)
    if ss.settings.auto_delete:
        cleanup_sessions(home, ss.settings.keep_hours)
    ss.ws = Workspace.create(home)
    ss.audit = AuditLog(ss.ws.path / "audit.jsonl")
    ss.catalog = None
    ss.plan = None
    ss.output = None
    ss.free = []
    ss.pack = None
    ss.admin_ok = False
    ss.deleted_msg = ""


def gateway() -> Gateway:
    s = st.session_state.settings
    client = make_client(s.provider, s.model or None)
    return Gateway(client, s.level, st.session_state.audit)


def sdc_level() -> Level:
    return Level(st.session_state.settings.sdc_level)


def load_catalog(folder: str):
    cat = Catalog.from_folder(folder)
    st.session_state.catalog = cat
    st.session_state.output = None
    st.session_state.free = []
    st.session_state.pack = None
    return cat


def demo_folder() -> Path:
    return app_home() / "demo_data"


# 가상데이터는 씨앗값이 고정돼 있어 만드는 파일 수와 순서가 늘 같다(아래 값은 진행률 표시에만 쓴다).
DEMO_FILES = 107
DEMO_STAGES = [(15, "내국인 유동인구·탐나는전 자료"), (31, "탐나는전·날씨 자료"),
               (86, "카드·외식업 포스 자료(월별)"), (DEMO_FILES, "요일·시간대·고객 자료")]


def load_demo():
    """시연용 가상데이터를 (없으면 만들고) 불러온다. 만드는 동안 단계 이름과 진행률을 보여 준다."""
    import threading
    from dataolle.synth.generator import generate
    d = demo_folder()
    if not (d / "_가상데이터_안내.txt").exists():
        d.mkdir(parents=True, exist_ok=True)
        box = st.container(border=True)
        box.markdown("**가상데이터를 만드는 중입니다.** 처음 한 번만 걸립니다. 창을 닫지 말고 기다려 주세요.")
        bar = box.progress(0.0, text="준비하는 중…")
        note = box.empty()
        fail = {}

        def work():
            try:
                generate(d, profile="demo", verbose=False)
            except Exception as exc:                                  # noqa: BLE001
                fail["e"] = exc

        th = threading.Thread(target=work, daemon=True)
        th.start()
        t0 = time.time()
        while th.is_alive():
            n = len(list(d.glob("Results_*.csv")))
            stage = next((lab for upto, lab in DEMO_STAGES if n < upto), DEMO_STAGES[-1][1])
            sec = time.time() - t0
            left = f" · 남은 시간 약 {int((sec / n) * (DEMO_FILES - n))}초" if n >= 5 else ""
            bar.progress(min(n / DEMO_FILES, 0.99), text=f"{stage} — 파일 {n}/{DEMO_FILES}개{left}")
            note.caption(f"{int(sec)}초 지남 · 이 폴더에 만듭니다: {d}")
            time.sleep(0.7)
        th.join()
        if "e" in fail:
            bar.empty()
            note.empty()
            st.error(f"가상데이터를 만들지 못했습니다: {fail['e']}")
            return
        bar.progress(1.0, text=f"가상데이터 {DEMO_FILES}개 파일을 만들었습니다 — {int(time.time() - t0)}초")
        note.empty()
    with st.spinner("만든 자료를 읽는 중…"):
        load_catalog(str(d))


def steps_bar(now: int):
    """지금 어디까지 왔는지 한 줄로 — 처음 쓰는 사람이 순서를 헷갈리지 않게."""
    labels = ["데이터 불러오기", "질문하기", "결과 보기", "반출 신청"]
    parts = []
    for i, t in enumerate(labels):
        if i:
            parts.append("<span class='sep'>›</span>")
        parts.append(f"<span class='{'on' if i == now else ''}'>{t}</span>")
    st.markdown("<div class='do-step'>" + "".join(parts) + "</div>", unsafe_allow_html=True)


def top_banner():
    """'가상데이터' 표시는 화면 맨 위에 얇은 띠로 늘 보이게 한다."""
    cat = st.session_state.catalog
    if cat is not None and cat.synthetic:
        st.markdown("<div class='do-ribbon'><b>가상데이터</b> — 실제와 같은 형식으로 만든 가짜 자료입니다. "
                    "화면과 보고서의 모든 숫자는 시연용이며, 실제 의사결정에 쓰지 마세요.</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 사이드바
# ---------------------------------------------------------------------------
def sidebar():
    ss = st.session_state
    s = ss.settings
    with st.sidebar:
        st.markdown("### 데이터올레")
        st.caption("탐나는전 캐시백 설계 에이전트")
        st.markdown("<hr class='do-hr' style='margin:0.8rem 0 1rem'>", unsafe_allow_html=True)
        cat = ss.catalog
        if cat is None:
            st.markdown("**데이터** 아직 없음")
        else:
            n_ds = len(cat.inventory.by_dataset) if cat.inventory is not None else len(cat.tables)
            st.markdown(f"**데이터** {n_ds}종 불러옴"
                        + ("  \n<span class='do-tag syn'>가상데이터</span>" if cat.synthetic else ""),
                        unsafe_allow_html=True)
        gw = gateway()
        ai = PROVIDER_LABEL.get(s.provider, s.provider) if gw.enabled else "사용 안 함(규칙 기반)"
        st.markdown(f"**AI** {ai}")
        if gw.enabled and gw.external:
            st.caption(LEVEL_TEXT[s.level])
        st.caption(f"결과 자동 삭제 {'켜짐' if s.auto_delete else '꺼짐'}")
        n, size = ss.ws.stats()
        st.caption(f"이번 작업 폴더: 파일 {n}개 · {size / 1e6:.1f}MB · 세션 {ss.ws.id}")
        st.markdown("<hr class='do-hr' style='margin:1rem 0'>", unsafe_allow_html=True)
        if st.button("작업 끝내기(결과 삭제)", width="stretch", key="finish"):
            confirm_delete()
        if ss.deleted_msg:
            st.success(ss.deleted_msg)


@st.dialog("작업을 끝낼까요?")
def confirm_delete():
    ss = st.session_state
    n, size = ss.ws.stats()
    st.write(f"이번 작업에서 만든 그림·보고서·반출 묶음·기록 **파일 {n}개**({size / 1e6:.1f}MB)와 메모리의 분석 데이터를 모두 지웁니다. "
             "반출 신청에 쓸 파일은 먼저 옮겨 두세요.")
    st.caption("센터의 원본 CSV(요청파일 폴더)는 지우지 않습니다. 원본 정리는 센터 운영 절차를 따릅니다.")
    c1, c2 = st.columns(2)
    if c2.button("취소", type="primary", width="stretch", key="del_no"):
        st.rerun()
    if c1.button("지우고 끝내기", width="stretch", key="del_yes"):
        if ss.catalog is not None:
            try:
                ss.catalog.con.close()
            except Exception:
                pass
        n, size = ss.ws.delete()
        settings = ss.settings
        for k in list(ss.keys()):
            del ss[k]
        _init()
        st.session_state.settings = settings
        st.session_state.deleted_msg = f"파일 {n}개({size / 1e6:.1f}MB)를 삭제했습니다. 새 작업을 시작할 수 있습니다."
        st.rerun()


# ---------------------------------------------------------------------------
# ① 질문
# ---------------------------------------------------------------------------
EX_LABEL = ["동네 진단 — 일도1동", "예산 400억 원 배분", "전통시장 추가 적립 효과",
            "앞당긴 소비 걸러 내기", "쏠림 막는 배분", "창업 시장분석 — 애월읍 카페"]


def empty_state():
    """데이터가 없을 때 첫 화면 — 설명 두 줄과 주 버튼 하나."""
    st.markdown("### 탐나는전 캐시백 설계 에이전트")
    st.markdown("<div class='do-sub'>질문 한 줄을 넣으면 데이터를 고르고 진단·효과 추정·예산 배분안·보고서·실행물·"
                "반출 서류까지 만듭니다. 숫자는 모두 이 PC에서 계산합니다.<br>"
                "먼저 데이터가 있어야 합니다. 센터에서 받은 CSV가 없으면 아래 버튼으로 바로 해 볼 수 있습니다.</div>",
                unsafe_allow_html=True)
    if st.button("가상데이터로 바로 체험하기", type="primary", key="demo_q"):
        load_demo()
        st.rerun()
    st.caption("실제와 같은 형식의 가짜 자료를 만듭니다(처음 한 번 1~2분, 약 700MB). 모든 결과에 '가상데이터'라고 표시됩니다.")
    st.markdown(
        "<div class='do-concl' style='grid-template-columns:1fr 1fr 1fr'>"
        "<div class='do-cell'><div class='do-k'>1. 진단</div><div class='do-v2'>어디에 얼마나 쓰이나</div>"
        "<div class='do-d'>동네·업종·요일·날씨별 사용 흐름과 적립이 닿지 않는 곳을 찾습니다.</div></div>"
        "<div class='do-cell'><div class='do-k'>2. 처방</div><div class='do-v2'>예산 안에서 어떻게 나눌까</div>"
        "<div class='do-d'>적립률 반응도에서 앞당긴 소비를 걸러 내고, 배분안 다섯 개의 소진일과 몫을 비교합니다.</div></div>"
        "<div class='do-cell'><div class='do-k'>3. 실행물·반출</div><div class='do-v2'>바로 쓸 서류까지</div>"
        "<div class='do-d'>보고서 PDF, 개선안·안내문·홍보 문구, 반출 신청 문구와 점검표를 만듭니다.</div></div>"
        "</div>", unsafe_allow_html=True)
    st.markdown("<hr class='do-hr'>", unsafe_allow_html=True)
    st.markdown("센터에서 받은 CSV가 있으면 **② 데이터** 탭에서 폴더를 지정하세요. 어떤 자료인지, 몇 월인지 자동으로 알아냅니다.")


def ask_box():
    """예시 질문 · 입력칸 · 실행 단추."""
    ss = st.session_state
    st.markdown("**예시 질문** — 누르면 아래 칸에 들어갑니다. 고쳐 써도 됩니다.")
    for row in ((0, 1, 2), (3, 4, 5)):
        cols = st.columns(len(row))
        for c, i in zip(cols, row):
            if c.button(EX_LABEL[i], width="stretch", help=EXAMPLES[i]):
                ss.q = EXAMPLES[i]
    q = st.text_area("질문", value=ss.get("q", ""), height=90, label_visibility="collapsed",
                     placeholder="예) 일도1동, 탐나는전으로 동네를 살릴 수 있을까요?")
    b1, b2, _ = st.columns([1.3, 1, 2])
    if b1.button("에이전트에게 맡기기", type="primary", width="stretch"):
        if not q.strip():
            st.warning("질문을 입력해 주세요.")
        else:
            ss.q, ss.mode, ss.plan = q, "agent", None
            run_agent(q)
            st.rerun()
    if b2.button("직접 설정해서 분석", width="stretch"):
        if not q.strip():
            st.warning("질문을 입력해 주세요.")
        else:
            with st.spinner("계획을 세우는 중…"):
                ss.plan = ai_plan(q, gateway())
                ss.q, ss.mode = q, "manual"


def tab_question():
    ss = st.session_state
    steps_bar(0 if ss.catalog is None else (2 if ss.get("output") is not None else 1))

    if ss.catalog is None:
        empty_state()
        return

    # 답이 있으면 답부터 보여 준다 — 내려보지 않아도 결론이 먼저 보이게
    if ss.get("mode") == "agent" and ss.get("agent") is not None:
        show_agent(ss.agent)
        st.markdown("<hr class='do-hr'>", unsafe_allow_html=True)
        with st.expander("다른 질문 하기"):
            ask_box()
        return

    st.markdown("### 무엇을 알고 싶으세요?")
    st.markdown("<div class='do-sub'>사용 패턴 진단 → 적립률 반응도 추정(앞당긴 소비 분리) → 예산 배분안 시뮬레이션"
                "(소진일·전통시장 몫·쏠림) → 개선안·안내문 → 효과 확인까지 한 번에 합니다. "
                "질문 속 숫자는 외부 AI로 보내지 않습니다.</div>", unsafe_allow_html=True)
    ask_box()

    plan = ss.plan
    if plan is None:
        return
    st.markdown("<hr class='do-hr'>", unsafe_allow_html=True)
    st.markdown(f"#### 분석 계획 <span class='do-tag ai'>{'AI 계획' if plan.by_ai else '규칙 기반 계획'}</span>",
                unsafe_allow_html=True)
    if plan.notes:
        st.markdown(md(plan.notes))
    for p in plan.problems:
        st.caption(f"참고: {p}")
    if plan.course == "market":
        c1, c2 = st.columns(2)
        bizs = list(BIZ_LINK) + [b for b in biz_names() if b not in BIZ_LINK]
        biz = c1.selectbox("업종", bizs, index=bizs.index(plan.market.get("biz")) if plan.market.get("biz") in bizs else 0)
        names = bjd_names()
        bjd = c2.selectbox("후보 법정동", names, index=names.index(plan.market["bjd"]) if plan.market.get("bjd") in names else 0)
        persona = st.text_input("보고서 대상(선택)", value=f"{bjd} {BIZ_LINK.get(biz, {}).get('label', biz)} 창업 준비자")
        plan.market.update({"biz": biz, "bjd": bjd, "persona": persona})
        st.caption("창업 시장분석: 시장 규모·추세 → 입지 비교(특화지수) → 고객 구성 → 요일·시간대 → 객단가 → 외식업 POS → 유동인구")
    elif plan.course == "status":
        names = bjd_names()
        plan.market["bjd"] = st.selectbox("대상 법정동", names, index=names.index(plan.market["bjd"]) if plan.market.get("bjd") in names else 0)
        st.caption("동네 현황자료: 소비 규모·추세 → 업종 구성·특화 업종 → 이용자 구성 → 요일·시간대 → 유동인구")
    elif plan.course == "tamna":
        c1, c2 = st.columns(2)
        kinds = {"all": "제주 전체", "city": "시", "hd": "동네(행정동) — 동네 진단 카드", "emd": "읍면동(법정동) — 동네 진단 카드"}
        cur = plan.market.get("scope", "all")
        sc = c1.selectbox("분석 범위", list(kinds), index=list(kinds).index(cur) if cur in kinds else 0, format_func=kinds.get)
        plan.market["scope"] = sc
        if sc == "city":
            plan.market["value"] = c2.selectbox("시", ["제주시", "서귀포시"],
                                                index=1 if plan.market.get("value") == "서귀포시" else 0)
        elif sc == "emd":
            names = bjd_names()
            plan.market["value"] = c2.selectbox("읍면동", names,
                                                index=names.index(plan.market["value"]) if plan.market.get("value") in names else 0)
        elif sc == "hd":
            units = sorted(UNIT_GROUP, key=lambda u: (["원도심", "신도심", "그 밖의 동", "읍면"].index(UNIT_GROUP[u]), u))
            plan.market["value"] = c2.selectbox("동네", units, index=units.index(plan.market["value"]) if plan.market.get("value") in units else 0,
                                                format_func=lambda u: f"{u} ({UNIT_GROUP[u]})")
        else:
            plan.market["value"] = None
        months = ss.catalog.months("tamna") if ss.catalog is not None and ss.catalog.has("tamna") else []
        opts = ["(분석 안 함)"] + (months[3:-1] if len(months) >= 6 else [])
        ev = (plan.market.get("event") or "")[:6]
        pick = st.selectbox("정책 시작 월(전통시장 vs 일반 가맹점 전후 비교, 선택)", opts, index=opts.index(ev) if ev in opts else 0)
        plan.market["event"] = None if pick == "(분석 안 함)" else pick
        if plan.market["event"]:
            plan.market["event_name"] = st.text_input("정책 이름(선택)", value=plan.market.get("event_name", ""))
        if "policy" not in plan.market:
            plan.market["policy"] = parse_policy(plan.question, effect_question=bool(plan.market.get("event")))
        pk = plan.market["policy"]
        st.markdown("**처방 설계(적립 예산 배분)**")
        p1, p2, p3, p4 = st.columns(4)
        bud = p1.number_input("적립 예산(억 원)", min_value=0.0, value=float((pk.get("budget") or 0) / 1e8), step=10.0,
                              help="0이면 2026년 비율(예산 425억 원 ÷ 2025년 사용액 6,800억 원 = 6.25%)을 자료 사용액에 적용")
        mon = p2.number_input("이어 갈 기간(개월)", min_value=1, max_value=12, value=int(pk.get("months", 12)))
        stt = p3.number_input("시작 월", min_value=1, max_value=12, value=int(pk.get("start", 1)))
        goals = list(GOAL_LABEL)
        gl = p4.selectbox("우선순위", goals, index=goals.index(pk.get("goal", "auto")) if pk.get("goal", "auto") in goals else 0,
                          format_func=GOAL_LABEL.get)
        plan.market["policy"] = {"budget": bud * 1e8 if bud else None, "months": int(mon), "start": int(stt), "goal": gl}
        st.caption("탐나는전 캐시백 설계: (동네 진단 카드) → 사용 규모 → 전통시장·오일장 → 침투지수·사각지대 → 원도심·신도심 상권 → "
                   "인구 구조·동네 유형 → 학령인구 감소 경보 → 요일·비 오는 날 → 정책 전후 효과 → 동네 유형별 처방 → "
                   "효과 추정(적립률 반응도·앞당긴 소비·백테스트) → 처방 설계(A 현행·B 균등·C 동네 살리기 집중·D 사각지대 보강·"
                   "E 쏠림 방지, 효과 확인 뒤 C′) → 실행물 묶음")
    elif plan.course == "before_after":
        c1, c2, c3 = st.columns(3)
        names = bjd_names()
        plan.market["bjd"] = c1.selectbox("처리 지역(법정동)", names,
                                          index=names.index(plan.market["bjd"]) if plan.market.get("bjd") in names else 0)
        bizs = ["(전체 업종)"] + list(BIZ_LINK) + [b for b in biz_names() if b not in BIZ_LINK]
        cur = plan.market.get("biz") or "(전체 업종)"
        pick = c2.selectbox("업종", bizs, index=bizs.index(cur) if cur in bizs else 0)
        plan.market["biz"] = None if pick == "(전체 업종)" else pick
        months = ss.catalog.months("card") if ss.catalog is not None and ss.catalog.has("card") else []
        opts = months[3:-2] if len(months) >= 6 else [plan.market.get("event", "202507")]
        ev = plan.market.get("event")
        plan.market["event"] = c3.selectbox("사건 시작 월", opts, index=opts.index(ev) if ev in opts else len(opts) // 2)
        plan.market["event_name"] = st.text_input("사건 이름(선택, 예: ○○ 개장)", value=plan.market.get("event_name", ""))
        st.caption("사건·정책 전후 효과: 사건 전 추이가 비슷한 법정동을 비교 지역으로 자동 선택 → 이중차분 효과 → 공간 위약 검정 → 논문·보고용 문단")
    else:
        for a in plan.analyses:
            m = MODELS[a.dataset]
            st.markdown(f"- **{m.title}** · {m.measure(a.measure).label}"
                        + (f" · {m.dim(a.rows).label}별" if a.rows else "")
                        + (f" × {m.dim(a.cols).label}" if a.cols else "")
                        + (f" · 조건 {', '.join(f'{m.dim(k).label}={v}' for k, v in a.filters.items())}" if a.filters else "")
                        + (f" · {a.transform}" if a.transform != "none" else ""))
    checklist(plan)
    if ss.catalog is None:
        st.info("② 데이터 탭에서 센터 CSV 폴더(또는 시연용 가상데이터)를 먼저 불러오세요.")
        return
    if st.button("분석 실행", type="primary"):
        run_plan(plan)
    if ss.get("flash"):
        st.success(ss.flash)


def checklist(plan):
    ss = st.session_state
    reg = default_registry()
    cat = ss.catalog
    with st.expander("필요한 데이터와 다운로드 체크리스트", expanded=cat is None):
        if plan.course in COURSE_NEEDS:
            needs = COURSE_NEEDS[plan.course]
        else:
            needs = {ds: 12 for ds in plan.datasets}
        end = None
        if cat is not None and cat.inventory is not None:
            months = sorted({m for ds in needs for m in cat.inventory.months(ds)})
            end = months[-1] if months else None
        end = end or "202512"
        items = []
        for ds, n in needs.items():
            if n == 0:                                         # 비교용 스냅숏 1개월(2017년 같은 달)
                items += make_plan([ds], f"2017{end[4:]}", f"2017{end[4:]}")
            else:
                items += make_plan([ds], month_range("200001", end)[-n], end)
        if cat is not None and cat.inventory is not None:
            items = check_plan(items, cat.inventory)
        rows = []
        for ds, n in needs.items():
            mine = [i for i in items if i.dataset_id == ds]
            done = sum(1 for i in mine if i.status == "done")
            have_any = cat is not None and cat.inventory is not None and bool(cat.inventory.months(ds))
            label = f"2017년 {int(end[4:])}월(비교용 1개월)" if n == 0 else f"최근 {n}개월"
            rows.append({"데이터": f"{ds} {reg.get(ds).name_ko}", "필요 기간": label,
                         "받음": f"{done}/{len(mine)}" + (" (다른 달 있음)" if n == 0 and not done and have_any else "")})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=36 * (len(rows) + 1) + 3)
        st.caption("센터 PC에서는 한 번에 1개월씩 내려받습니다. 아래 문안을 운영자에게 전달하면 미리 '요청파일' 폴더에 받아 둡니다.")
        st.code(operator_request_text(items), language=None, wrap_lines=True)


def hx(text) -> str:
    """HTML 칸에 넣는 글 — 꺾쇠와 물결표를 안전하게 바꾼다."""
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("~", "&#126;"))


def timeline_html(steps) -> str:
    """'계획 → 도구 실행 → 결과'를 세로 타임라인으로."""
    rows = []
    for st_ in steps:
        kind = getattr(st_, "kind", "")
        detail = getattr(st_, "detail", "")
        d = f"<div class='d'>{hx(detail)}</div>" if detail else ""
        rows.append(f"<div class='row {kind}'><div class='k'>{STEP_LABEL.get(kind, '진행')}</div>"
                    f"<div class='t'>{hx(getattr(st_, 'title', ''))}</div>{d}</div>")
    return "<div class='do-tl'>" + "".join(rows) + "</div>"


def saw_html(by_ai: bool) -> str:
    """이 작품의 차별점 — 숫자는 이 PC가 계산한다는 것을 화면에서 바로 보이게."""
    seen = ("질문(숫자는 가림) · 데이터 목록 · 사실 번호와 설명 · 동네 이름" if by_ai
            else "없음 — AI를 쓰지 않고 규칙 기반으로 돌렸습니다. 밖으로 나간 내용이 없습니다.")
    return (f"<div class='do-saw'><div><b>AI가 본 것</b>{seen}</div>"
            "<div><b>AI가 보지 못한 것</b>센터 원본 자료 · 실제 수치 · 개인 정보</div></div>"
            "<div class='do-d' style='max-width:860px'>표·그림의 숫자는 모두 이 PC(DuckDB)가 계산했습니다. "
            "실제로 보낸 내용은 관리자 탭의 전송 기록에서 볼 수 있습니다.</div>")


def run_agent(q: str):
    ss = st.session_state
    gw = gateway()
    out_dir = ss.ws.sub("결과")
    with st.status("에이전트가 일하는 중입니다 — 보통 20~40초 걸립니다.", expanded=True) as status:
        rail = status.empty()
        seen = []

        def prog(step):
            seen.append(step)
            rail.markdown(timeline_html(seen), unsafe_allow_html=True)

        res = Agent(ss.catalog, out_dir, gw if gw.enabled else None, sdc_level(), progress=prog).run(q)
        status.update(label=f"에이전트가 일을 마쳤습니다 — {len(seen)}단계", state="complete", expanded=False)
    ss.agent = res
    ss.output = res.outputs[-1] if res.outputs else None
    ss.free = list(res.free)
    ss.pack = None
    ss.action_pack = res.packs[-1] if res.packs else None
    ss.plan = res.plan


def answer_md(text: str) -> str:
    """'[진단] … [처방] … [실행물] … [효과 확인]' 답을 단계별 문단으로."""
    import re
    parts = re.split(r"\s*\[(진단|처방|실행물|효과 확인)\]\s*", text or "")
    if len(parts) == 1:
        return md(text)
    out = [md(parts[0].strip())] if parts[0].strip() else []
    for tag, body in zip(parts[1::2], parts[2::2]):
        out.append(f"**{tag}** · {STAGE_NOTE.get(tag, '')}\n\n{md(body.strip())}")
    return "\n\n".join(out)


def conclusion_card(out):
    """답 맨 위 결론 칸 — 추천안·예산 유지(소진일)·전통시장 몫만 크게."""
    pol = out.report.extra.get("policy")
    if not pol:
        return
    S, rec, tab = pol["scen"], pol["rec"], pol["table"]
    row = next((i for i in tab.index if str(i).startswith(f"{rec}안")), None)
    base = next((i for i in tab.index if str(i).startswith("A안")), None)
    cells = [f"<div class='do-cell lead'><div class='do-k'>추천하는 배분안</div>"
             f"<div class='do-v'>{hx(rec)}안 {hx(S[rec].name)}</div>"
             f"<div class='do-d'>{hx(pol.get('why', ''))}</div></div>"]
    for col, label in (("예산 유지", "예산이 버티는 기간"), ("전통시장", "전통시장에 가는 몫")):
        if row is None or col not in tab.columns:
            continue
        now = f"지금 방식(A안) {hx(tab.at[base, col])}" if base is not None else ""
        cells.append(f"<div class='do-cell'><div class='do-k'>{label}</div>"
                     f"<div class='do-v2'>{hx(tab.at[row, col])}</div><div class='do-d'>{now}</div></div>")
    st.markdown("<div class='do-concl'>" + "".join(cells) + "</div>", unsafe_allow_html=True)


def next_steps(out):
    """답 바로 아래 '다음 단계' 세 가지 — 어느 탭으로 가야 하는지 찾지 않게."""
    ss = st.session_state
    pol = out.report.extra.get("policy")
    st.markdown("**다음 단계**")
    c1, c2, c3 = st.columns(3)
    if out.pdf:
        c1.download_button("보고서 PDF 받기", out.pdf.read_bytes(), out.pdf.name, "application/pdf",
                           width="stretch", key="ns_pdf")
    else:
        c1.caption("PDF 변환 실패 — ③ 분석 결과 탭에서 인쇄(Ctrl+P)로 저장하세요.")
    if c2.button("실행물 만들기", width="stretch", key="ns_pack", disabled=not pol):
        with st.spinner("도청 개선안·상인회 안내문·홍보 문구·어르신 안내문·일정표를 만드는 중…"):
            gw = gateway()
            try:
                ss.action_pack = make_action_pack(out, ss.ws.sub("결과") / "실행물", pol["rec"],
                                                  gateway=gw if gw.enabled else None)
            except (ValueError, KeyError) as e:
                st.error(str(e))
    if c3.button("반출 묶음 만들기", width="stretch", key="ns_export"):
        with st.spinner("반출할 파일을 모으고 설명자료를 만드는 중…"):
            ap = ss.get("action_pack")
            extra = ([(ap.pdf, f"실행물 묶음 '{ap.report.title}'({ap.scenario}안) — 개선안 초안·안내문·홍보 문구·일정표")]
                     if ap is not None and ap.pdf else [])
            use = {"market": "business", "status": "status", "before_after": "paper",
                   "tamna": "policy"}.get(ss.plan.course if ss.plan else "market", "business")
            ss.pack = build_pack(out.report, ss.ws.sub("반출"), use=use, report_pdf=out.pdf, extra_pdfs=extra)
        if ss.pack is not None and ss.pack.ok:
            st.success(f"반출용 파일 {len(ss.pack.files)}개를 모았습니다. "
                       "④ 반출 신청 탭에서 신청서에 넣을 문구와 점검표를 보세요.")
    if ss.get("action_pack") is not None:
        st.caption(f"실행물 묶음을 만들었습니다({ss.action_pack.scenario}안). 아래 '처방 설계' 칸에서 내려받으세요.")


def show_agent(res):
    ss = st.session_state
    badge = "AI 에이전트" if res.by_ai else "규칙 기반 에이전트"
    st.markdown(f"### 답 <span class='do-tag ai'>{badge}</span>", unsafe_allow_html=True)
    out = res.outputs[-1] if res.outputs else None
    if out is not None:
        conclusion_card(out)
        next_steps(out)
        st.markdown("<hr class='do-hr'>", unsafe_allow_html=True)
    st.markdown(answer_md(res.answer))
    st.markdown(saw_html(res.by_ai), unsafe_allow_html=True)
    if res.request_text:
        with st.expander("운영자 사전 다운로드 요청 문안(센터 운영자에게 전달하면 '요청파일' 폴더에 받아 둡니다)"):
            st.code(res.request_text, language=None, wrap_lines=True)
    with st.expander("에이전트가 한 일 — 계획 › 도구 실행 › 결과"):
        st.markdown(timeline_html(res.steps), unsafe_allow_html=True)
        for pb in res.problems[:3]:
            st.caption(f"참고: {pb}")
    if out is not None:
        st.markdown("<hr class='do-hr'>", unsafe_allow_html=True)
        policy_panel(out, "agent")


def scen_table(pol):
    """A~E 비교표 — 추천 행을 바탕색으로 드러낸다.
    적립률 규칙은 문장이 길어 칸 안에서 잘리므로 표 밖으로 빼서 목록으로 적는다."""
    tab = pol["table"].reset_index()
    rules = []
    if "적립률" in tab.columns:
        rules = list(zip(tab.iloc[:, 0].astype(str).tolist(), tab["적립률"].astype(str).tolist()))
        tab = tab.drop(columns=["적립률"])
    rec = str(pol["rec"])

    def mark(row):
        on = str(row.iloc[0]).startswith(f"{rec}안")
        return ["background-color: #eef2f8; font-weight: 700" if on else "" for _ in row]

    st.dataframe(tab.style.apply(mark, axis=1), hide_index=True, width="stretch",
                 height=36 * (len(tab) + 1) + 3,
                 column_config={"적립 방식": st.column_config.TextColumn(width="medium")})
    if rules:
        st.markdown("**적립 방식별 규칙**\n" + "\n".join(f"- **{md(k)}** — {md(v)}" for k, v in rules))


def policy_panel(out, key: str):
    """처방 설계(적립 예산 배분) 표·그림과 실행물 묶음 만들기·내려받기."""
    ss = st.session_state
    pol = out.report.extra.get("policy")
    if not pol:
        return
    st.markdown("#### 처방 설계 — 탐나는전 적립 예산 배분")
    S = pol["scen"]
    st.markdown(md(f"추천: **{pol['rec']}안 {S[pol['rec']].name}** — {pol['why']}."))
    scen_table(pol)
    if pol["figures"].get("burn"):
        st.image(str(pol["figures"]["burn"]))
    keys = list(pol["order"])
    cur = ss.get("action_pack")
    default = keys.index(cur.scenario) if cur is not None and cur.scenario in keys else keys.index(pol["rec"])
    c1, c2 = st.columns([2, 1])
    k = c1.selectbox("실행물에 쓸 배분안", keys, index=default, key=f"{key}_sc",
                     format_func=lambda x: f"{x}안 {S[x].name}" + (" (추천)" if x == pol["rec"] else ""))
    if c2.button("실행물 묶음 만들기", key=f"{key}_btn", type="primary", width="stretch"):
        with st.spinner("도청 개선안·상인회 안내문·홍보 문구·어르신 안내문·일정표를 만드는 중…"):
            gw = gateway()
            try:
                ss.action_pack = make_action_pack(out, ss.ws.sub("결과") / "실행물", k, gateway=gw if gw.enabled else None)
            except (ValueError, KeyError) as e:
                st.error(str(e))
    pk = ss.get("action_pack")
    if pk is None:
        st.caption("배분안을 고르고 '실행물 묶음 만들기'를 누르면 바로 쓰는 문서 8종을 만듭니다.")
        return
    st.markdown(f"**실행물 묶음 — {pk.scenario}안** " + ("(홍보 문구: AI가 다듬음)" if pk.polished else "(홍보 문구: 규칙 기반)"))
    st.caption(" · ".join(pk.titles[:-1]))
    d1, d2 = st.columns(2)
    if pk.pdf:
        d1.download_button("실행물 묶음 PDF", pk.pdf.read_bytes(), pk.pdf.name, "application/pdf", width="stretch", key=f"{key}_pdf")
    if pk.docx:
        d2.download_button("편집용 DOCX(센터 안)", pk.docx.read_bytes(), pk.docx.name,
                           "application/vnd.openxmlformats-officedocument.wordprocessingml.document", width="stretch",
                           key=f"{key}_docx")
    with st.expander("홍보 문구 미리보기(복사해서 쓰기)"):
        for lab, kk in (("현수막 1안", "banner1"), ("현수막 2안", "banner2"), ("현수막 3안", "banner3"), ("SNS", "sns"),
                        ("문자(단문)", "sms_short"), ("문자(장문)", "sms_long"), ("관광객(국문)", "tour_ko"), ("관광객(영문)", "tour_en")):
            st.markdown(f"**{lab}** · {len(pk.copy[kk])}자")
            st.code(pk.copy[kk], language=None, wrap_lines=True)
    st.caption(f"효과 확인: 시행 12주 뒤 “{pol['question']}”라고 물으면 전후 효과를 판정하고 다음 배분(C′안)을 다시 계산합니다.")


def run_plan(plan):
    ss = st.session_state
    out_dir = ss.ws.sub("결과")
    gw = gateway()
    if plan.course in COURSE_NEEDS:
        prog = st.empty()
        mk = plan.market
        with st.spinner("분석하는 중… (보통 10초 안팎)"):
            try:
                if plan.course == "market":
                    ss.output = MarketCourse(ss.catalog, out_dir, sdc_level()).run(
                        MarketInput(biz=mk["biz"], bjd=mk["bjd"], persona=mk.get("persona", "")),
                        progress=lambda m: prog.caption(m), gateway=gw if gw.enabled else None)
                elif plan.course == "status":
                    ss.output = StatusCourse(ss.catalog, out_dir, sdc_level()).run_status(mk["bjd"], progress=lambda m: prog.caption(m))
                elif plan.course == "tamna":
                    ss.output = TamnaCourse(ss.catalog, out_dir, sdc_level()).run(
                        TamnaInput(scope=mk.get("scope", "all"), value=mk.get("value"), event=mk.get("event"),
                                   event_name=mk.get("event_name", ""), policy=PolicyInput(**(mk.get("policy") or {}))),
                        progress=lambda m: prog.caption(m), gateway=gw if gw.enabled else None)
                else:
                    ss.output = BeforeAfterCourse(ss.catalog, out_dir, sdc_level()).run(
                        BeforeAfterInput(treat=[mk["bjd"]], event=mk["event"], biz=mk.get("biz"), event_name=mk.get("event_name", "")),
                        progress=lambda m: prog.caption(m))
            except (ValueError, KeyError) as e:
                st.error(str(e))
                return
        prog.empty()
        ss.pack = None
        ss.action_pack = None
        ss.flash = f"완료({ss.output.seconds:.0f}초). ③ 분석 결과 탭에서 보고서를 확인하세요."
        st.rerun()
    else:
        ss.free = []
        fs = FactStore()
        for i, a in enumerate(plan.analyses, start=1):
            try:
                r = run_safe(a, ss.catalog, fs, sdc_level())
            except (ValueError, KeyError) as e:
                st.error(f"{MODELS[a.dataset].title}: {e}")
                continue
            img = render_chart(r, out_dir / f"자유분석_{i:02d}.png")
            ss.free.append((r, img, fs))
        ss.flash = "완료. ③ 분석 결과 탭에서 확인하세요."
        st.rerun()


# ---------------------------------------------------------------------------
# ② 데이터
# ---------------------------------------------------------------------------
def tab_data():
    ss = st.session_state
    s = ss.settings
    st.markdown("### 데이터 불러오기")
    st.markdown("<div class='do-sub'>센터에서 내려받은 CSV(Results_….csv) 폴더를 지정하면 어떤 데이터·몇 월인지 자동으로 알아냅니다. "
                "원본은 이 PC 메모리 안에서만 읽습니다.</div>", unsafe_allow_html=True)
    c1, c2 = st.columns([4, 1])
    folder = c1.text_input("CSV 폴더", value=s.data_folder, label_visibility="collapsed")
    if c2.button("폴더 읽기", type="primary", width="stretch"):
        if not Path(folder).exists():
            st.error("폴더가 없습니다. 경로를 확인해 주세요.")
        else:
            with st.spinner("파일을 확인하고 불러오는 중…"):
                load_catalog(folder)
            st.rerun()
    with st.expander("센터 데이터가 없나요? 시연용 가상데이터로 체험하기", expanded=ss.catalog is None):
        st.write("실제와 같은 형식(헤더·따옴표·NA·월별 파일)의 **가상데이터**를 만들어 불러옵니다. 모든 결과에 '가상데이터' 표시가 붙습니다.")
        if st.button("가상데이터 만들고 불러오기", type="primary" if ss.catalog is None else "secondary", key="demo_d"):
            load_demo()
            st.rerun()
    cat = ss.catalog
    if cat is None or cat.inventory is None:
        return
    inv = cat.inventory
    reg = default_registry()
    rows = []
    for ds in sorted(inv.by_dataset):
        months = inv.months(ds)
        spec = reg.get(ds)
        rows.append({"데이터": f"{ds} {spec.name_ko}", "기간": f"{months[0]}~{months[-1]}" if months else "-",
                     "개월": len(months), "파일": len(inv.files_for(ds)),
                     "분석 지원": "예" if ds in {i for m in MODELS.values() for i in m.dataset_ids} else "목록만"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=36 * (len(rows) + 1) + 3)
    warns = []
    for rec in inv.unknown:
        warns.append(f"알 수 없는 파일: {Path(rec.path).name}")
    for rec in inv.ambiguous:
        warns.append(f"헤더가 같은 데이터가 둘 이상: {Path(rec.path).name}")
    for ds, ym, paths in inv.duplicates():
        warns.append(f"같은 달 파일 중복({ds} {ym}): 최신 파일만 사용")
    for rec in inv.files:
        if rec.status == "public":                         # 공공데이터 파일은 센터 형식 경고 대상이 아님
            continue
        for w in rec.warnings:
            warns.append(f"{Path(rec.path).name}: {w}")
    for w in warns[:30]:
        st.caption(f"확인 필요: {w}")


# ---------------------------------------------------------------------------
# ③ 결과
# ---------------------------------------------------------------------------
def tab_result():
    ss = st.session_state
    out = ss.output
    if out is None and not ss.free:
        st.info("① 질문 탭에서 분석을 실행하면 결과가 여기에 나타납니다.")
        return
    if out is not None:
        rep = out.report
        st.markdown(f"### {rep.title}")
        if rep.synthetic:
            st.caption("가상데이터(시연용)로 만든 결과입니다. 실제 의사결정에 쓰지 마세요.")
        conclusion_card(out)
        ok = sum(1 for r in rep.results if r.check.passed)
        st.caption(f"그림 {len(out.figures)}개 · 표·그림 반출 검수 {ok}/{len(rep.results)} 통과 · "
                   f"비공개(x) 처리한 칸 {sum(r.check.masked_cells for r in rep.results)}개 · "
                   f"만드는 데 {out.seconds:.0f}초")
        d1, d2, d3 = st.columns(3)
        if out.pdf:
            d1.download_button("보고서 PDF", out.pdf.read_bytes(), out.pdf.name, "application/pdf", width="stretch")
        else:
            d1.caption("PDF 자동 변환 실패: 아래 미리보기에서 인쇄(Ctrl+P) → PDF로 저장")
        if out.docx:
            d2.download_button("편집용 DOCX", out.docx.read_bytes(), out.docx.name,
                               "application/vnd.openxmlformats-officedocument.wordprocessingml.document", width="stretch")
        d3.download_button("미리보기 HTML", out.html.read_bytes(), out.html.name, "text/html", width="stretch")
        st.caption("반출 신청에는 PDF·PNG를 쓰세요(텍스트·HTML·엑셀은 원칙적으로 반출 불가). DOCX는 센터 안 편집용입니다.")
        plan_text = rep.extra.get("plan_text", "")
        if plan_text:
            st.markdown(f"#### {rep.extra.get('plan_label', '사업계획서 붙여넣기용 문단')}")
            st.code(plan_text, language=None, wrap_lines=True)
        if rep.extra.get("methods_text"):
            st.markdown("#### 논문용 문단 — 데이터 및 방법")
            st.code(rep.extra["methods_text"], language=None, wrap_lines=True)
            st.markdown("#### 논문용 문단 — 결과")
            st.code(rep.extra["results_text"], language=None, wrap_lines=True)
        policy_panel(out, "result")
        with st.expander("보고서 미리보기", expanded=True):
            import streamlit.components.v1 as components
            components.html(out.html.read_text(encoding="utf-8"), height=900, scrolling=True)
        with st.expander("근거 보기(모든 수치의 출처)"):
            st.dataframe(pd.DataFrame([{"번호": f.id, "설명": f.label, "값": f.text, "유형": f.kind.value, "표·그림": f.source}
                                       for f in out.facts]), hide_index=True, width="stretch")
    for r, img, fs in ss.free:
        st.markdown(f"#### {r.title}")
        for sent in r.sentences:
            st.markdown(md(fs.render(sent)))
        if img:
            st.image(str(img))
        st.dataframe(r.display, width="stretch")
        st.caption(f"{r.citation()} · {r.kind.value} · 비공개 {r.check.masked_cells}칸 · "
                   + ("반출 기준 검수 통과" if r.check.passed else "검수 보완 필요"))
        for n in r.notes:
            st.caption(f"참고: {n}")


# ---------------------------------------------------------------------------
# ④ 반출 신청
# ---------------------------------------------------------------------------
def tab_export():
    ss = st.session_state
    out = ss.output
    st.markdown("### 반출 신청 도우미")
    st.markdown("<div class='do-sub'>반출할 파일(PDF·PNG)을 한 폴더에 모으고, 신청서에 붙여 넣을 활용 사유·설명과 심사자용 점검표를 만듭니다. "
                "최종 승인은 센터 심사로 결정됩니다.</div>", unsafe_allow_html=True)
    if out is None:
        st.info("먼저 ① 질문 탭에서 보고서를 만드세요.")
        return
    labels = {"business": "사업계획서", "status": "기관 현황자료", "paper": "논문", "policy": "정책 협의 자료"}
    default = {"market": 0, "status": 1, "before_after": 2, "tamna": 3}.get(ss.plan.course if ss.plan else "market", 0)
    use = st.radio("활용처", list(labels), index=default, format_func=lambda k: f"{labels[k]} — {USE_PRESETS[k]}", horizontal=False)
    if st.button("반출 묶음 만들기", type="primary"):
        with st.spinner("파일을 모으고 설명자료를 만드는 중…"):
            ap = ss.get("action_pack")
            what = "개선안 초안·안내문·홍보 문구·일정표"
            extra = [(ap.pdf, f"실행물 묶음 '{ap.report.title}'({ap.scenario}안) — {what}")] if ap is not None and ap.pdf else []
            ss.pack = build_pack(out.report, ss.ws.sub("반출"), use=use, report_pdf=out.pdf, extra_pdfs=extra)
    pack = ss.pack
    if pack is None:
        return
    for p in pack.problems:
        st.error(p)
    for w in pack.warnings:
        st.warning(w)
    if pack.ok:
        st.markdown(f"반출 신청용 파일 **{len(pack.files)}개**를 한 폴더에 모았습니다.")
        st.caption(f"{pack.folder}")
    if os.name == "nt" and st.button("폴더 열기"):
        os.startfile(pack.folder)  # noqa: S606
    st.markdown("**① 활용 사유(신청서에 붙여 넣기)**")
    st.code(pack.purpose, language=None, wrap_lines=True)
    st.markdown("**② 설명란 문구**")
    st.code(pack.description, language=None, wrap_lines=True)
    st.markdown("**③ 표·그림별 점검표** (설명자료 PDF에도 들어 있음)")
    st.dataframe(pack.checklist, hide_index=True, width="stretch")
    st.markdown("**④ 파일 목록**")
    files_df = pd.DataFrame([{"파일": f.path.name, "구분": f.role, "내용": f.desc} for f in pack.files])
    st.dataframe(files_df, hide_index=True, width="stretch", height=36 * (len(files_df) + 1) + 3)


# ---------------------------------------------------------------------------
# ⚙ 관리자
# ---------------------------------------------------------------------------
def tab_admin():
    ss = st.session_state
    s: Settings = ss.settings
    home = app_home()
    if not ss.admin_ok:
        st.markdown("### 관리자 설정")
        if not s.has_pin:
            st.write("처음 사용합니다. 센터 담당자용 PIN(숫자 4~8자리)을 만들어 주세요.")
            p1 = st.text_input("새 PIN", type="password", key="pin_new")
            p2 = st.text_input("PIN 확인", type="password", key="pin_chk")
            if st.button("PIN 만들기", type="primary"):
                if not (p1.isdigit() and 4 <= len(p1) <= 8):
                    st.error("숫자 4~8자리로 입력하세요.")
                elif p1 != p2:
                    st.error("두 PIN이 다릅니다.")
                else:
                    s.set_pin(p1)
                    s.save(home)
                    ss.admin_ok = True
                    st.rerun()
        else:
            pin = st.text_input("관리자 PIN", type="password", key="pin_in")
            if st.button("확인", type="primary"):
                if s.check_pin(pin):
                    ss.admin_ok = True
                    st.rerun()
                else:
                    st.error("PIN이 맞지 않습니다.")
        return

    st.markdown("### AI 연결")
    providers = ["anthropic", "openai", "gemini", "ollama", "none"]
    c1, c2 = st.columns(2)
    prov = c1.selectbox("AI 공급자", providers, index=providers.index(s.provider), format_func=lambda p: PROVIDER_LABEL[p])
    model = c2.text_input("모델(비우면 기본값)", value=s.model, placeholder=DEFAULT_MODELS.get(prov, ""))
    if prov in ENV_KEYS:
        env = ENV_KEYS[prov]
        st.caption(f"현재 키: {mask_key(os.environ.get(env))} (data\\.env 의 {env})")
        newkey = st.text_input("API 키 입력·변경", type="password")
        if st.button("키 저장"):
            save_env_key(env, newkey.strip(), home)
            st.success("저장했습니다.")
        if prov == "anthropic":
            ws = st.text_input("워크스페이스 ID(선택: 워크스페이스가 지정되지 않은 키일 때만, wrkspc_…)",
                               value=os.environ.get("ANTHROPIC_WORKSPACE_ID", ""))
            if st.button("워크스페이스 ID 저장"):
                save_env_key("ANTHROPIC_WORKSPACE_ID", ws.strip(), home)
                st.success("저장했습니다.")
    if st.button("설정 저장 후 연결 테스트"):
        s.provider, s.model = prov, model.strip()
        s.save(home)
        gw = gateway()
        if not gw.enabled:
            st.info("AI 사용 안 함 모드이거나 키가 없습니다. 규칙 기반으로 모든 분석이 동작합니다.")
        else:
            r = gw.call("연결 테스트", "Reply with the single word OK.", "ping", max_tokens=5)
            if r:
                st.success(f"연결 성공: {r.model} · {r.seconds:.1f}초")
            else:
                st.error(gw.last_error)
    st.markdown("### 외부 AI 전송 레벨")
    lvl = st.radio("레벨", [0, 1, 2], index=s.level, format_func=lambda v: LEVEL_TEXT[v], label_visibility="collapsed")
    if lvl > 0:
        st.warning("센터 공식 답변: 집계 결과의 외부 AI 전송은 사전 승인되지 않습니다. 레벨 1·2는 기관 책임 아래에서만 사용하세요. "
                   "레벨 0에서도 모든 기능이 동작합니다(숫자는 이 PC에서 채움).")
    st.markdown("### 반출 검수 수준")
    lv_names = {"literal": "문언 기준(표시 수치 10 미만만)", "conservative": "보수(바탕 인원 10 미만도 가림, 권장)",
                "strict": "엄격(비율도 바탕 인원 10 미만이면 가림)"}
    sdc = st.radio("검수", list(lv_names), index=list(lv_names).index(s.sdc_level), format_func=lambda k: lv_names[k],
                   label_visibility="collapsed")
    st.markdown("### 결과 삭제")
    auto = st.toggle("결과 자동 삭제 ON (작업 끝내기·다음 실행 때 이전 세션 결과를 지움)", value=s.auto_delete)
    hours = st.number_input("자동 삭제 대상: 이 시간보다 오래된 세션 폴더", min_value=1, max_value=168, value=s.keep_hours)
    if st.button("설정 저장", type="primary"):
        s.provider, s.model, s.level, s.sdc_level, s.auto_delete, s.keep_hours = prov, model.strip(), lvl, sdc, auto, int(hours)
        s.save(home)
        st.success("저장했습니다.")
    sessions = list_sessions(home)
    st.dataframe(pd.DataFrame(sessions) if sessions else pd.DataFrame({"세션": []}), hide_index=True, width="stretch")
    if sessions and st.button("현재 세션을 뺀 모든 세션 결과 삭제"):
        n = cleanup_sessions(home, 0, keep=ss.ws.id)
        st.success(f"세션 폴더 {n}개를 삭제했습니다.")
    st.markdown("### 전송 기록(감사 로그)")
    tot = ss.audit.totals()
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("AI 호출", tot["calls"])
    a2.metric("입력 토큰", tot["input_tokens"])
    a3.metric("출력 토큰", tot["output_tokens"])
    a4.metric("외부로 보낸 센터 수치", tot["values_sent"])
    recs = [r for r in ss.audit.records if r.get("event") == "llm_call"]
    if recs:
        st.dataframe(pd.DataFrame([{k: r.get(k) for k in ("time", "purpose", "provider", "model", "level", "values_sent",
                                                           "masked_numbers", "terms_redacted", "ok", "input_tokens", "output_tokens")}
                                   for r in recs]), hide_index=True, width="stretch")
        with st.expander("마지막 호출에서 실제로 보낸 내용"):
            st.code(recs[-1].get("sent_user", "")[:6000], language=None, wrap_lines=True)
    if st.button("관리자 모드 나가기"):
        ss.admin_ok = False
        st.rerun()


def main():
    _init()
    st.markdown(CSS, unsafe_allow_html=True)
    sidebar()
    top_banner()
    t1, t2, t3, t4, t5 = st.tabs(["① 질문", "② 데이터", "③ 분석 결과", "④ 반출 신청", "관리자"])
    with t1:
        tab_question()
    with t2:
        tab_data()
    with t3:
        tab_result()
    with t4:
        tab_export()
    with t5:
        tab_admin()


main()
