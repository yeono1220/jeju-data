"""설치 점검 — 설치.bat 뒤에 한 번 실행합니다(점검.bat 더블클릭 또는 .venv\\Scripts\\python tools\\preflight.py).

하는 일
  ① Python 버전  ② 패키지가 고정 버전대로 깔렸는지  ③ 확장 모듈(.pyd·.dll)이 실제로 불러와지는지
  ④ 집계(DuckDB)·그림(matplotlib)·문서(python-docx) 한 바퀴  ⑤ 설치 폴더 쓰기 권한  ⑥ 8501 포트  ⑦ AI 키 유무
센터 PC처럼 서명 없는 확장 모듈 실행을 제한하는 환경에서는 ③이 '차단'으로 나옵니다.
그때는 결과 파일(data\\점검결과.txt)의 '허용 등록이 필요한 폴더와 파일'을 운영자에게 전달하면 됩니다.
외부로 보내는 것은 없습니다.
"""
from __future__ import annotations

import importlib
import os
import socket
import sys
import tempfile
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# 확장 모듈(.pyd)이 들어 있어 보안정책의 영향을 받는 패키지: (불러올 이름, 설명)
NATIVE = [("numpy", "수치 계산"), ("pandas", "표 처리"), ("duckdb", "집계(SQL)"), ("pyarrow", "표 전달"),
          ("matplotlib", "그림"), ("PIL", "그림 저장(pillow)"), ("kiwisolver", "그림 배치"), ("contourpy", "그림 보조"),
          ("pydantic_core", "입력 검증"), ("lxml", "문서(DOCX)"), ("rpds", "설정 검증"), ("markupsafe", "화면 보조"),
          ("google.protobuf", "화면 통신"), ("httptools", "화면 서버"), ("websockets", "화면 통신"),
          ("watchdog", "파일 감시"), ("charset_normalizer", "글자 인식")]
PURE = ["streamlit", "docx", "pypdf", "jinja2", "requests"]

lines: list[str] = []
bad: list[str] = []


def say(mark: str, text: str) -> None:
    line = f"[{mark}] {text}"
    lines.append(line)
    print(line)


def locked() -> dict[str, str]:
    out = {}
    p = ROOT / "requirements-lock.txt"
    if p.exists():
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln and not ln.startswith("#") and "==" in ln:
                k, v = ln.split("==", 1)
                out[k.lower().replace("_", "-")] = v
    return out


def native_files() -> tuple[Path | None, list[Path]]:
    site = next((Path(p) for p in sys.path if p.endswith("site-packages") and Path(p).exists()), None)
    if site is None:
        return None, []
    files = [f for ext in ("*.pyd", "*.dll", "*.so") for f in site.rglob(ext)]
    return site, sorted(files)


def main() -> int:
    say("정보", f"설치 폴더: {ROOT}")
    v = sys.version_info
    ok = (3, 12) <= (v.major, v.minor) <= (3, 14)
    say("통과" if ok else "실패", f"Python {v.major}.{v.minor}.{v.micro} ({'64' if sys.maxsize > 2**32 else '32'}비트) — 3.13 권장, 3.12~3.14 지원")
    if not ok:
        bad.append("Python 3.13을 설치한 뒤 설치.bat을 다시 실행하세요.")
    if sys.prefix == sys.base_prefix:
        say("주의", "프로그램 전용 환경(.venv)이 아닌 Python으로 실행했습니다. 점검.bat으로 실행하세요.")

    # ② 고정 버전
    want, diff, missing = locked(), [], []
    for name, ver in want.items():
        try:
            have = metadata.version(name)
            if have != ver:
                diff.append(f"{name} {have}(고정 {ver})")
        except metadata.PackageNotFoundError:
            missing.append(name)
    if missing:
        say("실패", f"깔리지 않은 패키지 {len(missing)}개: {', '.join(missing[:8])}")
        bad.append("설치.bat을 다시 실행하세요(인터넷에서 pypi.org 접속 필요).")
    else:
        say("통과", f"패키지 {len(want)}개 설치됨" + (f" — 버전이 다른 것: {', '.join(diff[:6])}" if diff else " (고정 버전과 같음)"))

    # ③ 확장 모듈
    blocked = []
    for mod, what in NATIVE:
        try:
            importlib.import_module(mod)
        except ModuleNotFoundError:
            continue                                  # 운영체제에 따라 없는 선택 패키지
        except Exception as e:                        # 보안정책 차단은 보통 ImportError: DLL load failed
            blocked.append(f"{mod}({what}): {type(e).__name__} {str(e)[:90]}")
    site, files = native_files()
    if blocked:
        say("차단", f"확장 모듈 {len(blocked)}개를 불러오지 못했습니다(보안정책이 서명 없는 .pyd·.dll 실행을 막았을 수 있음).")
        for b in blocked:
            say("  ·", b)
        bad.append("아래 '허용 등록이 필요한 폴더와 파일'을 센터 운영자(보안 담당)에게 전달해 허용 등록을 요청하세요.")
    else:
        say("통과", f"확장 모듈 불러오기 정상(.pyd·.dll {len(files)}개)")
    for mod in PURE:
        try:
            importlib.import_module(mod)
        except Exception as e:
            say("실패", f"{mod}: {type(e).__name__} {str(e)[:90]}")
            bad.append("설치.bat을 다시 실행하세요.")

    # ④ 한 바퀴
    if not blocked:
        try:
            import duckdb
            import pandas as pd
            n = duckdb.connect().execute("SELECT SUM(x) FROM (VALUES (1),(2),(3)) t(x)").fetchone()[0]
            assert n == 6 and pd.DataFrame({"a": [1, 2]})["a"].sum() == 3
            say("통과", "집계(DuckDB·pandas)")
        except Exception as e:
            say("실패", f"집계: {type(e).__name__} {str(e)[:90]}")
            bad.append("집계 엔진이 동작하지 않습니다. 이 결과 파일을 문의 때 함께 보내 주세요.")
        try:
            from dataolle.report.charts import setup_font
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            font = setup_font()
            with tempfile.TemporaryDirectory() as d:
                fig, ax = plt.subplots(figsize=(2, 1))
                ax.set_title("한글 시험")
                fig.savefig(Path(d) / "t.png")
                plt.close(fig)
            say("통과", f"그림(matplotlib), 한글 글꼴: {font or '기본 글꼴(한글이 네모로 보일 수 있음)'}")
        except Exception as e:
            say("실패", f"그림: {type(e).__name__} {str(e)[:90]}")
        try:
            import docx
            with tempfile.TemporaryDirectory() as d:
                doc = docx.Document()
                doc.add_paragraph("시험")
                doc.save(Path(d) / "t.docx")
            say("통과", "문서(DOCX)")
        except Exception as e:
            say("실패", f"문서: {type(e).__name__} {str(e)[:90]}")

    # ⑤ 쓰기 권한 ⑥ 포트 ⑦ PDF 변환기·AI 키
    data = ROOT / "data"
    try:
        data.mkdir(exist_ok=True)
        t = data / "_write_test.tmp"
        t.write_text("ok", encoding="utf-8")
        t.unlink()
        say("통과", f"쓰기 권한: {data}")
    except Exception as e:
        say("실패", f"쓰기 권한 없음: {data} ({e})")
        bad.append("압축을 바탕화면이나 C:\\dataolle처럼 쓸 수 있는 폴더에 다시 풀어 주세요.")
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", 8501))
        say("통과", "8501 포트 사용 가능")
    except OSError:
        say("주의", "8501 포트를 다른 프로그램이 쓰고 있습니다(데이터올레가 이미 켜져 있으면 정상). 설치 매뉴얼 9절 참고.")
    finally:
        s.close()
    edge = [p for p in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
                        r"C:\Program Files\Google\Chrome\Application\chrome.exe") if Path(p).exists()]
    if os.name == "nt":
        say("통과" if edge else "주의", "PDF 변환용 브라우저: " + (edge[0] if edge else "Edge·Chrome을 찾지 못함(보고서는 미리보기에서 Ctrl+P로 저장)"))
    env = data / ".env"
    key = ""
    if env.exists():
        for ln in env.read_text(encoding="utf-8", errors="ignore").splitlines():
            if ln.strip().startswith("ANTHROPIC_API_KEY="):
                key = ln.split("=", 1)[1].strip()
    say("정보", f"AI 키({env}): " + ("있음 — 실행 후 관리자 탭의 '연결 테스트'로 확인" if key else "없음 — AI 없이 규칙 기반으로 모든 기능이 동작"))

    # 허용 등록 안내
    lines.append("")
    lines.append("허용 등록이 필요한 폴더와 파일(서명 없는 오픈소스 확장 모듈) — 위에 [차단]이 있을 때만 필요")
    lines.append(f"  실행 파일: {Path(sys.executable)}")
    if site is not None:
        lines.append(f"  폴더(하위 포함): {site}")
        tops: dict[str, int] = {}
        for f in files:
            top = f.relative_to(site).parts[0]
            tops[top] = tops.get(top, 0) + 1
        lines.append("  패키지별 확장 모듈 수: " + ", ".join(f"{k} {n}" for k, n in sorted(tops.items())))
        lines.append(f"  파일 전체 목록({len(files)}개):")
        lines.extend(f"    {f}" for f in files)

    print()
    if bad:
        print("결과: 조치가 필요합니다.")
        for b in dict.fromkeys(bad):
            print("  →", b)
    else:
        print("결과: 정상입니다. 실행.bat을 더블클릭하세요.")
    lines.insert(0, "데이터올레 설치 점검 결과 — " + ("조치 필요" if bad else "정상"))
    try:
        out = data / "점검결과.txt"
        out.write_text("\n".join(lines + ["", "조치:"] + [f"  → {b}" for b in dict.fromkeys(bad)]), encoding="utf-8")
        print(f"자세한 결과(확장 모듈 목록 포함): {out}")
    except Exception:
        pass
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
