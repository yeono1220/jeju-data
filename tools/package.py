"""제출용 ZIP 만들기(파일 이름 UTF-8 표시, 금지 파일 검사).

사용:
  python tools/package.py                              # 평가용 키 없이
  set EVAL_ANTHROPIC_KEY=sk-ant-...  &  python tools/package.py   # 평가용 키를 data\\.env에 넣어 패키징

금지: 센터 원본 CSV, DAKER 제공 허브 데이터목록 xlsx, 코드·문서 속 API 키(평가용 data\\.env 제외), 개인 설정(data\\settings.json).
"""
from __future__ import annotations

import os
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["README.md", "설치.bat", "실행.bat", "점검.bat", "install.bat", "run.bat", "check.bat", "requirements-lock.txt", "pyproject.toml",
           ".env.example", ".gitignore", ".streamlit/config.toml", "src", "tests", "tools", "docs"]
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".venv", ".git"}       # 맨 위 data\(키·세션)만 따로 제외 — spec\data 등은 포함
SKIP_SUFFIX = {".pyc", ".csv", ".xlsx", ".xls", ".parquet", ".duckdb", ".db"}
KEY_PAT = re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}|sk-[A-Za-z0-9]{32,}|AIza[0-9A-Za-z_\-]{30,}")
TEXT_SUFFIX = {".py", ".md", ".txt", ".toml", ".bat", ".json", ".html", ".css", ".example", ".cfg", ".ini", ""}


def files() -> list[Path]:
    out = []
    for item in INCLUDE:
        p = ROOT / item
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                rel = f.relative_to(ROOT)
                if f.is_file() and not (set(rel.parts) & SKIP_PARTS) and rel.parts[0] != "data" \
                        and f.suffix.lower() not in SKIP_SUFFIX and not f.name.startswith("."):
                    out.append(f)
    return out


def check(paths: list[Path]) -> list[str]:
    problems = []
    for f in paths:
        rel = f.relative_to(ROOT).as_posix()
        if "데이터목록" in f.name:
            problems.append(f"재배포 금지 파일: {rel}")
        if f.stat().st_size > 50 * 1024 * 1024:
            problems.append(f"50MB 넘는 파일: {rel}")
        if f.suffix.lower() in TEXT_SUFFIX:
            text = f.read_text(encoding="utf-8", errors="ignore")
            if KEY_PAT.search(text):
                problems.append(f"API 키로 보이는 문자열: {rel}")
    return problems


def main():
    paths = files()
    problems = check(paths)
    if problems:
        print("패키징 중단:\n  " + "\n  ".join(problems))
        sys.exit(1)
    key = os.environ.get("EVAL_ANTHROPIC_KEY", "").strip()
    out = ROOT.parent / f"데이터올레_최종결과물_{datetime.now():%Y%m%d}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in paths:
            z.write(f, Path("dataolle") / f.relative_to(ROOT))
        if key:
            ws = os.environ.get("EVAL_ANTHROPIC_WORKSPACE_ID", "").strip()
            z.writestr("dataolle/data/.env", "# 평가용 키(사용량 제한, 평가 후 폐기)\nANTHROPIC_API_KEY=" + key + "\n"
                       + (f"ANTHROPIC_WORKSPACE_ID={ws}\n" if ws else ""))
    print(f"완료: {out} ({out.stat().st_size / 1e6:.1f}MB, 파일 {len(paths)}개{', 평가용 키 포함' if key else ''})")


if __name__ == "__main__":
    main()
