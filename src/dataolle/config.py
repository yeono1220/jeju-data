"""설정·작업 폴더·관리자 PIN.

모든 파일은 프로그램 폴더 안 data\\ 아래에만 만든다(센터 PC의 다른 곳을 건드리지 않음).
  data\\settings.json   AI 공급자·모델·전송 레벨·결과 자동 삭제(ON/OFF)·관리자 PIN(해시)
  data\\.env            API 키(평가용 키는 매뉴얼 안내에 따라 여기에 넣음, 평가 후 삭제)
  data\\sessions\\<id>\\  세션별 결과물(그림·보고서·반출 묶음·감사 로그) → '작업 끝내기'로 통째 삭제
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path


def app_home() -> Path:
    env = os.environ.get("DATAOLLE_HOME")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for parent in here.parents:                       # 설치 폴더(pyproject.toml 또는 실행.bat 있는 곳)
        if (parent / "실행.bat").exists() or (parent / "pyproject.toml").exists():
            return parent / "data"
    return Path.home() / "dataolle_data"


def default_data_folder() -> str:
    """센터 운영자가 미리 내려받아 두는 바탕화면 '요청파일' 폴더."""
    for base in (Path.home() / "Desktop", Path.home() / "바탕 화면", Path.home() / "OneDrive" / "바탕 화면",
                 Path.home() / "OneDrive" / "Desktop"):
        if (base / "요청파일").exists():
            return str(base / "요청파일")
    return str(Path.home() / "Desktop" / "요청파일")


@dataclass
class Settings:
    provider: str = "anthropic"          # anthropic | openai | gemini | ollama | none
    model: str = ""                      # 비우면 공급자 기본 모델
    level: int = 0                       # 외부 AI 전송 레벨(0 권장)
    auto_delete: bool = True             # 결과 자동 삭제 ON/OFF
    keep_hours: int = 12                 # 자동 삭제 ON일 때, 이 시간보다 오래된 세션 폴더는 시작 시 삭제
    data_folder: str = field(default_factory=default_data_folder)
    pin_hash: str = ""
    pin_salt: str = ""
    sdc_level: str = "conservative"      # literal | conservative | strict

    @classmethod
    def load(cls, home: Path | None = None) -> "Settings":
        p = (home or app_home()) / "settings.json"
        if p.exists():
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
                return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})
            except Exception:
                pass
        return cls()

    def save(self, home: Path | None = None) -> None:
        h = home or app_home()
        h.mkdir(parents=True, exist_ok=True)
        (h / "settings.json").write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    # -- 관리자 PIN -------------------------------------------------------------
    def set_pin(self, pin: str) -> None:
        self.pin_salt = secrets.token_hex(16)
        self.pin_hash = _hash_pin(pin, self.pin_salt)

    def check_pin(self, pin: str) -> bool:
        return bool(self.pin_hash) and secrets.compare_digest(self.pin_hash, _hash_pin(pin, self.pin_salt))

    @property
    def has_pin(self) -> bool:
        return bool(self.pin_hash)


def _hash_pin(pin: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", pin.encode("utf-8"), bytes.fromhex(salt), 200_000).hex()


# ---------------------------------------------------------------------------
# .env (API 키)
# ---------------------------------------------------------------------------
def load_env(home: Path | None = None) -> dict:
    p = (home or app_home()) / ".env"
    out = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    for k, v in out.items():
        if v and not os.environ.get(k):
            os.environ[k] = v
    return out


def save_env_key(name: str, value: str, home: Path | None = None) -> None:
    h = home or app_home()
    h.mkdir(parents=True, exist_ok=True)
    p = h / ".env"
    lines = p.read_text(encoding="utf-8").splitlines() if p.exists() else []
    lines = [l for l in lines if not l.strip().startswith(f"{name}=")]
    if value:
        lines.append(f"{name}={value}")
    p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    if value:
        os.environ[name] = value
    else:
        os.environ.pop(name, None)


def mask_key(v: str | None) -> str:
    if not v:
        return "(없음)"
    return v[:6] + "…" + v[-4:] if len(v) > 12 else "설정됨"


# ---------------------------------------------------------------------------
# 세션 작업 폴더
# ---------------------------------------------------------------------------
@dataclass
class Workspace:
    root: Path
    id: str

    @property
    def path(self) -> Path:
        return self.root / self.id

    @classmethod
    def create(cls, home: Path | None = None) -> "Workspace":
        root = (home or app_home()) / "sessions"
        sid = datetime.now().strftime("%Y%m%d_%H%M%S_") + secrets.token_hex(3)
        ws = cls(root, sid)
        ws.path.mkdir(parents=True, exist_ok=True)
        (ws.path / ".session").write_text(datetime.now().isoformat(timespec="seconds"), encoding="utf-8")
        return ws

    def sub(self, name: str) -> Path:
        p = self.path / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    def stats(self) -> tuple[int, int]:
        files = [f for f in self.path.rglob("*") if f.is_file()] if self.path.exists() else []
        return len(files), sum(f.stat().st_size for f in files)

    def delete(self) -> tuple[int, int]:
        """세션 폴더 통째 삭제. (삭제한 파일 수, 바이트) 반환."""
        n, size = self.stats()
        if self.path.exists():
            shutil.rmtree(self.path, ignore_errors=True)
        return n, size


def list_sessions(home: Path | None = None) -> list[dict]:
    root = (home or app_home()) / "sessions"
    out = []
    if root.exists():
        for d in sorted(root.iterdir()):
            if d.is_dir():
                files = [f for f in d.rglob("*") if f.is_file()]
                out.append({"id": d.name, "files": len(files), "MB": round(sum(f.stat().st_size for f in files) / 1e6, 1),
                            "modified": datetime.fromtimestamp(d.stat().st_mtime).strftime("%Y-%m-%d %H:%M")})
    return out


def cleanup_sessions(home: Path | None = None, older_than_hours: float = 12, keep: str | None = None) -> int:
    """자동 삭제 ON일 때 시작 시 호출: 오래된 세션 폴더(창을 닫고 떠난 이용자의 결과)를 지운다."""
    root = (home or app_home()) / "sessions"
    n = 0
    if not root.exists():
        return 0
    limit = time.time() - older_than_hours * 3600
    for d in root.iterdir():
        if d.is_dir() and d.name != keep and d.stat().st_mtime < limit:
            shutil.rmtree(d, ignore_errors=True)
            n += 1
    return n
