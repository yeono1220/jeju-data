"""센터 CSV 자동 인식기.

센터에서 내려받은 파일은 이름이 전부 `Results_YYYYMMDD_HHMMSS.csv`라서 이름만으로는
무슨 데이터인지 알 수 없다. 헤더(영문 컬럼명)를 명세서 v13의 컬럼 구성과 대조해
데이터셋을 판별하고, 기간 컬럼에서 수록 월을 읽어 낸다.

센터 CSV 형식(게시판 공식 답변): 헤더=명세서 영문 컬럼명, 쉼표 구분, 모든 값 큰따옴표,
결측 `NA`, 파티션 컬럼(pt_year/pt_month) 미포함, UTF-8(BOM 없음).
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from dataolle.spec import PARTITION_COLS, SpecRegistry, default_registry

CENTER_NAME_RE = re.compile(r"^Results_(\d{8})_(\d{6})(?:\s*\(\d+\))?\.csv$", re.IGNORECASE)

# 헤더가 완전히 같은 데이터셋 쌍 → 값으로 구별
AMBIGUOUS_GROUPS = [
    {"1-02", "1-05"},   # 관광업종 인당 금액: 내국인 vs 중국인
    {"1-03", "1-06"},   # 관광테마 인당 금액: 내국인 vs 중국인
    {"2-01", "2-02"},   # 거래처정보: 구매 vs 판매
    {"5-05", "5-06"},   # 음식물쓰레기: 제주시 vs 서귀포
]

SAMPLE_LINES = 200


@dataclass
class Fingerprint:
    path: Path
    encoding: str
    has_bom: bool
    header: list[str]
    sample_rows: list[list[str]]
    all_quoted: bool | None        # 데이터 행의 모든 값이 큰따옴표로 감싸져 있는지
    na_token_seen: bool
    center_filename: bool          # Results_YYYYMMDD_HHMMSS.csv 형식인지
    downloaded_at: datetime | None


@dataclass
class Candidate:
    dataset_id: str
    name_ko: str
    score: float
    exact: bool
    missing: list[str] = field(default_factory=list)   # 명세서엔 있는데 파일에 없는 컬럼
    extra: list[str] = field(default_factory=list)     # 파일엔 있는데 명세서에 없는 컬럼


@dataclass
class Recognition:
    fingerprint: Fingerprint
    status: str                    # ok | ambiguous | unknown
    best: Candidate | None
    candidates: list[Candidate]
    period_column: str | None = None
    months: list[str] = field(default_factory=list)      # YYYYMM
    month_rows: dict[str, int] = field(default_factory=dict)
    row_count: int | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def path(self) -> Path:
        return self.fingerprint.path

    @property
    def dataset_id(self) -> str | None:
        return self.best.dataset_id if self.best and self.status == "ok" else None

    def summary(self) -> str:
        if self.status == "unknown" or not self.best:
            return f"{self.path.name}: 알 수 없는 형식"
        months = f"{self.months[0]}~{self.months[-1]}" if len(self.months) > 1 else (self.months[0] if self.months else "기간 미상")
        flag = "" if self.status == "ok" else " (확인 필요)"
        return f"{self.path.name}: {self.best.dataset_id} {self.best.name_ko} · {months}{flag}"


# ---------------------------------------------------------------------------
# 1) 파일 지문 읽기
# ---------------------------------------------------------------------------
def _read_head_bytes(path: Path, limit: int = 256 * 1024) -> bytes:
    with open(path, "rb") as f:
        return f.read(limit)


def _decode(raw: bytes) -> tuple[str, str, bool]:
    has_bom = raw.startswith(b"\xef\xbb\xbf")
    for enc in ("utf-8-sig", "cp949"):
        try:
            text = raw.decode(enc)
            return text, ("utf-8" if enc == "utf-8-sig" else enc), has_bom
        except UnicodeDecodeError:
            # 버퍼 끝에서 멀티바이트 문자가 잘린 경우를 위해 마지막 몇 바이트를 버리고 재시도
            for cut in (1, 2, 3):
                try:
                    text = raw[:-cut].decode(enc)
                    return text, ("utf-8" if enc == "utf-8-sig" else enc), has_bom
                except UnicodeDecodeError:
                    continue
    return raw.decode("utf-8", errors="replace"), "unknown", has_bom


def _fields_all_quoted(line: str) -> bool:
    """한 줄의 모든 필드가 큰따옴표로 감싸져 있는지(따옴표 안 쉼표 고려)."""
    if not line.strip():
        return True
    # 따옴표로 감싼 필드만으로 이루어진 줄인지 정규식으로 확인
    return re.fullmatch(r'\s*"(?:[^"]|"")*"(?:\s*,\s*"(?:[^"]|"")*")*\s*', line) is not None


def fingerprint(path: str | Path) -> Fingerprint:
    path = Path(path)
    raw = _read_head_bytes(path)
    text, encoding, has_bom = _decode(raw)
    lines = text.splitlines()
    if len(raw) >= 256 * 1024 and lines:
        lines = lines[:-1]  # 마지막 줄은 잘렸을 수 있음
    reader = csv.reader(io.StringIO("\n".join(lines[: SAMPLE_LINES + 1])))
    rows = [r for r in reader]
    header = [h.strip().strip("﻿").lower() for h in rows[0]] if rows else []
    data = rows[1:]
    data_lines = lines[1: SAMPLE_LINES + 1]
    all_quoted = all(_fields_all_quoted(l) for l in data_lines) if data_lines else None
    na_seen = any("NA" in r for r in data)
    m = CENTER_NAME_RE.match(path.name)
    downloaded_at = None
    if m:
        try:
            downloaded_at = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        except ValueError:
            downloaded_at = None
    return Fingerprint(path, encoding, has_bom, header, data, all_quoted, na_seen, bool(m), downloaded_at)


# ---------------------------------------------------------------------------
# 2) 헤더 대조
# ---------------------------------------------------------------------------
def match_header(header: list[str], registry: SpecRegistry | None = None, top: int = 3) -> list[Candidate]:
    registry = registry or default_registry()
    h = [c for c in header if c]
    hs = set(h)
    out: list[Candidate] = []
    for spec in registry:
        csv_cols = spec.csv_columns
        s_csv, s_all = set(csv_cols), set(spec.columns)
        if hs == s_csv or hs == s_all:
            # 순서까지 같으면 가산점
            order_bonus = 0.001 if [c for c in h if c not in PARTITION_COLS] == list(csv_cols) else 0.0
            out.append(Candidate(spec.id, spec.name_ko, 1.0 + order_bonus, True))
            continue
        inter = len(hs & s_all)
        if not inter:
            continue
        coverage = inter / len(hs)                       # 파일 컬럼 중 명세서에 있는 비율
        recall = len(hs & s_csv) / max(1, len(s_csv))    # 명세서(CSV) 컬럼 중 파일에 있는 비율
        if coverage < 0.8:
            continue
        score = 0.99 * (0.6 * coverage + 0.4 * recall)   # 완전 일치(1.0)보다 항상 낮게
        out.append(Candidate(
            spec.id, spec.name_ko, round(score, 4), False,
            missing=[c for c in csv_cols if c not in hs],
            extra=[c for c in h if c not in s_all],
        ))
    out.sort(key=lambda c: (-c.score, c.dataset_id))
    return out[:top]


def _disambiguate(cands: list[Candidate], fp: Fingerprint) -> tuple[Candidate | None, str | None]:
    """헤더가 같은 데이터셋 쌍을 값으로 구별. 구별 못 하면 (None, 사유)."""
    ids = {c.dataset_id for c in cands}
    by_id = {c.dataset_id: c for c in cands}
    col = {name: i for i, name in enumerate(fp.header)}

    def values(column: str) -> set[str]:
        i = col.get(column)
        return {r[i] for r in fp.sample_rows if i is not None and i < len(r)}

    if ids <= {"1-02", "1-05"} or ids <= {"1-03", "1-06"}:
        vals = values("user_type")
        chinese = any("중국" in v for v in vals)
        pick = ("1-05" if chinese else "1-02") if "1-02" in ids or "1-05" in ids else ("1-06" if chinese else "1-03")
        if pick in by_id:
            return by_id[pick], None
    if ids <= {"2-01", "2-02"}:
        vals = " ".join(values("sale_buy_code"))
        if "구매" in vals and "판매" not in vals:
            return by_id.get("2-01"), None
        if "판매" in vals and "구매" not in vals:
            return by_id.get("2-02"), None
        return None, "구매/판매 거래처 파일은 헤더가 같아 자동 구별이 어려움 → 사용자 확인 필요"
    if ids <= {"5-05", "5-06"}:
        return None, "제주시/서귀포 음식물쓰레기 파일은 헤더가 같아 자동 구별이 어려움 → 사용자 확인 필요"
    return None, "후보가 여러 개"


# ---------------------------------------------------------------------------
# 3) 기간(수록 월) 읽기
# ---------------------------------------------------------------------------
def scan_periods(path: Path, period_column: str, encoding: str = "utf-8") -> tuple[dict[str, int], int]:
    """파일 전체를 훑어 월(YYYYMM)별 행 수를 센다. DuckDB가 있으면 DuckDB로 빠르게."""
    try:
        import duckdb  # noqa: WPS433

        con = duckdb.connect()
        enc_opt = "" if encoding in ("utf-8", "unknown") else f", encoding='{encoding}'"
        q = f"""
            SELECT substr(CAST("{period_column}" AS VARCHAR), 1, 6) AS ym, count(*) AS n
            FROM read_csv('{_sql_path(path)}', header=true, all_varchar=true, quote='"',
                          nullstr='NA'{enc_opt})
            GROUP BY 1 ORDER BY 1
        """
        rows = con.execute(q).fetchall()
        con.close()
        months = {str(ym): int(n) for ym, n in rows if ym}
        return months, sum(months.values())
    except Exception:
        import pandas as pd  # noqa: WPS433

        s = pd.read_csv(path, usecols=[period_column], dtype=str, encoding=("utf-8" if encoding == "unknown" else encoding))
        ym = s[period_column].dropna().str.slice(0, 6)
        counts = ym.value_counts().sort_index()
        return {k: int(v) for k, v in counts.items()}, int(len(s))


def _sql_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


# ---------------------------------------------------------------------------
# 4) 종합
# ---------------------------------------------------------------------------
def recognize(path: str | Path, registry: SpecRegistry | None = None, scan: bool = True) -> Recognition:
    registry = registry or default_registry()
    fp = fingerprint(path)
    warnings: list[str] = []

    if fp.encoding == "cp949":
        warnings.append("인코딩이 CP949(EUC-KR)입니다. 엑셀로 열었다 저장한 파일일 수 있어요 — 코드 앞자리 0(예: 연령 0009)이 사라졌는지 확인하세요.")
    if fp.has_bom:
        warnings.append("UTF-8 BOM이 있습니다(센터 원본은 BOM 없음). 다른 프로그램으로 다시 저장된 파일일 수 있어요.")
    if fp.all_quoted is False:
        warnings.append("값이 큰따옴표로 감싸져 있지 않습니다(센터 원본은 모든 값에 따옴표). 엑셀 등으로 재저장된 파일일 수 있어요.")

    cands = match_header(fp.header, registry)
    if not cands:
        return Recognition(fp, "unknown", None, [], warnings=warnings + ["명세서의 어떤 데이터셋과도 헤더가 맞지 않습니다."])

    top = cands[0]
    ties = [c for c in cands if abs(c.score - top.score) < 1e-6]
    status, best = "ok", top
    if len(ties) > 1:
        pick, reason = _disambiguate(ties, fp)
        if pick:
            best = pick
        else:
            status = "ambiguous"
            warnings.append(reason or "후보가 여러 개입니다.")
    elif not top.exact:
        if top.missing:
            warnings.append(f"명세서 컬럼 일부가 없습니다: {', '.join(top.missing)} (조회 시 컬럼을 골랐을 수 있음)")
        if top.extra:
            warnings.append(f"명세서에 없는 컬럼이 있습니다: {', '.join(top.extra)}")
        if top.score < 0.9:
            status = "ambiguous"

    rec = Recognition(fp, status, best, cands, warnings=warnings)
    spec = registry.get(best.dataset_id)
    rec.period_column = spec.period_column if spec.period_column in fp.header else None
    if scan and rec.period_column:
        months, total = scan_periods(fp.path, rec.period_column, fp.encoding)
        rec.month_rows = months
        rec.months = sorted(months)
        rec.row_count = total
        if len(rec.months) > 1 and fp.center_filename:
            warnings.append(f"한 파일에 {len(rec.months)}개월이 들어 있습니다(센터 조회는 1개월 단위). 운영자가 합쳐 준 파일인지 확인하세요.")
    return rec
