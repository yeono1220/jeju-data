"""탐나는전 적립률 이력 — 반응도 추정(자연실험)의 기준표.

보도로 확인한 구간만 '확인', 나머지는 '가정'으로 표시한다. 실제 운영에서는 담당 부서의 공식 이력으로 바꿔 넣는다
(설치 폴더의 data/rates.csv가 있으면 그 값이 우선: 열 ym,rate,limit,source).

2023~2024년(2026-11-19 기사 원문으로 확인)
  · 2023년: 현장 할인 5~10%(적립이 아님) — 경향신문 2024.5.28. 방식이 달라 이력에서 뺀다.
  · 2024년 1~5월: 적립 5%(연매출 5억 이하)·3%(5억 초과~10억 이하), 6월 1일부터 7% — 경향신문·제주의소리 2024.5.28.
  · 2024년 8.26~9.22: 14%, 월 한도 70만→100만 원 — 한국일보·스트레이트뉴스 2024.8.23.
    도 발표: 행사 기간 결제액 325억 원, 7% 기간의 1.7배 이상 — 헤드라인제주 2024.9.29.
  · 2024년 10월 1일~예산 소진 시: 10%(한도 70만 원) — 헤드라인제주 2024.9.29.
    12월 12일 예산(연 154억 원) 소진으로 중단, 예비비 10억 원으로 12월 23일 0시 재개 — 제주도 보도자료 2024.12.6·12.20.
    이 표는 12월도 10%로 둔다(중단 11일을 날수 평균하면 더 낮다 → '가정'으로 표시, 공식 이력으로 교체 가능).
  · 2025년 1월: 10% 유지, 1월 31일에 3월 말까지 연장 — 제주도 보도자료 2024.12.20·2025.1.31.
    같은 보도자료의 도 발표: 2024년 10~12월 10% 때 평시 7% 대비 월 결제액 +15.7%.
  달 중간에 바뀐 달(2024년 8·9월)은 날수로 평균한 값을 쓴다.

확인한 출처
  · 한라일보 2025.3.12: 당초 적립률 7% → 2025년 1~3월 한시 10%, 4~6월 15%(월 한도 70만→200만 원).
    도 발표: 7%→10% 때 월평균 사용액 +21.6%, 2024년 추석 연휴 7%→14%(한도 100만 원) 때 +39.2%.
  · 경향신문 2025.8.31: 2025년 9월 1일부터 '현행 10%'→13%(월 한도 70만 원) → 7~8월은 10%.
  · 제이누리 2026.3.3·7.13, 뉴스서울 2026.3.15, 파이낸셜뉴스 2026.9.4: 2026년 1월 10%, 2월 20%, 7.13 예산 소진 중단,
    9.5~10.31 10%(한도 100만 원).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

BASE_BEFORE_2025 = 7.0      # 2024년 6~8월의 기본 적립률


@dataclass(frozen=True)
class RateSpan:
    first: str            # YYYYMM
    last: str             # YYYYMM
    rate: float           # %
    limit: int            # 월 적립 대상 결제 한도(만 원)
    source: str
    confirmed: bool = True


SPANS: tuple[RateSpan, ...] = (
    # 2023년은 '현장 할인 5~10%'(결제할 때 바로 깎아 줌)라 적립과 방식이 다르다 → 이력에 넣지 않는다(추정에서 제외).
    RateSpan("202401", "202405", 5.0, 70, "경향신문·제주의소리 2024.5.28(5억 이하 5%, 5억 초과~10억 이하 3% → 대표값 5%)"),
    RateSpan("202406", "202407", 7.0, 70, "경향신문·제주의소리 2024.5.28(6월 1일부터 7%)"),
    RateSpan("202408", "202408", 8.4, 70, "한국일보·스트레이트뉴스 2024.8.23(8.26부터 14% → 25일 7%·6일 14%의 날수 평균)"),
    RateSpan("202409", "202409", 12.1, 100, "헤드라인제주 2024.9.29(9.22까지 14% → 22일 14%·8일 7%의 날수 평균)"),
    RateSpan("202410", "202412", 10.0, 70, "헤드라인제주 2024.9.29, 제주도 보도자료 2024.12.6·12.20(12월 12~22일은 예산 소진으로 중단 — 12월도 10%로 둔 근사)", confirmed=False),
    RateSpan("202501", "202503", 10.0, 70, "제주도 보도자료 2024.12.20·2025.1.31, 한라일보 2025.3.12"),
    RateSpan("202504", "202506", 15.0, 200, "한라일보 2025.3.12"),
    RateSpan("202507", "202508", 10.0, 70, "경향신문 2025.8.31('현행 10%')"),
    RateSpan("202509", "202512", 13.0, 70, "경향신문 2025.8.31"),
    RateSpan("202601", "202601", 10.0, 70, "뉴스서울 2026.3.15"),
    RateSpan("202602", "202602", 20.0, 70, "제이누리 2026.3.3"),
    RateSpan("202603", "202606", 10.0, 70, "제이누리 2026.7.13", confirmed=False),
    RateSpan("202609", "202610", 10.0, 100, "파이낸셜뉴스 2026.9.4"),
)
# 적립이 아닌 다른 방식이던 기간(표에는 보여 주되 추정에는 쓰지 않는다)
OTHER_METHOD = (("202301", "202312", "현장 할인 5~10%(결제할 때 바로 깎아 줌) — 경향신문 2024.5.28. 방식이 달라 추정에서 제외"),)
# 도가 발표한 과거 반응(외부 근거). (이전 적립률, 이후 적립률, 월평균 사용액 변화 %, 출처)
OFFICIAL_RESPONSES = ((7.0, 10.0, 21.6, "제주도 발표(한라일보 2025.3.12)"),
                      (7.0, 14.0, 39.2, "제주도 발표, 2024년 추석 연휴·한도 100만 원(한라일보 2025.3.12)"))


_CACHE: dict = {}


def _load_override() -> dict[str, tuple[float, int, str]]:
    """설치 폴더 data/rates.csv(있으면). 파일 수정 시각이 같으면 다시 읽지 않는다."""
    from dataolle.config import app_home
    p = Path(app_home()) / "rates.csv"
    key = (str(p), p.stat().st_mtime if p.exists() else None)
    if _CACHE.get("key") == key:
        return _CACHE["val"]
    out: dict[str, tuple[float, int, str]] = {}
    if p.exists():
        with open(p, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                try:
                    out[str(row["ym"]).strip()[:6]] = (float(row["rate"]), int(float(row.get("limit") or 70)),
                                                      row.get("source") or "data/rates.csv")
                except (KeyError, ValueError):
                    continue
    _CACHE["key"], _CACHE["val"] = key, out
    return out


def rate_of(ym: str) -> float | None:
    """그 달의 기본 적립률(%). 이력 밖이면 None."""
    info = span_of(ym)
    return info[0] if info else None


def span_of(ym: str) -> tuple[float, int, str, bool] | None:
    ym = str(ym)[:6]
    try:
        ov = _load_override()
    except Exception:                       # 설정 폴더가 없을 때(테스트·도구)
        ov = {}
    if ym in ov:
        r, lim, src = ov[ym]
        return r, lim, src, True
    for s in SPANS:
        if s.first <= ym <= s.last:
            return s.rate, s.limit, s.source, s.confirmed
    return None


def history_rows(months: list[str]) -> list[dict]:
    rows = []
    for ym in months:
        sp = span_of(ym)
        other = next((t for a, b, t in OTHER_METHOD if a <= ym <= b), None)
        rows.append({"연월": f"{ym[:4]}.{ym[4:]}", "적립률": (f"{sp[0]:g}%" if sp else ("적립 아님" if other else "모름")),
                     "월 한도": (f"{sp[1]}만 원" if sp else "-"),
                     "근거": (sp[2] + ("" if sp[3] else " — 가정") if sp else (other or "이력 없음"))})
    return rows


__all__ = ["rate_of", "span_of", "SPANS", "OFFICIAL_RESPONSES", "history_rows", "BASE_BEFORE_2025"]
