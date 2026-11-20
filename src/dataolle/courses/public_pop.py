"""공개 실데이터: 행정안전부 주민등록 인구통계(읍면동별 총인구·65세 이상·0~14세).

센터 유동인구(통신사 추정) 대신 '고령 동네' 판정과 '주민 1인당 적립액'의 분모로 쓴다.
원본은 jumin.mois.go.kr의 연령별 인구현황(행정동, 5세 단위) 엑셀이고, tools/build_public_pop.py가
spec/data/public_pop.json으로 발췌한다. 다른 달 자료로 바꾸려면 엑셀을 새로 받아 그 도구를 다시 돌린다.
센터 데이터가 아니므로 반출 제한이 없고 외부 전송 가림 대상도 아니다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

PATH = Path(__file__).resolve().parents[1] / "spec" / "data" / "public_pop.json"
# 분석 단위(센터 행정동 묶음) → 주민등록 행정동. 이름이 같으면 그대로, 묶인 단위는 가운뎃점으로 나눈다.
MIN_COVER = 0.9


@dataclass
class PublicPop:
    table: pd.DataFrame          # index=분석 단위, 열 total·age65·child014·aging(%)
    ym: str
    source: str
    missing: list[str]

    @property
    def label(self) -> str:
        return f"행정안전부 주민등록 인구통계({self.ym[:4]}년 {int(self.ym[4:])}월)" if self.ym else "행정안전부 주민등록 인구통계"


@lru_cache(maxsize=1)
def _raw() -> dict | None:
    if not PATH.exists():
        return None
    try:
        return json.loads(PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def _parts(unit: str) -> list[str]:
    """'송산·정방·중앙·천지동' → ['송산동', '정방동', '중앙동', '천지동']"""
    if "·" not in unit:
        return [unit]
    suffix = unit[-1]
    return [p if p.endswith(suffix) else p + suffix for p in unit.split("·")]


def for_units(units) -> PublicPop | None:
    """분석 단위 목록에 맞춘 공식 인구. 90% 이상 맞아야 쓴다(일부 시만 들어 있으면 None → 센터 자료로 대신)."""
    raw = _raw()
    units = [u for u in dict.fromkeys(units) if isinstance(u, str) and u]
    if raw is None or not units:
        return None
    by = {u["name"]: u for u in raw.get("units", [])}
    rows, missing = {}, []
    for unit in units:
        parts = _parts(unit)
        if not all(p in by for p in parts):
            missing.append(unit)
            continue
        rows[unit] = {k: sum(by[p][k] for p in parts) for k in ("total", "age65", "child014")}
    if len(rows) < MIN_COVER * len(units):
        return None
    t = pd.DataFrame.from_dict(rows, orient="index").astype(float)
    t["aging"] = t["age65"] / t["total"] * 100
    return PublicPop(t, str(raw.get("ym", "")), f"{raw.get('provider', '')} {raw.get('title', '')}".strip(), missing)


__all__ = ["for_units", "PublicPop", "PATH"]
