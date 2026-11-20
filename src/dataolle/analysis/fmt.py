"""숫자를 보고서용 한국어 표기로."""
from __future__ import annotations

import math


def fmt_value(v, unit: str, pct: bool = False) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "x"
    if unit == "p":
        return f"{v:.3f}"
    if pct or unit == "%":
        return f"{v:,.1f}%"
    if unit == "%p":
        return f"{v:+,.1f}%p"
    if unit == "원":
        a = abs(v)
        if a >= 1e8:
            return f"{v / 1e8:,.1f}억 원"
        if a >= 1e4 and a >= 1e6:
            return f"{v / 1e4:,.0f}만 원"
        return f"{v:,.0f}원"
    if unit == "회(정수)":
        return f"{v:,.0f}회"
    if unit in ("회", "배"):
        return f"{v:,.1f}{unit}"
    if unit == "지수":
        return f"{v:,.1f}"
    if unit == "위":
        return f"{int(v)}위"
    if unit in ("명", "건", "개"):
        if abs(v) >= 100 or float(v).is_integer():
            return f"{v:,.0f}{unit}"
        return f"{v:,.1f}{unit}"
    if unit == "시각":                       # 분(다음 날은 1440 이상) → '21:40', '다음 날 09:15'
        m = int(round(v))
        day = "다음 날 " if m >= 1440 else ""
        m %= 1440
        return f"{day}{m // 60:02d}:{m % 60:02d}"
    if unit == "시":
        return f"{int(round(v))}시"
    if unit == "편" and 0 < abs(v) < 1:          # 시뮬레이션 평균(기대값)이 1편이 안 될 때
        return "1편 미만"
    if unit in ("편", "석", "곳", "일"):
        if abs(v) >= 10 or abs(v - round(v)) < 0.05:
            return f"{v:,.0f}{unit}"
        return f"{v:,.1f}{unit}"
    return f"{v:,.1f}{unit}" if not float(v).is_integer() else f"{v:,.0f}{unit}"


def fmt_month(ym: str) -> str:
    ym = str(ym)
    if len(ym) >= 6 and ym[:6].isdigit():
        return f"{ym[:4]}년 {int(ym[4:6])}월"
    return ym


def fmt_period(start: str | None, end: str | None) -> str:
    if not start:
        return ""
    s, e = str(start)[:6], str(end or start)[:6]
    if s == e:
        return f"{s[:4]}.{s[4:6]}"
    return f"{s[:4]}.{s[4:6]}~{e[:4]}.{e[4:6]}"
