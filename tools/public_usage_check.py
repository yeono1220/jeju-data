"""공개 실데이터(도청 누리집의 탐나는전 월별 사용액)로 '적립률 1%p당 사용액 반응'의 크기를 거칠게 재 본다.

사용: python tools/public_usage_check.py
월 합계뿐이라 업종별 분석(앞당긴 소비 걸러 내기)은 할 수 없다. 적립률이 바뀐 때마다 '바뀐 뒤 하루 평균 ÷ 비교 기간 하루 평균'을
구하고 1%p당으로 바꾼다. 비교 기간을 어떻게 잡느냐에 따라 값이 크게 달라진다는 것 자체가 결과다.
"""
from __future__ import annotations

import calendar
import json
import math
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "src" / "dataolle" / "spec" / "data" / "public_usage.json"

# (설명, 바뀐 뒤 달, 비교 달, 적립률 차이 %p, 주의)
EVENTS = (
    ("2024.6 적립 5%→7%", ["202406", "202407"], ["202403", "202404", "202405"], 2.0, "비교: 직전 석 달"),
    ("2024.6 적립 5%→7%", ["202406", "202407"], ["202401", "202402", "202403", "202404", "202405"], 2.0, "비교: 1~5월 전체"),
    ("2024 추석 7%→14%(9월 평균 12.1%)", ["202409"], ["202406", "202407"], 5.1, "추석이 낀 달"),
    ("2024.10 7%→10%", ["202410", "202411", "202412"], ["202406", "202407"], 3.0, "12월 12~22일 적립 중단 포함"),
    ("2026.2 10%→20%", ["202602"], ["202603", "202604"], 10.0, "비교: 바로 뒤 두 달, 설이 낀 달"),
    ("2026.2 10%→20%", ["202602"], ["202601"], 10.0, "비교: 바로 앞 달"),
)


def load() -> dict[str, float]:
    return {m["ym"]: float(m["used"]) for m in json.loads(DATA.read_text(encoding="utf-8"))["months"]}


def daily(used: dict[str, float], months: list[str]) -> float:
    days = sum(calendar.monthrange(int(m[:4]), int(m[4:]))[1] for m in months)
    return sum(used[m] for m in months) / days


def rows() -> list[dict]:
    used = load()
    out = []
    for name, after, before, dr, note in EVENTS:
        a, b = daily(used, after), daily(used, before)
        out.append({"event": name, "before": b, "after": a, "change_pct": (a / b - 1) * 100, "dr": dr,
                    "pct_per_pp": (math.exp(math.log(a / b) / dr) - 1) * 100, "note": note})
    return out


if __name__ == "__main__":
    rs = rows()
    for r in rs:
        print(f"{r['event']:<34} 하루 평균 {r['before']:7.0f} → {r['after']:7.0f}백만 원 ({r['change_pct']:+5.1f}%) "
              f"→ 1%p당 {r['pct_per_pp']:.1f}%   [{r['note']}]")
    v = [r["pct_per_pp"] for r in rs]
    print(f"범위: 1%p당 {min(v):.1f}% ~ {max(v):.1f}% (비교 기간에 따라 달라짐, 월 합계라 앞당긴 소비·명절을 못 가름)")
