"""가상 탐나는전(1-32)·일일 날씨(5-02) 생성기 — 시연·검증용, 실제 데이터 아님.

가정(실제 분포를 흉내 낸 것):
  - 탐나는전은 주로 도민이 쓴다 → 지역 규모(SIZE)에 비례, 관광 계절성은 약함.
  - 요일: 금·토 높고 일요일 낮음(학원·병원 휴무), 설 연휴 직전 농축수산·전통시장 급증, 연휴 당일 감소.
  - 비(일강수량 5mm 이상) 오는 날: 전통시장 −22%, 의류·잡화 −8%, 그 밖 −1~3%.
  - 오일장: 장날(끝자리 날짜)에 시장 매출이 크게 늘고 나머지 날은 상설 점포만.
  - 검증용 가상 효과: 2025년 6월부터 전통시장 사용액 +12%(가상 시나리오 '전통시장 추가 할인').
  - 검증용 '정답' 반응(에이전트가 찾아내야 하는 값): 적립률 1%p당 사용액 ×exp(0.045)(약 +4.6%), 기준은 7%.
    적립률은 courses/rates.py의 보도 이력(2024.1~5 5%, 6월~ 7%, 추석 14%, 10월~ 10%, 2025.4~6 15%, 7~8 10%, 9월 13%)을 따른다(2023년은 7%로 둠 — 현장 할인 시기라 추정에는 쓰이지 않는다).
  - 학원 선결제(앞당긴 소비): 15% 기간(2025.4~6)에 학원 사용액 +40%·건당 금액 ×1.6(몇 달치 미리 결제),
    바로 뒤 7~8월은 −60%(미리 낸 만큼 빔). 순수 반응은 다른 업종과 같다.
"""
from __future__ import annotations

import calendar
from datetime import date, timedelta

import numpy as np
import pandas as pd

from dataolle.synth import world as W

# (업종 코드, 업종명, 표준업종변환, 비중, 건당 금액, 비 민감도)
TAMNA_BIZ = [
    ("TB01", "일반음식점", "한식 음식점업", 0.21, 19000, 0.97),
    ("TB02", "휴게음식점", "비알콜 음료점업", 0.07, 7500, 0.99),
    ("TB03", "슈퍼마켓·마트", "슈퍼마켓", 0.14, 26000, 0.98),
    ("TB04", "편의점", "체인화 편의점", 0.05, 9000, 1.0),
    ("TB05", "농축수산물", "과실 및 채소 소매업", 0.06, 31000, 0.92),
    ("TB06", "학원", "일반 교습 학원", 0.13, 175000, 1.0),
    ("TB07", "병원·의원", "일반 의원", 0.07, 24000, 1.0),
    ("TB08", "약국", "의약품 및 의료용품 소매업", 0.04, 14000, 1.0),
    ("TB09", "미용", "두발 미용업", 0.03, 32000, 0.98),
    ("TB10", "주유소", "차량용 주유소 운영업", 0.08, 52000, 0.99),
    ("TB11", "의류·잡화", "남녀용 겉옷 및 셔츠 소매업", 0.04, 41000, 0.92),
    ("TB12", "생활서비스", "세탁업", 0.03, 21000, 1.0),
    ("TB13", "숙박", "여관업", 0.02, 68000, 1.0),
]
ACQUIRERS = [("01", "제주은행", 0.34), ("02", "NH농협카드", 0.22), ("03", "신한카드", 0.14), ("04", "KB국민카드", 0.12),
             ("05", "비씨카드", 0.10), ("06", "삼성카드", 0.08)]
# (시장명, 법정동, 장날 끝자리(None=상설), 하루 기준 사용액(원))
MARKETS = [
    ("동문재래시장", "이도일동", None, 16_000_000), ("동문공설시장", "일도일동", None, 7_000_000),
    ("제주시민속오일시장", "도두일동", (2, 7), 26_000_000), ("서귀포매일올레시장", "서귀동", None, 13_000_000),
    ("서귀포향토오일시장", "동홍동", (4, 9), 15_000_000), ("한림민속오일시장", "한림읍", (4, 9), 9_000_000),
    ("세화민속오일시장", "구좌읍", (5, 0), 6_000_000), ("중문향토오일시장", "중문동", (3, 8), 5_000_000),
    ("대정오일시장", "대정읍", (1, 6), 6_500_000), ("고성오일시장", "애월읍", (2, 7), 5_500_000),
    ("성산오일시장", "성산읍", (1, 6), 4_500_000), ("표선오일시장", "표선면", (2, 7), 4_000_000),
]
MARKET_BIZ = [("TB05", 0.45), ("TB01", 0.20), ("TB11", 0.20), ("TB03", 0.15)]
OBSERVATORIES = [("301", "제주"), ("302", "서귀포"), ("303", "고산"), ("304", "성산"), ("305", "한림"), ("306", "표선")]
WET_P = [0.45, 0.40, 0.36, 0.34, 0.34, 0.46, 0.44, 0.40, 0.38, 0.26, 0.30, 0.40]   # 월별 비 온 날 확률
WET_MM = [6, 7, 9, 11, 13, 22, 20, 24, 21, 12, 9, 7]                                 # 비 온 날 평균 강수량
TEMP = [6.8, 7.9, 10.6, 14.6, 18.6, 22.0, 26.4, 27.5, 24.0, 19.2, 13.8, 8.9]
SEOLLAL = {"2025": [date(2025, 1, 28), date(2025, 1, 29), date(2025, 1, 30)],
           "2023": [date(2023, 1, 21), date(2023, 1, 22), date(2023, 1, 23), date(2023, 1, 24)],
           "2024": [date(2024, 2, 9), date(2024, 2, 10), date(2024, 2, 11), date(2024, 2, 12)]}
DAILY_TOTAL = 1_250_000_000          # 제주 전체 하루 평균 탐나는전 사용액(가정)
WEEKDAY = [1.00, 1.00, 1.00, 1.02, 1.08, 1.05, 0.83]
TRUE_B = 0.045                       # 정답 반응: log(사용액) / 적립률 %p
PREPAY = {"window": ("202504", "202506"), "up": 0.40, "ticket": 1.6, "post": ("202507", "202508"), "down": 0.60}


def true_rate(ym: str) -> float:
    from dataolle.courses.rates import SPANS
    for sp in SPANS:
        if sp.first <= ym <= sp.last:
            return sp.rate
    return 7.0


EVENT = {"from": date(2025, 6, 1), "lift": 1.12,
         "desc": "가상 시나리오: 2025년 6월부터 전통시장 탐나는전 사용액이 12% 늘도록 설정('전통시장 추가 할인' 가정)"}


# 바람(가정): 관측소별 일 최대풍속의 월평균(m/s, 공항 인근 '제주' 기준)과 지역 배수. 겨울은 북서 계절풍, 여름은 남풍, 7~9월 태풍.
WIND_MEAN = [9.2, 8.9, 8.6, 8.0, 6.8, 6.3, 6.6, 6.6, 6.8, 6.8, 7.6, 8.9]
WIND_LOCAL = {"301": 1.0, "302": 0.85, "303": 1.25, "304": 1.05, "305": 1.1, "306": 0.95}


def jeju_wind(d: date, code: str = "301", idx: int = 0) -> tuple[float, str, float, float]:
    """(일 최대풍속 m/s, 그 시각 HHMM, 평균풍속, 최대풍속 풍향 도) — 같은 날은 관측소가 달라도 같은 기상 흐름을 공유."""
    g = np.random.default_rng(int(d.strftime("%Y%m%d")) + 101)
    m = d.month
    base = float(g.gamma(9.0, WIND_MEAN[m - 1] / 9.0))
    u = g.random()
    if m in (11, 12, 1, 2, 3):
        dr = g.normal(325, 18) if u < 0.7 else g.uniform(0, 360)
    elif m in (6, 7, 8, 9):
        dr = g.normal(185, 25) if u < 0.6 else g.uniform(0, 360)
    else:
        dr = g.normal(325, 25) if u < 0.4 else (g.normal(185, 25) if u < 0.7 else g.uniform(0, 360))
    hh = int(np.clip(round(g.normal(14, 2.2)), 8, 20))
    mm = int(g.integers(0, 6)) * 10
    storm = g.random()
    if m in (7, 8, 9) and storm < 0.012:                      # 태풍
        base, dr, hh = float(g.uniform(19, 28)), float(g.normal(150, 30)), int(g.integers(10, 17))
    elif m in (12, 1, 2) and storm < 0.012:                   # 겨울 폭풍(강풍·윈드시어)
        base, dr, hh = float(g.uniform(17, 22)), float(g.normal(320, 12)), int(g.integers(9, 16))
    loc = np.random.default_rng(int(d.strftime("%Y%m%d")) * 10 + 7 + idx)
    w = max(0.8, base * WIND_LOCAL.get(code, 1.0) * float(loc.lognormal(0, 0.08)))
    return (round(w, 1), f"{hh:02d}{mm:02d}", round(w * float(loc.uniform(0.35, 0.5)), 1),
            round(float((dr + loc.normal(0, 10)) % 360), 1))


def _days(ym: str) -> list[date]:
    y, m = int(ym[:4]), int(ym[4:6])
    return [date(y, m, d) for d in range(1, calendar.monthrange(y, m)[1] + 1)]


def _zip(city: str, idx: int) -> str:
    return f"{630 if city == '제주시' else 635}{idx:02d}"


class TamnaWorld:
    def __init__(self, reg, seed: int = 20261120):
        self.rng = np.random.default_rng(seed + 32)
        self.areas = []
        counters = {"제주시": 0, "서귀포시": 0}
        for city, name, _area in reg.bjd_list():
            counters[city] += 1
            self.areas.append((city, name, W.size_of(name), _zip(city, counters[city] % 100)))
        tot = sum(a[2] for a in self.areas)
        self.share = {a[1]: a[2] / tot for a in self.areas}
        from dataolle.analysis.semantic import BJD_UNIT, UNIT_GROUP
        self.group = {a[1]: UNIT_GROUP.get(BJD_UNIT.get((a[0], a[1]), a[1]), "그 밖의 동") for a in self.areas}
        # 동네 유형별 침투 수준(원도심↑, 외곽 섬·면↓) — 가정
        self.level = {a[1]: W.TAMNA_GROUP_LEVEL.get(self.group[a[1]], 1.0) * W.TAMNA_LOW_AREAS.get(a[1], 1.0) for a in self.areas}
        self.biz_presence = {(a[1], b[0]): (self.rng.random() < min(1.0, 0.55 + a[2])) for a in self.areas for b in TAMNA_BIZ}
        self._rain_cache: dict[date, float] = {}

    # -- 날씨 -------------------------------------------------------------------
    def rain(self, d: date) -> float:
        """제주 전체 기준 강수량(관측소 평균에 가까운 값)."""
        if d not in self._rain_cache:
            r = np.random.default_rng(int(d.strftime("%Y%m%d")))
            wet = r.random() < WET_P[d.month - 1]
            self._rain_cache[d] = float(r.lognormal(np.log(WET_MM[d.month - 1]), 0.8)) if wet else 0.0
        return self._rain_cache[d]

    def weather_month(self, ym: str) -> pd.DataFrame:
        rows = []
        for d in _days(ym):
            base = self.rain(d)
            r = np.random.default_rng(int(d.strftime("%Y%m%d")) + 7)
            for i, (code, _name) in enumerate(OBSERVATORIES):
                local = 0.0 if (base > 0 and r.random() < 0.12) else base * float(r.lognormal(0, 0.45))
                t = TEMP[d.month - 1] + r.normal(0, 2.2)
                low = round(t - r.uniform(2, 6), 1)
                low_t = f"{int(r.integers(300, 700)):04d}"
                high = round(t + r.uniform(2, 6), 1)
                high_t = f"{int(r.integers(1200, 1600)):04d}"
                r.uniform(3, 14), r.integers(900, 1900), r.uniform(1, 6), r.uniform(0, 360)   # 예전 난수 순서 유지(비·기온 값 불변)
                w, w_t, w_avg, w_dir = jeju_wind(d, code, i)
                rows.append({"base_date": d.strftime("%Y%m%d"), "observatory_code": code,
                             "average_temperature": round(t, 1), "lowest_temperature": low,
                             "lowest_temperature_time": low_t, "highest_temperature": high,
                             "highest_temperature_time": high_t,
                             "daily_rainfall": round(local, 1), "maximum_wind_speed": w,
                             "maximum_wind_speed_time": w_t, "average_wind_speed": w_avg,
                             "maximum_wind_speed_direction": w_dir})
        return pd.DataFrame(rows)

    # -- 탐나는전 ----------------------------------------------------------------
    def _day_factor(self, d: date) -> float:
        f = WEEKDAY[d.weekday()] * W.SEASON_RES[d.month - 1] * (1 + 0.004 * ((d.year - 2024) * 12 + d.month))
        seol = SEOLLAL.get(str(d.year), [])
        if d in seol:
            f *= 0.55
        elif seol and timedelta(0) < (seol[0] - d) <= timedelta(days=7):
            f *= 1.22
        return f

    def _pre_seol(self, d: date) -> bool:
        seol = SEOLLAL.get(str(d.year), [])
        return bool(seol) and timedelta(0) < (seol[0] - d) <= timedelta(days=7)

    def _split(self, amount: float, ticket: float):
        """금액을 매입사 1~2곳으로 나누고 건수를 만든다."""
        k = int(self.rng.integers(1, 3))
        pick = self.rng.choice(len(ACQUIRERS), size=k, replace=False, p=[a[2] for a in ACQUIRERS])
        w = self.rng.dirichlet(np.ones(k) * 2)
        for i, share in zip(pick, w):
            amt = amount * share
            cnt = max(0, int(round(amt / (ticket * float(self.rng.lognormal(0, 0.15))))))
            if cnt == 0:
                continue
            yield ACQUIRERS[i][0], ACQUIRERS[i][1], int(round(amt, -1)), cnt

    def tamna_month(self, ym: str) -> pd.DataFrame:
        rows = []
        biz = {b[0]: b for b in TAMNA_BIZ}
        area_of = {a[1]: a for a in self.areas}
        t_m = (int(ym[:4]) - 2025) * 12 + int(ym[4:6]) - 1
        trend = {n: (1 + 0.5 * W.CARD_GROUP_TREND.get(g, 0.0)) ** t_m for n, g in self.group.items()}
        rate_f = float(np.exp(TRUE_B * (true_rate(ym) - 7.0)))
        in_win = PREPAY["window"][0] <= ym <= PREPAY["window"][1]
        in_post = PREPAY["post"][0] <= ym <= PREPAY["post"][1]
        edu_f = (1 + PREPAY["up"]) if in_win else ((1 - PREPAY["down"]) if in_post else 1.0)
        edu_t = PREPAY["ticket"] if in_win else 1.0
        for d in _days(ym):
            ymd = d.strftime("%Y%m%d")
            day_f = self._day_factor(d)
            wet = self.rain(d) >= 5
            for city, name, _size, zipc in self.areas:
                for code, nm, ind, weight, ticket, rain_s in TAMNA_BIZ:
                    if not self.biz_presence[(name, code)]:
                        continue
                    f = day_f * (rain_s if wet else 1.0) * self.level[name] * trend[name] * rate_f
                    if code == "TB05" and self._pre_seol(d):
                        f *= 1.5
                    tk = ticket
                    if code == "TB06":
                        f *= edu_f
                        tk = ticket * edu_t
                        if d.weekday() == 6:
                            f *= 0.25
                    amount = DAILY_TOTAL * self.share[name] * weight * f * float(self.rng.lognormal(0, 0.10))
                    for ccd, cnm, amt, cnt in self._split(amount, tk):
                        rows.append((ymd, zipc, ccd, cnm, code, nm, ind, amt, cnt, name, None))
            for mname, bjd, cycle, base in MARKETS:
                if bjd not in area_of:
                    continue
                city, _, _, zipc = area_of[bjd]
                if cycle is None:
                    mf = 1.0
                else:
                    mf = 3.2 if (d.day % 10 in cycle) else 0.18
                f = day_f * mf * (0.78 if wet else 1.0) * (1.9 if self._pre_seol(d) else 1.0) * rate_f
                if d >= EVENT["from"]:
                    f *= EVENT["lift"]
                for code, share in MARKET_BIZ:
                    amount = base * share * f * float(self.rng.lognormal(0, 0.10))
                    for ccd, cnm, amt, cnt in self._split(amount, biz[code][4]):
                        rows.append((ymd, zipc, ccd, cnm, code, biz[code][1], biz[code][2], amt, cnt, bjd, mname))
        return pd.DataFrame(rows, columns=["base_ymd", "zip_code", "card_cd", "card_nm", "biz_cd", "biz_nm", "induty",
                                           "amount", "cnt", "emd", "mrkt_nm"])
