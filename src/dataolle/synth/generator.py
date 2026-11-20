"""가상 센터 CSV 생성기.

센터 공식 CSV 형식을 그대로 재현한다(게시판 공식 답변 기준):
헤더=명세서 영문 컬럼명, 쉼표 구분, 모든 값 큰따옴표, 결측 NA, 파티션 컬럼 미포함, UTF-8(BOM 없음),
파일명 Results_YYYYMMDD_HHMMSS.csv, 1개월 1파일.

생성되는 숫자는 모두 가정에 기반한 가짜 값이다(데모·검증 전용).
"""
from __future__ import annotations

import argparse
import calendar
import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from dataolle.ingest.inventory import month_range
from dataolle.spec import default_registry
from dataolle.synth import world as W

SYNTH_NOTICE = "가상데이터(데모·검증용) — 실제 제주빅데이터센터 데이터가 아닙니다."

USER_TOUR = [("제주도민", "도민"), ("내국인", "개별"), ("내국인", "단체"), ("법인", "NA"),
             ("중국", "개별"), ("중국", "단체"), ("일본", "개별"), ("일본", "단체"),
             ("동남아", "개별"), ("동남아", "단체"), ("기타외국", "개별"), ("기타외국", "단체")]
AGES = ["10", "20", "30", "40", "50", "60", "70", "NA"]
SEXES = ["남성", "여성", "알수없음"]
WEEKDAYS = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
TIME5 = ["오전", "점심", "오후", "저녁", "밤"]
YOUNG_W = np.array([0.4, 0.6, 0.3, 0.0, -0.3, -0.5, -0.7])

# 가상 법정동 코드(8자리) 순서 — 실제 코드와 다를 수 있음
_JEJU_DONG_ORDER = ["일도일동", "일도이동", "이도일동", "이도이동", "삼도일동", "삼도이동", "용담일동", "용담이동", "용담삼동",
                    "건입동", "화북일동", "화북이동", "삼양일동", "삼양이동", "삼양삼동", "봉개동", "아라일동", "아라이동",
                    "오라일동", "오라이동", "오라삼동", "노형동", "외도일동", "외도이동", "이호일동", "이호이동", "도두일동",
                    "도두이동", "도남동", "도련일동", "도련이동", "용강동", "회천동", "오등동", "월평동", "영평동", "연동",
                    "도평동", "해안동", "내도동"]
_SGP_DONG_ORDER = ["서귀동", "법환동", "서호동", "호근동", "동홍동", "서홍동", "상효동", "하효동", "신효동", "보목동", "토평동",
                   "중문동", "회수동", "대포동", "월평동", "강정동", "도순동", "하원동", "색달동", "하예동", "상예동", "영남동"]
_EUPMYEON_CD = {("제주시", "한림읍"): "50110250", ("제주시", "애월읍"): "50110253", ("제주시", "구좌읍"): "50110256",
                ("제주시", "조천읍"): "50110259", ("제주시", "한경면"): "50110310", ("제주시", "추자면"): "50110320",
                ("제주시", "우도면"): "50110330", ("서귀포시", "대정읍"): "50130250", ("서귀포시", "남원읍"): "50130253",
                ("서귀포시", "성산읍"): "50130259", ("서귀포시", "안덕면"): "50130310", ("서귀포시", "표선면"): "50130320"}


def bjd_code(city: str, name: str) -> str:
    if (city, name) in _EUPMYEON_CD:
        return _EUPMYEON_CD[(city, name)]
    order = _JEJU_DONG_ORDER if city == "제주시" else _SGP_DONG_ORDER
    prefix = "5011" if city == "제주시" else "5013"
    return f"{prefix}01{order.index(name) + 1:02d}"


def _holidays(ym: str) -> int:
    return sum(1 for d in W.HOLIDAYS.get(ym[:4], []) if d[:2] == ym[4:6])


def _weekday_days(ym: str) -> np.ndarray:
    y, m = int(ym[:4]), int(ym[4:6])
    counts = np.zeros(7)
    for d in range(1, calendar.monthrange(y, m)[1] + 1):
        counts[calendar.weekday(y, m, d)] += 1
    return counts


def _tilted(base: list[float], young: float) -> np.ndarray:
    p = np.array(base) * np.exp(young * YOUNG_W)
    return p / p.sum()


class SynthWorld:
    """업종·지역 입점 현황은 시드로 고정(같은 세계), 월별 수치만 달라진다."""

    def __init__(self, seed: int = 20261120):
        self.seed = seed
        self.reg = default_registry()
        self.biz_names = self.reg.biz_names()
        rng = np.random.default_rng(seed)
        rows = []
        for city, name, area in self.reg.bjd_list():
            size, tour = W.size_of(name), W.tourism_of(name)
            for code, prof in W.BIZ.items():
                only = prof.get("only")
                if only is not None:
                    if name not in only:
                        continue
                    p = prof["presence"]
                else:
                    p = prof["presence"] * min(1.0, 0.35 + size / 1.5)
                    if prof["tour"] > 0.8:
                        p *= 0.5 + 0.8 * tour
                if rng.random() < min(p, 1.0):
                    rows.append(dict(city=city, bjd=name, area_type=area, dstrct_se=W.REGION_OF[(city, name)],
                                     biz_cd=code, biz_nm=self.biz_names[code], size=size, tourism=tour,
                                     static_noise=float(rng.lognormal(0, 0.25))))
        self.pairs = pd.DataFrame(rows)
        from dataolle.analysis.semantic import BJD_UNIT, UNIT_GROUP
        self.pairs["agrp"] = [UNIT_GROUP.get(BJD_UNIT.get((c, b), b), "그 밖의 동")
                              for c, b in zip(self.pairs["city"], self.pairs["bjd"])]
        self._pos_static = self._make_pos_static(rng)

    # ------------------------------------------------------------------ 카드 1-08
    def card_month(self, ym: str) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed + int(ym))
        m = int(ym[4:6]) - 1
        trend = 1.0 + 0.02 * (int(ym[:4]) - 2025)
        P = self.pairs
        n = len(P)
        prof = [W.BIZ[c] for c in P["biz_cd"]]
        tour_aff = np.array([p["tour"] for p in prof])
        ts = np.clip(P["tourism"].to_numpy() * tour_aff * 1.15, 0, 0.97)
        base = np.array([p["base_tx"] for p in prof]) * P["size"].to_numpy() * P["static_noise"].to_numpy()
        t_m = (int(ym[:4]) - 2025) * 12 + m                         # 2025년 1월부터 지난 개월 수
        grp_trend = np.array([(1 + W.CARD_GROUP_TREND.get(g, 0.0)) ** t_m for g in P["agrp"]])
        tx = base * (ts * W.SEASON_TOUR[m] + (1 - ts) * W.SEASON_RES[m]) * trend * grp_trend * rng.lognormal(0, 0.06, n)
        for ev in W.DEMO_EVENTS:                       # 시연용 가상 효과(코스 3 검증용)
            if ym >= ev["from"]:
                hit = ((P["bjd"] == ev["bjd"]) & (P["biz_cd"] == ev["biz_cd"])).to_numpy()
                tx = np.where(hit, tx * ev["lift"], tx)
        corp = np.array([p["corp"] for p in prof])
        fw = np.array([p["foreign"] for p in prof])
        mix = np.tile(np.array(W.TOURIST_MIX), (n, 1))
        mix[:, 1:] *= fw[:, None]
        mix /= mix.sum(axis=1, keepdims=True)
        grp = np.array([W.GROUP_SHARE[k] for k in ["내국인", "중국", "일본", "동남아", "기타외국"]])
        ut = np.zeros((n, len(USER_TOUR)))
        ut[:, 0] = (1 - ts) * (1 - corp)
        ut[:, 3] = corp
        for i, nat in enumerate(["내국인", "중국", "일본", "동남아", "기타외국"]):
            j = 1 + 2 * i if i == 0 else 2 + 2 * i
            share = ts * (1 - corp) * mix[:, i]
            ut[:, j] = share * (1 - grp[i])
            ut[:, j + 1] = share * grp[i]
        # 나이×성별 분포(조합별)
        female = np.array([p["female"] for p in prof])
        young = np.array([p["young"] for p in prof])
        expected = np.zeros((n, len(USER_TOUR), len(AGES), len(SEXES)))
        for k, (utype, _) in enumerate(USER_TOUR):
            for r in range(n):
                if utype == "법인":
                    a = np.zeros(8); a[7] = 1.0
                    s = np.array([0.0, 0.0, 1.0])
                elif utype in ("제주도민", "내국인"):
                    a = np.append(_tilted(W.AGE_RES if utype == "제주도민" else W.AGE_TOUR, young[r]) * 0.995, 0.005)
                    s = np.array([(1 - female[r]) * 0.99, female[r] * 0.99, 0.01])
                else:
                    a = np.append(_tilted(W.AGE_TOUR, young[r]) * 0.4, 0.6)
                    s = np.array([0.25, 0.25, 0.5])
                expected[r, k] = tx[r] * ut[r, k] * np.outer(a, s)
        use_cnt = rng.poisson(expected)
        idx = np.nonzero(use_cnt)
        cnt = use_cnt[idx]
        r_i, k_i, a_i, s_i = idx
        utypes = np.array([u for u, _ in USER_TOUR])[k_i]
        vpu = np.where(utypes == "제주도민", np.array([p["vpu"] for p in prof])[r_i],
                       np.where(utypes == "법인", 1.5, np.where(utypes == "내국인", 1.15, 1.1)))
        users = np.clip(np.round(cnt / vpu * rng.lognormal(0, 0.08, len(cnt))), 1, cnt).astype(int)
        mult = np.where(utypes == "제주도민", 1.0, np.where(utypes == "법인", 1.3, np.where(utypes == "내국인", 1.12, 1.25)))
        ticket = np.array([p["ticket"] for p in prof])[r_i]
        amt = np.round(cnt * ticket * mult * rng.lognormal(0, 0.12, len(cnt)), -1).astype(np.int64)
        out = pd.DataFrame({
            "base_year_month": ym,
            "dstrct_se": P["dstrct_se"].to_numpy()[r_i], "city": P["city"].to_numpy()[r_i],
            "area_type": P["area_type"].to_numpy()[r_i], "bjd": P["bjd"].to_numpy()[r_i],
            "biz_cd": P["biz_cd"].to_numpy()[r_i], "biz_nm": P["biz_nm"].to_numpy()[r_i],
            "user_type": utypes, "tour_type": np.array([t for _, t in USER_TOUR])[k_i],
            "age": np.array(AGES)[a_i], "sex": np.array(SEXES)[s_i],
            "user_cnt": users, "use_cnt": cnt.astype(int), "use_amt": amt,
        })
        return out

    # ------------------------------------------------------------------ 요일 1-12 / 시간대 1-14
    def _split(self, card: pd.DataFrame, probs: np.ndarray, labels: list[str], col: str, rng) -> pd.DataFrame:
        cnt = card["use_cnt"].to_numpy()
        parts = rng.multinomial(cnt, probs)                     # (N, K)
        vpu = cnt / card["user_cnt"].to_numpy()
        frames = []
        for j, lab in enumerate(labels):
            c = parts[:, j]
            keep = c > 0
            if not keep.any():
                continue
            sub = card.loc[keep].copy()
            cj = c[keep]
            sub[col] = lab
            sub["use_cnt"] = cj
            sub["user_cnt"] = np.clip(np.round(cj / np.maximum(1, vpu[keep] * 0.75)), 1, cj).astype(int)
            sub["use_amt"] = np.round(card["use_amt"].to_numpy()[keep] * cj / cnt[keep] * rng.lognormal(0, 0.05, keep.sum()), -1).astype(np.int64)
            frames.append(sub)
        return pd.concat(frames, ignore_index=True)

    def week_month(self, card: pd.DataFrame, ym: str) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed + int(ym) + 12)
        days = _weekday_days(ym)
        keys = card["biz_cd"].map(lambda c: W.BIZ[c]["week"]).to_numpy()
        probs = np.array([days * np.array(W.WEEK[k]) for k in keys])
        probs /= probs.sum(axis=1, keepdims=True)
        out = self._split(card, probs, WEEKDAYS, "week", rng)
        h = _holidays(ym)
        if h:
            n_days = days.sum()
            ph = np.clip(h / n_days * np.array([W.HOLIDAY_FACTOR[k] for k in keys]), 0, 0.9)
            hc = rng.binomial(card["use_cnt"].to_numpy(), ph)
            keep = hc > 0
            hol = card.loc[keep].copy()
            hol["week"] = "공휴일"           # 공휴일은 원래 요일에도 함께 집계됨(중복)
            vpu = hol["use_cnt"].to_numpy() / hol["user_cnt"].to_numpy()
            hol["use_amt"] = np.round(hol["use_amt"].to_numpy() * hc[keep] / hol["use_cnt"].to_numpy(), -1).astype(np.int64)
            hol["use_cnt"] = hc[keep]
            hol["user_cnt"] = np.clip(np.round(hc[keep] / np.maximum(1, vpu * 0.75)), 1, hc[keep]).astype(int)
            out = pd.concat([out, hol], ignore_index=True)
        cols = list(self.reg.get("1-12").csv_columns)
        return out[cols]

    def time_month(self, card: pd.DataFrame, ym: str) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed + int(ym) + 14)
        keys = card["biz_cd"].map(lambda c: W.BIZ[c]["time"]).to_numpy()
        probs = np.array([W.TIME5[k] for k in keys], dtype=float)
        probs /= probs.sum(axis=1, keepdims=True)
        out = self._split(card, probs, TIME5, "time", rng)
        cols = list(self.reg.get("1-14").csv_columns)
        return out[cols]

    # ------------------------------------------------------------------ 매장 수 1-17
    STORE_PER_UNIT = {"I56111": 40, "I56220": 25, "I56219": 15, "G47122": 8, "G47121": 6, "I56112": 6, "I56113": 5,
                      "I56114": 5, "I56192": 7, "G47711": 2, "I55111": 2, "I55112": 3, "I55113": 1.5, "L69110": 3,
                      "R91121": 0.6, "G47215": 6, "G47813": 4, "G47842": 5, "H49231": 10, "S96122": 3}
    CORP_OWNER = {"G47130": 1.0, "G47119": 0.9, "I55111": 0.6, "I55113": 0.7, "H51100": 1.0, "H50121": 0.8, "L69110": 0.6,
                  "R91121": 0.9, "R91249": 1.0, "G47711": 0.4, "G47122": 0.3}

    def stores_month(self, ym: str) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed + int(ym) + 17)
        months_from = (int(ym[:4]) - 2023) * 12 + int(ym[4:6])
        g = self.pairs.groupby(["city", "biz_cd", "biz_nm"], as_index=False)["size"].sum()
        rows = []
        for _, r in g.iterrows():
            spu = self.STORE_PER_UNIT.get(r.biz_cd, 2.5)
            total = r["size"] * spu * (1 + 0.004 * months_from)
            corp = self.CORP_OWNER.get(r.biz_cd, 0.12)
            for owner, share in (("내국인 개인", 1 - corp), ("법인", corp)):
                c = int(round(total * share * rng.lognormal(0, 0.03)))
                if c > 0:
                    rows.append(dict(base_year_month=ym, city=r.city, owner_gb=owner, biz_cd=r.biz_cd, biz_nm=r.biz_nm, store_cnt=c))
        return pd.DataFrame(rows)[list(self.reg.get("1-17").csv_columns)]

    # ------------------------------------------------------------------ 외식업 POS 1-19 / 1-22 / 1-24
    def _make_pos_static(self, rng) -> pd.DataFrame:
        pos = self.reg.codes["pos_biz"]
        mids: dict[str, list] = {}
        for mid, mid_nm, small, small_nm in pos:
            mids.setdefault(mid, []).append((small, small_nm, mid_nm))
        rows = []
        for city, name, _ in self.reg.bjd_list():
            size, tour = W.size_of(name), W.tourism_of(name)
            for mid, smalls in mids.items():
                prof = W.POS_MID[mid]
                weights = np.array([W.POS_SMALL_WEIGHT.get(s, 1.0) for s, _, _ in smalls])
                weights = weights / weights.mean()
                for (small, small_nm, mid_nm), w in zip(smalls, weights):
                    lam = prof["density"] * size * w / len(smalls) * 3.0 * (0.7 + 0.6 * tour * prof["tour"])
                    k = rng.poisson(lam)
                    if k > 0:
                        rows.append(dict(city=city, bjd=name, mid=mid, mid_nm=mid_nm, small=small, small_nm=small_nm,
                                         stores=int(k), tourism=tour, noise=float(rng.lognormal(0, 0.15))))
        return pd.DataFrame(rows)

    def pos_month(self, ym: str) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed + int(ym) + 19)
        m = int(ym[4:6]) - 1
        S = self._pos_static
        prof = S["mid"].map(W.POS_MID)
        tour_w = S["tourism"] * prof.map(lambda p: p["tour"])
        season = tour_w * W.SEASON_TOUR[m] + (1 - tour_w) * W.SEASON_RES[m]
        stores = np.maximum(1, S["stores"].to_numpy() + rng.integers(-1, 2, len(S)))
        sale = np.round(stores * prof.map(lambda p: p["per_store"]).to_numpy() * season.to_numpy() * S["noise"].to_numpy()
                        * rng.lognormal(0, 0.08, len(S)), -2).astype(np.int64)
        ticket = prof.map(lambda p: p["ticket"]).to_numpy()
        use = np.maximum(1, np.round(sale / ticket * rng.lognormal(0, 0.05, len(S)))).astype(np.int64)
        out = pd.DataFrame({
            "base_year_month": ym,
            "city_cd": S["city"].map(lambda c: "5011" if c == "제주시" else "5013"), "city_nm": S["city"],
            "bjd_cd": [bjd_code(c, b) for c, b in zip(S["city"], S["bjd"])], "bjd_nm": S["bjd"],
            "biz_mid_cd": S["mid"], "biz_mid_nm": S["mid_nm"], "biz_small_cd": S["small"], "biz_small_nm": S["small_nm"],
            "sale_amt": sale, "store_cnt": stores, "use_cnt": use,
        })
        return out[list(self.reg.get("1-19").csv_columns)]

    def pos_customers_month(self, pos: pd.DataFrame, ym: str) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed + int(ym) + 22)
        sel = pos[pos.apply(lambda r: (r["city_nm"], r["bjd_nm"]) in W.POS_DETAIL_BJD and r["biz_small_cd"] in W.POS_DETAIL_BIZ, axis=1)]
        codes = self.reg.codes
        sexes, ages, weeks, times = codes["pos_sex"], codes["pos_age"], codes["pos_week"], codes["pos_time"]
        frames = []
        for _, r in sel.iterrows():
            tkey, akey, wkey = W.POS_PATTERN[r["biz_small_cd"]]
            female = 0.6 if akey == "young" else 0.45
            p = np.einsum("i,j,k,l->ijkl", np.array([1 - female, female]), np.array(W.POS_AGE6[akey]),
                          np.array(W.POS_WEEK[wkey]), np.array(W.POS_TIME7[tkey]))
            p = p / p.sum()
            cnt = rng.multinomial(int(r["use_cnt"]), p.ravel()).reshape(p.shape)
            idx = np.nonzero(cnt)
            c = cnt[idx]
            ticket = r["sale_amt"] / max(1, r["use_cnt"])
            frames.append(pd.DataFrame({
                "base_year_month": ym, "city_cd": r["city_cd"], "city_nm": r["city_nm"], "bjd_cd": r["bjd_cd"], "bjd_nm": r["bjd_nm"],
                "biz_mid_cd": r["biz_mid_cd"], "biz_mid_nm": r["biz_mid_nm"], "biz_small_cd": r["biz_small_cd"], "biz_small_nm": r["biz_small_nm"],
                "sex": np.array(sexes)[idx[0]], "age": np.array(ages)[idx[1]], "week": np.array(weeks)[idx[2]], "time": np.array(times)[idx[3]],
                "user_cnt": np.maximum(1, np.round(c / 1.12)).astype(int), "use_cnt": c,
                "use_amt": np.round(c * ticket * rng.lognormal(0, 0.06, len(c)), -1).astype(np.int64),
            }))
        out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=self.reg.get("1-22").csv_columns)
        return out[list(self.reg.get("1-22").csv_columns)]

    def pos_menu_month(self, pos: pd.DataFrame, ym: str) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed + int(ym) + 24)
        month = int(ym[4:6])
        summer = 1.6 if month in (6, 7, 8) else (0.6 if month in (12, 1, 2) else 1.0)
        present = pos.groupby(["city_nm", "bjd_nm", "city_cd", "bjd_cd"])["biz_mid_cd"].apply(set).reset_index()
        menus = self.reg.codes["menu_subset"]
        large_nm = self.reg.codes["menu_large"]
        rows = []
        for _, r in present.iterrows():
            if (r["city_nm"], r["bjd_nm"]) not in W.POS_DETAIL_BJD:
                continue
            tour = W.tourism_of(r["bjd_nm"])
            for large, mid, mid_nm, small, small_nm, sub, sub_nm in menus:
                if W.MENU_NEEDS[large] not in r["biz_mid_cd"]:
                    continue
                cnt = W.MENU_POP[sub] * (1 + 0.8 * tour if sub in W.JEJU_SPECIAL else 1.0) * (summer if sub in W.SUMMER_MENU else 1.0)
                cnt = max(1, int(round(cnt * rng.lognormal(0, 0.15))))
                price = int(round(W.MENU_PRICE[sub] * (1 + 0.08 * tour) * rng.lognormal(0, 0.03), -2))
                rows.append(dict(base_year_month=ym, city_cd=r["city_cd"], city_nm=r["city_nm"], bjd_cd=r["bjd_cd"], bjd_nm=r["bjd_nm"],
                                 menu_large_cd=large, menu_large_nm=large_nm[large], menu_mid_cd=mid, menu_mid_nm=mid_nm,
                                 menu_small_cd=small, menu_small_nm=small_nm, menu_sub_cd=sub, menu_sub_nm=sub_nm,
                                 store_sale_amt=cnt * price, store_sale_cnt=cnt, unit_price=price))
        return pd.DataFrame(rows)[list(self.reg.get("1-24").csv_columns)]

    # ------------------------------------------------------------------ 유동인구 4-10(v01) / 4-11(v02)
    @staticmethod
    def resident_profile(short: str, year_frac: float) -> tuple[float, np.ndarray]:
        """행정동의 (총인구 배율, 14개 연령대 인구 비중×배율) — 동네 유형별 고령화·아동 감소 추세(가정)."""
        from dataolle.analysis.semantic import UNIT_GROUP, _unit_of_hdong
        p = dict(W.POP_GROUP[UNIT_GROUP.get(_unit_of_hdong(short), "그 밖의 동")])
        p.update(W.POP_OVERRIDE.get(short, {}))
        dt = year_frac - 2025.0
        base = np.array(W.POP_AGE_RES)
        child, working, old = base[:2] / base[:2].sum(), base[2:12] / base[2:12].sum(), base[12:] / base[12:].sum()
        total = (1 + p["g_total"]) ** dt
        c = p["share14"] * (1 + p["g_child"]) ** dt
        o = p["share65"] * (1 + p["g_old"]) ** dt
        w = max(total - c - o, 0.3 * total)
        return total, np.concatenate([c * child, w * working, o * old])

    def population_month(self, ym: str, emd_subset: list[str] | None = None, max_days: int | None = None,
                         dataset: str = "4-11") -> pd.DataFrame:
        rng = np.random.default_rng(self.seed + int(ym) + 411)
        y, m = int(ym[:4]), int(ym[4:6])
        n_days = calendar.monthrange(y, m)[1] if max_days is None else min(max_days, calendar.monthrange(y, m)[1])
        hd = self.reg.hdong_list()
        region_of_hdong = {}
        for city, mapping in self.reg.codes["hdong_approx"].items():
            if city.startswith("_"):
                continue
            for bjd, hds in mapping.items():
                for h in hds:
                    region_of_hdong.setdefault(f"{city} {h}", W.REGION_OF[(city, bjd)])
        if emd_subset:
            hd = [(c, n) for c, n in hd if n.split(" ", 1)[1] in emd_subset or n in emd_subset]
        E, H, A = len(hd), 24, 14
        short = [n.split(" ", 1)[1] for _, n in hd]
        R0 = np.array([W.HDONG_POP.get(s, 8000) for s in short], dtype=float)
        prof = [self.resident_profile(s, y + (m - 1) / 12) for s in short]
        R = R0 * np.array([t for t, _ in prof])                       # 그해 총 거주인구(가정)
        res_cnt = R0[:, None] * np.array([a for _, a in prof])         # 행정동 × 연령대 거주인구
        work_w = np.array([W.HDONG_WORK.get(s, 0.6) for s in short])
        tour_w = np.array([W.HDONG_TOUR.get(s, 0.1) for s in short])
        home, work, visit = np.array(W.HOME_24), np.array(W.WORK_24), np.array(W.VISIT_24)
        a_res, a_work, a_tour = np.array(W.POP_AGE_RES), np.array(W.POP_AGE_WORK), np.array(W.POP_AGE_TOUR)
        season = W.SEASON_TOUR[m - 1]
        ages = [c for c, _ in self.reg.codes["pop_age_skt"]]
        frames = []
        for d in range(1, n_days + 1):
            wd = calendar.weekday(y, m, d)
            weekend = wd >= 5
            wf = 0.35 if weekend else 1.0
            vf = 1.2 if weekend else 1.0
            shape = (E, H, 2, A)            # 행정동, 시간, 성별, 연령
            # 거주지 '제주'
            j_res = res_cnt[:, None, None, :] * home[None, :, None, None] * 0.5
            j_work = R[:, None, None, None] * 0.35 * work_w[:, None, None, None] * work[None, :, None, None] * wf * 0.5 * a_work[None, None, None, :]
            j_vis = R[:, None, None, None] * 0.18 * (0.5 + tour_w)[:, None, None, None] * visit[None, :, None, None] * vf * 0.5 * a_res[None, None, None, :]
            # 거주지 '그외'(도외)
            o_res = R[:, None, None, None] * 0.025 * home[None, :, None, None] * 0.5 * a_work[None, None, None, :]
            o_work = R[:, None, None, None] * 0.02 * work[None, :, None, None] * wf * 0.5 * a_work[None, None, None, :]
            o_vis = R[:, None, None, None] * 0.6 * tour_w[:, None, None, None] * season * visit[None, :, None, None] * (vf * 1.05) * 0.5 * a_tour[None, None, None, :]
            for resd, (pr, pw, pv) in (("제주", (j_res, j_work, j_vis)), ("그외", (o_res, o_work, o_vis))):
                noise = rng.lognormal(0, 0.06, size=(3,) + shape)
                e_i, h_i, s_i, a_i = np.indices(shape).reshape(4, -1)
                frames.append(pd.DataFrame({
                    "base_date": f"{ym}{d:02d}", "time": (h_i + 1).astype(str),
                    "dstrct_se": np.array([region_of_hdong[n] for _, n in hd])[e_i],
                    "emd_cd": np.array([c for c, _ in hd])[e_i], "emd_nm": np.array([n for _, n in hd])[e_i],
                    "sex": np.array(["남성", "여성"])[s_i], "age": np.array(ages)[a_i],
                    "resd_pop_cnt": np.round((pr * noise[0]).ravel(), 4), "work_pop_cnt": np.round((pw * noise[1]).ravel(), 4),
                    "visit_pop_cnt": np.round((pv * noise[2]).ravel(), 4), "resd": resd,
                }))
        out = pd.concat(frames, ignore_index=True)
        return out[list(self.reg.get(dataset).csv_columns)]          # 4-10(v01)은 지역구분(dstrct_se) 열이 없음


# ---------------------------------------------------------------------------
# 파일 쓰기
# ---------------------------------------------------------------------------
def write_center_csv(df: pd.DataFrame, path: Path) -> Path:
    """센터 형식(모든 값 큰따옴표, NA, UTF-8 BOM 없음)으로 저장."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL, encoding="utf-8", na_rep="NA", lineterminator="\n")
    return path


PROFILES = {
    # 데이터셋별 생성 월 범위(시작, 끝) 또는 월 목록. 1-22·4-10·4-11은 파일이 커서 일부만.
    # 4-10(v01)은 '매년 같은 달 1개월씩'(인구 추세용 스냅숏) — 센터에서도 이렇게 받으면 다운로드 수가 줄어듦.
    "demo": {"1-08": ("202501", "202512"), "1-12": ("202510", "202512"), "1-14": ("202510", "202512"),
             "1-17": ("202501", "202512"), "1-19": ("202501", "202512"), "1-22": ("202510", "202512"),
             "1-24": ("202501", "202512"), "4-11": ("202512", "202512"),
             "4-10": ["201712", "201912", "202112", "202312"],
             "1-32": ("202301", "202509"), "5-02": ("202410", "202509")},
    "card12": {"1-08": ("202501", "202512")},          # 코스 3 검증용(카드 12개월만)
    "tamna": {"1-08": ("202501", "202509"), "1-32": ("202401", "202509"), "5-02": ("202410", "202509"),
              "4-10": ["201909"], "4-11": ["202509"]},  # 코스 4 검증용
    "small": {"1-08": ("202510", "202512"), "1-12": ("202512", "202512"), "1-14": ("202512", "202512"),
              "1-17": ("202512", "202512"), "1-19": ("202510", "202512"), "1-22": ("202512", "202512"),
              "1-24": ("202512", "202512"), "4-11": ("202512", "202512")},
}


def _months_of(spec) -> list[str]:
    return list(spec) if isinstance(spec, list) else month_range(*spec)


def _plan_of(profile: str) -> dict:
    return {ds: set(_months_of(v)) for ds, v in PROFILES[profile].items()}


def generate(out_dir: str | Path, profile: str = "demo", seed: int = 20261120, pop_days: int | None = 7,
             pop_emd: list[str] | None = None, verbose: bool = True, pop_hist_days: int | None = 2) -> list[dict]:
    """프로필에 맞춰 월별 가상 CSV를 만들고 매니페스트(정답표)를 돌려준다."""
    out_dir = Path(out_dir)
    world = SynthWorld(seed)
    plan = _plan_of(profile)
    weather_frames = []
    ts = datetime(2026, 11, 9, 10, 0, 0)
    manifest = []

    def emit(ds: str, ym: str, df: pd.DataFrame):
        nonlocal ts
        ts += timedelta(seconds=41)
        name = f"Results_{ts:%Y%m%d_%H%M%S}.csv"
        write_center_csv(df, out_dir / name)
        manifest.append({"file": name, "dataset_id": ds, "month": ym, "rows": int(len(df))})
        if verbose:
            print(f"  {name}  {ds}  {ym}  {len(df):,}행")

    all_months = sorted({ym for months in plan.values() for ym in months})
    in_range = lambda ds, ym: ds in plan and ym in plan[ds]
    tw = None
    for ym in all_months:
        card = world.card_month(ym) if any(in_range(d, ym) for d in ("1-08", "1-12", "1-14")) else None
        if in_range("1-08", ym):
            emit("1-08", ym, card)
        if in_range("1-12", ym):
            emit("1-12", ym, world.week_month(card, ym))
        if in_range("1-14", ym):
            emit("1-14", ym, world.time_month(card, ym))
        if in_range("1-17", ym):
            emit("1-17", ym, world.stores_month(ym))
        pos = world.pos_month(ym) if any(in_range(d, ym) for d in ("1-19", "1-22", "1-24")) else None
        if in_range("1-19", ym):
            emit("1-19", ym, pos)
        if in_range("1-22", ym):
            emit("1-22", ym, world.pos_customers_month(pos, ym))
        if in_range("1-24", ym):
            emit("1-24", ym, world.pos_menu_month(pos, ym))
        if in_range("4-10", ym):
            emit("4-10", ym, world.population_month(ym, emd_subset=pop_emd, max_days=pop_hist_days, dataset="4-10"))
        if in_range("4-11", ym):
            emit("4-11", ym, world.population_month(ym, emd_subset=pop_emd, max_days=pop_days))
        if in_range("1-32", ym) or in_range("5-02", ym):
            if tw is None:
                from dataolle.synth.tamna import TamnaWorld
                tw = TamnaWorld(world.reg, seed)
            if in_range("1-32", ym):
                emit("1-32", ym, tw.tamna_month(ym))
            if in_range("5-02", ym):
                wdf = tw.weather_month(ym)
                emit("5-02", ym, wdf)
                weather_frames.append(wdf)
    public = []
    (out_dir / "_가상데이터_안내.txt").write_text(
        SYNTH_NOTICE + "\n센터 CSV 형식(모든 값 큰따옴표, 결측 NA, 파티션 컬럼 없음, UTF-8)을 재현한 데모 파일입니다.\n"
        "파일명만으로는 데이터 종류를 알 수 없도록 실제 센터처럼 Results_날짜_시간.csv로 저장했습니다.\n"
        + "".join(f"시연용 가상 효과: {ev['desc']}\n" for ev in W.DEMO_EVENTS)
        + ("시연용 가상 효과: " + __import__("dataolle.synth.tamna", fromlist=["EVENT"]).EVENT["desc"] + "\n"
           "시연용 가상 반응(정답): 적립률 1%p당 탐나는전 사용액 약 +4.6%(보도된 적립률 이력 적용), "
           "2025년 4~6월 15% 기간 학원 선결제(+40%, 건당 금액 ×1.6) 뒤 7~8월 −60%\n"
           if "1-32" in plan else ""),
        encoding="utf-8")
    (out_dir / "_manifest.json").write_text(json.dumps({"notice": SYNTH_NOTICE, "seed": seed, "profile": profile,
                                                        "files": manifest, "public_files": public}, ensure_ascii=False, indent=1),
                                            encoding="utf-8")
    return manifest


def main():
    ap = argparse.ArgumentParser(description="데이터올레 가상 센터 CSV 생성기(데모·검증용)")
    ap.add_argument("out_dir")
    ap.add_argument("--profile", default="demo", choices=sorted(PROFILES))
    ap.add_argument("--seed", type=int, default=20261120)
    ap.add_argument("--pop-days", type=int, default=7, help="유동인구(4-11) 생성 일수(전체 달은 0)")
    ap.add_argument("--pop-hist-days", type=int, default=2, help="유동인구 v01(4-10) 스냅숏 달의 생성 일수(전체 달은 0)")
    args = ap.parse_args()
    generate(args.out_dir, args.profile, args.seed, pop_days=(None if args.pop_days == 0 else args.pop_days),
             pop_hist_days=(None if args.pop_hist_days == 0 else args.pop_hist_days))


if __name__ == "__main__":
    main()
