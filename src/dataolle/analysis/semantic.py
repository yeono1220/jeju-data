"""시맨틱 모델: 데이터셋별 '차원(무엇으로 나눌지)'과 '지표(무엇을 셀지)'를 사람이 미리 정의한다.

LLM은 이 목록 안에서만 고르게 하고(값은 보지 않음), 실제 SQL은 코드가 만든다.
데이터 함정은 여기서 구조적으로 막는다.
- 요일별 표(1-12)의 '공휴일' 행은 원래 요일에도 중복 집계 → 기본으로 제외
- '월평동'은 제주시·서귀포시에 모두 있음 → 이름 뒤에 시를 붙여 구분
- 결측(NA)은 '알수없음'·'미상'으로 표시
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from importlib import resources

from dataolle.sdc.engine import AggKind


@dataclass(frozen=True)
class Dim:
    key: str
    expr: str
    label: str
    kind: str = "category"          # time | geo | category
    order: tuple[str, ...] = ()
    rollup: str | None = None       # 범주처리 시 올라갈 상위 차원 key


@dataclass(frozen=True)
class Measure:
    key: str
    label: str
    unit: str
    kind: AggKind
    expr: str | None = None         # 기본 지표: SQL 집계식
    num: str | None = None          # 평균 지표: 분자 지표 key
    den: str | None = None          # 평균 지표: 분모 지표 key
    support: str | None = None      # 반출 검수에 쓸 바탕 인원·건수 지표 key
    scale: float = 1.0              # 평균 계산 후 곱할 값


@dataclass
class DatasetModel:
    key: str
    dataset_ids: tuple[str, ...]
    title: str
    dims: dict[str, Dim]
    measures: dict[str, Measure]
    period_dim: str
    base_where: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()

    def dim(self, key: str) -> Dim:
        if key not in self.dims:
            raise KeyError(f"[{self.title}] 사용할 수 없는 차원: {key} (가능: {', '.join(self.dims)})")
        return self.dims[key]

    def measure(self, key: str) -> Measure:
        if key not in self.measures:
            raise KeyError(f"[{self.title}] 사용할 수 없는 지표: {key} (가능: {', '.join(self.measures)})")
        return self.measures[key]

    def base_measures(self, key: str) -> list[Measure]:
        """지표 계산에 필요한 기본 지표들(평균이면 분자·분모, 검수용 바탕 지표 포함)."""
        m = self.measure(key)
        need = [m.num, m.den] if m.expr is None else [m.key]
        if m.support:
            need.append(m.support)
        out, seen = [], set()
        for k in need:
            if k and k not in seen:
                seen.add(k)
                out.append(self.measure(k))
        return out


# ---------------------------------------------------------------------------
# 공통 표현식
# ---------------------------------------------------------------------------
CARD_AGE = ("19세 이하", "20대", "30대", "40대", "50대", "60대", "70세 이상", "알수없음")
_card_age_expr = ("CASE age WHEN '10' THEN '19세 이하' WHEN '20' THEN '20대' WHEN '30' THEN '30대' WHEN '40' THEN '40대' "
                  "WHEN '50' THEN '50대' WHEN '60' THEN '60대' WHEN '70' THEN '70세 이상' ELSE '알수없음' END")
_card_age_band = ("CASE WHEN age IN ('10','20') THEN '20대 이하' WHEN age IN ('30','40') THEN '30·40대' "
                  "WHEN age IN ('50','60') THEN '50·60대' WHEN age = '70' THEN '70세 이상' ELSE '알수없음' END")
TOUR = ("도민", "개별관광", "단체관광", "미상(법인 등)")
_tour_expr = ("CASE tour_type WHEN '도민' THEN '도민' WHEN '개별' THEN '개별관광' WHEN '단체' THEN '단체관광' "
              "ELSE '미상(법인 등)' END")
USER_TYPE = ("제주도민", "내국인", "법인", "중국", "일본", "동남아", "기타외국")
USER_GROUP = ("제주도민", "내국인", "외국인", "법인")
_user_group = "CASE WHEN user_type IN ('중국','일본','동남아','기타외국') THEN '외국인' ELSE user_type END"
SEX = ("남성", "여성", "알수없음")
REGION7 = ("제주시 동부", "제주시 중부(동)", "제주시 중부(서)", "제주시 서부", "서귀포시 동부", "서귀포시 중부", "서귀포시 서부")


def _quarter(col: str) -> str:
    return (f"substr({col},1,4) || '년 ' || CAST((CAST(substr({col},5,2) AS INTEGER) + 2) // 3 AS VARCHAR) || '분기'")


# ---------------------------------------------------------------------------
# 동네 단위(행정동)와 동네 유형(원도심·신도심·그 밖의 동·읍면)
# 법정동 자료(카드·탐나는전)와 행정동 자료(유동인구)를 같은 단위로 맞추기 위한 대응표(근사, codes.json)
# ---------------------------------------------------------------------------
_CODES = json.loads(resources.files("dataolle.spec").joinpath("data/codes.json").read_text(encoding="utf-8"))
_AG = _CODES.get("area_groups", {"원도심": [], "신도심": [], "merge": {}})
AREA_GROUPS = ("원도심", "신도심", "그 밖의 동", "읍면")


def _unit_of_hdong(h: str) -> str:
    for unit, members in _AG.get("merge", {}).items():
        if h in members:
            return unit
    return h


def group_of_unit(u: str) -> str:
    first = _AG.get("merge", {}).get(u, [u])[0]
    if first in _AG["원도심"]:
        return "원도심"
    if first in _AG["신도심"]:
        return "신도심"
    if first.endswith("읍") or first.endswith("면"):
        return "읍면"
    return "그 밖의 동"


BJD_UNIT: dict[tuple[str, str], str] = {}       # (시, 법정동) → 동네(행정동 단위)
for _city, _mp in _CODES.get("hdong_approx", {}).items():
    if _city.startswith("_"):
        continue
    for _bjd, _hds in _mp.items():
        BJD_UNIT[(_city, _bjd)] = sorted({_unit_of_hdong(h) for h in _hds})[0]
HD_UNIT: dict[tuple[str, str], str] = {}        # (시, 행정동) → 동네
for _code, _name in _CODES.get("hdong43", []):
    _c, _h = _name.split(" ", 1)
    HD_UNIT[(_c, _h)] = _unit_of_hdong(_h)
UNIT_GROUP = {u: group_of_unit(u) for u in set(BJD_UNIT.values()) | set(HD_UNIT.values())}
UNIT_CITY = {u: c for (c, _), u in list(BJD_UNIT.items()) + list(HD_UNIT.items())}


def _sq(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def _name_case(city_expr: str, name_expr: str, pairs: dict[tuple[str, str], str], fallback: str) -> str:
    by_name: dict[str, dict[str, str]] = defaultdict(dict)
    for (c, n), v in pairs.items():
        by_name[n][c] = v
    whens = []
    for n, cmap in sorted(by_name.items()):
        vals = set(cmap.values())
        if len(vals) == 1:
            whens.append(f"WHEN {name_expr} = {_sq(n)} THEN {_sq(next(iter(vals)))}")
        else:                                   # 같은 이름이 두 시에 있음(월평동)
            whens += [f"WHEN {city_expr} = {_sq(c)} AND {name_expr} = {_sq(n)} THEN {_sq(v)}" for c, v in sorted(cmap.items())]
    return f"CASE {' '.join(whens)} ELSE {fallback} END"


def unit_expr(city_expr: str, name_expr: str, hdong_names: bool = True) -> str:
    """법정동(또는 행정동) 이름 → 동네(행정동 단위) SQL 식."""
    pairs = dict(BJD_UNIT)
    if hdong_names:
        for k, v in HD_UNIT.items():
            pairs.setdefault(k, v)
    return _name_case(city_expr, name_expr, pairs, name_expr)


def group_expr(city_expr: str, name_expr: str, hdong_names: bool = True) -> str:
    """법정동(또는 행정동) 이름 → 동네 유형 SQL 식."""
    pairs = {k: UNIT_GROUP[v] for k, v in BJD_UNIT.items()}
    if hdong_names:
        for k, v in HD_UNIT.items():
            pairs.setdefault(k, UNIT_GROUP[v])
    fb = f"CASE WHEN {name_expr} LIKE '%읍' OR {name_expr} LIKE '%면' THEN '읍면' ELSE '그 밖의 동' END"
    return _name_case(city_expr, name_expr, pairs, fb)


def _card_time_dims(bjd_expr: str, has_city: bool) -> dict[str, Dim]:
    d = {
        "month": Dim("month", "base_year_month", "연월", "time", rollup="quarter"),
        "quarter": Dim("quarter", _quarter("base_year_month"), "분기", "time", rollup="year"),
        "year": Dim("year", "substr(base_year_month,1,4)", "연도", "time"),
        "region": Dim("region", "dstrct_se", "지역구분(권역)", "geo", REGION7, rollup="city"),
        "bjd": Dim("bjd", bjd_expr, "법정동", "geo", rollup="region"),
        "biz": Dim("biz", "biz_nm", "업종", "category"),
        "user_type": Dim("user_type", "user_type", "이용자구분", "category", USER_TYPE, rollup="user_group"),
        "user_group": Dim("user_group", _user_group, "이용자구분(묶음)", "category", USER_GROUP),
        "tour_type": Dim("tour_type", _tour_expr, "관광구분", "category", TOUR),
        "age": Dim("age", _card_age_expr, "연령대", "category", CARD_AGE, rollup="age_band"),
        "age_band": Dim("age_band", _card_age_band, "연령대(묶음)", "category", ("20대 이하", "30·40대", "50·60대", "70세 이상", "알수없음")),
        "sex": Dim("sex", "coalesce(sex,'알수없음')", "성별", "category", SEX),
    }
    if has_city:
        d["city"] = Dim("city", "city", "시군구", "geo", ("제주시", "서귀포시"))
        d["area"] = Dim("area", "area_type", "도심구분", "category", ("도심", "읍면"))
    else:
        d["city"] = Dim("city", "CASE WHEN dstrct_se LIKE '서귀포%' THEN '서귀포시' ELSE '제주시' END", "시군구", "geo", ("제주시", "서귀포시"))
    cexpr = d["city"].expr
    d["hd"] = Dim("hd", unit_expr(cexpr, "bjd", hdong_names=False), "동네(행정동)", "geo", rollup="agrp")
    d["agrp"] = Dim("agrp", group_expr(cexpr, "bjd", hdong_names=False), "동네 유형", "geo", AREA_GROUPS)
    return d


_CARD_MEASURES = {
    "amt": Measure("amt", "이용금액", "원", AggKind.SIMPLE, expr="SUM(use_amt)", support="users"),
    "cnt": Measure("cnt", "이용건수", "건", AggKind.SIMPLE, expr="SUM(use_cnt)", support="users"),
    "users": Measure("users", "이용자수(중복 포함 합)", "명", AggKind.SIMPLE, expr="SUM(user_cnt)", support="users"),
    "amt_per_cnt": Measure("amt_per_cnt", "건당 이용금액(객단가)", "원", AggKind.AVERAGE, num="amt", den="cnt", support="users"),
    "amt_per_user": Measure("amt_per_user", "1인당 이용금액", "원", AggKind.AVERAGE, num="amt", den="users", support="users"),
    "cnt_per_user": Measure("cnt_per_user", "1인당 이용건수", "회", AggKind.AVERAGE, num="cnt", den="users", support="users"),
}
_CARD_CAVEATS = (
    "비씨카드 결제분만 집계된 표본 자료입니다(다른 카드사·현금 제외). 시장 전체 규모가 아니라 상대 비교에 적합합니다.",
    "이용자구분은 카드 실소유자 기준이며, 이용자수는 구분별로 센 값을 더한 것이라 중복이 포함됩니다.",
)

_bjd_card = "CASE WHEN bjd = '월평동' THEN '월평동(' || city || ')' ELSE bjd END"
_bjd_from_region = ("CASE WHEN bjd = '월평동' THEN '월평동(' || CASE WHEN dstrct_se LIKE '서귀포%' THEN '서귀포시' ELSE '제주시' END || ')' "
                    "ELSE bjd END")

CARD = DatasetModel("card", ("1-08",), "카드 이용 현황(읍면동)", _card_time_dims(_bjd_card, True), dict(_CARD_MEASURES),
                    "month", caveats=_CARD_CAVEATS)

CARD_WEEK = DatasetModel(
    "card_week", ("1-12",), "카드 이용 현황(요일별)",
    {**_card_time_dims(_bjd_from_region, False),
     "week": Dim("week", "week", "요일", "category", ("월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일")),
     "daytype": Dim("daytype", "CASE WHEN week IN ('토요일','일요일') THEN '주말' ELSE '평일' END", "평일·주말", "category", ("평일", "주말"))},
    dict(_CARD_MEASURES), "month",
    base_where=("week <> '공휴일'",),
    caveats=_CARD_CAVEATS + ("요일별 표의 '공휴일' 행은 원래 요일에도 함께 집계되어 있어(중복) 합계 계산에서 제외했습니다.",),
)

CARD_TIME = DatasetModel(
    "card_time", ("1-14",), "카드 이용 현황(시간대)",
    {**_card_time_dims(_bjd_from_region, False),
     "time": Dim("time", "time", "시간대", "category", ("오전", "점심", "오후", "저녁", "밤"))},
    dict(_CARD_MEASURES), "month",
    caveats=_CARD_CAVEATS + ("시간대 구분: 오전 07~11시, 점심 11~14시, 오후 14~18시, 저녁 18~22시, 밤 22시~07시.",),
)

STORES = DatasetModel(
    "stores", ("1-17",), "제주도 내 업종별 매장 수",
    {"month": Dim("month", "base_year_month", "연월", "time", rollup="quarter"),
     "quarter": Dim("quarter", _quarter("base_year_month"), "분기", "time"),
     "year": Dim("year", "substr(base_year_month,1,4)", "연도", "time"),
     "city": Dim("city", "city", "시군구", "geo", ("제주시", "서귀포시")),
     "owner": Dim("owner", "owner_gb", "매장주 구분", "category"),
     "biz": Dim("biz", "biz_nm", "업종", "category")},
    {"stores": Measure("stores", "매장 수", "개", AggKind.SIMPLE, expr="SUM(store_cnt)", support="stores")},
    "month", caveats=("비씨카드 가맹 매장 기준 매장 수입니다. 법정동 단위가 아닌 시 단위입니다.",),
)

_bjd_pos = "CASE WHEN bjd_nm = '월평동' THEN '월평동(' || city_nm || ')' ELSE bjd_nm END"
_POS_DIMS = {
    "month": Dim("month", "base_year_month", "연월", "time", rollup="quarter"),
    "quarter": Dim("quarter", _quarter("base_year_month"), "분기", "time", rollup="year"),
    "year": Dim("year", "substr(base_year_month,1,4)", "연도", "time"),
    "city": Dim("city", "city_nm", "시군구", "geo", ("제주시", "서귀포시")),
    "bjd": Dim("bjd", _bjd_pos, "법정동", "geo", rollup="city"),
    "pos_mid": Dim("pos_mid", "biz_mid_nm", "외식업 중분류", "category"),
    "pos_biz": Dim("pos_biz", "biz_small_nm", "외식업 소분류", "category", rollup="pos_mid"),
}
POS = DatasetModel(
    "pos", ("1-19",), "외식업 포스 이용 현황(읍면동)", dict(_POS_DIMS),
    {"sales": Measure("sales", "총매출액", "원", AggKind.SIMPLE, expr="SUM(sale_amt)", support="samples"),
     "samples": Measure("samples", "표본 점포 수", "개", AggKind.SIMPLE, expr="SUM(store_cnt)", support="samples"),
     "cnt": Measure("cnt", "이용건수", "건", AggKind.SIMPLE, expr="SUM(use_cnt)", support="samples"),
     "sales_per_store": Measure("sales_per_store", "표본 점포당 매출액", "원", AggKind.AVERAGE, num="sales", den="samples", support="samples"),
     "amt_per_cnt": Measure("amt_per_cnt", "건당 결제금액", "원", AggKind.AVERAGE, num="sales", den="cnt", support="samples")},
    "month",
    caveats=("외식업 POS(나이스지니데이터) 표본 점포 자료입니다. 표본수는 전체 점포 수가 아닙니다.",
             "표본 점포가 10곳 미만인 칸의 매출은 개별 점포가 드러날 수 있어 비공개 처리합니다."),
)

POS_CUST = DatasetModel(
    "pos_cust", ("1-22",), "외식업 포스 고객 이용 현황(읍면동)",
    {**_POS_DIMS,
     "sex": Dim("sex", "sex", "성별", "category", ("남성", "여성")),
     "age": Dim("age", "CASE age WHEN '20' THEN '20대 이하' WHEN '70' THEN '70대 이상' ELSE age || '대' END", "연령대", "category",
                ("20대 이하", "30대", "40대", "50대", "60대", "70대 이상")),
     "week": Dim("week", "week || '요일'", "요일", "category", ("월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일")),
     "time": Dim("time", "time", "시간대", "category", ("00~06", "06~09", "09~12", "12~15", "15~18", "18~21", "21~24"))},
    {"users": Measure("users", "이용자수", "명", AggKind.SIMPLE, expr="SUM(user_cnt)", support="users"),
     "cnt": Measure("cnt", "이용건수", "건", AggKind.SIMPLE, expr="SUM(use_cnt)", support="users"),
     "amt": Measure("amt", "이용금액", "원", AggKind.SIMPLE, expr="SUM(use_amt)", support="users"),
     "amt_per_cnt": Measure("amt_per_cnt", "건당 결제금액", "원", AggKind.AVERAGE, num="amt", den="cnt", support="users")},
    "month", caveats=("외식업 POS 표본 점포의 고객 결제 자료입니다.",),
)

POS_MENU = DatasetModel(
    "pos_menu", ("1-24",), "외식업 포스 메뉴 이용 현황(읍면동)",
    {"month": Dim("month", "base_year_month", "연월", "time"),
     "city": Dim("city", "city_nm", "시군구", "geo", ("제주시", "서귀포시")),
     "bjd": Dim("bjd", _bjd_pos, "법정동", "geo"),
     "menu_large": Dim("menu_large", "menu_large_nm", "메뉴 대분류", "category"),
     "menu_mid": Dim("menu_mid", "menu_mid_nm", "메뉴 중분류", "category"),
     "menu_small": Dim("menu_small", "menu_small_nm", "메뉴 소분류", "category"),
     "menu": Dim("menu", "menu_sub_nm", "메뉴", "category")},
    {"avg_cnt": Measure("avg_cnt", "점포당 월 판매건수(평균)", "건", AggKind.AVERAGE, expr="AVG(store_sale_cnt)"),
     "avg_price": Measure("avg_price", "판매단가(평균)", "원", AggKind.AVERAGE, expr="AVG(unit_price)"),
     "avg_sales": Measure("avg_sales", "점포당 월 매출액(평균)", "원", AggKind.AVERAGE, expr="AVG(store_sale_amt)")},
    "month", caveats=("메뉴 자료는 점포당 평균값만 제공되어 표본 점포 수를 알 수 없습니다. 지역 간 비교 시 해석에 주의하세요.",),
)

POP_AGE_BAND = ("19세 이하", "20대", "30대", "40대", "50대", "60대", "70세 이상")
_pop_age_band = ("CASE WHEN age IN ('0009','1014','1519') THEN '19세 이하' WHEN age IN ('2024','2529') THEN '20대' "
                 "WHEN age IN ('3034','3539') THEN '30대' WHEN age IN ('4044','4549') THEN '40대' "
                 "WHEN age IN ('5054','5559') THEN '50대' WHEN age IN ('6064','6569') THEN '60대' ELSE '70세 이상' END")
POP_AGE3 = ("0~14세", "15~64세", "65세 이상")
_pop_age3 = "CASE WHEN age IN ('0009','1014') THEN '0~14세' WHEN age IN ('6569','7000') THEN '65세 이상' ELSE '15~64세' END"
_dow = "dayofweek(strptime(base_date, '%Y%m%d'))"
_pop_name = "regexp_replace(emd_nm, '^(제주시|서귀포시) ', '')"
_pop_city = "CASE WHEN emd_nm LIKE '서귀포시%' THEN '서귀포시' ELSE '제주시' END"
_NIGHT = "time IN ('2','3','4','5')"
POP = DatasetModel(
    "pop", ("4-10", "4-11"), "내국인 유동인구(읍면동)",
    {"date": Dim("date", "base_date", "날짜", "time"),
     "month": Dim("month", "substr(base_date,1,6)", "연월", "time", rollup="year"),
     "year": Dim("year", "substr(base_date,1,4)", "연도", "time"),
     "city": Dim("city", _pop_city, "시", "geo", ("제주시", "서귀포시")),
     "hd": Dim("hd", unit_expr(_pop_city, _pop_name), "동네(행정동)", "geo", rollup="agrp"),
     "agrp": Dim("agrp", group_expr(_pop_city, _pop_name), "동네 유형", "geo", AREA_GROUPS),
     "age3": Dim("age3", _pop_age3, "연령 3구분", "category", POP_AGE3),
     "weekday": Dim("weekday", f"CASE {_dow} WHEN 1 THEN '월요일' WHEN 2 THEN '화요일' WHEN 3 THEN '수요일' WHEN 4 THEN '목요일' "
                               f"WHEN 5 THEN '금요일' WHEN 6 THEN '토요일' ELSE '일요일' END", "요일", "category",
                    ("월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일")),
     "daytype": Dim("daytype", f"CASE WHEN {_dow} IN (0, 6) THEN '주말' ELSE '평일' END", "평일·주말", "category", ("평일", "주말")),
     "hour": Dim("hour", "lpad(time, 2, '0') || '시'", "시각", "time"),
     "region": Dim("region", "COALESCE(dstrct_se, '(권역 정보 없음)')", "지역구분(권역)", "geo", REGION7),
     "emd": Dim("emd", _pop_name, "행정동", "geo", rollup="region"),
     "sex": Dim("sex", "sex", "성별", "category", ("남성", "여성")),
     "age": Dim("age", _pop_age_band, "연령대", "category", POP_AGE_BAND),
     "resd": Dim("resd", "CASE resd WHEN '제주' THEN '도내 거주' ELSE '도외 거주' END", "거주지", "category", ("도내 거주", "도외 거주"))},
    {"visit": Measure("visit", "방문인구(시간대 합, 연인원)", "명", AggKind.SIMPLE, expr="SUM(visit_pop_cnt)", support="visit"),
     "resd_pop": Measure("resd_pop", "거주인구(시간대 합, 연인원)", "명", AggKind.SIMPLE, expr="SUM(resd_pop_cnt)", support="resd_pop"),
     "work": Measure("work", "근무인구(시간대 합, 연인원)", "명", AggKind.SIMPLE, expr="SUM(work_pop_cnt)", support="work"),
     "visit_hourly": Measure("visit_hourly", "시간당 평균 방문인구", "명", AggKind.SIMPLE,
                             expr="SUM(visit_pop_cnt) / NULLIF(COUNT(DISTINCT base_date || '-' || time), 0)", support="visit_hourly"),
     "total_hourly": Measure("total_hourly", "시간당 평균 유동인구(거주+근무+방문)", "명", AggKind.SIMPLE,
                             expr="SUM(resd_pop_cnt + work_pop_cnt + visit_pop_cnt) / NULLIF(COUNT(DISTINCT base_date || '-' || time), 0)",
                             support="total_hourly"),
     # 새벽(02~05시) 거주인구 평균 = 그 동네에 사는 사람 수의 추정(대부분 집에 있는 시간)
     "resd_night": Measure("resd_night", "야간 거주인구(02~05시 평균, 추정)", "명", AggKind.SIMPLE,
                           expr=(f"SUM(resd_pop_cnt) FILTER (WHERE {_NIGHT}) / "
                                 f"NULLIF(COUNT(DISTINCT base_date || '-' || time) FILTER (WHERE {_NIGHT}), 0)"),
                           support="resd_night")},
    "month",
    caveats=("유동인구는 매시 정각에 측정한 추정 인구(통신사)입니다. 시간대를 더한 값은 '연인원'으로 실제 사람 수가 아닙니다.",
             "행정동 단위 자료라 카드 자료(법정동)와 바로 맞대기 어렵습니다. 비교 시 근사 대응표를 사용합니다(추정)."),
)

_dow_ymd = "dayofweek(strptime(base_ymd, '%Y%m%d'))"
_tamna_city = "CASE WHEN zip_code >= '63500' THEN '서귀포시' ELSE '제주시' END"
TAMNA = DatasetModel(
    "tamna", ("1-32",), "탐나는전 이용 현황",
    {"date": Dim("date", "base_ymd", "날짜", "time"),
     "week": Dim("week", "strftime(date_trunc('week', strptime(base_ymd, '%Y%m%d')), '%Y%m%d')", "주(월요일 시작)", "time"),
     "month": Dim("month", "substr(base_ymd,1,6)", "연월", "time"),
     "weekday": Dim("weekday", f"CASE {_dow_ymd} WHEN 1 THEN '월요일' WHEN 2 THEN '화요일' WHEN 3 THEN '수요일' WHEN 4 THEN '목요일' "
                               f"WHEN 5 THEN '금요일' WHEN 6 THEN '토요일' ELSE '일요일' END", "요일", "category",
                    ("월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일")),
     "daytype": Dim("daytype", f"CASE WHEN {_dow_ymd} IN (0, 6) THEN '주말' ELSE '평일' END", "평일·주말", "category", ("평일", "주말")),
     "city": Dim("city", _tamna_city, "시", "geo", ("제주시", "서귀포시")),
     "emd": Dim("emd", f"CASE WHEN emd = '월평동' THEN '월평동(' || {_tamna_city} || ')' ELSE emd END", "읍면동", "geo", rollup="city"),
     "hd": Dim("hd", unit_expr(_tamna_city, "emd"), "동네(행정동)", "geo", rollup="agrp"),
     "agrp": Dim("agrp", group_expr(_tamna_city, "emd"), "동네 유형", "geo", AREA_GROUPS),
     "biz": Dim("biz", "biz_nm", "업종", "category"),
     "induty": Dim("induty", "induty", "표준업종", "category"),
     "card": Dim("card", "card_nm", "매입사", "category"),
     "market": Dim("market", "COALESCE(NULLIF(mrkt_nm, ''), '시장 외')", "전통시장", "category"),
     "mflag": Dim("mflag", "CASE WHEN mrkt_nm IS NULL OR mrkt_nm = '' THEN '일반 가맹점' ELSE '전통시장' END", "가맹점 구분",
                  "category", ("전통시장", "일반 가맹점"))},
    {"amt": Measure("amt", "탐나는전 사용금액", "원", AggKind.SIMPLE, expr="SUM(amount)", support="cnt"),
     "cnt": Measure("cnt", "탐나는전 결제건수", "건", AggKind.SIMPLE, expr="SUM(cnt)", support="cnt"),
     "amt_per_cnt": Measure("amt_per_cnt", "건당 사용금액", "원", AggKind.AVERAGE, num="amt", den="cnt", support="cnt")},
    "month",
    caveats=("탐나는전 결제 자료(승인일 기준)입니다. 결제건수 10건 미만인 칸은 금액도 비공개로 처리합니다.",
             "읍면동은 자료의 읍면동 표기를 따르며, 시 구분은 우편번호(63500 이상=서귀포시)로 판별했습니다."),
)
WEATHER = DatasetModel(
    "weather", ("5-02",), "일일 날씨 정보",
    {"date": Dim("date", "base_date", "날짜", "time"),
     "month": Dim("month", "substr(base_date,1,6)", "연월", "time"),
     "obs": Dim("obs", "observatory_code", "관측소", "category")},
    {"rain": Measure("rain", "일강수량(관측소 평균)", "mm", AggKind.APPLIED, expr="AVG(daily_rainfall)"),
     "temp": Measure("temp", "평균 기온(관측소 평균)", "℃", AggKind.APPLIED, expr="AVG(average_temperature)")},
    "month",
    caveats=("날씨 자료는 반출 가능 데이터입니다. 제주 전체 값은 관측소 평균입니다.",),
)
MODELS: dict[str, DatasetModel] = {m.key: m for m in (CARD, CARD_WEEK, CARD_TIME, STORES, POS, POS_CUST, POS_MENU, POP,
                                                      TAMNA, WEATHER)}
MODEL_OF_DATASET = {ds: m.key for m in MODELS.values() for ds in m.dataset_ids}
