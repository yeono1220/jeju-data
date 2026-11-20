"""제주빅데이터센터 데이터 명세 레지스트리.

명세서 v13(2026.11.04)에서 추출한 71종 데이터셋의 컬럼 구성과 반출 구분,
핵심 데이터셋의 한글 컬럼명, 코드표를 한곳에서 제공한다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

# 센터 CSV 다운로드에는 포함되지 않는 파티션 컬럼(게시판 공식 답변 2026-11-06)
PARTITION_COLS = frozenset({"pt_year", "pt_month"})

# 기간을 나타내는 컬럼 후보(우선순위 순)
PERIOD_COLUMNS = ("base_year_month", "base_ymd", "base_date")

# 여러 데이터셋에 공통으로 쓰이는 컬럼의 한글명(핵심 데이터셋 정의가 없을 때 사용)
GENERIC_LABELS = {
    "base_year_month": "년월", "base_date": "날짜", "base_ymd": "승인일자", "pt_year": "년", "pt_month": "월",
    "dstrct_se": "지역구분", "city": "시군구", "area_type": "도심구분", "bjd": "법정동", "emd": "읍면동",
    "biz_cd": "업종코드", "biz_nm": "업종명", "user_type": "이용자구분", "tour_type": "관광구분",
    "age": "연령대", "sex": "성별", "gender": "성별", "week": "요일구분", "time": "시간대",
    "user_cnt": "이용자수", "use_cnt": "이용건수", "use_amt": "이용금액", "store_cnt": "매장수",
    "day_of_week": "요일", "base_hour": "시간대", "user_count": "이용자수",
    "emd_cd": "행정동코드", "emd_nm": "행정동명", "resd": "거주지행정구역",
    "resd_pop_cnt": "거주인구", "work_pop_cnt": "근무인구", "visit_pop_cnt": "방문인구",
}


@dataclass(frozen=True)
class DatasetSpec:
    id: str
    name_ko: str
    category: str
    raw_exportable: bool          # 명세서 '반출' 열: 원본 반출 가능 여부
    provider: str
    period: str | None            # 명세서에 적힌 수록 기간(텍스트)
    columns: tuple[str, ...]      # 명세서 컬럼(파티션 포함)
    labels: tuple[tuple[str, str], ...] = ()   # (영문, 한글) 쌍
    dtypes: tuple[tuple[str, str], ...] = ()   # (영문, 명세서 Data Type) 쌍

    @property
    def csv_columns(self) -> tuple[str, ...]:
        """센터에서 실제로 내려받는 CSV의 컬럼(파티션 제외)."""
        return tuple(c for c in self.columns if c not in PARTITION_COLS)

    @property
    def restricted(self) -> bool:
        """반출 불가 데이터(카드·POS·유동인구 등): 가공 결과만 조건부 반출, 10 미만 규칙 적용."""
        return not self.raw_exportable

    @property
    def period_column(self) -> str | None:
        for col in PERIOD_COLUMNS:
            if col in self.columns:
                return col
        return None

    def label(self, column: str) -> str:
        own = dict(self.labels)
        return own.get(column) or GENERIC_LABELS.get(column) or column

    def dtype(self, column: str) -> str | None:
        return dict(self.dtypes).get(column)

    def citation(self, period_text: str) -> str:
        """반출 기준의 출처 표기 형식: 제주빅데이터센터, 데이터셋(이름, 기간)."""
        return f"제주빅데이터센터, {self.name_ko}({period_text})"


class SpecRegistry:
    def __init__(self, datasets: list[DatasetSpec], codes: dict, value_semantics: dict | None = None):
        self._by_id = {d.id: d for d in datasets}
        self.codes = codes
        self.value_semantics = value_semantics or {}

    @classmethod
    def from_dir(cls, data_dir: Path = DATA_DIR) -> "SpecRegistry":
        compact = json.loads((data_dir / "center_spec_compact.json").read_text(encoding="utf-8"))
        core = json.loads((data_dir / "core_columns.json").read_text(encoding="utf-8"))
        codes = json.loads((data_dir / "codes.json").read_text(encoding="utf-8"))
        core_cols = core.get("columns", {})
        datasets = []
        for d in compact["datasets"]:
            detail = core_cols.get(d["id"], [])
            datasets.append(DatasetSpec(
                id=d["id"],
                name_ko=d["ko"],
                category=d["cat"],
                raw_exportable=(d["exp"] == "가능"),
                provider=d["prov"],
                period=d.get("per"),
                columns=tuple(d["cols"]),
                labels=tuple((c[0], c[1]) for c in detail),
                dtypes=tuple((c[0], c[2]) for c in detail),
            ))
        return cls(datasets, codes, core.get("value_semantics"))

    def get(self, dataset_id: str) -> DatasetSpec:
        try:
            return self._by_id[dataset_id]
        except KeyError as exc:
            raise KeyError(f"명세서에 없는 데이터셋: {dataset_id}") from exc

    def __contains__(self, dataset_id: str) -> bool:
        return dataset_id in self._by_id

    def __iter__(self):
        return iter(self._by_id.values())

    def __len__(self) -> int:
        return len(self._by_id)

    def search(self, text: str) -> list[DatasetSpec]:
        """데이터셋 이름(한글) 부분 일치 검색."""
        t = text.replace(" ", "")
        return [d for d in self if t in d.name_ko.replace(" ", "")]

    # ---- 코드표 도우미 -------------------------------------------------
    def bjd_list(self) -> list[tuple[str, str, str]]:
        """(시군구, 법정동, 도심구분) 74행. '월평동'은 제주시·서귀포시에 각각 있음."""
        rows = []
        for city, parts in self.codes["bjd"].items():
            rows += [(city, name, "도심") for name in parts["dong"]]
            rows += [(city, name, "읍면") for name in parts["eupmyeon"]]
        return rows

    def biz_names(self) -> dict[str, str]:
        return {code: name for code, name in self.codes["biz42"]}

    def hdong_list(self) -> list[tuple[str, str]]:
        return [(code, name) for code, name in self.codes["hdong43"]]


@lru_cache(maxsize=1)
def default_registry() -> SpecRegistry:
    return SpecRegistry.from_dir()
