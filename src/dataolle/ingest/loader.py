"""센터 CSV를 DuckDB 테이블로 적재.

- 모든 값이 큰따옴표인 문자열 CSV를 읽어, 명세서 Data Type(또는 컬럼명 규칙)에 맞춰 숫자로 변환
- 같은 데이터셋의 월별 파일을 한 테이블로 합침
- 원본은 디스크에 복사하지 않고 메모리(DuckDB in-memory)에서만 다룸
"""
from __future__ import annotations

import re
from pathlib import Path

from dataolle.spec import SpecRegistry, default_registry

_DOUBLE_PAT = re.compile(r"(_pop_cnt$|^average_|^lowest_temperature$|^highest_temperature$|rainfall|_speed$|_direction$|_rate$|^latitude$|^longitude$|_latitude$|_longitude$)")
_INT_PAT = re.compile(r"(_cnt$|_amt$|_count$|^amount$|^cnt$|_price$|^sale_amt$|_tot$|^employees_cnt$)")
_TEXT_EXCEPT = {"base_year_month", "base_date", "base_ymd", "time", "age", "emd_cd", "bjd_cd", "city_cd", "zip_code"}


def numeric_kind(spec_dtype: str | None, column: str) -> str | None:
    """'BIGINT' | 'DOUBLE' | None(문자열 유지)."""
    if column in _TEXT_EXCEPT or column.endswith("_time") or column.endswith("_cd") or column.endswith("_id"):
        return None
    if spec_dtype in ("INT", "BIGINT"):
        return "BIGINT"
    if spec_dtype == "DOUBLE":
        return "DOUBLE"
    if _DOUBLE_PAT.search(column):
        return "DOUBLE"
    if _INT_PAT.search(column):
        return "BIGINT"
    return None


def table_name(dataset_id: str) -> str:
    return "d_" + dataset_id.replace("-", "_")


def _sql_str(path: Path) -> str:
    return "'" + str(path).replace("\\", "/").replace("'", "''") + "'"


def load_dataset(con, dataset_id: str, files: list[Path], registry: SpecRegistry | None = None,
                 name: str | None = None) -> str:
    """파일들을 하나의 DuckDB 테이블로 적재하고 테이블 이름을 돌려준다."""
    registry = registry or default_registry()
    if not files:
        raise ValueError(f"{dataset_id}: 적재할 파일이 없습니다.")
    spec = registry.get(dataset_id)
    name = name or table_name(dataset_id)
    file_list = "[" + ", ".join(_sql_str(Path(f)) for f in files) + "]"
    src = (f"read_csv({file_list}, header=true, all_varchar=true, quote='\"', nullstr='NA', "
           f"union_by_name=true, filename=true)")
    present = {r[0].lower() for r in con.execute(f"DESCRIBE SELECT * FROM {src}").fetchall()}
    selects = []
    for col in spec.csv_columns:
        if col not in present:
            continue
        kind = numeric_kind(spec.dtype(col), col)
        q = f'"{col}"'
        if kind == "DOUBLE":
            selects.append(f"TRY_CAST({q} AS DOUBLE) AS {q}")
        elif kind == "BIGINT":
            selects.append(f"CAST(ROUND(TRY_CAST({q} AS DOUBLE)) AS BIGINT) AS {q}")
        else:
            selects.append(f"NULLIF(TRIM({q}), '') AS {q}")
    selects.append("filename AS _source_file")
    selects.append(f"'{dataset_id}' AS _ds")               # 같은 지표를 여러 판(v01·v02)으로 합칠 때 출처 구분용
    con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT {', '.join(selects)} FROM {src}")
    return name
