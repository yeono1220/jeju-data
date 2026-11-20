"""분석 실행기: AnalysisSpec → SQL(DuckDB) → 집계표 → 반출 검수 → (응용집계 변환) → 사실(fact).

원자료와 1차 집계표(long)는 이 PC의 메모리 안에서만 쓰이고, 보고서·화면·외부 AI로 나가는 것은
반출 검수를 통과한 표(display)와 사실 저장소(FactStore)뿐이다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from dataolle.analysis.facts import FactStore
from dataolle.analysis.fmt import fmt_month, fmt_period, fmt_value
from dataolle.analysis.josa import josa
from dataolle.analysis.semantic import MODEL_OF_DATASET, MODELS, DatasetModel, Dim
from dataolle.analysis.spec import AnalysisSpec
from dataolle.ingest.inventory import Inventory, scan_folder
from dataolle.ingest.loader import load_dataset
from dataolle.sdc.engine import (MASK_TOKEN, Action, AggKind, CheckResult, Level, protect_average, protect_pivot,
                                 protect_series)

SYNTH_MARKER = "_가상데이터_안내.txt"
# 센터 밖 공개 자료(공공데이터포털 등)의 출처 기관: 데이터셋 이름 → 기관(출처 줄 '기관, 데이터셋(기간)')
PUBLIC_PROVIDERS: dict[str, str] = {}
TRANSFORM_LABEL = {"share": "구성비", "yoy": "전년 동기 대비 증감률", "index": "지수(첫 기간=100)", "rank": "순위"}


# ---------------------------------------------------------------------------
# 카탈로그: 이번 세션에 불러온 데이터
# ---------------------------------------------------------------------------
class Catalog:
    def __init__(self, con: duckdb.DuckDBPyConnection | None = None):
        self.con = con or duckdb.connect()
        self.tables: dict[str, str] = {}
        self.files: dict[str, list[Path]] = {}
        self.synthetic = False
        self.inventory: Inventory | None = None
        self.public: dict = {}                    # 센터 밖 공공데이터(현재 코스에서는 쓰지 않음, 확장용)
        self.public_problems: list[str] = []

    @classmethod
    def from_folder(cls, folder: str | Path, con=None) -> "Catalog":
        cat = cls(con)
        cat.load_inventory(scan_folder(folder))
        return cat

    def load_inventory(self, inv: Inventory) -> list[str]:
        self.inventory = inv
        groups: dict[str, list[tuple[str, list[Path]]]] = {}
        for ds_id in sorted(inv.by_dataset):
            key = MODEL_OF_DATASET.get(ds_id)
            if not key:
                continue
            files = inv.files_for(ds_id)
            if files:
                groups.setdefault(key, []).append((ds_id, files))
        loaded = []
        for key, items in groups.items():
            if len(items) == 1:
                ds_id, files = items[0]
                self.tables[key] = load_dataset(self.con, ds_id, files, name=f"t_{key}")
            else:
                # 같은 지표의 여러 판(예: 유동인구 v01 2017~2025.4 + v02 2025.5~)을 열 이름 기준으로 이어 붙임(없는 열은 NULL)
                parts = [load_dataset(self.con, ds, fs, name=f"t_{key}__{ds.replace('-', '_')}") for ds, fs in items]
                self.con.execute(f"CREATE OR REPLACE TABLE t_{key} AS "
                                 + " UNION ALL BY NAME ".join(f"SELECT * FROM {t}" for t in parts))
                for t in parts:
                    self.con.execute(f"DROP TABLE {t}")
                self.tables[key] = f"t_{key}"
            files = [f for _, fs in items for f in fs]
            self.files[key] = files
            if any((Path(f).parent / SYNTH_MARKER).exists() for f in files):
                self.synthetic = True
            loaded.append(key)
        return loaded

    def has(self, key: str) -> bool:
        return key in self.tables or key in self.public

    def months(self, key: str) -> list[str]:
        model = MODELS[key]
        d = model.dim(model.period_dim)
        rows = self.con.execute(f"SELECT DISTINCT {d.expr} AS p FROM {self.tables[key]} ORDER BY 1").fetchall()
        return [str(r[0]) for r in rows if r[0] is not None]

    def values(self, key: str, dim: str, limit: int = 400) -> list[str]:
        """차원의 값 목록(법정동 이름·업종명 같은 공개 코드값). 외부 AI 안내에 써도 되는 정보."""
        model = MODELS[key]
        d = model.dim(dim)
        where = f" WHERE {' AND '.join(model.base_where)}" if model.base_where else ""
        rows = self.con.execute(f"SELECT DISTINCT {d.expr} AS v FROM {self.tables[key]}{where} ORDER BY 1 LIMIT {limit}").fetchall()
        vals = [str(r[0]) for r in rows if r[0] is not None]
        if d.order:
            pos = {v: i for i, v in enumerate(d.order)}
            vals.sort(key=lambda v: (pos.get(v, len(pos)), v))
        return vals


# ---------------------------------------------------------------------------
# 결과
# ---------------------------------------------------------------------------
@dataclass
class AnalysisResult:
    spec: AnalysisSpec
    model: DatasetModel
    title: str
    subject: str
    value_label: str
    unit: str
    kind: AggKind
    table: pd.DataFrame                  # 숫자표(가린 칸 = NaN), 합계 제외
    display: pd.DataFrame                # 보고서용 문자열표(가린 칸 'x', 합계 포함 가능)
    check: CheckResult
    sql: str
    params: list
    sources: list[str]
    period: tuple[str | None, str | None]
    notes: list[str] = field(default_factory=list)
    fact_ids: list[str] = field(default_factory=list)
    sentences: list[str] = field(default_factory=list)
    synthetic: bool = False
    dims_override: tuple | None = None                  # 파생표(비교 지역별 등)의 행·열 차원
    cite_parts: list[tuple[str, str | None, str | None]] | None = None   # 파생표의 (데이터셋 이름, 시작월, 끝월)
    method: str = ""                                    # 파생표 계산 방법(점검표용)
    part_files: dict[str, list[str]] | None = None      # 데이터셋 이름 → 사용 파일(여러 데이터셋을 쓴 표)

    @property
    def rows_dim(self) -> Dim | None:
        if self.dims_override is not None:
            return self.dims_override[0]
        return self.model.dims.get(self.spec.rows) if self.spec.rows else None

    @property
    def cols_dim(self) -> Dim | None:
        if self.dims_override is not None:
            return self.dims_override[1]
        return self.model.dims.get(self.spec.cols) if self.spec.cols else None

    @property
    def period_text(self) -> str:
        return fmt_period(*self.period)

    def source_parts(self) -> list[tuple[str, str | None, str | None, list[str]]]:
        """(센터 공식 데이터셋 이름, 시작월, 끝월, 사용 파일) 목록."""
        if self.cite_parts:
            if self.part_files:
                return [(n, a, b, list(self.part_files.get(n, []))) for n, a, b in self.cite_parts]
            files = list(self.sources) if len({n for n, _, _ in self.cite_parts}) == 1 else []
            return [(n, a, b, files) for n, a, b in self.cite_parts]
        from dataolle.spec import default_registry
        reg = default_registry()
        return [(reg.get(i).name_ko, self.period[0], self.period[1], list(self.sources)) for i in self.model.dataset_ids]

    @property
    def dataset_names(self) -> list[str]:
        """센터의 공식 데이터셋 이름(반출 신청서의 '원본데이터명'과 같아야 함)."""
        return list(dict.fromkeys(n for n, _, _, _ in self.source_parts()))

    def citations(self) -> list[str]:
        """반출 기준 출처 표기: 제주빅데이터센터, 데이터셋(이름, 기간). 센터 밖 공개 자료는 제공기관 이름으로."""
        return list(dict.fromkeys(cite_label(n, a, b) for n, a, b, _ in self.source_parts()))

    def citation(self) -> str:
        return " / ".join(self.citations())


def cite_label(name: str, a: str | None, b: str | None) -> str:
    return f"{PUBLIC_PROVIDERS.get(name, '제주빅데이터센터')}, {name}({fmt_period(a, b)})"


def actual_sources(cat, key: str, period: tuple[str | None, str | None] | None = None) -> list[tuple[str, str, str, list[str]]]:
    """보조 자료(key 묶음)에 든 데이터셋별 (공식 이름, 실제 첫 달, 실제 끝 달, 파일 이름).

    출처 줄은 심사자가 다운로드 기록과 맞춰 보는 곳이라, 분석 기간이 아니라 '실제로 불러온 달'을 적는다.
    period를 주면 그 기간과 겹치는 달만 적고, 겹치는 달이 없으면 불러온 달 전부를 적는다.
    유동인구처럼 여러 판(v01·v02)이 한 묶음이면 판마다 따로 돌려준다."""
    inv = getattr(cat, "inventory", None)
    if inv is None:
        return []
    from dataolle.spec import default_registry
    reg = default_registry()
    out = []
    for ds in sorted(inv.by_dataset):
        if MODEL_OF_DATASET.get(ds) != key:
            continue
        months = inv.months(ds)
        if not months:
            continue
        sel = months
        if period and period[0] and period[1]:
            sel = [m for m in months if str(period[0])[:6] <= m[:6] <= str(period[1])[:6]] or months
        files = [Path(f).name for f in inv.files_for(ds, sel)]
        out.append((reg.get(ds).name_ko, sel[0][:6], sel[-1][:6], files))
    return out


def add_actual_sources(res, cat, key: str, period: tuple[str | None, str | None] | None = None) -> None:
    """파생 결과에 보조 자료를 출처로 더한다(실제 기간·파일). 이미 같은 이름이 있으면 기간을 넓히고 파일을 합친다."""
    parts = list(res.cite_parts or [])
    pf = dict(res.part_files or {})
    for nm, a, b, files in actual_sources(cat, key, period):
        same = [i for i, p in enumerate(parts) if p[0] == nm]
        if same:                                      # 같은 데이터셋이 이미 있으면 한 줄로 합친다
            olds = [parts[i] for i in same]
            a = min([a] + [p[1] for p in olds if p[1]])
            b = max([b] + [p[2] for p in olds if p[2]])
            parts = [p for i, p in enumerate(parts) if i not in same]
        parts.append((nm, a, b))
        pf[nm] = list(dict.fromkeys(list(pf.get(nm, [])) + files))
    res.cite_parts = parts
    res.part_files = pf


def is_public(name: str) -> bool:
    return name in PUBLIC_PROVIDERS


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def _subject(spec: AnalysisSpec, model: DatasetModel) -> str:
    """문장 주어: 지역 → 그 밖의 조건 순서(예: '애월읍 비알콜 음료점업')."""
    if spec.subject:
        return spec.subject
    geo, other = [], []
    for k, vals in spec.filters.items():
        if not vals:
            continue
        text = "·".join(vals[:3]) + (f" 외 {len(vals) - 3}" if len(vals) > 3 else "")
        dim = model.dims.get(k)
        (geo if dim is not None and dim.kind == "geo" else other).append(text)
    parts = geo + other
    return " ".join(parts) if parts else "제주 전체"


def _order_index(values: pd.Series, dim: Dim) -> list:
    idx = list(values.index)
    if dim.order:
        pos = {v: i for i, v in enumerate(dim.order)}
        return sorted(idx, key=lambda v: (pos.get(v, len(pos) + (1 if v == "기타" else 0)), str(v)))
    if dim.kind == "time":
        return sorted(idx, key=str)
    rest = [v for v in idx if v != "기타"]
    ordered = values.reindex(rest).sort_values(ascending=False, na_position="last").index.tolist()
    return ordered + (["기타"] if "기타" in idx else [])


def _value(df: pd.DataFrame, model: DatasetModel, mkey: str) -> pd.Series:
    m = model.measure(mkey)
    if m.expr is not None:
        return df[m.key].astype(float)
    num, den = df[m.num].astype(float), df[m.den].astype(float)
    return (num / den.replace(0, np.nan)) * m.scale


def _month_minus(ym: str, months: int) -> str:
    y, m = int(ym[:4]), int(ym[4:6])
    t = y * 12 + (m - 1) - months
    return f"{t // 12:04d}{t % 12 + 1:02d}"


def run_analysis(spec: AnalysisSpec, catalog: Catalog, facts: FactStore | None = None,
                 level: Level = Level.CONSERVATIVE) -> AnalysisResult:
    problems = spec.problems()
    if problems:
        raise ValueError("; ".join(problems))
    model = MODELS[spec.dataset]
    if not catalog.has(model.key):
        raise ValueError(f"'{model.title}' 데이터가 아직 불러와지지 않았습니다.")
    facts = facts if facts is not None else FactStore()
    table = catalog.tables[model.key]
    measure = model.measure(spec.measure)
    bases = model.base_measures(spec.measure)
    dims = [model.dim(k) for k in (spec.rows, spec.cols) if k]

    where, params = list(model.base_where), []
    for k, vals in spec.filters.items():
        if vals:
            where.append(f"({model.dim(k).expr}) IN ({', '.join(['?'] * len(vals))})")
            params += list(vals)
    if spec.period:
        where.append(f"({model.dim(model.period_dim).expr}) BETWEEN ? AND ?")
        params += [spec.period[0], spec.period[1]]
    where_sql = f" WHERE {' AND '.join(where)}" if where else ""
    select = [f"{d.expr} AS d{i}" for i, d in enumerate(dims)] + [f"{b.expr} AS {b.key}" for b in bases]
    group = f" GROUP BY {', '.join(f'd{i}' for i in range(len(dims)))}" if dims else ""
    sql = f"SELECT {', '.join(select)} FROM {table}{where_sql}{group}"
    long = catalog.con.execute(sql, params).df()
    sup_key = measure.support
    if sup_key and len(long):
        # 여러 기간을 합치면 같은 사람이 기간마다 다시 세어짐 → 바탕 인원은 '기간별 합계의 최댓값'(실제 인원의 하한)으로 판정
        sup_m = model.measure(sup_key)
        pexpr = model.dim(model.period_dim).expr
        dsel = [f"{d.expr} AS d{i}" for i, d in enumerate(dims)]
        gcols = [f"d{i}" for i in range(len(dims))]
        inner = (f"SELECT {', '.join(dsel + [f'{pexpr} AS _p', f'{sup_m.expr} AS _s'])} FROM {table}{where_sql} "
                 f"GROUP BY {', '.join(gcols + ['_p'])}")
        outer = f"SELECT {', '.join(gcols + ['MAX(_s) AS _support'])} FROM ({inner}) GROUP BY {', '.join(gcols)}" if dims \
            else f"SELECT MAX(_s) AS _support FROM ({inner})"
        sup_df = catalog.con.execute(outer, params).df()
        long = long.merge(sup_df, on=gcols, how="left") if dims else long.assign(_support=sup_df["_support"].iloc[0])

    pdim = model.dim(model.period_dim)
    meta = catalog.con.execute(f"SELECT min({pdim.expr}), max({pdim.expr}) FROM {table}{where_sql}", params).fetchone()
    srcs = catalog.con.execute(f"SELECT DISTINCT _source_file FROM {table}{where_sql} ORDER BY 1", params).fetchall()
    sources = [Path(s[0]).name for s in srcs if s[0]]
    period = (str(meta[0])[:6] if meta and meta[0] else None, str(meta[1])[:6] if meta and meta[1] else None)

    notes = list(model.caveats)
    if any("월평동" in v for v in spec.filters.get("bjd", [])):
        notes.append("'월평동'은 제주시와 서귀포시에 모두 있어 시를 붙여 구분했습니다.")

    # 상위 N + 기타
    if dims and spec.top_n and dims[0].kind != "time":
        tot = long.groupby("d0")[[b.key for b in bases]].sum()
        avg_expr = measure.expr is not None and measure.kind == AggKind.AVERAGE
        score = (long.groupby("d0")[measure.key].mean() if avg_expr else _value(tot.reset_index(), model, spec.measure))
        if not avg_expr:
            score.index = tot.index
        keep = set(score.sort_values(ascending=False).head(spec.top_n).index)
        if len(score) > len(keep):
            gcols = [f"d{i}" for i in range(len(dims))]
            if avg_expr:
                long = long[long["d0"].isin(keep)].copy()
                notes.append(f"상위 {spec.top_n}개만 표시했습니다.")
            else:
                long["d0"] = long["d0"].where(long["d0"].isin(keep), "기타")
                agg = {b.key: "sum" for b in bases}
                if "_support" in long.columns:
                    agg["_support"] = "max"
                long = long.groupby(gcols, as_index=False).agg(agg)
                notes.append(f"상위 {spec.top_n}개 외 항목은 '기타'로 묶었습니다.")

    long["_value"] = _value(long, model, spec.measure)
    sup_col = "_support" if "_support" in long.columns else None
    value_label = measure.label
    unit = measure.unit

    # 피벗
    if not dims:
        values = pd.DataFrame({value_label: [long["_value"].iloc[0] if len(long) else np.nan]}, index=["전체"])
        support = pd.DataFrame({value_label: [long[sup_col].iloc[0] if sup_col and len(long) else np.nan]}, index=["전체"])
    elif len(dims) == 1:
        s = long.set_index("d0")["_value"]
        order = _order_index(s, dims[0])
        values = s.reindex(order).to_frame(value_label)
        support = (long.set_index("d0")[sup_col].reindex(order).to_frame(value_label) if sup_col else None)
    else:
        values = long.pivot_table(index="d0", columns="d1", values="_value", aggfunc="sum")
        support = long.pivot_table(index="d0", columns="d1", values=sup_col, aggfunc="max") if sup_col else None
        rows_order = _order_index(values.sum(axis=1, min_count=1), dims[0])
        cols_order = _order_index(values.sum(axis=0, min_count=1), dims[1])
        values = values.reindex(index=rows_order, columns=cols_order)
        if support is not None:
            support = support.reindex(index=rows_order, columns=cols_order)
    values.index.name = dims[0].label if dims else None
    if len(dims) == 2:
        values.columns.name = dims[1].label

    # 반출 검수
    title = spec.title or _auto_title(spec, model, value_label)
    if measure.kind == AggKind.AVERAGE:
        check = protect_average(values, support if sup_col else None, title, level)
    elif len(dims) == 2:
        check = protect_pivot(values.fillna(0), title, AggKind.SIMPLE, support.fillna(0) if support is not None else None,
                              level, show_totals=spec.totals)
    else:
        col = values.columns[0]
        check = protect_series(values[col].fillna(0), title, AggKind.SIMPLE,
                               support[col].fillna(0) if support is not None else None, level, show_total=spec.totals)
    body = check.table.copy()
    for lab in ("합계",):
        if lab in body.index:
            body = body.drop(index=lab)
        if lab in body.columns:
            body = body.drop(columns=lab)
    numeric = body.replace(MASK_TOKEN, np.nan).apply(pd.to_numeric, errors="coerce")
    numeric = numeric.where(~values.reindex_like(numeric).isna())          # 원래 없던 칸은 NaN 유지
    kind = measure.kind

    # 응용집계 변환
    if spec.transform != "none":
        numeric, unit, notes2 = _transform(spec, values, numeric, support, dims, unit)
        notes += notes2
        kind = AggKind.APPLIED
        value_label = f"{measure.label} {TRANSFORM_LABEL[spec.transform]}"

    display = _display(numeric, check, unit, kind, spec, values)
    res = AnalysisResult(spec, model, title, _subject(spec, model), value_label, unit, kind, numeric, display, check,
                         sql, params, sources, period, notes, synthetic=catalog.synthetic)
    if len(model.dataset_ids) > 1:
        # 여러 판을 이어 붙인 표: 실제로 쓰인 데이터셋별 기간·파일을 따로 적는다(반출 신청서 원본데이터명과 일치)
        from dataolle.spec import default_registry
        reg = default_registry()
        rows = catalog.con.execute(f"SELECT _ds, min({pdim.expr}), max({pdim.expr}) FROM {table}{where_sql} GROUP BY 1 ORDER BY 1",
                                   params).fetchall()
        fl = catalog.con.execute(f"SELECT DISTINCT _ds, _source_file FROM {table}{where_sql} ORDER BY 2", params).fetchall()
        res.cite_parts = [(reg.get(ds).name_ko, str(a)[:6] if a else None, str(b)[:6] if b else None) for ds, a, b in rows]
        res.part_files = {reg.get(ds).name_ko: [Path(f).name for d, f in fl if d == ds and f] for ds, _, _ in rows}
        if len({r[0] for r in rows}) > 1 and "4-10" in {r[0] for r in rows}:
            res.notes.append("2017년~2025년 4월은 유동인구 v01(2022년 6월까지 KT, 이후 SKT), 2025년 5월부터는 v02(SKT) 자료입니다. "
                             "통신사·추정 방식이 바뀌어 인구 '수준'은 끊길 수 있으므로 구성비(비율) 중심으로 해석하세요.")
    _make_facts(res, facts)
    return res


def mask_ratio(res: AnalysisResult) -> float:
    """표 본문에서 비공개(x) 칸의 비율(원래 빈 칸 제외)."""
    d = res.display
    body = d.drop(index=[i for i in d.index if i == "합계"], columns=[c for c in d.columns if c == "합계"])
    vals = body.to_numpy().ravel()
    shown = [v for v in vals if v != "-"]
    return (sum(1 for v in shown if v == MASK_TOKEN) / len(shown)) if shown else 0.0


def _rollup_step(spec: AnalysisSpec, model: DatasetModel) -> tuple[AnalysisSpec, str] | None:
    """한 단계 위로 묶기: 열(범주) 먼저, 그다음 행. 필터에 쓰인 차원은 그대로 둔다."""
    for slot in ("cols", "rows"):
        key = getattr(spec, slot)
        if not key:
            continue
        dim = model.dims.get(key)
        if dim is None or not dim.rollup or dim.rollup not in model.dims or key in spec.filters:
            continue
        up = model.dim(dim.rollup)
        other = spec.rows if slot == "cols" else spec.cols
        if up.key == other:
            continue
        upd = {slot: up.key}
        if spec.top_n and slot == "rows" and up.kind != dim.kind:
            upd["top_n"] = None
        return spec.model_copy(update=upd), f"{dim.label} → {up.label}"
    return None


def run_safe(spec: AnalysisSpec, catalog: Catalog, facts: FactStore | None = None, level: Level = Level.CONSERVATIVE,
             max_mask_ratio: float = 0.25, max_steps: int = 2) -> AnalysisResult:
    """비공개 칸이 많아 표가 비면 상위 단위로 묶어(범주처리) 다시 집계한다. 반출 기준이 인정하는 처리 방식."""
    model = MODELS[spec.dataset]
    cur, steps = spec, []
    trial = run_analysis(cur, catalog, FactStore(), level)
    for _ in range(max_steps):
        if mask_ratio(trial) <= max_mask_ratio:
            break
        nxt = _rollup_step(cur, model)
        if nxt is None:
            break
        cand, desc = nxt
        cand_res = run_analysis(cand, catalog, FactStore(), level)
        steps.append((desc, mask_ratio(trial), mask_ratio(cand_res)))
        cur, trial = cand, cand_res
    final = run_analysis(cur, catalog, facts, level) if (steps or facts is not None) else trial
    for desc, before, after in reversed(steps):
        final.check.actions.insert(0, Action("rollup", f"범주처리: {desc}(비공개 칸 {before:.0%} → {after:.0%})"))
    if steps:
        final.notes.append("비공개 칸이 많아 " + ", ".join(d for d, _, _ in steps) + "로 묶어 다시 집계했습니다(범주처리).")
    return final


def _auto_title(spec: AnalysisSpec, model: DatasetModel, value_label: str) -> str:
    subj = _subject(spec, model)
    parts = [subj]
    short = {"month": "월", "quarter": "분기", "year": "연도", "date": "일"}
    if spec.rows:
        parts.append(f"{short.get(spec.rows, model.dim(spec.rows).label)}별")
    if spec.cols:
        parts.append(f"× {short.get(spec.cols, model.dim(spec.cols).label)}")
    parts.append(value_label)
    if spec.transform != "none":
        parts.append(TRANSFORM_LABEL[spec.transform])
    return " ".join(parts)


def _transform(spec, values, numeric, support, dims, unit):
    notes = []
    if spec.transform == "share":
        if numeric.shape[1] > 1:
            totals = values.sum(axis=1)
            out = numeric.div(totals.replace(0, np.nan), axis=0) * 100
        else:
            total = values.iloc[:, 0].sum()
            out = numeric / total * 100 if total else numeric * np.nan
        return out, "%", notes
    if spec.transform == "rank":
        return numeric.rank(ascending=False, method="min"), "위", notes
    if spec.transform == "yoy":
        time_on_rows = spec.rows in ("month", "quarter", "year")
        frame = numeric if time_on_rows else numeric.T
        sup = None if support is None else (support if time_on_rows else support.T)
        out = pd.DataFrame(np.nan, index=frame.index, columns=frame.columns)
        lag = {"month": 12, "quarter": 4, "year": 1}[spec.rows if time_on_rows else spec.cols]
        idx = [str(i) for i in frame.index]
        found = False
        for i, t in enumerate(idx):
            if lag == 12:
                prev = _month_minus(t, 12)
            elif lag == 1:
                prev = str(int(t[:4]) - 1) + t[4:]
            else:
                prev = str(int(t[:4]) - 1) + t[4:]
            if prev in idx:
                j = idx.index(prev)
                cur, base = frame.iloc[i], frame.iloc[j]
                ok = cur.notna() & base.notna() & (base != 0)
                if sup is not None:
                    ok &= (sup.iloc[i] >= 10) & (sup.iloc[j] >= 10)
                out.iloc[i] = np.where(ok, (cur / base - 1) * 100, np.nan)
                found = True
        if not found:
            notes.append("전년 같은 기간 자료가 없어 증감률 대신 첫 기간=100 지수로 보여 줍니다.")
            first = frame.apply(lambda c: c.dropna().iloc[0] if c.notna().any() else np.nan)
            out = frame / first * 100
            return (out if time_on_rows else out.T), "지수", notes
        return (out if time_on_rows else out.T), "%", notes
    if spec.transform == "index":
        first = numeric.apply(lambda c: c.dropna().iloc[0] if c.notna().any() else np.nan)
        return numeric / first * 100, "지수", notes
    return numeric, unit, notes


def _display(numeric: pd.DataFrame, check: CheckResult, unit: str, kind: AggKind, spec: AnalysisSpec,
             values: pd.DataFrame) -> pd.DataFrame:
    disp = numeric.copy().astype(object)
    for r in range(numeric.shape[0]):
        for c in range(numeric.shape[1]):
            v = numeric.iat[r, c]
            original_missing = pd.isna(values.reindex_like(numeric).iat[r, c])
            disp.iat[r, c] = "-" if original_missing else (MASK_TOKEN if pd.isna(v) else fmt_value(v, unit))
    if kind != AggKind.APPLIED and spec.totals and kind == AggKind.SIMPLE:
        tbl = check.table
        if "합계" in tbl.columns and numeric.shape[1] > 1:
            disp["합계"] = [fmt_value(v, unit) if not isinstance(v, str) else v for v in tbl["합계"].reindex(disp.index)]
        if "합계" in tbl.index:
            row = tbl.loc["합계"]
            disp.loc["합계"] = [fmt_value(row[c], unit) if not isinstance(row[c], str) else row[c] for c in disp.columns]
    return disp


# ---------------------------------------------------------------------------
# 사실(fact)과 기본 문장
# ---------------------------------------------------------------------------
def _label_of(v) -> str:
    s = str(v)
    return fmt_month(s) if (len(s) == 6 and s.isdigit()) else s


def _make_facts(res: AnalysisResult, store: FactStore) -> None:
    """검수를 통과한 숫자만으로 사실을 만들고, 자리표시자가 들어간 기본 문장을 만든다."""
    t = res.table
    if t.empty:
        return
    subj, vl, unit, kind = res.subject, res.value_label, res.unit, res.kind
    rd = res.rows_dim
    add = lambda label, v, u=unit, k=kind, ch=False: store.add(label, v, u, k, res.title, is_change=ch)
    if t.shape[1] == 1 and rd is not None:
        s = t.iloc[:, 0].dropna()
        s = s[[i for i in s.index if i != "기타"]]
        if s.empty:
            return
        if rd.kind == "time" and len(s) >= 2:
            hi, lo = s.idxmax(), s.idxmin()
            f_hi = add(f"{subj} {vl}: 가장 높은 {rd.label}({_label_of(hi)})", s[hi])
            f_lo = add(f"{subj} {vl}: 가장 낮은 {rd.label}({_label_of(lo)})", s[lo])
            res.fact_ids += [f_hi.id, f_lo.id]
            res.sentences.append(f"{josa(subj + '의 ' + vl, '은')} {_label_of(hi)}에 가장 높았고({{{{{f_hi.id}}}}}), "
                                 f"{_label_of(lo)}에 가장 낮았습니다({{{{{f_lo.id}}}}}).")
            first, last = s.index[0], s.index[-1]
            if kind != AggKind.APPLIED and res.spec.rows in ("month", "quarter", "year") and s[first]:
                ch = (s[last] / s[first] - 1) * 100
                f_ch = add(f"{subj} {vl}: {_label_of(first)} 대비 {_label_of(last)} 변화율", ch, "%", AggKind.APPLIED, True)
                res.fact_ids.append(f_ch.id)
                res.sentences.append(f"{josa(_label_of(first), '과')} 비교하면 {_label_of(last)}에는 {{{{{f_ch.id}.abs}}}} "
                                     f"{{{{{f_ch.id}.dir}}}}했습니다.")
        else:
            top = s.idxmax()
            f_top = add(f"{subj} {vl}: 1위 {rd.label}({top})", s[top])
            res.fact_ids.append(f_top.id)
            quoted = "'" + str(top) + "'"
            sent = f"{rd.label}별로 보면 {josa(quoted, '이')} 가장 컸습니다({{{{{f_top.id}}}}})"
            if kind == AggKind.SIMPLE and res.spec.transform == "none" and "합계" in res.check.table.index:
                full_total = res.check.table.loc["합계"].iloc[0]
                if not isinstance(full_total, str) and full_total:
                    sh = s[top] / float(full_total) * 100
                    f_sh = add(f"{subj} {vl}: 1위 {rd.label}({top})의 비중", sh, "%", AggKind.APPLIED)
                    res.fact_ids.append(f_sh.id)
                    sent += f". 전체의 {{{{{f_sh.id}}}}}를 차지합니다"
            res.sentences.append(sent + ".")
            if len(s) >= 3:
                top3 = [str(x) for x in s.sort_values(ascending=False).index[:3]]
                res.sentences.append(f"상위 3개는 {', '.join(top3)} 순입니다.")
    elif t.shape[1] > 1 and rd is not None:
        stacked = t.drop(index=[i for i in t.index if i == "기타"]).stack().dropna()
        if stacked.empty:
            return
        (r, c) = stacked.idxmax()
        f = add(f"{subj} {vl}: 가장 큰 칸({_label_of(r)}·{c})", stacked.max())
        res.fact_ids.append(f.id)
        res.sentences.append(f"가장 큰 값은 {_label_of(r)}의 '{c}'입니다({{{{{f.id}}}}}).")
