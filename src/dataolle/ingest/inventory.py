"""받은 파일 목록(인벤토리)과 다운로드 계획(체크리스트).

센터 데이터는 한 번에 1개월씩만 조회·다운로드할 수 있고, 여러 달은 운영자가
바탕화면 '요청파일' 폴더에 월별로 넣어 준다(게시판 공식 답변). 그래서
- 폴더를 훑어 어떤 데이터의 몇 월이 있는지,
- 계획 대비 빠진 달과 중복 파일이 있는지,
- 운영자에게 미리 요청할 목록
을 만들어 준다.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from dataolle.ingest.recognizer import Recognition, recognize
from dataolle.spec import SpecRegistry, default_registry


def month_range(start: str, end: str) -> list[str]:
    """'202501', '202512' → ['202501', ..., '202512']."""
    y, m = int(start[:4]), int(start[4:6])
    ey, em = int(end[:4]), int(end[4:6])
    out = []
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}{m:02d}")
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


@dataclass
class Inventory:
    files: list[Recognition]
    by_dataset: dict[str, dict[str, list[Path]]] = field(default_factory=dict)

    def months(self, dataset_id: str) -> list[str]:
        return sorted(self.by_dataset.get(dataset_id, {}))

    def files_for(self, dataset_id: str, months: list[str] | None = None) -> list[Path]:
        """데이터셋의 파일 목록(중복 월은 가장 최근 다운로드 하나만)."""
        chosen: list[Path] = []
        seen: set[Path] = set()
        for ym, paths in sorted(self.by_dataset.get(dataset_id, {}).items()):
            if months and ym not in months:
                continue
            p = sorted(paths, key=lambda x: x.name)[-1]
            if p not in seen:
                chosen.append(p)
                seen.add(p)
        return chosen

    def duplicates(self) -> list[tuple[str, str, list[Path]]]:
        out = []
        for ds, months in self.by_dataset.items():
            for ym, paths in months.items():
                uniq = sorted(set(paths))
                if len(uniq) > 1:
                    out.append((ds, ym, uniq))
        return out

    def missing(self, dataset_id: str, start: str, end: str) -> list[str]:
        have = set(self.months(dataset_id))
        return [ym for ym in month_range(start, end) if ym not in have]

    @property
    def unknown(self) -> list[Recognition]:
        return [r for r in self.files if r.status == "unknown"]

    @property
    def ambiguous(self) -> list[Recognition]:
        return [r for r in self.files if r.status == "ambiguous"]


def scan_folder(folder: str | Path, recursive: bool = True, registry: SpecRegistry | None = None) -> Inventory:
    registry = registry or default_registry()
    folder = Path(folder)
    pattern = "**/*.csv" if recursive else "*.csv"
    recs = [recognize(p, registry) for p in sorted(folder.glob(pattern)) if p.is_file()]
    by_ds: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for r in recs:
        if r.dataset_id:
            for ym in r.months or ["(기간없음)"]:
                by_ds[r.dataset_id][ym].append(r.path)
    return Inventory(recs, {k: dict(v) for k, v in by_ds.items()})


# ---------------------------------------------------------------------------
# 다운로드 계획
# ---------------------------------------------------------------------------
@dataclass
class PlanItem:
    dataset_id: str
    month: str
    conditions: dict = field(default_factory=dict)
    status: str = "todo"             # todo | done | duplicate
    files: list[Path] = field(default_factory=list)


def make_plan(dataset_ids: list[str], start: str, end: str, conditions: dict | None = None) -> list[PlanItem]:
    return [PlanItem(ds, ym, dict(conditions or {})) for ds in dataset_ids for ym in month_range(start, end)]


def check_plan(plan: list[PlanItem], inv: Inventory) -> list[PlanItem]:
    for item in plan:
        paths = sorted(set(inv.by_dataset.get(item.dataset_id, {}).get(item.month, [])))
        item.files = paths
        item.status = "todo" if not paths else ("duplicate" if len(paths) > 1 else "done")
    return plan


def plan_progress(plan: list[PlanItem]) -> tuple[int, int]:
    done = sum(1 for p in plan if p.status in ("done", "duplicate"))
    return done, len(plan)


def operator_request_text(plan: list[PlanItem], registry: SpecRegistry | None = None, requester: str = "") -> str:
    """운영자 사전 다운로드 요청 문안(게시판: 별도 요청 시 운영자가 미리 받아 '요청파일' 폴더에 넣어 줌)."""
    registry = registry or default_registry()
    grouped: dict[str, list[str]] = defaultdict(list)
    conds: dict[str, dict] = {}
    for p in plan:
        if p.status == "todo":
            grouped[p.dataset_id].append(p.month)
            conds[p.dataset_id] = p.conditions
    if not grouped:
        return "요청할 파일이 없습니다. 계획한 파일을 모두 받았어요."
    lines = ["[제주빅데이터센터 데이터 사전 다운로드 요청]"]
    if requester:
        lines.append(f"요청자: {requester}")
    for ds, months in grouped.items():
        spec = registry.get(ds)
        months = sorted(months)
        span = f"{months[0][:4]}.{months[0][4:]} ~ {months[-1][:4]}.{months[-1][4:]}" if len(months) > 1 else f"{months[0][:4]}.{months[0][4:]}"
        cond = ", ".join(f"{spec.label(k)}={v}" for k, v in conds.get(ds, {}).items()) or "조건 없음(전체)"
        lines.append(f"- {ds} {spec.name_ko}: {span} ({len(months)}개월, 월별 파일) / 조건: {cond}")
    lines.append("받은 파일은 바탕화면 '요청파일' 폴더에 넣어 주시면 에이전트가 자동으로 인식합니다.")
    return "\n".join(lines)
