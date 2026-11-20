"""사실(fact) 저장소: 보고서 문장에 들어가는 모든 숫자는 여기서만 나온다.

LLM은 숫자를 직접 쓰지 않고 {{f3}}, {{f3.dir}} 같은 자리표시자만 쓴다.
코드가 자리표시자를 실제 값으로 채우므로 '없는 숫자'가 보고서에 들어갈 수 없다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from dataolle.analysis.fmt import fmt_value
from dataolle.analysis.josa import josa
from dataolle.sdc.engine import AggKind

PLACEHOLDER = re.compile(r"\{\{\s*(f\d+)(?:\.(dir|abs))?\s*\}\}")
# 값이 채워진 뒤에야 받침을 알 수 있으므로 템플릿은 '(으)로', '(이)가' 같은 표시를 쓰고 렌더링 때 고른다.
_JOSA_MARK = re.compile(r"(\S+?)\((으|이|은|을|과)\)(로|가|는|를|와)")
_MARK_TO_J = {("으", "로"): "으로", ("이", "가"): "이", ("은", "는"): "은", ("을", "를"): "을", ("과", "와"): "과"}


def apply_josa(text: str) -> str:
    """'9,504원(으)로' → '9,504원으로', '1.5배(으)로' → '1.5배로'."""
    def rep(m):
        j = _MARK_TO_J.get((m.group(2), m.group(3)))
        return josa(m.group(1), j) if j else m.group(0)
    return _JOSA_MARK.sub(rep, text)


@dataclass
class Fact:
    id: str
    label: str
    value: float | None
    unit: str
    kind: AggKind
    source: str = ""
    masked: bool = False
    is_change: bool = False          # 증감률·차이처럼 방향이 있는 값

    @property
    def text(self) -> str:
        if self.masked or self.value is None:
            return "비공개"
        return fmt_value(self.value, self.unit)

    @property
    def abs_text(self) -> str:
        if self.masked or self.value is None:
            return "비공개"
        return fmt_value(abs(self.value), self.unit)

    @property
    def direction(self) -> str:
        if self.masked or self.value is None:
            return "변화"
        if abs(self.value) < 0.05:
            return "변화 없음"
        return "증가" if self.value > 0 else "감소"

    def min_egress_level(self) -> int:
        """이 값을 외부 AI에 보내려면 필요한 최소 전송 레벨(0=보낼 수 없음 기준). 응용집계=1, 단순집계=2."""
        return 1 if self.kind == AggKind.APPLIED else 2


@dataclass
class FactStore:
    facts: dict[str, Fact] = field(default_factory=dict)

    def add(self, label: str, value, unit: str, kind: AggKind, source: str = "", masked: bool = False,
            is_change: bool = False) -> Fact:
        fid = f"f{len(self.facts) + 1}"
        v = None if value is None else float(value)
        f = Fact(fid, label, v, unit, kind, source, masked or v is None, is_change)
        self.facts[fid] = f
        return f

    def get(self, fid: str) -> Fact:
        return self.facts[fid]

    def __iter__(self):
        return iter(self.facts.values())

    def __len__(self):
        return len(self.facts)

    def render(self, text: str) -> str:
        """자리표시자를 실제 값으로 채운다. 없는 id는 '확인 불가'로."""
        def rep(m):
            f = self.facts.get(m.group(1))
            if f is None:
                return "확인 불가"
            if m.group(2) == "dir":
                return f.direction
            if m.group(2) == "abs":
                return f.abs_text
            return f.text
        return apply_josa(PLACEHOLDER.sub(rep, text))

    def table_for_prompt(self, level: int) -> list[dict]:
        """외부 AI에 줄 사실 목록. 레벨이 허용하지 않는 값은 숫자 없이 이름·단위만."""
        rows = []
        for f in self:
            item = {"id": f.id, "설명": f.label, "단위": f.unit, "유형": f.kind.value}
            if not f.masked and level >= f.min_egress_level():
                item["값"] = f.text
            else:
                item["값"] = "(로컬에서 채움)"
            if f.is_change:
                item["방향표기"] = f"{{{{{f.id}.dir}}}}"
            rows.append(item)
        return rows
