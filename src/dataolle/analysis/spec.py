"""분석 계획(AnalysisSpec): LLM 또는 규칙이 만드는 '무엇을 어떻게 셀지'의 설계도."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from dataolle.analysis.semantic import MODELS


class AnalysisSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str = Field(description="시맨틱 모델 키: card, card_week, card_time, stores, pos, pos_cust, pos_menu, pop")
    measure: str = Field(description="지표 키(모델별 measures)")
    rows: str | None = Field(None, description="행으로 나눌 차원 키")
    cols: str | None = Field(None, description="열로 나눌 차원 키(선택)")
    filters: dict[str, list[str]] = Field(default_factory=dict, description="차원 키 → 포함할 값 목록")
    period: list[str] | None = Field(None, description="[시작 YYYYMM, 끝 YYYYMM]")
    transform: Literal["none", "share", "yoy", "index", "rank"] = "none"
    top_n: int | None = Field(None, ge=1, le=30)
    totals: bool = True
    chart: Literal["auto", "line", "bar", "hbar", "stacked", "heatmap", "none"] = "auto"
    title: str | None = None
    subject: str | None = Field(None, description="문장 주어(예: '애월읍 카페·음료점'). 없으면 필터로 자동 작성")

    @field_validator("period")
    @classmethod
    def _period(cls, v):
        if v is None:
            return v
        if len(v) != 2 or not all(len(x) == 6 and x.isdigit() for x in v):
            raise ValueError("period는 ['YYYYMM', 'YYYYMM'] 형식이어야 합니다.")
        return sorted(v)

    def problems(self) -> list[str]:
        """시맨틱 모델 기준 검증. 빈 목록이면 실행 가능."""
        out = []
        model = MODELS.get(self.dataset)
        if model is None:
            return [f"알 수 없는 데이터셋 키: {self.dataset}"]
        if self.measure not in model.measures:
            out.append(f"{model.title}에 없는 지표: {self.measure}")
        for k in (self.rows, self.cols, *self.filters.keys()):
            if k and k not in model.dims:
                out.append(f"{model.title}에 없는 차원: {k}")
        if self.rows and self.cols and self.rows == self.cols:
            out.append("행과 열에 같은 차원을 쓸 수 없습니다.")
        if self.transform == "yoy" and not (self.rows in ("month", "quarter", "year") or self.cols in ("month", "quarter", "year")):
            out.append("전년 대비 증감률은 기간 차원(월·분기·연)이 있어야 합니다.")
        if self.cols and not self.rows:
            out.append("열 차원만 있고 행 차원이 없습니다.")
        return out
