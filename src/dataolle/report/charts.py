"""보고서용 차트(PNG). 반출 기준상 시각화는 그림파일(png, jpg)이어야 하고, 그림 안의 수치도 기준을 따른다.

- 가린 칸(x)은 막대·점을 그리지 않는다(높이로 값이 드러나지 않게).
- 숫자 라벨은 검수를 통과한 값만, 그것도 선택적으로(최댓값·최솟값·강조 항목).
- 색: 검증된 기본 팔레트(dataviz reference palette, light). 한 계열은 slot 1 한 색, 비교 대상 강조는 파랑+회색.
- 가상데이터면 '가상데이터 · 시연용' 워터마크.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from dataolle.analysis.engine import AnalysisResult  # noqa: E402
from dataolle.analysis.fmt import fmt_month, fmt_value  # noqa: E402
from dataolle.sdc.engine import AggKind  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SURFACE = "#ffffff"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
DIMMED = "#cfd8e3"          # 강조하지 않는 막대(파랑 계열의 아주 옅은 톤)
LABEL_ON = {"#eda100": INK}  # 노랑 위 글자는 검정

_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\malgun.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]
_font_ready = False


_BOLD_SIBLINGS = {r"C:\Windows\Fonts\malgun.ttf": r"C:\Windows\Fonts\malgunbd.ttf",
                  "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc": "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"}


def setup_font() -> str | None:
    global _font_ready
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                bold = _BOLD_SIBLINGS.get(path)
                if bold and os.path.exists(bold):
                    font_manager.fontManager.addfont(bold)        # 제목 굵은 글씨용
                font_manager.fontManager.addfont(path)
                name = font_manager.FontProperties(fname=path).get_name()
                plt.rcParams["font.family"] = [name, "DejaVu Sans"]
                plt.rcParams["axes.unicode_minus"] = False
                _font_ready = True
                return name
            except Exception:
                continue
    plt.rcParams["axes.unicode_minus"] = False
    return None


def _compact(v: float, unit: str) -> str:
    if unit == "원":
        a = abs(v)
        if a >= 1e8:
            return f"{v / 1e8:,.1f}억"
        if a >= 1e4:
            return f"{v / 1e4:,.0f}만"
        return f"{v:,.0f}"
    if unit in ("%", "지수"):
        return f"{v:,.0f}" if unit == "지수" else f"{v:,.0f}%"
    return f"{v:,.0f}"


def _axis_formatter(values, unit: str) -> FuncFormatter:
    """축 눈금은 한 가지 단위로(0, 5,000만, 1.0억처럼 섞이지 않게)."""
    arr = np.asarray(values, dtype=float)
    vmax = float(np.nanmax(np.abs(arr))) if np.isfinite(arr).any() else 0.0
    if unit in ("%", "%p"):
        return FuncFormatter(lambda v, _: f"{v:,.0f}%" if abs(v - round(v)) < 1e-9 else f"{v:,.1f}%")
    if unit in ("지수", "위"):
        return FuncFormatter(lambda v, _: f"{v:,.0f}")
    scale, suffix = (1e8, "억") if vmax >= 1e8 else ((1e4, "만") if vmax >= 1e5 else (1, ""))

    def f(v, _):
        x = v / scale
        txt = f"{x:,.0f}" if abs(x - round(x)) < 1e-9 else f"{x:,.1f}"
        return "0" if v == 0 else txt + suffix
    return FuncFormatter(f)


def _time_labels(index) -> list[str]:
    """YYYYMM → '1월', 첫 눈금과 1월에는 연도를 함께('2025\n1월')."""
    out, prev_year = [], None
    for i in index:
        s = str(i)
        if len(s) == 6 and s.isdigit():
            y, m = s[:4], int(s[4:])
            out.append(f"{y}\n{m}월" if (prev_year is None or y != prev_year) else f"{m}월")
            prev_year = y
        elif len(s) == 8 and s.isdigit():               # 일·주 단위(YYYYMMDD) → '10/14', 해가 바뀌면 연도 표기
            y, m, d = s[:4], int(s[4:6]), int(s[6:])
            out.append(f"{y}\n{m}/{d}" if (prev_year is None or y != prev_year) else f"{m}/{d}")
            prev_year = y
        elif "년 " in s:                       # '2025년 3분기'
            y, rest = s.split("년 ", 1)
            out.append(f"{y}\n{rest}" if y != prev_year else rest)
            prev_year = y
        else:
            out.append(s)
    return out


def _style(ax, horizontal: bool = False):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    if horizontal:
        ax.spines["left"].set_visible(False)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=8.5, length=0)
    ax.grid(axis="x" if horizontal else "y", color=GRID, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)


def _decorate(fig, res: AnalysisResult, title: str | None, unit_note: str | None = None):
    fig.patch.set_facecolor(SURFACE)
    fig.text(0.012, 0.975, title or res.title, ha="left", va="top", fontsize=11, color=INK, fontweight="bold")
    unit = unit_note or (res.unit if res.unit not in ("지수", "위") else ("첫 기간=100" if res.unit == "지수" else "순위"))
    fig.text(0.012, 0.905, f"단위: {unit} · 기간: {res.period_text}", ha="left", va="top", fontsize=8, color=MUTED)
    status = "반출 기준 검수 통과" if res.check.passed else "검수 보완 필요"
    if res.check.masked_cells:
        status += f" · x={res.check.masked_cells}칸 비공개"
    cites = res.citations()
    cite = " / ".join(cites)
    if len(cite) > 105 and len(cites) > 1:                 # 출처가 길면 첫 자료 + 외 N종(전체는 보고서 부록 1)
        cite = f"{cites[0]} 외 {len(cites) - 1}종(전체 출처는 부록 1)"
    fig.text(0.012, 0.012, f"출처: {cite}", ha="left", va="bottom", fontsize=7, color=MUTED)
    fig.text(0.988, 0.012, status, ha="right", va="bottom", fontsize=7, color=MUTED)
    if res.synthetic:
        fig.text(0.5, 0.5, "가상데이터 · 시연용", ha="center", va="center", rotation=18, fontsize=34, color=INK, alpha=0.06,
                 fontweight="bold")


def _labels(index) -> list[str]:
    return [fmt_month(str(i)) if (len(str(i)) == 6 and str(i).isdigit()) else str(i) for i in index]


def choose_kind(res: AnalysisResult) -> str:
    if res.spec.chart != "auto":
        return res.spec.chart
    rd = res.rows_dim
    if rd is not None and rd.kind == "time" and len(res.table) >= 3 and res.table.shape[1] <= 6:
        return "line"
    if res.table.shape[1] > 1:
        return "stacked" if (res.spec.transform == "share" and res.table.shape[1] <= 8 and len(res.table) <= 6) else "heatmap"
    n = len(res.table)
    longest = max((len(str(i)) for i in res.table.index), default=0)
    return "hbar" if (n > 7 or longest > 6) else "bar"


def _legend_rows(labels, width: float) -> tuple[int, int]:
    """범례를 한 줄에 둘지 두 줄로 나눌지(라벨 길이로 폭을 어림)."""
    est = sum(0.45 + 0.13 * len(str(l)) for l in labels)
    if est <= width * 0.9:
        return 1, len(labels)
    return 2, int(np.ceil(len(labels) / 2))


def render_chart(res: AnalysisResult, path: str | Path, kind: str | None = None, highlight: list[str] | None = None,
                 title: str | None = None, width: float = 7.4, dpi: int = 200, ref_line: float | None = None,
                 ref_label: str = "", event: str | None = None, event_label: str = "", unit_note: str | None = None,
                 groups: dict | None = None, ref_x: float | None = None, ref_y: float | None = None,
                 x_label: str = "", y_label: str = "", y_unit: str | None = None) -> Path | None:
    """분석 결과를 PNG로. 그릴 게 없으면 None."""
    if not _font_ready:
        setup_font()
    kind = kind or choose_kind(res)
    if kind == "none" or res.table.empty:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "scatter":
        return _scatter_chart(res, path, title, width, dpi, unit_note, groups or {}, ref_x, ref_y, x_label, y_label,
                              y_unit or res.unit)
    t = res.table
    highlight = set(highlight or [])
    dropped_other = False
    if kind in ("bar", "hbar") and "기타" in t.index and len(t) > 1:
        t = t.drop(index="기타")          # '기타'(상위 외 합계)는 막대 하나가 축을 독차지하므로 그림에서 제외, 표에는 남김
        dropped_other = True
    if kind == "hbar":
        height = min(6.2, 1.25 + 0.32 * len(t))
    elif kind == "heatmap":
        height = min(6.4, 1.6 + 0.34 * len(t))
    else:
        height = 3.5
    has_legend = kind == "stacked" or (kind == "line" and t.shape[1] > 1)
    leg_rows, leg_cols = _legend_rows(t.columns, width) if has_legend else (0, 0)
    if kind == "stacked":
        height = max(height, 1.9 + 0.42 * len(t) + 0.2 * leg_rows)
    fig = plt.figure(figsize=(width, height), dpi=dpi)
    top = 1 - (0.62 + 0.3 * leg_rows) / height
    longest = max((len(l) for l in _labels(t.index)), default=4)
    side = min(0.34, max(0.1, 0.035 + 0.0175 * longest)) if kind in ("hbar", "stacked", "heatmap") else 0.1
    right = 0.86 if kind != "hbar" else 0.94 - side
    ax = fig.add_axes([side, 0.42 / height + 0.05, right if kind != "stacked" else 0.96 - side, top - 0.42 / height - 0.08])
    unit = res.unit
    if kind == "line":
        _line(ax, t, unit, leg_cols)
        if event is not None and event in [str(i) for i in t.index]:
            xe = [str(i) for i in t.index].index(event) - 0.5
            ax.axvline(xe, color=INK2, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
            ax.annotate(event_label or "사건 시작", (xe, 1.0), xycoords=("data", "axes fraction"), xytext=(4, -10),
                        textcoords="offset points", fontsize=7.5, color=INK2, va="top")
        if ref_line is not None:
            ax.axhline(ref_line, color=MUTED, linewidth=0.8, zorder=1)
    elif kind in ("bar", "hbar"):
        _bars(ax, t.iloc[:, 0], unit, horizontal=(kind == "hbar"), highlight=highlight, ref_line=ref_line, ref_label=ref_label)
    elif kind == "stacked":
        _stacked(ax, t, unit, leg_cols)
    elif kind == "heatmap":
        ax.set_position([side, 0.42 / height + 0.06, 0.86 - side, top - 0.42 / height - 0.1])
        _heatmap(fig, ax, t, unit)
    _decorate(fig, res, title, unit_note)
    if dropped_other:
        fig.text(0.988, 0.905, "상위 외 '기타'는 그림에서 제외(표에 포함)", ha="right", va="top", fontsize=7.5, color=MUTED)
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
    return path


def _line(ax, t: pd.DataFrame, unit: str, leg_cols: int = 4):
    _style(ax)
    x = np.arange(len(t))
    multi = t.shape[1] > 1
    for j, col in enumerate(t.columns):
        y = t[col].to_numpy(dtype=float)
        color = SERIES[j % len(SERIES)]
        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=4.5, markerfacecolor=color,
                markeredgecolor=SURFACE, markeredgewidth=1.2, label=str(col), zorder=3)
        if not multi:
            ok = ~np.isnan(y)
            if ok.any():
                for idx in {int(np.nanargmax(y)), int(np.nanargmin(y)), int(np.where(ok)[0][-1])}:
                    ax.annotate(fmt_value(y[idx], unit), (x[idx], y[idx]), textcoords="offset points", xytext=(0, 7),
                                ha="center", fontsize=8, color=INK2)
            for idx in np.where(~ok)[0]:
                ax.annotate("x", (x[idx], 0), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8, color=MUTED)
    step = max(1, int(np.ceil(len(x) / 13)))
    tl = _time_labels(t.index)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(tl[::step], fontsize=8)
    ax.yaxis.set_major_formatter(_axis_formatter(t.to_numpy(dtype=float), unit))
    arr = t.to_numpy(dtype=float)
    if unit == "지수" and np.isfinite(arr).any():
        # 지수(기준=100)는 0부터 그리면 선이 납작해져 차이가 안 보임 → 값 범위에 맞추되 100 기준선은 포함
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        lo, hi = min(lo, 100.0), max(hi, 100.0)
        pad = max((hi - lo) * 0.18, 2.0)
        ax.set_ylim(lo - pad, hi + pad)
    elif unit not in ("%",) or np.nanmin(arr) >= 0:
        ax.set_ylim(bottom=0, top=np.nanmax(arr) * 1.18 if np.isfinite(np.nanmax(arr)) else 1)
    if multi:
        ax.legend(frameon=False, fontsize=8, ncol=max(1, leg_cols), loc="lower left", bbox_to_anchor=(0, 1.0))
        last = len(x) - 1
        for j, col in enumerate(t.columns[:4]):            # 4개 이하 계열은 끝점에 이름도 붙임(색만으로 구분하지 않게)
            y = t[col].to_numpy(dtype=float)
            if not np.isnan(y[last]):
                ax.annotate(str(col), (x[last], y[last]), textcoords="offset points", xytext=(6, 0), va="center",
                            fontsize=7.5, color=INK2)
        ax.set_xlim(-0.4, len(x) - 1 + 0.9)


def _bars(ax, s: pd.Series, unit: str, horizontal: bool, highlight: set, ref_line: float | None = None,
          ref_label: str = ""):
    _style(ax, horizontal)
    labels = _labels(s.index)
    vals = s.to_numpy(dtype=float)
    pos = np.arange(len(s))
    emphasize = bool(highlight)
    colors = [SERIES[0] if (not emphasize or lab in highlight or str(idx) in highlight) else DIMMED
              for lab, idx in zip(labels, s.index)]
    plot_vals = np.nan_to_num(vals, nan=0.0)
    if horizontal:
        ax.barh(pos, plot_vals, height=0.62, color=colors, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.set_yticks(pos)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(_axis_formatter(vals, unit))
        vmax = np.nanmax(vals) if np.isfinite(vals).any() else 1
        vmin = np.nanmin(vals) if np.isfinite(vals).any() else 0
        if vmin < 0:                                   # 음수(감소)가 있으면 0 기준선을 두고 양쪽으로
            span = max(abs(vmin), abs(vmax))
            ax.set_xlim(min(vmin, 0) - span * 0.25, max(vmax, 0) + span * 0.25)
            ax.axvline(0, color=AXIS, linewidth=0.9, zorder=2)
        else:
            ax.set_xlim(0, vmax * 1.22)
    else:
        ax.bar(pos, plot_vals, width=0.62, color=colors, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.set_xticks(pos)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.yaxis.set_major_formatter(_axis_formatter(vals, unit))
        vmax = np.nanmax(vals) if np.isfinite(vals).any() else 1
        ax.set_ylim(0, vmax * 1.22)
    if ref_line is not None:
        if horizontal:
            ax.axvline(ref_line, color=INK2, linewidth=1, linestyle=(0, (4, 3)), zorder=4)
            ax.annotate(ref_label or f"기준 {ref_line:g}", (ref_line, -0.6), xytext=(3, 0), textcoords="offset points",
                        fontsize=7.5, color=INK2, va="center")
            fin = vals[np.isfinite(vals)]
            lo = min(float(fin.min()) if fin.size else 0.0, ref_line, 0.0)
            hi = max(float(fin.max()) if fin.size else 0.0, ref_line, 0.0)
            span = hi - lo if hi > lo else 1.0
            ax.set_xlim(lo - (span * 0.25 if lo < 0 else 0), hi + span * 0.22)
        else:
            ax.axhline(ref_line, color=INK2, linewidth=1, linestyle=(0, (4, 3)), zorder=4)
            ax.annotate(ref_label or f"기준 {ref_line:g}", (len(vals) - 0.5, ref_line), xytext=(0, 3),
                        textcoords="offset points", fontsize=7.5, color=INK2, ha="right")
    # 선택적 라벨: 최댓값, 강조 항목, 가린 칸 표시
    ok = ~np.isnan(vals)
    pick = set()
    if ok.any():
        pick.add(int(np.nanargmax(vals)))
        if np.nanmin(vals) < 0:                         # 감소(음수) 막대는 가장 크게 줄어든 항목에도 값 표시
            pick.add(int(np.nanargmin(vals)))
    pick |= {i for i, (lab, idx) in enumerate(zip(labels, s.index)) if lab in highlight or str(idx) in highlight}
    if len(vals) <= 8:
        pick |= set(np.where(ok)[0].tolist())
    for i in range(len(vals)):
        if np.isnan(vals[i]):
            txt, color = "x", MUTED
        elif i in pick:
            txt, color = fmt_value(vals[i], unit), INK2
        else:
            continue
        if horizontal:
            neg = plot_vals[i] < 0
            ax.annotate(txt, (plot_vals[i], pos[i]), textcoords="offset points", xytext=(-4 if neg else 4, 0), va="center",
                        ha="right" if neg else "left", fontsize=8, color=color)
        else:
            ax.annotate(txt, (pos[i], plot_vals[i]), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8, color=color)


def _stacked(ax, t: pd.DataFrame, unit: str, leg_cols: int = 5):
    _style(ax, horizontal=True)
    labels = _labels(t.index)
    pos = np.arange(len(t))
    left = np.zeros(len(t))
    for j, col in enumerate(t.columns):
        vals = np.nan_to_num(t[col].to_numpy(dtype=float), nan=0.0)
        color = SERIES[j % len(SERIES)]
        ax.barh(pos, vals, left=left, height=0.62, color=color, edgecolor=SURFACE, linewidth=1.5, label=str(col), zorder=3)
        for i, v in enumerate(vals):
            if v >= 9:
                ax.text(left[i] + v / 2, pos[i], f"{v:.0f}%", ha="center", va="center", fontsize=7.5,
                        color=LABEL_ON.get(color, "#ffffff"))
        left += vals
    ax.set_yticks(pos)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(frameon=False, fontsize=8, ncol=max(1, leg_cols), loc="lower left", bbox_to_anchor=(0, 1.0))
    if t.isna().any().any():
        ax.text(1.0, -0.12, "x: 비공개 칸은 그리지 않음", transform=ax.transAxes, ha="right", fontsize=7, color=MUTED)


def _scatter_chart(res: AnalysisResult, path: Path, title: str | None, width: float, dpi: int, unit_note: str | None,
                   groups: dict, ref_x: float | None, ref_y: float | None, x_label: str, y_label: str, y_unit: str) -> Path | None:
    """동네 지도: 첫 열 = 가로축, 둘째 열 = 세로축, 점 = 동네. 색 = 동네 유형(고정 순서), 기준선 = 제주 전체 값.
    가린 칸(x)이 있는 동네는 그리지 않는다."""
    t = res.table.iloc[:, :2].astype(float).dropna()
    if len(t) < 3:
        return None
    height = 4.6
    fig = plt.figure(figsize=(width, height), dpi=dpi)
    order = []
    for g in groups.values():
        if g not in order:
            order.append(g)
    from dataolle.analysis.semantic import AREA_GROUPS
    order = [g for g in AREA_GROUPS if g in order] + [g for g in order if g not in AREA_GROUPS]
    leg_rows = 1 if order else 0
    top = 1 - (0.62 + 0.3 * leg_rows) / height
    ax = fig.add_axes([0.1, 0.42 / height + 0.1, 0.86, top - 0.42 / height - 0.14])
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    xs, ys = t.iloc[:, 0].to_numpy(), t.iloc[:, 1].to_numpy()
    for k, g in enumerate(order or [None]):
        idx = [i for i, lab in enumerate(t.index) if groups.get(lab) == g] if g else list(range(len(t)))
        if not idx:
            continue
        c = SERIES[k % len(SERIES)]
        ax.scatter(xs[idx], ys[idx], s=46, color=c, edgecolor=SURFACE, linewidth=1.6, zorder=3, label=g)
    if ref_x is not None:
        ax.axvline(ref_x, color=INK2, linewidth=0.9, linestyle=(0, (4, 3)), zorder=2)
        ax.annotate("제주 전체", (ref_x, 1.0), xycoords=("data", "axes fraction"), xytext=(3, -9), textcoords="offset points",
                    fontsize=7.5, color=INK2, va="top")
    if ref_y is not None:
        ax.axhline(ref_y, color=INK2, linewidth=0.9, linestyle=(0, (4, 3)), zorder=2)
    # 이름표: 기준선에서 가장 멀리 떨어진 동네 위주로 최대 10곳. 글자 상자끼리 겹치면 왼쪽·아래로 옮겨 보고, 그래도 겹치면 건너뜀
    from matplotlib import patheffects
    cx = ref_x if ref_x is not None else float(np.median(xs))
    cy = ref_y if ref_y is not None else float(np.median(ys))
    sx, sy = (np.ptp(xs) or 1.0), (np.ptp(ys) or 1.0)
    dist = ((xs - cx) / sx) ** 2 + ((ys - cy) / sy) ** 2
    ax_w, ax_h = width * 0.86, height * (top - 0.42 / height - 0.14)          # 축 영역 크기(인치)
    boxes: list[tuple[float, float, float, float]] = []                      # 축 비율 좌표의 글자 상자
    halo = [patheffects.withStroke(linewidth=2.6, foreground=SURFACE)]
    for i in np.argsort(-dist)[:18]:
        px, py = (xs[i] - xs.min() + sx * 0.12) / (sx * 1.31), (ys[i] - ys.min() + sy * 0.12) / (sy * 1.24)
        label = str(t.index[i])
        tw, th = (0.075 * len(label) + 0.05) / ax_w, 0.15 / ax_h             # 글자 폭·높이(축 비율)
        placed = False
        for ox, oy, ha in ((5, 3, "left"), (-5, 3, "right"), (5, -10, "left"), (-5, -10, "right")):
            x0 = px + ox / 72 / ax_w - (tw if ha == "right" else 0)
            y0 = py + oy / 72 / ax_h
            box = (x0, y0, x0 + tw, y0 + th)
            hit_pts = any(x0 - 0.008 < (xs[k] - xs.min() + sx * 0.12) / (sx * 1.31) < box[2] + 0.008 and
                          y0 - 0.01 < (ys[k] - ys.min() + sy * 0.12) / (sy * 1.24) < box[3] + 0.01 for k in range(len(xs)) if k != i)
            if box[2] > 1.0 or box[0] < 0 or hit_pts or any(not (box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3])
                                                         for b in boxes):
                continue
            boxes.append(box)
            ax.annotate(label, (xs[i], ys[i]), textcoords="offset points", xytext=(ox, oy), ha=ha, fontsize=7.5, color=INK2,
                        path_effects=halo, zorder=4)
            placed = True
            break
        if placed and len(boxes) >= 10:
            break
    ax.xaxis.set_major_formatter(_axis_formatter(xs, res.unit))
    ax.yaxis.set_major_formatter(_axis_formatter(ys, y_unit))
    padx, pady = sx * 0.12, sy * 0.12
    ax.set_xlim(xs.min() - padx, xs.max() + padx * 1.6)
    ax.set_ylim(ys.min() - pady, ys.max() + pady)
    if x_label:
        ax.set_xlabel(x_label, fontsize=8.5, color=INK2)
    if y_label:
        ax.set_ylabel(y_label, fontsize=8.5, color=INK2)
    if order:
        ax.legend(frameon=False, fontsize=8, ncol=len(order), loc="lower left", bbox_to_anchor=(0, 1.0), handletextpad=0.3)
    _decorate(fig, res, title, unit_note)
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
    return path


def _heatmap(fig, ax, t: pd.DataFrame, unit: str):
    vals = t.to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    vmin, vmax = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ)
    cmap.set_bad("#f0efec")
    im = ax.imshow(np.ma.masked_invalid(vals), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(t.shape[1]))
    ax.set_xticklabels([str(c) for c in t.columns], fontsize=8, color=INK2)
    ax.set_yticks(range(t.shape[0]))
    ax.set_yticklabels(_labels(t.index), fontsize=8, color=INK2)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    for (i, j), v in np.ndenumerate(vals):
        if np.isnan(v):
            ax.text(j, i, "x", ha="center", va="center", fontsize=7.5, color=MUTED)
    cax = fig.add_axes([0.88, ax.get_position().y0, 0.018, ax.get_position().height])
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, colors=MUTED, length=0)
    cb.ax.yaxis.set_major_formatter(_axis_formatter(vals, unit))


# ---------------------------------------------------------------------------
# 처방 설계(예산 배분 시뮬레이션) 그림
# ---------------------------------------------------------------------------
# 적립 방식별 색: 현행(A)은 비교 기준이라 회색, 나머지는 검증된 팔레트에서 고정 순서로(방식이 빠져도 색이 바뀌지 않게)
SCENARIO_COLORS = {"A": MUTED, "B": SERIES[1], "C": SERIES[0], "D": SERIES[2], "E": SERIES[4], "C′": SERIES[6]}


def _scenario_color(label: str, j: int) -> str:
    key = str(label).split(" ")[0].replace("안", "")
    return SCENARIO_COLORS.get(key, SERIES[j % len(SERIES)])


def render_burn_chart(res: AnalysisResult, path: str | Path, budget: float, exhaust: dict | None = None,
                      title: str | None = None, width: float = 7.4, dpi: int = 200, budget_label: str = "예산") -> Path | None:
    """적립 예산 소진 그림. res.table: 행=계획 월 끝(첫 행 '시작'), 열=적립 방식, 값=누적 적립 필요액(원).
    선이 예산(점선)을 넘는 곳이 예산 소진. exhaust: {열 이름: (x 위치, 라벨)}."""
    if not _font_ready:
        setup_font()
    t = res.table
    if t.empty:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height = 3.8
    fig = plt.figure(figsize=(width, height), dpi=dpi)
    ax = fig.add_axes([0.1, 0.17, 0.72, 0.58])
    _style(ax)
    x = np.arange(len(t))
    arr = t.to_numpy(dtype=float)
    vmax = float(max(np.nanmax(arr), budget))
    ends = []
    for j, col in enumerate(t.columns):
        c = _scenario_color(col, j)
        y = t[col].to_numpy(dtype=float)
        ax.plot(x, y, color=c, linewidth=2, marker="o", markersize=3.6, markerfacecolor=c, markeredgecolor=SURFACE,
                markeredgewidth=1, zorder=3, label=str(col))
        ends.append((y[-1], str(col)))
    # 끝점 이름(4개 이하) — 값이 가까우면 위아래로 벌림
    ends.sort()
    last_y = None
    for yv, name in ends:
        yy = yv if last_y is None or yv - last_y > vmax * 0.07 else last_y + vmax * 0.07
        ax.annotate(name.split(" ")[0], (x[-1], yv), xytext=(x[-1] + 0.25, yy), textcoords="data", va="center",
                    fontsize=7.5, color=INK2)
        last_y = yy
    ax.axhline(budget, color=INK2, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate(budget_label, (0, budget), xytext=(3, 4), textcoords="offset points", fontsize=7.8, color=INK2)
    for col, (xe, lab) in (exhaust or {}).items():
        j = list(t.columns).index(col) if col in t.columns else 0
        c = _scenario_color(col, j)
        ax.plot([xe], [budget], marker="X", markersize=9, color=c, markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=5)
        ax.annotate(lab, (xe, budget), xytext=(0, -14), textcoords="offset points", ha="center", va="top", fontsize=7.8,
                    color=INK, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in t.index], fontsize=8)
    ax.yaxis.set_major_formatter(_axis_formatter([vmax], "원"))
    ax.set_ylim(0, vmax * 1.12)
    ax.set_xlim(-0.3, len(x) - 1 + 1.3)
    ax.legend(frameon=False, fontsize=8, ncol=max(1, len(t.columns)), loc="lower left", bbox_to_anchor=(0, 1.0))
    _decorate(fig, res, title, unit_note="원(누적 적립액)")
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
    return path


def render_pair_bars(res: AnalysisResult, path: str | Path, title: str | None = None, width: float = 7.4, dpi: int = 200,
                     unit_note: str | None = None) -> Path | None:
    """묶은 가로 막대: 행=대상(원도심·전통시장…), 열=적립 방식 2~3개. 막대 수가 적어 모든 막대에 값을 붙인다."""
    if not _font_ready:
        setup_font()
    t = res.table
    if t.empty:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n, k = len(t), max(1, t.shape[1])
    height = min(6.4, 1.55 + 0.30 * n * k + 0.25)
    fig = plt.figure(figsize=(width, height), dpi=dpi)
    labels = [str(i) for i in t.index]
    longest = max((len(l) for l in labels), default=4)
    side = min(0.32, max(0.12, 0.035 + 0.0175 * longest))
    top = 1 - 0.92 / height
    ax = fig.add_axes([side, 0.42 / height + 0.05, 0.9 - side, top - 0.42 / height - 0.08])
    _style(ax, horizontal=True)
    bh = 0.8 / k
    arr = t.to_numpy(dtype=float)
    vmax = float(np.nanmax(arr)) if np.isfinite(arr).any() else 1.0
    unit = res.unit
    for j, col in enumerate(t.columns):
        vals = t[col].to_numpy(dtype=float)
        pos = np.arange(n) + (j - (k - 1) / 2) * bh
        c = _scenario_color(col, j)
        ax.barh(pos, np.nan_to_num(vals, nan=0.0), height=bh * 0.9, color=c, edgecolor=SURFACE, linewidth=1.2,
                label=str(col), zorder=3)
        for p, v in zip(pos, vals):
            txt = "x" if np.isnan(v) else fmt_value(v, unit)
            ax.annotate(txt, (0 if np.isnan(v) else v, p), textcoords="offset points", xytext=(4, 0), va="center",
                        fontsize=7.6, color=INK2)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, vmax * 1.25)
    ax.xaxis.set_major_formatter(_axis_formatter(arr, unit))
    ax.legend(frameon=False, fontsize=8, ncol=k, loc="lower left", bbox_to_anchor=(0, 1.0))
    _decorate(fig, res, title, unit_note)
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
    return path


def render_response_chart(res: AnalysisResult, path: str | Path, title: str | None = None, window: tuple | None = None,
                          width: float = 7.4, dpi: int = 200) -> Path | None:
    """적립률 반응 그림: 달별 하루 평균 사용액(실제) vs 추정 반응으로 그린 예측, 아래 칸에 그 달 적립률.
    res.table: 행=연월(YYYY.MM), 열='실제', '예측(순수 반응)', '적립률'. window: 높은 적립률 기간(행 이름 둘)."""
    if not _font_ready:
        setup_font()
    t = res.table
    if t.empty or "실제" not in t.columns:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(width, 4.3), dpi=dpi)
    ax = fig.add_axes([0.1, 0.36, 0.86, 0.43])
    ax2 = fig.add_axes([0.1, 0.12, 0.86, 0.16], sharex=ax)
    _style(ax)
    _style(ax2)
    x = np.arange(len(t))
    labels = [str(i) for i in t.index]
    if window and window[0] in labels and window[1] in labels:
        a, b = labels.index(window[0]), labels.index(window[1])
        for axx in (ax, ax2):
            axx.axvspan(a - 0.5, b + 0.5, color=GRID, alpha=0.6, zorder=0, linewidth=0)
        ax.annotate("높은 적립률 기간", ((a + b) / 2, 1.0), xycoords=("data", "axes fraction"), ha="center", va="bottom",
                    fontsize=7.5, color=INK2)
    act = t["실제"].to_numpy(dtype=float)
    pred = t["예측(순수 반응)"].to_numpy(dtype=float)
    ax.plot(x, act, color=SERIES[0], linewidth=2, marker="o", markersize=3.6, label="실제", zorder=3)
    ax.plot(x, pred, color=SERIES[1], linewidth=1.6, linestyle=(0, (4, 2)), marker="s", markersize=3, label="예측(순수 반응)", zorder=3)
    vmax = float(np.nanmax(np.r_[act, pred]))
    ax.yaxis.set_major_formatter(_axis_formatter([vmax], "원"))
    ax.set_ylim(0, vmax * 1.15)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.04))
    plt.setp(ax.get_xticklabels(), visible=False)
    if "적립률" in t.columns:
        r = t["적립률"].to_numpy(dtype=float)
        ax2.step(x, r, where="mid", color=INK2, linewidth=1.5)
        for k, (xi, ri) in enumerate(zip(x, r)):
            if k == 0 or ri != r[k - 1]:                      # 적립률이 바뀌는 달에만 표시
                ax2.annotate(f"{ri:g}%", (xi, ri), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=6.5, color=INK2)
        ax2.set_ylim(0, max(r) * 1.45)
        ax2.set_yticks([])
        ax2.annotate("적립률", (0, 1.0), xycoords="axes fraction", xytext=(-4, 2), textcoords="offset points", ha="right",
                     va="bottom", fontsize=7.5, color=INK2)
    step = 1 if len(labels) <= 12 else (2 if len(labels) <= 24 else 3)
    ax2.set_xticks(x[::step])
    ax2.set_xticklabels([f"{l[2:4]}.{l[5:]}" if len(labels) > 12 else l for l in labels[::step]], fontsize=7, rotation=0)
    _decorate(fig, res, title, unit_note="원(하루 평균 탐나는전 사용액)")
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
    return path
