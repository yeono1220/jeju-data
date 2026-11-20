"""한국어 조사 자동 선택(은/는, 이/가, 을/를, 과/와, 으로/로)."""
from __future__ import annotations

import re

_PAIRS = {"은": ("은", "는"), "는": ("은", "는"), "이": ("이", "가"), "가": ("이", "가"),
          "을": ("을", "를"), "를": ("을", "를"), "과": ("과", "와"), "와": ("과", "와"),
          "으로": ("으로", "로"), "로": ("으로", "로")}


def _last_hangul(word: str) -> str | None:
    w = word.rstrip(" )]}'\"’”")
    if w.endswith("%") or w.endswith("%p"):
        return "트"                      # 퍼센트, 퍼센트포인트
    for ch in reversed(re.sub(r"[\s\)\]\}'\"’”·.,%]+$", "", word)):
        if "가" <= ch <= "힣":
            return ch
        if ch.isdigit():
            return "영일이삼사오육칠팔구"[int(ch)]
        if ch.isalpha():
            return None
    return None


def has_batchim(word: str) -> bool:
    ch = _last_hangul(word)
    if ch is None:
        return False
    return (ord(ch) - 0xAC00) % 28 != 0


def josa(word: str, j: str) -> str:
    """josa('이용금액', '은') → '이용금액은', josa('인구', '은') → '인구는'."""
    with_b, without_b = _PAIRS[j]
    if j in ("으로", "로"):
        ch = _last_hangul(word)
        rieul = ch is not None and (ord(ch) - 0xAC00) % 28 == 8
        return word + (without_b if (not has_batchim(word) or rieul) else with_b)
    return word + (with_b if has_batchim(word) else without_b)
