"""질문 50개 → 계획 정확도(규칙 기반, AI 없이). 결과 설명자료의 '계획 정확도' 근거.
탐나는전 캐시백 설계 질문은 예산·기간·우선순위를 이 PC가 질문에서 바르게 읽는지도 본다(courses.policy.parse_policy)."""
import json
from pathlib import Path

from dataolle.ai.planner import rule_plan

QS = json.loads((Path(__file__).parent / "data" / "questions.json").read_text(encoding="utf-8"))


def score(item) -> bool:
    p = rule_plan(item["q"])
    if p.course != item["course"]:
        return False
    if item["course"] in ("market", "status", "before_after", "tamna"):
        if not all(p.market.get(k) == item[k] for k in ("bjd", "biz", "event", "value") if k in item):
            return False
        if item["course"] == "tamna" and any(k in item for k in ("budget", "months", "goal")):
            from dataolle.courses.policy import parse_policy
            kw = parse_policy(item["q"], effect_question=bool(p.market.get("event")))
            return all(kw.get(k) == item[k] for k in ("budget", "months", "goal") if k in item)
        return True
    a = p.analyses[0]
    return a.dataset == item["dataset"] and a.rows == item["rows"]


def test_rule_planner_accuracy():
    ok = [score(i) for i in QS]
    misses = [i["q"] for i, s in zip(QS, ok) if not s]
    acc = sum(ok) / len(ok)
    print(f"rule planner accuracy {acc:.0%} ({sum(ok)}/{len(ok)}), misses: {misses}")
    assert acc >= 0.9, misses
