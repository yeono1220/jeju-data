"""AI 연결 점검 — Claude 키를 넣은 뒤 한 번 실행해 보세요.

    python tools/ai_check.py

① 짧은 문장 응답(연결·키·모델 확인) ② 도구 사용 한 바퀴(에이전트가 쓰는 요청 형식 확인)를 차례로 시험합니다.
센터 데이터는 보내지 않습니다(고정된 시험 문장만 보냄). 키는 data\\.env 의 ANTHROPIC_API_KEY 를 씁니다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataolle.ai.clients import LLMError, make_client  # noqa: E402
from dataolle.config import app_home, load_env  # noqa: E402

TOOL = [{"name": "lookup_area", "description": "동네 이름을 받아 그 동네의 유형을 돌려준다(시험용).",
         "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}]


def main() -> int:
    load_env(app_home())
    c = make_client("anthropic")
    if not c.available:
        print("ANTHROPIC_API_KEY가 없습니다. 관리자 탭에서 키를 저장하거나 data\\.env에 넣으세요.")
        return 1
    print(f"모델: {c.model}")
    try:
        r = c.complete("Reply with the single word OK.", "ping", max_tokens=16)
        print(f"① 문장 응답: {r.text.strip()!r} · {r.seconds:.1f}초 · 입력 {r.input_tokens} / 출력 {r.output_tokens} 토큰")
        msgs = [{"role": "user", "content": "일도1동은 어떤 동네 유형인지 lookup_area 도구로 확인하고 한 문장으로 답하세요."}]
        r1 = c.chat("도구를 써서 답하세요.", msgs, TOOL, max_tokens=2048)
        uses = [b for b in r1.raw.get("content", []) if b.get("type") == "tool_use"]
        if not uses:
            print(f"② 도구 사용: 도구를 부르지 않았습니다(stop_reason={r1.raw.get('stop_reason')}). 응답: {r1.text[:200]!r}")
            return 2
        u = uses[0]
        print(f"② 도구 호출: {u['name']}({json.dumps(u.get('input'), ensure_ascii=False)})")
        msgs += [{"role": "assistant", "content": r1.raw["content"]},
                 {"role": "user", "content": [{"type": "tool_result", "tool_use_id": u["id"], "content": "{\"유형\": \"원도심(시험값)\"}"}]}]
        r2 = c.chat("도구를 써서 답하세요.", msgs, TOOL, max_tokens=2048)
        print(f"   최종 답: {r2.text.strip()[:200]!r} (stop_reason={r2.raw.get('stop_reason')})")
    except LLMError as e:
        print(f"실패: {e}")
        return 3
    print("AI 연결과 도구 사용 형식이 정상입니다. 화면에서 '🤖 에이전트에게 맡기기'를 쓰면 AI 에이전트로 동작합니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
