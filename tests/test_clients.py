"""Claude 요청 형식: 5.x 모델은 temperature 없이 effort로, 워크스페이스 ID가 있으면 머리글로, 도구 사용 블록은 그대로 돌려줌."""
import dataolle.ai.clients as C


def _capture(monkeypatch, reply):
    seen = {}

    def fake_post(url, body, headers, timeout):
        seen.update(url=url, body=body, headers=headers)
        return reply
    monkeypatch.setattr(C, "_post", fake_post)
    return seen


def test_claude5_request_has_no_sampling_params(monkeypatch):
    seen = _capture(monkeypatch, {"content": [{"type": "thinking", "thinking": "", "signature": "s"},
                                              {"type": "text", "text": "OK"}], "stop_reason": "end_turn", "usage": {}})
    monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)
    r = C.AnthropicClient(model="claude-haiku-5-5", api_key="k").complete("sys", "ping", max_tokens=5)
    assert r.text == "OK"                                     # thinking 블록은 건너뛰고 text만
    b = seen["body"]
    assert "temperature" not in b and "top_p" not in b and b["output_config"]["effort"] == "low"
    assert b["max_tokens"] >= 2048                            # 사고 토큰 여유
    assert seen["headers"]["x-api-key"] == "k" and "anthropic-workspace-id" not in seen["headers"]


def test_older_claude_keeps_temperature_and_workspace_header(monkeypatch):
    seen = _capture(monkeypatch, {"content": [{"type": "text", "text": "OK"}], "usage": {}})
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_test")
    C.AnthropicClient(model="claude-haiku-4-5", api_key="k").complete("sys", "ping")
    assert "temperature" in seen["body"] and "output_config" not in seen["body"]
    assert seen["headers"]["anthropic-workspace-id"] == "wrkspc_test"


def test_tool_chat_passes_tools_and_returns_raw_blocks(monkeypatch):
    blocks = [{"type": "tool_use", "id": "tu1", "name": "check_data", "input": {}}]
    seen = _capture(monkeypatch, {"content": blocks, "stop_reason": "tool_use", "usage": {"input_tokens": 3}})
    tools = [{"name": "check_data", "description": "d", "input_schema": {"type": "object", "properties": {}}}]
    r = C.AnthropicClient(model="claude-haiku-5-5", api_key="k").chat("sys", [{"role": "user", "content": "q"}], tools)
    assert seen["body"]["tools"] == tools and seen["body"]["tool_choice"] == {"type": "auto"}
    assert r.raw["content"] == blocks and r.input_tokens == 3


def test_refusal_and_workspace_errors_are_friendly(monkeypatch):
    import pytest
    _capture(monkeypatch, {"content": [], "stop_reason": "refusal"})
    with pytest.raises(C.LLMError):
        C.AnthropicClient(model="claude-haiku-5-5", api_key="k").complete("s", "u")
