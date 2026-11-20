"""LLM 연결(표준 라이브러리 urllib만 사용 — 설치 패키지를 늘리지 않기 위해).

지원: Claude(기본) · OpenAI · Gemini · Ollama(로컬) · 없음(규칙 기반).
모든 호출은 ai.gateway.Gateway를 거쳐서만 이뤄진다(전송 레벨 검사·감사 기록).
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

DEFAULT_MODELS = {
    "anthropic": "claude-haiku-5-5",
    "openai": "gpt-5-mini",
    "gemini": "gemini-2.5-flash",
    "ollama": "qwen2.5:3b",
}
ENDPOINTS = {
    "anthropic": "https://api.anthropic.com/v1/messages",
    "openai": "https://api.openai.com/v1/chat/completions",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
    "ollama": "http://127.0.0.1:11434/api/chat",
}
PROVIDER_LABEL = {"anthropic": "Claude(Anthropic)", "openai": "OpenAI", "gemini": "Gemini(Google)", "ollama": "Ollama(로컬)",
                  "none": "AI 사용 안 함"}


class LLMError(RuntimeError):
    """사용자에게 그대로 보여 줄 수 있는 한국어 오류."""


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    raw: dict = field(default_factory=dict, repr=False)


def _post(url: str, body: dict, headers: dict, timeout: float) -> dict:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"content-type": "application/json", **headers}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", "replace")[:300]
        except Exception:
            pass
        if e.code in (401, 403):
            raise LLMError(f"API 키가 올바르지 않거나 권한이 없습니다(HTTP {e.code}). 관리자 설정에서 키를 확인하세요.") from e
        if "anthropic-workspace-id" in detail:
            raise LLMError("이 Claude API 키는 워크스페이스가 지정되지 않은 키입니다. Claude Console에서 키를 만들 때 워크스페이스를 지정하거나, "
                           "관리자 탭에 워크스페이스 ID(wrkspc_…)를 입력하세요.") from e
        if e.code == 429:
            raise LLMError("AI 사용 한도를 넘었거나 요청이 너무 많습니다(HTTP 429). 잠시 후 다시 시도하세요.") from e
        raise LLMError(f"AI 서버 오류(HTTP {e.code}): {detail}") from e
    except urllib.error.URLError as e:
        raise LLMError(f"AI 서버에 연결하지 못했습니다({e.reason}). 인터넷·방화벽 설정을 확인하거나 'AI 사용 안 함'으로 진행하세요.") from e
    except TimeoutError as e:
        raise LLMError("AI 응답 시간이 초과되었습니다. 'AI 사용 안 함'으로도 같은 분석을 할 수 있습니다.") from e


class LLMClient:
    provider = "none"

    def __init__(self, model: str | None = None, api_key: str | None = None, base_url: str | None = None,
                 timeout: float = 45.0):
        self.model = model or DEFAULT_MODELS.get(self.provider, "")
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return True

    @property
    def label(self) -> str:
        return f"{PROVIDER_LABEL.get(self.provider, self.provider)} · {self.model}" if self.model else PROVIDER_LABEL["none"]

    def complete(self, system: str, user: str, max_tokens: int = 1200, temperature: float = 0.2) -> LLMResponse:
        raise NotImplementedError


class NullClient(LLMClient):
    """AI를 쓰지 않는 모드. 호출하면 None 대신 예외 없이 빈 응답 — 상위 계층이 규칙 기반으로 진행."""
    provider = "none"

    @property
    def available(self) -> bool:
        return False

    def complete(self, system: str, user: str, max_tokens: int = 1200, temperature: float = 0.2) -> LLMResponse:
        raise LLMError("AI 사용 안 함 모드입니다.")


class AnthropicClient(LLMClient):
    """Claude Messages API. Claude 5.x(예: claude-haiku-5-5)는 temperature·top_p·top_k를 받지 않고(400 오류),
    적응형 사고(thinking)가 기본으로 켜져 있어 응답 앞에 thinking 블록이 올 수 있다 → 블록은 type으로 고르고,
    사고 토큰도 max_tokens에 들어가므로 여유를 두며, 사고량은 output_config.effort로 줄인다."""
    provider = "anthropic"
    effort = "low"                    # 문장 다듬기·계획처럼 짧은 일은 낮게(빠르고 저렴)

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _new_api(self) -> bool:
        m = (self.model or "").lower()
        return not any(old in m for old in ("claude-3", "claude-2", "claude-instant", "-4-", "-4-5", "sonnet-4", "opus-4", "haiku-4"))

    def _body(self, system: str, messages: list, max_tokens: int, temperature: float | None, tools: list | None = None,
              effort: str | None = None) -> dict:
        body = {"model": self.model, "max_tokens": max_tokens, "system": system, "messages": messages}
        if self._new_api():
            body["output_config"] = {"effort": effort or self.effort}
        elif temperature is not None:
            body["temperature"] = temperature
        if tools:
            body["tools"] = tools
            body["tool_choice"] = {"type": "auto"}
        return body

    def _send(self, body: dict) -> dict:
        if not self.api_key:
            raise LLMError("Claude API 키가 없습니다. 관리자 설정에서 입력하세요.")
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}
        ws = os.environ.get("ANTHROPIC_WORKSPACE_ID", "").strip()
        if ws:                                  # 워크스페이스가 지정되지 않은 키는 이 머리글이 꼭 필요함
            headers["anthropic-workspace-id"] = ws
        out = _post(self.base_url or ENDPOINTS["anthropic"], body, headers, self.timeout)
        if out.get("stop_reason") == "refusal":
            raise LLMError("AI가 이 요청을 처리하지 않았습니다(안전 정책). 규칙 기반으로 계속합니다.")
        return out

    def complete(self, system: str, user: str, max_tokens: int = 1200, temperature: float = 0.2) -> LLMResponse:
        t0 = time.time()
        out = self._send(self._body(system, [{"role": "user", "content": user}], max(max_tokens, 2048), temperature))
        text = "".join(b.get("text", "") for b in out.get("content", []) if b.get("type") == "text")
        if not text and out.get("stop_reason") == "max_tokens":
            raise LLMError("AI 응답이 길이 제한에 걸려 비어 있습니다. 잠시 후 다시 시도하세요.")
        u = out.get("usage", {})
        return LLMResponse(text, self.provider, out.get("model", self.model), u.get("input_tokens", 0),
                           u.get("output_tokens", 0), time.time() - t0, out)

    def chat(self, system: str, messages: list, tools: list, max_tokens: int = 4096, effort: str | None = None) -> LLMResponse:
        """도구 사용(tool use) 한 걸음. raw['content']의 블록(thinking·text·tool_use)을 그대로 다음 요청에 되돌려 보낸다."""
        t0 = time.time()
        out = self._send(self._body(system, messages, max_tokens, None, tools, effort))
        text = "".join(b.get("text", "") for b in out.get("content", []) if b.get("type") == "text")
        u = out.get("usage", {})
        return LLMResponse(text, self.provider, out.get("model", self.model), u.get("input_tokens", 0),
                           u.get("output_tokens", 0), time.time() - t0, out)


class OpenAIClient(LLMClient):
    provider = "openai"

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def complete(self, system: str, user: str, max_tokens: int = 1200, temperature: float = 0.2) -> LLMResponse:
        if not self.api_key:
            raise LLMError("OpenAI API 키가 없습니다. 관리자 설정에서 입력하세요.")
        t0 = time.time()
        body = {"model": self.model, "max_completion_tokens": max(max_tokens, 4000),     # 추론 토큰 여유
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        out = _post(self.base_url or ENDPOINTS["openai"], body, {"authorization": f"Bearer {self.api_key}"}, self.timeout)
        ch = (out.get("choices") or [{}])[0]
        u = out.get("usage", {})
        return LLMResponse(ch.get("message", {}).get("content", "") or "", self.provider, out.get("model", self.model),
                           u.get("prompt_tokens", 0), u.get("completion_tokens", 0), time.time() - t0, out)


class GeminiClient(LLMClient):
    provider = "gemini"

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def complete(self, system: str, user: str, max_tokens: int = 1200, temperature: float = 0.2) -> LLMResponse:
        if not self.api_key:
            raise LLMError("Gemini API 키가 없습니다. 관리자 설정에서 입력하세요.")
        t0 = time.time()
        url = (self.base_url or ENDPOINTS["gemini"]).format(model=self.model)
        body = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"maxOutputTokens": max(max_tokens, 4000), "temperature": temperature}}
        out = _post(url, body, {"x-goog-api-key": self.api_key}, self.timeout)
        cands = out.get("candidates") or [{}]
        text = "".join(p.get("text", "") for p in cands[0].get("content", {}).get("parts", []))
        u = out.get("usageMetadata", {})
        return LLMResponse(text, self.provider, self.model, u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0),
                           time.time() - t0, out)


class OllamaClient(LLMClient):
    """이 PC에서 도는 로컬 모델(외부 전송 없음). GTX1060 3GB면 3B급 소형 모델 권장."""
    provider = "ollama"

    def complete(self, system: str, user: str, max_tokens: int = 1200, temperature: float = 0.2) -> LLMResponse:
        t0 = time.time()
        body = {"model": self.model, "stream": False, "options": {"temperature": temperature, "num_predict": max_tokens},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        out = _post(self.base_url or ENDPOINTS["ollama"], body, {}, max(self.timeout, 120))
        return LLMResponse(out.get("message", {}).get("content", ""), self.provider, self.model,
                           out.get("prompt_eval_count", 0), out.get("eval_count", 0), time.time() - t0, out)


CLIENTS = {"anthropic": AnthropicClient, "openai": OpenAIClient, "gemini": GeminiClient, "ollama": OllamaClient,
           "none": NullClient}
ENV_KEYS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}


def make_client(provider: str = "anthropic", model: str | None = None, api_key: str | None = None,
                base_url: str | None = None) -> LLMClient:
    cls = CLIENTS.get(provider, NullClient)
    key = api_key or os.environ.get(ENV_KEYS.get(provider, ""), "") or None
    return cls(model=model, api_key=key, base_url=base_url)


def is_external(provider: str) -> bool:
    """이 PC 밖으로 나가는 연결인지(Ollama·없음은 외부 전송 없음)."""
    return provider in ("anthropic", "openai", "gemini")
