from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Any

import requests

from reddit_insights.config import settings


class LLMProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMResponse:
    provider: str
    model: str
    text: str
    raw: dict[str, Any]


class ChatProvider:
    provider_name: str
    model: str

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> LLMResponse:
        raise NotImplementedError


def _post_json(url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(settings.llm_max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=settings.llm_request_timeout)
            if response.status_code in {408, 429, 500, 502, 503, 504} and attempt < settings.llm_max_retries - 1:
                time.sleep(2**attempt)
                continue
            if response.status_code >= 400:
                raise LLMProviderError(f"Provider request failed with HTTP {response.status_code}: {response.text[:500]}")
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < settings.llm_max_retries - 1:
                time.sleep(2**attempt)
                continue
    raise LLMProviderError(f"Provider request failed after retries: {last_error}")


class GroqProvider(ChatProvider):
    provider_name = "groq"

    def __init__(self, model: str | None = None) -> None:
        self.model = model or settings.groq_model
        self.api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not self.api_key:
            raise LLMProviderError("GROQ_API_KEY is not set in the environment.")

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        raw = _post_json(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            payload=payload,
        )
        try:
            text = raw["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError(f"Unexpected Groq response shape: {raw}") from exc
        return LLMResponse(provider=self.provider_name, model=self.model, text=text, raw=raw)


class GeminiProvider(ChatProvider):
    provider_name = "gemini"

    def __init__(self, model: str | None = None) -> None:
        self.model = model or settings.gemini_model
        self.api_key = (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
        if not self.api_key:
            raise LLMProviderError("GOOGLE_API_KEY or GEMINI_API_KEY is not set in the environment.")

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> LLMResponse:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        raw = _post_json(url, headers={"Content-Type": "application/json"}, payload=payload)
        try:
            parts = raw["candidates"][0]["content"].get("parts", [])
            text = "".join(part.get("text", "") for part in parts).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError(f"Unexpected Gemini response shape: {raw}") from exc
        if not text:
            raise LLMProviderError(f"Gemini returned an empty text response: {raw}")
        return LLMResponse(provider=self.provider_name, model=self.model, text=text, raw=raw)


def build_provider(name: str) -> ChatProvider:
    normalized = name.strip().lower()
    if normalized == "groq":
        return GroqProvider()
    if normalized == "gemini":
        return GeminiProvider()
    raise ValueError(f"Unsupported provider '{name}'. Expected one of: groq, gemini")


def build_providers(names: list[str] | None = None) -> list[ChatProvider]:
    provider_names = names or ["groq", "gemini"]
    return [build_provider(name) for name in provider_names]
