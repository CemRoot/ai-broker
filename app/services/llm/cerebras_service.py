"""
Cerebras LLM service — primary analysis engine (OpenAI-compatible).

Base URL: https://api.cerebras.ai/v1
Model: llama3.1-8b
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from app.core.config import Settings
from app.core.logging import get_logger
from app.services.llm.groq_service import LLMResponse

log = get_logger("cerebras")


class CerebrasService:
    """Async Cerebras chat completions with httpx (OpenAI-compatible)."""

    def __init__(self, settings: Settings, *, http_client: httpx.AsyncClient | None = None) -> None:
        self._api_key = settings.cerebras_api_key
        self.model = settings.cerebras_model
        self.base_url = settings.cerebras_base_url.rstrip("/")
        self._client = http_client
        self._owns_client = http_client is None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0))
        return self._client

    async def close(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def analyze(self, prompt: str, system: str | None = None) -> LLMResponse:
        """Send a chat completion request to Cerebras."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return await self._chat(messages)

    async def create_chat_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
        }
        if tools:
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice
        return await self._post_chat(payload)

    async def _chat(self, messages: list[dict[str, Any]]) -> LLMResponse:
        t0 = time.perf_counter()
        data = await self._post_chat({"model": self.model, "messages": messages})
        elapsed = time.perf_counter() - t0

        msg = ((data.get("choices") or [{}])[0].get("message") or {}) if isinstance(data, dict) else {}
        text = str(msg.get("content") or "").strip()
        usage = data.get("usage") if isinstance(data, dict) else {}

        resp = LLMResponse(
            text=text,
            model=self.model,
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            total_tokens=int(usage.get("total_tokens") or 0),
            elapsed_seconds=round(elapsed, 3),
        )
        log.info(
            "Cerebras OK | model=%s | tokens=%d | %.1fs",
            resp.model,
            resp.total_tokens,
            resp.elapsed_seconds,
        )
        return resp

    async def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self._api_key:
            raise ValueError("CEREBRAS_API_KEY is not set")
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        client = await self._get_client()
        try:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            log.error("Cerebras error: %s", exc)
            raise
