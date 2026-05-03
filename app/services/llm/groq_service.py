"""
Groq LLM service — primary analysis engine.

Model: ``llama-3.3-70b-versatile`` (Faz 0 kapanış kararı).
Includes usage tracking (token counters) and structured logging.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.core.config import Settings
from app.core.logging import get_logger

log = get_logger("groq")

GROQ_MODEL = "llama-3.3-70b-versatile"


@dataclass
class LLMResponse:
    """Unified response from any LLM service."""

    text: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    elapsed_seconds: float = 0.0
    error: str | None = None


@dataclass
class UsageStats:
    """In-memory daily usage counters."""

    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    errors: int = 0
    daily_reset: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def record(self, resp: LLMResponse) -> None:
        self._maybe_reset()
        self.total_requests += 1
        self.total_input_tokens += resp.input_tokens
        self.total_output_tokens += resp.output_tokens
        self.total_tokens += resp.total_tokens
        if resp.error:
            self.errors += 1

    def to_dict(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "errors": self.errors,
            "daily_reset": self.daily_reset.isoformat(),
        }

    def _maybe_reset(self) -> None:
        """Reset counters at UTC midnight."""
        now = datetime.now(timezone.utc)
        if now.date() > self.daily_reset.date():
            self.total_requests = 0
            self.total_input_tokens = 0
            self.total_output_tokens = 0
            self.total_tokens = 0
            self.errors = 0
            self.daily_reset = now


class GroqService:
    """Async Groq chat completions with usage tracking."""

    def __init__(self, settings: Settings) -> None:
        self._api_key = settings.groq_api_key
        self.model = GROQ_MODEL
        self.usage = UsageStats()
        self._client = None  # lazy init

    def _get_client(self):
        """Lazy-init the Groq client (import is heavy)."""
        if self._client is None:
            from groq import Groq

            if not self._api_key:
                raise ValueError("GROQ_API_KEY is not set")
            self._client = Groq(api_key=self._api_key)
        return self._client

    async def analyze(self, prompt: str, system: str | None = None) -> LLMResponse:
        """Send a chat completion request to Groq.

        Parameters
        ----------
        prompt:
            User message content.
        system:
            Optional system prompt.
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        t0 = time.perf_counter()
        try:
            completion = await asyncio.to_thread(
                self._get_client().chat.completions.create,
                messages=messages,
                model=self.model,
            )
            elapsed = time.perf_counter() - t0

            text = (completion.choices[0].message.content or "").strip()
            usage = completion.usage

            resp = LLMResponse(
                text=text,
                model=self.model,
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                total_tokens=getattr(usage, "total_tokens", 0) or 0,
                elapsed_seconds=round(elapsed, 3),
            )
            self.usage.record(resp)

            log.info(
                "Groq OK | model=%s | tokens=%d | %.1fs",
                self.model,
                resp.total_tokens,
                resp.elapsed_seconds,
            )
            return resp

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            log.error("Groq error after %.1fs: %s", elapsed, exc)
            resp = LLMResponse(
                model=self.model,
                elapsed_seconds=round(elapsed, 3),
                error=str(exc),
            )
            self.usage.record(resp)
            raise

    async def analyze_multimodal(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
    ) -> LLMResponse:
        """Chat completion with arbitrary OpenAI-style messages (text + optional vision parts)."""
        use_model = model or self.model
        t0 = time.perf_counter()
        try:
            completion = await asyncio.to_thread(
                self._get_client().chat.completions.create,
                messages=messages,
                model=use_model,
            )
            elapsed = time.perf_counter() - t0

            text = (completion.choices[0].message.content or "").strip()
            usage = completion.usage

            resp = LLMResponse(
                text=text,
                model=use_model,
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                total_tokens=getattr(usage, "total_tokens", 0) or 0,
                elapsed_seconds=round(elapsed, 3),
            )
            self.usage.record(resp)
            log.info(
                "Groq multimodal OK | model=%s | tokens=%d | %.1fs",
                use_model,
                resp.total_tokens,
                resp.elapsed_seconds,
            )
            return resp

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            log.error("Groq multimodal error after %.1fs: %s", elapsed, exc)
            resp = LLMResponse(
                model=use_model,
                elapsed_seconds=round(elapsed, 3),
                error=str(exc),
            )
            self.usage.record(resp)
            raise
