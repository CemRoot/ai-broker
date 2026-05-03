"""
Ollama LLM service — fallback when Groq is unavailable.

Model: ``deepseek-r1:14b`` (Faz 0 kapanış kararı).
"""

from __future__ import annotations

import asyncio
import time

from app.core.config import Settings
from app.core.logging import get_logger
from app.services.llm.groq_service import LLMResponse

log = get_logger("ollama")

# Substrings that indicate Ollama daemon is not running
_UNREACHABLE_NEEDLES = (
    "connection refused",
    "failed to connect",
    "connect error",
    "connection error",
    "timeout",
    "name or service not known",
    "actively refused",
    "errno 61",
    "errno 111",
)


class OllamaService:
    """Sync Ollama chat wrapped in ``asyncio.to_thread``."""

    def __init__(self, settings: Settings) -> None:
        self._host = settings.ollama_base_url
        self.model = settings.ollama_model
        self._client = None

    def _get_client(self):
        if self._client is None:
            from ollama import Client as OllamaClient

            self._client = OllamaClient(host=self._host)
        return self._client

    async def analyze(self, prompt: str, system: str | None = None) -> LLMResponse:
        """Send a chat request to the local Ollama daemon.

        Raises ``RuntimeError("Ollama offline")`` if the daemon is unreachable.
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        t0 = time.perf_counter()
        try:
            resp_raw = await asyncio.to_thread(
                self._get_client().chat,
                model=self.model,
                messages=messages,
            )
            elapsed = time.perf_counter() - t0

            text = (resp_raw.message.content or "").strip()

            resp = LLMResponse(
                text=text,
                model=self.model,
                elapsed_seconds=round(elapsed, 3),
                # Ollama doesn't provide standard usage stats
            )
            log.info("Ollama OK | model=%s | %.1fs", self.model, elapsed)
            return resp

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            if self._is_unreachable(exc):
                log.error("Ollama offline (%.1fs): %s", elapsed, exc)
                raise RuntimeError("Ollama offline") from exc
            log.error("Ollama error after %.1fs: %s", elapsed, exc)
            raise

    @staticmethod
    def _is_unreachable(exc: BaseException) -> bool:
        err = str(exc).lower()
        return any(n in err for n in _UNREACHABLE_NEEDLES) or isinstance(
            exc, (ConnectionError, TimeoutError, OSError)
        )
