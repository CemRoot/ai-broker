"""
Legacy Ollama LLM service (no longer used for generation in Faz 3+).

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
        self._backup_host = (getattr(settings, "ollama_backup_base_url", "") or "").strip()
        self.model = settings.ollama_model
        self._clients: dict[str, object] = {}

    def _get_client_for_host(self, host: str):
        client = self._clients.get(host)
        if client is None:
            from ollama import Client as OllamaClient

            client = OllamaClient(host=host)
            self._clients[host] = client
        return client

    async def analyze(self, prompt: str, system: str | None = None) -> LLMResponse:
        """Send a chat request to the local Ollama daemon.

        Raises ``RuntimeError("Ollama offline")`` if the daemon is unreachable.
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        hosts = [self._host]
        if self._backup_host and self._backup_host not in hosts:
            hosts.append(self._backup_host)

        last_exc: Exception | None = None
        saw_unreachable = False
        for idx, host in enumerate(hosts):
            t0 = time.perf_counter()
            try:
                resp_raw = await asyncio.to_thread(
                    self._get_client_for_host(host).chat,
                    model=self.model,
                    messages=messages,
                )
                elapsed = time.perf_counter() - t0
                text = (resp_raw.message.content or "").strip()
                resp = LLMResponse(
                    text=text,
                    model=self.model,
                    elapsed_seconds=round(elapsed, 3),
                )
                log.info("Ollama OK | host=%s | model=%s | %.1fs", host, self.model, elapsed)
                return resp
            except Exception as exc:
                last_exc = exc
                elapsed = time.perf_counter() - t0
                unreachable = self._is_unreachable(exc)
                saw_unreachable = saw_unreachable or unreachable
                if idx < len(hosts) - 1:
                    log.warning(
                        "Ollama primary failed (host=%s, %.1fs): %s | trying backup host=%s",
                        host,
                        elapsed,
                        exc,
                        hosts[idx + 1],
                    )
                    continue
                if unreachable:
                    log.error("Ollama offline (host=%s, %.1fs): %s", host, elapsed, exc)
                else:
                    log.error("Ollama error after %.1fs (host=%s): %s", elapsed, host, exc)

        if last_exc is None:
            raise RuntimeError("Ollama offline")
        if saw_unreachable:
            raise RuntimeError("Ollama offline") from last_exc
        raise last_exc

    @staticmethod
    def _is_unreachable(exc: BaseException) -> bool:
        err = str(exc).lower()
        return any(n in err for n in _UNREACHABLE_NEEDLES) or isinstance(
            exc, (ConnectionError, TimeoutError, OSError)
        )
