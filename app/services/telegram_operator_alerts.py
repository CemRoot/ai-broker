"""
Push high-signal failures to Telegram (CEO allow-list) so VPS issues are visible on-phone.

Uses HTML parse mode; bodies are escaped. Cooldown + dedupe keys avoid flood loops.
"""

from __future__ import annotations

import asyncio
import html
import time
import traceback
from typing import Any

from app.core.logging import get_logger

log = get_logger("telegram_alerts")

_sink: Any | None = None


class OperatorAlertSink:
    """Send short operational alerts to ``TELEGRAM_ALLOWED_USER_IDS``."""

    def __init__(self, application: Any | None, settings: Any) -> None:
        self._application = application
        self._settings = settings
        self._enabled = bool(getattr(settings, "telegram_operator_alerts_enabled", True))
        self._cooldown = float(getattr(settings, "telegram_operator_alert_cooldown_sec", 45.0) or 45.0)
        self._last: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def send(
        self,
        *,
        category: str,
        summary: str,
        detail: str | None = None,
        dedupe_key: str | None = None,
    ) -> None:
        if not self._enabled:
            return
        app = self._application
        if app is None:
            return
        try:
            bot = app.bot
        except Exception:
            return
        ids = getattr(self._settings, "allowed_user_ids", set()) or set()
        if not ids:
            return

        key = dedupe_key or category
        now = time.monotonic()
        async with self._lock:
            last = self._last.get(key, 0.0)
            if now - last < self._cooldown:
                return
            self._last[key] = now

        cat = html.escape((category or "Alert").strip()[:120], quote=False)
        summ = html.escape((summary or "").strip()[:800], quote=False)
        parts = [f"🚨 <b>{cat}</b>", summ]
        if detail:
            tail = html.escape(detail.strip()[:2800], quote=False)
            parts.append(f"<pre>{tail}</pre>")
        text = "\n".join(parts)
        if len(text) > 4096:
            text = text[:4090] + "…"

        for uid in ids:
            try:
                await bot.send_message(chat_id=int(uid), text=text, parse_mode="HTML")
            except Exception as exc:
                log.debug("operator alert send failed uid=%s: %s", uid, exc)


def configure_operator_alerts(application: Any | None, settings: Any) -> None:
    """Call once from FastAPI lifespan after the PTB ``Application`` is built (or ``None``)."""
    global _sink
    _sink = OperatorAlertSink(application, settings)


async def fire_operator_alert(
    *,
    category: str,
    summary: str,
    detail: str | None = None,
    dedupe_key: str | None = None,
) -> None:
    """Best-effort Telegram alert; never raises to callers."""
    s = _sink
    if s is None:
        return
    try:
        await s.send(
            category=category,
            summary=summary,
            detail=detail,
            dedupe_key=dedupe_key,
        )
    except Exception as exc:
        log.debug("fire_operator_alert failed: %s", exc)


def format_exc_brief(exc: BaseException, *, limit: int = 1200) -> str:
    """Compact traceback for Telegram ``<pre>`` blocks."""
    try:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    except Exception:
        tb = repr(exc)
    tb = tb.strip()
    if len(tb) > limit:
        tb = tb[: limit - 1] + "…"
    return tb
