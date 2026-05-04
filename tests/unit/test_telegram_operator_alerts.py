"""Unit tests for Telegram operator alert sink (no real Telegram API)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.config import Settings
from app.services.telegram_operator_alerts import OperatorAlertSink, configure_operator_alerts, fire_operator_alert


@pytest.mark.asyncio
async def test_operator_alert_sink_respects_cooldown() -> None:
    app = MagicMock()
    app.bot = MagicMock()
    app.bot.send_message = AsyncMock()
    s = Settings()
    s.telegram_allowed_user_ids = "1"
    s.telegram_operator_alerts_enabled = True
    s.telegram_operator_alert_cooldown_sec = 60.0
    sink = OperatorAlertSink(app, s)
    await sink.send(category="T", summary="one", dedupe_key="k")
    await sink.send(category="T", summary="two", dedupe_key="k")
    assert app.bot.send_message.await_count == 1


@pytest.mark.asyncio
async def test_fire_operator_alert_no_configure_is_safe() -> None:
    configure_operator_alerts(None, Settings())
    await fire_operator_alert(category="X", summary="y")
