"""Health payload shape (no full app lifespan — avoids Telegram polling in tests)."""

from __future__ import annotations

from unittest.mock import MagicMock

from app.api.routes import build_health_payload
from app.core.config import Settings


def test_health_telegram_disabled_when_no_token():
    s = Settings(telegram_bot_token="")
    body = build_health_payload(settings=s, db=None, bot_app=None)
    assert body["telegram"]["mode"] == "disabled"
    assert body["telegram"]["bot_token_configured"] is False
    assert body["telegram"]["handlers_ready"] is False


def test_health_webhook_mode_when_url_set():
    s = Settings(
        telegram_bot_token="123:abc",
        telegram_webhook_url="https://example.com",
    )
    body = build_health_payload(settings=s, db=None, bot_app=MagicMock())
    assert body["telegram"]["mode"] == "webhook"
    assert body["telegram"]["webhook_base_configured"] is True


def test_health_web_ui_flag():
    s_off = Settings(web_ui_enabled=False)
    body = build_health_payload(settings=s_off, db=None, bot_app=None)
    assert body["phase"] == "5"
    assert body["web_ui"]["enabled"] is False

    s_on = Settings(web_ui_enabled=True)
    body2 = build_health_payload(settings=s_on, db=None, bot_app=None)
    assert body2["web_ui"]["enabled"] is True


def test_health_memory_db_flags():
    s = Settings(supabase_db_url="postgresql://u:p@h/db")
    db = MagicMock()
    db.get_pool.return_value = None
    db.last_connect_error = "TimeoutError: ..."
    body = build_health_payload(settings=s, db=db, bot_app=None)
    assert body["memory_db"]["supabase_db_url_configured"] is True
    assert body["memory_db"]["asyncpg_pool_ready"] is False
    assert body["memory_db"]["last_connect_error"] == "TimeoutError: ..."

    db.get_pool.return_value = object()
    db.last_connect_error = None
    body2 = build_health_payload(settings=s, db=db, bot_app=None)
    assert body2["memory_db"]["asyncpg_pool_ready"] is True
    assert body2["memory_db"]["last_connect_error"] is None
