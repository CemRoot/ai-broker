"""Unit tests for Telegram PTB wiring and `/memory` handler (no live Telegram API)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram.ext import Application

from app.bot.app import register_handlers
from app.bot.handlers import help_handler, memory_handler, start_handler
from app.core.config import Settings


@pytest.fixture
def ptb_application() -> Application:
    # Token shape satisfies PTB parser; never used against real API in these tests.
    return Application.builder().token("123456789:ABCDEF-test-token-unit-tests").build()


def test_register_handlers_stores_retriever_in_bot_data(ptb_application: Application) -> None:
    """Regression: Telegram handlers must receive the same RAGRetriever instance as the app (see lifespan order)."""
    sentinel = object()
    register_handlers(
        ptb_application,
        retriever=sentinel,
        settings=Settings(),
    )
    assert ptb_application.bot_data["retriever"] is sentinel


def test_register_handlers_registers_memory_command(ptb_application: Application) -> None:
    register_handlers(ptb_application, settings=Settings())
    groups = ptb_application.handlers
    assert 0 in groups
    # At least one handler group contains CommandHandler("memory", ...)
    all_handlers = [h for group in groups.values() for h in group]
    assert any(getattr(h, "commands", None) == frozenset({"memory"}) for h in all_handlers)


@pytest.mark.asyncio
async def test_memory_handler_calls_list_recent_memories() -> None:
    sent: list[str] = []

    async def record_send(text: str, **_kwargs: object) -> None:
        sent.append(text)

    update = MagicMock()
    update.effective_chat = MagicMock()
    update.effective_chat.send_message = AsyncMock(side_effect=record_send)
    update.effective_user = MagicMock()
    update.effective_user.id = 1

    retriever = MagicMock()
    retriever.list_recent_memories = AsyncMock(
        return_value=[
            {
                "memory_type": "LESSON",
                "context": "Analyzed AMD. Action decided: SELL.",
                "outcome": "OPEN",
                "pnl_percent": None,
                "created_at": "2026-04-30T12:00:00+00:00",
            }
        ]
    )

    context = MagicMock()
    context.bot_data = {"allowed_ids": {1}, "retriever": retriever}
    context.args = ["amd"]

    await memory_handler(update, context)

    retriever.list_recent_memories.assert_awaited_once_with("AMD", limit=10)
    blob = "\n".join(sent)
    assert "AMD" in blob
    assert "Analyzed AMD" in blob


@pytest.mark.asyncio
async def test_memory_handler_empty_memories_message() -> None:
    sent: list[str] = []

    update = MagicMock()
    update.effective_chat = MagicMock()
    update.effective_chat.send_message = AsyncMock(side_effect=lambda text, **_k: sent.append(text))
    update.effective_user = MagicMock()
    update.effective_user.id = 1

    retriever = MagicMock()
    retriever.list_recent_memories = AsyncMock(return_value=[])

    context = MagicMock()
    context.bot_data = {"allowed_ids": {1}, "retriever": retriever}
    context.args = ["NVDA"]

    await memory_handler(update, context)

    blob = "\n".join(sent)
    assert "kayıtlı anı bulunamadı" in blob


@pytest.mark.asyncio
async def test_memory_handler_no_retriever_message() -> None:
    sent: list[str] = []

    update = MagicMock()
    update.effective_chat = MagicMock()
    update.effective_chat.send_message = AsyncMock(side_effect=lambda text, **_k: sent.append(text))
    update.effective_user = MagicMock()
    update.effective_user.id = 1

    context = MagicMock()
    context.bot_data = {"allowed_ids": {1}}
    context.args = ["AMD"]

    await memory_handler(update, context)

    blob = "\n".join(sent)
    assert "Retriever" in blob or "aktif değil" in blob


@pytest.mark.asyncio
async def test_start_handler_points_to_help() -> None:
    """``/start`` is now a short welcome that funnels users to ``/help`` for the full reference."""
    sent: list[str] = []

    update = MagicMock()
    update.effective_chat = MagicMock()
    update.effective_chat.send_message = AsyncMock(side_effect=lambda text, **_k: sent.append(text))
    update.effective_user = MagicMock()
    update.effective_user.id = 1

    context = MagicMock()
    context.bot_data = {"allowed_ids": {1}}

    await start_handler(update, context)

    blob = "\n".join(sent)
    # Welcome should link to the new categorized command reference.
    assert "/help" in blob
    # Quick-start basics should still be discoverable from /start alone.
    assert "/portfolio" in blob and "/paper" in blob


@pytest.mark.asyncio
async def test_help_handler_lists_all_commands() -> None:
    """``/help`` is the canonical command index — every registered command must appear."""
    sent: list[str] = []

    update = MagicMock()
    update.effective_chat = MagicMock()
    update.effective_chat.send_message = AsyncMock(side_effect=lambda text, **_k: sent.append(text))
    update.effective_user = MagicMock()
    update.effective_user.id = 1

    context = MagicMock()
    context.bot_data = {"allowed_ids": {1}}

    await help_handler(update, context)

    blob = "\n".join(sent)
    for cmd in (
        "/portfolio", "/paper", "/analyze", "/news", "/memory",
        "/runpaper", "/punishments", "/usage",
    ):
        assert cmd in blob, f"Missing {cmd} in /help output"
