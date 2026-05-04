"""
PTB Application builder and handler registration.

The ``Application`` instance is created in the FastAPI lifespan and shared
via ``app.state.bot_app``.
"""

from __future__ import annotations

from telegram import BotCommand
from telegram.ext import Application, CommandHandler, MessageHandler, filters

from app.core.config import Settings
from app.core.logging import get_logger
from app.bot.handlers import (
    analyze_handler,
    chat_handler,
    help_handler,
    news_handler,
    portfolio_handler,
    start_handler,
    usage_handler,
    memory_handler,
    paper_handler,
    runpaper_handler,
    punishments_handler,
)


# Telegram BotFather menu — single flat list (Telegram has no native grouping in the
# menu UI). Order matters: most-used first; descriptions use plain ASCII because
# Telegram strips most non-emoji glyphs from the menu strip on iOS.
BOT_MENU_COMMANDS: list[BotCommand] = [
    BotCommand("portfolio", "T212 portfoy ve acik pozisyonlar"),
    BotCommand("paper", "PaperAgent (sanal hesap, T212 golgesi)"),
    BotCommand("analyze", "Hisse analizi (orn: /analyze AAPL)"),
    BotCommand("news", "Hisse haber ozeti (orn: /news NVDA)"),
    BotCommand("memory", "RAG hafiza gecmisi (orn: /memory NVDA)"),
    BotCommand("runpaper", "Manuel PaperAgent cycle tetikle"),
    BotCommand("punishments", "Aktif ticker cezalari"),
    BotCommand("usage", "Gunluk LLM token kullanimi"),
    BotCommand("help", "Detayli komut rehberi"),
    BotCommand("start", "Karsilama mesaji"),
]

log = get_logger("bot.app")


def build_bot_application(settings: Settings) -> Application:
    """Create a PTB ``Application`` (not started — lifespan manages lifecycle)."""
    if not settings.telegram_bot_token:
        log.warning("TELEGRAM_BOT_TOKEN is empty — bot will not function")

    application = (
        Application.builder()
        .token(settings.telegram_bot_token)
        .build()
    )
    return application


def register_handlers(
    application: Application,
    *,
    t212=None,
    groq=None,
    ollama=None,
    retriever=None,
    settings: Settings | None = None,
    http_client=None,
    paper_broker=None,
    paper_agent=None,
    punishment_engine=None,
) -> None:
    """Register all command handlers and inject service references into bot_data."""
    application.bot_data["t212"] = t212
    application.bot_data["groq"] = groq
    application.bot_data["ollama"] = ollama
    application.bot_data["retriever"] = retriever
    application.bot_data["settings"] = settings
    application.bot_data["http_client"] = http_client
    application.bot_data["paper_broker"] = paper_broker
    application.bot_data["paper_agent"] = paper_agent
    application.bot_data["punishment_engine"] = punishment_engine

    # User allow-list
    if settings:
        application.bot_data["allowed_ids"] = settings.allowed_user_ids
    else:
        application.bot_data["allowed_ids"] = set()

    application.add_handler(CommandHandler("start", start_handler))
    application.add_handler(CommandHandler("help", help_handler))
    application.add_handler(CommandHandler("portfolio", portfolio_handler))
    application.add_handler(CommandHandler("analyze", analyze_handler))
    application.add_handler(CommandHandler("news", news_handler))
    application.add_handler(CommandHandler("usage", usage_handler))
    application.add_handler(CommandHandler("memory", memory_handler))
    application.add_handler(CommandHandler("paper", paper_handler))
    application.add_handler(CommandHandler("runpaper", runpaper_handler))
    application.add_handler(CommandHandler("punishments", punishments_handler))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat_handler))

    log.info(
        "Bot handlers registered: 10 commands + free-text chat (allowed_ids=%s)",
        application.bot_data["allowed_ids"] or "all",
    )


async def install_bot_menu(application: Application) -> None:
    """Push :data:`BOT_MENU_COMMANDS` to Telegram so the client shows them in the menu UI."""
    try:
        await application.bot.set_my_commands(BOT_MENU_COMMANDS)
        log.info("Telegram bot menu commands installed (%d entries)", len(BOT_MENU_COMMANDS))
    except Exception as exc:
        log.warning("set_my_commands failed: %s", exc)
