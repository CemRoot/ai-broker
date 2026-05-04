"""
AI Broker — FastAPI application entry point.

Start with::

    uvicorn app.main:app --reload          # development (polling mode)
    uvicorn app.main:app --host 0.0.0.0    # production  (webhook mode)

The ``lifespan`` context manager handles startup / shutdown of shared
resources (httpx client, PTB bot, LLM services) — no ``@app.on_event``.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.services.t212.client import T212Client
from app.services.llm.groq_service import GroqService
from app.services.llm.ollama_service import OllamaService
from app.bot.app import build_bot_application, install_bot_menu, register_handlers
from app.api.routes import router as api_router
from app.web.routes import router as web_router
from app.services.trump_monitor import TrumpMonitor
from app.memory.database import SupabaseDatabase, DatabaseSettings
from app.memory.embedder import OllamaEmbedder
from app.memory.retriever import RAGRetriever
from app.services.paper.broker import PaperBroker
from app.services.paper.t212_mirror_poller import T212MirrorPoller
from app.services.market_clock import MarketClock
from app.services.screener import SPScreener
from app.tools.executor import ToolDeps, ToolExecutor
from app.agents.paper_agent import PaperAgent, PaperAgentDeps
from app.agents.punishment import PunishmentEngine
from app.agents.position_monitor import PositionMonitor

log = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle (replaces deprecated ``on_event``)."""
    settings = get_settings()
    setup_logging(settings.log_level)

    if (settings.telegram_webhook_url or "").strip() and not (
        settings.telegram_webhook_secret or ""
    ).strip():
        log.critical(
            "TELEGRAM_WEBHOOK_SECRET is empty while TELEGRAM_WEBHOOK_URL is set — "
            "POST /telegram/webhook will return HTTP 500 until a secret is configured."
        )
    if (settings.internal_api_key or "").strip():
        log.info("INTERNAL_API_KEY is set — /internal/* requires X-Internal-Api-Key")

    log.info("AI Broker starting — Phase 5 (Faz 5 kickoff: web UI optional + unified analyze + TOON)")

    # ── Database (Supabase / pgvector) ──────────────────────────────
    db_settings = DatabaseSettings()
    supabase_db = SupabaseDatabase(db_settings)
    # Prefer central ``get_settings()`` so ``SUPABASE_DB_URL`` matches the rest of the app
    # (``DatabaseSettings`` also reads `.env`, but cwd/env duplication caused “no DB” confusion).
    await supabase_db.connect(dsn=settings.supabase_db_url or None)

    # ── httpx client (shared, for T212) ─────────────────────────────
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, connect=5.0),
    )
    t212 = T212Client(http_client, settings)
    log.info("T212 client ready — base=%s", settings.t212_api_url)

    # ── LLM services ────────────────────────────────────────────────
    groq_svc = GroqService(settings) if settings.groq_api_key else None
    ollama_svc = OllamaService(settings)
    if groq_svc:
        log.info("Groq service ready — model=%s", groq_svc.model)
    else:
        log.warning("GROQ_API_KEY empty — Groq disabled, Ollama-only mode")
    log.info("Ollama fallback ready — model=%s host=%s", ollama_svc.model, settings.ollama_base_url)

    # ── RAG retriever (must exist before Telegram handler registration) ──
    # register_handlers stores ``retriever`` in ``bot_data``. Previously it was
    # passed as ``app.state.retriever`` before assignment, so Telegram always
    # saw ``None`` and never wrote to ``trade_memories``.
    embedder = OllamaEmbedder(http_client=http_client)
    retriever = RAGRetriever(db=supabase_db, embedder=embedder)
    
    # ── Paper Broker ────────────────────────────────────────────────
    paper_broker = PaperBroker(
        db=supabase_db,
        paper_executes_on_t212=settings.paper_executes_on_t212,
    )
    if (
        settings.paper_executes_on_t212
        and settings.paper_t212_sync_supabase_ledger
        and supabase_db.get_pool()
    ):
        try:
            await paper_broker.sync_ledger_from_t212_client(t212)
            log.info("T212 → Supabase shadow ledger initial sync OK")
        except Exception as exc:
            log.warning("T212 initial shadow ledger sync failed: %s", exc)

    # ── Faz 3: Tools, screener, market clock ────────────────────────
    market_clock = MarketClock()
    screener = SPScreener(settings=settings, http_client=http_client)
    tool_executor = ToolExecutor(
        ToolDeps(
            settings=settings,
            db=supabase_db,
            http_client=http_client,
            groq=groq_svc,
            ollama=ollama_svc,
            retriever=retriever,
            paper_broker=paper_broker,
            screener=screener,
            t212=t212,
        )
    )

    punishment_engine = PunishmentEngine(db=supabase_db, retriever=retriever)
    position_monitor = PositionMonitor(
        paper_broker=paper_broker,
        groq=groq_svc,
        ollama=ollama_svc,
        tool_executor=tool_executor,
        t212=t212,
        paper_executes_on_t212=settings.paper_executes_on_t212,
    )

    # ── Telegram bot ────────────────────────────────────────────────
    bot_app = None
    if settings.telegram_bot_token:
        bot_app = build_bot_application(settings)
        await bot_app.initialize()

        if settings.telegram_webhook_url:
            # Webhook mode: set webhook with Telegram servers
            webhook_url = f"{settings.telegram_webhook_url.rstrip('/')}/telegram/webhook"
            await bot_app.bot.set_webhook(
                url=webhook_url,
                secret_token=settings.telegram_webhook_secret or None,
            )
            log.info("Telegram webhook set: %s", webhook_url)
        else:
            # Polling mode: start updater for local dev
            await bot_app.start()
            await bot_app.updater.start_polling(drop_pending_updates=True)
            log.info("Telegram polling started (dev mode)")
    else:
        log.warning("TELEGRAM_BOT_TOKEN empty — bot disabled")

    # ── Faz 3: Paper Agent (event loop) ──────────────────────────────
    paper_agent = PaperAgent(
        PaperAgentDeps(
            settings=settings,
            db=supabase_db,
            paper_broker=paper_broker,
            groq=groq_svc,
            ollama=ollama_svc,
            retriever=retriever,
            tool_executor=tool_executor,
            market_clock=market_clock,
            telegram_application=bot_app,
            punishment_engine=punishment_engine,
            position_monitor=position_monitor,
            t212=t212,
        )
    )

    if bot_app:
        register_handlers(
            bot_app,
            t212=t212,
            groq=groq_svc,
            ollama=ollama_svc,
            retriever=retriever,
            settings=settings,
            http_client=http_client,
            paper_broker=paper_broker,
            paper_agent=paper_agent,
            punishment_engine=punishment_engine,
        )
        await install_bot_menu(bot_app)

    trump_monitor = TrumpMonitor(
        settings=settings,
        groq=groq_svc,
        http_client=http_client,
        telegram_application=bot_app,
        db=supabase_db,
        retriever=retriever,
    )
    trump_monitor.set_paper_agent(paper_agent)
    trump_task = None
    if getattr(settings, "trump_monitor_enabled", True):
        trump_task = asyncio.create_task(trump_monitor.run_with_reconnect())
        log.info("TrumpMonitor background task started")
    else:
        log.info(
            "TrumpMonitor disabled (TRUMP_MONITOR_ENABLED=false) — WebSocket + "
            "in-process REST puller skipped; /internal/trump/pull still available"
        )

    trump_pull_task = None
    if (
        getattr(settings, "trump_monitor_enabled", True)
        and int(getattr(settings, "trump_pull_interval_sec", 0) or 0) > 0
    ):
        async def _trump_pull_loop() -> None:
            interval = max(15, int(settings.trump_pull_interval_sec))
            log.info(
                "TrumpMonitor in-process REST pull every %ss (WebSocket fallback)",
                interval,
            )
            await asyncio.sleep(min(15, interval))
            while True:
                try:
                    summary = await trump_monitor.pull_recent_statuses(limit=5)
                    if (summary or {}).get("new_posts", 0) > 0:
                        log.info("TrumpMonitor pull: %s", summary)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    log.warning("TrumpMonitor pull error: %s", exc)
                await asyncio.sleep(interval)

        trump_pull_task = asyncio.create_task(_trump_pull_loop())

    paper_task = None
    if settings.paper_agent_enabled:
        paper_task = asyncio.create_task(paper_agent.run_forever())
        log.info("PaperAgent background task started")
    else:
        log.info("PaperAgent disabled (set PAPER_AGENT_ENABLED=true to run loop)")

    mirror_poll_task = None
    if (
        settings.paper_executes_on_t212
        and settings.paper_t212_mirror_poller_enabled
        and supabase_db.get_pool()
    ):
        mirror_poller = T212MirrorPoller(
            settings=settings,
            db=supabase_db,
            t212=t212,
            paper_broker=paper_broker,
        )
        mirror_poll_task = asyncio.create_task(mirror_poller.run_forever())
        log.info(
            "T212MirrorPoller started (every %ss; reconcile external=%s)",
            max(15, settings.paper_t212_pending_poll_sec),
            settings.paper_t212_reconcile_external_orders,
        )

    shadow_ledger_task = None
    if (
        settings.paper_executes_on_t212
        and settings.paper_t212_sync_supabase_ledger
        and supabase_db.get_pool()
        and not settings.paper_t212_mirror_poller_enabled
    ):
        async def _t212_shadow_ledger_loop() -> None:
            interval = float(max(30, settings.paper_t212_pending_poll_sec))
            log.info("T212 shadow ledger sync only (mirror poller off), every %ss", interval)
            while True:
                try:
                    await paper_broker.sync_ledger_from_t212_client(t212)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    log.warning("T212 shadow ledger sync failed: %s", exc)
                await asyncio.sleep(interval)

        shadow_ledger_task = asyncio.create_task(_t212_shadow_ledger_loop())

    # ── Expose on app.state ─────────────────────────────────────────
    app.state.db = supabase_db
    app.state.embedder = embedder
    app.state.retriever = retriever
    app.state.t212 = t212
    app.state.groq = groq_svc
    app.state.ollama = ollama_svc
    app.state.bot_app = bot_app
    app.state.settings = settings
    app.state.http_client = http_client
    app.state.trump_task = trump_task
    app.state.trump_monitor = trump_monitor
    app.state.paper_broker = paper_broker
    app.state.paper_agent = paper_agent
    app.state.punishment_engine = punishment_engine
    app.state.position_monitor = position_monitor
    app.state.market_clock = market_clock
    app.state.screener = screener
    app.state.tool_executor = tool_executor
    app.state.paper_task = paper_task
    app.state.mirror_poll_task = mirror_poll_task
    app.state.shadow_ledger_task = shadow_ledger_task
    app.state.trump_pull_task = trump_pull_task

    log.info(
        "Integrations — telegram=%s, supabase_pool=%s, supabase_dsn_in_env=%s",
        "webhook"
        if settings.telegram_webhook_url
        else ("polling" if settings.telegram_bot_token else "off"),
        supabase_db.get_pool() is not None,
        bool((settings.supabase_db_url or "").strip()),
    )
    log.info("AI Broker ready ✅")
    yield

    # ── SHUTDOWN ────────────────────────────────────────────────────
    log.info("AI Broker shutting down...")

    if trump_task:
        trump_task.cancel()
        try:
            await trump_task
        except asyncio.CancelledError:
            log.info("TrumpMonitor task stopped")

    if trump_pull_task:
        trump_pull_task.cancel()
        try:
            await trump_pull_task
        except asyncio.CancelledError:
            log.info("TrumpMonitor REST pull loop stopped")

    if paper_task:
        paper_task.cancel()
        try:
            await paper_task
        except asyncio.CancelledError:
            log.info("PaperAgent task stopped")

    if mirror_poll_task:
        mirror_poll_task.cancel()
        try:
            await mirror_poll_task
        except asyncio.CancelledError:
            log.info("T212MirrorPoller stopped")

    if shadow_ledger_task:
        shadow_ledger_task.cancel()
        try:
            await shadow_ledger_task
        except asyncio.CancelledError:
            log.info("T212 shadow ledger sync stopped")

    if bot_app:
        if not settings.telegram_webhook_url and bot_app.updater:
            # Polling mode: stop updater
            await bot_app.updater.stop()
            await bot_app.stop()
        await bot_app.shutdown()
        log.info("Telegram bot shut down")

    await http_client.aclose()
    log.info("httpx client closed — goodbye")
    
    await supabase_db.close()
    log.info("Database connection pool closed")


# ── FastAPI app ─────────────────────────────────────────────────────

app = FastAPI(
    title="AI Broker",
    description="Personal AI trading advisor — Phase 5 (optional /ui, Trump monitor, unified analyze, TOON optional)",
    version="1.4.0",
    lifespan=lifespan,
)

app.include_router(api_router)
app.include_router(web_router)

_cfg = get_settings()
_cors_origins = [
    o.strip()
    for o in (getattr(_cfg, "public_dashboard_cors_origins", "") or "").split(",")
    if o.strip()
]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_methods=["GET", "HEAD", "OPTIONS"],
        allow_headers=["*"],
    )
