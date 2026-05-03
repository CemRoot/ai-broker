#!/usr/bin/env python3
"""
Telegram olmadan tek seferlik PaperAgent döngüsü — ``/runpaper`` ile aynı çağrı.

``main.py`` lifespan ile aynı PaperAgent bağımlılıkları (RAG, ceza, position monitor),
yalnızca bot ve T212 polling yok.

Kullanım::

    PYTHONPATH=. uv run python scripts/smoke_runpaper_local.py
    PYTHONPATH=. uv run python scripts/smoke_runpaper_local.py --dry-run

``--dry-run``: ``allow_trades=False`` (LLM analizi çalışır, broker işlemi yok).

Gereksinimler: ``SUPABASE_DB_URL``, Groq veya Ollama; yerelde RAG için ``ollama pull nomic-embed-text``.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.agents.paper_agent import PaperAgent, PaperAgentDeps
from app.agents.position_monitor import PositionMonitor
from app.agents.punishment import PunishmentEngine
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.memory.database import DatabaseSettings, SupabaseDatabase
from app.memory.embedder import OllamaEmbedder
from app.memory.retriever import RAGRetriever
from app.services.llm.groq_service import GroqService
from app.services.llm.ollama_service import OllamaService
from app.services.market_clock import MarketClock
from app.services.paper.broker import PaperBroker
from app.services.screener import SPScreener
from app.services.t212.client import T212Client
from app.tools.executor import ToolDeps, ToolExecutor

log = get_logger("smoke_runpaper")


async def _run(*, allow_trades: bool) -> int:
    settings = get_settings()
    setup_logging(settings.log_level)

    if not (settings.supabase_db_url or "").strip():
        log.error("SUPABASE_DB_URL is empty — cannot run PaperAgent smoke")
        return 1
    if not (settings.groq_api_key or "").strip():
        log.warning("GROQ_API_KEY empty — smoke will use Ollama only (ensure Ollama is running)")

    db = SupabaseDatabase(DatabaseSettings())
    await db.connect(dsn=settings.supabase_db_url or None)
    if not db.get_pool():
        log.error("Supabase pool not available — check DSN / network")
        await db.close()
        return 1

    http_client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))
    try:
        groq_svc = GroqService(settings) if settings.groq_api_key else None
        ollama_svc = OllamaService(settings)
        t212 = T212Client(http_client, settings)
        embedder = OllamaEmbedder()
        retriever = RAGRetriever(db=db, embedder=embedder)
        paper_broker = PaperBroker(
            db=db,
            paper_executes_on_t212=settings.paper_executes_on_t212,
        )
        market_clock = MarketClock()
        screener = SPScreener(settings=settings, http_client=http_client)
        tool_executor = ToolExecutor(
            ToolDeps(
                settings=settings,
                db=db,
                http_client=http_client,
                groq=groq_svc,
                ollama=ollama_svc,
                retriever=retriever,
                paper_broker=paper_broker,
                screener=screener,
                t212=t212,
            )
        )
        punishment_engine = PunishmentEngine(db=db, retriever=retriever)
        position_monitor = PositionMonitor(
            paper_broker=paper_broker,
            groq=groq_svc,
            ollama=ollama_svc,
            tool_executor=tool_executor,
            t212=t212,
            paper_executes_on_t212=settings.paper_executes_on_t212,
        )
        agent = PaperAgent(
            PaperAgentDeps(
                settings=settings,
                db=db,
                paper_broker=paper_broker,
                groq=groq_svc,
                ollama=ollama_svc,
                retriever=retriever,
                tool_executor=tool_executor,
                market_clock=market_clock,
                telegram_application=None,
                punishment_engine=punishment_engine,
                position_monitor=position_monitor,
                t212=t212,
            )
        )

        log.info("Starting PaperAgent.run_cycle('MANUAL', allow_trades=%s)", allow_trades)
        text, decisions = await agent.run_cycle("MANUAL", allow_trades=allow_trades)
        log.info(
            "OK — analysis_chars=%s decisions=%s",
            len(text or ""),
            len(decisions or []),
        )
        if decisions:
            log.info("Decision tickers: %s", [d.get("ticker") for d in decisions[:20]])
        return 0
    except Exception as exc:
        log.exception("PaperAgent smoke failed: %s", exc)
        return 1
    finally:
        await http_client.aclose()
        await db.close()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run one PaperAgent cycle locally (no Telegram), like /runpaper",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="allow_trades=False — analysis only, no paper broker trades",
    )
    args = p.parse_args()
    code = asyncio.run(_run(allow_trades=not args.dry_run))
    sys.exit(code)


if __name__ == "__main__":
    main()
