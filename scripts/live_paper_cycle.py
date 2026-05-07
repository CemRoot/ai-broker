"""
CTO live smoke: real T212 demo + real LLM + real tools + real RAG.

Run a single PaperAgent cycle end-to-end with allow_trades=True. Used for
operational verification only — the running uvicorn is not required (in fact
should be stopped to avoid double T212/Telegram traffic).

Usage:
    PYTHONPATH=. uv run python scripts/live_paper_cycle.py [--event OPEN|MIDDAY|CLOSE|MANUAL]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

import httpx


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", default="MANUAL")
    parser.add_argument("--no-trades", action="store_true", help="Run cycle without sending orders")
    args = parser.parse_args()

    from app.agents.paper_agent import PaperAgent, PaperAgentDeps
    from app.agents.position_monitor import PositionMonitor
    from app.agents.punishment import PunishmentEngine
    from app.core.config import get_settings
    from app.memory.database import DatabaseSettings, SupabaseDatabase
    from app.memory.embedder import EmbedderSettings, OllamaEmbedder
    from app.memory.retriever import RAGRetriever
    from app.services.llm.groq_service import GroqService
    from app.services.llm.ollama_service import OllamaService
    from app.services.market_clock import MarketClock
    from app.services.paper.broker import PaperBroker
    from app.services.screener import SPScreener
    from app.services.t212.client import T212Client
    from app.tools.executor import ToolDeps, ToolExecutor

    settings = get_settings()

    db = SupabaseDatabase(DatabaseSettings())
    for attempt in range(1, 4):
        await db.connect(dsn=settings.supabase_db_url or None)
        if db.get_pool() is not None:
            break
        print(f"Supabase pool attempt {attempt} failed; retrying in 6s")
        await asyncio.sleep(6)
    if db.get_pool() is None:
        print("FATAL: cannot connect to Supabase")
        return 2

    http = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))
    try:
        groq = GroqService(settings) if settings.groq_api_key else None
        ollama = OllamaService(settings)
        embedder = OllamaEmbedder(EmbedderSettings(), http_client=http)
        retriever = RAGRetriever(db=db, embedder=embedder)
        t212 = T212Client(http, settings)
        paper_broker = PaperBroker(
            db=db,
            paper_executes_on_t212=settings.paper_executes_on_t212,
        )
        screener = SPScreener(settings=settings, http_client=http)
        tool_executor = ToolExecutor(
            ToolDeps(
                settings=settings,
                db=db,
                http_client=http,
                groq=groq,
                ollama=ollama,
                retriever=retriever,
                paper_broker=paper_broker,
                screener=screener,
                t212=t212 if settings.paper_executes_on_t212 else None,
            )
        )
        position_monitor = PositionMonitor(
            paper_broker=paper_broker,
            groq=groq,
            ollama=ollama,
            tool_executor=tool_executor,
            t212=t212 if settings.paper_executes_on_t212 else None,
            paper_executes_on_t212=settings.paper_executes_on_t212,
        )
        punishment_engine = PunishmentEngine(db=db, retriever=retriever)
        agent = PaperAgent(
            PaperAgentDeps(
                settings=settings,
                db=db,
                paper_broker=paper_broker,
                cerebras=None,
                groq=groq,
                ollama=ollama,
                retriever=retriever,
                tool_executor=tool_executor,
                market_clock=MarketClock(),
                telegram_application=None,
                punishment_engine=punishment_engine,
                position_monitor=position_monitor,
                t212=t212 if settings.paper_executes_on_t212 else None,
            )
        )

        print(f"\n=== PaperAgent.run_cycle event={args.event} allow_trades={not args.no_trades} ===\n")
        text, decisions = await agent.run_cycle(args.event, allow_trades=not args.no_trades)
        print("--- ANALYSIS TEXT ---")
        print(text[:4000])
        print("\n--- DECISIONS ---")
        print(json.dumps(decisions, indent=2)[:4000])
        print(f"\n--- {len(decisions)} decisions returned ---")
    finally:
        await http.aclose()
        await db.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
