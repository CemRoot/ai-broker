"""
Faz 3 checklist: Trump → PaperAgent emergency hook (mocked LLM, no trades).

Full host checks: ``PYTHONPATH=. uv run python scripts/faz3_e2e_check.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.paper_agent import PaperAgent, PaperAgentDeps
from app.core.config import Settings
from app.services.market_clock import MarketClock


@dataclass
class _FakeToolResult:
    reasoning_text: str
    decisions: list[dict[str, Any]]
    model: str = "mock"
    iterations: int = 1


@pytest.mark.asyncio
async def test_run_emergency_cycle_calls_llm_pipeline():
    """Simulates high-impact Trump path: one emergency cycle, no broker trades if model returns[][]."""
    settings = MagicMock(spec=Settings)
    settings.paper_starting_nav_usd = 20_000.0
    settings.paper_max_drawdown_pct = 30.0
    settings.allowed_user_ids = set()
    settings.paper_executes_on_t212 = False
    settings.paper_t212_extended_hours = True

    broker = AsyncMock()
    broker.get_balance.return_value = 20_000.0
    broker.get_positions.return_value = []

    tool_x = AsyncMock()
    tool_x.execute = AsyncMock(return_value="MACRO: test")

    deps = PaperAgentDeps(
        settings=settings,
        db=MagicMock(),
        paper_broker=broker,
        groq=MagicMock(),
        ollama=None,
        retriever=None,
        tool_executor=tool_x,
        market_clock=MarketClock(),
        telegram_application=None,
        punishment_engine=None,
        position_monitor=None,
    )
    agent = PaperAgent(deps)

    fake = _FakeToolResult(reasoning_text="ok", decisions=[])

    with patch("app.agents.paper_agent.analyze_with_tools", new_callable=AsyncMock, return_value=fake):
        with patch.object(PaperAgent, "_portfolio_risk_lines", new_callable=AsyncMock, return_value=("", False)):
            with patch.object(PaperAgent, "_active_punishments_prompt_block", new_callable=AsyncMock, return_value=""):
                with patch.object(PaperAgent, "_save_cycle_log", new_callable=AsyncMock):
                    with patch.object(PaperAgent, "_send_live_feed", new_callable=AsyncMock):
                        text, dec = await agent.run_emergency_cycle(
                            trigger="E2E_SIMULATION",
                            context={
                                "post_text": "Tariff headline",
                                "impact_score": 9.0,
                                "sentiment": "bearish",
                                "affected_tickers": ["NVDA"],
                            },
                        )
    assert text == "ok"
    assert dec == []
    broker.buy.assert_not_called()
    broker.sell.assert_not_called()


@pytest.mark.asyncio
async def test_trump_monitor_sets_paper_agent_reference():
    from app.core.config import get_settings
    from app.services.trump_monitor import TrumpMonitor

    settings = get_settings()
    mon = TrumpMonitor(
        settings=settings,
        groq=None,
        http_client=MagicMock(),
        telegram_application=None,
        db=None,
        retriever=None,
    )
    sentinel = object()
    mon.set_paper_agent(sentinel)
    assert mon._paper_agent is sentinel
