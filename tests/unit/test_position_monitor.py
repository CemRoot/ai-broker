"""Unit tests for PositionMonitor invalidation flow."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.position_monitor import PositionMonitor, InvalidationResult
from app.services.llm.groq_service import LLMResponse


@dataclass
class _FakePos:
    ticker: str
    shares: float
    avg_cost: float


@pytest.mark.asyncio
async def test_check_invalidations_skips_when_no_invalidation():
    broker = AsyncMock()
    broker.get_positions.return_value = [_FakePos("AAPL", 10.0, 150.0)]
    broker.get_latest_invalidation_for_ticker.return_value = None

    tools = AsyncMock()
    cerebras = None
    groq = MagicMock()

    mon = PositionMonitor(
        paper_broker=broker,
        cerebras=cerebras,
        groq=groq,
        tool_executor=tools,
    )
    out = await mon.check_invalidations()
    assert out == []
    tools.execute.assert_not_called()


@pytest.mark.asyncio
async def test_check_invalidations_forces_sell_when_llm_says_met(monkeypatch):
    broker = AsyncMock()
    broker.get_positions.return_value = [_FakePos("AAPL", 10.0, 150.0)]
    broker.get_latest_invalidation_for_ticker.return_value = "Close below 140 on 4h"

    tools = AsyncMock()
    tools.execute.return_value = '{"rsi_14": 30}'

    cerebras = MagicMock()
    cerebras.analyze = AsyncMock(
        return_value=LLMResponse(text='{"met": true, "reason": "broke support"}', model="x")
    )
    groq = MagicMock()

    mon = PositionMonitor(
        paper_broker=broker,
        cerebras=cerebras,
        groq=groq,
        tool_executor=tools,
    )
    out = await mon.check_invalidations()
    assert len(out) == 1
    assert out[0]["action"] == "SELL"
    assert out[0]["ticker"] == "AAPL"
    assert out[0]["shares"] == 10.0
    tools.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_parse_llm_answer_false():
    r = PositionMonitor._parse_llm_answer("noise then {\"met\": false, \"reason\": \"ok\"}")
    assert r == InvalidationResult(met=False, reason="ok")
