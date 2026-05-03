"""T212 mirror: immediate fill → record_mirror_trade; pending → enqueue for poller."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import app.agents.paper_agent as pa_mod
from app.agents.paper_agent import PaperAgent


@pytest.mark.asyncio
async def test_t212_buy_pending_enqueues_poller() -> None:
    t212 = AsyncMock()
    t212.place_market_order = AsyncMock(
        return_value={"id": 42_001, "filledQuantity": 0, "filledValue": 0},
    )
    paper_broker = AsyncMock()
    mock_db = MagicMock()
    mock_enqueue = AsyncMock()
    agent = PaperAgent.__new__(PaperAgent)
    agent.deps = SimpleNamespace(
        settings=SimpleNamespace(paper_t212_extended_hours=False),
        t212=t212,
        paper_broker=paper_broker,
        db=mock_db,
    )
    orig = pa_mod.enqueue_t212_pending_mirror
    pa_mod.enqueue_t212_pending_mirror = mock_enqueue
    try:
        r = await PaperAgent._t212_execute_buy(
            agent,
            ticker="AAPL",
            shares=2.0,
            px=100.0,
            reasoning="",
            stop_loss_f=None,
            target_f=None,
            invalid_s=None,
            cot=None,
            cycle_event=None,
            emergency=False,
        )
        assert r is None
        paper_broker.record_mirror_trade.assert_not_called()
        mock_enqueue.assert_awaited_once()
        assert mock_enqueue.await_args.kwargs["t212_order_id"] == 42_001
        assert mock_enqueue.await_args.kwargs["action"] == "BUY"
    finally:
        pa_mod.enqueue_t212_pending_mirror = orig


@pytest.mark.asyncio
async def test_t212_buy_filled_calls_record_mirror() -> None:
    t212 = AsyncMock()
    t212.place_market_order = AsyncMock(
        return_value={"id": 42_002, "filledQuantity": 2.0, "filledValue": 200.0},
    )
    paper_broker = AsyncMock()
    paper_broker.record_mirror_trade = AsyncMock(return_value=SimpleNamespace(id=1))
    agent = PaperAgent.__new__(PaperAgent)
    agent.deps = SimpleNamespace(
        settings=SimpleNamespace(paper_t212_extended_hours=False),
        t212=t212,
        paper_broker=paper_broker,
        db=MagicMock(),
    )
    mock_enqueue = AsyncMock()
    orig = pa_mod.enqueue_t212_pending_mirror
    pa_mod.enqueue_t212_pending_mirror = mock_enqueue
    try:
        await PaperAgent._t212_execute_buy(
            agent,
            ticker="AAPL",
            shares=2.0,
            px=100.0,
            reasoning="x",
            stop_loss_f=None,
            target_f=None,
            invalid_s=None,
            cot=None,
            cycle_event=None,
            emergency=False,
        )
        paper_broker.record_mirror_trade.assert_awaited_once()
        mock_enqueue.assert_not_called()
        kwargs = paper_broker.record_mirror_trade.await_args.kwargs
        assert kwargs["t212_order_id"] == 42_002
        assert kwargs["shares"] == 2.0
        assert kwargs["price"] == 100.0
    finally:
        pa_mod.enqueue_t212_pending_mirror = orig
