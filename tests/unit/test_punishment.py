"""Unit tests for PunishmentEngine."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.punishment import PunishmentEngine, Punishment


def _mock_pool(fetch_rows: list | None = None):
    db = MagicMock()
    pool = MagicMock()
    conn = AsyncMock()
    db.get_pool.return_value = pool

    class _AcquireCM:
        def __init__(self, c: AsyncMock) -> None:
            self._c = c

        async def __aenter__(self):
            return self._c

        async def __aexit__(self, *args):
            return None

    pool.acquire = MagicMock(side_effect=lambda: _AcquireCM(conn))
    if fetch_rows is not None:
        conn.fetch = AsyncMock(return_value=fetch_rows)
    conn.fetchrow = AsyncMock(return_value=None)
    conn.execute = AsyncMock()
    return db, conn


@pytest.mark.asyncio
async def test_check_and_punish_success_writes_success_memory():
    db, conn = _mock_pool(fetch_rows=[])
    retriever = AsyncMock()
    eng = PunishmentEngine(db=db, retriever=retriever)
    await eng.check_and_punish(ticker="AAPL", pnl_percent=6.5, reasoning="target hit", technical_at_entry=None)
    retriever.add_memory.assert_awaited_once()
    call_kw = retriever.add_memory.await_args.kwargs
    assert call_kw["memory_type"] == "SUCCESS"
    assert call_kw["ticker"] == "AAPL"
    conn.execute.assert_not_called()


@pytest.mark.asyncio
async def test_check_and_punish_three_loss_streak_cooldown():
    rows = [{"pnl_percent": -1.0}, {"pnl_percent": -2.0}]
    db, conn = _mock_pool(
        fetch_rows=rows,
    )
    retriever = AsyncMock()
    eng = PunishmentEngine(db=db, retriever=retriever)
    # Third losing SELL is current; fetch order is DESC so first row is current trade (-3)
    conn.fetch = AsyncMock(
        return_value=[
            {"pnl_percent": -3.0},
            {"pnl_percent": -2.0},
            {"pnl_percent": -1.0},
        ]
    )
    await eng.check_and_punish(ticker="TSLA", pnl_percent=-3.0, reasoning="stop", technical_at_entry=None)
    conn.execute.assert_awaited()
    retriever.add_memory.assert_awaited()
    assert retriever.add_memory.await_args.kwargs["memory_type"] == "LESSON"


@pytest.mark.asyncio
async def test_check_and_punish_noise_band_no_action():
    """Loss within ±3% noise band must not log to punishment_log nor RAG."""
    db, conn = _mock_pool(fetch_rows=[])
    retriever = AsyncMock()
    eng = PunishmentEngine(db=db, retriever=retriever)
    await eng.check_and_punish(
        ticker="AAPL", pnl_percent=-2.0, reasoning="noise stop-out", technical_at_entry=None
    )
    conn.execute.assert_not_called()
    retriever.add_memory.assert_not_called()


@pytest.mark.asyncio
async def test_check_and_punish_blowup_triggers_seven_day_cooldown():
    """Single -12% blow-up triggers a 7-day cooldown (circuit breaker), not the milder 3-day path."""
    db, conn = _mock_pool(fetch_rows=[])
    conn.fetch = AsyncMock(return_value=[{"pnl_percent": -12.0}])
    retriever = AsyncMock()
    eng = PunishmentEngine(db=db, retriever=retriever)
    await eng.check_and_punish(
        ticker="NVDA", pnl_percent=-12.0, reasoning="thesis broken", technical_at_entry=None
    )
    conn.execute.assert_awaited()
    sql_args = conn.execute.await_args.args
    assert "INSERT INTO punishment_log" in sql_args[0]
    assert sql_args[2] == "COOLDOWN"
    delta_days = (sql_args[4] - datetime.now(timezone.utc)).days
    assert delta_days >= 6
    retriever.add_memory.assert_awaited()
    assert retriever.add_memory.await_args.kwargs["memory_type"] == "LESSON"


@pytest.mark.asyncio
async def test_check_and_punish_win_streak_records_success_streak():
    """3+ consecutive winning SELLs trigger an extra SUCCESS_STREAK memory."""
    db, conn = _mock_pool()
    conn.fetch = AsyncMock(
        return_value=[
            {"pnl_percent": 11.0},
            {"pnl_percent": 8.0},
            {"pnl_percent": 6.5},
        ]
    )
    retriever = AsyncMock()
    eng = PunishmentEngine(db=db, retriever=retriever)
    await eng.check_and_punish(
        ticker="MSFT", pnl_percent=11.0, reasoning="breakout target hit", technical_at_entry=None
    )
    assert retriever.add_memory.await_count == 2
    contexts = [c.kwargs["context"] for c in retriever.add_memory.await_args_list]
    assert any("SUCCESS_STREAK" in c for c in contexts)
    assert any("SUCCESS [BIG_WIN]" in c for c in contexts)


@pytest.mark.asyncio
async def test_get_active_punishments_maps_rows():
    expires = datetime(2030, 1, 1, tzinfo=timezone.utc)
    db, conn = _mock_pool()
    conn.fetch = AsyncMock(
        return_value=[
            {"ticker": "nvda", "penalty_type": "COOLDOWN", "reason": "loss", "expires_at": expires},
        ]
    )
    eng = PunishmentEngine(db=db, retriever=None)
    out = await eng.get_active_punishments()
    assert len(out) == 1
    assert isinstance(out[0], Punishment)
    assert out[0].ticker == "NVDA"
    assert out[0].penalty_type == "COOLDOWN"
