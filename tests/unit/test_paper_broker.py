import pytest
from unittest.mock import AsyncMock, MagicMock
from app.services.paper.broker import (
    PaperBroker,
    PaperBrokerError,
    InsufficientFundsError,
    ShortSellingNotAllowedError,
)
from app.memory.database import SupabaseDatabase

@pytest.fixture
def mock_db():
    db = MagicMock(spec=SupabaseDatabase)
    pool = AsyncMock()
    db.get_pool.return_value = pool

    conn = AsyncMock()
    # asyncpg: ``async with pool.acquire()`` — acquire() is sync and returns an async CM, not a coroutine.
    acquire_cm = MagicMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=None)
    pool.acquire = MagicMock(return_value=acquire_cm)

    tx_ctx = MagicMock()
    tx_ctx.__aenter__ = AsyncMock(return_value=None)
    tx_ctx.__aexit__ = AsyncMock(return_value=None)
    conn.transaction = MagicMock(return_value=tx_ctx)

    return db, pool, conn

@pytest.mark.asyncio
async def test_paper_broker_get_balance(mock_db):
    db, pool, conn = mock_db
    conn.fetchrow.return_value = {'balance': 20000.0}
    
    broker = PaperBroker(db)
    balance = await broker.get_balance()
    
    assert balance == 20000.0
    conn.fetchrow.assert_called_once_with("SELECT balance FROM paper_account WHERE id = 1")

@pytest.mark.asyncio
async def test_paper_broker_buy_insufficient_funds(mock_db):
    db, pool, conn = mock_db
    conn.fetchrow.return_value = {'balance': 100.0} # Only $100
    
    broker = PaperBroker(db)
    
    with pytest.raises(InsufficientFundsError):
        # Trying to buy $200 worth of shares
        await broker.buy("AAPL", shares=2.0, price=100.0, reasoning="Test")

@pytest.mark.asyncio
async def test_paper_broker_buy_success(mock_db):
    db, pool, conn = mock_db
    
    # Define side effects for multiple fetchrow calls
    async def fetchrow_side_effect(*args, **kwargs):
        query = args[0]
        if "paper_account" in query:
            return {'balance': 20000.0}
        elif "paper_portfolio" in query:
            return None # No existing position
            
    conn.fetchrow.side_effect = fetchrow_side_effect
    conn.fetchval.return_value = 1 # Fake trade ID
    
    broker = PaperBroker(db)
    trade = await broker.buy("AAPL", shares=10.0, price=150.0, reasoning="Strong setup")
    
    assert trade.id == 1
    assert trade.ticker == "AAPL"
    assert trade.action == "BUY"
    assert trade.total_value == 1500.0

@pytest.mark.asyncio
async def test_paper_broker_sell_short_error(mock_db):
    db, pool, conn = mock_db
    
    # No existing position
    conn.fetchrow.return_value = None
    
    broker = PaperBroker(db)
    
    with pytest.raises(ShortSellingNotAllowedError):
        await broker.sell("AAPL", shares=10.0, price=150.0)

@pytest.mark.asyncio
async def test_paper_broker_sell_success(mock_db):
    db, pool, conn = mock_db
    
    async def fetchrow_side_effect(*args, **kwargs):
        query = args[0]
        if "paper_portfolio" in query:
            return {'ticker': 'AAPL', 'shares': 10.0, 'avg_cost': 100.0}
            
    conn.fetchrow.side_effect = fetchrow_side_effect
    conn.fetchval.return_value = 2 # Fake trade ID
    
    broker = PaperBroker(db)
    trade = await broker.sell("AAPL", shares=5.0, price=150.0)
    
    assert trade.id == 2
    assert trade.action == "SELL"
    assert trade.pnl_percent == 50.0  # Bought at 100, sold at 150
    assert trade.realized_pnl_usd == 250.0  # 5 shares * ($150 - $100)


@pytest.mark.asyncio
async def test_paper_broker_t212_mode_blocks_buy(mock_db):
    db, pool, conn = mock_db
    broker = PaperBroker(db, paper_executes_on_t212=True)
    with pytest.raises(PaperBrokerError, match="Virtual ledger buy is disabled"):
        await broker.buy("AAPL", shares=1.0, price=10.0)


@pytest.mark.asyncio
async def test_paper_broker_t212_mode_blocks_sell(mock_db):
    db, pool, conn = mock_db
    broker = PaperBroker(db, paper_executes_on_t212=True)
    with pytest.raises(PaperBrokerError, match="Virtual ledger sell is disabled"):
        await broker.sell("AAPL", shares=1.0, price=10.0)


@pytest.mark.asyncio
async def test_record_mirror_t212_requires_order_id(mock_db):
    db, pool, conn = mock_db
    broker = PaperBroker(db)
    with pytest.raises(PaperBrokerError, match="t212_order_id"):
        await broker.record_mirror_trade(
            ticker="AAPL",
            action="BUY",
            shares=1.0,
            price=10.0,
            total_value=10.0,
            t212_order_id=None,
            execution_broker="t212",
        )


@pytest.mark.asyncio
async def test_paper_broker_reset_all(mock_db):
    db, pool, conn = mock_db
    broker = PaperBroker(db)
    await broker.reset_all(starting_balance=20_000.0)
    assert conn.execute.await_count == 3
    calls = [c.args[0] for c in conn.execute.await_args_list]
    assert any("DELETE FROM paper_trades" in str(q) for q in calls)
    assert any("DELETE FROM paper_portfolio" in str(q) for q in calls)
    assert any("UPDATE paper_account" in str(q) for q in calls)
