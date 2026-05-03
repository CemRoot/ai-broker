import logging
from typing import Any, List, Optional, Dict
import asyncpg

from app.memory.database import SupabaseDatabase
from app.services.paper.models import PaperAccount, PaperPosition, PaperTrade
from app.services.t212.client import T212Client

log = logging.getLogger(__name__)

class PaperBrokerError(Exception):
    pass

class InsufficientFundsError(PaperBrokerError):
    pass

class ShortSellingNotAllowedError(PaperBrokerError):
    pass

class PaperBroker:
    """Manages virtual trades, balances, and PnL calculation for the autonomous agent."""

    def __init__(self, db: SupabaseDatabase, *, paper_executes_on_t212: bool = False):
        self.db = db
        self._paper_executes_on_t212 = bool(paper_executes_on_t212)

    async def _get_conn(self) -> asyncpg.Pool:
        pool = self.db.get_pool()
        if not pool:
            raise PaperBrokerError("Database pool is not initialized.")
        return pool

    async def t212_mirror_trade_exists(self, t212_order_id: int) -> bool:
        """True if ``paper_trades`` already has a T212 mirror row for this order id."""
        pool = self.db.get_pool()
        if not pool:
            return False
        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT 1 FROM paper_trades
                    WHERE t212_order_id = $1 AND execution_broker = 't212'
                    LIMIT 1
                    """,
                    int(t212_order_id),
                )
        except Exception as exc:
            log.debug("t212_mirror_trade_exists failed (schema 004 applied?): %s", exc)
            return False
        return row is not None

    async def sync_supabase_ledger_from_t212(
        self,
        *,
        account_summary: dict[str, Any],
        positions: list[Any],
    ) -> None:
        """Overwrite ``paper_account`` cash and ``paper_portfolio`` rows from T212 (account currency)."""
        if not self._paper_executes_on_t212:
            return
        cash_block = account_summary.get("cash") or {}
        cash = float(cash_block.get("availableToTrade") or 0.0)

        pool = self.db.get_pool()
        if not pool:
            log.warning("sync_supabase_ledger_from_t212: no DB pool")
            return

        from app.services.t212.ticker_map import t212_to_yfinance

        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO paper_account (id, balance)
                    VALUES (1, $1)
                    ON CONFLICT (id) DO UPDATE SET balance = EXCLUDED.balance, updated_at = NOW()
                    """,
                    cash,
                )
                await conn.execute("DELETE FROM paper_portfolio")
                for p in positions:
                    qty = float(getattr(p, "quantity", 0.0) or 0.0)
                    if qty <= 1e-9:
                        continue
                    t212_sym = p.ticker if hasattr(p, "ticker") else ""
                    yf = t212_to_yfinance(str(t212_sym))
                    avg = float(getattr(p, "average_price_paid", 0.0) or 0.0)
                    cur_px = float(getattr(p, "current_price", 0.0) or 0.0)
                    cur_val = qty * cur_px if cur_px > 0 else None
                    await conn.execute(
                        """
                        INSERT INTO paper_portfolio (ticker, shares, avg_cost, current_value, status)
                        VALUES ($1, $2, $3, $4, 'OPEN')
                        """,
                        yf,
                        qty,
                        avg,
                        cur_val,
                    )
        log.info(
            "T212 → Supabase ledger sync: cash=%.2f, open positions=%s",
            cash,
            len([p for p in positions if float(getattr(p, "quantity", 0.0) or 0.0) > 1e-9]),
        )

    async def sync_ledger_from_t212_client(self, t212: T212Client) -> None:
        """Fetch summary + positions from T212 and sync shadow ledger (rate-limited client)."""
        if not self._paper_executes_on_t212:
            return
        summary = await t212.get_account_summary()
        positions = await t212.get_positions()
        await self.sync_supabase_ledger_from_t212(account_summary=summary, positions=positions)

    async def get_balance(self) -> float:
        """Returns the current available cash balance in USD."""
        pool = await self._get_conn()
        async with pool.acquire() as conn:
            row = await conn.fetchrow("SELECT balance FROM paper_account WHERE id = 1")
            if not row:
                raise PaperBrokerError("Paper account not found in database.")
            return float(row['balance'])

    async def get_positions(self) -> List[PaperPosition]:
        """Returns all open positions."""
        pool = await self._get_conn()
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM paper_portfolio WHERE status = 'OPEN' AND shares > 0")
            return [PaperPosition.model_validate(dict(row)) for row in rows]

    async def get_recent_trades(self, limit: int = 10) -> List[PaperTrade]:
        """Return recent paper trades (most recent first)."""
        pool = await self._get_conn()
        lim = max(1, min(int(limit), 50))
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM paper_trades ORDER BY created_at DESC LIMIT $1",
                lim,
            )
            return [PaperTrade.model_validate(dict(row)) for row in rows]

    async def get_latest_invalidation_for_ticker(self, ticker: str) -> Optional[str]:
        """Latest non-empty invalidation_condition from a BUY row for this ticker."""
        pool = await self._get_conn()
        t = ticker.upper().strip()
        if not t:
            return None
        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT invalidation_condition
                    FROM paper_trades
                    WHERE ticker = $1
                      AND action = 'BUY'
                      AND invalidation_condition IS NOT NULL
                      AND TRIM(invalidation_condition) <> ''
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    t,
                )
        except Exception:
            return None
        if not row or row.get("invalidation_condition") is None:
            return None
        s = str(row["invalidation_condition"]).strip()
        return s or None

    async def buy(
        self,
        ticker: str,
        shares: float,
        price: float,
        reasoning: str = "",
        *,
        stop_loss: float | None = None,
        target: float | None = None,
        invalidation_condition: str | None = None,
        chain_of_thought: str | None = None,
        cycle_event: str | None = None,
        emergency: bool = False,
    ) -> PaperTrade:
        """Executes a virtual buy order, deducting balance and adding to portfolio."""
        if shares <= 0 or price <= 0:
            raise ValueError("Shares and price must be positive.")
        if self._paper_executes_on_t212:
            raise PaperBrokerError(
                "Virtual ledger buy is disabled when paper_executes_on_t212=True; "
                "execute on T212 first, then record_mirror_trade with t212_order_id."
            )

        total_cost = shares * price
        pool = await self._get_conn()
        
        async with pool.acquire() as conn:
            async with conn.transaction():
                # 1. Check balance
                account = await conn.fetchrow("SELECT balance FROM paper_account WHERE id = 1 FOR UPDATE")
                if not account:
                    raise PaperBrokerError("Paper account not found.")
                    
                current_balance = float(account['balance'])
                if current_balance < total_cost:
                    raise InsufficientFundsError(f"Insufficient funds: Have ${current_balance:.2f}, need ${total_cost:.2f}")
                
                # 2. Deduct balance
                new_balance = current_balance - total_cost
                await conn.execute("UPDATE paper_account SET balance = $1, updated_at = NOW() WHERE id = 1", new_balance)
                
                # 3. Update or Insert portfolio
                pos = await conn.fetchrow("SELECT * FROM paper_portfolio WHERE ticker = $1 FOR UPDATE", ticker)
                if pos:
                    current_shares = float(pos['shares'])
                    current_avg_cost = float(pos['avg_cost'])
                    
                    # Calculate new avg cost
                    total_spent_before = current_shares * current_avg_cost
                    new_shares = current_shares + shares
                    new_avg_cost = (total_spent_before + total_cost) / new_shares
                    
                    await conn.execute("""
                        UPDATE paper_portfolio 
                        SET shares = $1, avg_cost = $2, status = 'OPEN', updated_at = NOW()
                        WHERE ticker = $3
                    """, new_shares, new_avg_cost, ticker)
                else:
                    await conn.execute("""
                        INSERT INTO paper_portfolio (ticker, shares, avg_cost, status)
                        VALUES ($1, $2, $3, 'OPEN')
                    """, ticker, shares, price)
                
                # 4. Insert trade log
                # Try extended schema (Faz 3). If the DB hasn't applied it yet, fall back.
                try:
                    trade_id = await conn.fetchval(
                        """
                        INSERT INTO paper_trades (
                            ticker, action, shares, price, total_value, reasoning,
                            stop_loss, target, invalidation_condition,
                            chain_of_thought, cycle_event, emergency
                        )
                        VALUES ($1, 'BUY', $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                        RETURNING id
                        """,
                        ticker,
                        shares,
                        price,
                        total_cost,
                        reasoning,
                        stop_loss,
                        target,
                        invalidation_condition,
                        chain_of_thought,
                        cycle_event,
                        bool(emergency),
                    )
                except Exception:
                    trade_id = await conn.fetchval(
                        """
                        INSERT INTO paper_trades (ticker, action, shares, price, total_value, reasoning)
                        VALUES ($1, 'BUY', $2, $3, $4, $5)
                        RETURNING id
                        """,
                        ticker,
                        shares,
                        price,
                        total_cost,
                        reasoning,
                    )
                
                log.info(f"PAPER BUY: {shares} {ticker} @ ${price:.2f}. Cost: ${total_cost:.2f}")
                
                return PaperTrade(
                    id=trade_id,
                    ticker=ticker,
                    action="BUY",
                    shares=shares,
                    price=price,
                    total_value=total_cost,
                    reasoning=reasoning,
                    stop_loss=stop_loss,
                    target=target,
                    invalidation_condition=invalidation_condition,
                    chain_of_thought=chain_of_thought,
                    cycle_event=cycle_event,
                    emergency=bool(emergency),
                )

    async def sell(
        self,
        ticker: str,
        shares: float,
        price: float,
        reasoning: str = "",
        *,
        stop_loss: float | None = None,
        target: float | None = None,
        invalidation_condition: str | None = None,
        chain_of_thought: str | None = None,
        cycle_event: str | None = None,
        emergency: bool = False,
    ) -> PaperTrade:
        """Executes a virtual sell order, checking short constraints and calculating PnL."""
        if shares <= 0 or price <= 0:
            raise ValueError("Shares and price must be positive.")
        if self._paper_executes_on_t212:
            raise PaperBrokerError(
                "Virtual ledger sell is disabled when paper_executes_on_t212=True; "
                "execute on T212 first, then record_mirror_trade with t212_order_id."
            )

        total_revenue = shares * price
        pool = await self._get_conn()
        
        async with pool.acquire() as conn:
            async with conn.transaction():
                # 1. Check portfolio
                pos = await conn.fetchrow("SELECT * FROM paper_portfolio WHERE ticker = $1 FOR UPDATE", ticker)
                if not pos:
                    raise ShortSellingNotAllowedError(f"Cannot sell {ticker}: No open position found.")
                    
                current_shares = float(pos['shares'])
                avg_cost = float(pos['avg_cost'])
                
                if shares > current_shares:
                    raise ShortSellingNotAllowedError(f"Cannot sell {shares} shares of {ticker}: Only have {current_shares}.")
                    
                # 2. Update portfolio
                new_shares = current_shares - shares
                status = 'CLOSED' if new_shares == 0 else 'OPEN'
                
                await conn.execute("""
                    UPDATE paper_portfolio 
                    SET shares = $1, status = $2, updated_at = NOW()
                    WHERE ticker = $3
                """, new_shares, status, ticker)
                
                # 3. Add to balance
                await conn.execute("""
                    UPDATE paper_account 
                    SET balance = balance + $1, updated_at = NOW() 
                    WHERE id = 1
                """, total_revenue)
                
                # 4. Calculate PnL
                cost_basis = shares * avg_cost
                realized_pnl_dollars = total_revenue - cost_basis
                pnl_percent = (realized_pnl_dollars / cost_basis) * 100.0 if cost_basis > 0 else 0.0
                
                # 5. Insert trade log
                try:
                    trade_id = await conn.fetchval(
                        """
                        INSERT INTO paper_trades (
                            ticker, action, shares, price, total_value, reasoning,
                            pnl_percent, realized_pnl_usd,
                            stop_loss, target, invalidation_condition,
                            chain_of_thought, cycle_event, emergency
                        )
                        VALUES ($1, 'SELL', $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                        RETURNING id
                        """,
                        ticker,
                        shares,
                        price,
                        total_revenue,
                        reasoning,
                        pnl_percent,
                        realized_pnl_dollars,
                        stop_loss,
                        target,
                        invalidation_condition,
                        chain_of_thought,
                        cycle_event,
                        bool(emergency),
                    )
                except Exception:
                    trade_id = await conn.fetchval(
                        """
                        INSERT INTO paper_trades (
                            ticker, action, shares, price, total_value, reasoning,
                            pnl_percent, realized_pnl_usd
                        )
                        VALUES ($1, 'SELL', $2, $3, $4, $5, $6, $7)
                        RETURNING id
                        """,
                        ticker,
                        shares,
                        price,
                        total_revenue,
                        reasoning,
                        pnl_percent,
                        realized_pnl_dollars,
                    )
                
                log.info(
                    "PAPER SELL: %s %s @ $%.2f. Revenue: $%.2f, PnL: %.2f%% ($%.2f)",
                    shares, ticker, price, total_revenue, pnl_percent, realized_pnl_dollars,
                )
                
                return PaperTrade(
                    id=trade_id, ticker=ticker, action="SELL", shares=shares,
                    price=price, total_value=total_revenue, reasoning=reasoning,
                    pnl_percent=pnl_percent, realized_pnl_usd=realized_pnl_dollars,
                    stop_loss=stop_loss,
                    target=target,
                    invalidation_condition=invalidation_condition,
                    chain_of_thought=chain_of_thought,
                    cycle_event=cycle_event,
                    emergency=bool(emergency),
                )

    async def record_mirror_trade(
        self,
        *,
        ticker: str,
        action: str,
        shares: float,
        price: float,
        total_value: float,
        reasoning: str = "",
        stop_loss: float | None = None,
        target: float | None = None,
        invalidation_condition: str | None = None,
        chain_of_thought: str | None = None,
        cycle_event: str | None = None,
        emergency: bool = False,
        pnl_percent: float | None = None,
        realized_pnl_usd: float | None = None,
        t212_order_id: int | None = None,
        execution_broker: str = "t212",
    ) -> PaperTrade:
        """Audit row only: T212 (or other broker) executed the trade; Supabase ledger not updated."""
        pool = await self._get_conn()
        act = action.upper()
        if act not in ("BUY", "SELL"):
            raise ValueError("action must be BUY or SELL")
        eb = (execution_broker or "").strip().lower()
        if eb == "t212":
            if t212_order_id is None:
                raise PaperBrokerError("t212 mirror row requires t212_order_id (no fill without T212 order id).")
            try:
                oid_chk = int(t212_order_id)
            except (TypeError, ValueError) as e:
                raise PaperBrokerError("t212_order_id must be a positive integer") from e
            if oid_chk <= 0:
                raise PaperBrokerError("t212_order_id must be positive")
        async with pool.acquire() as conn:
            try:
                trade_id = await conn.fetchval(
                    """
                    INSERT INTO paper_trades (
                        ticker, action, shares, price, total_value, reasoning,
                        pnl_percent, realized_pnl_usd,
                        stop_loss, target, invalidation_condition,
                        chain_of_thought, cycle_event, emergency,
                        t212_order_id, execution_broker
                    )
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16)
                    RETURNING id
                    """,
                    ticker,
                    act,
                    shares,
                    price,
                    total_value,
                    reasoning,
                    pnl_percent,
                    realized_pnl_usd,
                    stop_loss,
                    target,
                    invalidation_condition,
                    chain_of_thought,
                    cycle_event,
                    bool(emergency),
                    t212_order_id,
                    execution_broker,
                )
            except Exception as exc:
                if eb == "t212":
                    raise PaperBrokerError(
                        "paper_trades must include t212_order_id and execution_broker "
                        "(apply sql/schemas/004_t212_paper_execution.sql)"
                    ) from exc
                trade_id = await conn.fetchval(
                    """
                    INSERT INTO paper_trades (
                        ticker, action, shares, price, total_value, reasoning,
                        pnl_percent, realized_pnl_usd,
                        stop_loss, target, invalidation_condition,
                        chain_of_thought, cycle_event, emergency
                    )
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
                    RETURNING id
                    """,
                    ticker,
                    act,
                    shares,
                    price,
                    total_value,
                    reasoning,
                    pnl_percent,
                    realized_pnl_usd,
                    stop_loss,
                    target,
                    invalidation_condition,
                    chain_of_thought,
                    cycle_event,
                    bool(emergency),
                )
        log.info(
            "PAPER MIRROR (%s): %s %s @ %.4f — t212_order_id=%s",
            execution_broker,
            act,
            ticker,
            price,
            t212_order_id,
        )
        return PaperTrade(
            id=trade_id,
            ticker=ticker,
            action=act,
            shares=shares,
            price=price,
            total_value=total_value,
            reasoning=reasoning,
            pnl_percent=pnl_percent,
            realized_pnl_usd=realized_pnl_usd,
            stop_loss=stop_loss,
            target=target,
            invalidation_condition=invalidation_condition,
            chain_of_thought=chain_of_thought,
            cycle_event=cycle_event,
            emergency=bool(emergency),
        )

    async def reset_all(self, *, starting_balance: float = 20_000.0) -> None:
        """
        Wipe paper trades and positions; set cash to ``starting_balance``.
        Does not touch ``punishment_log`` or ``trade_memories``.
        """
        pool = await self._get_conn()
        bal = float(starting_balance)
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("DELETE FROM paper_trades")
                await conn.execute("DELETE FROM paper_portfolio")
                await conn.execute(
                    "UPDATE paper_account SET balance = $1, updated_at = NOW() WHERE id = 1",
                    bal,
                )
        log.info("PAPER RESET: balance=%.2f, trades+positions cleared", bal)
