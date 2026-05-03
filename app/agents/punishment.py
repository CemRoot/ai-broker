"""
Punishment / learning engine (Faz 3g).

This is a minimal implementation aligned with the existing `punishment_log` table
from `sql/schemas/001_memory.sql`:
    (ticker, penalty_type, reason, expires_at, created_at)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from app.core.logging import get_logger
from app.memory.database import SupabaseDatabase
from app.memory.retriever import RAGRetriever

log = get_logger("punishment")


@dataclass(frozen=True)
class Punishment:
    ticker: str
    penalty_type: str
    reason: str
    expires_at: datetime | None


class PunishmentEngine:
    def __init__(self, *, db: SupabaseDatabase, retriever: RAGRetriever | None) -> None:
        self._db = db
        self._retriever = retriever

    async def get_active_punishments(self) -> list[Punishment]:
        pool = self._db.get_pool()
        if pool is None:
            return []
        q = """
        SELECT ticker, penalty_type, reason, expires_at
        FROM punishment_log
        WHERE expires_at IS NULL OR expires_at > NOW()
        ORDER BY created_at DESC
        LIMIT 200
        """
        async with pool.acquire() as conn:
            rows = await conn.fetch(q)
        out: list[Punishment] = []
        for r in rows:
            out.append(
                Punishment(
                    ticker=str(r.get("ticker", "")).upper(),
                    penalty_type=str(r.get("penalty_type", "")),
                    reason=str(r.get("reason", "")),
                    expires_at=r.get("expires_at"),
                )
            )
        return out

    async def is_punished(self, ticker: str) -> bool:
        t = ticker.upper().strip()
        if not t:
            return False
        pool = self._db.get_pool()
        if pool is None:
            return False
        q = """
        SELECT 1
        FROM punishment_log
        WHERE ticker = $1 AND (expires_at IS NULL OR expires_at > NOW())
        LIMIT 1
        """
        async with pool.acquire() as conn:
            row = await conn.fetchrow(q, t)
        return bool(row)

    async def check_and_punish(
        self,
        *,
        ticker: str,
        pnl_percent: float,
        reasoning: str,
        technical_at_entry: dict[str, Any] | None = None,
    ) -> None:
        """
        Tiered loss / reward ladder (after a SELL with realized pnl_percent).

        Wins:
          - pnl >= +10%           → SUCCESS memory (BIG_WIN)
          - +5% <= pnl < +10%     → SUCCESS memory (WIN)
          - +0% <= pnl < +5%      → no log (small profit, neutral)
          - 3 consecutive wins    → SUCCESS_STREAK memory (positive bias for next setups)

        Losses (graduated; harsher with magnitude / streak):
          - -3% < pnl < 0%        → INFO log only (within noise band)
          - -5% <= pnl <= -3%     → CONFIDENCE_CUT 1d
          - -8% < pnl < -5%       → CONFIDENCE_CUT 2d + LESSON memory
          - -10% < pnl <= -8%     → COOLDOWN 3d + LESSON memory (severe loss)
          - pnl <= -10%           → COOLDOWN 7d + LESSON memory (blow-up; circuit breaker)
          - 3+ consecutive losses → COOLDOWN 3d + LESSON memory (regardless of magnitude)
        """
        t = ticker.upper().strip()
        if not t:
            return

        if pnl_percent >= 10.0:
            await self._record_success(
                ticker=t, pnl_percent=pnl_percent, reasoning=reasoning, label="BIG_WIN"
            )
            await self._maybe_record_win_streak(t, reasoning)
            return
        if pnl_percent >= 5.0:
            await self._record_success(
                ticker=t, pnl_percent=pnl_percent, reasoning=reasoning, label="WIN"
            )
            await self._maybe_record_win_streak(t, reasoning)
            return
        if pnl_percent >= 0.0:
            return

        if pnl_percent > -3.0:
            log.info("Loss within noise band on %s: %.1f%% (no punishment)", t, pnl_percent)
            return

        streak = await self._consecutive_loss_streak(t)
        if streak >= 3:
            await self._apply_punishment(
                ticker=t,
                pnl_percent=pnl_percent,
                penalty_type="COOLDOWN",
                days=3,
                reasoning=f"{reasoning} (streak={streak} consecutive losing SELLs)",
                technical_at_entry=technical_at_entry,
            )
            return

        if pnl_percent <= -10.0:
            await self._apply_punishment(
                ticker=t,
                pnl_percent=pnl_percent,
                penalty_type="COOLDOWN",
                days=7,
                reasoning=f"{reasoning} (blow-up loss; 7d circuit breaker)",
                technical_at_entry=technical_at_entry,
            )
        elif pnl_percent <= -8.0:
            await self._apply_punishment(
                ticker=t,
                pnl_percent=pnl_percent,
                penalty_type="COOLDOWN",
                days=3,
                reasoning=reasoning,
                technical_at_entry=technical_at_entry,
            )
        elif pnl_percent <= -5.0:
            await self._apply_punishment(
                ticker=t,
                pnl_percent=pnl_percent,
                penalty_type="CONFIDENCE_CUT",
                days=2,
                reasoning=reasoning,
                technical_at_entry=technical_at_entry,
            )
        else:
            await self._apply_punishment(
                ticker=t,
                pnl_percent=pnl_percent,
                penalty_type="CONFIDENCE_CUT",
                days=1,
                reasoning=reasoning,
                technical_at_entry=technical_at_entry,
            )

    async def _consecutive_loss_streak(self, ticker: str) -> int:
        pool = self._db.get_pool()
        if pool is None:
            return 0
        q = """
        SELECT pnl_percent
        FROM paper_trades
        WHERE ticker = $1 AND UPPER(action) = 'SELL' AND pnl_percent IS NOT NULL
        ORDER BY created_at DESC
        LIMIT 30
        """
        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(q, ticker.upper())
        except Exception:
            return 0
        streak = 0
        for r in rows:
            try:
                p = float(r["pnl_percent"])
            except Exception:
                break
            if p < 0:
                streak += 1
            else:
                break
        return streak

    async def _apply_punishment(
        self,
        *,
        ticker: str,
        pnl_percent: float,
        penalty_type: str,
        days: int,
        reasoning: str,
        technical_at_entry: dict[str, Any] | None,
    ) -> None:
        expires_at = datetime.now(timezone.utc) + timedelta(days=max(1, int(days)))
        reason = (
            f"Loss on {ticker}: {pnl_percent:.1f}%. "
            f"Entry reasoning: {reasoning or 'N/A'}."
        )

        pool = self._db.get_pool()
        if pool is not None:
            q = """
            INSERT INTO punishment_log (ticker, penalty_type, reason, expires_at)
            VALUES ($1, $2, $3, $4)
            """
            async with pool.acquire() as conn:
                await conn.execute(q, ticker, penalty_type, reason, expires_at)

        if self._retriever:
            await self._retriever.add_memory(
                ticker=ticker,
                memory_type="LESSON",
                context=reason,
                outcome="LOSS",
                pnl_percent=float(pnl_percent),
            )

        log.info("Punishment applied: %s %s until %s", ticker, penalty_type, expires_at.isoformat())

    async def _record_success(
        self, *, ticker: str, pnl_percent: float, reasoning: str, label: str = "WIN"
    ) -> None:
        if not self._retriever:
            return
        ctx = (
            f"SUCCESS [{label}] on {ticker}: +{pnl_percent:.1f}%. "
            f"Setup/reasoning: {reasoning or 'N/A'}."
        )
        await self._retriever.add_memory(
            ticker=ticker,
            memory_type="SUCCESS",
            context=ctx,
            outcome="WIN",
            pnl_percent=float(pnl_percent),
        )

    async def _consecutive_win_streak(self, ticker: str) -> int:
        pool = self._db.get_pool()
        if pool is None:
            return 0
        q = """
        SELECT pnl_percent
        FROM paper_trades
        WHERE ticker = $1 AND UPPER(action) = 'SELL' AND pnl_percent IS NOT NULL
        ORDER BY created_at DESC
        LIMIT 30
        """
        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(q, ticker.upper())
        except Exception:
            return 0
        streak = 0
        for r in rows:
            try:
                p = float(r["pnl_percent"])
            except Exception:
                break
            if p > 0:
                streak += 1
            else:
                break
        return streak

    async def _maybe_record_win_streak(self, ticker: str, reasoning: str) -> None:
        if not self._retriever:
            return
        streak = await self._consecutive_win_streak(ticker)
        if streak < 3:
            return
        ctx = (
            f"SUCCESS_STREAK on {ticker}: {streak} consecutive winning SELLs. "
            f"Last setup/reasoning: {reasoning or 'N/A'}. "
            f"Pattern is reproducible — slightly favor similar setups; do not chase, keep risk discipline."
        )
        await self._retriever.add_memory(
            ticker=ticker,
            memory_type="SUCCESS",
            context=ctx,
            outcome="WIN",
            pnl_percent=0.0,
        )
        log.info("Win streak recorded for %s (streak=%d)", ticker, streak)

