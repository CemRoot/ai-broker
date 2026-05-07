"""
PositionMonitor (Faz 3h) — check invalidation conditions and suggest forced exits.

Runs on low-frequency ticks (OPEN/MIDDAY/CLOSE). Uses latest BUY `invalidation_condition`
from `paper_trades` (requires `003_paper_agent.sql` applied).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.core.logging import get_logger
from app.services.llm.cerebras_service import CerebrasService
from app.services.llm.groq_service import GroqService
from app.services.paper.broker import PaperBroker
from app.services.paper.models import PaperPosition
from app.services.t212.client import T212Client
from app.tools.executor import ToolExecutor

log = get_logger("position_monitor")


@dataclass(frozen=True)
class InvalidationResult:
    met: bool
    reason: str


class PositionMonitor:
    def __init__(
        self,
        *,
        paper_broker: PaperBroker,
        cerebras: CerebrasService | None,
        groq: GroqService | None,
        tool_executor: ToolExecutor,
        t212: T212Client | None = None,
        paper_executes_on_t212: bool = False,
    ) -> None:
        self._broker = paper_broker
        self._cerebras = cerebras
        self._groq = groq
        self._tools = tool_executor
        self._t212 = t212
        self._paper_executes_on_t212 = paper_executes_on_t212

    async def _positions_open(self) -> list[PaperPosition]:
        if self._paper_executes_on_t212 and self._t212:
            from app.services.t212.ticker_map import t212_to_yfinance

            raw = await self._t212.get_positions()
            return [
                PaperPosition(
                    ticker=t212_to_yfinance(p.ticker),
                    shares=float(p.quantity),
                    avg_cost=float(p.average_price_paid),
                    status="OPEN",
                )
                for p in raw
            ]
        return await self._broker.get_positions()

    async def check_invalidations(self) -> list[dict[str, Any]]:
        """
        Return SELL decisions when invalidation_condition appears met (LLM JSON gate).
        """
        forced: list[dict[str, Any]] = []
        positions = await self._positions_open()
        for pos in positions:
            inv = await self._broker.get_latest_invalidation_for_ticker(pos.ticker)
            if not inv:
                continue
            tech = await self._tools.execute("get_technical", {"ticker": pos.ticker})
            prompt = f"""You are a risk manager. Answer with ONE JSON object only (no markdown):
{{"met": true/false, "reason": "brief English explanation"}}

Ticker: {pos.ticker}
Open shares: {pos.shares}
Average cost: {pos.avg_cost}

Invalidation rule (English):
{inv}

Current technical snapshot:
{tech}

Has the invalidation condition been MET now? If unsure, answer met=false."""
            res = await self._ask_llm(prompt)
            if res.met:
                log.info("Invalidation MET for %s: %s", pos.ticker, res.reason)
                forced.append(
                    {
                        "ticker": pos.ticker,
                        "action": "SELL",
                        "shares": float(pos.shares),
                        "confidence": 0.95,
                        "edge_depth": "DEEP",
                        "hypothesis": "Invalidation exit",
                        "reasoning": f"Invalidation triggered: {res.reason}",
                        "stop_loss": None,
                        "target": None,
                        "invalidation_condition": inv,
                    }
                )
        return forced

    async def _ask_llm(self, prompt: str) -> InvalidationResult:
        if self._cerebras:
            try:
                resp = await self._cerebras.analyze(prompt, system=None)
                return self._parse_llm_answer(resp.text or "")
            except Exception as exc:
                log.warning("Cerebras invalidation query failed: %s", exc)

        if self._groq:
            resp = await self._groq.analyze(prompt, system=None)
            return self._parse_llm_answer((resp.text or "").strip())

        return InvalidationResult(met=False, reason="No LLM available")

    @staticmethod
    def _parse_llm_answer(text: str) -> InvalidationResult:
        import re

        raw = text.strip()
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            return InvalidationResult(met=False, reason="No JSON in model output")
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return InvalidationResult(met=False, reason="JSON parse failed")
        met = bool(obj.get("met", False))
        reason = str(obj.get("reason", "") or "").strip()
        return InvalidationResult(met=met, reason=reason or "N/A")
