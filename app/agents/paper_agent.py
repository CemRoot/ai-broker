"""
PaperAgent (Faz 3) — event-loop driven paper trading agent.

No APScheduler: we use MarketClock + asyncio sleep.
Cerebras is primary (tool-calling); Groq is fallback.
"""

from __future__ import annotations

import asyncio
import html
import json
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo
from typing import Any

from app.core.config import Settings
from app.core.logging import get_logger
from app.core.debug_probe import debug_probe
from app.memory.database import SupabaseDatabase
from app.memory.retriever import RAGRetriever
from app.services.llm.cerebras_service import CerebrasService
from app.services.llm.groq_service import GroqService
from app.services.llm.ollama_service import OllamaService
from app.services.llm.tool_calling import (
    ToolRunResult,
    _extract_json_array,
    analyze_with_tools,
)
from app.services.market_clock import MarketClock
from app.services.paper.account_currency import resolve_paper_account_currency
from app.services.paper.broker import PaperBroker
from app.services.paper.models import PaperTrade
from app.services.paper.t212_pending_mirror_store import enqueue_t212_pending_mirror
from app.services.telegram_operator_alerts import fire_operator_alert, format_exc_brief
from app.services.t212.client import T212Client
from app.services.t212.ticker_map import yfinance_to_t212
from app.tools.definitions import TOOLS
from app.tools.executor import ToolExecutor
from telegram.constants import ParseMode

from app.agents.punishment import PunishmentEngine
from app.agents.position_monitor import PositionMonitor

log = get_logger("paper_agent")

ET = ZoneInfo("America/New_York")


def _paper_agent_clock_block(settings: Settings) -> str:
    """Human-readable ET clock plus optional secondary zone (Telegram UX)."""
    now_et = datetime.now(tz=ET)
    et_line = now_et.strftime("%Y-%m-%d %H:%M ET")
    sec = (getattr(settings, "telegram_display_secondary_tz", None) or "").strip()
    if not sec:
        return (
            f"{et_line}\n"
            "(NYSE schedule uses US/Eastern; the Telegram bubble timestamp is your device local time.)"
        )
    try:
        zi = ZoneInfo(sec)
        loc = datetime.now(tz=zi)
        sub = sec.split("/")[-1].replace("_", " ")
        return (
            f"{et_line} · {loc.strftime('%Y-%m-%d %H:%M')} ({sub})\n"
            "(Anchors OPEN/MIDDAY/CLOSE are US/Eastern; Telegram bubble time follows your phone.)"
        )
    except Exception:
        return f"{et_line}\n(TELEGRAM_DISPLAY_SECONDARY_TZ={sec!r} is invalid — showing ET only.)"


def _paper_agent_clock_header(settings: Settings, *, with_seconds: bool = False) -> str:
    """One-line clock for Telegram headers (ET + optional secondary)."""
    now_et = datetime.now(tz=ET)
    fmt = "%H:%M:%S ET" if with_seconds else "%H:%M ET"
    et_part = now_et.strftime(fmt)
    sec = (getattr(settings, "telegram_display_secondary_tz", None) or "").strip()
    if not sec:
        return et_part
    try:
        loc = datetime.now(tz=ZoneInfo(sec))
        lfmt = "%H:%M:%S" if with_seconds else "%H:%M"
        zn = loc.tzname() or ""
        return f"{et_part} · {loc.strftime(lfmt)} {zn}".strip()
    except Exception:
        return et_part


def build_paper_system_prompt(account_currency: str) -> str:
    ac = (account_currency or "USD").upper()[:3]
    return f"""You are an autonomous paper trading agent (Alpha Arena-style discipline) managing a paper portfolio.
Account base currency is **{ac}** (NAV, cash, and broker balances are in {ac}).
US-listed stock prices from tools are usually in **USD**; the broker converts at execution — use tool prices as quoted; do not assume they are {ac} unless the tool says so.

Use tools for facts; never invent prices, indicator values, or news.
Available tools: get_macro_context, get_portfolio, screen_stocks, get_technical, get_news, get_memories,
check_t212_equity_instrument (when ``PAPER_EXECUTION_BACKEND=t212`` — verify the symbol exists as a **STOCK/ETF** on this Trading 212 account before any BUY).

**Macro vs tradable symbols:** SPY, QQQ, VIX are **benchmarks** for regime commentary only.
Do **not** put SPY/QQQ/^VIX in the JSON decisions array unless they are confirmed **T212 equity invest** names
(use ``check_t212_equity_instrument``). If you have no equity BUY edge, return **no row** for pure index context
(or a single portfolio-level SKIP) instead of repetitive ``SPY HOLD``.
Market data may be supplied to you in TOON (Token-Oriented Object Notation): a tabular,
JSON-equivalent format where the first row defines columns (e.g. ``per_ticker[3]{{ticker,technical,news,memories}}:``)
followed by one row per record. Treat it as the authoritative ground truth.

────────────────────────────────────────────────────────────────────────
ANALYSIS FRAMEWORK (think through the SIX sections in order, in your chain-of-thought)
────────────────────────────────────────────────────────────────────────
0) **CONTEXT SNAPSHOT (Risk Regime)**
   - From get_macro_context: VIX level + regime label (Low-vol / Normal / High-vol), index trend (benchmark ETFs such as SPY/QQQ vs 50/200 SMA — **macro only**, not default trade tickers), VIX direction.
   - News pulse: are there fresh macro catalysts (CPI / FOMC / NFP / earnings) in the next 1–5 days? state risk regime as RISK-ON / RISK-OFF / EVENT-WINDOW.

1) **RAW DATA DASHBOARD (per ticker on the watchlist)**
   For each ticker: pull get_technical (price, RSI, SMA20, ATR if present) + get_news (last 24–48h sentiment).
   Summarize one line per ticker:
   - Global Structure: Up / Sideways / Down vs 20-SMA, RSI band (oversold <30 / normal / overbought >70).
   - Relative strength vs index (qualitative).
   - News intensity: Quiet / Normal / Hot.

2) **NARRATIVE vs REALITY CHECK (per active theme)**
   For each catalyst story (earnings beat, AI deal, guidance cut, geopolitical, sector rotation):
   - Time: how old is the story (hours / days)?
   - Reality: did price actually respond, or is it flat?
   - Classify state: PRICED-IN | DIVERGENCE | ABSORPTION | FRESH IMPULSE.
   - Catalyst risk in next 1–5 days (CPI/Fed/earnings).

3) **FOMO MAP (per ticker you may trade)**
   - Upside Chase level: above which price recent buyers stampede in.
   - Downside Flush level: below which late longs are trapped and stops cluster.
   - This is your context for entries — never chase mid-range; prefer dips toward support or breakouts above defended highs.

4) **ALPHA SETUPS — Menu of Hypotheses (per ticker you may trade)**
   - **Hypothesis A** (primary): View, Timeframe (SCALP / SHORT-SWING / SWING), Alpha Type (FLOW / MEAN-REVERSION / NARRATIVE / SENTIMENT), Edge Depth (DEEP / MODERATE / SHALLOW), Risk Regime (TIGHT / NORMAL / WIDE), Edge Freshness (NEW / AGING / EXPIRED), Invalidation (concrete, testable), **Steel Man Risk** (best counter-argument).
   - **Hypothesis B** (optional fade / alternate).
   - Reject the trade if Steel Man Risk is comparable in weight and you cannot define a clean invalidation.

5) **EDGE QUALITY MATRIX (final classification before output)**
   - HIGH-CONVICTION: DEEP edge, structural tailwind, clear invalidation, confidence ≥ 0.70 → BUY allowed.
   - TACTICAL SKEW: MODERATE edge, counter-trend or short-window setup, confidence 0.60–0.69 → BUY only with tight stop.
   - NO EDGE: SHALLOW edge or unclear invalidation → HOLD or SKIP (never BUY).

────────────────────────────────────────────────────────────────────────
HARD RULES (the math is non-negotiable; the executor enforces them)
────────────────────────────────────────────────────────────────────────
- **No short selling.** SELL only what you already hold.
- **Position cap:** max ~20% of NAV per single ticker (executor clips automatically).
- **Risk per trade:** plan for ~1% of NAV at risk per BUY (R = 0.01 × NAV). Then shares ≈ R / |entry − stop|. The executor will warn if you violate.
- **Stop placement:** stop_loss must be at most −8% below entry (≤ 8% distance). Prefer ATR-aware stops (≈ 2.5 × ATR-14) when ATR is available.
- **Reward / Risk ≥ 2.0** is REQUIRED for BUY: target − entry ≥ 2 × (entry − stop). Setups below 2:1 are SKIP, not BUY.
- **invalidation_condition** (English, testable) is mandatory for every BUY — the price/indicator level that proves the thesis wrong.
- **Confidence gate:** < 0.60 → only HOLD or SKIP (no BUY / SELL initiation).
- **Active punishment:** any ticker listed in ACTIVE PUNISHMENTS is auto-SKIP (do not even propose BUY/SELL on it).

────────────────────────────────────────────────────────────────────────
ORDER TYPES (Trading 212 Public API — pick the right tool for the setup)
────────────────────────────────────────────────────────────────────────
The broker supports four order types. Select by setup quality, not by habit. The executor
will reject malformed combinations.
- **MARKET** (default): take immediate liquidity at the current Last Traded Price.
  Use for emergency exits, time-sensitive entries, and small sizes when slippage is
  acceptable. No `limit_price` / `stop_price` needed.
- **LIMIT**: only execute at `limit_price` or better.
  Use for patient entries near support / pullback zones (BUY at or below limit_price)
  or planned profit-taking (SELL at or above limit_price). Specify `limit_price`.
- **STOP**: triggers a market order once Last Traded Price hits `stop_price`.
  Use for breakout entries (BUY stop above resistance) or stop-loss exits
  (SELL stop below support). Specify `stop_price`.
- **STOP_LIMIT**: when `stop_price` is hit, a limit order at `limit_price` is placed —
  bounds slippage on stop triggers. Specify both `stop_price` AND `limit_price`.
- **time_validity**: `DAY` (expires at session close) or `GOOD_TILL_CANCEL` (persists).
  Default to `DAY` for tactical setups; `GOOD_TILL_CANCEL` only for swing thesis with a
  far-out trigger.
- For BUY orders, the **risk math uses `effective_entry`** = limit_price (LIMIT/STOP_LIMIT)
  or stop_price (STOP buy) or current price (MARKET). Plan stop_loss/target accordingly.

────────────────────────────────────────────────────────────────────────
OUTPUT FORMAT
────────────────────────────────────────────────────────────────────────
1) Chain-of-thought in English (bullet style ok, no markdown code fences). Walk through sections 0→5 briefly.
2) Then ONE JSON array of decision objects (no markdown fences). Each object fields:
   ticker, action (BUY|SELL|HOLD|SKIP), shares, price (optional), confidence (0–1),
   edge_depth (DEEP|MODERATE|SHALLOW), hypothesis, hypothesis_b (optional),
   narrative_vs_reality (optional), steel_man_risk (optional), risk_regime (TIGHT|NORMAL|WIDE, optional),
   edge_freshness (NEW|AGING|EXPIRED, optional),
   order_type (MARKET|LIMIT|STOP|STOP_LIMIT, default MARKET),
   limit_price (required for LIMIT and STOP_LIMIT),
   stop_price (required for STOP and STOP_LIMIT),
   time_validity (DAY|GOOD_TILL_CANCEL, default DAY),
   reasoning, stop_loss, target, invalidation_condition (required for BUY)."""


@dataclass(frozen=True)
class PaperAgentDeps:
    settings: Settings
    db: SupabaseDatabase
    paper_broker: PaperBroker
    cerebras: CerebrasService | None
    groq: GroqService | None
    ollama: OllamaService | None
    retriever: RAGRetriever | None
    tool_executor: ToolExecutor
    market_clock: MarketClock
    telegram_application: Any | None  # PTB Application
    punishment_engine: PunishmentEngine | None = None
    position_monitor: PositionMonitor | None = None
    t212: T212Client | None = None


class PaperAgent:
    def __init__(self, deps: PaperAgentDeps) -> None:
        self.deps = deps
        self._lock = asyncio.Lock()
        self._emergency_count = 0
        self._emergency_day_et: date | None = None
        self._last_cycle_text: str | None = None
        self._last_cycle_json: list[dict[str, Any]] | None = None
        self._last_cycle_at_utc: str | None = None
        self._nav_peak: float = float(deps.settings.paper_starting_nav_usd)
        self._last_drawdown_pct: float = 0.0

    def reset_risk_state(self) -> None:
        """Reset peak NAV tracker (e.g. after `/paper reset confirm`)."""
        self._nav_peak = float(self.deps.settings.paper_starting_nav_usd)
        self._last_drawdown_pct = 0.0

    async def _resolve_account_currency(self) -> str:
        return await resolve_paper_account_currency(self.deps.settings, self.deps.t212)

    def _reset_emergency_count_if_new_et_day(self) -> None:
        """Allow up to 10 emergency cycles per America/New_York calendar day."""
        today = datetime.now(tz=ET).date()
        if self._emergency_day_et != today:
            self._emergency_count = 0
            self._emergency_day_et = today

    async def _active_punishments_prompt_block(self) -> str:
        eng = self.deps.punishment_engine
        if not eng:
            return ""
        try:
            rows = await eng.get_active_punishments()
        except Exception:
            return ""
        if not rows:
            return ""
        lines = ["ACTIVE PUNISHMENTS (do not open new trades in these tickers; use SKIP or HOLD):"]
        for p in rows[:40]:
            exp = p.expires_at.strftime("%Y-%m-%d %H:%M UTC") if p.expires_at else "unknown"
            lines.append(f"- {p.ticker}: {p.penalty_type} until {exp} — {p.reason[:200]}")
        return "\n".join(lines) + "\n"

    async def _estimate_nav_mtm(self) -> tuple[float, float, float]:
        """Marked-to-market NAV: (nav, cash, unrealized). Account currency when using T212."""
        if self.deps.settings.paper_executes_on_t212 and self.deps.t212:
            try:
                s = await self.deps.t212.get_account_summary()
                total = float(s.get("totalValue") or 0)
                cash = float((s.get("cash") or {}).get("availableToTrade") or 0)
                unreal = float((s.get("investments") or {}).get("unrealizedProfitLoss") or 0)
                return total, cash, unreal
            except Exception:
                return 0.0, 0.0, 0.0

        cash = await self.deps.paper_broker.get_balance()
        positions = await self.deps.paper_broker.get_positions()
        invested = sum(p.shares * p.avg_cost for p in positions)
        unreal = 0.0
        if positions:
            prices = await asyncio.gather(
                *[asyncio.to_thread(self._fetch_yf_price, p.ticker) for p in positions]
            )
            for p, px in zip(positions, prices, strict=True):
                if px is not None:
                    unreal += p.shares * float(px) - p.shares * p.avg_cost
        nav = cash + invested + unreal
        return nav, cash, unreal

    async def _portfolio_risk_lines(self, *, account_currency: str) -> tuple[str, bool]:
        """
        Update peak NAV and return (prompt paragraph, halt_new_buys).
        Drawdown is from peak marked-to-market NAV; halts new BUYs when above config threshold.
        """
        ac = (account_currency or "USD").upper()[:3]
        try:
            nav, _cash, _unreal = await self._estimate_nav_mtm()
        except Exception:
            return "", False
        start = float(self.deps.settings.paper_starting_nav_usd)
        self._nav_peak = max(self._nav_peak, nav, start)
        if self._nav_peak <= 1e-6:
            return "", False
        dd_pct = max(0.0, (self._nav_peak - nav) / self._nav_peak * 100.0)
        self._last_drawdown_pct = dd_pct
        limit = float(self.deps.settings.paper_max_drawdown_pct)
        halt = dd_pct >= limit
        if halt:
            log.warning(
                "PaperAgent: max drawdown — %.2f%% from peak (limit %.2f%%); new BUYs disabled",
                dd_pct,
                limit,
            )
        extra = " NEW BUYS DISABLED (max drawdown from peak)." if halt else ""
        para = (
            f"Portfolio risk snapshot: est NAV {nav:,.2f} {ac}, peak NAV {self._nav_peak:,.2f} {ac}, "
            f"drawdown {dd_pct:.1f}% from peak (limit {limit:.1f}%){extra}"
        )
        return para, halt

    @property
    def last_cycle(self) -> dict[str, Any]:
        return {
            "at_utc": self._last_cycle_at_utc,
            "analysis": self._last_cycle_text,
            "decisions": self._last_cycle_json,
            "drawdown_from_peak_pct": self._last_drawdown_pct,
        }

    async def run_cycle(self, event_type: str, *, allow_trades: bool = True) -> tuple[str, list[dict[str, Any]]]:
        # region agent log
        debug_probe(
            run_id="pre-fix",
            hypothesis_id="H3",
            location="app/agents/paper_agent.py:319",
            message="run_cycle entry",
            data={"event_type": event_type, "allow_trades": bool(allow_trades)},
        )
        # endregion
        self._reset_emergency_count_if_new_et_day()
        acct_cur = await self._resolve_account_currency()
        punish_block = await self._active_punishments_prompt_block()
        risk_para, halt_buys = await self._portfolio_risk_lines(account_currency=acct_cur)
        macro_snip = ""
        try:
            macro_snip = await self.deps.tool_executor.execute("get_macro_context", {})
            if len(macro_snip) > 400:
                macro_snip = macro_snip[:400] + "…"
        except Exception:
            macro_snip = ""
        macro_block = f"MACRO (from tools):\n{macro_snip}\n\n" if macro_snip else ""
        clock_block = _paper_agent_clock_block(self.deps.settings)
        sched_note = ""
        if event_type in ("PREMARKET", "TICK"):
            sched_note = (
                "NOTE: Broker orders fire only on OPEN / MIDDAY / CLOSE (regular session, ET). "
                "This event is preparatory or between-anchor ticks — decisions here do not place tickets.\n\n"
            )
        user_message = f"""Market Event: {event_type}
Current Time:
{clock_block}

{sched_note}{risk_para}

{macro_block}{punish_block}
Analyze the market and make your trading decisions.
Use tools to gather data. Follow the trading rules.
"""
        system_prompt = build_paper_system_prompt(acct_cur)

        result = await analyze_with_tools(
            cerebras=self.deps.cerebras,
            groq=self.deps.groq,
            tool_executor=self.deps.tool_executor,
            system_prompt=system_prompt,
            user_message=user_message,
            tools=TOOLS,
            max_iterations=10,
        )
        if (
            not (result.decisions or [])
            and "temporarily unavailable" in (result.reasoning_text or "").lower()
            and self.deps.ollama is not None
        ):
            # region agent log
            debug_probe(
                run_id="pre-fix",
                hypothesis_id="H6",
                location="app/agents/paper_agent.py:370",
                message="fallback to local prepass via ollama",
                data={"event_type": event_type, "model_before": result.model},
            )
            # endregion
            result = await self._run_cycle_local_prepass(
                system_prompt=system_prompt,
                user_message=user_message,
                macro_snip=macro_snip or "",
            )

        analysis_text = (result.reasoning_text or "").strip()
        decisions = result.decisions or []
        # region agent log
        debug_probe(
            run_id="pre-fix",
            hypothesis_id="H4",
            location="app/agents/paper_agent.py:365",
            message="run_cycle decisions parsed",
            data={
                "event_type": event_type,
                "decisions_count": len(decisions),
                "first_rows": [
                    {
                        "ticker": str(d.get("ticker", "")),
                        "action": str(d.get("action", "")),
                        "confidence": d.get("confidence"),
                    }
                    for d in decisions[:3]
                ],
            },
        )
        # endregion

        # If punishments are active, force-skip punished tickers.
        if self.deps.punishment_engine and decisions:
            for d in decisions:
                t = str(d.get("ticker", "")).upper().strip()
                if not t:
                    continue
                if await self.deps.punishment_engine.is_punished(t):
                    d["action"] = "SKIP"
                    d["reasoning"] = (str(d.get("reasoning", "")) + " (skipped: active punishment)").strip()

        self._last_cycle_text = analysis_text
        self._last_cycle_json = decisions
        self._last_cycle_at_utc = datetime.now(timezone.utc).isoformat()

        # Persist cycle log (minimal): store in daily_reports as PAPER_CYCLE.
        await self._save_cycle_log(event_type=event_type, analysis=analysis_text, decisions=decisions)

        # Execute trades only during allowed windows (e.g., OPEN/MIDDAY/CLOSE, emergency).
        if allow_trades:
            for d in decisions:
                await self._apply_decision(
                    d,
                    cycle_event=event_type,
                    chain_of_thought=analysis_text,
                    emergency=False,
                    halt_new_buys=halt_buys,
                )

            # Invalidation checks (best-effort; low frequency).
            if self.deps.position_monitor:
                try:
                    forced = await self.deps.position_monitor.check_invalidations()
                    for fd in forced:
                        inv_text = str(fd.get("invalidation_condition") or "").strip()
                        trade = await self._apply_decision(
                            fd,
                            cycle_event=f"{event_type}:INVALIDATION",
                            chain_of_thought="Position invalidation check.",
                            emergency=False,
                            halt_new_buys=halt_buys,
                        )
                        if trade and trade.action == "SELL":
                            await self._send_invalidation_alert(
                                ticker=trade.ticker,
                                condition=inv_text,
                                trade=trade,
                                currency=acct_cur,
                            )
                except Exception:
                    pass

        # Add a compact decision memory per ticker (if RAG is available).
        if self.deps.retriever:
            for d in decisions:
                t = str(d.get("ticker", "")).upper().strip()
                if not t:
                    continue
                ctx = json.dumps(d, ensure_ascii=False)
                await self.deps.retriever.add_memory(
                    ticker=t,
                    memory_type="DECISION",
                    context=ctx,
                    outcome="OPEN",
                    pnl_percent=None,
                )

        # Live feed (best-effort)
        await self._send_live_feed(
            event_type=event_type,
            decisions=decisions,
            macro_snippet=macro_snip or None,
            nav_summary_line=risk_para or None,
            account_currency=acct_cur,
        )

        return analysis_text, decisions

    async def run_forever(self) -> None:
        log.info("PaperAgent started — waiting for market ticks/events")
        while True:
            try:
                event_type, cadence = await self.deps.market_clock.wait_for_next_tick()
                allow_trades = event_type in ("OPEN", "MIDDAY", "CLOSE")
                async with self._lock:
                    await self.run_cycle(event_type, allow_trades=allow_trades)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.error("PaperAgent cycle error: %s", exc)
                await fire_operator_alert(
                    category="PaperAgent",
                    summary=f"run_forever: cycle crashed — {type(exc).__name__}",
                    detail=format_exc_brief(exc),
                    dedupe_key=f"paper_agent_loop:{type(exc).__name__}",
                )
                await asyncio.sleep(60)

    async def run_emergency_cycle(self, *, trigger: str, context: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
        async with self._lock:
            self._reset_emergency_count_if_new_et_day()
            if self._emergency_count >= 10:
                log.warning("Daily emergency limit reached")
                return "", []

            self._emergency_count += 1
            now_et = datetime.now(tz=ET).strftime("%Y-%m-%d %H:%M ET")
            affected = [str(x).upper().strip() for x in (context.get("affected_tickers") or []) if x]
            affected = [x for x in affected if x][:32]
            if affected:
                focus = (
                    "Prioritize these symbols from the trigger (use tools; do not assume prices): "
                    + ", ".join(affected)
                    + ". Then review any other open positions for correlated risk."
                )
            else:
                focus = (
                    "No tickers were extracted from the trigger — review all open positions "
                    "and overall exposure using tools."
                )
            acct_cur = await self._resolve_account_currency()
            punish_block = await self._active_punishments_prompt_block()
            risk_para, halt_buys = await self._portfolio_risk_lines(account_currency=acct_cur)
            macro_snip = ""
            try:
                macro_snip = await self.deps.tool_executor.execute("get_macro_context", {})
                if len(macro_snip) > 400:
                    macro_snip = macro_snip[:400] + "…"
            except Exception:
                macro_snip = ""
            macro_block = f"MACRO (from tools):\n{macro_snip}\n\n" if macro_snip else ""
            user_message = f"""EMERGENCY TRIGGER: {trigger}
TIME: {now_et}

{risk_para}

{macro_block}{focus}

{punish_block}
CONTEXT (JSON):
{json.dumps(context, ensure_ascii=False)}

Use tools to check portfolio exposure and decide whether to HOLD/SELL/REDUCE risk.
Return JSON decisions array.
"""
            result = await analyze_with_tools(
                cerebras=self.deps.cerebras,
                groq=self.deps.groq,
                tool_executor=self.deps.tool_executor,
                system_prompt=build_paper_system_prompt(acct_cur),
                user_message=user_message,
                tools=TOOLS,
                max_iterations=10,
            )
            if (
                not (result.decisions or [])
                and "temporarily unavailable" in (result.reasoning_text or "").lower()
                and self.deps.ollama is not None
            ):
                # region agent log
                debug_probe(
                    run_id="pre-fix",
                    hypothesis_id="H6",
                    location="app/agents/paper_agent.py:548",
                    message="emergency fallback to local prepass via ollama",
                    data={"trigger": trigger, "model_before": result.model},
                )
                # endregion
                result = await self._run_cycle_local_prepass(
                    system_prompt=build_paper_system_prompt(acct_cur),
                    user_message=user_message,
                    macro_snip=macro_snip or "",
                )
            analysis_text = (result.reasoning_text or "").strip()
            decisions = result.decisions or []
            await self._save_cycle_log(event_type=f"EMERGENCY:{trigger}", analysis=analysis_text, decisions=decisions)
            for d in decisions:
                await self._apply_decision(
                    d,
                    cycle_event=f"EMERGENCY:{trigger}",
                    chain_of_thought=analysis_text,
                    emergency=True,
                    halt_new_buys=halt_buys,
                )
            await self._send_live_feed(
                event_type=f"EMERGENCY:{trigger}",
                decisions=decisions,
                is_emergency=True,
                emergency_context=context,
                macro_snippet=macro_snip or None,
                nav_summary_line=risk_para or None,
                account_currency=acct_cur,
            )
            return analysis_text, decisions

    async def _apply_decision(
        self,
        d: dict[str, Any],
        *,
        cycle_event: str | None,
        chain_of_thought: str | None,
        emergency: bool,
        halt_new_buys: bool = False,
    ) -> PaperTrade | None:
        action = str(d.get("action", "")).upper().strip()
        ticker = str(d.get("ticker", "")).upper().strip()
        if not ticker or action not in ("BUY", "SELL", "HOLD", "SKIP"):
            return None

        confidence = float(d.get("confidence", 0.0) or 0.0)
        if confidence < 0.60 and action in ("BUY", "SELL"):
            return None

        shares = float(d.get("shares", 0.0) or 0.0)
        if action in ("BUY", "SELL") and shares <= 0:
            return None

        # Order routing: MARKET (default), LIMIT, STOP, STOP_LIMIT.
        order_type = str(d.get("order_type", "MARKET") or "MARKET").upper().strip()
        if order_type not in ("MARKET", "LIMIT", "STOP", "STOP_LIMIT"):
            log.warning(
                "Unsupported order_type %r for %s — defaulting to MARKET", order_type, ticker
            )
            order_type = "MARKET"
        time_validity = str(d.get("time_validity", "DAY") or "DAY").upper().strip()
        if time_validity not in ("DAY", "GOOD_TILL_CANCEL"):
            time_validity = "DAY"

        def _coerce_price(val: Any) -> float | None:
            try:
                return float(val) if val is not None else None
            except (TypeError, ValueError):
                return None

        limit_price = _coerce_price(d.get("limit_price"))
        stop_price_param = _coerce_price(d.get("stop_price"))

        # Required-field validation per order type (executor layer; LLM is reminded by prompt).
        if order_type in ("LIMIT", "STOP_LIMIT") and (limit_price is None or limit_price <= 0):
            log.warning(
                "Skipping %s %s: order_type=%s requires positive limit_price (got %r)",
                action, ticker, order_type, d.get("limit_price"),
            )
            return None
        if order_type in ("STOP", "STOP_LIMIT") and (stop_price_param is None or stop_price_param <= 0):
            log.warning(
                "Skipping %s %s: order_type=%s requires positive stop_price (got %r)",
                action, ticker, order_type, d.get("stop_price"),
            )
            return None

        # Determine price (LLM may omit it). Use yfinance last price as fallback.
        price = d.get("price")
        try:
            px = float(price) if price is not None else None
        except Exception:
            px = None

        if px is None:
            px = await asyncio.to_thread(self._fetch_yf_price, ticker)
        if px is None:
            log.warning("No price for %s; cannot execute %s", ticker, action)
            return None

        # `effective_entry` is the price the risk math should be based on (where the
        # order is *planned* to fill, not the LTP). For MARKET orders this is `px`;
        # for LIMIT/STOP_LIMIT it's `limit_price`; for a STOP buy it's `stop_price`.
        if order_type == "MARKET":
            effective_entry = px
        elif order_type == "LIMIT":
            effective_entry = float(limit_price)  # type: ignore[arg-type]
        elif order_type == "STOP":
            effective_entry = float(stop_price_param)  # type: ignore[arg-type]
        else:  # STOP_LIMIT
            effective_entry = float(limit_price)  # type: ignore[arg-type]

        # Risk sizing: ~20% NAV cap is enforced at broker layer for both ledgers.
        # T212 mode uses live ``totalValue`` (account currency) as a NAV proxy; account
        # currency vs USD price difference is approximated 1:1 — coarse but always tighter
        # than relying on the LLM, and T212 will still reject under-funded orders.
        if action == "BUY":
            if self.deps.settings.paper_executes_on_t212 and self.deps.t212:
                try:
                    summ = await self.deps.t212.get_account_summary()
                    cash_avail = float((summ.get("cash") or {}).get("availableToTrade") or 0.0)
                    total_val = float(summ.get("totalValue") or cash_avail)
                    nav_proxy = max(total_val, 1.0)
                    max_pos_value = 0.20 * nav_proxy
                    order_value = shares * px
                    if order_value > max_pos_value:
                        clipped = max(0.0, max_pos_value / px)
                        log.warning(
                            "T212 BUY %s clipped %.4f → %.4f shares (~20%% NAV cap; total=%.2f %s)",
                            ticker, shares, clipped, nav_proxy, summ.get("currency") or "?",
                        )
                        shares = clipped
                except Exception as exc:
                    log.warning("T212 NAV cap pre-check failed for %s: %s", ticker, exc)
            else:
                cash = await self.deps.paper_broker.get_balance()
                positions = await self.deps.paper_broker.get_positions()
                invested_cost = sum(p.shares * p.avg_cost for p in positions)
                nav_est = cash + invested_cost
                max_pos_value = 0.20 * max(nav_est, 1.0)
                order_value = shares * px
                if order_value > max_pos_value:
                    shares = max_pos_value / px
            if shares <= 0:
                return None

        reasoning = str(d.get("reasoning", "") or d.get("hypothesis", "") or "").strip()
        stop_loss = d.get("stop_loss")
        target = d.get("target")
        invalidation_condition = d.get("invalidation_condition")
        try:
            stop_loss_f = float(stop_loss) if stop_loss is not None else None
        except Exception:
            stop_loss_f = None
        try:
            target_f = float(target) if target is not None else None
        except Exception:
            target_f = None
        invalid_s = str(invalidation_condition).strip() if invalidation_condition else None

        cot = (chain_of_thought or "").strip()
        if cot and len(cot) > 5000:
            cot = cot[:5000] + "…"
        if action == "BUY":
            if halt_new_buys:
                log.warning("Skipping BUY %s: max drawdown halt active", ticker)
                return None
            if self.deps.settings.paper_executes_on_t212 and self.deps.t212:
                ok_sym, why = await self.deps.t212.is_us_equity_instrument_tradeable(ticker)
                if not ok_sym:
                    log.warning("Skipping BUY %s: %s", ticker, why)
                    return None
            # Mathematical guard: enforce stop within 8% AND reward/risk ≥ 2.0.
            # The system prompt asks the LLM to honor these; the executor enforces.
            if stop_loss_f is None or target_f is None:
                log.warning(
                    "Skipping BUY %s: stop_loss/target missing (system prompt requires both for BUY)",
                    ticker,
                )
                return None
            entry_for_risk = effective_entry
            if not (0 < stop_loss_f < entry_for_risk) or not (target_f > entry_for_risk):
                log.warning(
                    "Skipping BUY %s [%s]: invalid stop/target geometry (entry=%.4f stop=%.4f target=%.4f)",
                    ticker, order_type, entry_for_risk, stop_loss_f, target_f,
                )
                return None
            stop_dist = entry_for_risk - stop_loss_f
            target_dist = target_f - entry_for_risk
            stop_pct = stop_dist / entry_for_risk
            rr = target_dist / stop_dist if stop_dist > 0 else 0.0
            if stop_pct > 0.08:
                log.warning(
                    "Skipping BUY %s [%s]: stop too wide (%.2f%% > 8%%); tighten stop",
                    ticker, order_type, stop_pct * 100.0,
                )
                return None
            if rr < 2.0:
                log.warning(
                    "Skipping BUY %s [%s]: reward/risk %.2f < 2.0 (target=%.4f stop=%.4f entry=%.4f)",
                    ticker, order_type, rr, target_f, stop_loss_f, entry_for_risk,
                )
                return None
            if self.deps.settings.paper_executes_on_t212:
                if not self.deps.t212:
                    log.error("paper_executes_on_t212 but T212Client missing")
                    return None
                return await self._t212_execute_buy(
                    ticker=ticker,
                    shares=shares,
                    px=px,
                    reasoning=reasoning,
                    stop_loss_f=stop_loss_f,
                    target_f=target_f,
                    invalid_s=invalid_s,
                    cot=cot or None,
                    cycle_event=cycle_event,
                    emergency=emergency,
                    order_type=order_type,
                    limit_price=limit_price,
                    stop_price=stop_price_param,
                    time_validity=time_validity,
                    effective_entry=effective_entry,
                )
            return await self.deps.paper_broker.buy(
                ticker,
                shares=shares,
                price=px,
                reasoning=reasoning,
                stop_loss=stop_loss_f,
                target=target_f,
                invalidation_condition=invalid_s,
                chain_of_thought=cot or None,
                cycle_event=cycle_event,
                emergency=emergency,
            )
        if action == "SELL":
            if self.deps.settings.paper_executes_on_t212:
                if not self.deps.t212:
                    log.error("paper_executes_on_t212 but T212Client missing")
                    return None
                trade = await self._t212_execute_sell(
                    ticker=ticker,
                    shares=shares,
                    px=px,
                    reasoning=reasoning,
                    stop_loss_f=stop_loss_f,
                    target_f=target_f,
                    invalid_s=invalid_s,
                    cot=cot or None,
                    cycle_event=cycle_event,
                    emergency=emergency,
                    order_type=order_type,
                    limit_price=limit_price,
                    stop_price=stop_price_param,
                    time_validity=time_validity,
                )
            else:
                trade = await self.deps.paper_broker.sell(
                    ticker,
                    shares=shares,
                    price=px,
                    reasoning=reasoning,
                    stop_loss=stop_loss_f,
                    target=target_f,
                    invalidation_condition=invalid_s,
                    chain_of_thought=cot or None,
                    cycle_event=cycle_event,
                    emergency=emergency,
                )
            if self.deps.punishment_engine and trade and trade.pnl_percent is not None:
                try:
                    await self.deps.punishment_engine.check_and_punish(
                        ticker=ticker,
                        pnl_percent=float(trade.pnl_percent),
                        reasoning=reasoning,
                        technical_at_entry=None,
                    )
                except Exception:
                    pass
            return trade
        return None

    async def _dispatch_t212_order(
        self,
        *,
        ticker: str,
        signed_quantity: float,
        order_type: str,
        limit_price: float | None,
        stop_price: float | None,
        time_validity: str,
        extended_hours: bool,
    ) -> dict[str, Any]:
        """Route an order to the right T212 endpoint based on ``order_type``.

        Quantity sign convention follows T212 docs: positive=BUY, negative=SELL.
        Required prices are validated upstream by ``_apply_decision``.
        """
        assert self.deps.t212 is not None
        ot = (order_type or "MARKET").upper().strip()
        if ot == "MARKET":
            return await self.deps.t212.place_market_order(
                ticker, signed_quantity, extended_hours=extended_hours,
            )
        if ot == "LIMIT":
            return await self.deps.t212.place_limit_order(
                ticker, signed_quantity,
                limit_price=float(limit_price),  # type: ignore[arg-type]
                time_validity=time_validity,  # type: ignore[arg-type]
                extended_hours=extended_hours,
            )
        if ot == "STOP":
            return await self.deps.t212.place_stop_order(
                ticker, signed_quantity,
                stop_price=float(stop_price),  # type: ignore[arg-type]
                time_validity=time_validity,  # type: ignore[arg-type]
                extended_hours=extended_hours,
            )
        if ot == "STOP_LIMIT":
            return await self.deps.t212.place_stop_limit_order(
                ticker, signed_quantity,
                stop_price=float(stop_price),  # type: ignore[arg-type]
                limit_price=float(limit_price),  # type: ignore[arg-type]
                time_validity=time_validity,  # type: ignore[arg-type]
                extended_hours=extended_hours,
            )
        raise ValueError(f"Unsupported order_type: {order_type!r}")

    async def _has_t212_buy_exposure(self, *, t212_ticker: str) -> tuple[bool, str]:
        """Return ``(True, reason)`` if the ticker already has long exposure or a queued BUY.

        Why:
        - Users may submit external mobile/web orders while ``PAPER_T212_RECONCILE_EXTERNAL_ORDERS=true``.
        - Supabase mirror is eventually consistent (poll-based), so a second BUY can be emitted before
          DB catches up unless we check broker-side truth just before placing a new BUY.
        """
        assert self.deps.t212 is not None

        # 1) Filled/open long exposure on broker.
        try:
            pos = await self.deps.t212.get_positions(ticker=t212_ticker)
            if isinstance(pos, list) and pos and float(getattr(pos[0], "quantity", 0.0) or 0.0) > 0:
                return True, "existing open position on T212"
        except Exception as exc:
            log.warning("T212 BUY exposure check (positions) failed for %s: %s", t212_ticker, exc)

        # 2) Pending BUY orders in broker queue (NEW/PENDING/PARTIAL).
        try:
            pending = await self.deps.t212.get_all_pending_orders()
            if not isinstance(pending, list):
                pending = []
            for o in pending:
                if str(o.get("ticker") or "").upper() != t212_ticker.upper():
                    continue
                side = str(o.get("side") or "").upper()
                qty = float(o.get("quantity") or 0.0)
                if side == "BUY" or (not side and qty > 0):
                    status = str(o.get("status") or "").upper()
                    return True, f"pending BUY already queued on T212 (status={status or 'UNKNOWN'})"
        except Exception as exc:
            log.warning("T212 BUY exposure check (pending orders) failed for %s: %s", t212_ticker, exc)

        return False, ""

    async def _t212_execute_buy(
        self,
        *,
        ticker: str,
        shares: float,
        px: float,
        reasoning: str,
        stop_loss_f: float | None,
        target_f: float | None,
        invalid_s: str | None,
        cot: str | None,
        cycle_event: str | None,
        emergency: bool,
        order_type: str = "MARKET",
        limit_price: float | None = None,
        stop_price: float | None = None,
        time_validity: str = "DAY",
        effective_entry: float | None = None,
    ) -> PaperTrade | None:
        assert self.deps.t212 is not None
        t212_sym = yfinance_to_t212(ticker)
        has_exp, why = await self._has_t212_buy_exposure(t212_ticker=t212_sym)
        if has_exp:
            log.warning("T212 BUY skipped for %s (%s): %s", ticker, t212_sym, why)
            return None
        ext = self.deps.settings.paper_t212_extended_hours
        try:
            order = await self._dispatch_t212_order(
                ticker=t212_sym,
                signed_quantity=shares,  # buy → positive
                order_type=order_type,
                limit_price=limit_price,
                stop_price=stop_price,
                time_validity=time_validity,
                extended_hours=ext,
            )
        except Exception as exc:
            log.error("T212 BUY [%s] failed %s: %s", order_type, ticker, exc)
            return None
        oid = order.get("id")
        try:
            oid_int = int(oid) if oid is not None else None
        except (TypeError, ValueError):
            oid_int = None
        fq = float(order.get("filledQuantity") or 0)
        fv = float(order.get("filledValue") or 0)
        if oid_int is None:
            log.warning(
                "T212 BUY returned no order id for %s — mirror row skipped (markets closed or API shape changed)",
                ticker,
            )
            return None
        # Queued market orders (e.g. weekend): persist id for T212MirrorPoller.
        if fq <= 0 or fv == 0:
            log.info(
                "T212 BUY pending fill %s order_id=%s — enqueued for mirror poller",
                ticker,
                oid_int,
            )
            await enqueue_t212_pending_mirror(
                self.deps.db,
                t212_order_id=oid_int,
                yf_ticker=ticker,
                action="BUY",
                meta={
                    "reasoning": reasoning or "",
                    "stop_loss": stop_loss_f,
                    "target": target_f,
                    "invalidation_condition": invalid_s,
                    "chain_of_thought": cot,
                    "cycle_event": cycle_event,
                    "emergency": bool(emergency),
                    "source": "agent",
                },
            )
            return None
        exec_price = abs(fv / fq)
        exec_shares = fq
        total_row = abs(fv)
        trade = await self.deps.paper_broker.record_mirror_trade(
            ticker=ticker,
            action="BUY",
            shares=exec_shares,
            price=exec_price,
            total_value=total_row,
            reasoning=reasoning,
            stop_loss=stop_loss_f,
            target=target_f,
            invalidation_condition=invalid_s,
            chain_of_thought=cot,
            cycle_event=cycle_event,
            emergency=emergency,
            t212_order_id=oid_int,
        )
        if getattr(self.deps.settings, "paper_t212_sync_supabase_ledger", True) and self.deps.t212:
            try:
                await self.deps.paper_broker.sync_ledger_from_t212_client(self.deps.t212)
            except Exception as exc:
                log.warning("T212 ledger sync after BUY mirror failed: %s", exc)
        try:
            acct_cur = await self._resolve_account_currency()
            nav_now, _, _ = await self._estimate_nav_mtm()
            await self._send_trade_notification(
                trade=trade,
                order_type=order_type,
                time_validity=time_validity,
                reasoning=reasoning,
                stop_loss=stop_loss_f,
                target=target_f,
                invalidation=invalid_s,
                cycle_event=cycle_event,
                emergency=emergency,
                currency=acct_cur,
                nav_now=nav_now,
                order_id=oid_int,
            )
        except Exception as exc:
            log.warning("BUY trade notification failed for %s: %s", ticker, exc)
        return trade

    async def _t212_execute_sell(
        self,
        *,
        ticker: str,
        shares: float,
        px: float,
        reasoning: str,
        stop_loss_f: float | None,
        target_f: float | None,
        invalid_s: str | None,
        cot: str | None,
        cycle_event: str | None,
        emergency: bool,
        order_type: str = "MARKET",
        limit_price: float | None = None,
        stop_price: float | None = None,
        time_validity: str = "DAY",
    ) -> PaperTrade | None:
        assert self.deps.t212 is not None
        t212_sym = yfinance_to_t212(ticker)
        pos_list = await self.deps.t212.get_positions(ticker=t212_sym)
        if not pos_list:
            log.warning("T212 SELL skipped — no open position for %s (%s)", ticker, t212_sym)
            return None
        pos = pos_list[0]
        held = float(pos.quantity)
        sell_qty = min(float(shares), held)
        if sell_qty <= 0:
            return None
        avg = float(pos.average_price_paid)
        ext = self.deps.settings.paper_t212_extended_hours
        try:
            order = await self._dispatch_t212_order(
                ticker=t212_sym,
                signed_quantity=-sell_qty,  # sell → negative
                order_type=order_type,
                limit_price=limit_price,
                stop_price=stop_price,
                time_validity=time_validity,
                extended_hours=ext,
            )
        except Exception as exc:
            log.error("T212 SELL [%s] failed %s: %s", order_type, ticker, exc)
            return None
        oid = order.get("id")
        try:
            oid_int = int(oid) if oid is not None else None
        except (TypeError, ValueError):
            oid_int = None
        fq = abs(float(order.get("filledQuantity") or 0))
        fv = float(order.get("filledValue") or 0)
        if oid_int is None:
            log.warning(
                "T212 SELL returned no order id for %s — mirror row skipped (markets closed or API shape changed)",
                ticker,
            )
            return None
        if fq <= 0 or fv == 0:
            log.info(
                "T212 SELL pending fill %s order_id=%s — enqueued for mirror poller",
                ticker,
                oid_int,
            )
            await enqueue_t212_pending_mirror(
                self.deps.db,
                t212_order_id=oid_int,
                yf_ticker=ticker,
                action="SELL",
                meta={
                    "reasoning": reasoning or "",
                    "stop_loss": stop_loss_f,
                    "target": target_f,
                    "invalidation_condition": invalid_s,
                    "chain_of_thought": cot,
                    "cycle_event": cycle_event,
                    "emergency": bool(emergency),
                    "avg_price_paid": avg,
                    "source": "agent",
                },
            )
            return None
        exec_price = abs(fv / fq)
        exec_shares = fq
        proceeds = abs(fv)
        cost_basis = exec_shares * avg
        realized = proceeds - cost_basis
        pnl_pct = (realized / cost_basis * 100.0) if cost_basis > 0 else None
        trade = await self.deps.paper_broker.record_mirror_trade(
            ticker=ticker,
            action="SELL",
            shares=exec_shares,
            price=exec_price,
            total_value=proceeds,
            reasoning=reasoning,
            stop_loss=stop_loss_f,
            target=target_f,
            invalidation_condition=invalid_s,
            chain_of_thought=cot,
            cycle_event=cycle_event,
            emergency=emergency,
            pnl_percent=pnl_pct,
            realized_pnl_usd=realized,
            t212_order_id=oid_int,
        )
        if getattr(self.deps.settings, "paper_t212_sync_supabase_ledger", True) and self.deps.t212:
            try:
                await self.deps.paper_broker.sync_ledger_from_t212_client(self.deps.t212)
            except Exception as exc:
                log.warning("T212 ledger sync after SELL mirror failed: %s", exc)
        try:
            acct_cur = await self._resolve_account_currency()
            nav_now, _, _ = await self._estimate_nav_mtm()
            await self._send_trade_notification(
                trade=trade,
                order_type=order_type,
                time_validity=time_validity,
                reasoning=reasoning,
                stop_loss=stop_loss_f,
                target=target_f,
                invalidation=invalid_s,
                cycle_event=cycle_event,
                emergency=emergency,
                currency=acct_cur,
                nav_now=nav_now,
                order_id=oid_int,
            )
        except Exception as exc:
            log.warning("SELL trade notification failed for %s: %s", ticker, exc)
        return trade

    async def _send_trade_notification(
        self,
        *,
        trade: PaperTrade,
        order_type: str,
        time_validity: str,
        reasoning: str,
        stop_loss: float | None,
        target: float | None,
        invalidation: str | None,
        cycle_event: str | None,
        emergency: bool,
        currency: str = "USD",
        nav_now: float | None = None,
        order_id: int | None = None,
    ) -> None:
        """Send a professional, per-trade Telegram alert.

        Format aims to be CEO-readable: WHY (hypothesis), HOW (order type), GUARDRAILS
        (stop/target/invalidation), and CONTEXT (cycle, account snapshot).
        """
        app = self.deps.telegram_application
        if not app:
            return
        allowed = self.deps.settings.allowed_user_ids
        if not allowed:
            return

        ac = (currency or "USD").upper()[:3]
        action = (trade.action or "").upper().strip()
        is_buy = action == "BUY"
        emoji = "🟢" if is_buy else "🔻"
        verb_tr = "ALIM" if is_buy else "SATIM"

        order_label = order_type.upper().replace("_", "-")
        validity_label = "GTC" if time_validity == "GOOD_TILL_CANCEL" else "DAY"
        emergency_tag = " 🚨" if emergency else ""

        head = (
            f"{emoji} <b>{verb_tr}{emergency_tag}</b> · <b>{html.escape(trade.ticker)}</b> · "
            f"<code>{order_label}</code> @ <b>{trade.price:,.2f} {ac}</b> · <code>{validity_label}</code>"
        )

        notional = float(trade.shares) * float(trade.price)
        nav_pct = ""
        if nav_now and nav_now > 0:
            nav_pct = f" ≈ %{(notional / nav_now) * 100:.1f} NAV"
        notional_line = (
            f"📦 <b>Adet</b>: <code>{float(trade.shares):,.4f}</code> "
            f"(toplam <b>{notional:,.2f} {ac}</b>{html.escape(nav_pct)})"
        )

        guardrail_parts: list[str] = []
        if is_buy:
            if target is not None:
                tgt_pct = (float(target) - float(trade.price)) / float(trade.price) * 100.0
                guardrail_parts.append(f"🎯 Hedef {float(target):,.2f} ({tgt_pct:+.1f}%)")
            if stop_loss is not None:
                stop_pct = (float(stop_loss) - float(trade.price)) / float(trade.price) * 100.0
                guardrail_parts.append(f"🛑 Stop {float(stop_loss):,.2f} ({stop_pct:+.1f}%)")
            if (target is not None) and (stop_loss is not None) and float(trade.price) > float(stop_loss):
                rr_num = float(target) - float(trade.price)
                rr_den = float(trade.price) - float(stop_loss)
                if rr_den > 0:
                    guardrail_parts.append(f"⚖️ R/R {rr_num / rr_den:.2f}")
        else:
            if trade.realized_pnl_usd is not None and trade.pnl_percent is not None:
                pnl_emoji = "💰" if float(trade.realized_pnl_usd) >= 0 else "🩸"
                guardrail_parts.append(
                    f"{pnl_emoji} Realize PnL: {float(trade.realized_pnl_usd):+,.2f} {ac} "
                    f"({float(trade.pnl_percent):+.2f}%)"
                )

        why_line = ""
        clean_reasoning = (reasoning or "").strip().replace("\n", " ")
        if len(clean_reasoning) > 320:
            clean_reasoning = clean_reasoning[:320].rstrip() + "…"
        if clean_reasoning:
            why_line = (
                f"🧠 <b>{'Aldım' if is_buy else 'Sattım'} çünkü</b>: "
                f"{html.escape(clean_reasoning)}"
            )

        invalid_line = ""
        if is_buy and invalidation:
            inv = invalidation.strip().replace("\n", " ")
            if len(inv) > 200:
                inv = inv[:200] + "…"
            invalid_line = f"🚫 <b>Geçersizleştirici</b>: {html.escape(inv)}"

        footer_bits: list[str] = ["🤖 PaperAgent"]
        if cycle_event:
            footer_bits.append(cycle_event[:20])
        footer_bits.append(_paper_agent_clock_header(self.deps.settings, with_seconds=False))
        if order_id is not None:
            footer_bits.append(f"#T212-{order_id}")
        footer = " · ".join(html.escape(b) for b in footer_bits)

        lines = [head, notional_line]
        if guardrail_parts:
            lines.append(" · ".join(html.escape(x) for x in guardrail_parts))
        if why_line:
            lines.append(why_line)
        if invalid_line:
            lines.append(invalid_line)
        lines.append(f"🤖 <i>{footer}</i>")
        msg = "\n".join(lines)

        for uid in allowed:
            try:
                await app.bot.send_message(chat_id=uid, text=msg, parse_mode=ParseMode.HTML)
            except Exception:
                pass

    async def _send_invalidation_alert(
        self,
        *,
        ticker: str,
        condition: str,
        trade: PaperTrade,
        currency: str = "USD",
    ) -> None:
        app = self.deps.telegram_application
        if not app:
            return
        allowed = self.deps.settings.allowed_user_ids
        if not allowed:
            return
        ac = (currency or "USD").upper()[:3]
        cond = (condition or "").strip() or "see trade log"
        if len(cond) > 200:
            cond = cond[:200] + "…"
        pnl = ""
        if trade.pnl_percent is not None:
            pnl = f" | PnL: {float(trade.pnl_percent):+.1f}%"
        msg = (
            f"🔴 <b>{html.escape(ticker)} · İNVALİDASYON</b>\n"
            f"Koşul: {html.escape(cond)}\n"
            f"Otomatik satıldı <code>{trade.shares:.2f}</code> @ <b>{trade.price:.2f} {ac}</b>"
            f"{html.escape(pnl)}"
        )
        for uid in allowed:
            try:
                await app.bot.send_message(chat_id=uid, text=msg, parse_mode=ParseMode.HTML)
            except Exception:
                pass

    @staticmethod
    def _encode_prepass_payload(payload: dict[str, Any]) -> tuple[str, str]:
        """Pack the local-prepass payload as TOON when available; otherwise return raw plaintext.

        Returns ``(body, encoding)`` where ``encoding`` is ``"toon"`` or ``"plain"``.
        TOON cuts ~30–55% of input tokens vs JSON for the same dict (tabular ``per_ticker``).
        """
        try:
            import toon_format

            return toon_format.encode(payload), "toon"
        except Exception as exc:
            log.warning("toon_format unavailable for prepass payload (%s); falling back to plaintext", exc)
            return "", "plain"

    async def _run_cycle_local_prepass(
        self,
        *,
        system_prompt: str,
        user_message: str,
        macro_snip: str,
    ) -> ToolRunResult:
        """Prepass strategy: fan out tools in Python (real data), then ask the LLM once."""
        tx = self.deps.tool_executor
        s = self.deps.settings

        held: list[str] = []
        try:
            positions = await self.deps.paper_broker.get_positions()
            held = [p.ticker.upper() for p in positions if p.shares and p.shares > 0]
        except Exception:
            held = []

        candidates: list[str] = []
        try:
            scr = await tx.execute("screen_stocks", {"min_volume_ratio": 1.5})
            for line in (scr or "").splitlines():
                tk = line.split(":", 1)[0].strip().upper()
                if tk and tk.isalpha() and 1 <= len(tk) <= 5 and tk not in candidates:
                    candidates.append(tk)
                if len(candidates) >= int(getattr(s, "paper_local_prepass_top_k", 5)):
                    break
        except Exception:
            candidates = []

        universe: list[str] = []
        for tk in held + candidates:
            if tk and tk not in universe:
                universe.append(tk)
        if not universe:
            universe = ["SPY"]
        max_universe = max(int(getattr(s, "paper_local_prepass_top_k", 5)), 3) + len(held)
        universe = universe[:max_universe]

        log.info("Local-prepass universe (%d): %s", len(universe), ",".join(universe))

        async def gather_one(tk: str) -> dict[str, str]:
            tech = await tx.execute("get_technical", {"ticker": tk})
            news = await tx.execute("get_news", {"ticker": tk, "days": 2})
            mem = await tx.execute("get_memories", {"ticker": tk, "top_k": 3})
            return {
                "ticker": tk,
                "technical": (tech or "").strip(),
                "news": (news or "").strip(),
                "memories": (mem or "").strip(),
            }

        per_ticker_rows = await asyncio.gather(
            *(gather_one(tk) for tk in universe), return_exceptions=False
        )

        portfolio_block = ""
        try:
            portfolio_block = await tx.execute("get_portfolio", {})
        except Exception:
            portfolio_block = ""

        macro_extra = macro_snip if macro_snip else await tx.execute("get_macro_context", {})

        payload: dict[str, Any] = {
            "portfolio": (portfolio_block or "").strip(),
            "macro": (macro_extra or "").strip(),
            "universe": list(universe),
            "per_ticker": per_ticker_rows,
        }

        body, encoding = self._encode_prepass_payload(payload)
        if encoding == "toon":
            data_block = (
                "Below is the market data in TOON (Token-Oriented Object Notation; tabular, "
                "header row defines columns, then one row per ticker). Parse it as facts.\n\n"
                f"```toon\n{body}\n```"
            )
        else:
            data_block = (
                "=================== TOOL OUTPUTS (real data; ground truth) ===================\n\n"
                f"PORTFOLIO:\n{payload['portfolio']}\n\n"
                f"MACRO:\n{payload['macro']}\n\n"
                f"PER-TICKER (universe = {','.join(universe)}):\n\n"
                + "\n".join(
                    f"=== {r['ticker']} ===\nTECHNICAL:\n{r['technical']}\n\n"
                    f"NEWS:\n{r['news']}\n\nMEMORIES:\n{r['memories']}\n"
                    for r in per_ticker_rows
                )
                + "\n\n=================== END TOOL OUTPUTS ===================\n"
            )

        enriched = (
            f"{user_message}\n\n"
            f"{data_block}\n\n"
            "Now produce: (1) a brief chain-of-thought, then (2) the decisions JSON array.\n"
            "Use ONLY the data above for facts. If data is missing, choose SKIP/HOLD."
        )

        t0 = time.perf_counter()
        resp = None
        if self.deps.cerebras:
            try:
                resp = await self.deps.cerebras.analyze(enriched, system=system_prompt)
            except Exception as exc:
                log.error("Local-prepass Cerebras failed: %s", exc)
                await fire_operator_alert(
                    category="PaperAgent · Cerebras",
                    summary="Local prepass: Cerebras analyze failed.",
                    detail=format_exc_brief(exc),
                    dedupe_key="paper_agent_local_prepass_cerebras",
                )
        if resp is None and self.deps.groq:
            try:
                resp = await self.deps.groq.analyze(enriched, system=system_prompt)
            except Exception as exc:
                log.error("Local-prepass Groq failed: %s", exc)
                await fire_operator_alert(
                    category="PaperAgent · Groq",
                    summary="Local prepass: Groq analyze failed.",
                    detail=format_exc_brief(exc),
                    dedupe_key="paper_agent_local_prepass_groq",
                )
        if resp is None and self.deps.ollama:
            try:
                resp = await self.deps.ollama.analyze(enriched, system=system_prompt)
            except Exception as exc:
                log.error("Local-prepass Ollama failed: %s", exc)
                await fire_operator_alert(
                    category="PaperAgent · Ollama",
                    summary="Local prepass: Ollama analyze failed.",
                    detail=format_exc_brief(exc),
                    dedupe_key="paper_agent_local_prepass_ollama",
                )
        if resp is None:
            return ToolRunResult(
                reasoning_text="No LLM available for prepass; returning empty decisions.",
                decisions=[],
                model="none",
                iterations=0,
            )
        elapsed = time.perf_counter() - t0
        log.info("Local-prepass OK (%.1fs, model=%s)", elapsed, resp.model)
        reasoning, decisions = _extract_json_array(resp.text)
        return ToolRunResult(
            reasoning_text=reasoning or resp.text,
            decisions=decisions,
            model=resp.model,
            iterations=1,
        )

    async def _save_cycle_log(self, *, event_type: str, analysis: str, decisions: list[dict[str, Any]]) -> None:
        pool = self.deps.db.get_pool()
        if pool is None:
            return
        content = (
            f"EVENT: {event_type}\n"
            f"UTC: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"ANALYSIS:\n{analysis}\n\n"
            f"DECISIONS_JSON:\n{json.dumps(decisions, ensure_ascii=False)}\n"
        )
        # Use daily_reports table (exists in 001_memory.sql) to keep it Docker-safe.
        # We store `ticker` as event_type to keep unique(report_date, report_type, ticker) useful.
        query = """
        INSERT INTO daily_reports (report_date, content, report_type, ticker)
        VALUES (CURRENT_DATE, $1, 'PAPER_CYCLE', $2)
        ON CONFLICT (report_date, report_type, ticker)
        DO UPDATE SET content = EXCLUDED.content, created_at = NOW()
        """
        try:
            async with pool.acquire() as conn:
                await conn.execute(query, content, event_type[:10])
        except Exception as exc:
            log.warning("Failed to persist cycle log: %s", exc)

    async def _send_live_feed(
        self,
        *,
        event_type: str,
        decisions: list[dict[str, Any]],
        is_emergency: bool = False,
        emergency_context: dict[str, Any] | None = None,
        macro_snippet: str | None = None,
        nav_summary_line: str | None = None,
        account_currency: str = "USD",
    ) -> None:
        app = self.deps.telegram_application
        if not app:
            return
        allowed = self.deps.settings.allowed_user_ids
        if not allowed:
            return

        # B1: premarket / between-anchor ticks — no trades; keep Telegram concise.
        if not is_emergency and event_type in ("PREMARKET", "TICK"):
            clk = _paper_agent_clock_header(self.deps.settings, with_seconds=False)
            parts = [
                f"🤖 <b>AI Broker</b> · <code>{html.escape(event_type)}</code> · {html.escape(clk)}",
                "⏸ <i>Pre-market / between-event</i> — no broker orders (OPEN / MIDDAY / CLOSE ET only).",
                "Full cycle → Supabase <code>daily_reports</code> · <code>/paper log</code>",
            ]
            if decisions:
                parts.append("<b>Snapshot (max 3)</b>")
                for d in decisions[:3]:
                    t = str(d.get("ticker", "")).upper().strip()
                    a = str(d.get("action", "")).upper().strip()
                    parts.append(f"• <b>{html.escape(t)}</b> · <code>{html.escape(a)}</code>")
            else:
                parts.append("(No decision rows.)")
            msg = "\n".join(parts)
            for uid in allowed:
                try:
                    await app.bot.send_message(chat_id=uid, text=msg, parse_mode=ParseMode.HTML)
                except Exception:
                    pass
            return

        # Keep feed compact.
        head = (
            f"🤖 <b>AI Broker</b> · <code>{html.escape(event_type)}</code> · "
            f"{html.escape(_paper_agent_clock_header(self.deps.settings, with_seconds=False))}"
        )
        if is_emergency:
            head = (
                f"🚨 <b>ACİL</b> · <code>{html.escape(event_type)}</code> · "
                f"{html.escape(_paper_agent_clock_header(self.deps.settings, with_seconds=True))}"
            )
            extra: list[str] = []
            ctx = emergency_context or {}
            score = ctx.get("impact_score")
            sent = str(ctx.get("sentiment", "") or "").upper()
            if score is not None:
                urg = "HIGH" if float(score) >= 8 else ("MED" if float(score) >= 6 else "LOW")
                extra.append(
                    f"⚡ <b>Aciliyet</b>: {html.escape(urg)} · "
                    f"<b>Etki</b>: {html.escape(str(score))}/10 {html.escape(sent)}".strip()
                )
            snippet = str(ctx.get("post_text", "") or "").strip().replace("\n", " ")
            if snippet and len(snippet) > 160:
                snippet = snippet[:160] + "…"
            if snippet:
                extra.append(f"📝 \"{html.escape(snippet)}\"")
            img = ctx.get("image_analysis")
            if img and str(img).strip():
                ia = str(img).strip().replace("\n", " ")
                if len(ia) > 120:
                    ia = ia[:120] + "…"
                extra.append(f"📷 {html.escape(ia)}")
            head = head + ("\n" + "\n".join(extra) if extra else "")

        sub: list[str] = []
        if nav_summary_line:
            sub.append(html.escape(nav_summary_line[:350] + ("…" if len(nav_summary_line) > 350 else "")))
        if macro_snippet:
            m = macro_snippet.replace("\n", " ").strip()
            if len(m) > 280:
                m = m[:280] + "…"
            sub.append(f"📊 <b>Macro</b>: {html.escape(m)}")
        if sub:
            head = head + "\n" + "—" * 24 + "\n" + "\n".join(sub)

        if not decisions:
            msg = head + "\n\nNo decisions."
        else:
            lines = [head, "—" * 24]
            for d in decisions[:5]:
                t = str(d.get("ticker", "")).upper().strip()
                a = str(d.get("action", "")).upper().strip()
                conf = d.get("confidence", None)
                edge = str(d.get("edge_depth", "") or "").strip()
                hyp = (d.get("hypothesis") or d.get("reasoning") or "").strip()
                if hyp and len(hyp) > 140:
                    hyp = hyp[:140] + "…"
                conf_s = f"{float(conf):.2f}" if conf is not None else "N/A"
                edge_s = f" | {edge}" if edge else ""
                lines.append(
                    f"<b>{html.escape(t)}</b> · <code>{html.escape(a)}</code> · "
                    f"conf=<code>{html.escape(conf_s)}</code>{html.escape(edge_s)}"
                )
                if hyp:
                    lines.append(f"  {html.escape(hyp)}")
            try:
                nav_now, cash_now, _ = await self._estimate_nav_mtm()
                start = float(self.deps.settings.paper_starting_nav_usd)
                ret_pct = (nav_now - start) / start * 100.0 if start else 0.0
                ac = (account_currency or "USD").upper()[:3]
                lines.append("—" * 24)
                lines.append(
                    f"📊 <b>NAV</b> ~{nav_now:,.0f} {ac} ({ret_pct:+.1f}% vs {start:,.0f} {ac} start) · "
                    f"<b>Cash</b> {cash_now:,.0f} {ac}"
                )
            except Exception:
                pass
            msg = "\n".join(lines)

        for uid in allowed:
            try:
                await app.bot.send_message(chat_id=uid, text=msg, parse_mode=ParseMode.HTML)
            except Exception:
                # Don't spam logs for transient errors.
                pass

    @staticmethod
    def _fetch_yf_price(symbol: str) -> float | None:
        import yfinance as yf

        try:
            t = yf.Ticker(symbol)
            if hasattr(t, "fast_info") and "lastPrice" in t.fast_info:
                return float(t.fast_info["lastPrice"])
            info = t.info
            v = info.get("currentPrice") or info.get("regularMarketPrice")
            return float(v) if v is not None else None
        except Exception:
            return None
