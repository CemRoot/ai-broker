"""
Tool execution layer for Faz 3 Paper Agent.

Each tool returns a compact **English** string (log-friendly, LLM-friendly).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from app.core.config import Settings
from app.core.logging import get_logger
from app.memory.database import SupabaseDatabase
from app.memory.retriever import RAGRetriever
from app.services.paper.broker import PaperBroker
from app.services.llm.cerebras_service import CerebrasService
from app.services.llm.groq_service import GroqService

log = get_logger("tools.executor")


def _try_toon(payload: dict, *, header: str) -> str | None:
    """Encode ``payload`` as TOON when toon-format is installed; otherwise None.

    The wrapper prefixes a one-line ``header`` so the LLM still sees a familiar
    title (e.g. ``S&P 500 SCREENER (TOON):``). Falls back gracefully so the
    caller can keep its plaintext path.
    """
    try:
        import toon_format

        body = toon_format.encode(payload)
        return f"{header}\n```toon\n{body}\n```"
    except Exception:
        return None


@dataclass(frozen=True)
class ToolDeps:
    settings: Settings
    db: SupabaseDatabase | None
    http_client: Any  # httpx.AsyncClient (kept Any to avoid import cycles)
    cerebras: CerebrasService | None
    groq: GroqService | None
    retriever: RAGRetriever | None
    paper_broker: PaperBroker | None
    # Faz 3d: `app/services/screener.py` will provide SPScreener; keep Any for now
    screener: Any | None = None
    #: When ``paper_executes_on_t212``, portfolio tool reads from T212 API.
    t212: Any | None = None


class ToolExecutor:
    def __init__(self, deps: ToolDeps) -> None:
        self.deps = deps
        self._technical_cache: dict[str, tuple[float, str]] = {}

    async def execute(self, tool_name: str, arguments: dict | None = None) -> str:
        args = arguments or {}
        name = (tool_name or "").strip()
        try:
            if name == "get_news":
                return await self._get_news(
                    ticker=str(args.get("ticker", "")).upper(),
                    days=int(args.get("days", 2) or 2),
                )
            if name == "get_technical":
                return await self._get_technical(ticker=str(args.get("ticker", "")).upper())
            if name == "get_memories":
                return await self._get_memories(
                    ticker=str(args.get("ticker", "")).upper(),
                    query=str(args.get("query", "") or ""),
                    top_k=int(args.get("top_k", 5) or 5),
                )
            if name == "get_portfolio":
                return await self._get_portfolio()
            if name == "get_macro_context":
                return await self._get_macro_context()
            if name == "screen_stocks":
                return await self._screen_stocks(
                    min_volume_ratio=float(args.get("min_volume_ratio", 1.5) or 1.5)
                )
            if name == "check_t212_equity_instrument":
                return await self._check_t212_equity_instrument(
                    ticker=str(args.get("ticker", "") or "").strip(),
                )

            return f"ERROR: Unknown tool '{name}'."
        except Exception as exc:
            log.error("Tool %s failed: %s", name, exc)
            return f"ERROR: Tool '{name}' failed: {type(exc).__name__}: {exc}"

    async def _get_news(self, ticker: str, days: int = 2) -> str:
        if not ticker:
            return "ERROR: ticker is required."
        if not self.deps.settings.finnhub_api_key:
            return f"NEWS FOR {ticker}: ERROR: FINNHUB_API_KEY not configured."
        if not self.deps.http_client:
            return f"NEWS FOR {ticker}: ERROR: HTTP client not initialised."

        # Lazy imports: keep module import fast for CLI/tests.
        from app.services.finnhub_news import fetch_company_news
        from app.services.news_pipeline import analyze_news_batch

        articles = await fetch_company_news(
            self.deps.http_client,
            api_key=self.deps.settings.finnhub_api_key,
            symbol=ticker,
            days=max(1, min(days, 7)),
        )
        if not articles:
            return f"NEWS FOR {ticker} (last {days} days): No articles."

        scored, model_used = await analyze_news_batch(
            symbol=ticker,
            articles=articles,
            cerebras=self.deps.cerebras,
            groq=self.deps.groq,
        )

        # Compact sentiment rollup
        pos = sum(1 for a in scored if a.get("relevant") and a.get("sentiment") == "positive")
        neg = sum(1 for a in scored if a.get("relevant") and a.get("sentiment") == "negative")
        neu = sum(1 for a in scored if a.get("relevant") and a.get("sentiment") == "neutral")
        total_rel = pos + neg + neu
        score = 0.0
        if total_rel:
            score = (pos - neg) / float(total_rel)
        stance = "BULLISH" if score > 0.15 else ("BEARISH" if score < -0.15 else "MIXED/NEUTRAL")

        lines: list[str] = [f"NEWS FOR {ticker} (last {days} days) | model={model_used}:"]
        for i, a in enumerate(scored[:8]):  # keep small; LLM can ask again if needed
            title = (a.get("title") or "").strip()
            sent = (a.get("sentiment") or "neutral").upper()
            summ = (a.get("summary") or "").strip()
            if not title:
                continue
            lines.append(f"- [{i}] {title}")
            if summ:
                lines.append(f"  Sentiment: {sent} | Summary: {summ}")
        lines.append("")
        lines.append(f"OVERALL SENTIMENT: {stance} (score: {score:+.2f}, relevant: {total_rel})")
        return "\n".join(lines).strip()

    async def _get_technical(self, ticker: str) -> str:
        if not ticker:
            return "ERROR: ticker is required."
        ttl = int(getattr(self.deps.settings, "paper_technical_cache_ttl_sec", 0) or 0)
        if ttl > 0:
            now = time.monotonic()
            hit = self._technical_cache.get(ticker.upper())
            if hit and hit[0] > now:
                return hit[1]
        # Lazy imports: pandas/pandas_ta can be heavy to import.
        from app.tools.technical import get_technical_summary
        from app.tools.technical_extended import get_extended_price_features

        summary = await get_technical_summary(ticker)
        ext = await get_extended_price_features(ticker)

        lines: list[str] = [f"TECHNICAL ANALYSIS FOR {ticker}:"]
        if summary.error:
            lines.append(f"ERROR: {summary.error}")
            return "\n".join(lines)

        price = summary.last_close
        if price is not None:
            lines.append(f"Price: ${price:.2f}")
        if summary.rsi_14 is not None:
            rsi = summary.rsi_14
            regime = "OVERBOUGHT" if rsi > 70 else ("OVERSOLD" if rsi < 30 else "NEUTRAL")
            lines.append(f"RSI(14): {rsi:.1f} ({regime})")
        if summary.sma_20 is not None:
            lines.append(f"SMA20: ${summary.sma_20:.2f}")
        if summary.volume is not None:
            lines.append(f"Volume (last): {summary.volume:,.0f}")

        if ext and not ext.error:
            # Keep a compact subset
            f = ext.features
            keys = ["ret_1d", "ret_5d", "volatility_10d", "volume_ratio_5d", "ma5_vs_ma20", "rsi_14"]
            parts = []
            for k in keys:
                v = f.get(k)
                if v is None:
                    continue
                parts.append(f"{k}={v:.4f}")
            if parts:
                lines.append("Price-features: " + " | ".join(parts))
        out = "\n".join(lines).strip()
        if ttl > 0:
            self._technical_cache[ticker.upper()] = (time.monotonic() + float(ttl), out)
        return out

    async def _get_memories(self, ticker: str, query: str = "", top_k: int = 5) -> str:
        if not ticker:
            return "ERROR: ticker is required."
        if not self.deps.retriever:
            return f"PAST EXPERIENCES FOR {ticker}: ERROR: retriever not initialised."

        q = (query or "").strip()
        if not q:
            q = f"Trading lessons and outcomes for {ticker}"

        rows = await self.deps.retriever.search_similar_memories(
            ticker=ticker,
            query_text=q,
            top_k=max(1, min(int(top_k), 10)),
            match_threshold=0.45,
        )
        if not rows:
            return f"PAST EXPERIENCES FOR {ticker}: No similar memories found."

        lines = [f"PAST EXPERIENCES FOR {ticker} (top {min(len(rows), 10)} similar):"]
        for r in rows[:10]:
            mtype = (r.get("memory_type") or "UNKNOWN").upper()
            ctx = (r.get("context") or "").strip().replace("\n", " ")
            out = (r.get("outcome") or "").upper()
            pnl = r.get("pnl_percent")
            tail = []
            if out:
                tail.append(f"Outcome: {out}")
            if pnl is not None:
                try:
                    tail.append(f"PnL: {float(pnl):+.1f}%")
                except Exception:
                    pass
            suffix = " | " + " | ".join(tail) if tail else ""
            if len(ctx) > 220:
                ctx = ctx[:220] + "…"
            lines.append(f"- [{mtype}] {ctx}{suffix}")
        return "\n".join(lines).strip()

    async def _get_portfolio(self) -> str:
        if self.deps.settings.paper_executes_on_t212 and self.deps.t212:
            return await self._get_t212_portfolio_string()

        broker = self.deps.paper_broker
        if not broker:
            return "CURRENT PAPER PORTFOLIO: ERROR: paper broker not initialised."

        balance = await broker.get_balance()
        positions = await broker.get_positions()
        cur = (self.deps.settings.paper_account_currency or "USD").upper()[:3]

        rows: list[dict] = []
        total_value = balance
        for p in positions:
            live_price = await asyncio.to_thread(self._fetch_yf_price, p.ticker)
            if live_price is None:
                price = None
                pnl_pct = None
                total_value += p.shares * p.avg_cost
            else:
                price = float(live_price)
                pnl_pct = ((price - p.avg_cost) / p.avg_cost) * 100.0 if p.avg_cost else 0.0
                total_value += p.shares * price
            rows.append({
                "ticker": p.ticker,
                "shares": round(p.shares, 4),
                "avg_cost": round(p.avg_cost, 4),
                "current_yf": round(price, 4) if price is not None else None,
                "pnl_pct": round(pnl_pct, 2) if pnl_pct is not None else None,
            })

        if self.deps.settings.use_toon_prompts:
            packed = _try_toon(
                {
                    "currency": cur,
                    "cash": round(balance, 2),
                    "total_value_approx": round(total_value, 2),
                    "positions": rows,
                },
                header="CURRENT PAPER PORTFOLIO (TOON):",
            )
            if packed is not None:
                return packed

        lines = ["CURRENT PAPER PORTFOLIO:",
                 f"Account currency (ledger): {cur}",
                 f"Cash: {balance:,.2f} {cur}"]
        if not rows:
            lines.append("POSITIONS: none")
            lines.append(f"Total Value (approx): {total_value:,.2f} {cur}")
            return "\n".join(lines)
        lines.append("")
        lines.append("POSITIONS:")
        for r in rows:
            if r["current_yf"] is None:
                lines.append(f"- {r['ticker']}: {r['shares']:.2f} @ {r['avg_cost']:.2f} {cur} | Current: N/A")
            else:
                lines.append(
                    f"- {r['ticker']}: {r['shares']:.2f} @ {r['avg_cost']:.2f} {cur} | "
                    f"Current (yfinance, often USD): {r['current_yf']:.2f} | PnL: {r['pnl_pct']:+.1f}%"
                )
        lines.append("")
        lines.append(f"Total Value (approx): {total_value:,.2f} {cur}")
        return "\n".join(lines).strip()

    async def _get_t212_portfolio_string(self) -> str:
        from app.services.t212.ticker_map import t212_to_yfinance

        t212 = self.deps.t212
        try:
            summary = await t212.get_account_summary()
            positions = await t212.get_positions()
        except Exception as exc:
            return f"T212 PORTFOLIO (execution backend): ERROR — {type(exc).__name__}: {exc}"

        cur = summary.get("currency") or "?"
        cash = summary.get("cash") or {}
        inv = summary.get("investments") or {}
        rows = [
            {
                "yf": t212_to_yfinance(p.ticker),
                "t212_ticker": p.ticker,
                "qty": round(p.quantity, 4),
                "avg": round(p.average_price_paid, 2),
                "last": round(p.current_price, 2),
                "ppl_pct": round(p.pnl_percent, 2),
            }
            for p in sorted(positions, key=lambda x: x.ticker)
        ]
        if self.deps.settings.use_toon_prompts:
            packed = _try_toon(
                {
                    "account_currency": cur,
                    "total_value": float(summary.get("totalValue") or 0),
                    "available_to_trade": float(cash.get("availableToTrade") or 0),
                    "investments_mv": float(inv.get("currentValue") or 0),
                    "unrealized_pl": float(inv.get("unrealizedProfitLoss") or 0),
                    "positions": rows,
                },
                header="T212 ACCOUNT (Paper Agent execution backend) (TOON):",
            )
            if packed is not None:
                return packed

        lines = [
            "T212 ACCOUNT (Paper Agent uses this broker for execution):",
            f"Account currency: {cur}",
            f"Total value: {float(summary.get('totalValue') or 0):,.2f} {cur}",
            f"Available to trade: {float(cash.get('availableToTrade') or 0):,.2f} {cur}",
            f"Investments MV: {float(inv.get('currentValue') or 0):,.2f} {cur} | "
            f"Unrealized P/L: {float(inv.get('unrealizedProfitLoss') or 0):,.2f} {cur}",
            "",
            "OPEN POSITIONS:",
        ]
        if not rows:
            lines.append("- (none)")
        else:
            for r in rows:
                lines.append(
                    f"- {r['yf']} [{r['t212_ticker']}] qty={r['qty']:.4f} avg={r['avg']:.2f} "
                    f"last={r['last']:.2f} ppl%={r['ppl_pct']:+.2f}%"
                )
        return "\n".join(lines).strip()

    async def _get_macro_context(self) -> str:
        now = datetime.now(timezone.utc)
        vix = await asyncio.to_thread(self._fetch_yf_price, "^VIX")
        if vix is None:
            vix = await asyncio.to_thread(self._fetch_vix_close_download)
        if vix is None:
            vix_str = "N/A"
            regime = "UNKNOWN"
        else:
            vix_str = f"{float(vix):.1f}"
            vv = float(vix)
            regime = "HIGH" if vv > 25 else ("NORMAL" if vv >= 15 else "LOW")

        lines = ["MACRO CONTEXT:"]
        lines.append(f"Timestamp (UTC): {now.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"VIX: {vix_str} ({regime})")
        lines.append(f"Market regime (rule-of-thumb): VIX {regime} — use with price action, not alone.")

        # Fear & Greed (CNN) — single endpoint covers composite + put/call + VIX cross-check
        if getattr(self.deps.settings, "macro_fear_greed_enabled", True) and self.deps.http_client:
            try:
                from app.services.macro import CNNFearGreedClient

                snap = await CNNFearGreedClient(self.deps.http_client).fetch()
                if snap is not None:
                    lines.extend(snap.to_lines())
            except Exception as exc:
                log.warning("CNN F&G enrichment skipped: %s", exc)

        lines.append("")
        lines.append("Upcoming scheduled events (static reference list; verify dates):")
        lines.append("- FOMC decision days (typically 8x/year) — check official Fed calendar.")
        lines.append("- Monthly US CPI release (BLS schedule).")
        lines.append("- Monthly US NFP (first Friday).")
        lines.append("")
        lines.append("Recent high-impact Trump / Truth Social posts (DB, last 24h):")
        trump_lines = await self._fetch_recent_trump_posts(hours=24, limit=5)
        if not trump_lines:
            lines.append("- (No rows in trump_posts table or DB unavailable.)")
        else:
            lines.extend(trump_lines)
        return "\n".join(lines).strip()

    async def _fetch_recent_trump_posts(self, *, hours: int, limit: int) -> list[str]:
        db = self.deps.db
        if not db or db.get_pool() is None:
            return []
        pool = db.get_pool()
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max(1, hours))
        q = """
        SELECT posted_at, impact_score, sentiment, post_text, affected_tickers
        FROM trump_posts
        WHERE posted_at >= $1
        ORDER BY posted_at DESC
        LIMIT $2
        """
        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(q, cutoff, max(1, min(limit, 20)))
        except Exception as exc:
            log.warning("trump_posts read failed: %s", exc)
            return []
        out: list[str] = []
        for r in rows:
            posted = r.get("posted_at")
            when = posted.isoformat() if posted else ""
            imp = r.get("impact_score")
            sent = (r.get("sentiment") or "").upper()
            txt = (r.get("post_text") or "").strip().replace("\n", " ")
            if len(txt) > 160:
                txt = txt[:160] + "…"
            tickers = r.get("affected_tickers") or []
            tick_s = ", ".join(str(t) for t in tickers[:8]) if tickers else ""
            out.append(
                f"- [{when}] Impact {imp}/10 {sent} | Tickers: {tick_s or 'n/a'} | \"{txt}\""
            )
        return out

    async def _check_t212_equity_instrument(self, ticker: str) -> str:
        if not ticker:
            return "ERROR: ticker is required."
        if not self.deps.settings.paper_executes_on_t212:
            return (
                "T212 EQUITY CHECK: skipped (PAPER_EXECUTION_BACKEND is not t212 — "
                "virtual ledger mode has no broker instrument list)."
            )
        if not self.deps.t212:
            return "T212 EQUITY CHECK: ERROR: T212 client not configured."
        try:
            ok, detail = await self.deps.t212.is_us_equity_instrument_tradeable(ticker)
        except Exception as exc:
            return f"T212 EQUITY CHECK {ticker.upper()}: ERROR: {type(exc).__name__}: {exc}"
        sym = ticker.upper().strip()
        if ok:
            return f"T212 EQUITY CHECK {sym}: OK — tradeable as {detail} (STOCK/ETF on this account)."
        return f"T212 EQUITY CHECK {sym}: NO — {detail}"

    async def _screen_stocks(self, min_volume_ratio: float = 1.5) -> str:
        if not self.deps.screener:
            return "S&P 500 SCREENER RESULTS: ERROR: screener not initialised."
        rows = await self.deps.screener.get_candidates(min_volume_ratio=min_volume_ratio)
        if not rows:
            return "S&P 500 SCREENER RESULTS: No candidates."

        toon_rows: list[dict] = []
        for r in rows[:5]:
            toon_rows.append({
                "ticker": str(r.get("ticker") or r.get("symbol") or "?"),
                "vol_ratio": round(float(r.get("volume_ratio") or 0.0), 2),
                "momentum_1d_pct": round(float(r.get("momentum_1d") or 0.0), 2),
                "sector": str(r.get("sector") or ""),
            })
        if self.deps.settings.use_toon_prompts:
            packed = _try_toon({"candidates": toon_rows}, header="S&P 500 SCREENER RESULTS (TOON):")
            if packed is not None:
                return packed
        lines = ["S&P 500 SCREENER RESULTS:"]
        for r in toon_rows:
            sec_s = f" | Sector: {r['sector']}" if r["sector"] else ""
            lines.append(
                f"- {r['ticker']}: Vol ratio {r['vol_ratio']:.2f}x | "
                f"Momentum {r['momentum_1d_pct']:+.2f}%{sec_s}"
            )
        return "\n".join(lines).strip()

    @staticmethod
    def _fetch_yf_price(symbol: str) -> float | None:
        # Lazy import: yfinance import is non-trivial.
        import yfinance as yf

        try:
            t = yf.Ticker(symbol)
            if hasattr(t, "fast_info") and "lastPrice" in t.fast_info:
                return t.fast_info["lastPrice"]
            info = t.info
            return info.get("currentPrice") or info.get("regularMarketPrice")
        except Exception:
            return None

    @staticmethod
    def _fetch_vix_close_download() -> float | None:
        """Last daily close for ^VIX (fallback when fast_info is empty)."""
        import yfinance as yf

        try:
            raw = yf.download("^VIX", period="5d", interval="1d", progress=False, auto_adjust=True)
            if raw is None or raw.empty:
                return None
            if hasattr(raw.columns, "levels"):
                raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
            close = raw["Close"].astype(float)
            return float(close.iloc[-1])
        except Exception:
            return None
