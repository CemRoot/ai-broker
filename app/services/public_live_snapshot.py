"""
Read-only aggregates for the public live dashboard (``GET /public/live``).

No secrets, no internal API key — intended for a small audience showcase
(``PUBLIC_DASHBOARD_ENABLED``). Truncates long LLM text defensively.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import Request

from app.core.logging import get_logger
from app.services.paper.account_currency import resolve_paper_account_currency
from app.services.t212.ticker_map import t212_to_yfinance

log = get_logger("public_live")

# Server-side cache for the T212 fetch inside ``/public/live``.
#
# Why: T212 Public API enforces *per-endpoint* rate limits (e.g. ``equity/positions``
# ≈ 1 req / 5 s, ``account/info`` ≈ 1 req / 30 s). The dashboard polls every 60 s
# and ``T212MirrorPoller`` polls every 90 s — when those align we get a 429 storm
# that delays every public-live response by 6+ seconds. A short snapshot cache
# (30 s) shields T212 from the public surface without hiding real movement: paper
# trades execute on cycle boundaries (≥ minutes apart), so a half-minute lag is
# invisible to humans watching the dashboard.
# Aligned with the dashboard's 60 s polling cadence so each client refresh
# typically reads the cache instead of triggering an upstream T212 round-trip.
_T212_CACHE_TTL_SEC: float = 60.0
_t212_cache: dict[str, Any] | None = None
_t212_cache_at: float = 0.0
_t212_cache_lock = asyncio.Lock()

# Whole-snapshot cache — Supabase (cycles + trades + ledger + memories) and the
# T212 fetch are both rolled into a single dict; with 15 s TTL several browser
# refreshes inside the same minute share one full computation instead of 4–10
# seconds of repeated DB round-trips. Background loops (PaperAgent, mirror
# poller) still write to Supabase directly and are unaffected.
_SNAPSHOT_CACHE_TTL_SEC: float = 15.0
_snapshot_cache: dict[str, Any] | None = None
_snapshot_cache_at: float = 0.0
_snapshot_cache_lock = asyncio.Lock()

_MAX_ANALYSIS = 900
_MAX_REASON = 320
_SECRETISH = re.compile(
    r"(?i)\b("
    r"gsk_[a-z0-9]{10,}|"
    r"sk-[a-z0-9]{10,}|"
    r"Bearer\s+[a-z0-9\-._]{20,}|"
    r"postgresql://[^\s]+"
    r")\S*"
)


def redact_secrets(text: str) -> str:
    return _SECRETISH.sub("[redacted]", text or "")


def truncate(text: str, max_len: int) -> str:
    t = (text or "").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 1] + "…"


async def _fetch_t212_account_cached(t212: Any, currency: str) -> dict[str, Any]:
    """Return a cached T212 ``account+positions`` snapshot for the public dashboard.

    A ~30 s in-process cache absorbs concurrent ``/public/live`` callers and
    background pollers so we don't churn through the per-endpoint rate budget.
    On error we cache the failure briefly too — repeatedly hammering a degraded
    upstream just makes things worse.
    """
    global _t212_cache, _t212_cache_at
    now = time.monotonic()
    if _t212_cache is not None and (now - _t212_cache_at) < _T212_CACHE_TTL_SEC:
        # Stamp ``cached_age_s`` so the dashboard can hint at staleness.
        cached = dict(_t212_cache)
        cached["cached_age_s"] = round(now - _t212_cache_at, 1)
        return cached

    async with _t212_cache_lock:
        # Re-check inside the lock to avoid a thundering herd.
        now = time.monotonic()
        if _t212_cache is not None and (now - _t212_cache_at) < _T212_CACHE_TTL_SEC:
            cached = dict(_t212_cache)
            cached["cached_age_s"] = round(now - _t212_cache_at, 1)
            return cached

        fetched_iso = datetime.now(timezone.utc).isoformat()
        out: dict[str, Any] = {
            "ok": False,
            "fetched_at": fetched_iso,
            "error": None,
            "account_currency": currency,
            "total_value": None,
            "cash_available": None,
            "cash_blocked": None,
            "position_count": 0,
            "positions": [],
            "pending_orders": [],
            "pending_orders_count": 0,
            "pending_orders_error": None,
            "cached_age_s": 0.0,
        }
        try:
            summary = await t212.get_account_summary()
            pos_live = await t212.get_positions()
            # Pending orders — separate ``GET /equity/orders`` (1 req / 5s).
            # If this fails, we still want positions/summary to render, so we
            # downgrade the failure to a list and an inline error.
            pending_raw: list[dict[str, Any]] = []
            pending_error: str | None = None
            try:
                pending_raw = await t212.get_all_pending_orders()
            except Exception as exc:
                pending_error = f"{type(exc).__name__}: {exc!s}"[:200]
                log.warning("public live T212 pending orders fetch failed: %s", exc)
            acct_cur = str(summary.get("currency") or currency or "USD").upper()[:3]
            tv = float(summary.get("totalValue") or 0.0)
            cash_b = summary.get("cash") or {}
            avail = float(cash_b.get("availableToTrade") or 0.0)
            blocked = float(cash_b.get("blocked") or 0.0)
            pos_out: list[dict[str, Any]] = []
            for p in sorted(pos_live, key=lambda x: x.ticker):
                yf = t212_to_yfinance(p.ticker)
                qty = float(p.quantity or 0.0)
                avg = float(p.average_price_paid or 0.0)
                last = float(p.current_price or 0.0)
                cur_val = qty * last if last > 0 and qty > 0 else None
                pos_out.append(
                    {
                        "t212_ticker": p.ticker,
                        "symbol": yf,
                        "quantity": qty,
                        "average_price_paid": round(avg, 6),
                        "current_price": round(last, 6),
                        "current_value": round(cur_val, 2) if cur_val is not None else None,
                        "ppl": round(float(p.pnl), 4),
                        "ppl_percent": round(float(p.pnl_percent), 4),
                    }
                )
            pending_out: list[dict[str, Any]] = []
            for o in pending_raw:
                # Trading 212 schema: id, ticker, quantity (signed), type
                # (MARKET|LIMIT|STOP|STOP_LIMIT), limitPrice, stopPrice,
                # filledQuantity, filledValue, value (notional, signed),
                # status, creationTime, strategy ('QUANTITY'|'VALUE').
                # See https://t212public-api-docs.redoc.ly/.
                qty = float(o.get("quantity") or 0.0)
                value_signed = o.get("value")
                notional = abs(float(value_signed)) if value_signed is not None else None
                pending_out.append(
                    {
                        "order_id": o.get("id"),
                        "t212_ticker": o.get("ticker"),
                        "symbol": t212_to_yfinance(o.get("ticker") or ""),
                        "side": "BUY" if qty >= 0 else "SELL",
                        "quantity": abs(qty),
                        "type": (o.get("type") or "").upper(),
                        "limit_price": (
                            round(float(o["limitPrice"]), 4)
                            if o.get("limitPrice") is not None
                            else None
                        ),
                        "stop_price": (
                            round(float(o["stopPrice"]), 4)
                            if o.get("stopPrice") is not None
                            else None
                        ),
                        "notional": round(notional, 2) if notional is not None else None,
                        "status": o.get("status"),
                        "created_at": o.get("creationTime"),
                    }
                )
            out = {
                "ok": True,
                "fetched_at": fetched_iso,
                "error": None,
                "account_currency": acct_cur,
                "total_value": round(tv, 2),
                "cash_available": round(avail, 2),
                "cash_blocked": round(blocked, 2),
                "position_count": len(pos_out),
                "positions": pos_out,
                "pending_orders": pending_out,
                "pending_orders_count": len(pending_out),
                "pending_orders_error": pending_error,
                "cached_age_s": 0.0,
            }
        except Exception as exc:
            log.warning("public live T212 fetch failed: %s", exc)
            out["error"] = f"{type(exc).__name__}: {exc!s}"[:240]

        _t212_cache = out
        _t212_cache_at = time.monotonic()
        return dict(out)


def parse_paper_cycle_content(content: str) -> tuple[str, list[dict[str, Any]]]:
    """Split ``daily_reports`` PAPER_CYCLE row into analysis excerpt + decisions list."""
    raw = content or ""
    if "ANALYSIS:" not in raw or "DECISIONS_JSON:" not in raw:
        return truncate(redact_secrets(raw), _MAX_ANALYSIS), []
    try:
        a0 = raw.index("ANALYSIS:") + len("ANALYSIS:")
        j0 = raw.index("DECISIONS_JSON:")
        analysis = raw[a0:j0].strip()
        js = raw[j0 + len("DECISIONS_JSON:") :].strip()
        decisions = json.loads(js)
        if not isinstance(decisions, list):
            decisions = []
    except (ValueError, json.JSONDecodeError):
        return truncate(redact_secrets(raw), _MAX_ANALYSIS), []
    return truncate(redact_secrets(analysis), _MAX_ANALYSIS), decisions


async def build_public_live_snapshot(request: Request) -> dict[str, Any]:
    """Public dashboard snapshot, wrapped in a short whole-response cache."""
    global _snapshot_cache, _snapshot_cache_at

    now = time.monotonic()
    if _snapshot_cache is not None and (now - _snapshot_cache_at) < _SNAPSHOT_CACHE_TTL_SEC:
        cached = dict(_snapshot_cache)
        cached["snapshot_cached_age_s"] = round(now - _snapshot_cache_at, 1)
        return cached

    async with _snapshot_cache_lock:
        now = time.monotonic()
        if _snapshot_cache is not None and (now - _snapshot_cache_at) < _SNAPSHOT_CACHE_TTL_SEC:
            cached = dict(_snapshot_cache)
            cached["snapshot_cached_age_s"] = round(now - _snapshot_cache_at, 1)
            return cached

        out = await _compute_public_live_snapshot(request)
        _snapshot_cache = out
        _snapshot_cache_at = time.monotonic()
        return dict(out)


async def _compute_public_live_snapshot(request: Request) -> dict[str, Any]:
    from app.api.routes import build_health_payload

    settings = getattr(request.app.state, "settings", None)
    db = getattr(request.app.state, "db", None)
    bot_app = getattr(request.app.state, "bot_app", None)
    t212 = getattr(request.app.state, "t212", None)
    market_clock = getattr(request.app.state, "market_clock", None)

    health = build_health_payload(settings=settings, db=db, bot_app=bot_app)
    pool = db.get_pool() if db else None

    # Market session — give the dashboard enough signal to explain why a
    # weekend / overnight snapshot has no fresh decisions.
    market_session: dict[str, Any] | None = None
    if market_clock is not None:
        try:
            from zoneinfo import ZoneInfo

            et = ZoneInfo("America/New_York")
            now_et = datetime.now(tz=et)
            is_open = bool(market_clock.is_market_open(now_et))
            next_open_et = next_close_et = None
            if not is_open:
                next_open_et = market_clock.next_open(now_et).isoformat()
            else:
                next_close_et = market_clock.next_close(now_et).isoformat()
            market_session = {
                "is_open": is_open,
                "now_et": now_et.isoformat(),
                "next_open_et": next_open_et,
                "next_close_et": next_close_et,
            }
        except Exception as exc:
            log.warning("market_session probe failed: %s: %s", type(exc).__name__, exc)
            market_session = None

    currency = "USD"
    if settings:
        try:
            currency = await resolve_paper_account_currency(settings, t212)
        except Exception as exc:
            log.debug("resolve_paper_account_currency: %s", exc)
            currency = (getattr(settings, "paper_account_currency", None) or "USD").upper()[:3]

    starting_nav = float(getattr(settings, "paper_starting_nav_usd", 20_000.0) or 20_000.0)
    balance = 0.0
    ledger_updated_at: str | None = None
    positions: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    cycles: list[dict[str, Any]] = []
    memories_count = 0
    today_realized = 0.0
    pnl_by_day: list[dict[str, Any]] = []

    if pool:
        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow("SELECT balance, updated_at FROM paper_account WHERE id = 1")
                if row:
                    balance = float(row["balance"] or 0.0)
                    bu = row["updated_at"]
                    ledger_updated_at = bu.isoformat() if bu else None

                prow = await conn.fetch(
                    """
                    SELECT ticker, shares, avg_cost, current_value, updated_at
                    FROM paper_portfolio
                    WHERE status = 'OPEN' AND shares > 0
                    ORDER BY ticker
                    """
                )
                for r in prow:
                    positions.append(
                        {
                            "ticker": r["ticker"],
                            "shares": float(r["shares"] or 0.0),
                            "avg_cost": float(r["avg_cost"] or 0.0),
                            "current_value": float(r["current_value"])
                            if r["current_value"] is not None
                            else None,
                            "updated_at": r["updated_at"].isoformat()
                            if r["updated_at"]
                            else None,
                        }
                    )

                trows = await conn.fetch(
                    """
                    SELECT id, ticker, action, shares, price, total_value,
                           reasoning, pnl_percent, realized_pnl_usd,
                           stop_loss, target, cycle_event, emergency, created_at
                    FROM paper_trades
                    ORDER BY created_at DESC
                    LIMIT 40
                    """
                )
                for r in trows:
                    trades.append(
                        {
                            "id": int(r["id"]),
                            "ticker": r["ticker"],
                            "action": r["action"],
                            "shares": float(r["shares"] or 0.0),
                            "price": float(r["price"] or 0.0),
                            "total_value": float(r["total_value"] or 0.0),
                            "reasoning": truncate(redact_secrets(str(r["reasoning"] or "")), _MAX_REASON),
                            "pnl_percent": float(r["pnl_percent"])
                            if r["pnl_percent"] is not None
                            else None,
                            "realized_pnl": float(r["realized_pnl_usd"])
                            if r["realized_pnl_usd"] is not None
                            else None,
                            "stop_loss": float(r["stop_loss"]) if r["stop_loss"] is not None else None,
                            "target": float(r["target"]) if r["target"] is not None else None,
                            "cycle_event": r["cycle_event"],
                            "emergency": bool(r["emergency"]) if r["emergency"] is not None else False,
                            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                        }
                    )

                crows = await conn.fetch(
                    """
                    SELECT report_date, ticker, content, created_at
                    FROM daily_reports
                    WHERE report_type = 'PAPER_CYCLE'
                    ORDER BY created_at DESC
                    LIMIT 10
                    """
                )
                for r in crows:
                    excerpt, decisions = parse_paper_cycle_content(str(r["content"] or ""))
                    slim_decisions: list[dict[str, Any]] = []
                    for d in decisions[:12]:
                        if not isinstance(d, dict):
                            continue
                        sym = str(d.get("ticker", "") or "").upper().strip()
                        act = str(d.get("action", "") or "").upper().strip()
                        conf = d.get("confidence")
                        try:
                            conf_f = float(conf) if conf is not None else None
                        except (TypeError, ValueError):
                            conf_f = None
                        slim_decisions.append(
                            {
                                "ticker": sym,
                                "action": act or "—",
                                "confidence": conf_f,
                                "reasoning": truncate(
                                    redact_secrets(str(d.get("reasoning", "") or "")),
                                    220,
                                ),
                            }
                        )
                    cycles.append(
                        {
                            "event": str(r["ticker"] or ""),
                            "report_date": r["report_date"].isoformat()
                            if r.get("report_date")
                            else None,
                            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                            "analysis_excerpt": excerpt,
                            "decisions": slim_decisions,
                        }
                    )

                mrow = await conn.fetchrow("SELECT COUNT(*)::bigint AS c FROM trade_memories")
                if mrow:
                    memories_count = int(mrow["c"])

                tpnl = await conn.fetchrow(
                    """
                    SELECT COALESCE(SUM(realized_pnl_usd), 0)::double precision AS s
                    FROM paper_trades
                    WHERE action = 'SELL'
                      AND (created_at AT TIME ZONE 'UTC')::date
                          = (NOW() AT TIME ZONE 'UTC')::date
                    """
                )
                if tpnl:
                    today_realized = float(tpnl["s"] or 0.0)

                days = await conn.fetch(
                    """
                    SELECT day, COALESCE(SUM(realized), 0)::double precision AS realized
                    FROM (
                        SELECT (created_at AT TIME ZONE 'UTC')::date AS day,
                               realized_pnl_usd AS realized
                        FROM paper_trades
                        WHERE action = 'SELL' AND realized_pnl_usd IS NOT NULL
                    ) x
                    GROUP BY day
                    ORDER BY day DESC
                    LIMIT 14
                    """
                )
                for d in reversed(list(days)):
                    pnl_by_day.append(
                        {
                            "day": d["day"].isoformat() if d["day"] else None,
                            "realized": float(d["realized"] or 0.0),
                        }
                    )
        except Exception as exc:
            log.warning("public live snapshot DB error: %s", exc)

    pos_mtm = 0.0
    for p in positions:
        cv = p.get("current_value")
        if cv is not None:
            pos_mtm += float(cv)
        else:
            pos_mtm += float(p.get("shares") or 0.0) * float(p.get("avg_cost") or 0.0)

    ledger_nav = float(balance) + pos_mtm

    t212_account: dict[str, Any] | None = None
    if settings and getattr(settings, "paper_executes_on_t212", False) and t212 is not None:
        t212_account = await _fetch_t212_account_cached(t212, currency)

    nav_display = ledger_nav
    nav_display_source = "supabase_ledger"
    if t212_account and t212_account.get("ok") and t212_account.get("total_value") is not None:
        nav_display = float(t212_account["total_value"])
        nav_display_source = "t212_api"

    nav_delta = nav_display - starting_nav
    nav_change_pct = (nav_delta / starting_nav * 100.0) if starting_nav > 1e-9 else 0.0

    ledger_t212_total_drift: float | None = None
    if t212_account and t212_account.get("ok") and t212_account.get("total_value") is not None:
        ledger_t212_total_drift = round(
            float(ledger_nav) - float(t212_account["total_value"]),
            4,
        )

    display_currency = (
        str(t212_account.get("account_currency") or currency).upper()[:3]
        if (t212_account and t212_account.get("ok"))
        else currency
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ledger_updated_at": ledger_updated_at,
        "health": health,
        "market_session": market_session,
        "currency": display_currency,
        "starting_nav": starting_nav,
        "nav_estimate": round(ledger_nav, 2),
        "nav_display": round(nav_display, 2),
        "nav_display_source": nav_display_source,
        "nav_delta": round(nav_delta, 2),
        "nav_change_pct": round(nav_change_pct, 3),
        "ledger_t212_total_drift": ledger_t212_total_drift,
        "today_realized_pnl": round(today_realized, 2),
        "open_positions": positions,
        "position_count": len(positions),
        "memories_count": memories_count,
        "trades": trades,
        "cycles": cycles,
        "pnl_by_day": pnl_by_day,
        "paper_agent_enabled": bool(getattr(settings, "paper_agent_enabled", False)),
        "paper_execution_backend": (getattr(settings, "paper_execution_backend", None) or "supabase"),
        "t212_account": t212_account,
        "snapshot_cached_age_s": 0.0,
    }
