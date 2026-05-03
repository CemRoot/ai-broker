#!/usr/bin/env python3
"""Print paper trading state from Supabase (CLI). Uses SUPABASE_DB_URL from .env — does not echo secrets.

Run::

    PYTHONPATH=. uv run python scripts/inspect_paper_supabase.py
    PYTHONPATH=. uv run python scripts/inspect_paper_supabase.py --trades 25
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _mask_dsn(dsn: str) -> str:
    try:
        p = urlparse(dsn)
        host = p.hostname or "?"
        port = f":{p.port}" if p.port else ""
        db = (p.path or "/").lstrip("/") or "postgres"
        return f"{p.scheme}://***@{host}{port}/{db}"
    except Exception:
        return "(unparseable DSN)"


def _ensure_supabase_ssl(url: str) -> str:
    lower = url.lower()
    if "supabase.com" not in lower:
        return url
    if "sslmode=" in lower or "ssl=" in lower:
        return url
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}sslmode=require"


async def main() -> int:
    from app.core.config import Settings

    p = argparse.ArgumentParser()
    p.add_argument("--trades", type=int, default=20, help="max paper_trades rows to show")
    args = p.parse_args()

    raw = (Settings().supabase_db_url or "").strip()
    if not raw:
        print("FAIL: SUPABASE_DB_URL empty")
        return 1

    dsn = _ensure_supabase_ssl(raw)
    print("DSN (masked):", _mask_dsn(dsn))

    cfg = Settings()
    if cfg.paper_executes_on_t212:
        sync_on = getattr(cfg, "paper_t212_sync_supabase_ledger", True)
        print(
            "\n*** PAPER_EXECUTION_BACKEND=t212 ***\n"
            f"  paper_account / paper_portfolio = T212’nin **gölge kopyası** (PAPER_T212_SYNC_SUPABASE_LEDGER={sync_on}).\n"
            "  Nakit: API `cash.availableToTrade` (uygulamadaki “available”; pending rezerv ayrı).\n"
            "  Ayrıca: paper_trades (execution_broker=t212), paper_t212_pending_mirror.\n"
        )

    import asyncpg

    try:
        conn = await asyncpg.connect(dsn=dsn, timeout=25, command_timeout=60)
    except Exception as exc:
        print(f"FAIL: connect — {exc}")
        return 1

    try:
        row = await conn.fetchrow("SELECT id, balance, updated_at FROM paper_account WHERE id = 1")
        if row:
            print("\n=== paper_account (id=1) ===")
            print(f"  balance_usd: {float(row['balance']):,.2f}  updated_at: {row['updated_at']}")
        else:
            print("\n=== paper_account === (no row id=1 — run sql/schemas/002_paper_trading.sql)")

        pos = await conn.fetch(
            "SELECT ticker, shares, avg_cost, status, updated_at FROM paper_portfolio "
            "WHERE status = 'OPEN' AND shares > 0 ORDER BY ticker"
        )
        print(f"\n=== paper_portfolio OPEN ({len(pos)} rows) ===")
        for r in pos:
            print(
                f"  {r['ticker']}: shares={float(r['shares']):.4f} avg={float(r['avg_cost']):.2f} "
                f"updated={r['updated_at']}"
            )
        if not pos:
            print("  (none)")

        lim = max(1, min(int(args.trades), 100))
        tr = await conn.fetch(
            """
            SELECT ticker, action, shares, price, total_value, reasoning,
                   cycle_event, emergency, created_at
            FROM paper_trades
            ORDER BY created_at DESC
            LIMIT $1
            """,
            lim,
        )
        print(f"\n=== paper_trades (last {len(tr)} rows) ===")
        for r in tr:
            reason = (r["reasoning"] or "")[:80]
            if len((r["reasoning"] or "")) > 80:
                reason += "…"
            print(
                f"  {r['created_at']} | {r['action']:4} {r['ticker']:6} "
                f"sh={float(r['shares']):.4f} @ {float(r['price']):.2f} "
                f"evt={r['cycle_event']} emg={r['emergency']}"
            )
            if reason:
                print(f"      {reason}")
        if not tr:
            print("  (none)")

    finally:
        await conn.close()

    print("\nNOTE: Bu tablolar sanal paper portföy (Supabase). Trading 212 hesabınızla karıştırmayın.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
