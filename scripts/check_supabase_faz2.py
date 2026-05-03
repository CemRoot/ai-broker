#!/usr/bin/env python3
"""Verify Supabase Postgres (Faz 2): connect, pgvector, trade_memories, RPC.

Uses ``SUPABASE_DB_URL`` from ``.env`` (same as the app). Does not print credentials.
Run: ``uv run python scripts/check_supabase_faz2.py``
"""

from __future__ import annotations

import asyncio
import sys
from urllib.parse import urlparse

import asyncpg


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

    raw = (Settings().supabase_db_url or "").strip()
    if not raw:
        print("FAIL: SUPABASE_DB_URL is empty (set in .env)")
        return 1

    dsn = _ensure_supabase_ssl(raw)
    print("DSN (masked):", _mask_dsn(dsn))

    try:
        conn = await asyncpg.connect(dsn=dsn, timeout=25, command_timeout=60)
    except Exception as exc:
        print(f"FAIL: connect — {type(exc).__name__}: {exc}")
        return 1

    print("OK: connected")

    try:
        rows = await conn.fetch(
            "SELECT extname, extversion FROM pg_extension "
            "WHERE extname IN ('vector', 'plpgsql') ORDER BY extname"
        )
        names = {r["extname"] for r in rows}
        print("Extensions:", {r["extname"]: r["extversion"] for r in rows})
        if "vector" not in names:
            print("FAIL: extension 'vector' missing — run sql/schemas/001_memory.sql")
            return 1
    except Exception as exc:
        print(f"FAIL: pg_extension — {type(exc).__name__}: {exc}")
        await conn.close()
        return 1

    try:
        exists = await conn.fetchval(
            "SELECT to_regclass('public.trade_memories') IS NOT NULL"
        )
        if not exists:
            print("FAIL: table public.trade_memories missing — run sql/schemas/001_memory.sql")
            await conn.close()
            return 1
        print("OK: table trade_memories exists")
    except Exception as exc:
        print(f"FAIL: trade_memories — {type(exc).__name__}: {exc}")
        await conn.close()
        return 1

    try:
        cols = await conn.fetch(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'trade_memories'
            ORDER BY ordinal_position
            """
        )
        have = {c["column_name"] for c in cols}
        need = {
            "id",
            "ticker",
            "memory_type",
            "context",
            "outcome",
            "pnl_percent",
            "embedding",
            "created_at",
        }
        missing = need - have
        if missing:
            print("FAIL: trade_memories missing columns:", sorted(missing))
            await conn.close()
            return 1
        print("OK: trade_memories columns (subset):", sorted(need))
    except Exception as exc:
        print(f"FAIL: columns — {type(exc).__name__}: {exc}")
        await conn.close()
        return 1

    try:
        fn = await conn.fetchval(
            "SELECT proname FROM pg_proc JOIN pg_namespace n ON n.oid = pg_proc.pronamespace "
            "WHERE n.nspname = 'public' AND proname = 'match_trade_memories' LIMIT 1"
        )
        if not fn:
            print(
                "FAIL: function public.match_trade_memories missing — run sql/schemas/001_memory.sql"
            )
            await conn.close()
            return 1
        print("OK: function match_trade_memories exists")
    except Exception as exc:
        print(f"FAIL: match_trade_memories — {type(exc).__name__}: {exc}")
        await conn.close()
        return 1

    try:
        await conn.execute("SELECT '[0.1,0.2,0.3]'::vector(3)")
        print("OK: vector literal cast")
    except Exception as exc:
        print(f"WARN: vector cast test — {type(exc).__name__}: {exc}")

    await conn.close()
    print("DONE: Faz 2 DB checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
