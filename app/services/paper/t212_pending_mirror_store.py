"""Supabase queue: T212 order ids until filled (``paper_t212_pending_mirror``)."""

from __future__ import annotations

import json
import logging
from typing import Any

from app.memory.database import SupabaseDatabase

log = logging.getLogger(__name__)


async def enqueue_t212_pending_mirror(
    db: SupabaseDatabase,
    *,
    t212_order_id: int,
    yf_ticker: str,
    action: str,
    meta: dict[str, Any],
) -> None:
    pool = db.get_pool()
    if not pool:
        log.warning("enqueue_t212_pending_mirror: no DB pool")
        return
    act = action.upper().strip()
    if act not in ("BUY", "SELL"):
        raise ValueError("action must be BUY or SELL")
    payload = json.dumps(meta, default=str)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO paper_t212_pending_mirror (t212_order_id, yf_ticker, action, meta)
                VALUES ($1, $2, $3, $4::jsonb)
                ON CONFLICT (t212_order_id) DO NOTHING
                """,
                int(t212_order_id),
                yf_ticker.upper().strip()[:16],
                act,
                payload,
            )
    except Exception as exc:
        log.warning(
            "enqueue_t212_pending_mirror failed (apply sql/schemas/005_t212_pending_mirror.sql?): %s",
            exc,
        )


async def list_t212_pending_mirror(db: SupabaseDatabase) -> list[dict[str, Any]]:
    pool = db.get_pool()
    if not pool:
        return []
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT t212_order_id, yf_ticker, action, meta, created_at, last_poll_at
            FROM paper_t212_pending_mirror
            ORDER BY created_at ASC
            """
        )
    out: list[dict[str, Any]] = []
    for r in rows:
        meta = r["meta"]
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        out.append(
            {
                "t212_order_id": int(r["t212_order_id"]),
                "yf_ticker": str(r["yf_ticker"]),
                "action": str(r["action"]),
                "meta": meta if isinstance(meta, dict) else {},
            }
        )
    return out


async def delete_t212_pending_mirror(db: SupabaseDatabase, t212_order_id: int) -> None:
    pool = db.get_pool()
    if not pool:
        return
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM paper_t212_pending_mirror WHERE t212_order_id = $1",
            int(t212_order_id),
        )


async def touch_t212_pending_poll(db: SupabaseDatabase, t212_order_id: int) -> None:
    pool = db.get_pool()
    if not pool:
        return
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE paper_t212_pending_mirror SET last_poll_at = NOW() WHERE t212_order_id = $1",
            int(t212_order_id),
        )


async def pending_mirror_row_exists(db: SupabaseDatabase, t212_order_id: int) -> bool:
    pool = db.get_pool()
    if not pool:
        return False
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT 1 FROM paper_t212_pending_mirror WHERE t212_order_id = $1 LIMIT 1",
            int(t212_order_id),
        )
    return row is not None
