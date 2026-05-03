#!/usr/bin/env python3
"""Apply repo SQL schemas to Supabase Postgres in canonical order.

Reads ``SUPABASE_DB_URL`` from ``.env`` (via ``app.core.config.Settings``).
Does not print credentials.

Usage:
  uv run python scripts/apply_sql_schemas.py
  uv run python scripts/apply_sql_schemas.py --no-paper-drop   # skip DROP in 002 (keep paper_* data)

Run: from repository root.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import asyncpg

ROOT = Path(__file__).resolve().parent.parent

SCHEMA_FILES = [
    "sql/schemas/001_memory.sql",
    "sql/schemas/trump_posts.sql",
    "sql/schemas/002_paper_trading.sql",
    "sql/schemas/003_paper_agent.sql",
    "sql/schemas/004_t212_paper_execution.sql",
    "sql/schemas/005_t212_pending_mirror.sql",
]


def _ensure_supabase_ssl(url: str) -> str:
    lower = url.lower()
    if "supabase.com" not in lower:
        return url
    if "sslmode=" in lower or "ssl=" in lower:
        return url
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}sslmode=require"


def _mask_dsn(dsn: str) -> str:
    from urllib.parse import urlparse

    try:
        p = urlparse(dsn)
        host = p.hostname or "?"
        port = f":{p.port}" if p.port else ""
        db = (p.path or "/").lstrip("/") or "postgres"
        return f"{p.scheme}://***@{host}{port}/{db}"
    except Exception:
        return "(unparseable DSN)"


def split_sql_statements(sql: str) -> list[str]:
    """Split SQL on top-level ``;`` outside quotes, dollar-quoted blocks, and ``--`` comments."""
    buf: list[str] = []
    out: list[str] = []
    i = 0
    n = len(sql)

    def flush() -> None:
        s = "".join(buf).strip()
        buf.clear()
        if s:
            out.append(s)

    while i < n:
        if i + 1 < n and sql[i] == "-" and sql[i + 1] == "-":
            while i < n and sql[i] != "\n":
                buf.append(sql[i])
                i += 1
            continue
        if sql[i] == "'":
            buf.append(sql[i])
            i += 1
            while i < n:
                if sql[i] == "'" and i + 1 < n and sql[i + 1] == "'":
                    buf.append("''")
                    i += 2
                    continue
                if sql[i] == "'":
                    buf.append(sql[i])
                    i += 1
                    break
                buf.append(sql[i])
                i += 1
            continue
        if sql[i] == "$":
            if i + 1 < n and sql[i + 1] == "$":
                delim = "$$"
                j = i + 2
            else:
                k = i + 1
                while k < n and sql[k] != "$":
                    k += 1
                if k >= n:
                    raise ValueError("unclosed dollar-quote tag")
                tag = sql[i + 1 : k]
                delim = "$" + tag + "$"
                j = k + 1
            end = sql.find(delim, j)
            if end == -1:
                raise ValueError("unclosed dollar-quoted string")
            buf.append(sql[i : end + len(delim)])
            i = end + len(delim)
            continue
        if sql[i] == ";":
            flush()
            i += 1
            continue
        buf.append(sql[i])
        i += 1
    flush()
    return out


def _maybe_strip_paper_drops(text: str) -> tuple[str, int]:
    lines = text.splitlines()
    kept: list[str] = []
    skipped = 0
    for line in lines:
        if line.strip().upper().startswith("DROP TABLE"):
            skipped += 1
            continue
        kept.append(line)
    return "\n".join(kept), skipped


async def main() -> int:
    parser = argparse.ArgumentParser(description="Apply sql/schemas/*.sql to Supabase.")
    parser.add_argument(
        "--no-paper-drop",
        action="store_true",
        help="Omit DROP TABLE lines from 002_paper_trading.sql (preserve paper_* rows).",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))
    from app.core.config import Settings

    raw = (Settings().supabase_db_url or "").strip()
    if not raw:
        print("FAIL: SUPABASE_DB_URL is empty (set in .env)")
        return 1

    dsn = _ensure_supabase_ssl(raw)
    print("DSN (masked):", _mask_dsn(dsn))

    try:
        conn = await asyncpg.connect(dsn=dsn, timeout=60, command_timeout=120)
    except Exception as exc:
        print(f"FAIL: connect — {type(exc).__name__}: {exc}")
        return 1

    try:
        for rel in SCHEMA_FILES:
            path = ROOT / rel
            if not path.is_file():
                print(f"FAIL: missing file {rel}")
                return 1
            text = path.read_text(encoding="utf-8")
            if args.no_paper_drop and rel.endswith("002_paper_trading.sql"):
                text, nskip = _maybe_strip_paper_drops(text)
                if nskip:
                    print(f"INFO: {rel} — skipped {nskip} DROP TABLE line(s) (--no-paper-drop)")
            try:
                stmts = split_sql_statements(text)
            except ValueError as exc:
                print(f"FAIL: parse {rel} — {exc}")
                return 1
            for k, stmt in enumerate(stmts, 1):
                preview = stmt.replace("\n", " ")[:72]
                try:
                    await conn.execute(stmt)
                except Exception as exc:
                    print(f"FAIL: {rel} statement {k}/{len(stmts)}: {type(exc).__name__}: {exc}")
                    print(f"  preview: {preview!r}…")
                    return 1
                print(f"OK: {rel} [{k}/{len(stmts)}] {preview}…")
        print("Done: all schema files applied.")
        return 0
    finally:
        await conn.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
