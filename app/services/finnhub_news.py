"""Finnhub company-news fetch (async httpx)."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import httpx

from app.core.logging import get_logger

log = get_logger("finnhub_news")

FINNHUB_BASE = "https://finnhub.io/api/v1"


async def fetch_company_news(
    client: httpx.AsyncClient,
    *,
    api_key: str,
    symbol: str,
    days: int = 7,
    max_articles: int = 40,
) -> list[dict[str, Any]]:
    """Return articles as ``title`` / ``description`` / ``url`` dicts for ``news_pipeline``."""
    if not api_key:
        return []

    sym = symbol.upper().split(".")[0]
    end = date.today()
    start = end - timedelta(days=max(1, days))

    url = f"{FINNHUB_BASE}/company-news"
    params = {
        "symbol": sym,
        "from": start.isoformat(),
        "to": end.isoformat(),
        "token": api_key,
    }

    try:
        resp = await client.get(url, params=params, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        log.error("Finnhub company-news failed: %s", exc)
        raise

    if not isinstance(data, list):
        return []

    out: list[dict[str, Any]] = []
    for row in data[:max_articles]:
        if not isinstance(row, dict):
            continue
        headline = (row.get("headline") or row.get("title") or "").strip()
        if not headline:
            continue
        summary = (row.get("summary") or row.get("description") or "").strip()
        url_s = row.get("url") or row.get("source")
        out.append(
            {
                "title": headline,
                "description": summary,
                "url": url_s,
            }
        )

    return out
