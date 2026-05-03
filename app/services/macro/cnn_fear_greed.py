"""CNN Fear & Greed Index client.

The public CNN dataviz endpoint returns the daily composite score together
with seven sub-component scores. We use it for **three** signals at once:

1. ``score`` (0–100) + ``rating`` (extreme fear … extreme greed)
2. ``put_call_options`` sub-component (the canonical doc lists this as a
   separate data source; CNN's index already aggregates the CBOE feed)
3. ``market_volatility_vix`` sub-component (sanity check vs our own
   ``yfinance ^VIX`` fetch in ``ToolExecutor._get_macro_context``)

Endpoint
--------
``GET https://production.dataviz.cnn.io/index/fearandgreed/graphdata/{YYYY-MM-DD}``

Headers ``Origin: https://www.cnn.com`` + a real browser ``User-Agent`` are
required, otherwise Cloudflare returns 403. No authentication.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.logging import get_logger

log = get_logger("macro.cnn_fear_greed")

CNN_FG_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"

# Cloudflare in front of CNN's API rejects default Python clients.
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.cnn.com",
    "Referer": "https://www.cnn.com/markets/fear-and-greed",
}


def _bucket(score: float | None) -> str:
    """CNN's published bucket boundaries."""
    if score is None:
        return "n/a"
    s = float(score)
    if s < 25:
        return "extreme fear"
    if s < 45:
        return "fear"
    if s < 55:
        return "neutral"
    if s < 75:
        return "greed"
    return "extreme greed"


@dataclass(frozen=True)
class CNNFearGreedSnapshot:
    score: float | None
    rating: str | None
    timestamp: str | None
    previous_close: float | None
    previous_1_week: float | None
    previous_1_month: float | None
    put_call_score: float | None
    put_call_rating: str | None
    vix_score: float | None
    vix_rating: str | None
    safe_haven_score: float | None
    safe_haven_rating: str | None
    junk_bond_score: float | None
    junk_bond_rating: str | None

    def to_lines(self) -> list[str]:
        """Compact log/LLM-friendly multiline rendering."""
        lines: list[str] = []
        if self.score is not None:
            d_w = (
                f" ({(self.score - self.previous_1_week):+.1f} vs 1w)"
                if self.previous_1_week is not None
                else ""
            )
            d_m = (
                f" ({(self.score - self.previous_1_month):+.1f} vs 1m)"
                if self.previous_1_month is not None
                else ""
            )
            lines.append(
                f"CNN Fear & Greed: {self.score:.1f} ({self.rating or _bucket(self.score)}){d_w}{d_m}"
            )
        if self.put_call_score is not None:
            lines.append(
                f"  ↳ Put/Call options sub-score: {self.put_call_score:.1f} ({self.put_call_rating or _bucket(self.put_call_score)})"
            )
        if self.vix_score is not None:
            lines.append(
                f"  ↳ Volatility (VIX) sub-score: {self.vix_score:.1f} ({self.vix_rating or _bucket(self.vix_score)})"
            )
        if self.safe_haven_score is not None:
            lines.append(
                f"  ↳ Safe-haven demand sub-score: {self.safe_haven_score:.1f} ({self.safe_haven_rating or _bucket(self.safe_haven_score)})"
            )
        if self.junk_bond_score is not None:
            lines.append(
                f"  ↳ Junk-bond demand sub-score: {self.junk_bond_score:.1f} ({self.junk_bond_rating or _bucket(self.junk_bond_score)})"
            )
        return lines


def _safe_get(d: Any, *path: str) -> Any:
    cur = d
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


class CNNFearGreedClient:
    """Single-method async client; share an httpx.AsyncClient with the rest of the app."""

    def __init__(self, http_client: httpx.AsyncClient) -> None:
        self._http = http_client

    async def fetch(self, *, on_date: dt.date | None = None) -> CNNFearGreedSnapshot | None:
        date_str = (on_date or dt.date.today()).isoformat()
        url = f"{CNN_FG_URL}/{date_str}"
        try:
            r = await self._http.get(url, headers=_BROWSER_HEADERS, timeout=12.0)
        except Exception as exc:
            log.warning("CNN F&G request failed: %s", exc)
            return None
        if r.status_code != 200:
            log.warning("CNN F&G HTTP %s on %s", r.status_code, date_str)
            return None
        try:
            d = r.json()
        except Exception as exc:
            log.warning("CNN F&G JSON parse error: %s", exc)
            return None

        return CNNFearGreedSnapshot(
            score=_safe_get(d, "fear_and_greed", "score"),
            rating=_safe_get(d, "fear_and_greed", "rating"),
            timestamp=_safe_get(d, "fear_and_greed", "timestamp"),
            previous_close=_safe_get(d, "fear_and_greed", "previous_close"),
            previous_1_week=_safe_get(d, "fear_and_greed", "previous_1_week"),
            previous_1_month=_safe_get(d, "fear_and_greed", "previous_1_month"),
            put_call_score=_safe_get(d, "put_call_options", "score"),
            put_call_rating=_safe_get(d, "put_call_options", "rating"),
            vix_score=_safe_get(d, "market_volatility_vix", "score"),
            vix_rating=_safe_get(d, "market_volatility_vix", "rating"),
            safe_haven_score=_safe_get(d, "safe_haven_demand", "score"),
            safe_haven_rating=_safe_get(d, "safe_haven_demand", "rating"),
            junk_bond_score=_safe_get(d, "junk_bond_demand", "score"),
            junk_bond_rating=_safe_get(d, "junk_bond_demand", "rating"),
        )
