"""
Async HTTP client for the Trading 212 Public API.

Key design decisions
--------------------
* **Endpoint:** ``GET /api/v0/equity/positions`` (official docs, rate limit 1 req/s).
* **Auth:** HTTP Basic — ``Authorization: Basic Base64(key:secret)``.
* **Rate limit:** enforced client-side with ``asyncio.Lock`` + monotonic clock.
* **httpx timeout:** 30 s total, 5 s connect.
* **Retry:** automatic exponential back-off on 429 (max 3 attempts).

Usage::

    async with httpx.AsyncClient() as http:
        client = T212Client(http, settings)
        positions = await client.get_positions()
"""

from __future__ import annotations

import asyncio
import base64
import time
from typing import Any

import httpx

from app.core.config import Settings
from app.core.logging import get_logger
from app.services.t212.models import Position

log = get_logger("t212")

# Minimum gap between successive T212 API calls (seconds).
# Official limit is 1 req / 1 s — we use a small safety margin.
_MIN_INTERVAL: float = 1.05

# Max retry attempts on 429 Too Many Requests.
_MAX_RETRIES: int = 3

# Back-off multiplier for retries (seconds).
_BACKOFF_BASE: float = 2.0


class T212Client:
    """Async Trading 212 Public API client.

    Parameters
    ----------
    http_client:
        A shared ``httpx.AsyncClient`` (created in the FastAPI lifespan).
    settings:
        Application settings (T212 keys, base URL).
    """

    def __init__(self, http_client: httpx.AsyncClient, settings: Settings) -> None:
        self._http = http_client
        self._settings = settings
        self._base_url = settings.t212_api_url  # e.g. https://demo.trading212.com/api/v0
        self._auth_header = self._build_auth_header(
            settings.t212_demo_api_key,
            settings.t212_demo_api_secret,
        )
        # Rate-limit state
        self._lock = asyncio.Lock()
        self._last_request_at: float = 0.0

    # ── Public API ──────────────────────────────────────────────────

    async def get_positions(self, ticker: str | None = None) -> list[Position]:
        """Fetch all open positions (or filter by ``ticker``).

        Maps to ``GET /api/v0/equity/positions``.
        Rate limit: 1 request per second (enforced client-side).

        Parameters
        ----------
        ticker:
            Optional T212-format ticker, e.g. ``AAPL_US_EQ``.
        """
        params: dict[str, str] = {}
        if ticker:
            params["ticker"] = ticker

        resp = await self._request("GET", "/equity/positions", params=params)
        data = resp.json()

        if not isinstance(data, list):
            log.warning("Unexpected response type %s, expected list", type(data).__name__)
            return []

        return [Position.model_validate(item) for item in data]

    async def get_account_summary(self) -> dict[str, Any]:
        """``GET /equity/account/summary`` — cash, investments, totalValue (account currency)."""
        resp = await self._request("GET", "/equity/account/summary")
        return resp.json()

    async def place_market_order(
        self,
        ticker: str,
        quantity: float,
        *,
        extended_hours: bool = False,
    ) -> dict[str, Any]:
        """``POST /equity/orders/market`` — buy positive quantity, sell negative.

        Beta API is not idempotent; duplicate calls may duplicate orders.
        """
        q = round(float(quantity), 6)
        if q == 0:
            raise ValueError("T212 market order quantity must be non-zero")
        body: dict[str, Any] = {"ticker": ticker, "quantity": q}
        params: dict[str, str] | None = None
        if extended_hours:
            params = {"extendedHours": "true"}
        resp = await self._request(
            "POST",
            "/equity/orders/market",
            params=params,
            json_body=body,
        )
        return resp.json()

    async def get_pending_order(self, order_id: int) -> dict[str, Any] | None:
        """``GET /equity/orders/{id}``. Returns ``None`` if order left the pending queue (404)."""
        resp = await self._request(
            "GET",
            f"/equity/orders/{int(order_id)}",
            pass_through_status={404},
        )
        if resp.status_code == 404:
            return None
        return resp.json()

    async def get_all_pending_orders(self) -> list[dict[str, Any]]:
        """``GET /equity/orders`` — active (not fully filled / terminal) orders. Rate: 1 / 5s."""
        resp = await self._request("GET", "/equity/orders")
        data = resp.json()
        if isinstance(data, list):
            return data
        return []

    async def get_history_orders(
        self,
        *,
        limit: int = 50,
        cursor: int | None = None,
        ticker: str | None = None,
        next_page_path: str | None = None,
    ) -> dict[str, Any]:
        """``GET /equity/history/orders`` (paginated). Use ``next_page_path`` from prior response."""
        if next_page_path:
            path, params = self._parse_history_next_path(next_page_path)
            resp = await self._request("GET", path, params=params)
            return resp.json()
        lim = min(max(int(limit), 1), 50)
        params: dict[str, str] = {"limit": str(lim)}
        if cursor is not None:
            params["cursor"] = str(int(cursor))
        if ticker:
            params["ticker"] = ticker
        resp = await self._request("GET", "/equity/history/orders", params=params)
        return resp.json()

    @staticmethod
    def _parse_history_next_path(next_page_path: str) -> tuple[str, dict[str, str]]:
        """Turn ``/api/v0/equity/history/orders?limit=20&cursor=…`` into path under ``/api/v0`` + query params."""
        from urllib.parse import parse_qs, urlparse

        raw = (next_page_path or "").strip()
        if not raw:
            return "/equity/history/orders", {}
        if raw.startswith("http"):
            u = urlparse(raw)
            path_part, query = u.path, u.query
        else:
            parts = raw.split("?", 1)
            path_part = parts[0]
            query = parts[1] if len(parts) > 1 else ""
        if "/api/v0" in path_part:
            idx = path_part.index("/api/v0")
            rel = path_part[idx + len("/api/v0") :]
        else:
            rel = path_part
        path = rel if rel.startswith("/") else f"/{rel}"
        qsd = parse_qs(query)
        params = {k: v[0] for k, v in qsd.items() if v}
        return path, params

    # ── Internal helpers ────────────────────────────────────────────

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str] | None = None,
        json_body: Any | None = None,
        pass_through_status: set[int] | None = None,
    ) -> httpx.Response:
        """Rate-limited, authenticated request with retry on 429."""
        url = f"{self._base_url}{path}"
        headers = {"Authorization": self._auth_header}

        for attempt in range(1, _MAX_RETRIES + 1):
            await self._throttle()
            try:
                resp = await self._http.request(
                    method,
                    url,
                    headers=headers,
                    params=params,
                    json=json_body,
                )
            except httpx.TimeoutException:
                log.error("T212 request timeout: %s %s (attempt %d)", method, path, attempt)
                if attempt == _MAX_RETRIES:
                    raise
                await asyncio.sleep(_BACKOFF_BASE ** attempt)
                continue

            if resp.status_code == 429:
                wait = _BACKOFF_BASE ** attempt
                log.warning(
                    "T212 429 rate-limited — backing off %.1fs (attempt %d/%d)",
                    wait,
                    attempt,
                    _MAX_RETRIES,
                )
                await asyncio.sleep(wait)
                continue

            if resp.status_code >= 400:
                if pass_through_status and resp.status_code in pass_through_status:
                    return resp
                body_preview = resp.text[:300] if resp.text else "(empty)"
                log.error(
                    "T212 HTTP %d on %s %s: %s",
                    resp.status_code,
                    method,
                    path,
                    body_preview,
                )
                resp.raise_for_status()

            return resp

        # All retries exhausted (only reachable for 429s)
        msg = f"T212 rate-limit retries exhausted for {method} {path}"
        log.error(msg)
        raise httpx.HTTPStatusError(
            msg,
            request=httpx.Request(method, url),
            response=resp,  # type: ignore[possibly-undefined]
        )

    async def _throttle(self) -> None:
        """Ensure at least ``_MIN_INTERVAL`` seconds between API calls."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_request_at
            if elapsed < _MIN_INTERVAL:
                wait = _MIN_INTERVAL - elapsed
                log.debug("Throttling T212 request for %.2fs", wait)
                await asyncio.sleep(wait)
            self._last_request_at = time.monotonic()

    @staticmethod
    def _build_auth_header(api_key: str, api_secret: str) -> str:
        """Build ``Authorization: Basic <base64>`` header value."""
        if not api_key or not api_secret:
            log.warning("T212 API key or secret is empty — requests will fail with 401")
            return "Basic "
        credentials = f"{api_key}:{api_secret}"
        encoded = base64.b64encode(credentials.encode("utf-8")).decode("ascii")
        return f"Basic {encoded}"
