"""
Async HTTP client for the Trading 212 Public API.

Key design decisions
--------------------
* **Endpoints covered (official docs, https://t212public-api-docs.redoc.ly):**
    * ``GET   /api/v0/equity/metadata/instruments``         — ~1 req / 50 s (cached in-process)
    * ``GET   /api/v0/equity/positions``                    — 1 req / 5 s class
    * ``GET   /api/v0/equity/account/summary``              — 1 req / 30 s class
    * ``POST  /api/v0/equity/orders/market``                — 1 req / 1 s
    * ``POST  /api/v0/equity/orders/limit``                 — 1 req / 2 s
    * ``POST  /api/v0/equity/orders/stop``                  — 1 req / 2 s
    * ``POST  /api/v0/equity/orders/stop_limit``            — 1 req / 2 s
    * ``GET   /api/v0/equity/orders``                       — 1 req / 5 s
    * ``GET   /api/v0/equity/orders/{id}``                  — 1 req / 1 s
    * ``DELETE /api/v0/equity/orders/{id}``                 — 1 req / 2 s
    * ``GET   /api/v0/equity/history/orders``               — 1 req / 60 s (paged)
* **Auth:** HTTP Basic — ``Authorization: Basic Base64(key:secret)``.
* **Rate limit:** enforced client-side with ``asyncio.Lock`` + monotonic clock.
  We use a *single global* throttle of ``_MIN_INTERVAL`` (the slowest of the
  hot-path endpoints we use) so that any combination of order/queries respects
  the strictest documented bucket.
* **httpx timeout:** 30 s total, 5 s connect.
* **Retry:** automatic exponential back-off on 429 (max 3 attempts).

Usage::

    async with httpx.AsyncClient() as http:
        client = T212Client(http, settings)
        positions = await client.get_positions()
        order = await client.place_limit_order("AAPL_US_EQ", 1.5, limit_price=150.0)
"""

from __future__ import annotations

import asyncio
import base64
import time
from typing import Any, Literal

import httpx

from app.core.config import Settings
from app.core.logging import get_logger
from app.services.t212.models import Position
from app.services.t212.ticker_map import t212_to_yfinance, yfinance_to_t212

log = get_logger("t212")

# Minimum gap between successive T212 API calls (seconds).
# Sized to the slowest hot-path endpoint we use (limit/stop/stop_limit/cancel = 1 req / 2 s,
# per Context7 docs). Market orders are themselves 1 req / 1 s but interleaving with order
# variants would otherwise risk 429s, so we keep one safe global bucket + small margin.
_MIN_INTERVAL: float = 2.05

# Max retry attempts on 429 Too Many Requests.
_MAX_RETRIES: int = 3

# Back-off multiplier for retries (seconds).
_BACKOFF_BASE: float = 2.0

# Official instruments list is heavily rate-limited (~1 call / 50s). Cache + cooldown.
_INSTRUMENTS_API_COOLDOWN_SEC: float = 50.0
_INSTRUMENTS_CACHE_TTL_SEC: float = 3600.0
_TRADEABLE_EQUITY_TYPES: frozenset[str] = frozenset({"STOCK", "ETF"})

# Allowed values for the `timeValidity` request body field.
TimeValidity = Literal["DAY", "GOOD_TILL_CANCEL"]


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
        self._instruments_lock = asyncio.Lock()
        self._equity_instruments: list[dict[str, Any]] | None = None
        self._equity_instruments_loaded_mono: float = 0.0
        self._last_instruments_request_mono: float = -10_000.0
        self._tradeable_equity_tickers: frozenset[str] = frozenset()

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
        q = self._normalize_quantity(quantity, kind="market")
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

    async def place_limit_order(
        self,
        ticker: str,
        quantity: float,
        *,
        limit_price: float,
        time_validity: TimeValidity = "DAY",
        extended_hours: bool = False,
    ) -> dict[str, Any]:
        """``POST /equity/orders/limit`` — execute only at ``limit_price`` or better.

        * ``quantity`` positive → BUY (fills at ``<= limit_price``).
        * ``quantity`` negative → SELL (fills at ``>= limit_price``).
        * ``time_validity`` is required by the API: ``DAY`` or ``GOOD_TILL_CANCEL``.
        * Endpoint rate limit: **1 req / 2 s** (Context7 docs).
        * Endpoint is **not idempotent** in the public beta; do not retry blindly.
        """
        q = self._normalize_quantity(quantity, kind="limit")
        lp = self._normalize_price(limit_price, name="limit_price")
        body: dict[str, Any] = {
            "ticker": ticker,
            "quantity": q,
            "limitPrice": lp,
            "timeValidity": self._normalize_validity(time_validity),
        }
        params: dict[str, str] | None = {"extendedHours": "true"} if extended_hours else None
        resp = await self._request(
            "POST",
            "/equity/orders/limit",
            params=params,
            json_body=body,
        )
        return resp.json()

    async def place_stop_order(
        self,
        ticker: str,
        quantity: float,
        *,
        stop_price: float,
        time_validity: TimeValidity = "DAY",
        extended_hours: bool = False,
    ) -> dict[str, Any]:
        """``POST /equity/orders/stop`` — fires a Market order once LTP touches ``stop_price``.

        * Positive ``quantity`` → buy stop (entry).
        * Negative ``quantity`` → sell stop (commonly a stop-loss).
        * Endpoint rate limit: **1 req / 2 s**, **not idempotent**.
        """
        q = self._normalize_quantity(quantity, kind="stop")
        sp = self._normalize_price(stop_price, name="stop_price")
        body: dict[str, Any] = {
            "ticker": ticker,
            "quantity": q,
            "stopPrice": sp,
            "timeValidity": self._normalize_validity(time_validity),
        }
        params: dict[str, str] | None = {"extendedHours": "true"} if extended_hours else None
        resp = await self._request(
            "POST",
            "/equity/orders/stop",
            params=params,
            json_body=body,
        )
        return resp.json()

    async def place_stop_limit_order(
        self,
        ticker: str,
        quantity: float,
        *,
        stop_price: float,
        limit_price: float,
        time_validity: TimeValidity = "DAY",
        extended_hours: bool = False,
    ) -> dict[str, Any]:
        """``POST /equity/orders/stop_limit`` — once LTP hits ``stop_price`` a Limit order at ``limit_price`` is placed.

        * Positive ``quantity`` → buy; negative → sell.
        * Use to bound slippage on a stop trigger.
        * Endpoint rate limit: **1 req / 2 s**, **not idempotent**.
        """
        q = self._normalize_quantity(quantity, kind="stop_limit")
        sp = self._normalize_price(stop_price, name="stop_price")
        lp = self._normalize_price(limit_price, name="limit_price")
        body: dict[str, Any] = {
            "ticker": ticker,
            "quantity": q,
            "stopPrice": sp,
            "limitPrice": lp,
            "timeValidity": self._normalize_validity(time_validity),
        }
        params: dict[str, str] | None = {"extendedHours": "true"} if extended_hours else None
        resp = await self._request(
            "POST",
            "/equity/orders/stop_limit",
            params=params,
            json_body=body,
        )
        return resp.json()

    async def cancel_order(self, order_id: int) -> bool:
        """``DELETE /equity/orders/{id}`` — request cancellation of a pending order.

        Returns ``True`` on accepted cancel (HTTP 2xx), ``False`` if the order has
        already left the pending queue (HTTP 404). Other errors propagate.
        Cancellation is *requested*, not guaranteed (the order may already be filling).
        """
        resp = await self._request(
            "DELETE",
            f"/equity/orders/{int(order_id)}",
            pass_through_status={404},
        )
        if resp.status_code == 404:
            log.info("T212 cancel: order %s already terminal/unknown (HTTP 404)", order_id)
            return False
        return True

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

    async def fetch_equity_instruments_list(self, *, force: bool = False) -> list[dict[str, Any]]:
        """``GET /equity/metadata/instruments`` — account invest universe (STOCK/ETF).

        Cached for :data:`_INSTRUMENTS_CACHE_TTL_SEC`; consecutive upstream calls are
        spaced by :data:`_INSTRUMENTS_API_COOLDOWN_SEC` per official rate guidance.
        """
        async with self._instruments_lock:
            now = time.monotonic()
            if (
                not force
                and self._equity_instruments is not None
                and (now - self._equity_instruments_loaded_mono) < _INSTRUMENTS_CACHE_TTL_SEC
            ):
                return list(self._equity_instruments)
            if (
                self._equity_instruments is not None
                and (now - self._last_instruments_request_mono) < _INSTRUMENTS_API_COOLDOWN_SEC
            ):
                return list(self._equity_instruments)

            try:
                resp = await self._request("GET", "/equity/metadata/instruments")
                data = resp.json()
            except Exception as exc:
                log.error("T212 instruments fetch failed: %s: %s", type(exc).__name__, exc)
                if self._equity_instruments is not None:
                    return list(self._equity_instruments)
                return []

            if not isinstance(data, list):
                log.warning("T212 instruments: expected JSON list, got %s", type(data).__name__)
                data = []

            tradeable: set[str] = set()
            for rec in data:
                if not isinstance(rec, dict):
                    continue
                ttype = str(rec.get("type") or "").upper().strip()
                if ttype not in _TRADEABLE_EQUITY_TYPES:
                    continue
                tick = str(rec.get("ticker") or "").strip().upper()
                if tick:
                    tradeable.add(tick)

            self._equity_instruments = data
            self._tradeable_equity_tickers = frozenset(tradeable)
            self._equity_instruments_loaded_mono = time.monotonic()
            self._last_instruments_request_mono = self._equity_instruments_loaded_mono
            log.info(
                "T212 equity instruments cache refreshed (%d tradeable STOCK/ETF tickers)",
                len(self._tradeable_equity_tickers),
            )
            return list(self._equity_instruments)

    def tradeable_equity_instrument_count(self) -> int:
        """Count of cached STOCK/ETF tickers (0 until first successful fetch)."""
        return len(self._tradeable_equity_tickers)

    async def is_us_equity_instrument_tradeable(self, yf_or_t212_ticker: str) -> tuple[bool, str]:
        """Whether *yf_or_t212_ticker* resolves to a tradeable equity STOCK/ETF on this account."""
        raw = (yf_or_t212_ticker or "").strip().upper()
        if not raw:
            return False, "empty ticker"
        base = t212_to_yfinance(raw) if "_" in raw and raw.endswith("_EQ") else raw
        base = base.strip().upper()
        if not base:
            return False, "empty base symbol"
        candidate = yfinance_to_t212(base).strip().upper()
        await self.fetch_equity_instruments_list()
        if candidate in self._tradeable_equity_tickers:
            return True, candidate
        return (
            False,
            f"{candidate} not in this account's T212 equity STOCK/ETF list (CFD-only or unsupported)",
        )

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

    # ── Normalization helpers (defensive client-side validation) ────

    @staticmethod
    def _normalize_quantity(quantity: float, *, kind: str) -> float:
        """Return ``quantity`` rounded to 6 dp; raise if zero or non-finite.

        ``kind`` is purely for the error message (which order endpoint failed).
        """
        try:
            q = round(float(quantity), 6)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"T212 {kind} order: quantity must be a number, got {quantity!r}") from exc
        if q == 0:
            raise ValueError(f"T212 {kind} order: quantity must be non-zero")
        return q

    @staticmethod
    def _normalize_price(price: float, *, name: str) -> float:
        """Return ``price`` rounded to 4 dp; raise if not strictly positive."""
        try:
            p = round(float(price), 4)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"T212 order: {name} must be a number, got {price!r}") from exc
        if p <= 0:
            raise ValueError(f"T212 order: {name} must be > 0 (got {p})")
        return p

    @staticmethod
    def _normalize_validity(value: str) -> str:
        """Validate ``timeValidity`` against the API enum."""
        v = (value or "").upper().strip()
        if v not in ("DAY", "GOOD_TILL_CANCEL"):
            raise ValueError(
                f"T212 order: timeValidity must be 'DAY' or 'GOOD_TILL_CANCEL' (got {value!r})"
            )
        return v

    @staticmethod
    def _build_auth_header(api_key: str, api_secret: str) -> str:
        """Build ``Authorization: Basic <base64>`` header value."""
        if not api_key or not api_secret:
            log.warning("T212 API key or secret is empty — requests will fail with 401")
            return "Basic "
        credentials = f"{api_key}:{api_secret}"
        encoded = base64.b64encode(credentials.encode("utf-8")).decode("ascii")
        return f"Basic {encoded}"
