"""
Unit tests for the new T212Client order methods (LIMIT / STOP / STOP_LIMIT / cancel).

These tests verify:
* Request bodies match the official T212 Public API shapes (per Context7 docs).
* Quantity sign conventions (positive=BUY, negative=SELL) are preserved.
* Client-side validation rejects malformed inputs *before* hitting the network.
* The DELETE cancel endpoint maps 404 to ``False`` (already-terminal order).

The HTTP layer is faked with a ``RecordingClient`` that captures the outgoing
``method``, ``url``, ``params``, and ``json`` payloads — no real network calls.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from app.core.config import Settings
from app.services.t212.client import T212Client


# ── Helpers ─────────────────────────────────────────────────────────


class _FakeResponse:
    """Minimal ``httpx.Response`` stand-in covering what ``T212Client._request`` reads."""

    def __init__(self, status_code: int = 200, body: Any | None = None):
        self.status_code = status_code
        if body is None:
            self._body = {}
        else:
            self._body = body
        self.text = "[]" if isinstance(body, list) else "{}"

    def json(self) -> Any:
        return self._body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=httpx.Request("GET", "http://t/"),
                response=self,  # type: ignore[arg-type]
            )


class RecordingClient:
    """Captures every outgoing request and returns a canned response."""

    def __init__(self, response: _FakeResponse | None = None):
        self.calls: list[dict[str, Any]] = []
        self._response = response or _FakeResponse(200, {"id": 12345, "status": "LOCAL"})

    async def request(self, method: str, url: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append(
            {
                "method": method,
                "url": url,
                "params": kwargs.get("params"),
                "json": kwargs.get("json"),
                "headers": kwargs.get("headers"),
            }
        )
        return self._response


def _make_client(response: _FakeResponse | None = None) -> tuple[T212Client, RecordingClient]:
    fake_http = RecordingClient(response)
    settings = Settings(
        t212_demo_api_key="key",
        t212_demo_api_secret="secret",
        t212_base_url="https://demo.trading212.com",
    )
    client = T212Client(fake_http, settings)  # type: ignore[arg-type]
    # Bypass throttle for tests so they run fast.
    client._last_request_at = 0.0
    return client, fake_http


# ── place_limit_order ───────────────────────────────────────────────


class TestPlaceLimitOrder:
    @pytest.mark.asyncio
    async def test_buy_limit_default_validity(self):
        client, http = _make_client()
        await client.place_limit_order("AAPL_US_EQ", 1.5, limit_price=150.0)
        call = http.calls[-1]
        assert call["method"] == "POST"
        assert call["url"].endswith("/api/v0/equity/orders/limit")
        assert call["json"] == {
            "ticker": "AAPL_US_EQ",
            "quantity": 1.5,
            "limitPrice": 150.0,
            "timeValidity": "DAY",
        }
        assert call["params"] is None

    @pytest.mark.asyncio
    async def test_sell_limit_with_extended_hours_and_gtc(self):
        client, http = _make_client()
        await client.place_limit_order(
            "NVDA_US_EQ",
            -2.0,
            limit_price=900.5,
            time_validity="GOOD_TILL_CANCEL",
            extended_hours=True,
        )
        call = http.calls[-1]
        assert call["json"]["quantity"] == -2.0  # sell preserves negative sign
        assert call["json"]["timeValidity"] == "GOOD_TILL_CANCEL"
        assert call["params"] == {"extendedHours": "true"}

    @pytest.mark.asyncio
    async def test_zero_quantity_rejected_locally(self):
        client, _ = _make_client()
        with pytest.raises(ValueError, match="non-zero"):
            await client.place_limit_order("AAPL_US_EQ", 0, limit_price=150.0)

    @pytest.mark.asyncio
    async def test_negative_limit_price_rejected_locally(self):
        client, _ = _make_client()
        with pytest.raises(ValueError, match="limit_price"):
            await client.place_limit_order("AAPL_US_EQ", 1, limit_price=-1.0)

    @pytest.mark.asyncio
    async def test_invalid_time_validity_rejected_locally(self):
        client, _ = _make_client()
        with pytest.raises(ValueError, match="timeValidity"):
            await client.place_limit_order(
                "AAPL_US_EQ", 1, limit_price=150.0, time_validity="MONTH"  # type: ignore[arg-type]
            )


# ── place_stop_order ────────────────────────────────────────────────


class TestPlaceStopOrder:
    @pytest.mark.asyncio
    async def test_sell_stop_loss(self):
        client, http = _make_client()
        await client.place_stop_order("AAPL_US_EQ", -1.5, stop_price=140.0)
        call = http.calls[-1]
        assert call["url"].endswith("/api/v0/equity/orders/stop")
        assert call["json"] == {
            "ticker": "AAPL_US_EQ",
            "quantity": -1.5,  # sell stop = stop-loss
            "stopPrice": 140.0,
            "timeValidity": "DAY",
        }

    @pytest.mark.asyncio
    async def test_buy_stop_breakout(self):
        client, http = _make_client()
        await client.place_stop_order("NVDA_US_EQ", 1.0, stop_price=950.0)
        call = http.calls[-1]
        assert call["json"]["quantity"] == 1.0
        assert call["json"]["stopPrice"] == 950.0


# ── place_stop_limit_order ──────────────────────────────────────────


class TestPlaceStopLimitOrder:
    @pytest.mark.asyncio
    async def test_full_payload(self):
        client, http = _make_client()
        await client.place_stop_limit_order(
            "AAPL_US_EQ",
            -10.0,
            stop_price=145.0,
            limit_price=144.0,
        )
        call = http.calls[-1]
        assert call["url"].endswith("/api/v0/equity/orders/stop_limit")
        assert call["json"] == {
            "ticker": "AAPL_US_EQ",
            "quantity": -10.0,
            "stopPrice": 145.0,
            "limitPrice": 144.0,
            "timeValidity": "DAY",
        }


# ── cancel_order ────────────────────────────────────────────────────


class TestCancelOrder:
    @pytest.mark.asyncio
    async def test_accepted_returns_true(self):
        client, http = _make_client(_FakeResponse(200, {}))
        ok = await client.cancel_order(987654321)
        assert ok is True
        call = http.calls[-1]
        assert call["method"] == "DELETE"
        assert call["url"].endswith("/api/v0/equity/orders/987654321")

    @pytest.mark.asyncio
    async def test_404_returns_false(self):
        client, _ = _make_client(_FakeResponse(404, {}))
        ok = await client.cancel_order(111)
        assert ok is False


# ── equity/metadata/instruments ───────────────────────────────────


class TestEquityInstrumentsMetadata:
    @pytest.mark.asyncio
    async def test_fetch_builds_stock_etf_set_only(self):
        payload: list[dict[str, Any]] = [
            {"ticker": "AAPL_US_EQ", "type": "STOCK", "name": "Apple"},
            {"ticker": "SPY_CFD_US", "type": "CFD", "name": "S&P CFD"},
        ]
        client, http = _make_client(_FakeResponse(200, payload))
        out = await client.fetch_equity_instruments_list()
        assert len(out) == 2
        assert client.tradeable_equity_instrument_count() == 1
        ok, detail = await client.is_us_equity_instrument_tradeable("AAPL")
        assert ok is True
        assert detail == "AAPL_US_EQ"
        ok2, _ = await client.is_us_equity_instrument_tradeable("SPY")
        assert ok2 is False
        assert http.calls[-1]["url"].endswith("/api/v0/equity/metadata/instruments")
        n_calls = len(http.calls)
        await client.fetch_equity_instruments_list()
        assert len(http.calls) == n_calls  # cache hit — no second upstream GET
