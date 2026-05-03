"""Unit tests for FMP-based S&P screener (no live API)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.screener import SPScreener


@pytest.mark.asyncio
async def test_screener_filters_volume_etf_and_ranks():
    settings = MagicMock()
    settings.fmp_api_key = "test-key"

    payload = [
        {
            "symbol": "AAA",
            "exchangeShortName": "NYSE",
            "volume": 300,
            "avgVolume": 100,
            "changesPercentage": 5.0,
        },
        {
            "symbol": "LOWVOL",
            "exchangeShortName": "NYSE",
            "volume": 100,
            "avgVolume": 100,
            "changesPercentage": 10.0,
        },
        {"symbol": "ETFX", "isEtf": True, "exchangeShortName": "NYSE", "volume": 900, "avgVolume": 100},
        {
            "symbol": "BBB",
            "exchangeShortName": "NASDAQ",
            "volume": 250,
            "avgVolume": 100,
            "changesPercentage": 1.0,
        },
    ]

    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=payload)

    client = MagicMock()
    client.get = AsyncMock(return_value=resp)

    screener = SPScreener(settings, client)
    out = await screener.get_candidates(min_volume_ratio=1.5)

    tickers = [x["ticker"] for x in out]
    assert "LOWVOL" not in tickers
    assert "ETFX" not in tickers
    assert tickers[0] == "AAA"
    assert len(out) <= 5


@pytest.mark.asyncio
async def test_screener_returns_empty_without_key():
    settings = MagicMock()
    settings.fmp_api_key = ""
    client = MagicMock()
    screener = SPScreener(settings, client)
    assert await screener.get_candidates() == []
    client.get.assert_not_called()
