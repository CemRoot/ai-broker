"""
Unit tests for the technical analysis module.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

from app.tools.technical import _compute_technicals, TechnicalSummary


def _make_ohlcv_df(n: int = 60, base_price: float = 100.0) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame for testing."""
    np.random.seed(42)
    dates = pd.date_range(end="2026-04-28", periods=n, freq="B")
    close = base_price + np.cumsum(np.random.randn(n) * 2)
    return pd.DataFrame(
        {
            "Open": close - 1,
            "High": close + 2,
            "Low": close - 2,
            "Close": close,
            "Volume": np.random.randint(1_000_000, 10_000_000, size=n),
        },
        index=dates,
    )


class TestComputeTechnicals:
    @patch("app.tools.technical.yf.download")
    def test_happy_path(self, mock_dl):
        mock_dl.return_value = _make_ohlcv_df()

        result = _compute_technicals("AMD")

        assert result.symbol == "AMD"
        assert result.error is None
        assert result.rsi_14 is not None
        assert 0 <= result.rsi_14 <= 100
        assert result.sma_20 is not None
        assert result.last_close is not None
        assert result.volume is not None

    @patch("app.tools.technical.yf.download")
    def test_empty_dataframe(self, mock_dl):
        mock_dl.return_value = pd.DataFrame()

        result = _compute_technicals("DELISTED")

        assert result.error is not None
        assert "No OHLCV" in result.error

    @patch("app.tools.technical.yf.download")
    def test_none_return(self, mock_dl):
        mock_dl.return_value = None

        result = _compute_technicals("GONE")

        assert result.error is not None

    @patch("app.tools.technical.yf.download")
    def test_exception(self, mock_dl):
        mock_dl.side_effect = Exception("network error")

        result = _compute_technicals("ERR")

        assert result.error == "network error"

    @patch("app.tools.technical.yf.download")
    def test_multiindex_columns(self, mock_dl):
        """yfinance sometimes returns MultiIndex columns for single tickers."""
        df = _make_ohlcv_df()
        df.columns = pd.MultiIndex.from_tuples(
            [(col, "AMD") for col in df.columns]
        )
        mock_dl.return_value = df

        result = _compute_technicals("AMD")

        assert result.error is None
        assert result.last_close is not None


class TestTechnicalSummary:
    def test_summary_text_with_data(self):
        ts = TechnicalSummary(symbol="AMD", rsi_14=72.5, sma_20=108.0, last_close=112.0)
        text = ts.summary_text()
        assert "AMD" in text
        assert "$112.00" in text
        assert "72.5" in text
        assert "🔴" in text  # overbought

    def test_summary_text_oversold(self):
        ts = TechnicalSummary(symbol="X", rsi_14=25.0, sma_20=50.0, last_close=48.0)
        assert "🟢" in ts.summary_text()

    def test_summary_text_error(self):
        ts = TechnicalSummary(symbol="BAD", error="delisted")
        assert "⚠️" in ts.summary_text()

    def test_to_dict(self):
        ts = TechnicalSummary(symbol="T", rsi_14=50.0, sma_20=100.0, last_close=101.0)
        d = ts.to_dict()
        assert d["symbol"] == "T"
        assert d["rsi_14"] == 50.0
