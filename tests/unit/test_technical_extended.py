"""Tests for PokieTicker-style price feature row (synthetic OHLCV)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.tools.technical_extended import TECHNICAL_FEATURE_KEYS, _compute_price_features_df


def test_compute_price_features_last_row():
    n = 80
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    close = 100 + np.cumsum(np.random.default_rng(42).normal(0, 0.5, n))
    df = pd.DataFrame(
        {
            "open": close * 0.998,
            "high": close * 1.002,
            "low": close * 0.997,
            "close": close,
            "volume": 1e6 + np.arange(n),
        },
        index=idx,
    )
    feat = _compute_price_features_df(df)
    feat = feat.dropna(subset=["ret_10d", "rsi_14"], how="any")
    assert len(feat) >= 30
    last = feat.iloc[-1]
    for k in TECHNICAL_FEATURE_KEYS:
        assert k in last.index
        assert not pd.isna(last[k])
