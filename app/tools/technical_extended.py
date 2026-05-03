"""
PokieTicker-style **price-only** feature snapshot (last trading row).

News columns from ``FEATURE_COLS`` are omitted here — supplied separately via
``news_pipeline``. Logic aligned with ``external/PokieTicker/backend/ml/features.py``
(price block only, ``shift(1)`` for leakage control).

Usage::

    snap = await get_extended_price_features("AMD")
    print(snap.to_dict())
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yfinance as yf

from app.core.logging import get_logger

log = get_logger("technical_extended")

# Price / technical columns from PokieTicker FEATURE_COLS (no news_*)
TECHNICAL_FEATURE_KEYS = [
    "ret_1d",
    "ret_3d",
    "ret_5d",
    "ret_10d",
    "volatility_5d",
    "volatility_10d",
    "volume_ratio_5d",
    "gap",
    "ma5_vs_ma20",
    "rsi_14",
]


@dataclass
class ExtendedPriceSnapshot:
    symbol: str
    features: dict[str, float | None] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict:
        return {"symbol": self.symbol, "features": self.features, "error": self.error}

    def summary_text(self) -> str:
        if self.error:
            return f"{self.symbol}: ⚠️ {self.error}"
        parts = [f"📐 {self.symbol} (PokieTicker-style price features)"]
        for k in TECHNICAL_FEATURE_KEYS:
            v = self.features.get(k)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                parts.append(f"{k}={v:.4f}")
        return " | ".join(parts)


def _compute_price_features_df(df: pd.DataFrame) -> pd.DataFrame:
    """Expects columns: open, high, low, close, volume (lower case)."""
    df = df.copy()
    close = df["close"].astype(float)
    vol = df["volume"].astype(float) if "volume" in df.columns else pd.Series(0.0, index=df.index)

    df["ret_1d"] = close.pct_change(1).shift(1)
    df["ret_3d"] = close.pct_change(3).shift(1)
    df["ret_5d"] = close.pct_change(5).shift(1)
    df["ret_10d"] = close.pct_change(10).shift(1)

    df["volatility_5d"] = close.pct_change().rolling(5).std().shift(1)
    df["volatility_10d"] = close.pct_change().rolling(10).std().shift(1)

    avg_vol_5 = vol.rolling(5).mean().shift(1)
    df["volume_ratio_5d"] = vol.shift(1) / avg_vol_5.clip(lower=1)

    df["gap"] = (df["open"] / close.shift(1) - 1).shift(1)

    ma5 = close.rolling(5).mean().shift(1)
    ma20 = close.rolling(20).mean().shift(1)
    df["ma5_vs_ma20"] = ma5 / ma20.clip(lower=0.01) - 1

    delta = close.diff().shift(1)
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.clip(lower=1e-10)
    df["rsi_14"] = 100 - 100 / (1 + rs)

    return df


def _extended_from_yfinance(symbol: str, period: str = "1y") -> ExtendedPriceSnapshot:
    try:
        raw = yf.download(
            symbol,
            period=period,
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
        if raw is None or raw.empty:
            return ExtendedPriceSnapshot(symbol=symbol, error=f"No OHLCV for {symbol}")

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]

        df = raw.rename(columns={c: str(c).lower() for c in raw.columns})
        for col in ("open", "high", "low", "close"):
            if col not in df.columns:
                return ExtendedPriceSnapshot(symbol=symbol, error=f"Missing {col} column")

        if "volume" not in df.columns:
            df["volume"] = 0.0

        feat = _compute_price_features_df(df)
        feat = feat.dropna(subset=["ret_10d", "rsi_14"], how="any")
        if feat.empty:
            return ExtendedPriceSnapshot(symbol=symbol, error="Insufficient history for features")

        last = feat.iloc[-1]
        out: dict[str, float | None] = {}
        for k in TECHNICAL_FEATURE_KEYS:
            v = last.get(k)
            if v is not None and not pd.isna(v):
                out[k] = float(v)
            else:
                out[k] = None

        return ExtendedPriceSnapshot(symbol=symbol, features=out)
    except Exception as exc:
        log.error("extended technical failed for %s: %s", symbol, exc)
        return ExtendedPriceSnapshot(symbol=symbol, error=str(exc))


async def get_extended_price_features(symbol: str, period: str = "1y") -> ExtendedPriceSnapshot:
    return await asyncio.to_thread(_extended_from_yfinance, symbol.upper(), period)
