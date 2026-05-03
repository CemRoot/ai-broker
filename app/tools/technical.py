"""
Technical analysis module — yfinance OHLCV + pandas-ta indicators.

Generalised from the Faz 0 ``fetch_amd_technicals()`` single-ticker helper.
Future phases will extend this with PokieTicker's 31-feature set.

Usage::

    summary = await get_technical_summary("AMD")
    print(summary.rsi_14, summary.sma_20, summary.last_close)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pandas as pd
import pandas_ta as ta  # noqa: F401 — used via df.ta accessor
import yfinance as yf

from app.core.logging import get_logger

log = get_logger("technical")


@dataclass
class TechnicalSummary:
    """MVP indicator snapshot for a single symbol."""

    symbol: str
    rsi_14: float | None = None
    sma_20: float | None = None
    last_close: float | None = None
    volume: float | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "rsi_14": self.rsi_14,
            "sma_20": self.sma_20,
            "last_close": self.last_close,
            "volume": self.volume,
            "error": self.error,
        }

    def summary_text(self) -> str:
        """Human-readable one-liner for Telegram / LLM prompt."""
        if self.error:
            return f"{self.symbol}: ⚠️ {self.error}"
        parts = [f"📊 {self.symbol}"]
        if self.last_close is not None:
            parts.append(f"Price: ${self.last_close:.2f}")
        if self.rsi_14 is not None:
            emoji = "🔴" if self.rsi_14 > 70 else ("🟢" if self.rsi_14 < 30 else "⚪")
            parts.append(f"RSI(14): {self.rsi_14:.1f} {emoji}")
        if self.sma_20 is not None:
            parts.append(f"SMA(20): ${self.sma_20:.2f}")
        return " | ".join(parts)


def _compute_technicals(symbol: str, period: str = "3mo") -> TechnicalSummary:
    """Synchronous yfinance download + pandas-ta computation.

    Runs in a thread via ``asyncio.to_thread``.
    """
    try:
        df = yf.download(
            symbol,
            period=period,
            interval="1d",
            progress=False,
            auto_adjust=True,
        )

        if df is None or df.empty:
            return TechnicalSummary(symbol=symbol, error=f"No OHLCV data for {symbol}")

        # yfinance may return MultiIndex columns for single ticker
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

        close = df["Close"].astype(float)
        volume_series = df["Volume"].astype(float) if "Volume" in df.columns else None

        rsi_series = ta.rsi(close, length=14)
        sma_series = ta.sma(close, length=20)

        rsi_val = (
            float(rsi_series.iloc[-1])
            if rsi_series is not None and not pd.isna(rsi_series.iloc[-1])
            else None
        )
        sma_val = (
            float(sma_series.iloc[-1])
            if sma_series is not None and not pd.isna(sma_series.iloc[-1])
            else None
        )
        price = float(close.iloc[-1])
        vol = float(volume_series.iloc[-1]) if volume_series is not None else None

        return TechnicalSummary(
            symbol=symbol,
            rsi_14=rsi_val,
            sma_20=sma_val,
            last_close=price,
            volume=vol,
        )

    except Exception as exc:
        log.error("Technical analysis failed for %s: %s", symbol, exc)
        return TechnicalSummary(symbol=symbol, error=str(exc))


async def get_technical_summary(symbol: str, period: str = "3mo") -> TechnicalSummary:
    """Async wrapper — runs yfinance in a thread so it doesn't block the event loop."""
    return await asyncio.to_thread(_compute_technicals, symbol, period)
