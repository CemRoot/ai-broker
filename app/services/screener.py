"""
S&P 500 screener (Faz 3d).

Uses Financial Modeling Prep (FMP) API to obtain candidate tickers using a pure-Python filter.
The goal is NOT to scan thousands of tickers via yfinance — we only send a small shortlist to
downstream tools (technical/news).

When the FMP plan does not return live ``volume`` / ``changesPercentage`` (Basic plan returns
mostly static fields), we enrich the **top N candidates** with a single yfinance batch call so
the rank order actually reflects today's volume/momentum.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import Settings
from app.core.logging import get_logger

log = get_logger("screener")

# FMP migrated many routes to ``/stable/``; Basic tier often returns 403 on legacy ``/api/v3/stock-screener``.
STABLE_COMPANY_SCREENER = "https://financialmodelingprep.com/stable/company-screener"
LEGACY_STOCK_SCREENER_V3 = "https://financialmodelingprep.com/api/v3/stock-screener"


def _redact_secrets(msg: str) -> str:
    return re.sub(r"apikey=[^&\s\"']+", "apikey=***", msg, flags=re.IGNORECASE)


@dataclass(frozen=True)
class ScreenRow:
    ticker: str
    volume_ratio: float | None = None
    momentum_1d: float | None = None  # percent
    sector: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "volume_ratio": self.volume_ratio,
            "momentum_1d": self.momentum_1d,
            "sector": self.sector,
        }


class SPScreener:
    """
    Minimal FMP-based screener.

    Notes
    -----
    FMP responses vary by endpoint/plan; this implementation is defensive and will
    operate with partial fields.
    """

    def __init__(self, settings: Settings, http_client: httpx.AsyncClient) -> None:
        self._settings = settings
        self._http = http_client

    async def get_candidates(self, min_volume_ratio: float = 1.5) -> list[dict[str, Any]]:
        """
        Return top candidates (dicts) for the Paper Agent.

        Filter strategy (LLM-free):
        - volume > avgVolume * min_volume_ratio (if both fields exist)
        - prefer positive daily momentum (changesPercentage) when available
        - exchanges NYSE/NASDAQ when present
        """
        key = (self._settings.fmp_api_key or "").strip()
        if not key:
            log.warning("FMP_API_KEY missing; screener disabled.")
            return []

        data: list | None = None
        attempts: list[tuple[str, dict[str, Any]]] = [
            (
                STABLE_COMPANY_SCREENER,
                {"limit": 100, "apikey": key, "country": "US", "isEtf": False},
            ),
            (LEGACY_STOCK_SCREENER_V3, {"limit": 250, "apikey": key}),
        ]
        for url, params in attempts:
            try:
                resp = await self._http.get(url, params=params, timeout=45.0)
                if resp.status_code in (401, 403, 404):
                    log.warning(
                        "FMP screener %s HTTP %s — trying fallback if any",
                        url.rsplit("/", 1)[-1],
                        resp.status_code,
                    )
                    continue
                resp.raise_for_status()
                parsed = resp.json()
                if isinstance(parsed, list):
                    data = parsed
                    break
                log.warning("FMP screener unexpected JSON type from %s", url)
            except Exception as exc:
                log.error("FMP screener error (%s): %s", url.rsplit("/", 1)[-1], _redact_secrets(str(exc)))

        if not data:
            return []

        rows: list[ScreenRow] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            sym = (item.get("symbol") or item.get("ticker") or "").strip().upper()
            if not sym or len(sym) > 10:
                continue
            if item.get("isEtf") is True or str(item.get("type") or "").lower() == "etf":
                continue

            exchange = (item.get("exchangeShortName") or item.get("exchange") or "").upper()
            if exchange and exchange not in ("NYSE", "NASDAQ", "NMS", "NGM", "NCM"):
                continue

            volume = item.get("volume")
            avg_volume = item.get("avgVolume") or item.get("averageVolume")
            vol_ratio: float | None = None
            try:
                if volume is not None and avg_volume:
                    vol_ratio = float(volume) / max(1.0, float(avg_volume))
            except Exception:
                vol_ratio = None

            if vol_ratio is not None and vol_ratio < float(min_volume_ratio):
                continue

            mom = (
                item.get("changesPercentage")
                or item.get("changePercentage")
                or item.get("changePercent")
                or item.get("percentChange")
            )
            mom_pct: float | None = None
            try:
                if mom is not None:
                    mom_pct = float(mom)
            except Exception:
                mom_pct = None

            sector = item.get("sector") or None
            rows.append(ScreenRow(ticker=sym, volume_ratio=vol_ratio, momentum_1d=mom_pct, sector=sector))

        # ── Enrich first N rows with yfinance when FMP didn't carry vol/mom ───
        # FMP Basic plan returns mostly static fields (sector / marketCap) and
        # often omits ``volume`` and ``changesPercentage`` — without enrichment,
        # the rank order collapses to whatever order FMP serves the JSON in
        # (alphabetical-ish), which is useless. We pay one yfinance batch call
        # for the first ``enrich_top`` rows; this is a "shortlist", not a scan.
        enrich_top = min(30, len(rows))
        missing = [r for r in rows[:enrich_top] if r.volume_ratio is None or r.momentum_1d is None]
        if missing:
            try:
                enriched = await asyncio.to_thread(_yf_enrich, [r.ticker for r in missing])
                if enriched:
                    by_sym = {r.ticker: r for r in rows}
                    for sym, (vr, mm) in enriched.items():
                        if sym in by_sym:
                            old = by_sym[sym]
                            rows[rows.index(old)] = ScreenRow(
                                ticker=old.ticker,
                                volume_ratio=vr if old.volume_ratio is None else old.volume_ratio,
                                momentum_1d=mm if old.momentum_1d is None else old.momentum_1d,
                                sector=old.sector,
                            )
            except Exception as exc:
                log.warning("yfinance enrichment failed (continuing with raw FMP order): %s", exc)

        # Rank: volume_ratio (desc), then momentum (desc); None → 0.0 sentinel
        def key_fn(r: ScreenRow):
            vr = r.volume_ratio if r.volume_ratio is not None else 0.0
            mm = r.momentum_1d if r.momentum_1d is not None else 0.0
            return (vr, mm)

        rows.sort(key=key_fn, reverse=True)
        return [r.to_dict() for r in rows[:5]]


def _yf_enrich(tickers: list[str]) -> dict[str, tuple[float | None, float | None]]:
    """Batch yfinance OHLCV → (volume_ratio_5d, momentum_1d_pct).

    Runs in a worker thread because the underlying ``yfinance`` library is sync.
    Returns ``{symbol: (vol_ratio, momentum_pct)}`` for symbols where data was
    available.  Symbols with no rows are silently dropped — the caller already
    knows how to fall back to None.
    """
    if not tickers:
        return {}
    import yfinance as yf  # local import keeps module fast for tests/CLI

    out: dict[str, tuple[float | None, float | None]] = {}
    try:
        df = yf.download(
            tickers=" ".join(tickers),
            period="10d",
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
        )
    except Exception:
        return out

    for sym in tickers:
        try:
            sub = df[sym] if (sym in df.columns.get_level_values(0)) else df
            closes = sub["Close"].dropna()
            vols = sub["Volume"].dropna()
            if len(closes) < 2 or len(vols) < 2:
                continue
            momentum = float((closes.iloc[-1] / closes.iloc[-2]) - 1.0)
            avg_vol = float(vols.iloc[-6:-1].mean()) if len(vols) >= 6 else float(vols.iloc[:-1].mean())
            vol_ratio = float(vols.iloc[-1] / max(1.0, avg_vol)) if avg_vol > 0 else None
            out[sym] = (vol_ratio, momentum * 100.0)
        except Exception:
            continue
    return out

