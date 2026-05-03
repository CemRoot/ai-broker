"""
Trading 212 ticker → yfinance symbol normalisation.

T212 uses suffixed tickers like ``AAPL_US_EQ``.  yfinance expects plain
symbols like ``AAPL``.  This module handles the mapping.
"""

from __future__ import annotations

import re

# Known special-case mappings (T212 base → yfinance)
_SPECIAL: dict[str, str] = {
    "BRKb": "BRK-B",
    "BRKa": "BRK-A",
}

# Regex: strip common T212 exchange suffixes
#   _US_EQ, _UK_EQ, _DE_EQ, _FR_EQ, _NL_EQ, etc.
_SUFFIX_RE = re.compile(r"_[A-Z]{2}_EQ$")


def t212_to_yfinance(t212_ticker: str) -> str:
    """Convert a Trading 212 ticker to a yfinance-compatible symbol.

    Examples::

        >>> t212_to_yfinance("AAPL_US_EQ")
        'AAPL'
        >>> t212_to_yfinance("BRKb_US_EQ")
        'BRK-B'
        >>> t212_to_yfinance("AMZN")
        'AMZN'
    """
    base = _SUFFIX_RE.sub("", t212_ticker)
    return _SPECIAL.get(base, base)


def yfinance_to_t212(yf_symbol: str, exchange: str = "US", suffix: str = "EQ") -> str:
    """Best-effort reverse mapping (for display / logging only).

    >>> yfinance_to_t212("AAPL")
    'AAPL_US_EQ'
    """
    # Reverse special cases
    for t212_base, yf_base in _SPECIAL.items():
        if yf_symbol == yf_base:
            return f"{t212_base}_{exchange}_{suffix}"
    return f"{yf_symbol}_{exchange}_{suffix}"
