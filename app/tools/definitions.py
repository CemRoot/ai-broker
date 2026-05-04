"""
Tool definitions for the autonomous Paper Agent (Faz 3).

These are "function tools" intended for LLM tool-calling.
All tool outputs are expected to be returned as **English** strings.
"""

from __future__ import annotations

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_news",
            "description": "Fetch latest news and sentiment for a stock ticker (Finnhub + batch scoring).",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "days": {"type": "integer", "default": 2},
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_technical",
            "description": "Get technical analysis (RSI/SMA + PokieTicker-style price features when available).",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_memories",
            "description": "Retrieve past trading experiences and lessons for a ticker from RAG memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_portfolio",
            "description": "Get current paper trading portfolio: positions, P&L estimate, available cash.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_macro_context",
            "description": "Get macro context (VIX + notable risk headlines + recent high-impact Trump posts if available).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "screen_stocks",
            "description": "Screen S&P 500 for momentum/volume opportunities. Returns top 5 candidates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "min_volume_ratio": {"type": "number", "default": 1.5},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_t212_equity_instrument",
            "description": (
                "When execution is Trading 212 equity: verify a US-style symbol (e.g. AAPL) exists "
                "as a STOCK/ETF in this account's invest universe (not CFD-only). Call before BUY proposals."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Base symbol (AAPL) or T212 ticker (AAPL_US_EQ).",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
]

