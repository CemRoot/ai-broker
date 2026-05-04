"""Unit tests for public dashboard snapshot helpers."""

from __future__ import annotations

from app.services.public_live_snapshot import (
    parse_paper_cycle_content,
    redact_secrets,
    truncate,
)


def test_parse_paper_cycle_roundtrip():
    body = """EVENT: MIDDAY
UTC: 2026-05-04 12:00:00

ANALYSIS:
Macro calm. NVDA stretched.

DECISIONS_JSON:
[{"ticker": "NVDA", "action": "HOLD", "confidence": 0.55, "reasoning": "RSI high"}]
"""
    excerpt, dec = parse_paper_cycle_content(body)
    assert "Macro" in excerpt or "NVDA" in excerpt
    assert len(dec) == 1
    assert dec[0]["ticker"] == "NVDA"
    assert dec[0]["action"] == "HOLD"


def test_parse_paper_cycle_malformed():
    excerpt, dec = parse_paper_cycle_content(
        "no markers here gsk_abcdefghijklmnopqrstuvwxyz0123456789"
    )
    assert dec == []
    assert "[redacted]" in excerpt


def test_redact_bearer():
    s = redact_secrets("Authorization: Bearer abcdefghijklmnopqrstuvwxyz0123456789")
    assert "[redacted]" in s
    assert "abcdefghijklmnopqrst" not in s


def test_truncate():
    assert truncate("abc", 10) == "abc"
    assert len(truncate("x" * 20, 10)) == 10
