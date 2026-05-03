"""Tests for Faz 1.5-b analysis runner helpers (no live LLM)."""

from __future__ import annotations

from app.services.analysis_runner import (
    _build_news_digest,
    _build_user_message_plain,
)


def test_build_news_digest():
    merged = [
        {
            "relevant": True,
            "sentiment": "positive",
            "title": "Beat",
            "summary": "EPS up",
        },
        {"relevant": False, "title": "Weather", "summary": "Rain"},
    ]
    d = _build_news_digest(merged)
    assert "Beat" in d
    assert "positive" in d


def test_build_user_message_plain_has_sections():
    body = _build_user_message_plain(
        symbol="AMD",
        position_info="none",
        tech_text="RSI 50",
        extended_text="ret_1d=0.01",
        news_digest="Recent scored headlines:\n- [+] x: y",
        memories_digest=None,
    )
    assert "AMD" in body
    assert "RSI 50" in body
    assert "ret_1d" in body
    assert "Recent scored" in body
