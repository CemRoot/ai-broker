"""Unit tests for PokieTicker-style news batch helpers (no live LLM)."""

from __future__ import annotations

from app.services import news_pipeline as npipe


def test_extract_relevant_short_passthrough():
    text = "AMD announced new chips. Revenue beat."
    assert npipe.extract_relevant_text(text, "AMD") == text


def test_build_batch_prompt_plain_contains_titles():
    arts = [
        {"title": "AMD beats estimates", "description": "Strong data center sales."},
        {"title": "Unrelated weather", "description": "Rain in Seattle."},
    ]
    p = npipe.build_batch_prompt("AMD", arts, use_toon=False)
    assert "AMD" in p
    assert "[0]" in p and "[1]" in p
    assert "beats estimates" in p


def test_build_batch_prompt_toon_packs_articles_table():
    """When toon-format is installed (project default), prompt body uses TOON tabular form."""
    arts = [
        {"title": "AMD beats estimates", "description": "Strong data center sales."},
        {"title": "Macro update", "description": "Rates outlook."},
    ]
    p = npipe.build_batch_prompt("AMD", arts, use_toon=True)
    assert "AMD" in p
    assert "```toon" in p
    assert "articles[2]{i,title,extract}" in p
    assert "AMD beats estimates" in p


def test_parse_layer1_response_valid():
    raw = """Here is the result:
[{"i":0,"r":"y","s":"+","e":"Earnings beat","u":"Growth","d":""}]
"""
    scores = npipe.parse_layer1_response(raw, 2)
    assert len(scores) == 1
    assert scores[0]["i"] == 0


def test_parse_layer1_response_fenced():
    raw = '```json\n[{"i":0,"r":"n","s":"0","e":"","u":"","d":""}]\n```'
    scores = npipe.parse_layer1_response(raw, 1)
    assert len(scores) == 1


def test_merge_article_scores():
    arts = [{"title": "A", "description": ""}, {"title": "B", "description": ""}]
    scores = [{"i": 0, "r": "y", "s": "+", "e": "ok", "u": "u", "d": "d"}]
    m = npipe.merge_article_scores(arts, scores)
    assert m[0]["relevant"] is True
    assert m[0]["sentiment"] == "positive"
    assert m[1]["relevant"] is False
