"""Unit tests for Truth Social monitor helpers (no live network)."""

from __future__ import annotations

from app.services.trump_monitor import TrumpMonitor, _extract_json_object, _mastodon_html_to_plain


def test_extract_json_object_plain():
    assert _extract_json_object('{"impact_score": 8}') == {"impact_score": 8}


def test_extract_json_object_fenced():
    raw = 'Sure:\n```json\n{"sentiment": "bullish", "impact_score": 7}\n```'
    out = _extract_json_object(raw)
    assert out.get("impact_score") == 7
    assert out.get("sentiment") == "bullish"


def test_should_process_username_match():
    from app.core.config import Settings

    s = Settings(
        truth_social_access_token="x",
        trump_truth_account_username="realDonaldTrump",
    )
    m = TrumpMonitor(s, None, None, None)  # type: ignore[arg-type]
    status = {"account": {"id": "1", "username": "realDonaldTrump"}, "id": "99"}
    assert m._should_process_status(status) is True


def test_should_process_id_match():
    from app.core.config import Settings

    s = Settings(truth_social_access_token="x")
    m = TrumpMonitor(s, None, None, None)  # type: ignore[arg-type]
    m._trump_account_id = "42"
    status = {"account": {"id": "42", "username": "someone"}, "id": "99"}
    assert m._should_process_status(status) is True


def test_mastodon_html_to_plain_strips_tags():
    raw = "<p>Hello <span>world</span></p>"
    assert _mastodon_html_to_plain(raw) == "Hello world"


def test_status_plain_text_reblog_when_outer_empty():
    from app.core.config import Settings

    s = Settings(truth_social_access_token="x")
    m = TrumpMonitor(s, None, None, None)  # type: ignore[arg-type]
    status = {
        "content": "",
        "reblog": {"content": "<p>Inner boost text</p>"},
        "id": "1",
    }
    assert m._status_plain_text(status) == "Inner boost text"


def test_status_plain_text_media_only_placeholder():
    from app.core.config import Settings

    s = Settings(truth_social_access_token="x")
    m = TrumpMonitor(s, None, None, None)  # type: ignore[arg-type]
    status = {"content": "", "reblog": None, "media_attachments": [{"type": "image", "url": "https://x/i.jpg"}]}
    assert "[Media-only" in m._status_plain_text(status)
