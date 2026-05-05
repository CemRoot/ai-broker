"""Unit tests for Truth Social monitor helpers (no live network)."""

from __future__ import annotations

import pytest

from app.services.trump_monitor import TrumpMonitor, _extract_json_object, _mastodon_html_to_plain


def test_extract_json_object_plain():
    assert _extract_json_object('{"impact_score": 8}') == {"impact_score": 8}


def test_extract_json_object_fenced():
    raw = 'Sure:\n```json\n{"sentiment": "bullish", "impact_score": 7}\n```'
    out = _extract_json_object(raw)
    assert out.get("impact_score") == 7
    assert out.get("sentiment") == "bullish"


def test_cnn_media_split_filters_images_and_videos():
    from app.core.config import Settings

    s = Settings()
    m = TrumpMonitor(s, None, None, None)  # type: ignore[arg-type]
    images, videos = m._cnn_media_split(
        [
            "https://x/a.jpg",
            "https://x/b.png?x=1",
            "https://x/c.mp4",
            "https://x/d.mov?dl=1",
        ]
    )
    assert images == ["https://x/a.jpg", "https://x/b.png?x=1"]
    assert videos == ["https://x/c.mp4", "https://x/d.mov?dl=1"]


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


@pytest.mark.asyncio
async def test_pull_recent_statuses_skips_duplicate_and_empty(monkeypatch):
    from app.core.config import Settings

    s = Settings()
    m = TrumpMonitor(s, None, None, None)  # type: ignore[arg-type]
    m._last_processed_post_id = "100"

    async def fake_fetch():
        return 200, [
            {"id": "100", "content": "older", "created_at": "2026-05-04T23:31:46.926Z", "media": []},
            {"id": "101", "content": "", "created_at": "2026-05-04T23:31:46.926Z", "media": []},
            {"id": "102", "content": "new", "created_at": "2026-05-04T23:31:46.926Z", "media": []},
        ], ""

    calls: list[str] = []

    async def fake_on_post(status):
        calls.append(str(status.get("id")))

    monkeypatch.setattr(m, "_fetch_cnn_archive", fake_fetch)
    monkeypatch.setattr(m, "on_trump_post", fake_on_post)

    out = await m.pull_recent_statuses(limit=10)
    assert out["error_status"] == 200
    assert out["fetched"] == 3
    assert out["new_posts"] == 1
    assert calls == ["102"]


@pytest.mark.asyncio
async def test_pull_recent_statuses_image_passes_to_pipeline(monkeypatch):
    from app.core.config import Settings

    s = Settings()
    m = TrumpMonitor(s, None, None, None)  # type: ignore[arg-type]
    m._last_processed_post_id = None

    async def fake_last_id():
        return None

    async def fake_fetch():
        return 200, [
            {
                "id": "103",
                "content": "",
                "created_at": "2026-05-04T23:31:46.926Z",
                "media": ["https://x/img.jpg", "https://x/vid.mp4"],
            }
        ], ""

    captured: list[dict] = []

    async def fake_on_post(status):
        captured.append(status)

    monkeypatch.setattr(m, "_load_last_processed_post_id", fake_last_id)
    monkeypatch.setattr(m, "_fetch_cnn_archive", fake_fetch)
    monkeypatch.setattr(m, "on_trump_post", fake_on_post)

    out = await m.pull_recent_statuses(limit=10)
    assert out["new_posts"] == 1
    assert captured
    media = captured[0]["media_attachments"]
    assert len(media) == 1
    assert media[0]["url"] == "https://x/img.jpg"
