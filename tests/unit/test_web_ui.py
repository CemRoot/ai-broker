"""Web UI route (minimal app.state only — no full lifespan)."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.core.config import Settings
from app.web.routes import router as web_router


@pytest.mark.asyncio
async def test_ui_returns_404_when_disabled():
    app = FastAPI()
    app.state.settings = Settings(web_ui_enabled=False)
    app.include_router(web_router)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/ui")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_ui_returns_html_when_enabled():
    app = FastAPI()
    app.state.settings = Settings(web_ui_enabled=True)
    app.include_router(web_router)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/ui")
    assert r.status_code == 200
    assert "text/html" in (r.headers.get("content-type") or "")
    assert "Symbol analysis" in r.text


@pytest.mark.asyncio
async def test_ui_analyze_404_when_disabled():
    app = FastAPI()
    app.state.settings = Settings(web_ui_enabled=False)
    app.include_router(web_router)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post("/ui/analyze?symbol=AAPL")
    assert r.status_code == 404
