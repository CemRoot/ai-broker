"""Internal HTTP routes: optional API key gate (no full app lifespan)."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.routes import router as api_router
from app.core.config import Settings


@pytest.mark.asyncio
async def test_internal_positions_403_when_key_set_but_header_missing():
    app = FastAPI()
    app.state.settings = Settings(internal_api_key="secret")
    app.state.t212 = None
    app.include_router(api_router)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/internal/positions")
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_internal_positions_reaches_handler_when_key_matches():
    app = FastAPI()
    app.state.settings = Settings(internal_api_key="secret")
    app.state.t212 = None
    app.include_router(api_router)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get(
            "/internal/positions",
            headers={"X-Internal-Api-Key": "secret"},
        )
    assert r.status_code == 503


@pytest.mark.asyncio
async def test_internal_positions_no_key_required_when_setting_empty():
    app = FastAPI()
    app.state.settings = Settings(internal_api_key="")
    app.state.t212 = None
    app.include_router(api_router)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/internal/positions")
    assert r.status_code == 503
