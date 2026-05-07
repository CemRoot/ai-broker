"""Browser routes: static analysis page."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from app.services.analysis_runner import analysis_result_to_api_dict, run_symbol_analysis

router = APIRouter(tags=["web"])

_STATIC_DIR = Path(__file__).resolve().parent / "static"


@router.get("/ui")
async def web_analysis_ui(request: Request):
    """Single-page symbol analysis (browser calls ``POST /ui/analyze``).

    Disabled unless ``WEB_UI_ENABLED=true``. For production, terminate TLS and add auth
    at the reverse proxy; this surface is not multi-tenant hardened.
    """
    settings = getattr(request.app.state, "settings", None)
    if not settings or not getattr(settings, "web_ui_enabled", False):
        raise HTTPException(status_code=404, detail="Web UI disabled")
    path = _STATIC_DIR / "index.html"
    if not path.is_file():
        raise HTTPException(status_code=503, detail="UI bundle missing")
    return FileResponse(path, media_type="text/html; charset=utf-8")


@router.get("/finance")
async def public_finance_dashboard(request: Request):
    """Live paper dashboard (static HTML; polls ``/public/live``)."""
    settings = getattr(request.app.state, "settings", None)
    if not settings or not getattr(settings, "public_dashboard_enabled", False):
        raise HTTPException(status_code=404, detail="Public dashboard disabled")
    path = _STATIC_DIR / "dashboard.html"
    if not path.is_file():
        raise HTTPException(status_code=503, detail="Dashboard bundle missing")
    return FileResponse(path, media_type="text/html; charset=utf-8")


@router.post("/ui/analyze")
async def web_ui_analyze(
    request: Request,
    symbol: str,
    include_news: bool = False,
    extended: bool = False,
    use_toon: bool | None = None,
):
    """Same pipeline as ``POST /internal/analyze`` for the bundled HTML UI (no API key).

    Use ``/internal/analyze`` + ``INTERNAL_API_KEY`` for scripted clients.
    """
    settings = getattr(request.app.state, "settings", None)
    if not settings or not getattr(settings, "web_ui_enabled", False):
        raise HTTPException(status_code=404, detail="Web UI disabled")
    result = await run_symbol_analysis(
        symbol=symbol.upper(),
        settings=settings,
        http_client=request.app.state.http_client,
        t212=request.app.state.t212,
        cerebras=request.app.state.cerebras,
        groq=request.app.state.groq,
        retriever=getattr(request.app.state, "retriever", None),
        include_news=include_news,
        include_extended_technical=extended,
        use_toon=use_toon,
    )
    return analysis_result_to_api_dict(result)
