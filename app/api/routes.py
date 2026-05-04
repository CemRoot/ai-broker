"""
FastAPI routes: health, internal endpoints, and Telegram webhook.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as pkg_version

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from telegram import Update

from app.core.logging import get_logger
from app.services.analysis_runner import analysis_result_to_api_dict, run_symbol_analysis
from app.services.finnhub_news import fetch_company_news
from app.services.news_pipeline import analyze_news_batch
from app.services.public_live_snapshot import build_public_live_snapshot
from app.tools.technical_extended import get_extended_price_features

log = get_logger("api")

router = APIRouter()


async def require_internal_api_key(request: Request) -> None:
    """When ``INTERNAL_API_KEY`` is set, require matching ``X-Internal-Api-Key`` header."""
    settings = getattr(request.app.state, "settings", None)
    if not settings:
        raise HTTPException(503, "Application not ready")
    expected = (getattr(settings, "internal_api_key", None) or "").strip()
    if not expected:
        return
    got = (
        request.headers.get("X-Internal-Api-Key")
        or request.headers.get("x-internal-api-key")
        or ""
    ).strip()
    if got != expected:
        log.warning("Internal API key missing or invalid")
        raise HTTPException(403, "Unauthorized")


def _app_version() -> str:
    try:
        return pkg_version("ai-broker")
    except PackageNotFoundError:
        return "0.0.0"


def build_health_payload(*, settings, db, bot_app) -> dict:
    """JSON for ``GET /health`` (no secrets). Extracted for unit tests."""
    pool_ok = bool(db and db.get_pool())
    token_set = bool(settings and settings.telegram_bot_token)
    hook_base = (settings.telegram_webhook_url or "").strip() if settings else ""
    hook_secret = (settings.telegram_webhook_secret or "").strip() if settings else ""
    if settings and hook_base and hook_secret:
        telegram_mode = "webhook"
    elif settings and hook_base and not hook_secret:
        telegram_mode = "webhook_incomplete"
    elif token_set:
        telegram_mode = "polling"
    else:
        telegram_mode = "disabled"

    dsn_configured = bool(settings and (settings.supabase_db_url or "").strip())

    web_on = bool(settings and getattr(settings, "web_ui_enabled", False))
    dash_on = bool(settings and getattr(settings, "public_dashboard_enabled", False))
    return {
        "status": "ok",
        "phase": "5",
        "version": _app_version(),
        "web_ui": {"enabled": web_on},
        "public_dashboard": {"enabled": dash_on},
        "telegram": {
            "bot_token_configured": token_set,
            "mode": telegram_mode,
            "webhook_base_configured": bool(hook_base),
            "webhook_secret_configured": bool(hook_secret),
            "handlers_ready": bot_app is not None,
        },
        "memory_db": {
            "supabase_db_url_configured": dsn_configured,
            "asyncpg_pool_ready": pool_ok,
            "last_connect_error": (
                getattr(db, "last_connect_error", None) if db else None
            ),
        },
        "groq_configured": bool(settings and settings.groq_api_key),
    }


class NewsArticlePayload(BaseModel):
    model_config = {"extra": "ignore"}

    title: str = Field(..., min_length=1)
    description: str = ""


class NewsBatchRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=16)
    articles: list[NewsArticlePayload] = Field(..., min_length=1, max_length=150)


# ── Health ──────────────────────────────────────────────────────────

@router.get("/health", tags=["ops"])
async def health(request: Request):
    """Liveness + integration flags (no secrets). Use this when Telegram or DB “does nothing”."""
    return build_health_payload(
        settings=getattr(request.app.state, "settings", None),
        db=getattr(request.app.state, "db", None),
        bot_app=getattr(request.app.state, "bot_app", None),
    )


@router.get("/public/live", tags=["public"])
async def public_live_dashboard(request: Request):
    """Read-only paper snapshot for ``GET /finance`` (no ``INTERNAL_API_KEY``).

    With ``PAPER_EXECUTION_BACKEND=t212``, open positions and ``nav_display`` prefer
    live Trading 212 Public API values on each request; the Supabase ledger is still
    exposed for drift checks and trade history.
    """
    settings = getattr(request.app.state, "settings", None)
    if not settings or not getattr(settings, "public_dashboard_enabled", False):
        raise HTTPException(404, "Public dashboard disabled")
    return await build_public_live_snapshot(request)


# ── Internal / dev endpoints ────────────────────────────────────────

@router.get(
    "/internal/positions",
    tags=["internal"],
    dependencies=[Depends(require_internal_api_key)],
)
async def internal_positions(request: Request):
    """Fetch T212 open positions (dev/test convenience)."""
    t212 = request.app.state.t212
    if not t212:
        raise HTTPException(503, "T212 client not initialised")

    positions = await t212.get_positions()
    return [
        {
            "ticker": p.ticker,
            "quantity": p.quantity,
            "averagePrice": p.average_price_paid,
            "currentPrice": p.current_price,
            "pnl": round(p.pnl, 2),
            "pnlPercent": round(p.pnl_percent, 2),
        }
        for p in positions
    ]


@router.get(
    "/internal/technical/extended",
    tags=["internal"],
    dependencies=[Depends(require_internal_api_key)],
)
async def internal_technical_extended(request: Request, symbol: str):
    """PokieTicker-style price-only feature row (yfinance, last valid bar)."""
    snap = await get_extended_price_features(symbol.upper())
    return snap.to_dict()


@router.post(
    "/internal/news/batch",
    tags=["internal"],
    dependencies=[Depends(require_internal_api_key)],
)
async def internal_news_batch(request: Request, body: NewsBatchRequest):
    """Score submitted articles with Layer-1-style Groq batch (optional Ollama fallback)."""
    groq_svc = request.app.state.groq
    ollama_svc = request.app.state.ollama
    if not groq_svc and not ollama_svc:
        raise HTTPException(503, "No LLM service available")

    articles = [a.model_dump() for a in body.articles]
    try:
        merged, model = await analyze_news_batch(
            symbol=body.symbol,
            articles=articles,
            groq=groq_svc,
            ollama=ollama_svc,
        )
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from exc

    return {"symbol": body.symbol.upper(), "model": model, "articles": merged}


@router.get(
    "/internal/news/analyze",
    tags=["internal"],
    dependencies=[Depends(require_internal_api_key)],
)
async def internal_news_analyze(
    request: Request,
    symbol: str,
    limit: int = 25,
    days: int = 7,
):
    """Fetch Finnhub company news and run batch LLM scoring."""
    settings = request.app.state.settings
    http = request.app.state.http_client
    groq_svc = request.app.state.groq
    ollama_svc = request.app.state.ollama

    if not settings.finnhub_api_key:
        raise HTTPException(503, "FINNHUB_API_KEY not configured")
    if not groq_svc and not ollama_svc:
        raise HTTPException(503, "No LLM service available")

    lim = max(1, min(limit, 50))
    try:
        articles = await fetch_company_news(
            http,
            api_key=settings.finnhub_api_key,
            symbol=symbol.upper(),
            days=days,
            max_articles=lim,
        )
    except Exception as exc:
        log.error("Finnhub error: %s", exc)
        raise HTTPException(502, f"Finnhub failed: {exc}") from exc

    if not articles:
        return {"symbol": symbol.upper(), "model": "none", "articles": [], "note": "no articles"}

    sym = symbol.upper()
    try:
        merged, model = await analyze_news_batch(
            symbol=sym,
            articles=articles,
            groq=groq_svc,
            ollama=ollama_svc,
        )
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from exc

    return {"symbol": sym, "model": model, "articles": merged}


@router.post(
    "/internal/analyze",
    tags=["internal"],
    dependencies=[Depends(require_internal_api_key)],
)
async def internal_analyze(
    request: Request,
    symbol: str,
    include_news: bool = False,
    extended: bool = False,
    use_toon: bool | None = None,
):
    """Technical (+ optional Finnhub news batch + optional extended OHLC features) + one LLM call.

    Query params mirror Faz 1.5-b: ``include_news``, ``extended`` (PokieTicker-style row),
    ``use_toon`` (override ``USE_TOON_PROMPTS`` in .env when set).
    """
    result = await run_symbol_analysis(
        symbol=symbol.upper(),
        settings=request.app.state.settings,
        http_client=request.app.state.http_client,
        t212=request.app.state.t212,
        groq=request.app.state.groq,
        ollama=request.app.state.ollama,
        retriever=getattr(request.app.state, 'retriever', None),
        include_news=include_news,
        include_extended_technical=extended,
        use_toon=use_toon,
    )
    return analysis_result_to_api_dict(result)


@router.get(
    "/internal/usage",
    tags=["internal"],
    dependencies=[Depends(require_internal_api_key)],
)
async def internal_usage(request: Request):
    """Groq daily token / request counters."""
    groq_svc = request.app.state.groq
    if not groq_svc:
        raise HTTPException(503, "Groq service not initialised")
    return groq_svc.usage.to_dict()


@router.post(
    "/internal/trump/pull",
    tags=["internal"],
    dependencies=[Depends(require_internal_api_key)],
)
async def internal_trump_pull(request: Request, limit: int = 10):
    """REST polling fallback for Truth Social posts.

    Designed for the GitHub Actions cron worker (``.github/workflows/trump-pull-cron.yml``)
    so we don't depend on the WebSocket user-stream — which is silent unless the
    bot account follows ``@realDonaldTrump``. Returns ``{fetched, new_posts, error_status}``.
    Each new status is processed by the same impact LLM + Supabase write +
    Telegram alert + emergency PaperAgent path as the live stream.
    """
    monitor = getattr(request.app.state, "trump_monitor", None)
    if monitor is None:
        raise HTTPException(503, "TrumpMonitor not initialised")
    summary = await monitor.pull_recent_statuses(limit=limit)
    return summary


# ── Telegram webhook ───────────────────────────────────────────────

@router.post("/telegram/webhook", tags=["telegram"])
async def telegram_webhook(request: Request):
    """Receive Telegram updates via webhook.

    Flow (PTB v21+ compatible):
    1. Validate ``X-Telegram-Bot-Api-Secret-Token`` header.
    2. Parse JSON body → ``Update.de_json()``.
    3. ``await bot_app.process_update(update)`` (async).
    """
    bot_app = request.app.state.bot_app
    settings = request.app.state.settings

    if (settings.telegram_webhook_url or "").strip() and not (
        settings.telegram_webhook_secret or ""
    ).strip():
        log.critical(
            "TELEGRAM_WEBHOOK_SECRET empty while TELEGRAM_WEBHOOK_URL is set — refusing webhook"
        )
        raise HTTPException(500, "Webhook secret not configured")

    # 1. Secret token validation
    if settings.telegram_webhook_secret:
        header = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
        if header != settings.telegram_webhook_secret:
            log.warning("Webhook secret mismatch — rejecting request")
            raise HTTPException(403, "Invalid secret token")

    # 2. Parse update
    body = await request.json()
    update = Update.de_json(body, bot_app.bot)

    # 3. Process
    await bot_app.process_update(update)

    return {"ok": True}
