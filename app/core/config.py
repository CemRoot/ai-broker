"""
Centralised settings via pydantic-settings.

All env variables are loaded from `.env` and validated at startup.
Import with:  ``from app.core.config import get_settings``
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration — every field maps to an env variable."""

    # ── Trading 212 ──────────────────────────────────────────────────
    t212_base_url: str = "https://demo.trading212.com"
    t212_demo_api_key: str = ""
    t212_demo_api_secret: str = ""

    # ── LLM — Groq (primary) ────────────────────────────────────────
    groq_api_key: str = ""

    # ── LLM — Ollama (fallback) ─────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "deepseek-r1:14b"
    #: When true, the PaperAgent skips Groq and runs a local-prepass cycle:
    #: Python-side fans out ``get_macro_context``, ``get_portfolio``, ``screen_stocks``,
    #: and per-ticker ``get_technical`` / ``get_news`` / ``get_memories`` (real data),
    #: then asks Ollama once with the combined context. Used to keep Groq tokens
    #: untouched on dev machines (Apple Silicon Metal backend).
    prefer_local_llm: bool = False
    #: Ticker count cap for local-prepass screener candidates per cycle (keeps prompt small).
    paper_local_prepass_top_k: int = 5

    # ── Telegram ────────────────────────────────────────────────────
    telegram_bot_token: str = ""
    telegram_webhook_url: str = ""  # empty → polling mode (dev)
    telegram_webhook_secret: str = ""  # X-Telegram-Bot-Api-Secret-Token
    telegram_allowed_user_ids: str = ""  # comma-separated int IDs

    # ── Misc ────────────────────────────────────────────────────────
    finnhub_api_key: str = ""
    fmp_api_key: str = ""
    paper_agent_enabled: bool = False
    #: Starting NAV for drawdown baseline, ``/paper stats`` vs-start, and ``/paper reset`` (Supabase ledger).
    #: Numeric value is in **account currency** (``PAPER_ACCOUNT_CURRENCY``, or T212 summary when ``t212`` execution).
    #: Env name retains ``_usd`` for backward compatibility.
    paper_starting_nav_usd: float = 20_000.0
    #: Supabase-only paper ledger currency (ISO 4217). With ``PAPER_EXECUTION_BACKEND=t212``, T212 ``account/summary`` wins.
    paper_account_currency: str = "USD"
    #: Halt new BUY orders when drawdown from peak marked-to-market NAV exceeds this percent.
    paper_max_drawdown_pct: float = 30.0
    #: Cache `get_technical` results per ticker (seconds). 0 = disabled.
    paper_technical_cache_ttl_sec: int = 90
    #: ``supabase`` = virtual ledger (PaperBroker). ``t212`` = place orders on Trading 212 demo/live API + DB audit rows.
    paper_execution_backend: str = "supabase"
    #: Market orders: allow extended hours / queue when exchange closed (T212 API).
    paper_t212_extended_hours: bool = True
    #: When ``t212`` execution: poll pending T212 orders and write ``paper_trades`` mirror on fill.
    paper_t212_mirror_poller_enabled: bool = True
    #: Seconds between poller ticks (min 15 recommended; history endpoint is 6 req/min).
    paper_t212_pending_poll_sec: int = 90
    #: Each tick: ``GET /equity/orders`` to enqueue app/web pending orders not yet mirrored.
    paper_t212_reconcile_external_orders: bool = True
    #: When ``t212`` execution: overwrite ``paper_account`` / ``paper_portfolio`` from T212 API (shadow ledger).
    paper_t212_sync_supabase_ledger: bool = True
    log_level: str = "INFO"
    # Faz 1.5-b: pack user prompt with TOON when ``toon-format`` is installed (``uv pip install -e ".[integrations]"``)
    use_toon_prompts: bool = False

    # ── Faz 5 — Web UI ───────────────────────────────────────────────
    #: Serve ``GET /ui`` (static page → ``POST /internal/analyze``). Use proxy auth for public hosts.
    web_ui_enabled: bool = False
    #: When non-empty, ``/internal/*`` requires header ``X-Internal-Api-Key`` (same value). Browser ``/ui`` uses ``POST /ui/analyze`` instead.
    internal_api_key: str = ""

    # ── Faz 1.5-d — Truth Social / Trump monitor ────────────────────
    truth_social_email: str = ""
    truth_social_password: str = ""
    truth_social_access_token: str = ""
    truth_social_base_url: str = "https://truthsocial.com"
    #: Mastodon-compatible streaming: ``sse`` (HTTP event-stream, default) or ``websocket``
    truth_social_stream_transport: str = "sse"
    trump_truth_account_username: str = "realDonaldTrump"
    trump_impact_threshold: float = 5.0
    # Llama 3.2 vision preview was deprecated on Groq (Oct 2025). Llama 4 Maverick
    # was deprecated March 2026. Llama 4 Scout is the current production multimodal
    # model on Groq as of May 2026 (see https://console.groq.com/docs/models).
    groq_vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    #: In-process Trump REST polling fallback interval (seconds). Runs on top of the
    #: WebSocket user-stream so we still get posts when the bot account does not
    #: follow @realDonaldTrump or when Cloudflare hiccups silence the stream. Set
    #: to 0 to disable (e.g. when you fully rely on the GitHub Actions cron).
    trump_pull_interval_sec: int = 60
    #: Master switch for the Trump/Truth Social monitor. Set to ``false`` when
    #: the host's egress IP is rate-limited / 403'd by Truth Social Cloudflare
    #: (common on datacenter ranges like Oracle Cloud, Hetzner). Disables both
    #: the WebSocket consumer and the REST puller; ``POST /internal/trump/pull``
    #: still works for ad-hoc polling from elsewhere (e.g. GitHub Actions cron).
    trump_monitor_enabled: bool = True

    #: Enrich ``get_macro_context`` with the CNN Fear & Greed Index (free public
    #: endpoint). The single call also yields put/call options and VIX
    #: sub-component scores — replaces three rows in the canonical "Korku &
    #: Açgözlülük" data layer with one HTTP GET. Set false to skip when CNN is
    #: down or you don't want the extra network call.
    macro_fear_greed_enabled: bool = True

    # ── Faz 2: RAG ve Kalıcı Hafıza ─────────────────────────────────
    supabase_url: str = ""
    supabase_key: str = ""
    supabase_db_url: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore unrecognised env vars silently
    )

    # ── Helpers ─────────────────────────────────────────────────────

    @property
    def allowed_user_ids(self) -> set[int]:
        """Parse ``TELEGRAM_ALLOWED_USER_IDS`` into a set of ints."""
        raw = self.telegram_allowed_user_ids.strip()
        if not raw:
            return set()
        return {int(uid.strip()) for uid in raw.split(",") if uid.strip()}

    @property
    def t212_api_url(self) -> str:
        """Full base URL for T212 API, e.g. ``https://demo.trading212.com/api/v0``."""
        return f"{self.t212_base_url.rstrip('/')}/api/v0"

    @property
    def paper_executes_on_t212(self) -> bool:
        return (self.paper_execution_backend or "").strip().lower() == "t212"


@lru_cache
def get_settings() -> Settings:
    """Singleton settings instance, cached after first call."""
    return Settings()
