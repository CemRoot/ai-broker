import logging
import traceback
from typing import Optional
import asyncpg
from pydantic_settings import BaseSettings, SettingsConfigDict

log = logging.getLogger(__name__)

class DatabaseSettings(BaseSettings):
    supabase_db_url: str = ""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

class SupabaseDatabase:
    """Singleton for asyncpg connection pool to Supabase."""
    
    _instance: Optional['SupabaseDatabase'] = None
    _pool: Optional[asyncpg.Pool] = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(SupabaseDatabase, cls).__new__(cls)
        return cls._instance

    def __init__(self, settings: Optional[DatabaseSettings] = None):
        if not hasattr(self, '_settings'):
            self._settings = settings or DatabaseSettings()
            self._pool_size = 10
            self.last_connect_error: str | None = None

    def _ensure_supabase_ssl(self, url: str) -> str:
        """Supabase pooler/direct host requires TLS; asyncpg may hang without sslmode."""
        lower = url.lower()
        if "supabase.com" not in lower:
            return url
        if "sslmode=" in lower or "ssl=" in lower:
            return url
        sep = "&" if "?" in url else "?"
        return f"{url}{sep}sslmode=require"

    async def connect(self, dsn: str | None = None):
        """Initializes the connection pool.

        ``dsn`` overrides ``DatabaseSettings.supabase_db_url`` (use the same URI as
        ``Settings.supabase_db_url`` from ``app.core.config`` so `.env` is single-sourced).
        """
        if self._pool is not None:
            return

        url = (dsn or self._settings.supabase_db_url or "").strip()
        self.last_connect_error = None
        if not url:
            self.last_connect_error = "SUPABASE_DB_URL empty"
            log.warning("SUPABASE_DB_URL is not set. Memory/DB functions will not work.")
            return

        url = self._ensure_supabase_ssl(url)

        async def init_connection(conn):
            # Lazy import: ``pgvector.asyncpg`` can block for seconds at import time on some setups;
            # delaying until first pool connection keeps pytest imports and CLI fast.
            from pgvector.asyncpg import register_vector

            await register_vector(conn)

        try:
            # ``timeout``: fail fast if host/firewall/SSL wrong (default ~60s hang confuses users).
            self._pool = await asyncpg.create_pool(
                dsn=url,
                min_size=1,
                max_size=self._pool_size,
                timeout=25,
                command_timeout=60,
                init=init_connection,
            )
            log.info("Supabase asyncpg connection pool initialized (with pgvector support).")
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc!s}".strip() or repr(exc)
            self.last_connect_error = detail
            log.error(
                "Failed to initialize Supabase connection pool (%s). Traceback:\n%s",
                detail,
                traceback.format_exc(),
            )

    async def close(self):
        """Closes the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            log.info("Supabase asyncpg connection pool closed.")

    def get_pool(self) -> Optional[asyncpg.Pool]:
        """Returns the connection pool, or None if not connected."""
        return self._pool
