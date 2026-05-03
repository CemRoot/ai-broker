# syntax=docker/dockerfile:1.7

# ────────────────────────────────────────────────────────────────────────
# AI Broker — production image
# Strategy : two-stage, slim-bookworm (musl wheels would force pandas/numpy
#            to compile from source, which is not worth it on alpine).
# Layers   : (1) dep-only sync for cache, (2) project sync after app/ COPY.
# Runtime  : non-root, healthcheck, BuildKit cache mounts (uv + apt).
# ────────────────────────────────────────────────────────────────────────

ARG PY_BASE=python:3.12-slim-bookworm

# ── builder ────────────────────────────────────────────────────────────
FROM ${PY_BASE} AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/root/.cache/uv

WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends --no-install-suggests \
        ca-certificates \
        curl && \
    pip install --no-cache-dir "uv~=0.5.14"

# (1) Dependency sync — caches as long as pyproject.toml + uv.lock unchanged.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# (2) Project sync — only invalidated when app/ source actually changes.
COPY app ./app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


# ── runtime ────────────────────────────────────────────────────────────
FROM ${PY_BASE} AS runtime

# OCI metadata for registries (override at build with --build-arg).
ARG VCS_REF="dev"
ARG BUILD_DATE="1970-01-01T00:00:00Z"
LABEL org.opencontainers.image.title="ai-broker" \
      org.opencontainers.image.description="Personal AI trading advisor (FastAPI + Telegram + T212 + Supabase)" \
      org.opencontainers.image.source="https://github.com/dr-sam/ai-broker" \
      org.opencontainers.image.licenses="UNLICENSED" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends --no-install-suggests \
        ca-certificates \
        curl \
        tini && \
    useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser

COPY --from=builder --chown=appuser:appuser /app /app

USER appuser

EXPOSE 8000

# /health waits up to 90s for warm-up (Trump monitor, asyncpg pool, optional PaperAgent).
HEALTHCHECK --interval=30s --timeout=8s --start-period=90s --retries=3 \
    CMD curl -sf http://127.0.0.1:8000/health || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
