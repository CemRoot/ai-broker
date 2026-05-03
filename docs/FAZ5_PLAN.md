# Faz 5 — SaaS & open source (plan)

**Goal:** Make the advisor usable beyond a single operator while keeping the **recommendations-only** and **data/rate-limit** rules in `AI_BROKER_PROJECT.md` and `.cursor/rules`.

## Shipped in repo (first slice)

- **`WEB_UI_ENABLED`:** `GET /ui` serves a minimal page that calls `POST /ui/analyze` (same pipeline as `POST /internal/analyze`).
- **`INTERNAL_API_KEY` (optional):** when set, all `/internal/*` routes require header `X-Internal-Api-Key`. Scripted clients use `/internal/*`; the bundled browser UI uses `/ui/analyze` (no key in JS).
- **License:** `LICENSE` (MIT), `CONTRIBUTING.md`.
- **Health:** `GET /health` includes `web_ui.enabled`.

## Security note

Internal HTTP routes (`/internal/*`) are aimed at **trusted networks** (local dev, tunnel, or reverse proxy). For any public host:

- Terminate TLS and add **authentication** at the proxy (or app-level sessions later).
- Do not expose Groq/T212/Finnhub keys to browsers; the current UI only triggers server-side analysis.

## Remaining Faz 5 themes (not implemented here)

| Theme | Direction |
|-------|-----------|
| **Multi-user** | Auth (e.g. Supabase Auth), row-level isolation for preferences and optional per-user rate limits. |
| **Per-user T212** | Store encrypted or vault-referenced API key pairs per user; still **HTTP Basic** to Trading 212; demo vs live host per CEO policy. |
| **Freemium** | Feature flags + metering (e.g. extended technical, news batch caps) — product decision with CEO. |
| **GitHub publication** | Remove secrets from history, enable CI, tag releases; repo may already be public with LICENSE attached. |

## Changelog

Record phase progress in `CHANGELOG.md` with timestamps; keep this file as a high-level roadmap only.
