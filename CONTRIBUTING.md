# Contributing

Thanks for your interest in AI Broker.

## Basics

- **Python:** 3.12+, package manager **`uv`** (see `README.md`).
- **Secrets:** Never commit API keys or tokens. Use `.env` locally; only variable names and placeholders belong in `.env.example`.
- **Tests:** `pytest tests/ -v` should pass before you open a PR.
- **Scope:** The product is a **recommendation-only** advisor; keep changes aligned with the canonical Turkish roadmap (`AI_BROKER_PROJECT.md`, **local / not published**) and `.cursor/rules/*.mdc` where applicable.

## Pull requests

1. Describe **what** changed and **why** (user-visible behavior, env vars, or schema if relevant).
2. Update **`CHANGELOG.md`** under `[Unreleased]` with an ISO 8601 timestamp if your change affects behavior, configuration, or architecture (see `ai-broker-changelog.mdc`).
3. If you touch canonical design or phase scope, update **`AI_BROKER_PROJECT.md`** on your side (that file is **gitignored** for public repos; coordinate with the owner if the change must be shared).

## Questions

Product and architecture trade-offs go to the project owner (CEO); when in doubt, open an issue or ask in the PR description.
