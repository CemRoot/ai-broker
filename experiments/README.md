# Experiments

**Phase 0 only.** Benchmark scripts compare inference backends outside the shipped FastAPI app path — canonical stack decisions live in `AI_BROKER_PROJECT.md` and `.cursor/rules/ai-broker-core.mdc`.

| Path | Purpose |
|------|---------|
| `model_race/faz0_test.py` | Groq / Ollama / optional Cerebras latency + JSON discipline smoke; writes `results/faz0_results.json` |
| `model_race/faz0_embed_test.py` | `nomic-embed-text` cosine sanity (768-dim) |

Run from repo root with `uv sync --all-extras` as needed; never wire these into production `app.main` lifespans.
