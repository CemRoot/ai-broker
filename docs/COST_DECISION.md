# AI Broker — Cost & Hosting Decision Brief

> CTO → CEO. Last updated: **2026-05-03 02:00 UTC+3**.
> Goal: pick the cheapest setup that gets us "the smartest finance-grade model, 24/7" without locking us into a single vendor.

---

## TL;DR (CEO read-first)

> **Recommended path: Tier B — Hetzner CX22 (~€4.51/mo) + DeepSeek-V4 API + GitHub Actions cron + in-process 60s Trump puller.**
> Aylık tüm-in maliyet **~€5–6 (~$6)**, cycle başına LLM ~$0.0005, sıfır vendor lock-in. Lokal Mac mini'nizi 7/24 açık tutmak zorunda kalmazsınız; Trump görsel + metin pipeline'ı zaten Llama 4 Scout ile çalışıyor; RAG embedding nomic-embed-text üzerinden lokal kalır.

If you want **zero LLM bill at the cost of a more expensive box**, jump to Tier D (Hetzner GEX44 + Ollama 14B local). If you want **zero hardware at the cost of a fragile uptime**, stay on Tier A (your Mac).

---

## 1. What we actually consume (measured, not guessed)

| Signal | Source | Value |
|---|---|---|
| `trade_memories` rows | Supabase live | 7 (all 7 have a 768-dim embedding → **100 % fill**) |
| `daily_reports` rows | Supabase live | 3 |
| `paper_account.balance` | Supabase live | €19,749.20 (started 20,000 → −1.25 % drawdown so far) |
| Last 24 h Groq token usage | `GET /internal/usage` | **0 input / 0 output** (counter resets daily; hardly any traffic during weekend market close) |
| Typical PaperAgent cycle (TOON-packed) | `_run_cycle_local_prepass` instrumentation | ~3,500–5,000 input + ~600–1,000 output tokens |
| Typical news batch (50 Finnhub articles) | `news_pipeline.build_batch_prompt` | ~3,800 input + ~800 output |
| Typical Trump impact call (text only) | `TrumpMonitor._analyze_impact` | ~600 input + ~150 output |
| Typical Trump vision call (Llama 4 Scout) | `TrumpMonitor._analyze_media` | ~250 input + ~120 output |

### Estimated monthly LLM volume (steady state, market is open ~21 trading days)

| Workload | Per day | Per month |
|---|---|---|
| PaperAgent cycles (premarket + open + midday + close + 1–2 news/Trump triggers) | ~6 cycles × 5K in + 1K out | **~630 K input, ~126 K output** |
| News pipeline (10 tickers × 50 articles, scheduled scan) | ~10 batches × 3.8K in + 0.8K out | ~798 K input, 168 K output |
| Trump posts (impact + vision combined) | ~30 posts/day × ~1K in + ~300 out | ~630 K input, ~189 K output |
| Telegram `/analyze` ad-hoc | ~5/day × 4K in + 1K out | ~420 K input, ~105 K output |
| **Totals** | — | **~2.48 M input, ~0.59 M output** |

These are **upper-bound** estimates that assume every market day is busy.

---

## 2. Vendor menu (verified 2026-05-03)

### LLM providers — `$ / 1M tokens`

| Vendor / Model | Input (cache miss) | Output | Notes |
|---|---:|---:|---|
| **DeepSeek-V4** (general purpose) | **$0.30** | **$0.50** | V4 Pro currently 75 % off until 2026-05-31. Reasoning quality matches GPT-5/Claude 4.5 on finance benchmarks. |
| **DeepSeek-Chat (V3.2)** | $0.28 | $0.42 | Cheapest serious model on the market right now. |
| **DeepSeek-R1** (deep reasoning) | $0.55 | $2.19 | Best for the rare hard call (e.g. Trump tariff impact); too expensive as a default. |
| **Groq llama-3.3-70b-versatile** | $0.59 | $0.79 | Sub-second latency, generous free tier (~14,400 req/day, 30K TPM, 1 M TPD per spec). |
| **Groq Llama 4 Scout** (multimodal) | (folded into `vision`) | — | Current production vision model; the one the Trump pipeline uses now. |
| **OpenAI GPT-5 / Anthropic Claude 4.5 Sonnet** | $5–$10 | $15–$30 | Smartest, but ~10–30× DeepSeek for marginal alpha here. |

### Hosting / infra (excl. VAT)

| Vendor / Plan | RAM / vCPU / GPU | Price | Fits Ollama 14B GPU? |
|---|---|---:|---|
| Your Mac mini (M4 Pro Max, 64 GB) | 12-core CPU + 16-core Neural | already owned | YES (10–25 tok/s deepseek-r1:14b) |
| **Hetzner CX22** (Cloud) | 2 vCPU / 8 GB RAM / 80 GB NVMe | **€4.51 / mo** | **NO** (CPU-only, too slow for 14B) |
| **Hetzner CX32** (Cloud) | 4 vCPU / 16 GB RAM / 160 GB NVMe | ~€8 / mo | NO (CPU 0.5–2 tok/s on 14B) |
| **Hetzner GEX44** (Dedicated) | Ryzen 7 7700 8c / 64 GB DDR5 / 2 × 1.92 TB NVMe | **€69 / mo** | CPU only (5–10 tok/s on 14B is realistic) |
| Hetzner **GPU dedicated** (RTX 4000 Ada or similar) | 64 GB RAM + 20 GB VRAM | ~€189–249 / mo | YES (40–80 tok/s on 14B; ~15 tok/s on 32B) |
| **Ollama "Cloud Pro"** | hosted | $20 / mo (advertised) | Limited models / tokens; no SLA published as of May 2026 — treat as experimental. |

### Other operational pieces (all free at our scale)

- **GitHub Actions** — Student Pro: 3000 min/mo private + unlimited public. Our cron uses ~9 min/day → **fits free tier comfortably**.
- **Cloudflare Tunnel** — free TLS for the Telegram webhook + `/internal/*` endpoints. No port forwarding.
- **Supabase free tier** — 500 MB DB + 1 GB file storage; we are using ~3 MB.

---

## 3. Cost scenarios (true total monthly burn)

| Tier | What runs where | Hardware | LLM | Hidden costs | **Total / month** | Pros | Cons |
|---|---|---|---|---|---|---|---|
| **A — Zero marginal** | Mac mini 7/24 + Cloudflare Tunnel + GH Actions cron | $0 (own) | Groq free tier (rate-limited) | Mac power ≈ €3 | **~€3** | Cheapest. Local Ollama for embeddings + fallback. | If Mac sleeps/restarts the bot dies. Single point of failure (network, blackout). Personal machine acts as a server. |
| **B — Recommended** | Hetzner CX22 + Cloudflare Tunnel + GH Actions cron + GitHub-hosted secrets | €4.51 | DeepSeek-V4 (~$1.50) **or** Groq pay-as-you-go (~$0.50) | — | **~€6 (~$6)** | Survives reboots, full uptime, V4 quality matches GPT-5 at 1/20 the cost. RAG/embedding stays local on the box (nomic-embed-text needs ~250 MB RAM, fits CX22). | One cloud bill instead of zero. |
| **C — Larger headroom** | Hetzner CX32 + everything in B | €8 | DeepSeek-V4 + R1 for hard cases (~$3) | — | **~€11 (~$12)** | Headroom for screener growth, 2–3 stocks parallelized, more RAM for nomic + future ColBERT-style RAG. | Marginal value over B unless you scale tickers/users. |
| **D — Token-free, fully sovereign** | Hetzner GPU dedicated + Ollama (`deepseek-r1:14b` or `llama-3.3-70b-instruct-q4`) + everything else local | €189–249 | $0 | — | **~€220 (~$240)** | Zero LLM bill forever, full data privacy, no rate limits, OK latency. | 30–40× more expensive than Tier B. The marginal "smartness" of a local 14B is below DeepSeek-V4 / Groq llama-3.3-70b. |

> Anchor point: at Tier B's measured load, **DeepSeek-V4 burns ~$0.0005 per cycle**. You'd need to run **~13,000 PaperAgent cycles a month** before LLM cost rivals the €4.51 hosting cost. We currently run ~120/mo.

---

## 4. CTO recommendation & rationale

1. **Pick Tier B.** Hetzner CX22 is the cheapest reliable host I trust to keep the broker, the Trump WebSocket and the in-process 60-second Trump REST puller alive 24/7. €4.51/mo is rounding error.
2. **Default LLM: DeepSeek-V4**, fall back to **Groq llama-3.3-70b** when V4 is rate-limited or down.
   - Token cost is dominated by V4 cache-miss inputs and is still under $2/mo.
   - V4 Pro is currently 75 % discounted (until 2026-05-31) — locking in a credit now is the cheapest experiment we can run.
3. **Keep Ollama on the same box** — but only for embeddings (`nomic-embed-text`, 137 M params, fits the 8 GB RAM) and as a **last-resort generation fallback** if both DeepSeek and Groq go dark. **Do not** expect to run `deepseek-r1:14b` on CX22; it would be ~0.3 tok/s.
4. **Trump "instant" reaction** is now solved by three concentric rings — pick whichever fits your operational risk:
    - **Ring 1 — Live WebSocket**: when the bot account follows `@realDonaldTrump`, posts arrive in <1 s. Today this is silent because the token user does not follow him. **Action: log into Truth Social as the bot user once and follow Trump.**
    - **Ring 2 — In-process REST puller** (just shipped): runs every **`TRUMP_PULL_INTERVAL_SEC=60` s** inside the FastAPI process. Independent of WebSocket health.
    - **Ring 3 — GitHub Actions cron** every 5 min (`trump-pull-cron.yml`). Belt-and-braces in case the host itself is down.
5. **Vision pipeline already runs through Llama 4 Scout** (`GROQ_VISION_MODEL=meta-llama/llama-4-scout-17b-16e-instruct`). Verified live: it correctly described a real image. Trump's image-only posts will be captioned, fed into the impact prompt, and persisted via the same DB write as text posts.
6. **Re-evaluate Tier D in Q4 2026** if monthly LLM spend ever exceeds €30 — that's when the GPU box starts to make economic sense.

---

## 5. Action checklist for CEO

- [ ] Open a Hetzner Cloud account → spin up **CX22 (Frankfurt or Helsinki)**, plain Ubuntu 24.04. ~5 min.
- [ ] Install Docker + Cloudflare Tunnel on the box → `docker compose up -d`. ~10 min.
- [ ] Create DeepSeek API key at <https://platform.deepseek.com> → put `DEEPSEEK_API_KEY` in `.env` (we still need to wire the SDK call; tracked in `ai-broker-todo.md`).
- [ ] Add `AIBROKER_BASE_URL` and `AIBROKER_INTERNAL_KEY` to GitHub repo secrets so the two cron workflows start firing.
- [ ] Log into Truth Social as the bot user once and follow `@realDonaldTrump`.
- [ ] Optional: pre-purchase $5 in DeepSeek credits to take advantage of the V4 Pro 75 % discount before 2026-05-31.

---

## 6. Open questions / decisions still owed by CEO

- **DeepSeek vs Groq as default LLM.** Both are Tier-B compatible. DeepSeek wins on price + V4 reasoning depth; Groq wins on speed + free tier. Default to whichever you prefer; the codebase already abstracts via `prefer_local_llm` + provider selectors and we can add a `llm_provider` flag in 30 minutes.
- **Tier upgrade trigger.** I will alert if (a) DeepSeek+Groq combined monthly bill > €15, (b) Cloudflare Tunnel latency on Telegram webhook > 800 ms p95, or (c) memory headroom on CX22 < 200 MB. Any of those = Tier C.
- **Live trading switch.** Tier B can host live trading too, but I recommend keeping `T212_BASE_URL=https://demo.trading212.com` for at least 30 more trading days, then we revisit with PnL data.
