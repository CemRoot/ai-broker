# Trading 212 — LIVE switch (read me before flipping)

> CEO directive (`Faz 4 / 2026-05`): the bot must run autonomously without per-trade
> approval. By default it points at the **demo** host with demo API keys; this
> document describes how to flip it to **LIVE**.
>
> **Live trading risks real money.** Read every checkbox below before changing
> `T212_BASE_URL` on the VPS.

## 1. Pre-flight checklist

- [ ] PaperAgent has been running on **DEMO** for at least one full trading day with
      no fatal errors in `docker compose logs ai-broker | grep -iE "error|traceback"`.
- [ ] Last `/paper stats` shows a profile you accept (win rate, avg R/R, max DD).
- [ ] You have re-read **`AI_BROKER_PROJECT.md`** *Faz 0 — Kapanış kararları* — the
      live host requires a **separate** API key + secret pair (different from demo).
- [ ] You have set `PAPER_MAX_DRAWDOWN_PCT` in `.env` to a value that matches your
      LIVE risk tolerance (default 15%; below that PaperAgent halts new BUYs).
- [ ] You have decided on a *kill switch* shortcut — `PAPER_AGENT_ENABLED=false`
      in `.env` + `docker compose up -d --force-recreate ai-broker` stops the loop
      cleanly without touching open positions.

## 2. Get LIVE credentials from Trading 212

1. Open the **Trading 212 mobile/web app → Settings → API → Create API key**.
2. Pick the **LIVE** account (separate keys are issued for Demo vs Live).
3. Copy the **API key** and **API secret** (the secret is shown once).
4. Whitelist the VPS public IP if prompted (Oracle OCI ARM IP from `~/.ssh/config`).

## 3. Switch the VPS `.env`

SSH to the VPS and edit `/home/ubuntu/ai-broker/.env`:

```bash
ssh ai-broker
cd ~/ai-broker
cp .env .env.backup-$(date +%F)   # rollback safety net

# Replace these 3 lines:
T212_BASE_URL=https://live.trading212.com
T212_DEMO_API_KEY=<paste-LIVE-key-here>      # name kept for compat; value is LIVE
T212_DEMO_API_SECRET=<paste-LIVE-secret-here>
```

Then **force-recreate** so the env is re-read (a plain `restart` does not):

```bash
docker compose up -d --force-recreate ai-broker
```

Verify the client is talking to the LIVE host:

```bash
docker compose logs --since 1m ai-broker | grep -iE "T212|t212"
# expect: "T212 client initialised host=https://live.trading212.com"
```

## 4. Smoke-test before the next market open

```bash
# Use the internal API key set in .env (auto-generated, see CHANGELOG 2026-05-04 02:30)
curl -s -H "X-Internal-Api-Key: $(grep '^INTERNAL_API_KEY=' .env | cut -d= -f2)" \
     https://broker.cemkoyluoglu.codes/internal/positions | jq .
# Should now reflect your LIVE T212 positions, NOT the demo ones.
```

## 5. Rollback to DEMO

If anything looks wrong, restore from the backup:

```bash
ssh ai-broker
cd ~/ai-broker
cp .env.backup-YYYY-MM-DD .env
docker compose up -d --force-recreate ai-broker
```

## 6. Operating notes for LIVE

- Order endpoints (`/equity/orders/*`) are **non-idempotent in the public beta**;
  the client never auto-retries POST orders on transient errors — failures bubble
  up as `T212 BUY/SELL [TYPE] failed` log lines and the trade is *not* recorded.
  This is intentional: a duplicate live BUY would be expensive to unwind.
- Rate limits are global at the client (≥ 2.05 s between calls). With 4 order
  variants + portfolio polling, this caps the loop at ≈ 28 calls/minute, safely
  inside the documented buckets.
- If T212 returns HTTP 401 the auth header is wrong (typo in key/secret); the bot
  will not silently keep retrying — fix `.env` and force-recreate.
- Telegram notifications include the **T212 order id** in the footer (`#T212-…`)
  so you can cross-check fills against the T212 order history page.
