"""
Faz 1.5-d — Truth Social (Mastodon-compatible) Trump post monitor.

Streams the authenticated user's home timeline via SSE or WebSocket (Truth fork dependent),
filters posts by configured Trump account id, runs Groq impact + optional vision analysis,
logs rows (Supabase insert in Faz 2), and alerts Telegram when impact exceeds threshold.

Requires ``TRUTH_SOCIAL_ACCESS_TOKEN`` (Bearer). Email/password are reserved for future OAuth;
token must be obtained via Truth Social / Mastodon OAuth app flow.

The ``stream=user`` timeline includes accounts the token user **follows** — follow @realDonaldTrump
from that account or posts may not appear.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import httpx

from app.core.config import Settings
from app.core.logging import get_logger
from app.services.llm.groq_service import GroqService

if TYPE_CHECKING:
    from telegram.ext import Application

log = get_logger("trump_monitor")

IMPACT_SYSTEM = """You are a markets-focused assistant. Given a social post by a political figure, respond with ONE JSON object only (no markdown), keys:
{"impact_score": <float 1-10>, "sentiment": "bullish"|"bearish"|"neutral", "affected_sectors": [<strings>], "affected_tickers": [<uppercase tickers or empty>], "reasoning": <short string>}
Scores 8-10 = likely major market-moving; 4-7 = moderate; 1-3 = noise."""

VISION_PROMPT = (
    "Briefly describe any charts, tickers, logos, or economic imagery in this image "
    "that could matter for traders (one or two sentences)."
)


def _extract_json_object(text: str) -> dict[str, Any]:
    """Best-effort JSON object extraction from LLM output."""
    text = text.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {}


@dataclass
class TrumpPostRecord:
    post_id: str
    post_text: str
    image_analysis: str | None
    posted_at: datetime
    impact_score: float
    sentiment: str
    affected_sectors: list[str]
    affected_tickers: list[str]
    reasoning: str
    telegram_sent: bool = False


class TrumpMonitor:
    """Background Truth Social stream consumer."""

    def __init__(
        self,
        settings: Settings,
        groq: GroqService | None,
        http_client: httpx.AsyncClient | None,
        telegram_application: Application | None,
        db=None,
        retriever=None,
    ) -> None:
        self._settings = settings
        self._groq = groq
        self._http = http_client
        self._telegram_app = telegram_application
        self._db = db
        self._retriever = retriever
        self._paper_agent = None
        self._seen_ids: set[str] = set()
        self._trump_account_id: str | None = None
        self._backoff = 5.0

    @property
    def _api_base(self) -> str:
        return f"{self._settings.truth_social_base_url.rstrip('/')}/api/v1"

    def _auth_headers(self) -> dict[str, str]:
        tok = self._settings.truth_social_access_token.strip()
        return {"Authorization": f"Bearer {tok}"}

    def _truth_user_agent_headers(self) -> dict[str, str]:
        # Cloudflare/WAF often blocks default Python clients; mimic a modern browser.
        return {
            "Accept": "application/json",
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
        }

    async def _truth_get_json(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        accept: str = "application/json",
        timeout_s: float = 20.0,
    ) -> tuple[int, dict[str, Any] | None, str]:
        """
        Cloudflare-aware JSON GET for Truth Social endpoints.

        Returns: (status_code, json_dict_or_none, raw_text_prefix)
        """
        url = f"{self._api_base}{path}"
        headers = {
            **self._truth_user_agent_headers(),
            **self._auth_headers(),
            "Accept": accept,
        }

        # Prefer curl_cffi (curl-impersonate) to bypass WAF.
        try:
            from curl_cffi import requests as crequests  # type: ignore

            def _do() -> tuple[int, dict[str, Any] | None, str]:
                r = crequests.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=timeout_s,
                    impersonate="chrome",
                )
                text = (r.text or "")[:500]
                data: dict[str, Any] | None = None
                try:
                    data = r.json()
                except Exception:
                    data = None
                return int(r.status_code), data, text

            return await asyncio.to_thread(_do)
        except Exception:
            pass

        # Fallback: httpx (may be blocked by Cloudflare on some endpoints).
        if not self._http:
            return 0, None, "http client unavailable"
        try:
            r = await self._http.get(
                url,
                params=params,
                headers=headers,
                timeout=httpx.Timeout(timeout_s),
            )
            text = (r.text or "")[:500]
            data: dict[str, Any] | None = None
            try:
                data = r.json()
            except Exception:
                data = None
            return int(r.status_code), data, text
        except Exception as exc:
            return 0, None, str(exc)[:500]

    async def connect(self) -> None:
        """Resolve Trump account id for filtering (REST lookup)."""
        tok = self._settings.truth_social_access_token.strip()
        if not tok:
            log.warning("TrumpMonitor: TRUTH_SOCIAL_ACCESS_TOKEN empty — streaming disabled")
            return
        aid = await self._lookup_trump_account_id()
        if aid:
            self._trump_account_id = aid
            log.info("TrumpMonitor: resolved Trump account id=%s", aid)
        else:
            log.warning(
                "TrumpMonitor: could not resolve account %r — posts may not filter correctly",
                self._settings.trump_truth_account_username,
            )

    async def _lookup_trump_account_id(self) -> str | None:
        status, data, text = await self._truth_get_json(
            "/accounts/lookup",
            params={"acct": self._settings.trump_truth_account_username.strip()},
        )
        if status != 200 or not isinstance(data, dict):
            log.warning("TrumpMonitor lookup HTTP %s: %s", status, text[:200])
            return None
        return str(data.get("id", "")) or None

    async def on_trump_post(self, status: dict[str, Any]) -> None:
        """Handle one normalized Mastodon Status dict attributed to Trump."""
        sid = str(status.get("id", ""))
        if not sid:
            return
        if sid in self._seen_ids:
            return
        self._seen_ids.add(sid)
        # Bound memory
        if len(self._seen_ids) > 5000:
            self._seen_ids = set(list(self._seen_ids)[-2500:])

        content = self._status_plain_text(status)
        media_urls = self._status_media_urls(status)
        created = self._status_created_at(status)

        image_analysis: str | None = None
        if media_urls:
            image_analysis = await self._analyze_media(media_urls[0])

        impact = await self._analyze_impact(content, image_analysis)
        rec = TrumpPostRecord(
            post_id=sid,
            post_text=content[:8000],
            image_analysis=image_analysis,
            posted_at=created,
            impact_score=float(impact.get("impact_score", 0)),
            sentiment=str(impact.get("sentiment", "neutral"))[:16],
            affected_sectors=[str(x) for x in impact.get("affected_sectors", []) if x][:32],
            affected_tickers=[str(x).upper()[:12] for x in impact.get("affected_tickers", []) if x][:64],
            reasoning=str(impact.get("reasoning", ""))[:4000],
        )

        await self._save_to_db(rec)

        if rec.impact_score >= self._settings.trump_impact_threshold:
            await self._send_alert(rec)
            
        if self._retriever and rec.impact_score > 5:
            # High impact, add to global RAG memory
            mem_context = f"Trump posted: {rec.post_text[:1000]}... Impact: {rec.impact_score}. Affected sectors: {', '.join(rec.affected_sectors)}. Reasoning: {rec.reasoning}"
            for ticker in (rec.affected_tickers or ["SPY"]):
                await self._retriever.add_memory(
                    ticker=ticker,
                    memory_type="WARNING"
                    if str(rec.sentiment).lower() == "bearish"
                    else "INFO",
                    context=mem_context,
                    outcome="OPEN"
                )

        # Faz 3f: Emergency cycle hook (best-effort, no hard dependency).
        if self._paper_agent and rec.impact_score > 5:
            try:
                await self._paper_agent.run_emergency_cycle(
                    trigger="TRUMP_POST",
                    context={
                        "post_text": rec.post_text,
                        "image_analysis": rec.image_analysis,
                        "impact_score": rec.impact_score,
                        "sentiment": rec.sentiment,
                        "affected_tickers": rec.affected_tickers,
                        "affected_sectors": rec.affected_sectors,
                        "reasoning": rec.reasoning,
                    },
                )
            except Exception as exc:
                log.warning("PaperAgent emergency cycle failed: %s", exc)

    def set_paper_agent(self, paper_agent) -> None:
        """Attach a PaperAgent instance (Faz 3f)."""
        self._paper_agent = paper_agent

    async def pull_recent_statuses(self, *, limit: int = 10) -> dict[str, int]:
        """REST polling fallback used by ``POST /internal/trump/pull`` and the
        GitHub Actions cron worker.

        The ``stream=user`` WebSocket only carries posts authored by accounts the
        token user follows. If the operator has not followed ``@realDonaldTrump``
        from the bot account, posts never arrive on the live stream — this helper
        polls the public statuses endpoint and feeds each new status through the
        same ``on_trump_post`` pipeline (impact LLM + Supabase write + Telegram).

        Returns a small summary dict suitable for HTTP responses and CI logs.
        """
        if not self._trump_account_id:
            await self.connect()
        aid = self._trump_account_id
        if not aid:
            return {"fetched": 0, "new_posts": 0, "error_status": -1}
        status, data, text = await self._truth_get_json(
            f"/accounts/{aid}/statuses",
            params={"limit": int(max(1, min(40, limit))), "exclude_replies": "true"},
        )
        if status != 200 or not isinstance(data, list):
            log.warning("TrumpMonitor pull HTTP %s: %s", status, text[:160])
            return {"fetched": 0, "new_posts": 0, "error_status": status}
        new_count = 0
        for st in data:
            if not isinstance(st, dict):
                continue
            sid = str(st.get("id") or "")
            before = sid in self._seen_ids
            try:
                await self.on_trump_post(st)
            except Exception as exc:
                log.warning("TrumpMonitor pull on_trump_post error: %s", exc)
                continue
            if not before:
                new_count += 1
        return {"fetched": len(data), "new_posts": new_count, "error_status": status}

    def _status_plain_text(self, status: dict[str, Any]) -> str:
        # Prefer spoiler-free content; Mastodon uses HTML sometimes
        raw = status.get("content") or ""
        if isinstance(raw, str) and raw.startswith("<"):
            # Minimal strip tags
            raw = re.sub(r"<[^>]+>", " ", raw)
        return " ".join(raw.split())

    def _status_created_at(self, status: dict[str, Any]) -> datetime:
        s = status.get("created_at") or ""
        try:
            # ISO8601 e.g. 2025-01-01T12:00:00.000Z
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except Exception:
            return datetime.now(timezone.utc)

    def _status_media_urls(self, status: dict[str, Any]) -> list[str]:
        out: list[str] = []
        for att in status.get("media_attachments") or []:
            if not isinstance(att, dict):
                continue
            url = att.get("url") or att.get("preview_url")
            if url and att.get("type") == "image":
                out.append(str(url))
        return out

    async def _analyze_media(self, image_url: str) -> str | None:
        if not self._groq:
            log.warning("TrumpMonitor: Groq disabled — skipping vision")
            return None
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ]
        try:
            resp = await self._groq.analyze_multimodal(
                messages,
                model=self._settings.groq_vision_model,
            )
            return resp.text[:4000] if resp.text else None
        except Exception as exc:
            log.warning("TrumpMonitor vision failed: %s", exc)
            return None

    async def _analyze_impact(self, text: str, image_analysis: str | None) -> dict[str, Any]:
        if not self._groq:
            return {"impact_score": 0, "sentiment": "neutral", "reasoning": "Groq disabled"}
        body = f"Post text:\n{text[:6000]}\n"
        if image_analysis:
            body += f"\nImage notes:\n{image_analysis[:2000]}\n"
        try:
            resp = await self._groq.analyze(body, system=IMPACT_SYSTEM)
            data = _extract_json_object(resp.text)
            # Normalize lists
            if "affected_sectors" not in data:
                data["affected_sectors"] = []
            if "affected_tickers" not in data:
                data["affected_tickers"] = []
            return data
        except Exception as exc:
            log.warning("TrumpMonitor impact LLM failed: %s", exc)
            return {"impact_score": 0, "sentiment": "neutral", "reasoning": str(exc)}

    async def _save_to_db(self, rec: TrumpPostRecord) -> None:
        """Persist row to Supabase."""
        payload = {
            "post_id": rec.post_id,
            "post_text": rec.post_text[:500],
            "image_analysis": rec.image_analysis,
            "posted_at": rec.posted_at.isoformat(),
            "impact_score": rec.impact_score,
            "sentiment": rec.sentiment,
            "affected_sectors": (rec.affected_sectors or [])[:32],
            "affected_tickers": (rec.affected_tickers or [])[:64],
            "reasoning": rec.reasoning[:500],
            "telegram_sent": rec.telegram_sent,
        }
        log.info("Saving Trump post: %s", json.dumps(payload, ensure_ascii=False))
        
        if self._db and self._db.get_pool():
            try:
                query = """
                INSERT INTO trump_posts 
                (post_id, post_text, image_analysis, posted_at, impact_score, sentiment, affected_sectors, affected_tickers, reasoning, telegram_sent)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (post_id) DO NOTHING
                """
                async with self._db.get_pool().acquire() as conn:
                    await conn.execute(
                        query,
                        rec.post_id,
                        rec.post_text,
                        rec.image_analysis,
                        rec.posted_at,
                        rec.impact_score,
                        rec.sentiment,
                        (rec.affected_sectors or [])[:32],
                        (rec.affected_tickers or [])[:64],
                        rec.reasoning,
                        rec.telegram_sent
                    )
            except Exception as exc:
                log.error("Failed to save Trump post to DB: %s", exc)

    async def _send_alert(self, rec: TrumpPostRecord) -> None:
        if not self._telegram_app:
            log.info("TrumpMonitor: Telegram bot offline — alert skipped")
            return
        allowed = self._settings.allowed_user_ids
        if not allowed:
            log.warning("TrumpMonitor: TELEGRAM_ALLOWED_USER_IDS empty — alert skipped")
            return

        lines = [
            "Trump post — impact alert",
            f"Score: {rec.impact_score:.1f} (threshold {self._settings.trump_impact_threshold})",
            f"Sentiment: {rec.sentiment}",
            f"Sectors: {', '.join(rec.affected_sectors[:8]) or '—'}",
            f"Tickers: {', '.join(rec.affected_tickers[:12]) or '—'}",
            "",
            rec.post_text[:2800],
        ]
        if rec.reasoning:
            lines.extend(["", "Reasoning:", rec.reasoning[:1200]])
        text = "\n".join(lines)
        bot = self._telegram_app.bot
        for uid in allowed:
            try:
                await bot.send_message(chat_id=uid, text=text[:4096])
                rec.telegram_sent = True
            except Exception as exc:
                log.error("TrumpMonitor Telegram send failed uid=%s: %s", uid, exc)

    def _should_process_status(self, status: dict[str, Any]) -> bool:
        acct = status.get("account") or {}
        aid = str(acct.get("id", ""))
        uname = str(acct.get("username", "")).lower()
        target = self._settings.trump_truth_account_username.strip().lower()
        if self._trump_account_id and aid == self._trump_account_id:
            return True
        if target and uname == target.replace("@", "").split("@")[0]:
            return True
        return False

    async def _handle_sse_event(self, event_name: str | None, data_blob: str) -> None:
        ev = (event_name or "update").strip().lower()
        if ev in ("heartbeat", "ping"):
            return
        if ev != "update":
            return
        try:
            status = json.loads(data_blob)
        except json.JSONDecodeError:
            return
        if isinstance(status, dict) and self._should_process_status(status):
            await self.on_trump_post(status)

    async def _dispatch_ws_message(self, message: str) -> None:
        """Parse Mastodon-compatible WebSocket JSON frame."""
        try:
            obj = json.loads(message)
        except json.JSONDecodeError:
            return
        payload = obj.get("payload")
        if isinstance(payload, str):
            try:
                inner = json.loads(payload)
            except json.JSONDecodeError:
                return
            status = inner if isinstance(inner, dict) else None
        elif isinstance(payload, dict):
            status = payload
        else:
            return
        if status and isinstance(status, dict) and self._should_process_status(status):
            await self.on_trump_post(status)

    async def _consume_sse(self) -> None:
        tok = self._settings.truth_social_access_token.strip()
        if not tok:
            await asyncio.sleep(60)
            return

        url = f"{self._settings.truth_social_base_url.rstrip('/')}/api/v1/streaming"
        headers = {
            **self._auth_headers(),
            **self._truth_user_agent_headers(),
            "Accept": "text/event-stream",
        }
        log.info("TrumpMonitor: SSE connect %s stream=user", url)

        # Prefer curl_cffi for SSE to bypass Cloudflare.
        try:
            from curl_cffi import requests as crequests  # type: ignore

            q: asyncio.Queue[str | None] = asyncio.Queue(maxsize=2000)
            stop = threading.Event()
            loop = asyncio.get_running_loop()

            def _pump() -> None:
                try:
                    with crequests.Session() as s:
                        r = s.get(
                            url,
                            params={"stream": "user"},
                            headers=headers,
                            stream=True,
                            timeout=60,
                            impersonate="chrome",
                        )
                        try:
                            if int(getattr(r, "status_code", 0)) != 200:
                                try:
                                    body = (r.text or "")[:200]
                                except Exception:
                                    body = ""
                                loop.call_soon_threadsafe(q.put_nowait, None)
                                log.warning("TrumpMonitor SSE HTTP %s: %s", r.status_code, body)
                                return
                            for raw in r.iter_lines():
                                if stop.is_set():
                                    break
                                if raw is None:
                                    continue
                                line = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
                                def _offer(v: str) -> None:
                                    if q.full():
                                        # Drop lines under sustained backpressure rather than blocking the pump.
                                        return
                                    q.put_nowait(v)

                                loop.call_soon_threadsafe(_offer, line)
                        finally:
                            try:
                                r.close()
                            except Exception:
                                pass
                except Exception as exc:
                    log.warning("TrumpMonitor SSE (curl_cffi) failed: %s", exc)
                    try:
                        loop.call_soon_threadsafe(q.put_nowait, None)
                    except Exception:
                        pass

            t = threading.Thread(target=_pump, daemon=True)
            t.start()

            event_name: str | None = None
            data_lines: list[str] = []
            while True:
                line = await q.get()
                if line is None:
                    break
                if line.startswith("event:"):
                    event_name = line[6:].strip()
                    continue
                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
                    continue
                if line == "" and data_lines:
                    blob = "\n".join(data_lines)
                    data_lines.clear()
                    await self._handle_sse_event(event_name, blob)
                    event_name = None
            stop.set()
            return
        except Exception:
            pass

        # Fallback to httpx streaming (may be blocked by Cloudflare).
        if not self._http:
            await asyncio.sleep(60)
            return
        async with self._http.stream(
            "GET",
            url,
            params={"stream": "user"},
            headers=headers,
            timeout=httpx.Timeout(None, connect=30.0),
        ) as resp:
            resp.raise_for_status()
            event_name: str | None = None
            data_lines: list[str] = []

            async for raw_line in resp.aiter_lines():
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if line is None:
                    continue
                if line.startswith("event:"):
                    event_name = line[6:].strip()
                    continue
                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
                    continue
                if line == "" and data_lines:
                    blob = "\n".join(data_lines)
                    data_lines.clear()
                    await self._handle_sse_event(event_name, blob)
                    event_name = None

    async def _consume_websocket(self) -> None:
        try:
            from curl_cffi.requests import AsyncSession
        except ImportError:
            log.warning("TrumpMonitor WebSocket requires curl_cffi")
            await asyncio.sleep(60)
            return

        tok = self._settings.truth_social_access_token.strip()
        if not tok:
            await asyncio.sleep(60)
            return

        host = self._settings.truth_social_base_url.rstrip("/").replace("https://", "").replace("http://", "")
        uri = f"wss://{host}/api/v1/streaming?stream=user"
        log.info("TrumpMonitor: WebSocket connect (user stream) via curl_cffi")

        headers = {"Authorization": f"Bearer {tok}"}
        try:
            async with AsyncSession(impersonate="chrome", timeout=60.0) as session:
                async with session.ws_connect(uri, headers=headers) as ws:
                    while True:
                        message = await ws.recv()
                        if isinstance(message, tuple):
                            # curl_cffi ws.recv() returns a tuple (content, opcode)
                            message = message[0]
                        if isinstance(message, bytes):
                            message = message.decode("utf-8", errors="replace")
                        await self._dispatch_ws_message(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Re-raise so run_with_reconnect's outer except applies exponential
            # backoff; otherwise a Cloudflare 403 would loop instantly and flood
            # logs with hundreds of WARNINGs per second.
            log.warning("TrumpMonitor WebSocket failed: %s", exc)
            raise

    async def run_with_reconnect(self) -> None:
        """Long-running loop with exponential backoff + jitter."""
        await self.connect()

        while True:
            try:
                if not self._settings.truth_social_access_token.strip():
                    log.warning(
                        "TrumpMonitor idle — set TRUTH_SOCIAL_ACCESS_TOKEN (email/password OAuth not automated yet)"
                    )
                    await asyncio.sleep(300)
                    continue

                transport = self._settings.truth_social_stream_transport.strip().lower()
                if transport == "websocket":
                    await self._consume_websocket()
                else:
                    await self._consume_sse()

                self._backoff = 5.0

            except asyncio.CancelledError:
                log.info("TrumpMonitor cancelled")
                raise
            except Exception as exc:
                log.exception("TrumpMonitor stream error: %s", exc)
                jitter = random.uniform(0, min(self._backoff * 0.2, 30))
                await asyncio.sleep(self._backoff + jitter)
                self._backoff = min(self._backoff * 2, 900)
