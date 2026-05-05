"""
Faz 1.5-d — Trump monitor (CNN Truth archive source).

Fetches ``https://ix.cnn.io/data/truth-social/truth_archive.json`` on an interval,
analyzes post text/media with existing Groq pipelines, writes rows to DB, and sends
Telegram alerts for high-impact posts.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
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
CNN_TRUTH_ARCHIVE_URL = "https://ix.cnn.io/data/truth-social/truth_archive.json"

IMPACT_SYSTEM = """You are a markets-focused assistant. Given a social post by a political figure, respond with ONE JSON object only (no markdown), keys:
{"impact_score": <float 1-10>, "sentiment": "bullish"|"bearish"|"neutral", "affected_sectors": [<strings>], "affected_tickers": [<uppercase tickers or empty>], "reasoning": <short string>}
Scores 8-10 = likely major market-moving; 4-7 = moderate; 1-3 = noise."""

VISION_PROMPT = (
    "Briefly describe any charts, tickers, logos, or economic imagery in this image "
    "that could matter for traders (one or two sentences)."
)


def _mastodon_html_to_plain(raw: str) -> str:
    """Strip minimal HTML from Mastodon/Truth ``content`` fields."""
    if not isinstance(raw, str):
        return ""
    s = raw.strip()
    if not s:
        return ""
    if s.startswith("<"):
        s = re.sub(r"<[^>]+>", " ", s)
    return " ".join(s.split()).strip()


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
        self._last_processed_post_id: str | None = None
        self._backoff = 5.0

    async def _fetch_cnn_archive(self) -> tuple[int, list[dict[str, Any]], str]:
        """Fetch CNN Truth archive and normalize output to a post list."""
        if not self._http:
            return 0, [], "http client unavailable"
        try:
            resp = await self._http.get(
                CNN_TRUTH_ARCHIVE_URL,
                timeout=httpx.Timeout(20.0, connect=5.0),
            )
            code = int(resp.status_code)
            text = (resp.text or "")[:500]
            if code != 200:
                return code, [], text
            payload = resp.json()
        except Exception as exc:
            return 0, [], str(exc)[:500]

        if isinstance(payload, list):
            posts = [p for p in payload if isinstance(p, dict)]
            return 200, posts, ""
        if isinstance(payload, dict):
            # endpoint may return either one post or wrapped list
            if isinstance(payload.get("posts"), list):
                posts = [p for p in payload["posts"] if isinstance(p, dict)]
                return 200, posts, ""
            if payload.get("id"):
                return 200, [payload], ""
        return 200, [], ""

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

    async def _load_last_processed_post_id(self) -> str | None:
        """Best-effort seed from DB so restarts don't reprocess old posts."""
        if not self._db or not self._db.get_pool():
            return None
        try:
            query = """
            SELECT post_id
            FROM trump_posts
            ORDER BY posted_at DESC
            LIMIT 1
            """
            async with self._db.get_pool().acquire() as conn:
                row = await conn.fetchrow(query)
            if row and row.get("post_id"):
                return str(row["post_id"])
        except Exception as exc:
            log.warning("TrumpMonitor: failed loading last post_id from DB: %s", exc)
        return None

    @staticmethod
    def _id_to_int(post_id: str) -> int:
        try:
            return int(post_id.strip())
        except Exception:
            return 0

    @staticmethod
    def _cnn_media_split(media: list[str]) -> tuple[list[str], list[str]]:
        images: list[str] = []
        videos: list[str] = []
        for raw in media:
            u = str(raw or "").strip()
            if not u:
                continue
            ul = u.lower().split("?", 1)[0]
            if ul.endswith((".jpg", ".jpeg", ".png", ".webp")):
                images.append(u)
            elif ul.endswith((".mp4", ".mov")):
                videos.append(u)
        return images, videos

    @staticmethod
    def _cnn_to_status(post: dict[str, Any], image_urls: list[str]) -> dict[str, Any]:
        return {
            "id": str(post.get("id") or ""),
            "created_at": str(post.get("created_at") or ""),
            "content": str(post.get("content") or ""),
            "media_attachments": [{"type": "image", "url": u} for u in image_urls],
        }

    async def pull_recent_statuses(self, *, limit: int = 10) -> dict[str, int]:
        """Poll CNN archive and feed new entries through existing analysis pipeline."""
        if self._last_processed_post_id is None:
            self._last_processed_post_id = await self._load_last_processed_post_id()

        status, posts, text = await self._fetch_cnn_archive()
        if status != 200:
            log.warning("TrumpMonitor CNN archive HTTP %s: %s", status, text[:160])
            return {"fetched": 0, "new_posts": 0, "error_status": status}

        # Keep most recent entries first if endpoint returns long arrays.
        posts = posts[: int(max(1, min(200, limit)))] if posts else []
        new_count = 0
        last_id_int = self._id_to_int(self._last_processed_post_id or "")

        for post in reversed(posts):
            sid = str(post.get("id") or "").strip()
            if not sid:
                continue
            sid_int = self._id_to_int(sid)
            if sid in self._seen_ids:
                continue
            if last_id_int and sid_int and sid_int <= last_id_int:
                continue

            content = str(post.get("content") or "").strip()
            media = post.get("media") if isinstance(post.get("media"), list) else []
            image_urls, video_urls = self._cnn_media_split([str(x) for x in media])

            for vu in video_urls:
                log.info("TrumpMonitor CNN: skipping video media url=%s", vu[:180])

            if not content and not image_urls:
                continue

            normalized = self._cnn_to_status(post, image_urls)
            before = sid in self._seen_ids
            try:
                await self.on_trump_post(normalized)
            except Exception as exc:
                log.warning("TrumpMonitor CNN on_trump_post error: %s", exc)
                continue
            if not before:
                new_count += 1
                self._last_processed_post_id = sid
                if sid_int:
                    last_id_int = sid_int
        return {"fetched": len(posts), "new_posts": new_count, "error_status": 200}

    def _status_plain_text(self, status: dict[str, Any]) -> str:
        """Plain text for DB + LLM. Boosts/reblogs often have empty top-level ``content``."""
        main = _mastodon_html_to_plain(str(status.get("content") or ""))
        reblog = status.get("reblog")
        rb_dict = reblog if isinstance(reblog, dict) else None
        rb_text = _mastodon_html_to_plain(str(rb_dict.get("content") or "")) if rb_dict else ""

        if main and rb_text:
            return f"{main}\n\n↪ {rb_text}"
        if main:
            return main
        if rb_text:
            return rb_text

        sp = str(status.get("spoiler_text") or "").strip()
        if sp:
            return sp
        if rb_dict:
            sp2 = str(rb_dict.get("spoiler_text") or "").strip()
            if sp2:
                return sp2

        if status.get("media_attachments") or (rb_dict and rb_dict.get("media_attachments")):
            return "[Media-only post — no status text]"
        return "[No text in status]"

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

    async def run_with_reconnect(self) -> None:
        """Long-running CNN archive polling loop with exponential backoff + jitter."""

        while True:
            try:
                source = (self._settings.trump_stream_source or "cnn_archive").strip().lower()
                if source != "cnn_archive":
                    log.warning("TrumpMonitor: unknown TRUMP_STREAM_SOURCE=%r, using cnn_archive", source)
                await self.pull_recent_statuses(limit=40)
                await asyncio.sleep(max(15, int(self._settings.trump_pull_interval_sec or 60)))

                self._backoff = 5.0

            except asyncio.CancelledError:
                log.info("TrumpMonitor cancelled")
                raise
            except Exception as exc:
                log.exception("TrumpMonitor stream error: %s", exc)
                jitter = random.uniform(0, min(self._backoff * 0.2, 30))
                await asyncio.sleep(self._backoff + jitter)
                self._backoff = min(self._backoff * 2, 900)
