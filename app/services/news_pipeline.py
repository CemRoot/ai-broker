"""
PokieTicker Layer 1–style batch news scoring via Cerebras (Groq fallback).

Ported ideas: keyword extraction, single prompt for up to ``BATCH_SIZE`` articles,
compact JSON array output. No SQLite — stateless; persistence is Faz 2 (Supabase).

Reference: ``external/PokieTicker/backend/pipeline/layer1.py``.
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.core.logging import get_logger
from app.services.llm.cerebras_service import CerebrasService
from app.services.llm.groq_service import GroqService
from app.services.telegram_operator_alerts import fire_operator_alert, format_exc_brief

log = get_logger("news_pipeline")

BATCH_SIZE = 50
EXTRACT_THRESHOLD = 500

LAYER1_SYSTEM = (
    "You are a financial news analyst. Reply with a single valid JSON array only — "
    "no markdown fences, no commentary. Each element must have keys: "
    'i (int), r ("y"|"n"), s ("+"|"-"|"0"), e (string), u (string), d (string).'
)

# Subset of PokieTicker TICKER_KEYWORDS (extend as needed)
TICKER_KEYWORDS: dict[str, list[str]] = {
    "AAPL": ["apple", "aapl", "tim cook", "iphone", "ipad", "macbook", "ios", "macos"],
    "AMD": ["amd", "advanced micro", "lisa su", "radeon", "ryzen", "epyc"],
    "AMZN": ["amazon", "amzn", "andy jassy", "aws", "prime", "alexa"],
    "BABA": ["alibaba", "ali baba", "taobao", "tmall", "ant group"],
    "GLD": ["spdr gold", "gld", "gold trust", "gold etf"],
    "GOOGL": ["google", "alphabet", "googl", "goog", "youtube", "deepmind", "android"],
    "META": ["meta platforms", "facebook", "instagram", "whatsapp", "zuckerberg"],
    "MSFT": ["microsoft", "msft", "satya nadella", "windows", "azure", "xbox"],
    "NVDA": ["nvidia", "nvda", "jensen huang", "geforce", "cuda", "h100"],
    "TSLA": ["tesla", "tsla", "elon musk", "model 3", "model y", "cybertruck"],
}


def _get_keywords(symbol: str) -> list[str]:
    kws = [symbol.lower()]
    kws.extend(TICKER_KEYWORDS.get(symbol.upper(), []))
    return kws


def extract_relevant_text(description: str, symbol: str) -> str:
    """Short descriptions pass through; long ones keep company-relevant sentences."""
    if not description:
        return ""
    desc = description.strip()
    if len(desc) < EXTRACT_THRESHOLD:
        return desc

    keywords = _get_keywords(symbol)
    sentences = re.split(r"(?<=[.!?])\s+", desc)
    relevant: set[int] = set()
    for i, sent in enumerate(sentences):
        lower = sent.lower()
        if any(kw in lower for kw in keywords):
            for j in range(max(0, i - 1), min(len(sentences), i + 2)):
                relevant.add(j)

    if not relevant:
        return " ".join(sentences[:2])
    return " ".join(sentences[i] for i in sorted(relevant))


def _try_encode_articles_toon(symbol: str, articles: list[dict[str, Any]]) -> str | None:
    """Pack the articles as a TOON table; return None when toon-format is unavailable.

    Tabular ``articles[N]{i,title,extract}:`` rows save ~25–35% input tokens vs
    the bracketed plaintext form when the batch is large (BATCH_SIZE=50).
    """
    try:
        import toon_format
    except Exception:
        return None
    sym = symbol.upper()
    rows = []
    for i, art in enumerate(articles):
        title = (art.get("title") or "").strip()
        raw_desc = art.get("description") or art.get("summary") or ""
        extract = extract_relevant_text(str(raw_desc), sym)
        rows.append({"i": i, "title": title, "extract": extract})
    return toon_format.encode({"articles": rows})


def build_batch_prompt(
    symbol: str,
    articles: list[dict[str, Any]],
    *,
    use_toon: bool = False,
) -> str:
    """Single user prompt for up to ``BATCH_SIZE`` articles.

    Default is plaintext because empirical measurement on a 50-article AAPL batch
    (Finnhub, 2026-05-03) showed TOON only matched plaintext (-0.5%) — long
    title/extract strings dominate and the CSV-quoting overhead eats the header
    saving. ``use_toon=True`` is kept available for future numeric-heavy formats.
    """
    sym = symbol.upper()
    body: str
    if use_toon:
        toon = _try_encode_articles_toon(sym, articles)
        if toon is not None:
            body = (
                "Articles are encoded below in TOON (Token-Oriented Object Notation). "
                "Each row is one article; columns are i, title, extract.\n\n"
                f"```toon\n{toon}\n```"
            )
        else:
            body = _build_articles_plain(sym, articles)
    else:
        body = _build_articles_plain(sym, articles)

    return f"""Rate these {len(articles)} articles for {sym}. Return JSON array only.

{body}

Format: [{{"i":0,"r":"y"|"n","s":"+"|"-"|"0","e":"summary","u":"up reason","d":"down reason"}}]
r: "y" = article specifically discusses {sym}, "n" = irrelevant/brief mention
s: "+" positive, "-" negative, "0" neutral
e: ~10-word summary of what happened (empty if irrelevant)
u: why this could push {sym} stock UP (empty if none)
d: why this could push {sym} stock DOWN (empty if none)
JSON:"""


def _build_articles_plain(symbol: str, articles: list[dict[str, Any]]) -> str:
    sym = symbol.upper()
    lines: list[str] = []
    for i, art in enumerate(articles):
        title = (art.get("title") or "").strip()
        raw_desc = art.get("description") or art.get("summary") or ""
        extract = extract_relevant_text(str(raw_desc), sym)
        lines.append(f"[{i}] {title}")
        if extract:
            lines.append(f"  > {extract}")
    return "\n".join(lines)


def _strip_json_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


def parse_layer1_response(text: str, n_articles: int) -> list[dict[str, Any]]:
    """Extract JSON array from model output; tolerate extra prose."""
    raw = _strip_json_fences(text)
    start = raw.find("[")
    end = raw.rfind("]") + 1
    if start < 0 or end <= start:
        log.warning("No JSON array in model response")
        return []

    try:
        data = json.loads(raw[start:end])
    except json.JSONDecodeError:
        log.warning("JSON decode failed for layer1 response")
        return []

    if not isinstance(data, list):
        return []

    out: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        idx = item.get("i")
        if idx is None or not isinstance(idx, int) or idx < 0 or idx >= n_articles:
            continue
        out.append(item)
    return out


def merge_article_scores(
    articles: list[dict[str, Any]],
    scores: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach parsed scores to each article by index ``i``."""
    by_i = {s["i"]: s for s in scores if "i" in s}
    merged: list[dict[str, Any]] = []
    for i, art in enumerate(articles):
        s = by_i.get(i, {})
        raw_r = s.get("r", "n")
        relevant = raw_r in ("y", "relevant", "Y", True)
        raw_s = s.get("s", "0")
        if raw_s == "+":
            sentiment = "positive"
        elif raw_s == "-":
            sentiment = "negative"
        else:
            sentiment = "neutral"
        merged.append(
            {
                "title": art.get("title", ""),
                "url": art.get("url"),
                "relevant": relevant,
                "sentiment": sentiment,
                "summary": s.get("e", "") or "",
                "reason_up": s.get("u", "") or "",
                "reason_down": s.get("d", "") or "",
            }
        )
    return merged


async def analyze_news_batch(
    *,
    symbol: str,
    articles: list[dict[str, Any]],
    cerebras: CerebrasService | None,
    groq: GroqService | None,
) -> tuple[list[dict[str, Any]], str]:
    """Run Layer-1-style batch scoring. Chunks at ``BATCH_SIZE``.

    Returns
    -------
    (merged_results, model_name)
    """
    if not articles:
        return [], "none"

    sym = symbol.upper()
    all_merged: list[dict[str, Any]] = []
    model_used = "none"

    for offset in range(0, len(articles), BATCH_SIZE):
        chunk = articles[offset : offset + BATCH_SIZE]
        prompt = build_batch_prompt(sym, chunk, use_toon=False)

        text_out = ""
        if cerebras:
            try:
                resp = await cerebras.analyze(prompt, system=LAYER1_SYSTEM)
                text_out = resp.text
                model_used = resp.model
            except Exception as exc:
                log.warning("Cerebras news batch failed: %s", exc)
                await fire_operator_alert(
                    category="LLM · Cerebras",
                    summary=f"analyze_news_batch({sym}): Cerebras failed — trying Groq.",
                    detail=format_exc_brief(exc),
                    dedupe_key="llm_cerebras_news_batch",
                )

        if not text_out and groq:
            try:
                resp = await groq.analyze(prompt, system=LAYER1_SYSTEM)
                text_out = resp.text
                model_used = resp.model
            except Exception as exc:
                log.error("Groq news batch failed: %s", exc)
                await fire_operator_alert(
                    category="LLM · Groq",
                    summary=f"analyze_news_batch({sym}): Groq failed after Cerebras miss.",
                    detail=format_exc_brief(exc),
                    dedupe_key="llm_groq_news_batch",
                )
                raise RuntimeError(f"News batch LLM failed: {exc}") from exc

        if not text_out:
            await fire_operator_alert(
                category="LLM · News",
                summary=f"analyze_news_batch({sym}): no LLM output (Cerebras/Groq unavailable).",
                dedupe_key="llm_news_no_output",
            )
            raise RuntimeError("No LLM available for news batch (Cerebras/Groq unavailable)")

        scores = parse_layer1_response(text_out, len(chunk))
        merged = merge_article_scores(chunk, scores)
        all_merged.extend(merged)

    return all_merged, model_used
