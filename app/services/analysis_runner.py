"""
Shared symbol analysis pipeline for Telegram and ``/internal/analyze``.

Faz 1.5-b: optional Finnhub → news batch digest, optional extended price
features, optional TOON packing for the user message ([toon-format](https://github.com/toon-format/spec)).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.config import Settings
from app.core.logging import get_logger
from app.services.finnhub_news import fetch_company_news
from app.services.news_pipeline import analyze_news_batch
from app.services.t212.ticker_map import t212_to_yfinance
from app.tools.technical import TechnicalSummary, get_technical_summary
from app.tools.technical_extended import ExtendedPriceSnapshot, get_extended_price_features

log = get_logger("analysis_runner")


@dataclass
class AnalysisResult:
    symbol: str
    technical: TechnicalSummary
    extended: ExtendedPriceSnapshot | None = None
    position_info: str = ""
    news_model: str | None = None
    news_articles: list[dict] = field(default_factory=list)
    news_digest: str | None = None
    news_attempted: bool = False
    llm_text: str = ""
    llm_model: str = "none"
    prompt_encoding: str = "plain"  # "plain" | "toon"
    memories: list[dict] = field(default_factory=list)


_ANALYSIS_SYSTEM = """You are a concise financial analyst. Analyse the data provided and give a
BUY / SELL / HOLD recommendation with a short rationale (max 100 words).
Include confidence (0-1) and a suggested stop-loss level if applicable.
Respond in valid JSON with keys: action, confidence, stop_loss, target, reasoning.
Return only valid JSON (double quotes). No markdown fences."""


def _toon_available() -> bool:
    try:
        import toon_format  # noqa: F401

        return True
    except ImportError:
        return False


def _pack_payload_toon(data: dict) -> str:
    import toon_format

    return toon_format.encode(data)


def _build_news_digest(merged: list[dict], *, top: int = 5) -> str:
    rel = [m for m in merged if m.get("relevant")]
    lines: list[str] = []
    for m in rel[:top]:
        lines.append(
            f"- [{m.get('sentiment')}] {m.get('title', '')}: {m.get('summary', '')}"
        )
    if not lines:
        return "No relevant scored headlines in this window."
    return "Recent scored headlines:\n" + "\n".join(lines)


def _build_user_message_plain(
    *,
    symbol: str,
    position_info: str,
    tech_text: str,
    extended_text: str | None,
    news_digest: str | None,
    memories_digest: str | None,
) -> str:
    parts = [
        f"Symbol: {symbol}",
        f"Portfolio: {position_info}",
        f"Technical (MVP): {tech_text}",
    ]
    if extended_text:
        parts.append(f"Technical (extended): {extended_text}")
    if news_digest:
        parts.append(news_digest)
    if memories_digest:
        parts.append(memories_digest)
    parts.append(
        "Return only valid JSON with keys: action, confidence, stop_loss, target, reasoning."
    )
    return "\n\n".join(parts)


def _build_user_message_toon(
    *,
    symbol: str,
    position_info: str,
    technical: TechnicalSummary,
    extended: ExtendedPriceSnapshot | None,
    news_rows: list[dict],
    memories: list[dict] = None,
) -> str:
    payload: dict = {
        "symbol": symbol,
        "portfolio": position_info,
        "technical_mvp": technical.to_dict(),
    }
    if extended and not extended.error:
        payload["technical_extended"] = extended.features
    if news_rows:
        payload["recent_news"] = news_rows
    if memories:
        payload["past_experiences"] = memories
    body = _pack_payload_toon(payload)
    return (
        "Market context is encoded below in TOON (Token-Oriented Object Notation). "
        "Parse it, then output your recommendation as JSON only.\n\n"
        f"```toon\n{body}\n```\n\n"
        "Return only valid JSON with keys: action, confidence, stop_loss, target, reasoning."
    )


async def run_symbol_analysis(
    *,
    symbol: str,
    settings: Settings,
    http_client,
    t212,
    groq,
    ollama,
    retriever=None,
    include_news: bool = False,
    include_extended_technical: bool = False,
    use_toon: bool | None = None,
    news_max_articles: int = 20,
) -> AnalysisResult:
    """Full pipeline: technicals, optional T212 position, optional news batch, one LLM call."""
    sym = symbol.upper()
    tech = await get_technical_summary(sym)

    extended: ExtendedPriceSnapshot | None = None
    if include_extended_technical:
        extended = await get_extended_price_features(sym)

    position_info = "Not in portfolio"
    if t212:
        try:
            summary = await t212.get_account_summary()
            base_ccy = (summary.get("currency") or "USD").strip().upper()[:3]
            positions = await t212.get_positions()
            for p in positions:
                if t212_to_yfinance(p.ticker) == sym:
                    position_info = (
                        f"Holding {p.quantity:.2f} shares, "
                        f"avg {p.average_price_paid:.2f}, current {p.current_price:.2f} (API quote), "
                        f"P&L: {p.pnl:+.2f} {base_ccy} ({p.pnl_percent:+.1f}%)"
                    )
                    break
        except Exception as exc:
            log.warning("T212 fetch in analysis: %s", exc)
            position_info = f"T212 error: {exc}"

    news_articles: list[dict] = []
    news_digest: str | None = None
    news_model: str | None = None
    news_rows_for_toon: list[dict] = []

    news_attempted = include_news
    if include_news:
        if not settings.finnhub_api_key:
            log.warning("include_news requested but FINNHUB_API_KEY empty")
            news_digest = "(Haber isteniyor ama FINNHUB_API_KEY tanımlı değil.)"
        elif not http_client:
            log.warning("include_news requested but http_client missing")
            news_digest = "(HTTP istemcisi yok — haber çekilemedi.)"
        else:
            try:
                raw = await fetch_company_news(
                    http_client,
                    api_key=settings.finnhub_api_key,
                    symbol=sym,
                    days=7,
                    max_articles=max(1, min(news_max_articles, 40)),
                )
                if raw:
                    news_articles, news_model = await analyze_news_batch(
                        symbol=sym,
                        articles=raw,
                        groq=groq,
                        ollama=ollama,
                        prefer_local=bool(getattr(settings, "prefer_local_llm", False)),
                    )
                    news_digest = _build_news_digest(news_articles)
                    rel = [m for m in news_articles if m.get("relevant")][:5]
                    news_rows_for_toon = [
                        {
                            "sentiment": r.get("sentiment"),
                            "title": r.get("title"),
                            "summary": r.get("summary"),
                        }
                        for r in rel
                    ]
                else:
                    news_digest = "(Finnhub bu pencerede haber döndürmedi.)"
            except Exception as exc:
                log.error("News path in analyze failed: %s", exc)
                news_digest = f"(News fetch/score failed: {exc})"

    memories_digest: str | None = None
    memories_list: list[dict] = []
    if retriever:
        try:
            mems = await retriever.search_similar_memories(sym, f"Trading analysis and news for {sym}", top_k=3)
            if mems:
                memories_list = mems
                lines = ["Past experiences:"]
                for m in mems:
                    lines.append(f"- [{m.get('memory_type', 'INFO')}] {m.get('context', '')} (Outcome: {m.get('outcome', 'UNKNOWN')})")
                memories_digest = "\n".join(lines)
        except Exception as exc:
            log.error("Failed to fetch RAG memories: %s", exc)

    want_toon = settings.use_toon_prompts if use_toon is None else use_toon
    encoding = "toon" if want_toon and _toon_available() else "plain"
    if want_toon and encoding == "plain":
        log.warning("USE_TOON_PROMPTS set but toon-format not installed — falling back to plain")

    if encoding == "toon":
        user_msg = _build_user_message_toon(
            symbol=sym,
            position_info=position_info,
            technical=tech,
            extended=extended,
            news_rows=news_rows_for_toon,
            memories=memories_list,
        )
    else:
        ext_txt = extended.summary_text() if extended and not extended.error else None
        user_msg = _build_user_message_plain(
            symbol=sym,
            position_info=position_info,
            tech_text=tech.summary_text(),
            extended_text=ext_txt,
            news_digest=news_digest,
            memories_digest=memories_digest,
        )

    llm_text = ""
    llm_model = "none"

    prefer_local = bool(getattr(settings, "prefer_local_llm", False))
    if groq and not prefer_local:
        try:
            resp = await groq.analyze(user_msg, system=_ANALYSIS_SYSTEM)
            llm_text = resp.text
            llm_model = resp.model
        except Exception as exc:
            log.warning("Groq analyze failed: %s", exc)

    if not llm_text and ollama:
        try:
            resp = await ollama.analyze(user_msg, system=_ANALYSIS_SYSTEM)
            llm_text = resp.text
            llm_model = resp.model
        except Exception as exc:
            log.error("Ollama analyze failed: %s", exc)
            llm_text = f"⚠️ LLM error: {exc}"

    if not llm_text:
        llm_text = "⚠️ No LLM available."
        llm_model = "none"

    # Auto-save memory after successful analysis
    if retriever and llm_text and not llm_text.startswith("⚠️"):
        try:
            import json
            import re
            
            clean_text = llm_text.strip()
            # Basic cleanup if markdown is present
            if clean_text.startswith("```"):
                clean_text = re.sub(r"^```(json)?", "", clean_text)
                clean_text = re.sub(r"```$", "", clean_text).strip()
                
            data = json.loads(clean_text)
            action = data.get("action", "UNKNOWN")
            reasoning = data.get("reasoning", "")
            
            await retriever.add_memory(
                ticker=sym,
                memory_type="LESSON",
                context=f"Analyzed {sym}. Action decided: {action}. Reasoning: {reasoning}",
                outcome="OPEN"
            )
        except Exception as exc:
            log.warning("Could not auto-save memory after analysis: %s", exc)

    return AnalysisResult(
        symbol=sym,
        technical=tech,
        extended=extended,
        position_info=position_info,
        news_model=news_model,
        news_articles=news_articles,
        news_digest=news_digest,
        news_attempted=news_attempted,
        llm_text=llm_text,
        llm_model=llm_model,
        prompt_encoding=encoding,
        memories=memories_list,
    )


def analysis_result_to_api_dict(r: AnalysisResult) -> dict:
    """Shape for ``/internal/analyze`` JSON response."""
    out: dict = {
        "symbol": r.symbol,
        "technical": r.technical.to_dict(),
        "position": r.position_info,
        "llm": {"model": r.llm_model, "text": r.llm_text},
        "prompt_encoding": r.prompt_encoding,
    }
    if r.extended:
        out["technical_extended"] = r.extended.to_dict()
    if r.news_attempted:
        out["news"] = {
            "batch_model": r.news_model,
            "articles": r.news_articles,
            "digest": r.news_digest,
        }
    return out
