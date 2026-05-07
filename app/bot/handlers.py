"""
Telegram bot command handlers.

Each handler receives services from ``context.bot_data`` (injected at setup).
"""

from __future__ import annotations

import html
import re

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from app.core.logging import get_logger
from app.services.analysis_runner import run_symbol_analysis
from app.services.telegram_operator_alerts import fire_operator_alert, format_exc_brief
from app.services.paper.account_currency import resolve_paper_account_currency
from app.services.t212.ticker_map import t212_to_yfinance
from app.services.finnhub_news import fetch_company_news
from app.services.news_pipeline import analyze_news_batch
import asyncio
import statistics
import yfinance as yf

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    """Remove deepseek-r1 ``<think>...</think>`` blocks before showing to user."""
    cleaned = _THINK_TAG_RE.sub("", text or "").strip()
    return cleaned or (text or "").strip()

log = get_logger("bot.handlers")

# Telegram message hard limit
_MAX_MSG_LEN = 4096


# ── Helpers ─────────────────────────────────────────────────────────

def _check_user(update: Update, allowed_ids: set[int]) -> bool:
    """Return True only if ``TELEGRAM_ALLOWED_USER_IDS`` is non-empty and includes this user."""
    user = update.effective_user
    if not allowed_ids:
        log.warning(
            "TELEGRAM_ALLOWED_USER_IDS not set — access denied (user_id=%s)",
            user.id if user else None,
        )
        return False
    if user and user.id in allowed_ids:
        return True
    log.warning("Unauthorised user: %s (id=%s)", user, user.id if user else "?")
    return False


async def _send_long(
    update: Update,
    text: str,
    *,
    parse_mode: str | None = None,
) -> None:
    """Send a message, splitting into chunks if it exceeds Telegram limits."""
    if not update.effective_chat:
        return
    for i in range(0, len(text), _MAX_MSG_LEN):
        chunk = text[i : i + _MAX_MSG_LEN]
        kwargs: dict = {}
        if parse_mode:
            kwargs["parse_mode"] = parse_mode
        await update.effective_chat.send_message(chunk, **kwargs)


# ── Command handlers ───────────────────────────────────────────────

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/start`` — short welcome; full reference lives in /help."""
    allowed = context.bot_data.get("allowed_ids", set())
    if not _check_user(update, allowed):
        return
    body = "\n".join(
        [
            "<b>🤖 AI Broker — otonom paper broker</b>",
            "T212 demo'ya gerçek emir akıtır, kararları kendisi verir, sana profesyonel bildirim atar.",
            "",
            "<b>Hızlı başlangıç</b>",
            "• <code>/portfolio</code> — T212 hesap özeti + açık pozisyonlar",
            "• <code>/paper</code> — PaperAgent durumu (NAV, son cycle, açık trade'ler)",
            "• <code>/analyze AAPL</code> — anlık teknik + AI tezi",
            "• <code>/runpaper</code> — manuel cycle tetikle",
            "",
            "📖 <b>Tüm komutlar</b>: <code>/help</code>",
            "💬 <b>Sohbet</b>: komutla başlamayan her mesaj broker bağlamıyla cevaplanır "
            "(örn. \"AAPL'de neden BUY dedin?\").",
        ]
    )
    await _send_long(
        update,
        body,
        parse_mode=ParseMode.HTML,
    )


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/help`` — categorized command reference."""
    allowed = context.bot_data.get("allowed_ids", set())
    if not _check_user(update, allowed):
        return
    # HTML for stable typography (Telegram Markdown is easy to break on special chars).
    def esc(s: str) -> str:
        return html.escape(s, quote=False)

    body = "\n".join(
        [
            "<b>AI Broker — Komut Rehberi</b>",
            "",
            "<u>Portföy ve sanal hesap</u>",
            f"• <code>/portfolio</code> — {esc('T212 hesap özeti (cash, totalValue, P&L) + açık pozisyonlar')}",
            f"• <code>/paper</code> — {esc('PaperAgent NAV, nakit, bugünkü P&L, açık trade')}",
            f"• <code>/paper history</code> — {esc('Son 10 paper işlem (BUY/SELL, fiyat, reasoning özeti)')}",
            f"• <code>/paper stats</code> — {esc('Kazanma oranı, ortalama R/R, en iyi/kötü trade')}",
            f"• <code>/paper log</code> — {esc('Son cycle analizi + JSON kararlar')}",
            f"• <code>/paper reset confirm</code> — {esc('Sanal portföyü başlangıç nakdine sıfırla (geri alınamaz)')}",
            "",
            "<u>Analiz</u>",
            f"• <code>/analyze SYMBOL</code> — {esc('yfinance teknik (RSI, SMA, MACD) + AI tezi')}",
            f"• <code>/analyze SYMBOL news</code> — {esc('+ Finnhub haber duyarlılığı + birleşik AI')}",
            f"• <code>/analyze SYMBOL news full</code> — {esc('+ 31 feature genişletilmiş teknik')}",
            f"• <code>/news SYMBOL</code> — {esc('Haber duyarlılığı (LLM batch)')}",
            f"• <code>/memory SYMBOL</code> — {esc('RAG: dersler, başarı/uyarı')}",
            "",
            "<u>Paper Agent</u>",
            f"• <code>/runpaper</code> — {esc('Manuel cycle (programlı tick beklemeden)')}",
            f"• <code>/punishments</code> — {esc('Aktif ticker cezaları (peş peşe loss → geçici ban)')}",
            "",
            "<u>İzleme</u>",
            f"• <code>/usage</code> — {esc('Cerebras + Groq token / maliyet özeti')}",
            "",
            "<u>Sohbet (komutsuz)</u>",
            esc("Komutla başlamayan mesajlar broker bağlamıyla yanıtlanır. Örnek:"),
            f"• {esc('AAPL için neden BUY dedin?')}",
            f"• {esc('NVDA pozisyonum bugün nasıl?')}",
            "",
            "<u>Otomatik bildirimler</u>",
            esc(
                "BUY/SELL kartları, OPEN/MIDDAY/CLOSE özetleri, invalidasyon SELL, acil tetikleyiciler. "
                "Kritik LLM / PaperAgent hataları (Cerebras→Groq fallback vb.) aynı sohbete operatör uyarısı olarak düşer "
                "(TELEGRAM_OPERATOR_ALERTS_ENABLED)."
            ),
            "",
            "<u>T212 emir tipleri (PaperAgent)</u>",
            esc("MARKET, LIMIT, STOP, STOP_LIMIT — model uygun olanı seçer."),
        ]
    )
    await _send_long(update, body, parse_mode=ParseMode.HTML)


# ── Free-text chat handler ──────────────────────────────────────────

async def chat_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Free-text → LLM (broker context). No trade execution from chat."""
    allowed = context.bot_data.get("allowed_ids", set())
    if not _check_user(update, allowed):
        return
    if not update.message or not update.message.text:
        return
    text = update.message.text.strip()
    if not text:
        return

    cerebras = context.bot_data.get("cerebras")
    groq = context.bot_data.get("groq")
    t212 = context.bot_data.get("t212")
    paper_agent = context.bot_data.get("paper_agent")
    if cerebras is None and groq is None:
        await _send_long(update, "⚠️ Sohbet için LLM yapılandırılmamış.")
        return

    ctx_lines: list[str] = []
    if t212 is not None:
        try:
            summ = await t212.get_account_summary()
            cash = float((summ.get("cash") or {}).get("availableToTrade") or 0.0)
            cur = (summ.get("currency") or "?").upper()[:3]
            total = summ.get("totalValue")
            ctx_lines.append(f"T212 demo: cash {cash:.2f} {cur}, totalValue {total} {cur}")
            positions = await t212.get_positions()
            if positions:
                ctx_lines.append(
                    "Open positions: "
                    + "; ".join(
                        f"{p.ticker} qty={p.quantity:.2f} avg={p.average_price_paid:.2f} "
                        f"cur={p.current_price:.2f} pnl={p.pnl:+.2f}"
                        for p in positions[:10]
                    )
                )
            else:
                ctx_lines.append("Open positions: none")
        except Exception as exc:
            ctx_lines.append(f"T212 unavailable: {type(exc).__name__}: {exc}")

    last_cycle = getattr(paper_agent, "_last_cycle_text", None) if paper_agent else None
    last_cycle_at = getattr(paper_agent, "_last_cycle_at_utc", None) if paper_agent else None
    if last_cycle:
        snippet = last_cycle[:600]
        ctx_lines.append(f"Last PaperAgent cycle ({last_cycle_at}):\n{snippet}")

    ctx_block = "\n".join(ctx_lines) or "(no context)"
    system_prompt = (
        "You are AI Broker assistant talking to the CEO on Telegram. Reply in Turkish. "
        "Be concise (max ~6 sentences). Do NOT execute trades from chat — only answer, "
        "explain, or summarise. Use ONLY the CONTEXT for facts; if a fact is missing, say so. "
        "Never invent prices, positions, or quantities.\n\n"
        f"CONTEXT:\n{ctx_block}"
    )

    try:
        await update.effective_chat.send_chat_action("typing")
    except Exception:
        pass
    try:
        resp = None
        if cerebras:
            try:
                resp = await cerebras.analyze(text, system=system_prompt)
            except Exception as exc:
                log.warning("chat_handler Cerebras failed: %s", exc)
                await fire_operator_alert(
                    category="Telegram · Sohbet",
                    summary="chat_handler: Cerebras cevap üretemedi — Groq deneniyor.",
                    detail=format_exc_brief(exc),
                    dedupe_key="telegram_chat_cerebras",
                )
        if resp is None and groq:
            resp = await groq.analyze(text, system=system_prompt)
        if resp is None:
            raise RuntimeError("LLM unavailable")
        out = _strip_thinking(resp.text if resp else "")[:3500]
        if not out:
            out = "(boş cevap)"
        await _send_long(update, out)
    except Exception as exc:
        log.error("chat_handler LLM failed: %s", exc)
        await fire_operator_alert(
            category="Telegram · Sohbet",
            summary="chat_handler: LLM cevap üretemedi.",
            detail=format_exc_brief(exc),
            dedupe_key="telegram_chat_llm",
        )
        await _send_long(update, f"⚠️ Sohbet hatası: {type(exc).__name__}: {exc}")


async def portfolio_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/portfolio`` — list T212 open positions."""
    allowed = context.bot_data.get("allowed_ids", set())
    if not _check_user(update, allowed):
        return

    t212 = context.bot_data.get("t212")
    if not t212:
        await _send_long(update, "⚠️ T212 client not initialised.")
        return

    await _send_long(update, "⏳ Portföy çekiliyor...")

    try:
        summary = await t212.get_account_summary()
        positions = await t212.get_positions()
    except Exception as exc:
        log.error("portfolio_handler T212 error: %s", exc)
        await _send_long(update, f"❌ T212 hatası: {exc}")
        return

    acct_cur = (summary.get("currency") or "?").strip().upper()[:3] or "?"

    if not positions:
        await _send_long(update, "📭 Açık pozisyon yok.")
        return

    lines = ["📊 *Açık Pozisyonlar*\n", f"_Hesap para birimi: {acct_cur}_\n"]
    for p in positions:
        emoji = "🟢" if p.pnl >= 0 else "🔴"
        yf_sym = t212_to_yfinance(p.ticker)
        lines.append(
            f"{emoji} *{yf_sym}* — {p.quantity:.2f} adet\n"
            f"   Ort: {p.average_price_paid:.2f} → Güncel: {p.current_price:.2f} (emir fiyatı; US hisse genelde USD)\n"
            f"   P&L: {p.pnl:+.2f} {acct_cur} ({p.pnl_percent:+.1f}%)"
        )

    await _send_long(update, "\n".join(lines))


async def analyze_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/analyze SYMBOL`` — technical analysis + LLM."""
    allowed = context.bot_data.get("allowed_ids", set())
    if not _check_user(update, allowed):
        return

    if not context.args:
        await _send_long(
            update,
            "Kullanım:\n"
            "`/analyze AMD` — teknik + AI\n"
            "`/analyze AMD news` — + haber\n"
            "`/analyze AMD news full` — + extended teknik",
        )
        return

    symbol = context.args[0].upper()
    rest = [a.lower() for a in context.args[1:]]
    include_news = "news" in rest
    include_extended = "full" in rest or "extended" in rest

    settings = context.bot_data.get("settings")
    http = context.bot_data.get("http_client")
    cerebras_svc = context.bot_data.get("cerebras")
    groq_svc = context.bot_data.get("groq")
    retriever = context.bot_data.get("retriever")
    t212 = context.bot_data.get("t212")

    if not settings:
        await _send_long(update, "⚠️ Ayarlar yüklenemedi.")
        return

    hint = " + haber" if include_news else ""
    if include_extended:
        hint += " + ext.teknik"
    await _send_long(update, f"⏳ {symbol} analiz ediliyor{hint}...")

    result = await run_symbol_analysis(
        symbol=symbol,
        settings=settings,
        http_client=http,
        t212=t212,
        cerebras=cerebras_svc,
        groq=groq_svc,
        retriever=retriever,
        include_news=include_news,
        include_extended_technical=include_extended,
    )

    response_parts = [result.technical.summary_text()]
    if result.extended and not result.extended.error:
        response_parts.append("\n" + result.extended.summary_text())
    if result.news_digest:
        digest = result.news_digest
        if len(digest) > 2800:
            digest = digest[:2800] + "…"
        response_parts.append("\n📰 " + digest)
    response_parts.append(
        f"\n🧠 *AI* (`{result.llm_model}`, `{result.prompt_encoding}`):\n{result.llm_text}"
    )
    await _send_long(update, "\n".join(response_parts))


async def news_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/news SYMBOL`` — Finnhub company news + PokieTicker-style batch LLM scoring."""
    allowed = context.bot_data.get("allowed_ids", set())
    if not _check_user(update, allowed):
        return

    if not context.args:
        await _send_long(update, "Kullanım: `/news AMD`")
        return

    symbol = context.args[0].upper()
    settings = context.bot_data.get("settings")
    http = context.bot_data.get("http_client")
    cerebras_svc = context.bot_data.get("cerebras")
    groq_svc = context.bot_data.get("groq")

    if not settings or not getattr(settings, "finnhub_api_key", ""):
        await _send_long(
            update,
            "⚠️ `FINNHUB_API_KEY` .env içinde yok — haber çekilemiyor.",
        )
        return
    if not http:
        await _send_long(update, "⚠️ HTTP client hazır değil.")
        return
    if not cerebras_svc and not groq_svc:
        await _send_long(update, "⚠️ LLM servisleri hazır değil.")
        return

    await _send_long(update, f"⏳ {symbol} haberleri çekiliyor + AI skorlanıyor...")

    try:
        articles = await fetch_company_news(
            http,
            api_key=settings.finnhub_api_key,
            symbol=symbol,
            days=7,
            max_articles=30,
        )
    except Exception as exc:
        log.error("Finnhub fetch failed: %s", exc)
        await _send_long(update, f"❌ Finnhub hatası: {exc}")
        return

    if not articles:
        await _send_long(update, f"📭 Son günlerde {symbol} için haber dönmedi.")
        return

    try:
        merged, model = await analyze_news_batch(
            symbol=symbol,
            articles=articles,
            cerebras=cerebras_svc,
            groq=groq_svc,
        )
    except Exception as exc:
        log.error("news batch failed: %s", exc)
        await _send_long(update, f"❌ Haber analizi başarısız: {exc}")
        return

    relevant = [m for m in merged if m.get("relevant")]
    pos = sum(1 for m in relevant if m.get("sentiment") == "positive")
    neg = sum(1 for m in relevant if m.get("sentiment") == "negative")
    neu = sum(1 for m in relevant if m.get("sentiment") == "neutral")

    lines = [
        f"📰 *{symbol}* — model: `{model}`",
        f"Haber: {len(merged)} | İlgili: {len(relevant)} (+{pos} / -{neg} / nötr {neu})",
        "",
    ]
    for m in relevant[:8]:
        title = (m.get("title") or "")[:120]
        summ = (m.get("summary") or "")[:160]
        lines.append(f"• [{m.get('sentiment')}] *{title}*")
        if summ:
            lines.append(f"  _{summ}_")
    if len(relevant) > 8:
        lines.append(f"\n… ve {len(relevant) - 8} ilgili haber daha")

    await _send_long(update, "\n".join(lines))


async def usage_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/usage`` — daily Groq token usage."""
    allowed = context.bot_data.get("allowed_ids", set())
    if not _check_user(update, allowed):
        return

    groq_svc = context.bot_data.get("groq")
    if not groq_svc:
        await _send_long(update, "⚠️ Groq service not initialised.")
        return

    u = groq_svc.usage.to_dict()
    text = (
        "📈 *Günlük Groq Kullanımı*\n\n"
        f"İstek sayısı: {u['total_requests']}\n"
        f"Input token: {u['total_input_tokens']:,}\n"
        f"Output token: {u['total_output_tokens']:,}\n"
        f"Toplam token: {u['total_tokens']:,}\n"
        f"Hatalar: {u['errors']}\n"
        f"Sıfırlama: {u['daily_reset']}"
    )
    await _send_long(update, text)

async def memory_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/memory SYMBOL`` — search RAG memories."""
    allowed = context.bot_data.get("allowed_ids", set())
    if not _check_user(update, allowed):
        return

    if not context.args:
        await _send_long(update, "Kullanım: `/memory AMD`")
        return

    symbol = context.args[0].upper()
    retriever = context.bot_data.get("retriever")

    if not retriever:
        await _send_long(update, "⚠️ Hafıza sistemi (Retriever) aktif değil.")
        return

    await _send_long(update, f"⏳ {symbol} için geçmiş anılar taranıyor...")

    try:
        mems = await retriever.list_recent_memories(symbol, limit=10)
    except Exception as exc:
        log.error("Memory search failed: %s", exc)
        await _send_long(update, f"❌ Hafıza araması başarısız: {exc}")
        return

    if not mems:
        await _send_long(update, f"📭 {symbol} için kayıtlı anı bulunamadı.")
        return

    lines = [f"🧠 *{symbol} HAFIZASI*\n"]
    for m in mems:
        m_type = m.get("memory_type", "INFO").upper()
        emoji = "📌"
        if m_type == "SUCCESS":
            emoji = "✅"
        elif m_type == "WARNING":
            emoji = "⚠️"
        elif m_type == "LESSON":
            emoji = "📌"
            
        context_text = m.get("context", "")
        outcome = m.get("outcome", "")
        pnl = m.get("pnl_percent")
        created_at = m.get("created_at", "")
        
        when = f" — {created_at}" if created_at else ""
        line = f"{emoji} *{m_type}*{when}: \"{context_text}\""
        if outcome:
            line += f" (Sonuç: {outcome}"
            if pnl is not None:
                line += f", PnL: {pnl:+.1f}%"
            line += ")"
        lines.append(line)
        lines.append("")

    await _send_long(update, "\n".join(lines))


def _fetch_yf_price(symbol: str) -> float | None:
    try:
        t = yf.Ticker(symbol)
        # Try fast_info first
        if hasattr(t, "fast_info") and 'lastPrice' in t.fast_info:
            return t.fast_info['lastPrice']
        # Fallback to info
        info = t.info
        return info.get("currentPrice") or info.get("regularMarketPrice")
    except Exception as e:
        log.warning(f"yfinance fetch failed for {symbol}: {e}")
        return None


async def paper_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/paper`` — list virtual paper portfolio and balance."""
    allowed = context.bot_data.get("allowed_ids", set())
    if not _check_user(update, allowed):
        return

    broker = context.bot_data.get("paper_broker")
    if not broker:
        await _send_long(update, "⚠️ Paper Broker servisi aktif değil.")
        return

    # Subcommands
    sub = (context.args[0].lower() if context.args else "").strip()
    if sub in ("history", "trades"):
        await paper_history_handler(update, context)
        return
    if sub in ("stats",):
        await paper_stats_handler(update, context)
        return
    if sub in ("log", "cycle"):
        await paper_log_handler(update, context)
        return
    if sub in ("reset",):
        await paper_reset_handler(update, context)
        return

    settings = context.bot_data.get("settings")
    if settings and settings.paper_executes_on_t212:
        t212 = context.bot_data.get("t212")
        if not t212:
            await _send_long(update, "⚠️ Paper yürütme T212 üzerinde; T212 client yok.")
            return
        await _send_long(update, "⏳ T212 hesabı çekiliyor (Paper Agent)...")
        try:
            summary = await t212.get_account_summary()
            positions = await t212.get_positions()
        except Exception as exc:
            log.error("paper_handler T212: %s", exc)
            await _send_long(update, f"❌ T212 hatası: {exc}")
            return
        cur = summary.get("currency") or "?"
        tv = float(summary.get("totalValue") or 0)
        cash = summary.get("cash") or {}
        avail = float(cash.get("availableToTrade") or 0)
        lines = [
            "💼 *Paper Agent — broker: Trading 212*\n",
            f"Para birimi: *{cur}*",
            f"Toplam değer: *{tv:,.2f} {cur}*",
            f"Müsait nakit: *{avail:,.2f} {cur}*",
            "",
        ]
        if not positions:
            lines.append("📭 Açık pozisyon yok (T212).")
        else:
            lines.append("📈 *Açık pozisyonlar (T212)*:")
            for p in sorted(positions, key=lambda x: x.ticker):
                yf_sym = t212_to_yfinance(p.ticker)
                emoji = "🟢" if p.pnl >= 0 else "🔴"
                lines.append(
                    f"{emoji} *{yf_sym}* (`{p.ticker}`) — {p.quantity:.4f} adet\n"
                    f"   Ort: {p.average_price_paid:.2f} → Son: {p.current_price:.2f} | "
                    f"P&L: {p.pnl_percent:+.1f}%"
                )
        lines.append("\n_Audit kayıtları: Supabase `paper_trades` (T212 emir yansıması)._")
        await _send_long(update, "\n".join(lines))
        return

    await _send_long(update, "⏳ Sanal portföy çekiliyor...")

    ledger_settings = context.bot_data.get("settings")
    ledger_cur = (
        (ledger_settings.paper_account_currency or "USD").upper()[:3]
        if ledger_settings
        else "USD"
    )

    try:
        balance = await broker.get_balance()
        positions = await broker.get_positions()
    except Exception as exc:
        log.error("paper_handler DB error: %s", exc)
        await _send_long(update, f"❌ Veritabanı hatası: {exc}")
        return

    lines = ["💼 *Sanal Portföy (Paper Agent)*\n", f"_Defter para birimi: {ledger_cur}_\n"]
    lines.append(f"💵 Nakit Bakiye: *{balance:,.2f} {ledger_cur}*")
    
    if not positions:
        lines.append("📭 Açık pozisyon yok.")
    else:
        lines.append("\n📈 *Açık Pozisyonlar:*")
        total_unrealized = 0.0
        total_invested = 0.0
        
        for p in positions:
            cost_basis = p.shares * p.avg_cost
            total_invested += cost_basis
            
            # Fetch live price
            live_price = await asyncio.to_thread(_fetch_yf_price, p.ticker)
            
            if live_price is not None:
                current_value = p.shares * live_price
                unrealized_pnl = current_value - cost_basis
                total_unrealized += unrealized_pnl
                pnl_pct = (unrealized_pnl / cost_basis) * 100.0 if cost_basis > 0 else 0.0
                emoji = "🟢" if unrealized_pnl >= 0 else "🔴"
                price_str = f"{live_price:.2f} (yfinance, genelde USD)"
                pnl_str = f"{unrealized_pnl:,.2f} {ledger_cur} ({pnl_pct:+.1f}%)"
            else:
                emoji = "⚪"
                price_str = "N/A"
                pnl_str = "N/A"
                
            lines.append(
                f"{emoji} *{p.ticker}* — {p.shares:.2f} adet\n"
                f"   Ort. Maliyet: {p.avg_cost:.2f} {ledger_cur} → Güncel: {price_str}\n"
                f"   PnL: {pnl_str}"
            )
            
        lines.append(f"\n📊 Toplam Yatırım: *{total_invested:,.2f} {ledger_cur}*")
        lines.append(f"🔄 Toplam Unrealized PnL: *{total_unrealized:+,.2f} {ledger_cur}*")
        lines.append(
            f"💰 Net Varlık (Nakit+Pozisyon): *{(balance + total_invested + total_unrealized):,.2f} {ledger_cur}*"
        )

    await _send_long(update, "\n".join(lines))


async def paper_history_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/paper history`` — last 10 paper trades."""
    allowed = context.bot_data.get("allowed_ids", set())
    if not _check_user(update, allowed):
        return

    broker = context.bot_data.get("paper_broker")
    if not broker:
        await _send_long(update, "⚠️ Paper Broker servisi aktif değil.")
        return

    try:
        trades = await broker.get_recent_trades(limit=10)
    except Exception as exc:
        log.error("paper_history_handler error: %s", exc)
        await _send_long(update, f"❌ DB hatası: {exc}")
        return

    if not trades:
        await _send_long(update, "📭 No paper trades yet.")
        return

    settings = context.bot_data.get("settings")
    t212 = context.bot_data.get("t212")
    hist_cur = "USD"
    if settings:
        hist_cur = await resolve_paper_account_currency(settings, t212)

    lines = ["🧾 *PAPER TRADE HISTORY* (last 10)\n", f"_Tutarlar (ayna): {hist_cur}_\n"]
    for t in trades:
        when = t.created_at.isoformat() if t.created_at else ""
        pnl = ""
        if t.realized_pnl_usd is not None:
            pnl = f" | realized: {t.realized_pnl_usd:+.2f} {hist_cur}"
        elif t.pnl_percent is not None and t.action.upper() == "SELL":
            pnl = f" | pnl: {t.pnl_percent:+.1f}%"
        lines.append(f"- [{when}] {t.action} {t.ticker} {t.shares:.2f} @ {t.price:.2f} {hist_cur}{pnl}")

    await _send_long(update, "\n".join(lines))


async def paper_stats_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/paper stats`` — basic paper stats (lightweight)."""
    allowed = context.bot_data.get("allowed_ids", set())
    if not _check_user(update, allowed):
        return

    broker = context.bot_data.get("paper_broker")
    if not broker:
        await _send_long(update, "⚠️ Paper Broker servisi aktif değil.")
        return

    settings = context.bot_data.get("settings")
    try:
        trades = await broker.get_recent_trades(limit=200)
        if settings and settings.paper_executes_on_t212 and context.bot_data.get("t212"):
            t212 = context.bot_data["t212"]
            summary = await t212.get_account_summary()
            positions = await t212.get_positions()
            balance = float((summary.get("cash") or {}).get("availableToTrade") or 0)
            nav_est = float(summary.get("totalValue") or 0)
            acct_cur = summary.get("currency") or "?"
        else:
            balance = await broker.get_balance()
            positions = await broker.get_positions()
            invested_cost = sum(p.shares * p.avg_cost for p in positions)
            total_unrealized = 0.0
            for p in positions:
                live = await asyncio.to_thread(_fetch_yf_price, p.ticker)
                if live is not None:
                    total_unrealized += p.shares * float(live) - p.shares * p.avg_cost
            nav_est = balance + invested_cost + total_unrealized
            acct_cur = (
                (settings.paper_account_currency or "USD").upper()[:3] if settings else "USD"
            )
    except Exception as exc:
        log.error("paper_stats_handler error: %s", exc)
        await _send_long(update, f"❌ DB/T212 hatası: {exc}")
        return

    realized = 0.0
    wins = 0
    losses = 0
    closed = 0
    for t in trades:
        if t.action.upper() != "SELL":
            continue
        if t.realized_pnl_usd is None:
            continue
        closed += 1
        realized += float(t.realized_pnl_usd)
        if t.realized_pnl_usd >= 0:
            wins += 1
        else:
            losses += 1

    win_rate = (wins / closed * 100.0) if closed else 0.0

    start_capital = float(getattr(settings, "paper_starting_nav_usd", 20_000.0) or 20_000.0)
    total_return_pct = ((nav_est - start_capital) / start_capital) * 100.0 if start_capital else 0.0

    pnl_pcts: list[float] = []
    for t in trades:
        if t.action.upper() != "SELL" or t.pnl_percent is None:
            continue
        try:
            pnl_pcts.append(float(t.pnl_percent))
        except Exception:
            continue
    sharpe_note = "N/A (need ≥2 closed SELL rows with pnl_percent)"
    if len(pnl_pcts) >= 2:
        m = statistics.mean(pnl_pcts)
        sd = statistics.pstdev(pnl_pcts)
        if sd > 1e-9:
            sharpe_note = f"~{m / sd:.2f} (per-trade pnl% mean/σ, not annualized)"

    lines = ["📊 *PAPER AGENT STATS*"]
    if settings and settings.paper_executes_on_t212:
        lines.append(f"Broker: *T212* | Hesap para birimi: *{acct_cur}*")
        lines.append(f"Müsait nakit: *{balance:,.2f} {acct_cur}*")
        lines.append(f"Açık pozisyon (T212): *{len(positions)}*")
        lines.append(f"Toplam değer (T212): *{nav_est:,.2f} {acct_cur}*")
        lines.append(f"Başlangıç referansı: *{start_capital:,.0f} {acct_cur}* (`PAPER_STARTING_NAV_USD`)")
    else:
        lines.append(f"Cash: {balance:,.2f} {acct_cur}")
        lines.append(f"Open positions: {len(positions)}")
        lines.append(f"Est. NAV (cash + cost basis + unrealized): {nav_est:,.2f} {acct_cur}")
    lines.append(f"vs başlangıç ({start_capital:,.0f} {acct_cur}): {total_return_pct:+.1f}%")
    lines.append(f"Closed trades (SELL w/ realized in audit row): {closed}")
    lines.append(f"Win rate: {win_rate:.1f}% (W={wins}, L={losses})")
    lines.append(f"Realized (audit currency / row): *{realized:+,.2f}*")
    lines.append(f"Sharpe-like (trades): {sharpe_note}")
    await _send_long(update, "\n".join(lines))


async def paper_log_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/paper log`` — show last PaperAgent cycle (if agent wired)."""
    allowed = context.bot_data.get("allowed_ids", set())
    if not _check_user(update, allowed):
        return

    agent = context.bot_data.get("paper_agent")
    if not agent:
        await _send_long(update, "⚠️ PaperAgent not running / not wired yet.")
        return

    last = agent.last_cycle
    if not last:
        await _send_long(update, "No cycle log available.")
        return

    at = last.get("at_utc") or "N/A"
    dd = last.get("drawdown_from_peak_pct")
    dd_line = ""
    if dd is not None:
        try:
            dd_line = f"\nDrawdown from peak: {float(dd):.1f}%\n"
        except (TypeError, ValueError):
            dd_line = ""
    analysis = (last.get("analysis") or "").strip()
    if len(analysis) > 3500:
        analysis = analysis[:3500] + "…"
    decisions = last.get("decisions") or []
    try:
        decisions_text = json.dumps(decisions, ensure_ascii=False, indent=2)
    except Exception:
        decisions_text = "[]"
    if len(decisions_text) > 3500:
        decisions_text = decisions_text[:3500] + "\n... (truncated)"

    body_lines = [
        f"🧠 <b>LAST PAPER CYCLE</b> <code>{html.escape(str(at))}</code>",
    ]
    if dd_line:
        body_lines.append(f"📉 Drawdown from peak: <b>{html.escape(dd_line.strip().split(':', 1)[-1].strip())}</b>")
    body_lines.extend(
        [
            "",
            "<b>Analysis</b>",
            f"<pre>{html.escape(analysis or 'No analysis.')}</pre>",
            "",
            "<b>Decisions JSON</b>",
            f"<pre>{html.escape(decisions_text)}</pre>",
        ]
    )
    await _send_long(update, "\n".join(body_lines), parse_mode=ParseMode.HTML)


async def paper_reset_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/paper reset confirm`` — wipe paper trades/positions; restore starting cash."""
    allowed = context.bot_data.get("allowed_ids", set())
    if not _check_user(update, allowed):
        return

    args = [a.lower() for a in (context.args or [])]
    if len(args) < 2 or args[1] != "confirm":
        await _send_long(
            update,
            "⚠️ Bu komut tüm paper işlem geçmişini ve pozisyonları siler.\n"
            "Onaylamak için: `/paper reset confirm`",
        )
        return

    broker = context.bot_data.get("paper_broker")
    if not broker:
        await _send_long(update, "⚠️ Paper Broker servisi aktif değil.")
        return

    settings = context.bot_data.get("settings")
    start_nav = float(getattr(settings, "paper_starting_nav_usd", 20_000.0) or 20_000.0)
    t212 = context.bot_data.get("t212")
    reset_cur = "USD"
    if settings:
        reset_cur = await resolve_paper_account_currency(settings, t212)

    try:
        await broker.reset_all(starting_balance=start_nav)
    except Exception as exc:
        log.error("paper_reset_handler: %s", exc)
        await _send_long(update, f"❌ Sıfırlama hatası: {exc}")
        return

    if settings and settings.paper_executes_on_t212 and t212 and getattr(
        settings, "paper_t212_sync_supabase_ledger", True
    ):
        try:
            await broker.sync_ledger_from_t212_client(t212)
        except Exception as exc:
            log.warning("paper_reset_handler T212 resync: %s", exc)

    agent = context.bot_data.get("paper_agent")
    if agent and hasattr(agent, "reset_risk_state"):
        try:
            agent.reset_risk_state()
        except Exception:
            pass

    extra = ""
    if settings and settings.paper_executes_on_t212:
        extra = " T212 demo bakiyesi değişmedi — yalnızca Supabase `paper_*` sıfırlandı."
    await _send_long(
        update,
        f"✅ Paper portföy sıfırlandı. Nakit (defter): *{start_nav:,.2f} {reset_cur}*{extra}\n"
        "(ceza/hafıza tablolarına dokunulmadı).",
    )


async def runpaper_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/runpaper`` — manually trigger a PaperAgent cycle (dev/test)."""
    allowed = context.bot_data.get("allowed_ids", set())
    if not _check_user(update, allowed):
        return

    agent = context.bot_data.get("paper_agent")
    if not agent:
        await _send_long(update, "⚠️ PaperAgent not running / not wired yet.")
        return

    chat_id = update.effective_chat.id if update.effective_chat else None
    if chat_id is None:
        await _send_long(update, "⚠️ Chat context missing.")
        return

    await _send_long(update, "⏳ Running paper cycle...")

    async def _run_and_report() -> None:
        try:
            await agent.run_cycle("MANUAL", allow_trades=True)
            await context.bot.send_message(
                chat_id=chat_id,
                text="✅ Cycle complete. Use `/paper log` to view analysis.",
                parse_mode=ParseMode.HTML,
            )
        except Exception as exc:
            log.error("runpaper_handler error: %s", exc)
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"❌ Cycle error: {html.escape(str(exc))}",
                parse_mode=ParseMode.HTML,
            )

    # Run in background so webhook returns quickly; avoid Telegram timeout.
    asyncio.create_task(_run_and_report())


async def punishments_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/punishments`` — list active punishments (if engine wired)."""
    allowed = context.bot_data.get("allowed_ids", set())
    if not _check_user(update, allowed):
        return

    engine = context.bot_data.get("punishment_engine")
    if not engine:
        await _send_long(update, "⚠️ PunishmentEngine not wired yet.")
        return

    try:
        rows = await engine.get_active_punishments()
    except Exception as exc:
        await _send_long(update, f"❌ DB error: {exc}")
        return

    if not rows:
        await _send_long(update, "No active punishments.")
        return

    lines = ["🚫 *ACTIVE PUNISHMENTS*\n"]
    for p in rows[:50]:
        exp = p.expires_at.isoformat() if p.expires_at else "N/A"
        lines.append(f"- {p.ticker}: {p.penalty_type} until {exp} | {p.reason[:120]}")
    await _send_long(update, "\n".join(lines))
