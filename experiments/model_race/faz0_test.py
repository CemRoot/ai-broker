"""
Faz 0 — model race: T212 demo portfolio + AMD technicals, same prompt to three API/local LLMs.
"""

from __future__ import annotations

import ast
import asyncio
import base64
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

CEREBRAS_MODEL_ID = "llama3.1-8b"
GROQ_MODEL_ID = "llama-3.3-70b-versatile"

GROQ_DAILY_REQUEST_LIMIT = 14_400
PAPER_AGENT_LOOPS_PER_DAY = 96
# Faz 1 öncesi varsayım: yoğun pipeline’da döngü başına çoklu Groq tamamlama (~55k/gün senaryosu)
ESTIMATED_GROQ_REQUESTS_PER_PAPER_LOOP = 573
ANALIZ_MULTIPLIER_FOR_TOKEN_ESTIMATE = 10

import httpx
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from dotenv import load_dotenv
from ollama import Client as OllamaClient
from rich.console import Console
from rich.table import Table

load_dotenv()

MOCK_POSITIONS: list[dict[str, Any]] = [
    {"ticker": "AMD", "quantity": 7.8, "averagePrice": 167.0, "currentPrice": 245.0},
]

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    if v is not None and v.strip() != "":
        return v.strip()
    return default


def extract_positions(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = (
            payload.get("positions")
            or payload.get("openPositions")
            or payload.get("items")
            or payload.get("data")
            or []
        )
    else:
        items = []

    out: list[dict[str, Any]] = []
    for p in items:
        if not isinstance(p, dict):
            continue
        ticker = (
            p.get("ticker")
            or p.get("symbol")
            or p.get("instrumentTicker")
            or p.get("instrumentCode")
        )
        if not ticker:
            continue
        qty = float(
            p.get("quantity")
            or p.get("qty")
            or p.get("totalQuantity")
            or p.get("size")
            or 0
        )
        avg = float(
            p.get("averagePrice")
            or p.get("averageBuyPrice")
            or p.get("avgPrice")
            or 0
        )
        cur = float(
            p.get("currentPrice")
            or p.get("currentPriceInAccountCurrency")
            or p.get("price")
            or avg
        )
        out.append(
            {
                "ticker": str(ticker),
                "quantity": qty,
                "averagePrice": avg,
                "currentPrice": cur,
            }
        )
    return out


T212_DEMO_PORTFOLIO_URL = "https://demo.trading212.com/api/v0/equity/portfolio"


async def fetch_t212_portfolio() -> tuple[list[dict[str, Any]], bool, str | None]:
    """
    Trading 212 Public API (Faz 0 demo):
    - Auth: HTTP Basic, Base64(api_key:api_secret).
    - GET https://demo.trading212.com/api/v0/equity/portfolio (fixed URL for this script).
    """
    key = _env("T212_DEMO_API_KEY")
    secret = _env("T212_DEMO_API_SECRET")

    if not key or not secret:
        return (
            MOCK_POSITIONS[:3],
            True,
            "missing T212_DEMO_API_KEY or T212_DEMO_API_SECRET "
            "(Public API requires both for Basic auth)",
        )

    token = base64.b64encode(f"{key}:{secret}".encode()).decode("ascii")
    headers = {"Authorization": f"Basic {token}"}
    url = T212_DEMO_PORTFOLIO_URL

    err: str | None = None
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(url, headers=headers, timeout=30.0)
            if r.status_code >= 400:
                err = f"HTTP {r.status_code}: {r.text[:200]}"
            else:
                data = r.json()
                positions = extract_positions(data)
                return positions[:3], False, None
    except Exception as e:
        err = str(e)

    return MOCK_POSITIONS[:3], True, err


def fetch_amd_technicals() -> tuple[dict[str, Any], str | None]:
    try:
        df = yf.download("AMD", period="3mo", interval="1d", progress=False, auto_adjust=True)
        if df is None or df.empty:
            return {}, "no OHLCV for AMD"

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

        close = df["Close"].astype(float)
        rsi_series = ta.rsi(close, length=14)
        sma_series = ta.sma(close, length=20)

        rsi_val = float(rsi_series.iloc[-1]) if rsi_series is not None and not pd.isna(rsi_series.iloc[-1]) else None
        sma_val = float(sma_series.iloc[-1]) if sma_series is not None and not pd.isna(sma_series.iloc[-1]) else None
        price = float(close.iloc[-1])

        return (
            {
                "symbol": "AMD",
                "rsi_14": rsi_val,
                "sma_20": sma_val,
                "last_close": price,
            },
            None,
        )
    except Exception as e:
        return {}, str(e)


def build_prompt(t212_data: list[dict[str, Any]], tech: dict[str, Any]) -> str:
    rsi_f = float(tech["rsi_14"]) if tech.get("rsi_14") is not None else 0.0
    sma_f = float(tech["sma_20"]) if tech.get("sma_20") is not None else 0.0
    price_f = float(tech["last_close"]) if tech.get("last_close") is not None else 0.0
    portfolio = json.dumps(t212_data, ensure_ascii=False)
    return f"""Portföy: {portfolio}
AMD Teknik: RSI={rsi_f:.1f}, SMA20={sma_f:.2f}, Fiyat={price_f:.2f}

Sadece şu JSON'ı döndür, başka hiçbir şey yazma (geçerli JSON: çift tırnak kullan):
{{
  "action": "BUY veya SELL veya HOLD",
  "confidence": 0.0,
  "stop_loss": 0.0,
  "target": 0.0,
  "reasoning": "kısa açıklama max 20 kelime"
}}""".strip()


def strip_json_fences(text: str) -> str:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text


def parse_model_json(raw: str) -> tuple[bool, dict[str, Any] | None, str]:
    text = strip_json_fences(raw)
    try:
        return True, json.loads(text), ""
    except json.JSONDecodeError as e1:
        try:
            obj = ast.literal_eval(text)
            if isinstance(obj, dict):
                return True, obj, ""
        except (SyntaxError, ValueError):
            pass
        return False, None, str(e1)


def _is_ollama_unreachable(exc: BaseException) -> bool:
    err = str(exc).lower()
    needles = (
        "connection refused",
        "failed to connect",
        "connect error",
        "connection error",
        "timeout",
        "name or service not known",
        "actively refused",
        "errno 61",
        "errno 111",
    )
    return any(n in err for n in needles) or isinstance(
        exc, (ConnectionError, TimeoutError, OSError)
    )


def run_ollama(prompt: str) -> tuple[str, None]:
    host = _env("OLLAMA_BASE_URL", "http://localhost:11434")
    model = _env("OLLAMA_MODEL", "deepseek-r1:14b")
    try:
        client = OllamaClient(host=host)
        resp = client.chat(
            model=model or "deepseek-r1:14b",
            messages=[{"role": "user", "content": prompt}],
        )
        return (resp.message.content or "", None)
    except Exception as e:
        if _is_ollama_unreachable(e):
            raise RuntimeError("Ollama offline") from e
        raise


def _cerebras_is_model_missing(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(
        x in msg
        for x in (
            "model_not_found",
            "not_found_error",
            "does not exist",
            '"code": "model_not_found"',
            "error code: 404",
        )
    )


def _usage_dict_from_counts(
    prompt_tokens: int | None, completion_tokens: int | None, total_tokens: int | None
) -> dict[str, int] | None:
    if prompt_tokens is None and completion_tokens is None and total_tokens is None:
        return None
    pi = int(prompt_tokens or 0)
    co = int(completion_tokens or 0)
    tot = int(total_tokens if total_tokens is not None else pi + co)
    return {"input": pi, "output": co, "total": tot}


def run_cerebras(prompt: str) -> tuple[str, dict[str, int] | None]:
    """Cerebras: fixed Faz 0 model id (no env override, no fallback)."""
    from cerebras.cloud.sdk import Cerebras

    api_key = _env("CEREBRAS_API_KEY")
    if not api_key:
        raise ValueError("CEREBRAS_API_KEY not set")
    client = Cerebras(api_key=api_key)
    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=CEREBRAS_MODEL_ID,
        )
        text = (completion.choices[0].message.content or "").strip()
        u = completion.usage
        usage = _usage_dict_from_counts(
            getattr(u, "prompt_tokens", None),
            getattr(u, "completion_tokens", None),
            getattr(u, "total_tokens", None),
        )
        return text, usage
    except Exception as e:
        if _cerebras_is_model_missing(e):
            raise RuntimeError(f"Model bulunamadı: {CEREBRAS_MODEL_ID}") from e
        raise


def run_groq(prompt: str) -> tuple[str, dict[str, int] | None]:
    from groq import Groq

    api_key = _env("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")
    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=GROQ_MODEL_ID,
    )
    text = (completion.choices[0].message.content or "").strip()
    u = completion.usage
    usage = None
    if u is not None:
        usage = _usage_dict_from_counts(
            getattr(u, "prompt_tokens", None),
            getattr(u, "completion_tokens", None),
            getattr(u, "total_tokens", None),
        )
    return text, usage


def _truncate(s: str, max_len: int = 80) -> str:
    s = s.strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


async def timed_model_call(name: str, fn: Callable[..., Any], prompt: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    preview = ""
    err: str | None = None
    raw = ""
    usage: dict[str, int] | None = None
    try:
        out = await asyncio.to_thread(fn, prompt)
        if isinstance(out, tuple) and len(out) == 2:
            raw, usage = out[0], out[1]
        else:
            raw = out  # type: ignore[assignment]
        preview = (raw or "")[:200]
    except Exception as e:
        err = str(e) or "unknown error"
    elapsed = time.perf_counter() - t0

    json_ok = False
    parsed: dict[str, Any] | None = None
    parse_err = ""
    if raw and not err:
        json_ok, parsed, parse_err = parse_model_json(raw)

    action = ""
    confidence: Any = ""
    reasoning = ""
    if parsed:
        action = str(parsed.get("action", ""))
        confidence = parsed.get("confidence", "")
        reasoning = str(parsed.get("reasoning", "") or "")

    return {
        "model": name,
        "seconds": round(elapsed, 3),
        "json_ok": json_ok,
        "action": action or ("—" if err else ""),
        "confidence": confidence if confidence != "" else ("—" if err else ""),
        "reasoning": reasoning if reasoning else ("—" if err else ""),
        "reasoning_full": reasoning,
        "error": err,
        "parse_error": parse_err if not json_ok and raw and not err else None,
        "response_preview": preview,
        "raw_response": raw if err else None,
        "usage": usage,
    }


def build_prompt_scenario_portfolio_only(t212_data: list[dict[str, Any]]) -> str:
    portfolio = json.dumps(t212_data, ensure_ascii=False)
    return f"""Portföy: {portfolio}
Teknik veri yok, sadece pozisyonlara bak.
En riskli pozisyonu belirle.
Sadece geçerli JSON döndür (çift tırnak), başka metin yok:
{{
  "ticker": "SYMBOL",
  "risk_level": "low veya medium veya high",
  "reasoning": "kısa gerekçe"
}}""".strip()


def build_prompt_scenario_amd_urgent() -> str:
    return f"""AMD fiyatı son 1 saatte -%5 düştü.
RSI=45, hacim 3x ortalamanın üstünde.
Panik satış mı, fırsat mı?
Sadece geçerli JSON döndür (çift tırnak), başka metin yok:
{{
  "action": "BUY veya SELL veya HOLD",
  "confidence": 0.0,
  "reasoning": "kısa gerekçe"
}}""".strip()


async def run_groq_scenarios(
    t212_data: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    specs: list[tuple[str, str, Callable[[], str]]] = [
        ("scenario_2", "portfolio_risk_no_technicals", lambda: build_prompt_scenario_portfolio_only(t212_data)),
        ("scenario_3", "amd_urgent_intraday", build_prompt_scenario_amd_urgent),
    ]
    out: list[dict[str, Any]] = []
    for sid, label, builder in specs:
        prompt = builder()
        t0 = time.perf_counter()
        err: str | None = None
        raw = ""
        usage: dict[str, int] | None = None
        try:
            raw, usage = await asyncio.to_thread(run_groq, prompt)
        except Exception as e:
            err = str(e) or "unknown error"
        elapsed = time.perf_counter() - t0
        json_ok = False
        parsed: dict[str, Any] | None = None
        parse_err = ""
        if raw and not err:
            json_ok, parsed, parse_err = parse_model_json(raw)
        out.append(
            {
                "id": sid,
                "label": label,
                "model": f"Groq {GROQ_MODEL_ID}",
                "seconds": round(elapsed, 3),
                "json_ok": json_ok,
                "parsed": parsed,
                "parse_error": parse_err if not json_ok and raw and not err else None,
                "error": err,
                "usage": usage,
                "prompt": prompt,
                "raw_preview": (raw or "")[:500] if err else None,
            }
        )
    return out


def _usage_from_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"groq": None, "cerebras": None, "ollama": None}
    for r in results:
        m = r.get("model", "")
        u = r.get("usage")
        if m.startswith("Groq"):
            out["groq"] = u
        elif "Cerebras" in m:
            out["cerebras"] = u
        elif "Ollama" in m:
            out["ollama"] = u
    return out


def _print_token_usage_box(console: Console, results: list[dict[str, Any]]) -> None:
    groq_u = next((r.get("usage") for r in results if r.get("model", "").startswith("Groq") and r.get("usage")), None)
    cerebras_u = next(
        (r.get("usage") for r in results if "Cerebras" in r.get("model", "") and r.get("usage")), None
    )

    def fmt(u: dict[str, int] | None) -> str:
        if not u:
            return "— (yok veya hata)"
        return f"{u['input']} input + {u['output']} output (total {u['total']})"

    g10_in = (groq_u or {}).get("input", 0) * ANALIZ_MULTIPLIER_FOR_TOKEN_ESTIMATE
    g10_out = (groq_u or {}).get("output", 0) * ANALIZ_MULTIPLIER_FOR_TOKEN_ESTIMATE
    g10_tot = (groq_u or {}).get("total", 0) * ANALIZ_MULTIPLIER_FOR_TOKEN_ESTIMATE

    est_daily_req = PAPER_AGENT_LOOPS_PER_DAY * ESTIMATED_GROQ_REQUESTS_PER_PAPER_LOOP
    risk = "YÜKSEK" if est_daily_req > GROQ_DAILY_REQUEST_LIMIT else "DÜŞÜK"

    def short(s: str, n: int = 40) -> str:
        return s if len(s) <= n else s[: n - 1] + "…"

    lines = [
        "┌──────────────────────────────────────────┐",
        "│ TOKEN KULLANIMI                            │",
        f"│ Groq:     {short(fmt(groq_u))}",
        f"│ Cerebras: {short(fmt(cerebras_u))}",
        f"│ Tahmini günlük ({ANALIZ_MULTIPLIER_FOR_TOKEN_ESTIMATE} analiz, Groq token)",
        f"│   ~{g10_in} input + ~{g10_out} output (~{g10_tot} total)",
        f"│ Groq req/gün limiti: {GROQ_DAILY_REQUEST_LIMIT}",
        f"│ Paper agent ({PAPER_AGENT_LOOPS_PER_DAY} döngü/gün)",
        f"│   Tahmini ~{est_daily_req:,} req (varsayılan yük modeli)",
        f"│ ⚠ LİMİT AŞIMI RİSKİ: {risk}",
        "└──────────────────────────────────────────┘",
    ]
    console.print("\n".join(lines))


async def main() -> None:
    t212_data, used_mock, t212_err = await fetch_t212_portfolio()
    tech, tech_err = fetch_amd_technicals()
    prompt = build_prompt(t212_data, tech)

    model_jobs = [
        timed_model_call("Ollama deepseek-r1:14b", run_ollama, prompt),
        timed_model_call(f"Cerebras {CEREBRAS_MODEL_ID}", run_cerebras, prompt),
        timed_model_call(f"Groq {GROQ_MODEL_ID}", run_groq, prompt),
    ]
    results = await asyncio.gather(*model_jobs)

    scenarios = await run_groq_scenarios(t212_data)

    payload = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "prompt": prompt,
        "t212": {
            "used_mock": used_mock,
            "error": t212_err,
            "positions": t212_data,
        },
        "technical": {"data": tech, "error": tech_err},
        "models": results,
        "usage": _usage_from_results(results),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "faz0_results.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    scenarios_path = RESULTS_DIR / "faz0_scenarios.json"
    scenarios_path.write_text(
        json.dumps(
            {
                "run_at": datetime.now(timezone.utc).isoformat(),
                "groq_model": GROQ_MODEL_ID,
                "scenarios": scenarios,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    console = Console()
    table = Table(title="Faz 0 — Model race")
    table.add_column("Model", style="cyan")
    table.add_column("Süre(s)", justify="right")
    table.add_column("JSON OK", justify="center")
    table.add_column("Action")
    table.add_column("Confidence")
    table.add_column("Reasoning")

    for row in results:
        if row.get("error"):
            reason_cell = str(row["error"])
        else:
            reason_cell = _truncate(str(row.get("reasoning_full") or row.get("reasoning") or "—"), 100)
        table.add_row(
            row["model"],
            str(row["seconds"]),
            "✓" if row["json_ok"] else "✗",
            row.get("action") or "—",
            str(row.get("confidence", "—")),
            reason_cell,
        )

    console.print(table)

    scen_table = Table(title="Faz 0 — Groq ek senaryolar")
    scen_table.add_column("Senaryo", style="cyan")
    scen_table.add_column("Süre(s)", justify="right")
    scen_table.add_column("JSON OK", justify="center")
    scen_table.add_column("Not")
    for s in scenarios:
        note = str(s.get("error") or "") or (
            "parse: " + str(s.get("parse_error") or "ok") if not s.get("json_ok") else "ok"
        )
        scen_table.add_row(s.get("label", ""), str(s.get("seconds", "")), "✓" if s.get("json_ok") else "✗", note[:60])
    console.print(scen_table)

    _print_token_usage_box(console, results)
    console.print(f"\n[green]Saved:[/green] {out_path}")
    console.print(f"[green]Saved:[/green] {scenarios_path} (Groq senaryolar)")
    if t212_err and used_mock:
        console.print(f"[yellow]T212:[/yellow] {t212_err}")
    if tech_err:
        console.print(f"[yellow]Technical:[/yellow] {tech_err}")


if __name__ == "__main__":
    asyncio.run(main())
