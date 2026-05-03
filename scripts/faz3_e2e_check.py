#!/usr/bin/env python3
"""
Faz 3 uçtan uca doğrulama (CEO / CI için).

- Ollama modelleri (deepseek-r1:14b, nomic-embed-text)
- Ortam: FMP, Truth token, Supabase DSN (değer yazdırmaz)
- Supabase: ``scripts/check_supabase_faz2.py`` ile aynı kontroller
- Paper: bakiye + istatistik sorguları (``/paper stats`` yolu)
- İsteğe bağlı: tek PaperAgent döngüsü, işlem YOK (``allow_trades=False``) — Groq/Ollama gerekir

Çalıştırma::

    PYTHONPATH=. uv run python scripts/faz3_e2e_check.py
    PYTHONPATH=. uv run python scripts/faz3_e2e_check.py --skip-llm-cycle
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _check_ollama_models() -> tuple[bool, list[str]]:
    try:
        r = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return False, [str(e)]
    if r.returncode != 0:
        return False, [r.stderr or "ollama list failed"]
    out = r.stdout or ""
    need = ("deepseek-r1:14b", "nomic-embed-text")
    missing = [m for m in need if m not in out]
    return len(missing) == 0, missing


def _check_supabase() -> tuple[bool, str]:
    r = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "check_supabase_faz2.py")],
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        capture_output=True,
        text=True,
        timeout=120,
    )
    ok = r.returncode == 0
    tail = (r.stdout or "")[-500:] if r.stdout else (r.stderr or "")
    return ok, f"exit={r.returncode} tail={tail!r}"


async def _check_paper_paths() -> tuple[bool, str]:
    from app.core.config import get_settings
    from app.memory.database import DatabaseSettings, SupabaseDatabase
    from app.services.paper.broker import PaperBroker

    settings = get_settings()
    if not (settings.supabase_db_url or "").strip():
        return False, "SUPABASE_DB_URL empty"

    db = SupabaseDatabase(DatabaseSettings())
    await db.connect(dsn=settings.supabase_db_url or None)
    try:
        broker = PaperBroker(db)
        bal = await broker.get_balance()
        pos = await broker.get_positions()
        tr = await broker.get_recent_trades(limit=5)
        _ = (bal, len(pos), len(tr))
        return True, f"balance_ok cash={bal:.2f} positions={len(pos)} recent_trades={len(tr)}"
    finally:
        await db.close()


async def _optional_llm_cycle(skip: bool) -> tuple[bool, str]:
    if skip:
        return True, "skipped (--skip-llm-cycle)"

    import httpx

    from app.agents.paper_agent import PaperAgent, PaperAgentDeps
    from app.core.config import get_settings
    from app.memory.database import DatabaseSettings, SupabaseDatabase
    from app.services.llm.groq_service import GroqService
    from app.services.llm.ollama_service import OllamaService
    from app.services.market_clock import MarketClock
    from app.services.paper.broker import PaperBroker
    from app.services.screener import SPScreener
    from app.tools.executor import ToolDeps, ToolExecutor

    settings = get_settings()
    if not settings.groq_api_key and not settings.ollama_base_url:
        return False, "no GROQ_API_KEY and no Ollama — cannot run cycle"

    db = SupabaseDatabase(DatabaseSettings())
    await db.connect(dsn=settings.supabase_db_url or None)
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))
    try:
        groq_svc = GroqService(settings) if settings.groq_api_key else None
        ollama_svc = OllamaService(settings)
        paper_broker = PaperBroker(
            db=db,
            paper_executes_on_t212=settings.paper_executes_on_t212,
        )
        market_clock = MarketClock()
        screener = SPScreener(settings=settings, http_client=http_client)
        tool_executor = ToolExecutor(
            ToolDeps(
                settings=settings,
                db=db,
                http_client=http_client,
                groq=groq_svc,
                ollama=ollama_svc,
                retriever=None,
                paper_broker=paper_broker,
                screener=screener,
                t212=None,
            )
        )
        agent = PaperAgent(
            PaperAgentDeps(
                settings=settings,
                db=db,
                paper_broker=paper_broker,
                groq=groq_svc,
                ollama=ollama_svc,
                retriever=None,
                tool_executor=tool_executor,
                market_clock=market_clock,
                telegram_application=None,
                punishment_engine=None,
                position_monitor=None,
                t212=None,
            )
        )
        text, decisions = await agent.run_cycle("E2E_CHECK", allow_trades=False)
        return True, f"cycle_ok analysis_chars={len(text)} decisions={len(decisions)}"
    finally:
        await http_client.aclose()
        await db.close()


async def _async_main(skip_llm: bool) -> int:
    from app.core.config import get_settings

    s = get_settings()
    results: dict[str, object] = {}

    ok, meta = _check_ollama_models()
    results["ollama_deepseek_and_embed"] = {"ok": ok, "detail": meta}
    print("1) Ollama models:", "OK" if ok else f"FAIL missing={meta}")

    fmp = bool((s.fmp_api_key or "").strip())
    results["fmp_api_key"] = {"ok": fmp, "note": "optional for PaperAgent; required for screen_stocks"}
    print(
        "2) FMP_API_KEY:",
        "OK" if fmp else "WARN (empty — S&P screener disabled; set FMP_API_KEY for full Faz 3d)",
    )

    truth = bool((s.truth_social_access_token or "").strip())
    results["truth_social_token"] = {"ok": truth}
    print("3) TRUTH_SOCIAL_ACCESS_TOKEN:", "OK" if truth else "FAIL (empty)")

    sup_ok, sup_msg = _check_supabase()
    results["supabase"] = {"ok": sup_ok, "detail": sup_msg}
    print("4) Supabase:", "OK" if sup_ok else f"FAIL {sup_msg}")

    pap_ok, pap_msg = await _check_paper_paths()
    results["paper_stats_path"] = {"ok": pap_ok, "detail": pap_msg}
    print("5) Paper broker (/paper stats path):", "OK" if pap_ok else f"FAIL {pap_msg}")

    cyc_ok, cyc_msg = await _optional_llm_cycle(skip_llm)
    results["runpaper_cycle_no_trades"] = {"ok": cyc_ok, "detail": cyc_msg}
    print("6) PaperAgent cycle (no trades):", "OK" if cyc_ok else f"FAIL {cyc_msg}")

    print("7) Trump emergency: run `PYTHONPATH=. pytest tests/integration/test_faz3_e2e_checklist.py -q`")
    print(json.dumps({"summary": results}, indent=2))

    # FMP is optional for core paper loop; everything else should pass on a healthy dev machine.
    hard = [ok, truth, sup_ok, pap_ok, cyc_ok]
    return 0 if all(hard) else 1


def main() -> None:
    p = argparse.ArgumentParser(description="Faz 3 E2E checklist runner")
    p.add_argument(
        "--skip-llm-cycle",
        action="store_true",
        help="Do not call Groq/Ollama for a dry PaperAgent cycle",
    )
    args = p.parse_args()
    code = asyncio.run(_async_main(skip_llm=args.skip_llm_cycle))
    sys.exit(code)


if __name__ == "__main__":
    main()
