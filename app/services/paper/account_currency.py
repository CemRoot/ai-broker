"""Paper / T212 account currency (prompts, Telegram — not for duplicating T212 FX)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.core.config import Settings
    from app.services.t212.client import T212Client


async def resolve_paper_account_currency(
    settings: "Settings",
    t212: "T212Client | None",
) -> str:
    """ISO 4217 code: T212 summary when execution is on T212, else ``PAPER_ACCOUNT_CURRENCY``."""
    if settings.paper_executes_on_t212 and t212:
        try:
            s = await t212.get_account_summary()
            c = (s.get("currency") or "").strip().upper()
            if c:
                return c[:3]
        except Exception:
            pass
    raw = (getattr(settings, "paper_account_currency", None) or "USD").strip().upper()
    return (raw[:3] if raw else "USD") or "USD"
