"""Background poll: T212 pending orders → ``paper_trades`` mirror when filled (incl. app-placed)."""

from __future__ import annotations

import asyncio
import logging

import httpx

from app.core.config import Settings
from app.memory.database import SupabaseDatabase
from app.services.paper.broker import PaperBroker
from app.services.paper.t212_mirror_sync import (
    mirror_kwargs_from_filled_order,
    mirror_kwargs_from_historical_item,
    pending_order_drop_without_mirror,
    pending_order_ready_to_mirror,
)
from app.services.paper.t212_pending_mirror_store import (
    delete_t212_pending_mirror,
    enqueue_t212_pending_mirror,
    list_t212_pending_mirror,
    pending_mirror_row_exists,
    touch_t212_pending_poll,
)
from app.services.t212.client import T212Client
from app.services.t212.ticker_map import t212_to_yfinance, yfinance_to_t212

log = logging.getLogger(__name__)


class T212MirrorPoller:
    """Respects T212 rate limits: ~1 req/s between per-order polls; history paginated slowly."""

    def __init__(
        self,
        *,
        settings: Settings,
        db: SupabaseDatabase,
        t212: T212Client,
        paper_broker: PaperBroker,
    ) -> None:
        self._settings = settings
        self._db = db
        self._t212 = t212
        self._paper_broker = paper_broker
        self._reconcile_disabled_runtime = False
        self._history_disabled_runtime = False

    async def run_forever(self) -> None:
        interval = max(15, int(self._settings.paper_t212_pending_poll_sec))
        log.info("T212MirrorPoller interval=%ss reconcile=%s", interval, self._settings.paper_t212_reconcile_external_orders)
        while True:
            try:
                if self._settings.paper_executes_on_t212 and self._settings.paper_t212_mirror_poller_enabled:
                    await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("T212MirrorPoller tick failed")
            await asyncio.sleep(float(interval))

    async def tick(self) -> None:
        await self._reconcile_external_pending_orders()
        rows = await list_t212_pending_mirror(self._db)
        for row in rows:
            await self._process_pending_row(row)
            await asyncio.sleep(1.06)
        if self._settings.paper_t212_sync_supabase_ledger:
            try:
                await self._paper_broker.sync_ledger_from_t212_client(self._t212)
            except Exception as exc:
                log.warning("T212 → Supabase ledger sync failed: %s", exc)

    async def _reconcile_external_pending_orders(self) -> None:
        if not self._settings.paper_t212_reconcile_external_orders:
            return
        if self._reconcile_disabled_runtime:
            return
        try:
            orders = await self._t212.get_all_pending_orders()
        except httpx.HTTPStatusError as exc:
            if exc.response is not None and exc.response.status_code == 403:
                log.warning(
                    "T212 GET /equity/orders → 403; reconcile disabled. "
                    "API key needs 'orders:read' scope to mirror app/web pending orders."
                )
                self._reconcile_disabled_runtime = True
                return
            log.warning("T212 GET /equity/orders (reconcile) failed: %s", exc)
            return
        except Exception as exc:
            log.warning("T212 GET /equity/orders (reconcile) failed: %s", exc)
            return
        for o in orders:
            oid = o.get("id")
            if oid is None:
                continue
            try:
                oid_int = int(oid)
            except (TypeError, ValueError):
                continue
            if await self._paper_broker.t212_mirror_trade_exists(oid_int):
                continue
            if await pending_mirror_row_exists(self._db, oid_int):
                continue
            t212_sym = str(o.get("ticker") or "").strip()
            if not t212_sym:
                continue
            yf = t212_to_yfinance(t212_sym)
            side = str(o.get("side") or "").upper()
            if side not in ("BUY", "SELL"):
                continue
            meta = {
                "reasoning": "T212 pending order (reconcile: app/web/API). Mirror on fill only.",
                "stop_loss": None,
                "target": None,
                "invalidation_condition": None,
                "chain_of_thought": None,
                "cycle_event": "T212_RECONCILE",
                "emergency": False,
                "source": "external",
                "initiated_from": o.get("initiatedFrom"),
            }
            await enqueue_t212_pending_mirror(
                self._db,
                t212_order_id=oid_int,
                yf_ticker=yf,
                action=side,
                meta=meta,
            )

    async def _process_pending_row(self, row: dict) -> None:
        oid = int(row["t212_order_id"])
        yf = str(row["yf_ticker"])
        meta = row.get("meta") or {}
        if not isinstance(meta, dict):
            meta = {}
        if await self._paper_broker.t212_mirror_trade_exists(oid):
            await delete_t212_pending_mirror(self._db, oid)
            return

        po = await self._t212.get_pending_order(oid)
        if po is not None:
            if pending_order_drop_without_mirror(po):
                log.info("T212 pending mirror removed (cancelled/rejected, no fill) order_id=%s", oid)
                await delete_t212_pending_mirror(self._db, oid)
                return
            if pending_order_ready_to_mirror(po):
                try:
                    kwargs = mirror_kwargs_from_filled_order(
                        po, yf_ticker=yf, t212_order_id=oid, meta=meta
                    )
                    await self._paper_broker.record_mirror_trade(**kwargs)
                    log.info("T212 mirror from pending GET order_id=%s", oid)
                except Exception as exc:
                    log.warning("T212 mirror from pending GET failed order_id=%s: %s", oid, exc)
                await delete_t212_pending_mirror(self._db, oid)
                return
            await touch_t212_pending_poll(self._db, oid)
            return

        t212_sym = yfinance_to_t212(yf)
        hist = await self._find_history_item(oid, t212_sym)
        if hist is not None:
            try:
                kwargs = mirror_kwargs_from_historical_item(
                    hist, yf_ticker=yf, t212_order_id=oid, meta=meta
                )
                await self._paper_broker.record_mirror_trade(**kwargs)
                log.info("T212 mirror from history order_id=%s", oid)
            except Exception as exc:
                log.warning("T212 mirror from history failed order_id=%s: %s", oid, exc)
            await delete_t212_pending_mirror(self._db, oid)
            return

        await touch_t212_pending_poll(self._db, oid)

    async def _find_history_item(self, order_id: int, t212_ticker: str) -> dict | None:
        if self._history_disabled_runtime:
            return None
        next_path: str | None = None
        pages = 0
        while pages < 8:
            if pages > 0:
                await asyncio.sleep(11.0)
            try:
                if next_path is None:
                    data = await self._t212.get_history_orders(limit=50, ticker=t212_ticker)
                else:
                    data = await self._t212.get_history_orders(next_page_path=next_path)
            except httpx.HTTPStatusError as exc:
                if exc.response is not None and exc.response.status_code == 403:
                    log.warning(
                        "T212 GET /equity/history/orders → 403; history search disabled. "
                        "API key needs 'history:orders' scope for fill recovery after pending TTL."
                    )
                    self._history_disabled_runtime = True
                return None
            for item in data.get("items") or []:
                o = item.get("order") or {}
                if o.get("id") is not None and int(o["id"]) == int(order_id):
                    return item
            np = data.get("nextPagePath")
            if not np:
                break
            next_path = str(np)
            pages += 1
        return None
