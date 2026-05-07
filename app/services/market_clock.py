"""
MarketClock — asyncio-friendly NYSE session clock (no APScheduler/cron).

Design goals (Faz 3):
- Keep the PaperBroker event-driven; the agent loop can "wait" for time-based events.
- Use `exchange-calendars` for correct trading days (holidays, early closes).
- Provide a dynamic cadence: premarket can tick every 5 minutes; regular session ticks
  at a lower frequency, plus key decision moments.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from app.core.logging import get_logger
from app.core.debug_probe import debug_probe

log = get_logger("market_clock")

ET = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class MarketEvent:
    when_et: str  # "HH:MM"
    event_type: str  # OPEN / MIDDAY / CLOSE


class MarketClock:
    """NYSE clock using exchange-calendars ('XNYS')."""

    EVENTS: list[MarketEvent] = [
        MarketEvent("09:30", "OPEN"),
        MarketEvent("12:00", "MIDDAY"),
        MarketEvent("15:45", "CLOSE"),
    ]

    def __init__(
        self,
        *,
        premarket_window_minutes: int = 120,
        premarket_tick_seconds: int = 300,  # 5m
        regular_tick_seconds: int = 1800,  # 30m baseline
    ) -> None:
        self.premarket_window_minutes = max(0, int(premarket_window_minutes))
        self.premarket_tick_seconds = max(30, int(premarket_tick_seconds))
        self.regular_tick_seconds = max(60, int(regular_tick_seconds))
        self._cal = None
        self._fired_event_day: date | None = None
        self._fired_events: set[str] = set()

    def _calendar(self):
        if self._cal is None:
            from exchange_calendars import get_calendar

            self._cal = get_calendar("XNYS")
        return self._cal

    def now_et(self) -> datetime:
        return datetime.now(tz=ET)

    def is_market_open(self, when: datetime | None = None) -> bool:
        """True if market is open at the given moment (regular session minutes)."""
        dt = when or self.now_et()
        cal = self._calendar()
        # exchange-calendars expects pandas.Timestamp-like; it accepts tz-aware datetime.
        try:
            return bool(cal.is_open_on_minute(dt))
        except Exception:
            return False

    def next_open(self, when: datetime | None = None) -> datetime:
        """Next market open from `when` (ET)."""
        dt = when or self.now_et()
        cal = self._calendar()
        return cal.next_open(dt).to_pydatetime().astimezone(ET)

    def next_close(self, when: datetime | None = None) -> datetime:
        """Next market close from `when` (ET)."""
        dt = when or self.now_et()
        cal = self._calendar()
        return cal.next_close(dt).to_pydatetime().astimezone(ET)

    def _event_dt_on_day(self, day_et: datetime, when_et: str) -> datetime:
        hh, mm = when_et.split(":")
        return day_et.replace(hour=int(hh), minute=int(mm), second=0, microsecond=0)

    def _reset_fired_events(self, now: datetime) -> None:
        today = now.date()
        if self._fired_event_day != today:
            self._fired_event_day = today
            self._fired_events = set()

    def _next_unfired_event(self, now: datetime) -> MarketEvent | None:
        for ev in self.EVENTS:
            if ev.event_type in self._fired_events:
                continue
            dt = self._event_dt_on_day(now, ev.when_et)
            if dt <= now:
                return ev
        return None

    def _mark_fired(self, event_type: str) -> None:
        if event_type:
            self._fired_events.add(event_type)

    def next_decision_event(self, when: datetime | None = None) -> tuple[datetime, str]:
        """Return (datetime_et, event_type) for the next decision moment."""
        now = when or self.now_et()
        close_dt = self.next_close(now)
        day = now

        candidates: list[tuple[datetime, str]] = []
        for ev in self.EVENTS:
            dt = self._event_dt_on_day(day, ev.when_et)
            if dt > now and dt <= close_dt:
                candidates.append((dt, ev.event_type))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0]

        # No more events today → next open is the next OPEN decision.
        nxt_open = self.next_open(now)
        return nxt_open, "OPEN"

    async def wait_for_next_tick(self) -> tuple[str, int]:
        """
        Sleep until the next meaningful tick/event and return (event_type, suggested_cadence_seconds).

        event_type:
        - PREMARKET: premarket preparation tick
        - OPEN/MIDDAY/CLOSE: key decision times
        - TICK: regular low-frequency tick during session
        """
        now = self.now_et()
        self._reset_fired_events(now)

        # Critical ordering: during regular session, next_open() points to the NEXT day.
        # If we compute premarket window first, we can accidentally sleep all day.
        if self.is_market_open(now):
            pending = self._next_unfired_event(now)
            if pending:
                self._mark_fired(pending.event_type)
                # region agent log
                debug_probe(
                    run_id="pre-fix",
                    hypothesis_id="H5",
                    location="app/services/market_clock.py:151",
                    message="market_open pending event fired",
                    data={"event_type": pending.event_type, "now_et": now.isoformat()},
                )
                # endregion
                return pending.event_type, self.regular_tick_seconds

            next_ev_dt, next_ev_type = self.next_decision_event(now)
            to_ev = max(0, int((next_ev_dt - now).total_seconds()))

            # If we're at/after the event moment, fire it.
            if to_ev <= 3:
                self._mark_fired(next_ev_type)
                return next_ev_type, self.regular_tick_seconds

            # Otherwise tick at min(regular_tick, time-to-event)
            sleep_s = min(self.regular_tick_seconds, max(1, to_ev))
            await asyncio.sleep(sleep_s)
            # We may have crossed into event time; caller will re-check next loop.
            # region agent log
            debug_probe(
                run_id="pre-fix",
                hypothesis_id="H3",
                location="app/services/market_clock.py:171",
                message="market_open tick sleep",
                data={"sleep_s": sleep_s, "now_et": now.isoformat()},
            )
            # endregion
            return "TICK", self.regular_tick_seconds

        nxt_open = self.next_open(now)

        # Premarket window: [open - window, open)
        pre_start = nxt_open - timedelta(minutes=self.premarket_window_minutes)

        if now < pre_start:
            sleep_s = max(1, int((pre_start - now).total_seconds()))
            log.info("MarketClock sleeping until premarket window (%ss)", sleep_s)
            # region agent log
            debug_probe(
                run_id="pre-fix",
                hypothesis_id="H5",
                location="app/services/market_clock.py:181",
                message="market_closed sleep to premarket",
                data={"sleep_s": sleep_s, "now_et": now.isoformat(), "next_open_et": nxt_open.isoformat()},
            )
            # endregion
            await asyncio.sleep(sleep_s)
            return "PREMARKET", self.premarket_tick_seconds

        if now < nxt_open:
            # Inside premarket window
            sleep_s = min(
                self.premarket_tick_seconds,
                max(1, int((nxt_open - now).total_seconds())),
            )
            await asyncio.sleep(sleep_s)
            return "PREMARKET", self.premarket_tick_seconds

        # After close: wait until next premarket window (or open if window=0)
        sleep_to = pre_start if now < pre_start else self.next_open(now) - timedelta(
            minutes=self.premarket_window_minutes
        )
        sleep_s = max(60, int((sleep_to - now).total_seconds()))
        await asyncio.sleep(sleep_s)
        return "PREMARKET", self.premarket_tick_seconds
