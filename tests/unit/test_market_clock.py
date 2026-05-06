from app.services.market_clock import MarketClock
from app.services.market_clock import ET
from datetime import datetime
import pytest


def test_market_clock_instantiates():
    mc = MarketClock()
    now = mc.now_et()
    assert now.tzinfo is not None


def test_market_clock_next_decision_event_returns_tuple():
    mc = MarketClock()
    dt, ev = mc.next_decision_event()
    assert hasattr(dt, "tzinfo")
    assert isinstance(ev, str)
    assert ev in {"OPEN", "MIDDAY", "CLOSE"}


@pytest.mark.asyncio
async def test_wait_for_next_tick_during_open_session_does_not_sleep_until_tomorrow(monkeypatch):
    mc = MarketClock(regular_tick_seconds=1800)
    now = datetime(2026, 5, 6, 11, 20, 0, tzinfo=ET)  # regular session (ET)
    slept: list[int] = []

    monkeypatch.setattr(mc, "now_et", lambda: now)
    monkeypatch.setattr(mc, "is_market_open", lambda when=None: True)
    monkeypatch.setattr(
        mc,
        "next_decision_event",
        lambda when=None: (datetime(2026, 5, 6, 12, 0, 0, tzinfo=ET), "MIDDAY"),
    )

    async def _fake_sleep(seconds: int) -> None:
        slept.append(int(seconds))

    monkeypatch.setattr("app.services.market_clock.asyncio.sleep", _fake_sleep)

    event, cadence = await mc.wait_for_next_tick()

    assert event == "TICK"
    assert cadence == 1800
    assert slept == [1800]

