from app.services.market_clock import MarketClock


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

