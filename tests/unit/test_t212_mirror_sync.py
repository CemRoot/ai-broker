from __future__ import annotations

from app.services.paper.t212_mirror_sync import (
    mirror_kwargs_from_filled_order,
    pending_order_drop_without_mirror,
    pending_order_ready_to_mirror,
)


def test_pending_order_ready_filled() -> None:
    assert pending_order_ready_to_mirror(
        {"status": "FILLED", "filledQuantity": 1.0, "filledValue": 100.0, "side": "BUY"}
    )


def test_pending_order_not_ready_partial() -> None:
    assert not pending_order_ready_to_mirror(
        {"status": "PARTIALLY_FILLED", "filledQuantity": 0.5, "filledValue": 50.0, "side": "BUY"}
    )


def test_drop_cancelled_empty() -> None:
    assert pending_order_drop_without_mirror(
        {"status": "CANCELLED", "filledQuantity": 0, "filledValue": 0}
    )


def test_mirror_kwargs_buy() -> None:
    kw = mirror_kwargs_from_filled_order(
        {
            "status": "FILLED",
            "side": "BUY",
            "filledQuantity": 2.0,
            "filledValue": 200.0,
        },
        yf_ticker="AAPL",
        t212_order_id=99,
        meta={"reasoning": "x"},
    )
    assert kw["action"] == "BUY"
    assert kw["shares"] == 2.0
    assert kw["price"] == 100.0
    assert kw["t212_order_id"] == 99
