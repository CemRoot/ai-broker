"""Derive ``record_mirror_trade`` kwargs from T212 Order / history payloads (filled only)."""

from __future__ import annotations

from typing import Any


def _meta_defaults(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "reasoning": str(meta.get("reasoning") or ""),
        "stop_loss": meta.get("stop_loss"),
        "target": meta.get("target"),
        "invalidation_condition": meta.get("invalidation_condition"),
        "chain_of_thought": meta.get("chain_of_thought"),
        "cycle_event": meta.get("cycle_event"),
        "emergency": bool(meta.get("emergency", False)),
    }


def pending_order_ready_to_mirror(order: dict[str, Any]) -> bool:
    """FILLED, or CANCELLED/REJECTED after a partial leg (non-zero fill)."""
    st = str(order.get("status") or "")
    fq = float(order.get("filledQuantity") or 0)
    fv = float(order.get("filledValue") or 0)
    if fq <= 0 or fv == 0:
        return False
    if st == "FILLED":
        return True
    if st in ("CANCELLED", "REJECTED"):
        return True
    return False


def pending_order_drop_without_mirror(order: dict[str, Any]) -> bool:
    """Terminal with no fill — remove from pending queue."""
    st = str(order.get("status") or "")
    fq = float(order.get("filledQuantity") or 0)
    return st in ("CANCELLED", "REJECTED") and fq <= 0


def mirror_kwargs_from_filled_order(
    order: dict[str, Any],
    *,
    yf_ticker: str,
    t212_order_id: int,
    meta: dict[str, Any],
) -> dict[str, Any]:
    fq = float(order.get("filledQuantity") or 0)
    fv = float(order.get("filledValue") or 0)
    if fq <= 0 or fv == 0:
        raise ValueError("order not filled")
    side = str(order.get("side") or "").upper()
    if side == "BUY":
        action = "BUY"
    elif side == "SELL":
        action = "SELL"
    else:
        raise ValueError(f"unknown side {side!r}")
    exec_price = abs(fv / fq)
    exec_shares = fq
    total_row = abs(fv)
    m = _meta_defaults(meta)
    out: dict[str, Any] = {
        "ticker": yf_ticker.upper().strip(),
        "action": action,
        "shares": exec_shares,
        "price": exec_price,
        "total_value": total_row,
        "reasoning": m["reasoning"],
        "stop_loss": m["stop_loss"],
        "target": m["target"],
        "invalidation_condition": m["invalidation_condition"],
        "chain_of_thought": m["chain_of_thought"],
        "cycle_event": m["cycle_event"],
        "emergency": m["emergency"],
        "t212_order_id": int(t212_order_id),
        "execution_broker": "t212",
    }
    if action == "SELL":
        avg = meta.get("avg_price_paid")
        if avg is not None:
            avg_f = float(avg)
            cost_basis = exec_shares * avg_f
            realized = total_row - cost_basis
            out["realized_pnl_usd"] = realized
            out["pnl_percent"] = (realized / cost_basis * 100.0) if cost_basis > 0 else None
    return out


def mirror_kwargs_from_historical_item(
    item: dict[str, Any],
    *,
    yf_ticker: str,
    t212_order_id: int,
    meta: dict[str, Any],
) -> dict[str, Any]:
    order = item.get("order") or {}
    fill = item.get("fill") or {}
    oid = order.get("id")
    if oid is not None and int(oid) != int(t212_order_id):
        raise ValueError("history item order id mismatch")
    side = str(order.get("side") or "").upper()
    if side == "BUY":
        action = "BUY"
    elif side == "SELL":
        action = "SELL"
    else:
        raise ValueError(f"unknown side {side!r}")
    fq_o = float(order.get("filledQuantity") or 0)
    fv_o = float(order.get("filledValue") or 0)
    fq_f = abs(float(fill.get("quantity") or 0))
    price_f = float(fill.get("price") or 0)
    if fq_f > 0 and price_f > 0:
        exec_shares = fq_f
        exec_price = price_f
        total_row = abs(exec_shares * exec_price)
    elif fq_o > 0 and fv_o != 0:
        exec_shares = fq_o
        exec_price = abs(fv_o / fq_o)
        total_row = abs(fv_o)
    else:
        raise ValueError("history item has no fill quantities")
    m = _meta_defaults(meta)
    out: dict[str, Any] = {
        "ticker": yf_ticker.upper().strip(),
        "action": action,
        "shares": exec_shares,
        "price": exec_price,
        "total_value": total_row,
        "reasoning": m["reasoning"],
        "stop_loss": m["stop_loss"],
        "target": m["target"],
        "invalidation_condition": m["invalidation_condition"],
        "chain_of_thought": m["chain_of_thought"],
        "cycle_event": m["cycle_event"],
        "emergency": m["emergency"],
        "t212_order_id": int(t212_order_id),
        "execution_broker": "t212",
    }
    if action == "SELL":
        wi = fill.get("walletImpact") or {}
        realised = wi.get("realisedProfitLoss")
        if realised is not None:
            out["realized_pnl_usd"] = float(realised)
            avg = meta.get("avg_price_paid")
            if avg is not None:
                cost_basis = exec_shares * float(avg)
                if cost_basis > 0:
                    out["pnl_percent"] = float(realised) / cost_basis * 100.0
        elif meta.get("avg_price_paid") is not None:
            avg_f = float(meta["avg_price_paid"])
            cost_basis = exec_shares * avg_f
            realized = total_row - cost_basis
            out["realized_pnl_usd"] = realized
            out["pnl_percent"] = (realized / cost_basis * 100.0) if cost_basis > 0 else None
    return out
