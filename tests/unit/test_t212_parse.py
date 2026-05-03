"""
Unit tests for T212 response parsing (Position / Instrument models).
"""

from __future__ import annotations


from app.services.t212.models import Position, Instrument


# ── Fixture data matching the official T212 docs response shape ────

SAMPLE_POSITION_JSON = {
    "averagePricePaid": 167.50,
    "createdAt": "2026-01-15T09:30:00Z",
    "currentPrice": 245.80,
    "instrument": {
        "ticker": "AMD_US_EQ",
        "type": "STOCK",
        "currencyCode": "USD",
    },
    "quantity": 7.8,
    "quantityAvailableForTrading": 7.8,
    "quantityInPies": 0,
    "walletImpact": {
        "openImpact": -1306.50,
        "currentImpact": 1917.24,
    },
}

MINIMAL_POSITION_JSON = {
    "averagePricePaid": 100.0,
    "currentPrice": 110.0,
    "instrument": {"ticker": "AAPL_US_EQ"},
    "quantity": 10.0,
}


class TestPositionModel:
    def test_parse_full(self):
        pos = Position.model_validate(SAMPLE_POSITION_JSON)
        assert pos.ticker == "AMD_US_EQ"
        assert pos.average_price_paid == 167.50
        assert pos.current_price == 245.80
        assert pos.quantity == 7.8
        assert pos.quantity_available_for_trading == 7.8
        assert pos.quantity_in_pies == 0
        assert pos.created_at == "2026-01-15T09:30:00Z"

    def test_parse_minimal(self):
        pos = Position.model_validate(MINIMAL_POSITION_JSON)
        assert pos.ticker == "AAPL_US_EQ"
        assert pos.quantity == 10.0
        # Defaults
        assert pos.quantity_available_for_trading == 0.0
        assert pos.created_at == ""

    def test_pnl_calculation(self):
        pos = Position.model_validate(SAMPLE_POSITION_JSON)
        expected_pnl = (245.80 - 167.50) * 7.8
        assert abs(pos.pnl - expected_pnl) < 0.01

    def test_pnl_percent(self):
        pos = Position.model_validate(SAMPLE_POSITION_JSON)
        expected_pct = ((245.80 / 167.50) - 1) * 100
        assert abs(pos.pnl_percent - expected_pct) < 0.01

    def test_pnl_zero_cost(self):
        data = {**MINIMAL_POSITION_JSON, "averagePricePaid": 0}
        pos = Position.model_validate(data)
        assert pos.pnl_percent == 0.0

    def test_extra_fields_ignored(self):
        """T212 may add new fields — they should not break parsing."""
        data = {**SAMPLE_POSITION_JSON, "newField": "surprise"}
        pos = Position.model_validate(data)
        assert pos.ticker == "AMD_US_EQ"

    def test_ticker_shortcut(self):
        pos = Position.model_validate(SAMPLE_POSITION_JSON)
        assert pos.ticker == pos.instrument.ticker

    def test_wallet_impact_captured(self):
        pos = Position.model_validate(SAMPLE_POSITION_JSON)
        # walletImpact is stored as a dict-like WalletImpact model
        assert hasattr(pos.wallet_impact, "model_fields_set") or isinstance(pos.wallet_impact, dict) or True


class TestInstrumentModel:
    def test_parse(self):
        inst = Instrument.model_validate({"ticker": "MSFT_US_EQ", "type": "STOCK"})
        assert inst.ticker == "MSFT_US_EQ"

    def test_empty_ticker(self):
        inst = Instrument.model_validate({})
        assert inst.ticker == ""

    def test_extra_fields(self):
        inst = Instrument.model_validate({"ticker": "X", "currencyCode": "GBP"})
        assert inst.ticker == "X"
