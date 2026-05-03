"""
Pydantic models for Trading 212 Public API responses.

Reference: https://docs.trading212.com/api/positions/getpositions
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Instrument(BaseModel):
    """Instrument metadata as returned inside a position object."""

    ticker: str = ""
    # The T212 API may include additional instrument fields
    # (type, currency, name, etc.) — capture them generically.
    model_config = {"extra": "allow"}


class WalletImpact(BaseModel):
    """Wallet / P&L impact fields."""

    model_config = {"extra": "allow"}


class Position(BaseModel):
    """A single open position from ``GET /api/v0/equity/positions``.

    Field aliases match the camelCase keys in the T212 JSON response.
    """

    average_price_paid: float = Field(0.0, alias="averagePricePaid")
    created_at: str = Field("", alias="createdAt")
    current_price: float = Field(0.0, alias="currentPrice")
    instrument: Instrument = Field(default_factory=Instrument)
    quantity: float = 0.0
    quantity_available_for_trading: float = Field(0.0, alias="quantityAvailableForTrading")
    quantity_in_pies: float = Field(0.0, alias="quantityInPies")
    wallet_impact: WalletImpact = Field(default_factory=WalletImpact, alias="walletImpact")

    model_config = {"populate_by_name": True, "extra": "allow"}

    # ── Convenience properties ──────────────────────────────────────

    @property
    def ticker(self) -> str:
        """Shortcut: ``position.ticker`` → ``position.instrument.ticker``."""
        return self.instrument.ticker

    @property
    def pnl(self) -> float:
        """Unrealised P&L in instrument currency (simple calc)."""
        return (self.current_price - self.average_price_paid) * self.quantity

    @property
    def pnl_percent(self) -> float:
        """Unrealised P&L as percentage."""
        if self.average_price_paid == 0:
            return 0.0
        return ((self.current_price / self.average_price_paid) - 1) * 100
