"""
Unit tests for T212 ↔ yfinance ticker normalisation.
"""

from __future__ import annotations


from app.services.t212.ticker_map import t212_to_yfinance, yfinance_to_t212


class TestT212ToYfinance:
    def test_us_equity(self):
        assert t212_to_yfinance("AAPL_US_EQ") == "AAPL"

    def test_uk_equity(self):
        assert t212_to_yfinance("LLOY_UK_EQ") == "LLOY"

    def test_de_equity(self):
        assert t212_to_yfinance("SAP_DE_EQ") == "SAP"

    def test_already_plain(self):
        assert t212_to_yfinance("AMZN") == "AMZN"

    def test_brk_b_special(self):
        assert t212_to_yfinance("BRKb_US_EQ") == "BRK-B"

    def test_brk_a_special(self):
        assert t212_to_yfinance("BRKa_US_EQ") == "BRK-A"

    def test_empty_string(self):
        assert t212_to_yfinance("") == ""

    def test_no_suffix_match(self):
        """Strings ending with something other than _XX_EQ pass through."""
        assert t212_to_yfinance("RANDOM_TOKEN") == "RANDOM_TOKEN"


class TestYfinanceToT212:
    def test_basic(self):
        assert yfinance_to_t212("AAPL") == "AAPL_US_EQ"

    def test_custom_exchange(self):
        assert yfinance_to_t212("SAP", exchange="DE") == "SAP_DE_EQ"

    def test_brk_b_reverse(self):
        assert yfinance_to_t212("BRK-B") == "BRKb_US_EQ"

    def test_brk_a_reverse(self):
        assert yfinance_to_t212("BRK-A") == "BRKa_US_EQ"
