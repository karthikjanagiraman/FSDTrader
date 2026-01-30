#!/usr/bin/env python3
"""
Unit tests for crypto (fractional size) support.

Tests ensure that:
1. Float sizes work correctly in market data structures
2. Formatting displays correctly for both stocks and crypto
3. VWAP calculations work with fractional volumes
4. Order book cumulative sizes work with floats
"""
import pytest
import math
from unittest.mock import MagicMock

# Import the modules under test
from src.brain.context import _fmt_size
from src.market_data import (
    Trade,
    FootprintBar,
    DOMWall,
    LargePrint,
    OrderBook,
    TapeStream,
    FootprintTracker,
    CumulativeDelta,
    VolumeProfile,
    MarketMetrics,
)


class TestFmtSizeHelper:
    """Tests for the _fmt_size formatting helper."""

    def test_stock_whole_numbers(self):
        """Stock integer sizes should use comma formatting."""
        assert _fmt_size(100) == "100"
        assert _fmt_size(1500) == "1,500"
        assert _fmt_size(25000) == "25,000"
        assert _fmt_size(1000000) == "1,000,000"

    def test_stock_float_whole_numbers(self):
        """Float values that are whole numbers should display as integers."""
        assert _fmt_size(100.0) == "100"
        assert _fmt_size(1500.0) == "1,500"
        assert _fmt_size(25000.0) == "25,000"

    def test_crypto_small_fractions(self):
        """Small crypto fractions should use 8 decimal places."""
        assert _fmt_size(0.00157578) == "0.00157578"
        assert _fmt_size(0.00000001) == "0.00000001"
        assert _fmt_size(0.5) == "0.50000000"
        assert _fmt_size(0.99999999) == "0.99999999"

    def test_crypto_larger_fractions(self):
        """Crypto values >= 1 with fractions should use 4 decimal places."""
        assert _fmt_size(1.5) == "1.5000"
        assert _fmt_size(2.75) == "2.7500"
        assert _fmt_size(10.1234) == "10.1234"

    def test_zero(self):
        """Zero should display as '0'."""
        assert _fmt_size(0) == "0"
        assert _fmt_size(0.0) == "0"

    def test_signed_positive(self):
        """Positive values with signed=True should have + prefix."""
        assert _fmt_size(100, signed=True) == "+100"
        assert _fmt_size(1500, signed=True) == "+1,500"
        assert _fmt_size(0.05, signed=True) == "+0.05000000"
        assert _fmt_size(1.5, signed=True) == "+1.5000"

    def test_signed_negative(self):
        """Negative values should always show - prefix."""
        assert _fmt_size(-100) == "-100"
        assert _fmt_size(-100, signed=True) == "-100"
        assert _fmt_size(-0.05) == "-0.05000000"
        assert _fmt_size(-0.05, signed=True) == "-0.05000000"
        assert _fmt_size(-1500) == "-1,500"

    def test_signed_zero(self):
        """Zero with signed=True should still be '0'."""
        assert _fmt_size(0, signed=True) == "0"


class TestTradeDataclass:
    """Tests for Trade dataclass with float sizes."""

    def test_stock_trade(self):
        """Stock trades with integer-like sizes."""
        trade = Trade(price=245.50, size=100, time=1234567890.0, side="BUY")
        assert trade.price == 245.50
        assert trade.size == 100
        assert trade.side == "BUY"

    def test_crypto_trade(self):
        """Crypto trades with fractional sizes."""
        trade = Trade(price=83000.25, size=0.00157578, time=1234567890.0, side="SELL")
        assert trade.price == 83000.25
        assert trade.size == 0.00157578
        assert trade.side == "SELL"


class TestFootprintBar:
    """Tests for FootprintBar with float volumes."""

    def test_stock_footprint(self):
        """Stock footprint with integer-like volumes."""
        bar = FootprintBar(
            open=245.0,
            high=246.0,
            low=244.0,
            close=245.50,
            volume=10000,
            buy_volume=6000,
            sell_volume=4000,
            delta=2000,
        )
        assert bar.volume == 10000
        assert bar.delta == 2000

    def test_crypto_footprint(self):
        """Crypto footprint with fractional volumes."""
        bar = FootprintBar(
            open=83000.0,
            high=83100.0,
            low=82900.0,
            close=83050.0,
            volume=1.5,
            buy_volume=0.9,
            sell_volume=0.6,
            delta=0.3,
        )
        assert bar.volume == 1.5
        assert bar.buy_volume == 0.9
        assert bar.sell_volume == 0.6
        assert bar.delta == 0.3


class TestDOMWall:
    """Tests for DOMWall with float sizes."""

    def test_stock_wall(self):
        """Stock DOM wall with integer size."""
        wall = DOMWall(
            side="BID",
            price=245.00,
            size=5000,
            tier="MAJOR",
            distance_pct=0.5,
            percentile=92.0,
        )
        assert wall.size == 5000

    def test_crypto_wall(self):
        """Crypto DOM wall with fractional size."""
        wall = DOMWall(
            side="ASK",
            price=83500.00,
            size=2.5,
            tier="MASSIVE",
            distance_pct=0.6,
            percentile=96.0,
        )
        assert wall.size == 2.5


class TestOrderBook:
    """Tests for OrderBook with cumulative sizes."""

    def test_stock_cumulative_sizes(self):
        """Stock order book accumulates integer sizes."""
        book = OrderBook("TSLA")

        # Mock DOM levels
        class MockLevel:
            def __init__(self, price, size):
                self.price = price
                self.size = size

        dom_bids = [
            MockLevel(245.00, 100),
            MockLevel(245.00, 200),  # Same price - should accumulate
            MockLevel(244.50, 150),
        ]
        dom_asks = [
            MockLevel(245.50, 100),
            MockLevel(246.00, 200),
        ]

        book.update_from_dom_levels(dom_bids, dom_asks)

        # Check accumulation at same price
        assert book.bids[245.00] == 300  # 100 + 200
        assert book.bids[244.50] == 150
        assert book.asks[245.50] == 100

    def test_crypto_cumulative_sizes(self):
        """Crypto order book accumulates fractional sizes."""
        book = OrderBook("BTC")

        class MockLevel:
            def __init__(self, price, size):
                self.price = price
                self.size = size

        dom_bids = [
            MockLevel(83000.00, 0.5),
            MockLevel(83000.00, 0.25),  # Same price - should accumulate
            MockLevel(82900.00, 0.1),
        ]
        dom_asks = [
            MockLevel(83100.00, 0.3),
        ]

        book.update_from_dom_levels(dom_bids, dom_asks)

        # Check accumulation at same price with floats
        assert abs(book.bids[83000.00] - 0.75) < 0.0001  # 0.5 + 0.25
        assert abs(book.bids[82900.00] - 0.1) < 0.0001
        assert abs(book.asks[83100.00] - 0.3) < 0.0001

    def test_get_stack_to_wall_has_cumulative_size(self):
        """get_stack_to_wall should include cumulative_size field."""
        book = OrderBook("TSLA")

        class MockLevel:
            def __init__(self, price, size):
                self.price = price
                self.size = size

        dom_bids = [
            MockLevel(245.00, 100),
            MockLevel(244.50, 200),
            MockLevel(244.00, 150),
        ]

        book.update_from_dom_levels(dom_bids, [])

        stack = book.get_stack_to_wall("BID")

        # Should have cumulative_size in each level
        assert len(stack) > 0
        for level in stack:
            assert "cumulative_size" in level
            assert "price" in level
            assert "size" in level

        # Check cumulative calculation
        if len(stack) >= 2:
            # First level cumulative = its size
            assert stack[0]["cumulative_size"] == stack[0]["size"]
            # Second level cumulative = first + second
            assert stack[1]["cumulative_size"] == stack[0]["size"] + stack[1]["size"]


class TestCumulativeDelta:
    """Tests for CumulativeDelta with float sizes."""

    def test_stock_delta(self):
        """Stock CVD accumulates integer sizes."""
        cvd = CumulativeDelta("TSLA")

        cvd.add_trade(100, "BUY")
        cvd.add_trade(50, "SELL")
        cvd.add_trade(75, "BUY")

        assert cvd.session_delta == 125  # 100 - 50 + 75

    def test_crypto_delta(self):
        """Crypto CVD accumulates fractional sizes."""
        cvd = CumulativeDelta("BTC")

        cvd.add_trade(0.5, "BUY")
        cvd.add_trade(0.2, "SELL")
        cvd.add_trade(0.1, "BUY")

        assert abs(cvd.session_delta - 0.4) < 0.0001  # 0.5 - 0.2 + 0.1


class TestVolumeProfile:
    """Tests for VolumeProfile with float sizes."""

    def test_stock_volume(self):
        """Stock volume profile with integer sizes."""
        vp = VolumeProfile("TSLA")

        vp.add_trade(245.00, 100)
        vp.add_trade(245.50, 200)
        vp.add_trade(245.00, 150)

        assert vp.total_volume == 450
        assert vp.volume_at_price[245.00] == 250  # 100 + 150

    def test_crypto_volume(self):
        """Crypto volume profile with fractional sizes."""
        vp = VolumeProfile("BTC")

        vp.add_trade(83000.00, 0.5)
        vp.add_trade(83100.00, 0.25)
        vp.add_trade(83000.00, 0.1)

        assert abs(vp.total_volume - 0.85) < 0.0001
        assert abs(vp.volume_at_price[83000.00] - 0.6) < 0.0001


class TestMarketMetrics:
    """Tests for MarketMetrics VWAP calculation with float sizes."""

    def test_stock_vwap(self):
        """Stock VWAP calculation."""
        metrics = MarketMetrics("TSLA")

        metrics.add_trade(245.00, 100)
        metrics.add_trade(246.00, 100)

        # VWAP = (245*100 + 246*100) / 200 = 245.50
        assert metrics.vwap == 245.50
        assert metrics.session_volume == 200

    def test_crypto_vwap(self):
        """Crypto VWAP calculation with fractional sizes."""
        metrics = MarketMetrics("BTC")

        metrics.add_trade(83000.00, 0.5)
        metrics.add_trade(83100.00, 0.5)

        # VWAP = (83000*0.5 + 83100*0.5) / 1.0 = 83050
        assert metrics.vwap == 83050.00
        assert metrics.session_volume == 1.0

    def test_update_from_snapshot_ignores_nan(self):
        """update_from_snapshot should not overwrite with NaN values."""
        metrics = MarketMetrics("BTC")

        # First, set up valid VWAP from trades
        metrics.add_trade(83000.00, 0.5)
        initial_vwap = metrics.vwap
        initial_volume = metrics.session_volume

        # Mock ticker with NaN values (common for crypto)
        mock_ticker = MagicMock()
        mock_ticker.last = 83100.00
        mock_ticker.volume = float('nan')
        mock_ticker.vwap = float('nan')

        metrics.update_from_snapshot(mock_ticker)

        # VWAP and volume should NOT be overwritten with NaN
        assert not math.isnan(metrics.vwap)
        assert not math.isnan(metrics.session_volume)

    def test_update_from_snapshot_uses_valid_values(self):
        """update_from_snapshot should use valid (non-NaN) values."""
        metrics = MarketMetrics("TSLA")

        # Mock ticker with valid values
        mock_ticker = MagicMock()
        mock_ticker.last = 245.50
        mock_ticker.volume = 1000000
        mock_ticker.vwap = 245.25

        metrics.update_from_snapshot(mock_ticker)

        # Should use the valid values
        assert metrics.session_volume == 1000000
        assert metrics.vwap == 245.25


class TestTapeStream:
    """Tests for TapeStream with float sizes."""

    def test_crypto_trades(self):
        """TapeStream should handle crypto fractional sizes."""
        tape = TapeStream("BTC")
        tape.update_quotes(83000.00, 83100.00)

        # Mock tick with fractional size
        class MockTick:
            def __init__(self):
                self.price = 83050.00
                self.size = 0.00157578

        tape.on_tick(MockTick())

        assert len(tape.trades) == 1
        assert tape.trades[0].size == 0.00157578


class TestFootprintTracker:
    """Tests for FootprintTracker with float sizes."""

    def test_crypto_footprint(self):
        """FootprintTracker should handle crypto fractional sizes."""
        fp = FootprintTracker("BTC")

        fp.on_trade(83000.00, 0.5, "BUY")
        fp.on_trade(83050.00, 0.25, "SELL")

        bar = fp.current_bar
        assert bar.buy_volume == 0.5
        assert bar.sell_volume == 0.25
        assert bar.volume == 0.75
        assert bar.delta == 0.25  # 0.5 - 0.25


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
