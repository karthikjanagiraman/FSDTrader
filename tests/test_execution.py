#!/usr/bin/env python3
"""
FSDTrader Execution Module Unit Tests

Comprehensive tests for SimulatedExecutor and execution types.
Tests cover: order submission, fill logic, P&L calculation, risk limits,
position management, and state consistency.
"""
import sys
import pytest
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from execution import SimulatedExecutor, RiskLimits, get_executor
from execution.types import (
    PositionSide,
    Position,
    BracketOrder,
    OrderResult,
    TradeRecord,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def default_limits():
    """Default risk limits for testing."""
    return RiskLimits(
        max_position_size=100,
        max_daily_loss=-500.0,
        max_daily_trades=10,
        max_spread=0.15,
        min_stop_distance=0.10,
        max_stop_distance=0.30,
    )


@pytest.fixture
def executor(default_limits):
    """Create a fresh SimulatedExecutor for each test."""
    return SimulatedExecutor(risk_limits=default_limits, symbol="TEST")


@pytest.fixture
def executor_with_position(executor):
    """Create executor with an active long position."""
    # Submit and fill a long position
    executor._current_price = 100.0
    executor._current_time = 1000.0

    result = executor.submit_bracket_order(
        side="BUY",
        size=50,
        limit_price=100.0,
        stop_loss=99.80,
        profit_target=100.50,
    )
    assert result.success

    # Trigger fill
    executor.update(99.95, 1001.0)
    assert not executor._position.is_flat()

    return executor


# =============================================================================
# Types Tests
# =============================================================================

class TestPosition:
    """Tests for Position dataclass."""

    def test_default_position_is_flat(self):
        pos = Position()
        assert pos.is_flat()
        assert pos.side == PositionSide.FLAT
        assert pos.size == 0

    def test_long_position_not_flat(self):
        pos = Position(side=PositionSide.LONG, size=100, avg_entry=50.0)
        assert not pos.is_flat()

    def test_short_position_not_flat(self):
        pos = Position(side=PositionSide.SHORT, size=100, avg_entry=50.0)
        assert not pos.is_flat()

    def test_zero_size_is_flat(self):
        pos = Position(side=PositionSide.LONG, size=0)
        assert pos.is_flat()

    def test_to_dict(self):
        pos = Position(
            side=PositionSide.LONG,
            size=100,
            avg_entry=50.0,
            unrealized_pnl=25.50,
            entry_time=1000.0,
        )
        d = pos.to_dict()
        assert d["side"] == "LONG"
        assert d["size"] == 100
        assert d["avg_entry"] == 50.0
        assert d["unrealized_pnl"] == 25.5


class TestRiskLimits:
    """Tests for RiskLimits validation."""

    def test_valid_long_stop(self, default_limits):
        # Entry at 100, stop at 99.85 = $0.15 distance (valid)
        error = default_limits.validate_stop_distance(100.0, 99.85, "BUY")
        assert error is None

    def test_valid_short_stop(self, default_limits):
        # Entry at 100, stop at 100.15 = $0.15 distance (valid)
        error = default_limits.validate_stop_distance(100.0, 100.15, "SELL")
        assert error is None

    def test_stop_too_tight_long(self, default_limits):
        # Entry at 100, stop at 99.95 = $0.05 distance (too tight)
        error = default_limits.validate_stop_distance(100.0, 99.95, "BUY")
        assert error is not None
        assert "too tight" in error.lower()

    def test_stop_too_wide_long(self, default_limits):
        # Entry at 100, stop at 99.50 = $0.50 distance (too wide)
        error = default_limits.validate_stop_distance(100.0, 99.50, "BUY")
        assert error is not None
        assert "too wide" in error.lower()

    def test_stop_wrong_direction_long(self, default_limits):
        # Entry at 100, stop at 100.10 (above entry for long = invalid)
        error = default_limits.validate_stop_distance(100.0, 100.10, "BUY")
        assert error is not None
        assert "below entry" in error.lower()

    def test_stop_wrong_direction_short(self, default_limits):
        # Entry at 100, stop at 99.90 (below entry for short = invalid)
        error = default_limits.validate_stop_distance(100.0, 99.90, "SELL")
        assert error is not None
        assert "above entry" in error.lower()


class TestOrderResult:
    """Tests for OrderResult."""

    def test_success_result(self):
        result = OrderResult(success=True, order_id=1001, message="Order placed")
        assert result.success
        assert result.order_id == 1001
        assert result.error is None

    def test_failure_result(self):
        result = OrderResult(success=False, error="INVALID_PRICE")
        assert not result.success
        assert result.error == "INVALID_PRICE"

    def test_to_dict(self):
        result = OrderResult(success=True, order_id=1001, fill_price=100.0)
        d = result.to_dict()
        assert d["success"] is True
        assert d["order_id"] == 1001
        assert d["fill_price"] == 100.0


# =============================================================================
# SimulatedExecutor Tests
# =============================================================================

class TestSimulatedExecutorInit:
    """Tests for SimulatedExecutor initialization."""

    def test_default_initialization(self):
        executor = SimulatedExecutor()
        assert executor._position.is_flat()
        assert executor._daily_pnl == 0.0
        assert executor._daily_trades == 0
        assert executor._active_bracket is None

    def test_custom_risk_limits(self, default_limits):
        executor = SimulatedExecutor(risk_limits=default_limits, symbol="AAPL")
        assert executor.risk_limits.max_position_size == 100
        assert executor.symbol == "AAPL"

    def test_factory_function(self):
        executor = get_executor("simulated", symbol="TSLA")
        assert isinstance(executor, SimulatedExecutor)
        assert executor.symbol == "TSLA"


class TestBracketOrderSubmission:
    """Tests for bracket order submission."""

    def test_submit_long_bracket(self, executor):
        result = executor.submit_bracket_order(
            side="BUY",
            size=50,
            limit_price=100.0,
            stop_loss=99.80,
            profit_target=100.50,
        )
        assert result.success
        assert result.order_id is not None
        assert executor._active_bracket is not None
        assert executor._active_bracket.status == "PENDING"
        assert executor._active_bracket.side == "BUY"

    def test_submit_short_bracket(self, executor):
        result = executor.submit_bracket_order(
            side="SELL",
            size=50,
            limit_price=100.0,
            stop_loss=100.20,
            profit_target=99.50,
        )
        assert result.success
        assert executor._active_bracket.side == "SELL"

    def test_reject_already_in_position(self, executor_with_position):
        result = executor_with_position.submit_bracket_order(
            side="BUY",
            size=50,
            limit_price=100.0,
            stop_loss=99.80,
            profit_target=100.50,
        )
        assert not result.success
        assert "ALREADY_IN_POSITION" in result.error

    def test_reject_size_exceeds_limit(self, executor):
        result = executor.submit_bracket_order(
            side="BUY",
            size=200,  # Exceeds max_position_size of 100
            limit_price=100.0,
            stop_loss=99.80,
            profit_target=100.50,
        )
        assert not result.success
        assert "SIZE_EXCEEDS_LIMIT" in result.error

    def test_reject_invalid_stop_distance(self, executor):
        result = executor.submit_bracket_order(
            side="BUY",
            size=50,
            limit_price=100.0,
            stop_loss=99.95,  # Too tight ($0.05 < $0.10 min)
            profit_target=100.50,
        )
        assert not result.success
        assert "too tight" in result.error.lower()

    def test_reject_target_wrong_direction_long(self, executor):
        result = executor.submit_bracket_order(
            side="BUY",
            size=50,
            limit_price=100.0,
            stop_loss=99.80,
            profit_target=99.50,  # Below entry for long = invalid
        )
        assert not result.success
        assert "Target must be above" in result.error

    def test_reject_target_wrong_direction_short(self, executor):
        result = executor.submit_bracket_order(
            side="SELL",
            size=50,
            limit_price=100.0,
            stop_loss=100.20,
            profit_target=100.50,  # Above entry for short = invalid
        )
        assert not result.success
        assert "Target must be below" in result.error


class TestEntryFillLogic:
    """Tests for entry order fill detection."""

    def test_long_entry_fills_at_limit(self, executor):
        executor.submit_bracket_order(
            side="BUY",
            size=50,
            limit_price=100.0,
            stop_loss=99.80,
            profit_target=100.50,
        )

        # Price touches limit exactly
        executor.update(100.0, 1000.0)

        assert executor._active_bracket.status == "FILLED"
        assert executor._position.side == PositionSide.LONG
        assert executor._position.size == 50
        assert executor._position.avg_entry == 100.0

    def test_long_entry_fills_below_limit(self, executor):
        executor.submit_bracket_order(
            side="BUY",
            size=50,
            limit_price=100.0,
            stop_loss=99.80,
            profit_target=100.50,
        )

        # Price goes below limit (better fill)
        executor.update(99.90, 1000.0)

        assert executor._active_bracket.status == "FILLED"
        assert executor._position.avg_entry == 99.90  # Filled at better price

    def test_long_entry_not_fill_above_limit(self, executor):
        executor.submit_bracket_order(
            side="BUY",
            size=50,
            limit_price=100.0,
            stop_loss=99.80,
            profit_target=100.50,
        )

        # Price stays above limit
        executor.update(100.10, 1000.0)

        assert executor._active_bracket.status == "PENDING"
        assert executor._position.is_flat()

    def test_short_entry_fills_at_limit(self, executor):
        executor.submit_bracket_order(
            side="SELL",
            size=50,
            limit_price=100.0,
            stop_loss=100.20,
            profit_target=99.50,
        )

        executor.update(100.0, 1000.0)

        assert executor._active_bracket.status == "FILLED"
        assert executor._position.side == PositionSide.SHORT

    def test_short_entry_fills_above_limit(self, executor):
        executor.submit_bracket_order(
            side="SELL",
            size=50,
            limit_price=100.0,
            stop_loss=100.20,
            profit_target=99.50,
        )

        # Price goes above limit (better fill for short)
        executor.update(100.10, 1000.0)

        assert executor._active_bracket.status == "FILLED"
        assert executor._position.avg_entry == 100.10  # Filled at better price


class TestExitFillLogic:
    """Tests for stop/target fill detection."""

    def test_long_stop_hit(self, executor_with_position):
        stop_price = executor_with_position._active_bracket.stop_price  # 99.80

        # Price hits stop
        executor_with_position.update(stop_price, 2000.0)

        assert executor_with_position._position.is_flat()
        assert len(executor_with_position._trade_history) == 1
        assert executor_with_position._trade_history[0].exit_reason == "STOP"

    def test_long_stop_gaps_through(self, executor_with_position):
        # Price gaps below stop
        executor_with_position.update(99.50, 2000.0)

        assert executor_with_position._position.is_flat()
        # Fills at stop price (no slippage model)
        assert executor_with_position._trade_history[0].exit_price == 99.80

    def test_long_target_hit(self, executor_with_position):
        target_price = executor_with_position._active_bracket.target_price  # 100.50

        # Price hits target
        executor_with_position.update(target_price, 2000.0)

        assert executor_with_position._position.is_flat()
        assert executor_with_position._trade_history[0].exit_reason == "TARGET"

    def test_short_stop_hit(self, executor, default_limits):
        # Create short position
        executor._current_price = 100.0
        executor.submit_bracket_order(
            side="SELL",
            size=50,
            limit_price=100.0,
            stop_loss=100.20,
            profit_target=99.50,
        )
        executor.update(100.05, 1000.0)  # Fill entry
        assert executor._position.side == PositionSide.SHORT

        # Hit stop
        executor.update(100.20, 2000.0)

        assert executor._position.is_flat()
        assert executor._trade_history[0].exit_reason == "STOP"

    def test_short_target_hit(self, executor):
        executor._current_price = 100.0
        executor.submit_bracket_order(
            side="SELL",
            size=50,
            limit_price=100.0,
            stop_loss=100.20,
            profit_target=99.50,
        )
        executor.update(100.05, 1000.0)  # Fill entry

        # Hit target
        executor.update(99.50, 2000.0)

        assert executor._position.is_flat()
        assert executor._trade_history[0].exit_reason == "TARGET"


class TestPnLCalculation:
    """Tests for P&L calculation accuracy."""

    def test_long_profit(self, executor):
        executor.submit_bracket_order(
            side="BUY",
            size=100,
            limit_price=100.0,
            stop_loss=99.80,
            profit_target=100.50,
        )
        executor.update(100.0, 1000.0)  # Entry at 100
        executor.update(100.50, 2000.0)  # Exit at target 100.50

        # P&L = (100.50 - 100.00) * 100 = $50
        assert executor._daily_pnl == 50.0
        assert executor._trade_history[0].pnl == 50.0

    def test_long_loss(self, executor):
        executor.submit_bracket_order(
            side="BUY",
            size=100,
            limit_price=100.0,
            stop_loss=99.80,
            profit_target=100.50,
        )
        executor.update(100.0, 1000.0)  # Entry at 100
        executor.update(99.80, 2000.0)  # Exit at stop 99.80

        # P&L = (99.80 - 100.00) * 100 = -$20
        assert abs(executor._daily_pnl - (-20.0)) < 0.01
        assert abs(executor._trade_history[0].pnl - (-20.0)) < 0.01

    def test_short_profit(self, executor):
        executor._current_price = 100.0
        executor.submit_bracket_order(
            side="SELL",
            size=100,
            limit_price=100.0,
            stop_loss=100.20,
            profit_target=99.50,
        )
        executor.update(100.0, 1000.0)  # Entry at 100
        executor.update(99.50, 2000.0)  # Exit at target 99.50

        # P&L = (100.00 - 99.50) * 100 = $50
        assert executor._daily_pnl == 50.0

    def test_short_loss(self, executor):
        executor._current_price = 100.0
        executor.submit_bracket_order(
            side="SELL",
            size=100,
            limit_price=100.0,
            stop_loss=100.20,
            profit_target=99.50,
        )
        executor.update(100.0, 1000.0)  # Entry at 100
        executor.update(100.20, 2000.0)  # Exit at stop 100.20

        # P&L = (100.00 - 100.20) * 100 = -$20
        assert abs(executor._daily_pnl - (-20.0)) < 0.01

    def test_unrealized_pnl_long(self, executor_with_position):
        # Current position: LONG from ~99.95
        entry = executor_with_position._position.avg_entry

        # Price moves up
        executor_with_position.update(100.20, 1500.0)

        expected_unrealized = (100.20 - entry) * 50
        assert abs(executor_with_position._position.unrealized_pnl - expected_unrealized) < 0.01

    def test_cumulative_pnl(self, executor):
        # Trade 1: Profit
        executor.submit_bracket_order("BUY", 50, 100.0, 99.80, 100.50)
        executor.update(100.0, 1000.0)
        executor.update(100.50, 1500.0)  # +$25

        # Trade 2: Loss
        executor.submit_bracket_order("BUY", 50, 100.0, 99.80, 100.50)
        executor.update(100.0, 2000.0)
        executor.update(99.80, 2500.0)  # -$10

        # Cumulative: +$25 - $10 = $15
        assert abs(executor._daily_pnl - 15.0) < 0.01
        assert executor._daily_trades == 2


class TestModifyOrders:
    """Tests for order modification."""

    def test_modify_stop(self, executor_with_position):
        original_stop = executor_with_position._active_bracket.stop_price

        result = executor_with_position.modify_stop(99.75)

        assert result.success
        assert executor_with_position._active_bracket.stop_price == 99.75
        assert executor_with_position._active_bracket.stop_price != original_stop

    def test_modify_target(self, executor_with_position):
        result = executor_with_position.modify_target(100.75)

        assert result.success
        assert executor_with_position._active_bracket.target_price == 100.75

    def test_modify_stop_no_position(self, executor):
        result = executor.modify_stop(99.75)

        assert not result.success
        assert "NO_POSITION" in result.error

    def test_modify_target_no_position(self, executor):
        result = executor.modify_target(100.75)

        assert not result.success
        assert "NO_POSITION" in result.error


class TestExitPosition:
    """Tests for manual position exit."""

    def test_exit_at_market(self, executor_with_position):
        executor_with_position._current_price = 100.30

        result = executor_with_position.exit_position(reason="MANUAL")

        assert result.success
        assert result.fill_price == 100.30
        assert executor_with_position._position.is_flat()
        assert executor_with_position._trade_history[0].exit_reason == "MANUAL"

    def test_exit_no_position(self, executor):
        result = executor.exit_position()

        assert not result.success
        assert "NO_POSITION" in result.error


class TestCancelAll:
    """Tests for cancelling orders."""

    def test_cancel_pending_bracket(self, executor):
        executor.submit_bracket_order("BUY", 50, 100.0, 99.80, 100.50)

        result = executor.cancel_all()

        assert result.success
        assert executor._active_bracket is None

    def test_cancel_closes_position(self, executor_with_position):
        executor_with_position._current_price = 100.10

        result = executor_with_position.cancel_all()

        assert result.success
        assert executor_with_position._position.is_flat()


class TestAccountState:
    """Tests for account state output."""

    def test_flat_account_state(self, executor):
        state = executor.get_account_state()

        assert state["POSITION"] == 0
        assert state["POSITION_SIDE"] == "FLAT"
        assert state["AVG_ENTRY"] == 0.0
        assert state["UNREALIZED_PL"] == 0.0
        assert state["DAILY_PL"] == 0.0
        assert state["DAILY_TRADES"] == 0
        assert "BUYING_POWER" in state
        assert "DAILY_LOSS_REMAINING" in state

    def test_position_account_state(self, executor_with_position):
        state = executor_with_position.get_account_state()

        assert state["POSITION"] == 50
        assert state["POSITION_SIDE"] == "LONG"
        assert state["AVG_ENTRY"] > 0
        assert state["DAILY_TRADES"] == 1

    def test_daily_loss_remaining(self, executor):
        # Start with full budget
        state = executor.get_account_state()
        assert state["DAILY_LOSS_REMAINING"] == 500.0  # abs(-500)

        # After a loss
        executor._daily_pnl = -100.0
        state = executor.get_account_state()
        assert state["DAILY_LOSS_REMAINING"] == 400.0


class TestActiveOrders:
    """Tests for active orders output."""

    def test_no_active_orders(self, executor):
        orders = executor.get_active_orders()
        assert orders == []

    def test_pending_entry_order(self, executor):
        executor.submit_bracket_order("BUY", 50, 100.0, 99.80, 100.50)

        orders = executor.get_active_orders()

        assert len(orders) == 1
        assert orders[0]["purpose"] == "ENTRY"
        assert orders[0]["type"] == "LIMIT"
        assert orders[0]["side"] == "BUY"
        assert orders[0]["price"] == 100.0

    def test_filled_bracket_shows_stop_target(self, executor_with_position):
        orders = executor_with_position.get_active_orders()

        assert len(orders) == 2
        purposes = [o["purpose"] for o in orders]
        assert "STOP_LOSS" in purposes
        assert "PROFIT_TARGET" in purposes


class TestTradeHistory:
    """Tests for trade history."""

    def test_empty_history(self, executor):
        history = executor.get_trade_history()
        assert history == []

    def test_trade_record_created(self, executor_with_position):
        executor_with_position.update(100.50, 2000.0)  # Hit target

        history = executor_with_position.get_trade_history()

        assert len(history) == 1
        record = history[0]
        assert record.side == "LONG"
        assert record.size == 50
        assert record.exit_reason == "TARGET"
        assert record.pnl > 0

    def test_trade_record_duration(self, executor):
        executor.submit_bracket_order("BUY", 50, 100.0, 99.80, 100.50)
        executor.update(100.0, 1000.0)  # Entry at t=1000
        executor.update(100.50, 1060.0)  # Exit at t=1060

        record = executor.get_trade_history()[0]
        assert record.duration_seconds == 60.0

    def test_trade_record_context(self, executor):
        context = {
            "CVD_TREND": "RISING",
            "L2_IMBALANCE": 1.5,
            "TAPE_VELOCITY": "HIGH",
            "SPREAD": 0.02,
            "TIME_SESSION": "OPEN_DRIVE",
        }

        executor.submit_bracket_order(
            side="BUY",
            size=50,
            limit_price=100.0,
            stop_loss=99.80,
            profit_target=100.50,
            context=context,
        )
        executor.update(100.0, 1000.0)
        executor.update(100.50, 1500.0)

        record = executor.get_trade_history()[0]
        assert record.cvd_trend == "RISING"
        assert record.l2_imbalance == 1.5
        assert record.tape_velocity == "HIGH"


class TestReset:
    """Tests for executor reset."""

    def test_reset_clears_state(self, executor_with_position):
        executor_with_position._daily_pnl = 100.0

        executor_with_position.reset()

        assert executor_with_position._position.is_flat()
        assert executor_with_position._active_bracket is None
        assert executor_with_position._daily_pnl == 0.0
        assert executor_with_position._daily_trades == 0
        assert executor_with_position._trade_history == []


class TestExecuteCommand:
    """Tests for DSL command execution."""

    def test_execute_enter_long(self, executor):
        cmd = "ENTER_LONG|limit_price=100.0|stop_loss=99.80|profit_target=100.50|size=50"

        result = executor.execute(cmd, current_spread=0.02)

        assert result["success"]
        assert executor._active_bracket is not None

    def test_execute_enter_short(self, executor):
        cmd = "ENTER_SHORT|limit_price=100.0|stop_loss=100.20|profit_target=99.50|size=50"

        result = executor.execute(cmd, current_spread=0.02)

        assert result["success"]
        assert executor._active_bracket.side == "SELL"

    def test_execute_wait(self, executor):
        cmd = "WAIT|reasoning=Market conditions unfavorable"

        result = executor.execute(cmd)

        assert result["success"]
        assert result["action"] == "WAIT"

    def test_execute_rejects_wide_spread(self, executor):
        cmd = "ENTER_LONG|limit_price=100.0|stop_loss=99.80|profit_target=100.50"

        result = executor.execute(cmd, current_spread=0.20)  # Exceeds 0.15 max

        assert not result["success"]
        assert "SPREAD_TOO_WIDE" in result["error"]

    def test_execute_update_stop(self, executor_with_position):
        cmd = "UPDATE_STOP|new_price=99.85"

        result = executor_with_position.execute(cmd)

        assert result["success"]
        assert executor_with_position._active_bracket.stop_price == 99.85

    def test_execute_exit_position(self, executor_with_position):
        executor_with_position._current_price = 100.20
        cmd = "EXIT_POSITION|reasoning=Taking profits"

        result = executor_with_position.execute(cmd)

        assert result["success"]
        assert executor_with_position._position.is_flat()

    def test_execute_unknown_command(self, executor):
        cmd = "UNKNOWN_CMD|arg=value"

        result = executor.execute(cmd)

        assert not result["success"]
        assert "UNKNOWN_COMMAND" in result["error"]


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_immediate_fill_on_submit(self, executor):
        """Entry fills immediately if price already crossed."""
        executor._current_price = 99.90

        executor.submit_bracket_order("BUY", 50, 100.0, 99.80, 100.50)

        # Should fill immediately since price (99.90) <= limit (100.0)
        assert executor._active_bracket.status == "FILLED"
        assert executor._position.avg_entry == 99.90

    def test_multiple_updates_same_price(self, executor_with_position):
        """Multiple updates at same price don't cause issues."""
        for _ in range(10):
            executor_with_position.update(100.10, 1500.0)

        assert not executor_with_position._position.is_flat()
        assert len(executor_with_position._trade_history) == 0

    def test_price_exactly_at_stop_and_target(self, executor):
        """Price exactly between stop and target doesn't trigger either."""
        executor.submit_bracket_order("BUY", 50, 100.0, 99.80, 100.50)
        executor.update(100.0, 1000.0)  # Fill entry

        # Price at midpoint
        executor.update(100.15, 1500.0)

        assert not executor._position.is_flat()


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
