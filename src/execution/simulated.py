#!/usr/bin/env python3
"""
FSDTrader Execution Module: Simulated Executor

Virtual order execution for backtest and simulation modes.
Accurately simulates bracket orders with stop/target fills.
"""
import logging
from typing import Dict, Any, List, Optional

from .base import ExecutionProvider
from .types import (
    PositionSide,
    Position,
    BracketOrder,
    OrderResult,
    TradeRecord,
    RiskLimits,
    ActiveOrder,
)


logger = logging.getLogger(__name__)


class SimulatedExecutor(ExecutionProvider):
    """
    Simulated order execution for backtest/sim modes.

    Features:
    - Accurate bracket order simulation
    - Stop/target fill detection based on price updates
    - P&L calculation
    - Trade history tracking
    - Risk limit enforcement
    """

    def __init__(self, risk_limits: Optional[RiskLimits] = None, symbol: str = "TSLA"):
        """
        Initialize the simulated executor.

        Args:
            risk_limits: Risk control parameters
            symbol: Trading symbol (for logging)
        """
        self.symbol = symbol
        self.risk_limits = risk_limits or RiskLimits()
        self.logger = logging.getLogger(f"SIM_EXEC_{symbol}")

        # Position state
        self._position = Position()
        self._active_bracket: Optional[BracketOrder] = None

        # Order ID counter
        self._next_order_id = 1000

        # Daily stats
        self._daily_pnl: float = 0.0
        self._daily_trades: int = 0

        # Trade history
        self._trade_history: List[TradeRecord] = []

        # Current price (updated via update())
        self._current_price: float = 0.0
        self._current_time: float = 0.0

        # Entry context for trade records
        self._entry_context: Optional[Dict[str, Any]] = None

    def submit_bracket_order(
        self,
        side: str,
        size: int,
        limit_price: float,
        stop_loss: float,
        profit_target: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> OrderResult:
        """Submit a simulated bracket order."""
        # Validate position
        if not self._position.is_flat():
            return OrderResult(success=False, error="ALREADY_IN_POSITION")

        # Validate size
        if size > self.risk_limits.max_position_size:
            return OrderResult(
                success=False,
                error=f"SIZE_EXCEEDS_LIMIT: {size} > {self.risk_limits.max_position_size}"
            )

        # Validate stop distance
        stop_error = self.risk_limits.validate_stop_distance(limit_price, stop_loss, side)
        if stop_error:
            return OrderResult(success=False, error=stop_error)

        # Validate target direction
        if side == "BUY":  # Long
            if profit_target <= limit_price:
                return OrderResult(
                    success=False,
                    error=f"Target must be above entry for long (target={profit_target}, entry={limit_price})"
                )
        else:  # Short
            if profit_target >= limit_price:
                return OrderResult(
                    success=False,
                    error=f"Target must be below entry for short (target={profit_target}, entry={limit_price})"
                )

        # Create bracket order
        entry_id = self._get_next_order_id()
        stop_id = self._get_next_order_id()
        target_id = self._get_next_order_id()

        self._active_bracket = BracketOrder(
            entry_order_id=entry_id,
            stop_order_id=stop_id,
            target_order_id=target_id,
            entry_price=limit_price,
            stop_price=stop_loss,
            target_price=profit_target,
            side=side,
            quantity=size,
            status="PENDING",
        )

        # Store entry context
        self._entry_context = context

        self.logger.info(
            f"[SIM] Bracket submitted: {side} {size} @ ${limit_price:.2f}, "
            f"stop=${stop_loss:.2f}, target=${profit_target:.2f}"
        )

        # Check for immediate fill (if price already crossed)
        if self._current_price > 0:
            self._check_entry_fill()

        return OrderResult(
            success=True,
            order_id=entry_id,
            message=f"Bracket order submitted: {side} {size} @ ${limit_price:.2f}",
        )

    def modify_stop(self, new_price: float) -> OrderResult:
        """Modify stop loss price."""
        if self._position.is_flat():
            return OrderResult(success=False, error="NO_POSITION")

        if not self._active_bracket:
            return OrderResult(success=False, error="NO_ACTIVE_BRACKET")

        old_price = self._active_bracket.stop_price
        self._active_bracket.stop_price = new_price

        self.logger.info(f"[SIM] Stop modified: ${old_price:.2f} -> ${new_price:.2f}")

        return OrderResult(
            success=True,
            message=f"Stop modified to ${new_price:.2f}",
        )

    def modify_target(self, new_price: float) -> OrderResult:
        """Modify profit target price."""
        if self._position.is_flat():
            return OrderResult(success=False, error="NO_POSITION")

        if not self._active_bracket:
            return OrderResult(success=False, error="NO_ACTIVE_BRACKET")

        old_price = self._active_bracket.target_price
        self._active_bracket.target_price = new_price

        self.logger.info(f"[SIM] Target modified: ${old_price:.2f} -> ${new_price:.2f}")

        return OrderResult(
            success=True,
            message=f"Target modified to ${new_price:.2f}",
        )

    def exit_position(self, reason: str = "MANUAL") -> OrderResult:
        """Exit position at current market price."""
        if self._position.is_flat():
            return OrderResult(success=False, error="NO_POSITION")

        exit_price = self._current_price
        self._close_position(exit_price, reason, self._current_time)

        return OrderResult(
            success=True,
            fill_price=exit_price,
            message=f"Position closed at ${exit_price:.2f}",
        )

    def cancel_all(self) -> OrderResult:
        """Cancel all pending orders."""
        if self._active_bracket and self._active_bracket.status == "PENDING":
            self._active_bracket = None
            self.logger.info("[SIM] Pending bracket cancelled")
            return OrderResult(success=True, message="Pending orders cancelled")

        # If in position, close it
        if not self._position.is_flat():
            return self.exit_position(reason="CANCEL_ALL")

        return OrderResult(success=True, message="No orders to cancel")

    def update(self, current_price: float, timestamp: float) -> None:
        """
        Update executor with current market price.
        Checks for stop/target fills.
        """
        self._current_price = current_price
        self._current_time = timestamp

        # Update unrealized P&L
        if not self._position.is_flat():
            self._update_unrealized_pnl()

        if not self._active_bracket:
            return

        # Check entry fill
        if self._active_bracket.status == "PENDING":
            self._check_entry_fill()

        # Check stop/target fills
        elif self._active_bracket.status == "FILLED":
            self._check_exit_fills()

    def _check_entry_fill(self) -> None:
        """Check if entry order should fill."""
        if not self._active_bracket or self._active_bracket.status != "PENDING":
            return

        bracket = self._active_bracket
        should_fill = False
        fill_price = bracket.entry_price

        if bracket.side == "BUY":  # Long entry
            # Fill if price goes at or below limit
            if self._current_price <= bracket.entry_price:
                should_fill = True
                fill_price = min(bracket.entry_price, self._current_price)
        else:  # Short entry
            # Fill if price goes at or above limit
            if self._current_price >= bracket.entry_price:
                should_fill = True
                fill_price = max(bracket.entry_price, self._current_price)

        if should_fill:
            self._fill_entry(fill_price, self._current_time)

    def _fill_entry(self, fill_price: float, timestamp: float) -> None:
        """Process entry order fill."""
        bracket = self._active_bracket
        if not bracket:
            return

        # Update bracket
        bracket.status = "FILLED"
        bracket.fill_price = fill_price
        bracket.fill_time = timestamp

        # Update position
        self._position = Position(
            side=PositionSide.LONG if bracket.side == "BUY" else PositionSide.SHORT,
            size=bracket.quantity,
            avg_entry=fill_price,
            unrealized_pnl=0.0,
            entry_time=timestamp,
        )

        self._daily_trades += 1

        self.logger.info(
            f"[SIM] ENTRY FILLED: {bracket.side} {bracket.quantity} @ ${fill_price:.2f}"
        )

    def _check_exit_fills(self) -> None:
        """Check if stop or target should fill."""
        if not self._active_bracket or self._active_bracket.status != "FILLED":
            return

        bracket = self._active_bracket

        if self._position.side == PositionSide.LONG:
            # Stop hit (price at or below stop)
            if self._current_price <= bracket.stop_price:
                self._close_position(bracket.stop_price, "STOP", self._current_time)
            # Target hit (price at or above target)
            elif self._current_price >= bracket.target_price:
                self._close_position(bracket.target_price, "TARGET", self._current_time)

        elif self._position.side == PositionSide.SHORT:
            # Stop hit (price at or above stop)
            if self._current_price >= bracket.stop_price:
                self._close_position(bracket.stop_price, "STOP", self._current_time)
            # Target hit (price at or below target)
            elif self._current_price <= bracket.target_price:
                self._close_position(bracket.target_price, "TARGET", self._current_time)

    def _close_position(self, exit_price: float, reason: str, timestamp: float) -> None:
        """Close position and record trade."""
        if self._position.is_flat():
            return

        # Calculate P&L
        if self._position.side == PositionSide.LONG:
            pnl = (exit_price - self._position.avg_entry) * self._position.size
        else:
            pnl = (self._position.avg_entry - exit_price) * self._position.size

        # Create trade record
        duration = timestamp - self._position.entry_time
        record = TradeRecord(
            entry_time=self._position.entry_time,
            exit_time=timestamp,
            duration_seconds=duration,
            side=self._position.side.value,
            size=self._position.size,
            entry_price=self._position.avg_entry,
            exit_price=exit_price,
            pnl=pnl,
            exit_reason=reason,
        )

        # Add context if available
        if self._entry_context:
            record.cvd_trend = self._entry_context.get("CVD_TREND", "")
            record.l2_imbalance = self._entry_context.get("L2_IMBALANCE", 0)
            record.tape_velocity = self._entry_context.get("TAPE_VELOCITY", "")
            record.spread = self._entry_context.get("SPREAD", 0)
            record.time_session = self._entry_context.get("TIME_SESSION", "")

        self._trade_history.append(record)

        # Update daily P&L
        self._daily_pnl += pnl

        self.logger.info(
            f"[SIM] EXIT ({reason}): {self._position.side.value} {self._position.size} "
            f"@ ${exit_price:.2f} | P&L: ${pnl:+.2f}"
        )

        # Reset position
        self._position = Position()
        self._active_bracket = None
        self._entry_context = None

    def _update_unrealized_pnl(self) -> None:
        """Update unrealized P&L based on current price."""
        if self._position.is_flat():
            return

        if self._position.side == PositionSide.LONG:
            self._position.unrealized_pnl = (
                (self._current_price - self._position.avg_entry) * self._position.size
            )
        else:
            self._position.unrealized_pnl = (
                (self._position.avg_entry - self._current_price) * self._position.size
            )

    def _get_next_order_id(self) -> int:
        """Get next order ID."""
        order_id = self._next_order_id
        self._next_order_id += 1
        return order_id

    def get_position(self) -> Position:
        """Get current position."""
        return self._position

    def get_account_state(self) -> Dict[str, Any]:
        """Get account state for Brain context."""
        daily_loss_remaining = abs(self.risk_limits.max_daily_loss) - abs(min(0, self._daily_pnl))
        return {
            "POSITION": self._position.size,
            "POSITION_SIDE": self._position.side.value,
            "AVG_ENTRY": self._position.avg_entry,
            "UNREALIZED_PL": round(self._position.unrealized_pnl, 2),
            "DAILY_PL": round(self._daily_pnl, 2),
            "DAILY_TRADES": self._daily_trades,
            "BUYING_POWER": 100000.0,  # Simulated unlimited buying power
            "DAILY_LOSS_REMAINING": round(daily_loss_remaining, 2),
        }

    def get_active_orders(self) -> List[Dict[str, Any]]:
        """Get list of active orders."""
        orders = []

        if not self._active_bracket:
            return orders

        bracket = self._active_bracket

        # Entry order (if pending)
        if bracket.status == "PENDING":
            orders.append({
                "order_id": bracket.entry_order_id,
                "type": "LIMIT",
                "side": bracket.side,
                "price": bracket.entry_price,
                "size": bracket.quantity,
                "status": "PENDING",
                "purpose": "ENTRY",
            })

        # Stop and target (if entry filled)
        elif bracket.status == "FILLED":
            exit_side = "SELL" if bracket.side == "BUY" else "BUY"

            orders.append({
                "order_id": bracket.stop_order_id,
                "type": "STOP",
                "side": exit_side,
                "price": bracket.stop_price,
                "size": bracket.quantity,
                "status": "PENDING",
                "purpose": "STOP_LOSS",
            })

            orders.append({
                "order_id": bracket.target_order_id,
                "type": "LIMIT",
                "side": exit_side,
                "price": bracket.target_price,
                "size": bracket.quantity,
                "status": "PENDING",
                "purpose": "PROFIT_TARGET",
            })

        return orders

    def get_state(self) -> Dict[str, Any]:
        """Get complete execution state."""
        return {
            "position": self._position.to_dict(),
            "daily_stats": {
                "pnl": round(self._daily_pnl, 2),
                "trades": self._daily_trades,
            },
            "active_bracket": self._active_bracket.to_dict() if self._active_bracket else None,
            "limits": {
                "max_position": self.risk_limits.max_position_size,
                "max_daily_loss": self.risk_limits.max_daily_loss,
                "max_trades": self.risk_limits.max_daily_trades,
            },
            "trade_count": len(self._trade_history),
        }

    def get_trade_history(self) -> List[TradeRecord]:
        """Get completed trade history."""
        return self._trade_history

    def reset(self) -> None:
        """Reset executor state for new session."""
        self._position = Position()
        self._active_bracket = None
        self._daily_pnl = 0.0
        self._daily_trades = 0
        self._trade_history = []
        self._current_price = 0.0
        self._current_time = 0.0
        self._entry_context = None
        self.logger.info("[SIM] Executor reset")
