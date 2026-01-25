#!/usr/bin/env python3
"""
FSDTrader Execution Module: IBKR Executor

Real order execution via Interactive Brokers TWS API.
Uses ib_insync for order management.
"""
import logging
import time
from typing import Dict, Any, List, Optional

try:
    from ib_insync import IB, Stock, Order, Trade
except ImportError:
    IB = None
    Stock = None
    Order = None
    Trade = None

from .base import ExecutionProvider
from .types import (
    PositionSide,
    Position,
    BracketOrder,
    OrderResult,
    TradeRecord,
    RiskLimits,
)


logger = logging.getLogger(__name__)


class IBKRExecutor(ExecutionProvider):
    """
    Real order execution via IBKR TWS API.

    Uses ib_insync to:
    - Submit bracket orders (entry + stop + target)
    - Modify stop/target orders
    - Track fills and position
    - Calculate P&L
    """

    def __init__(
        self,
        ib: "IB",
        symbol: str,
        risk_limits: Optional[RiskLimits] = None,
    ):
        """
        Initialize IBKR executor.

        Args:
            ib: Connected IB instance from ib_insync
            symbol: Trading symbol
            risk_limits: Risk control parameters
        """
        if IB is None:
            raise ImportError("ib_insync is required for IBKRExecutor")

        self.ib = ib
        self.symbol = symbol
        self.contract = Stock(symbol, 'SMART', 'USD')
        self.risk_limits = risk_limits or RiskLimits()
        self.logger = logging.getLogger(f"IBKR_EXEC_{symbol}")

        # Qualify contract
        self.ib.qualifyContracts(self.contract)

        # Position state
        self._position = Position()
        self._active_bracket: Optional[BracketOrder] = None

        # Order tracking
        self._pending_orders: Dict[int, Trade] = {}

        # Daily stats
        self._daily_pnl: float = 0.0
        self._daily_trades: int = 0

        # Trade history
        self._trade_history: List[TradeRecord] = []

        # Entry context
        self._entry_context: Optional[Dict[str, Any]] = None

        # Current price for P&L
        self._current_price: float = 0.0

        # Hook into IBKR events
        self.ib.orderStatusEvent += self._on_order_status
        self.ib.execDetailsEvent += self._on_execution

    def submit_bracket_order(
        self,
        side: str,
        size: int,
        limit_price: float,
        stop_loss: float,
        profit_target: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> OrderResult:
        """Submit a bracket order to IBKR."""
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

        try:
            # Parent order (entry)
            parent = Order()
            parent.orderId = self.ib.client.getReqId()
            parent.action = side
            parent.totalQuantity = size
            parent.orderType = "LMT"
            parent.lmtPrice = limit_price
            parent.transmit = False  # Don't transmit yet

            # Take profit order
            take_profit = Order()
            take_profit.orderId = self.ib.client.getReqId()
            take_profit.action = "SELL" if side == "BUY" else "BUY"
            take_profit.totalQuantity = size
            take_profit.orderType = "LMT"
            take_profit.lmtPrice = profit_target
            take_profit.parentId = parent.orderId
            take_profit.transmit = False

            # Stop loss order
            stop_order = Order()
            stop_order.orderId = self.ib.client.getReqId()
            stop_order.action = "SELL" if side == "BUY" else "BUY"
            stop_order.totalQuantity = size
            stop_order.orderType = "STP"
            stop_order.auxPrice = stop_loss
            stop_order.parentId = parent.orderId
            stop_order.transmit = True  # Transmit all orders

            # Place all three orders
            parent_trade = self.ib.placeOrder(self.contract, parent)
            tp_trade = self.ib.placeOrder(self.contract, take_profit)
            sl_trade = self.ib.placeOrder(self.contract, stop_order)

            # Track orders
            self._pending_orders[parent.orderId] = parent_trade
            self._pending_orders[take_profit.orderId] = tp_trade
            self._pending_orders[stop_order.orderId] = sl_trade

            # Track bracket
            self._active_bracket = BracketOrder(
                entry_order_id=parent.orderId,
                stop_order_id=stop_order.orderId,
                target_order_id=take_profit.orderId,
                entry_price=limit_price,
                stop_price=stop_loss,
                target_price=profit_target,
                side=side,
                quantity=size,
                status="PENDING",
            )

            # Store entry context
            self._entry_context = context

            self._daily_trades += 1
            self.logger.info(
                f"Bracket submitted: {side} {size} @ ${limit_price:.2f}, "
                f"stop=${stop_loss:.2f}, target=${profit_target:.2f}"
            )

            return OrderResult(
                success=True,
                order_id=parent.orderId,
                message=f"Bracket order submitted: {side} {size} @ ${limit_price:.2f}",
            )

        except Exception as e:
            self.logger.error(f"Bracket submission failed: {e}")
            return OrderResult(success=False, error=str(e))

    def modify_stop(self, new_price: float) -> OrderResult:
        """Modify stop loss order."""
        if self._position.is_flat():
            return OrderResult(success=False, error="NO_POSITION")

        if not self._active_bracket:
            return OrderResult(success=False, error="NO_ACTIVE_BRACKET")

        order_id = self._active_bracket.stop_order_id
        if order_id not in self._pending_orders:
            return OrderResult(success=False, error="STOP_ORDER_NOT_FOUND")

        try:
            trade = self._pending_orders[order_id]
            trade.order.auxPrice = new_price
            self.ib.placeOrder(self.contract, trade.order)

            old_price = self._active_bracket.stop_price
            self._active_bracket.stop_price = new_price
            self.logger.info(f"Stop modified: ${old_price:.2f} -> ${new_price:.2f}")

            return OrderResult(
                success=True,
                message=f"Stop modified to ${new_price:.2f}",
            )
        except Exception as e:
            self.logger.error(f"Stop modification failed: {e}")
            return OrderResult(success=False, error=str(e))

    def modify_target(self, new_price: float) -> OrderResult:
        """Modify profit target order."""
        if self._position.is_flat():
            return OrderResult(success=False, error="NO_POSITION")

        if not self._active_bracket:
            return OrderResult(success=False, error="NO_ACTIVE_BRACKET")

        order_id = self._active_bracket.target_order_id
        if order_id not in self._pending_orders:
            return OrderResult(success=False, error="TARGET_ORDER_NOT_FOUND")

        try:
            trade = self._pending_orders[order_id]
            trade.order.lmtPrice = new_price
            self.ib.placeOrder(self.contract, trade.order)

            old_price = self._active_bracket.target_price
            self._active_bracket.target_price = new_price
            self.logger.info(f"Target modified: ${old_price:.2f} -> ${new_price:.2f}")

            return OrderResult(
                success=True,
                message=f"Target modified to ${new_price:.2f}",
            )
        except Exception as e:
            self.logger.error(f"Target modification failed: {e}")
            return OrderResult(success=False, error=str(e))

    def exit_position(self, reason: str = "MANUAL") -> OrderResult:
        """Exit position at market."""
        if self._position.is_flat():
            return OrderResult(success=False, error="NO_POSITION")

        try:
            # Cancel existing bracket orders
            self._cancel_bracket_orders()

            # Submit market exit order
            exit_side = "SELL" if self._position.side == PositionSide.LONG else "BUY"
            exit_order = Order()
            exit_order.action = exit_side
            exit_order.totalQuantity = self._position.size
            exit_order.orderType = "MKT"

            trade = self.ib.placeOrder(self.contract, exit_order)
            self._pending_orders[exit_order.orderId] = trade

            self.logger.info(f"Market exit submitted: {exit_side} {self._position.size}")

            return OrderResult(
                success=True,
                order_id=exit_order.orderId,
                message=f"Market exit order submitted",
            )
        except Exception as e:
            self.logger.error(f"Exit failed: {e}")
            return OrderResult(success=False, error=str(e))

    def cancel_all(self) -> OrderResult:
        """Cancel all pending orders."""
        try:
            # Cancel all tracked orders
            for order_id, trade in list(self._pending_orders.items()):
                try:
                    self.ib.cancelOrder(trade.order)
                except Exception:
                    pass

            self._pending_orders.clear()
            self._active_bracket = None

            # If in position, close it
            if not self._position.is_flat():
                return self.exit_position(reason="CANCEL_ALL")

            return OrderResult(success=True, message="All orders cancelled")
        except Exception as e:
            self.logger.error(f"Cancel all failed: {e}")
            return OrderResult(success=False, error=str(e))

    def update(self, current_price: float, timestamp: float) -> None:
        """Update current price for P&L calculation."""
        self._current_price = current_price
        self._update_unrealized_pnl()

    def _update_unrealized_pnl(self) -> None:
        """Update unrealized P&L."""
        if self._position.is_flat() or self._current_price == 0:
            return

        if self._position.side == PositionSide.LONG:
            self._position.unrealized_pnl = (
                (self._current_price - self._position.avg_entry) * self._position.size
            )
        else:
            self._position.unrealized_pnl = (
                (self._position.avg_entry - self._current_price) * self._position.size
            )

    def _cancel_bracket_orders(self) -> None:
        """Cancel remaining bracket orders."""
        if not self._active_bracket:
            return

        for order_id in [
            self._active_bracket.stop_order_id,
            self._active_bracket.target_order_id,
        ]:
            if order_id in self._pending_orders:
                try:
                    trade = self._pending_orders[order_id]
                    self.ib.cancelOrder(trade.order)
                except Exception:
                    pass

    # =========================================================================
    # IBKR Event Handlers
    # =========================================================================

    def _on_order_status(self, trade: Trade) -> None:
        """Handle order status updates from IBKR."""
        order_id = trade.order.orderId
        status = trade.orderStatus.status

        self.logger.debug(f"Order {order_id} status: {status}")

        if status == "Filled":
            self._handle_fill(trade)
        elif status == "Cancelled":
            self._handle_cancel(trade)

    def _on_execution(self, trade: Trade, fill) -> None:
        """Handle execution details."""
        self.logger.info(
            f"Execution: {fill.execution.side} {fill.execution.shares} "
            f"@ ${fill.execution.price:.2f}"
        )

    def _handle_fill(self, trade: Trade) -> None:
        """Process a filled order."""
        if not self._active_bracket:
            return

        order_id = trade.order.orderId
        fill_price = trade.orderStatus.avgFillPrice

        if order_id == self._active_bracket.entry_order_id:
            # Entry filled
            self._active_bracket.status = "FILLED"
            self._active_bracket.fill_price = fill_price
            self._active_bracket.fill_time = time.time()

            self._position = Position(
                side=PositionSide.LONG if self._active_bracket.side == "BUY" else PositionSide.SHORT,
                size=self._active_bracket.quantity,
                avg_entry=fill_price,
                entry_time=time.time(),
            )

            self.logger.info(f"ENTRY FILLED @ ${fill_price:.2f}")

        elif order_id in (self._active_bracket.stop_order_id,
                          self._active_bracket.target_order_id):
            # Exit filled
            reason = "STOP" if order_id == self._active_bracket.stop_order_id else "TARGET"
            self._close_position(fill_price, reason, time.time())

            # Cancel remaining bracket leg
            self._cleanup_bracket_orders(order_id)

    def _handle_cancel(self, trade: Trade) -> None:
        """Handle cancelled order."""
        order_id = trade.order.orderId
        if order_id in self._pending_orders:
            del self._pending_orders[order_id]

    def _cleanup_bracket_orders(self, filled_order_id: int) -> None:
        """Cancel remaining bracket leg after one side fills."""
        if not self._active_bracket:
            return

        other_order_id = None
        if filled_order_id == self._active_bracket.stop_order_id:
            other_order_id = self._active_bracket.target_order_id
        elif filled_order_id == self._active_bracket.target_order_id:
            other_order_id = self._active_bracket.stop_order_id

        if other_order_id and other_order_id in self._pending_orders:
            try:
                trade = self._pending_orders[other_order_id]
                self.ib.cancelOrder(trade.order)
            except Exception:
                pass

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

        if self._entry_context:
            record.cvd_trend = self._entry_context.get("CVD_TREND", "")
            record.l2_imbalance = self._entry_context.get("L2_IMBALANCE", 0)
            record.tape_velocity = self._entry_context.get("TAPE_VELOCITY", "")
            record.spread = self._entry_context.get("SPREAD", 0)
            record.time_session = self._entry_context.get("TIME_SESSION", "")

        self._trade_history.append(record)
        self._daily_pnl += pnl

        self.logger.info(
            f"EXIT ({reason}): {self._position.side.value} {self._position.size} "
            f"@ ${exit_price:.2f} | P&L: ${pnl:+.2f}"
        )

        # Reset
        self._position = Position()
        self._active_bracket = None
        self._entry_context = None

    # =========================================================================
    # State Getters
    # =========================================================================

    def get_position(self) -> Position:
        """Get current position."""
        return self._position

    def get_account_state(self) -> Dict[str, Any]:
        """Get account state for Brain context."""
        # Calculate remaining daily loss budget
        daily_loss_remaining = abs(self.risk_limits.max_daily_loss) - abs(min(0, self._daily_pnl))

        # Get buying power from IBKR if available
        buying_power = 100000.0  # Default
        try:
            account_values = self.ib.accountValues()
            for av in account_values:
                if av.tag == 'BuyingPower':
                    buying_power = float(av.value)
                    break
        except Exception:
            pass

        return {
            "POSITION": self._position.size,
            "POSITION_SIDE": self._position.side.value,
            "AVG_ENTRY": self._position.avg_entry,
            "UNREALIZED_PL": round(self._position.unrealized_pnl, 2),
            "DAILY_PL": round(self._daily_pnl, 2),
            "DAILY_TRADES": self._daily_trades,
            "BUYING_POWER": round(buying_power, 2),
            "DAILY_LOSS_REMAINING": round(daily_loss_remaining, 2),
        }

    def get_active_orders(self) -> List[Dict[str, Any]]:
        """Get list of active orders."""
        orders = []

        if not self._active_bracket:
            return orders

        bracket = self._active_bracket

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
        """Reset executor state."""
        # Cancel any pending orders first
        self.cancel_all()

        self._position = Position()
        self._active_bracket = None
        self._pending_orders.clear()
        self._daily_pnl = 0.0
        self._daily_trades = 0
        self._trade_history = []
        self._entry_context = None
        self.logger.info("Executor reset")
