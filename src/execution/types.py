#!/usr/bin/env python3
"""
FSDTrader Execution Module: Type Definitions

Core data classes for order execution and position tracking.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class PositionSide(Enum):
    """Position direction."""
    FLAT = "FLAT"
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class Position:
    """Current position state."""
    side: PositionSide = PositionSide.FLAT
    size: int = 0
    avg_entry: float = 0.0
    unrealized_pnl: float = 0.0
    entry_time: float = 0.0

    def is_flat(self) -> bool:
        """Check if position is flat."""
        return self.side == PositionSide.FLAT or self.size == 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for state output."""
        return {
            "side": self.side.value,
            "size": self.size,
            "avg_entry": self.avg_entry,
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "entry_time": self.entry_time,
        }


@dataclass
class BracketOrder:
    """
    Represents a bracket order (entry + stop + target).

    For LONG positions:
        - entry is a BUY limit order
        - stop is a SELL stop order (below entry)
        - target is a SELL limit order (above entry)

    For SHORT positions:
        - entry is a SELL limit order
        - stop is a BUY stop order (above entry)
        - target is a BUY limit order (below entry)
    """
    # Order IDs
    entry_order_id: int = 0
    stop_order_id: int = 0
    target_order_id: int = 0

    # Prices
    entry_price: float = 0.0
    stop_price: float = 0.0
    target_price: float = 0.0

    # Order details
    side: str = ""  # "BUY" (for long) or "SELL" (for short)
    quantity: int = 0

    # Status: PENDING -> FILLED (entry filled) -> CLOSED (stop/target hit)
    status: str = "PENDING"

    # Fill info
    fill_price: float = 0.0
    fill_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entry_order_id": self.entry_order_id,
            "stop_order_id": self.stop_order_id,
            "target_order_id": self.target_order_id,
            "entry_price": self.entry_price,
            "stop_price": self.stop_price,
            "target_price": self.target_price,
            "side": self.side,
            "quantity": self.quantity,
            "status": self.status,
            "fill_price": self.fill_price,
            "fill_time": self.fill_time,
        }


@dataclass
class OrderResult:
    """Result of an order operation."""
    success: bool
    order_id: Optional[int] = None
    error: Optional[str] = None
    fill_price: Optional[float] = None
    fill_size: Optional[int] = None
    message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {"success": self.success}
        if self.order_id is not None:
            result["order_id"] = self.order_id
        if self.error is not None:
            result["error"] = self.error
        if self.fill_price is not None:
            result["fill_price"] = self.fill_price
        if self.fill_size is not None:
            result["fill_size"] = self.fill_size
        if self.message is not None:
            result["message"] = self.message
        return result


@dataclass
class TradeRecord:
    """
    Record of a completed trade for backtest reporting.
    Created when a position is closed (stop, target, or manual exit).
    """
    # Timing
    entry_time: float = 0.0
    exit_time: float = 0.0
    duration_seconds: float = 0.0

    # Trade details
    side: str = ""  # "LONG" or "SHORT"
    size: int = 0
    entry_price: float = 0.0
    exit_price: float = 0.0
    pnl: float = 0.0
    exit_reason: str = ""  # "STOP", "TARGET", "MANUAL"

    # Market context at entry
    cvd_trend: str = ""
    l2_imbalance: float = 0.0
    tape_velocity: str = ""
    spread: float = 0.0
    time_session: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "duration_seconds": round(self.duration_seconds, 1),
            "side": self.side,
            "size": self.size,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "pnl": round(self.pnl, 2),
            "exit_reason": self.exit_reason,
            "context": {
                "cvd_trend": self.cvd_trend,
                "l2_imbalance": self.l2_imbalance,
                "tape_velocity": self.tape_velocity,
                "spread": self.spread,
                "time_session": self.time_session,
            }
        }


@dataclass
class RiskLimits:
    """
    Risk control parameters.
    These are enforced by both SimulatedExecutor and IBKRExecutor.
    """
    # Position limits
    max_position_size: int = 100

    # Daily limits
    max_daily_loss: float = -500.0
    max_daily_trades: int = 10

    # Entry conditions
    max_spread: float = 0.15  # Max spread for entry

    # Stop distance limits
    min_stop_distance: float = 0.10  # $0.10 minimum
    max_stop_distance: float = 0.30  # $0.30 maximum

    def validate_stop_distance(self, entry_price: float, stop_price: float, side: str) -> Optional[str]:
        """
        Validate stop distance.

        Args:
            entry_price: Entry limit price
            stop_price: Stop loss price
            side: "BUY" (long) or "SELL" (short)

        Returns:
            Error message if invalid, None if valid
        """
        if side == "BUY":  # Long position
            distance = entry_price - stop_price
            if distance < 0:
                return f"Stop must be below entry for long (stop={stop_price}, entry={entry_price})"
        else:  # Short position
            distance = stop_price - entry_price
            if distance < 0:
                return f"Stop must be above entry for short (stop={stop_price}, entry={entry_price})"

        if distance < self.min_stop_distance:
            return f"Stop too tight: ${distance:.2f} < ${self.min_stop_distance:.2f} minimum"
        if distance > self.max_stop_distance:
            return f"Stop too wide: ${distance:.2f} > ${self.max_stop_distance:.2f} maximum"

        return None


@dataclass
class ActiveOrder:
    """Representation of an active order for state output."""
    order_id: int
    type: str  # "LIMIT", "STOP", "MARKET"
    side: str  # "BUY", "SELL"
    price: float
    size: int
    status: str  # "PENDING", "PARTIAL", "FILLED", "CANCELLED"
    purpose: str  # "ENTRY", "STOP_LOSS", "PROFIT_TARGET", "EXIT"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "order_id": self.order_id,
            "type": self.type,
            "side": self.side,
            "price": self.price,
            "size": self.size,
            "status": self.status,
            "purpose": self.purpose,
        }
