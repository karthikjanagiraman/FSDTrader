#!/usr/bin/env python3
"""
FSDTrader Execution Module: Base Provider Interface

Abstract interface for order execution providers.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

from .types import OrderResult, Position, RiskLimits, TradeRecord


class ExecutionProvider(ABC):
    """
    Abstract interface for order execution.

    Implementations:
    - SimulatedExecutor: Virtual execution for backtest/sim
    - IBKRExecutor: Real execution via IBKR TWS API

    Both implementations MUST:
    - Enforce the same risk limits
    - Produce identical state output format
    - Handle bracket orders (entry + stop + target)
    """

    @abstractmethod
    def submit_bracket_order(
        self,
        side: str,
        size: int,
        limit_price: float,
        stop_loss: float,
        profit_target: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> OrderResult:
        """
        Submit a bracket order (entry + stop + target).

        Args:
            side: "BUY" for long, "SELL" for short
            size: Number of shares
            limit_price: Entry limit price
            stop_loss: Stop loss price
            profit_target: Take profit price
            context: Optional market context at entry (for trade records)

        Returns:
            OrderResult with success status and order IDs
        """
        pass

    @abstractmethod
    def modify_stop(self, new_price: float) -> OrderResult:
        """
        Modify stop loss price.

        Args:
            new_price: New stop price

        Returns:
            OrderResult with success status
        """
        pass

    @abstractmethod
    def modify_target(self, new_price: float) -> OrderResult:
        """
        Modify profit target price.

        Args:
            new_price: New target price

        Returns:
            OrderResult with success status
        """
        pass

    @abstractmethod
    def exit_position(self, reason: str = "MANUAL") -> OrderResult:
        """
        Exit current position at market.

        Args:
            reason: Exit reason for trade record

        Returns:
            OrderResult with success status
        """
        pass

    @abstractmethod
    def cancel_all(self) -> OrderResult:
        """
        Cancel all pending orders.

        Returns:
            OrderResult with success status
        """
        pass

    @abstractmethod
    def update(self, current_price: float, timestamp: float) -> None:
        """
        Update executor state with current market price.

        For SimulatedExecutor: Checks if stop/target should fill.
        For IBKRExecutor: Updates unrealized P&L calculation.

        Args:
            current_price: Current market price
            timestamp: Current timestamp (simulated or real)
        """
        pass

    @abstractmethod
    def get_position(self) -> Position:
        """
        Get current position.

        Returns:
            Position object with side, size, avg_entry, unrealized_pnl
        """
        pass

    @abstractmethod
    def get_account_state(self) -> Dict[str, Any]:
        """
        Get account state for Brain context.

        Returns:
            Dictionary with:
            - POSITION: int (size)
            - POSITION_SIDE: str (FLAT/LONG/SHORT)
            - AVG_ENTRY: float
            - UNREALIZED_PL: float
            - DAILY_PL: float
            - DAILY_TRADES: int
        """
        pass

    @abstractmethod
    def get_active_orders(self) -> List[Dict[str, Any]]:
        """
        Get list of active orders for Brain context.

        Returns:
            List of order dictionaries with:
            - order_id, type, side, price, size, status, purpose
        """
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Get complete execution state.

        Returns:
            Dictionary with position, daily_stats, active_bracket, limits
        """
        pass

    @abstractmethod
    def get_trade_history(self) -> List[TradeRecord]:
        """
        Get completed trade history.

        Returns:
            List of TradeRecord objects
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset executor state for new session.
        Clears position, orders, and daily stats.
        """
        pass

    # =========================================================================
    # Command Execution (DSL parsing)
    # =========================================================================

    def execute(self, command: str, current_spread: float = 0.0,
                context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a command string from the Brain.

        Command format: "TOOL_NAME|arg1=val1|arg2=val2|reasoning=..."

        Args:
            command: Command string from Brain
            current_spread: Current market spread for validation
            context: Optional market context for trade records

        Returns:
            Execution result dictionary
        """
        # Parse command
        parts = command.split("|")
        cmd_name = parts[0].upper()
        args = {}
        for part in parts[1:]:
            if "=" in part:
                key, value = part.split("=", 1)
                args[key.strip()] = value.strip()

        # Pre-flight risk checks
        if hasattr(self, 'risk_limits'):
            if self._daily_pnl <= self.risk_limits.max_daily_loss:
                return {"success": False, "error": "MAX_DAILY_LOSS_HIT", "command": command}
            if self._daily_trades >= self.risk_limits.max_daily_trades:
                return {"success": False, "error": "MAX_TRADES_HIT", "command": command}

        # Route to handler
        if cmd_name == "ENTER_LONG":
            return self._handle_enter_long(args, current_spread, context)
        elif cmd_name == "ENTER_SHORT":
            return self._handle_enter_short(args, current_spread, context)
        elif cmd_name == "UPDATE_STOP":
            return self._handle_update_stop(args)
        elif cmd_name == "UPDATE_TARGET":
            return self._handle_update_target(args)
        elif cmd_name == "EXIT_POSITION":
            return self._handle_exit_position(args)
        elif cmd_name == "WAIT":
            return {"success": True, "action": "WAIT", "reason": args.get("reasoning", "")}
        else:
            return {"success": False, "error": f"UNKNOWN_COMMAND: {cmd_name}"}

    def _handle_enter_long(self, args: Dict[str, str], spread: float,
                           context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle ENTER_LONG command."""
        try:
            limit_price = float(args.get("limit_price", 0))
            stop_loss = float(args.get("stop_loss", 0))
            profit_target = float(args.get("profit_target", 0))
            size = int(args.get("size", self.risk_limits.max_position_size))
        except (ValueError, TypeError) as e:
            return {"success": False, "error": f"INVALID_ARGS: {e}"}

        # Validate spread
        if spread > self.risk_limits.max_spread:
            return {"success": False, "error": f"SPREAD_TOO_WIDE: {spread} > {self.risk_limits.max_spread}"}

        result = self.submit_bracket_order(
            side="BUY",
            size=size,
            limit_price=limit_price,
            stop_loss=stop_loss,
            profit_target=profit_target,
            context=context,
        )
        return result.to_dict()

    def _handle_enter_short(self, args: Dict[str, str], spread: float,
                            context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle ENTER_SHORT command."""
        try:
            limit_price = float(args.get("limit_price", 0))
            stop_loss = float(args.get("stop_loss", 0))
            profit_target = float(args.get("profit_target", 0))
            size = int(args.get("size", self.risk_limits.max_position_size))
        except (ValueError, TypeError) as e:
            return {"success": False, "error": f"INVALID_ARGS: {e}"}

        if spread > self.risk_limits.max_spread:
            return {"success": False, "error": f"SPREAD_TOO_WIDE: {spread} > {self.risk_limits.max_spread}"}

        result = self.submit_bracket_order(
            side="SELL",
            size=size,
            limit_price=limit_price,
            stop_loss=stop_loss,
            profit_target=profit_target,
            context=context,
        )
        return result.to_dict()

    def _handle_update_stop(self, args: Dict[str, str]) -> Dict[str, Any]:
        """Handle UPDATE_STOP command."""
        try:
            new_price = float(args.get("new_price", 0))
        except (ValueError, TypeError) as e:
            return {"success": False, "error": f"INVALID_ARGS: {e}"}

        result = self.modify_stop(new_price)
        return result.to_dict()

    def _handle_update_target(self, args: Dict[str, str]) -> Dict[str, Any]:
        """Handle UPDATE_TARGET command."""
        try:
            new_price = float(args.get("new_price", 0))
        except (ValueError, TypeError) as e:
            return {"success": False, "error": f"INVALID_ARGS: {e}"}

        result = self.modify_target(new_price)
        return result.to_dict()

    def _handle_exit_position(self, args: Dict[str, str]) -> Dict[str, Any]:
        """Handle EXIT_POSITION command."""
        reason = args.get("reasoning", "MANUAL")
        result = self.exit_position(reason=reason)
        return result.to_dict()
