#!/usr/bin/env python3
"""
FSDTrader: Execution Layer
Order Management, Bracket Orders, and Position Tracking via IBKR

DSL Commands Handled:
- ENTER_LONG(limit_price, stop_loss, profit_target)
- ENTER_SHORT(limit_price, stop_loss, profit_target)
- UPDATE_STOP(new_price)
- UPDATE_TARGET(new_price)
- CANCEL_ALL()
- WAIT(reason)
"""
import logging
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import time

try:
    from ib_insync import (
        IB, Stock, Order, LimitOrder, StopOrder, 
        Trade, OrderStatus, Contract
    )
except ImportError:
    IB = None
    Stock = None


class PositionSide(Enum):
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


@dataclass
class BracketOrder:
    """Represents a bracket order (entry + stop + target)."""
    entry_order_id: int = 0
    stop_order_id: int = 0
    target_order_id: int = 0
    entry_price: float = 0.0
    stop_price: float = 0.0
    target_price: float = 0.0
    side: str = ""  # "BUY" or "SELL"
    quantity: int = 0
    status: str = "PENDING"  # PENDING, FILLED, PARTIAL, CANCELLED


@dataclass
class RiskLimits:
    """Hardcoded risk controls."""
    max_position_size: int = 100
    max_daily_loss: float = -500.0
    max_daily_trades: int = 10
    max_spread: float = 0.05
    min_stop_distance: float = 0.10
    max_stop_distance: float = 0.50


class OrderManager:
    """
    Manages order execution, position tracking, and risk enforcement.
    """
    
    def __init__(self, ib: IB, symbol: str, risk_limits: RiskLimits = None):
        self.ib = ib
        self.symbol = symbol
        self.contract = Stock(symbol, 'SMART', 'USD') if Stock else None
        self.logger = logging.getLogger(f"EXEC_{symbol}")
        
        # Risk limits
        self.limits = risk_limits or RiskLimits()
        
        # State
        self.position = Position()
        self.active_bracket: Optional[BracketOrder] = None
        self.pending_orders: Dict[int, Trade] = {}
        
        # Daily stats
        self.daily_pnl: float = 0.0
        self.daily_trades: int = 0
        self.session_start: float = time.time()
        
        # Hook into order events
        if self.ib:
            self.ib.orderStatusEvent += self._on_order_status
            self.ib.execDetailsEvent += self._on_execution
    
    # =========================================================================
    # PUBLIC API: Execute DSL Commands
    # =========================================================================
    
    def execute(self, command: str, current_spread: float = 0.0) -> Dict[str, Any]:
        """
        Parse and execute a DSL command.
        Returns execution result with status and details.
        """
        # Pre-flight checks
        if self.daily_pnl <= self.limits.max_daily_loss:
            return {"success": False, "error": "MAX_DAILY_LOSS_HIT", "command": command}
        
        if self.daily_trades >= self.limits.max_daily_trades:
            return {"success": False, "error": "MAX_TRADES_HIT", "command": command}
        
        if current_spread > self.limits.max_spread:
            return {"success": False, "error": "SPREAD_TOO_WIDE", "command": command}
        
        # Parse command
        cmd_name, args = self._parse_command(command)
        
        # Route to handler
        handlers = {
            "ENTER_LONG": self._enter_long,
            "ENTER_SHORT": self._enter_short,
            "UPDATE_STOP": self._update_stop,
            "UPDATE_TARGET": self._update_target,
            "CANCEL_ALL": self._cancel_all,
            "WAIT": self._wait,
        }
        
        handler = handlers.get(cmd_name)
        if not handler:
            return {"success": False, "error": "UNKNOWN_COMMAND", "command": command}
        
        return handler(args)
    
    # =========================================================================
    # COMMAND HANDLERS
    # =========================================================================
    
    def _enter_long(self, args: List[str]) -> Dict[str, Any]:
        """Handle ENTER_LONG(limit, stop, target)."""
        if self.position.side != PositionSide.FLAT:
            return {"success": False, "error": "ALREADY_IN_POSITION"}
        
        if len(args) != 3:
            return {"success": False, "error": "INVALID_ARGS", "expected": 3}
        
        try:
            limit_price = float(args[0])
            stop_price = float(args[1])
            target_price = float(args[2])
        except ValueError:
            return {"success": False, "error": "INVALID_PRICE_FORMAT"}
        
        # Validate stop distance
        stop_distance = limit_price - stop_price
        if stop_distance < self.limits.min_stop_distance:
            return {"success": False, "error": "STOP_TOO_TIGHT", 
                    "distance": stop_distance, "min": self.limits.min_stop_distance}
        if stop_distance > self.limits.max_stop_distance:
            return {"success": False, "error": "STOP_TOO_WIDE",
                    "distance": stop_distance, "max": self.limits.max_stop_distance}
        
        # Create bracket order
        quantity = self.limits.max_position_size
        
        if not self.ib:
            # Simulation mode
            self.logger.info(f"[SIM] ENTER_LONG: {quantity} @ {limit_price}, "
                           f"Stop: {stop_price}, Target: {target_price}")
            self.position = Position(
                side=PositionSide.LONG,
                size=quantity,
                avg_entry=limit_price,
                entry_time=time.time()
            )
            self.daily_trades += 1
            return {"success": True, "mode": "SIMULATION", "position": "LONG",
                    "entry": limit_price, "stop": stop_price, "target": target_price}
        
        # Real order submission
        return self._submit_bracket("BUY", quantity, limit_price, stop_price, target_price)
    
    def _enter_short(self, args: List[str]) -> Dict[str, Any]:
        """Handle ENTER_SHORT(limit, stop, target)."""
        if self.position.side != PositionSide.FLAT:
            return {"success": False, "error": "ALREADY_IN_POSITION"}
        
        if len(args) != 3:
            return {"success": False, "error": "INVALID_ARGS", "expected": 3}
        
        try:
            limit_price = float(args[0])
            stop_price = float(args[1])
            target_price = float(args[2])
        except ValueError:
            return {"success": False, "error": "INVALID_PRICE_FORMAT"}
        
        # Validate stop distance (for short, stop is above entry)
        stop_distance = stop_price - limit_price
        if stop_distance < self.limits.min_stop_distance:
            return {"success": False, "error": "STOP_TOO_TIGHT"}
        if stop_distance > self.limits.max_stop_distance:
            return {"success": False, "error": "STOP_TOO_WIDE"}
        
        quantity = self.limits.max_position_size
        
        if not self.ib:
            self.logger.info(f"[SIM] ENTER_SHORT: {quantity} @ {limit_price}")
            self.position = Position(
                side=PositionSide.SHORT,
                size=quantity,
                avg_entry=limit_price,
                entry_time=time.time()
            )
            self.daily_trades += 1
            return {"success": True, "mode": "SIMULATION", "position": "SHORT"}
        
        return self._submit_bracket("SELL", quantity, limit_price, stop_price, target_price)
    
    def _update_stop(self, args: List[str]) -> Dict[str, Any]:
        """Handle UPDATE_STOP(new_price)."""
        if self.position.side == PositionSide.FLAT:
            return {"success": False, "error": "NO_POSITION"}
        
        if len(args) != 1:
            return {"success": False, "error": "INVALID_ARGS"}
        
        try:
            new_stop = float(args[0])
        except ValueError:
            return {"success": False, "error": "INVALID_PRICE_FORMAT"}
        
        # Validate stop direction
        if self.position.side == PositionSide.LONG:
            if new_stop > self.position.avg_entry:
                self.logger.info(f"Trailing stop to profit: {new_stop}")
            if new_stop <= 0:
                return {"success": False, "error": "INVALID_STOP_PRICE"}
        
        if not self.ib or not self.active_bracket:
            self.logger.info(f"[SIM] UPDATE_STOP: {new_stop}")
            if self.active_bracket:
                self.active_bracket.stop_price = new_stop
            return {"success": True, "mode": "SIMULATION", "new_stop": new_stop}
        
        # Modify the stop order
        return self._modify_stop_order(new_stop)
    
    def _update_target(self, args: List[str]) -> Dict[str, Any]:
        """Handle UPDATE_TARGET(new_price)."""
        if self.position.side == PositionSide.FLAT:
            return {"success": False, "error": "NO_POSITION"}
        
        if len(args) != 1:
            return {"success": False, "error": "INVALID_ARGS"}
        
        try:
            new_target = float(args[0])
        except ValueError:
            return {"success": False, "error": "INVALID_PRICE_FORMAT"}
        
        if not self.ib or not self.active_bracket:
            self.logger.info(f"[SIM] UPDATE_TARGET: {new_target}")
            if self.active_bracket:
                self.active_bracket.target_price = new_target
            return {"success": True, "mode": "SIMULATION", "new_target": new_target}
        
        return self._modify_target_order(new_target)
    
    def _cancel_all(self, args: List[str]) -> Dict[str, Any]:
        """Handle CANCEL_ALL() - exits position and cancels orders."""
        if not self.ib:
            self.logger.info("[SIM] CANCEL_ALL")
            if self.position.side != PositionSide.FLAT:
                # Simulate exit
                self.position = Position()
                self.active_bracket = None
            return {"success": True, "mode": "SIMULATION", "action": "ALL_CANCELLED"}
        
        # Cancel all pending orders
        for order_id, trade in list(self.pending_orders.items()):
            self.ib.cancelOrder(trade.order)
        
        # If in position, submit market exit
        if self.position.side != PositionSide.FLAT:
            exit_side = "SELL" if self.position.side == PositionSide.LONG else "BUY"
            exit_order = Order()
            exit_order.action = exit_side
            exit_order.totalQuantity = self.position.size
            exit_order.orderType = "MKT"
            self.ib.placeOrder(self.contract, exit_order)
        
        self.active_bracket = None
        return {"success": True, "action": "ALL_CANCELLED"}
    
    def _wait(self, args: List[str]) -> Dict[str, Any]:
        """Handle WAIT(reason) - no-op."""
        reason = args[0] if args else "No reason"
        self.logger.debug(f"WAIT: {reason}")
        return {"success": True, "action": "WAIT", "reason": reason}
    
    # =========================================================================
    # IBKR ORDER SUBMISSION
    # =========================================================================
    
    def _submit_bracket(self, action: str, quantity: int, 
                        limit_price: float, stop_price: float, 
                        target_price: float) -> Dict[str, Any]:
        """Submit a bracket order to IBKR."""
        if not self.ib:
            return {"success": False, "error": "NO_CONNECTION"}
        
        try:
            # Parent order (entry)
            parent = Order()
            parent.orderId = self.ib.client.getReqId()
            parent.action = action
            parent.totalQuantity = quantity
            parent.orderType = "LMT"
            parent.lmtPrice = limit_price
            parent.transmit = False  # Don't transmit yet
            
            # Take profit order
            take_profit = Order()
            take_profit.orderId = self.ib.client.getReqId()
            take_profit.action = "SELL" if action == "BUY" else "BUY"
            take_profit.totalQuantity = quantity
            take_profit.orderType = "LMT"
            take_profit.lmtPrice = target_price
            take_profit.parentId = parent.orderId
            take_profit.transmit = False
            
            # Stop loss order
            stop_loss = Order()
            stop_loss.orderId = self.ib.client.getReqId()
            stop_loss.action = "SELL" if action == "BUY" else "BUY"
            stop_loss.totalQuantity = quantity
            stop_loss.orderType = "STP"
            stop_loss.auxPrice = stop_price
            stop_loss.parentId = parent.orderId
            stop_loss.transmit = True  # Transmit all orders
            
            # Place all three orders
            parent_trade = self.ib.placeOrder(self.contract, parent)
            tp_trade = self.ib.placeOrder(self.contract, take_profit)
            sl_trade = self.ib.placeOrder(self.contract, stop_loss)
            
            # Track orders
            self.pending_orders[parent.orderId] = parent_trade
            self.pending_orders[take_profit.orderId] = tp_trade
            self.pending_orders[stop_loss.orderId] = sl_trade
            
            # Track bracket
            self.active_bracket = BracketOrder(
                entry_order_id=parent.orderId,
                stop_order_id=stop_loss.orderId,
                target_order_id=take_profit.orderId,
                entry_price=limit_price,
                stop_price=stop_price,
                target_price=target_price,
                side=action,
                quantity=quantity,
                status="PENDING"
            )
            
            self.daily_trades += 1
            self.logger.info(f"Bracket submitted: {action} {quantity} @ {limit_price}")
            
            return {
                "success": True,
                "entry_order_id": parent.orderId,
                "stop_order_id": stop_loss.orderId,
                "target_order_id": take_profit.orderId
            }
            
        except Exception as e:
            self.logger.error(f"Bracket submission failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _modify_stop_order(self, new_price: float) -> Dict[str, Any]:
        """Modify existing stop order price."""
        if not self.active_bracket:
            return {"success": False, "error": "NO_ACTIVE_BRACKET"}
        
        order_id = self.active_bracket.stop_order_id
        if order_id not in self.pending_orders:
            return {"success": False, "error": "STOP_ORDER_NOT_FOUND"}
        
        try:
            trade = self.pending_orders[order_id]
            trade.order.auxPrice = new_price
            self.ib.placeOrder(self.contract, trade.order)
            
            self.active_bracket.stop_price = new_price
            self.logger.info(f"Stop updated to {new_price}")
            
            return {"success": True, "new_stop": new_price}
        except Exception as e:
            self.logger.error(f"Stop modification failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _modify_target_order(self, new_price: float) -> Dict[str, Any]:
        """Modify existing target order price."""
        if not self.active_bracket:
            return {"success": False, "error": "NO_ACTIVE_BRACKET"}
        
        order_id = self.active_bracket.target_order_id
        if order_id not in self.pending_orders:
            return {"success": False, "error": "TARGET_ORDER_NOT_FOUND"}
        
        try:
            trade = self.pending_orders[order_id]
            trade.order.lmtPrice = new_price
            self.ib.placeOrder(self.contract, trade.order)
            
            self.active_bracket.target_price = new_price
            self.logger.info(f"Target updated to {new_price}")
            
            return {"success": True, "new_target": new_price}
        except Exception as e:
            self.logger.error(f"Target modification failed: {e}")
            return {"success": False, "error": str(e)}
    
    # =========================================================================
    # EVENT HANDLERS
    # =========================================================================
    
    def _on_order_status(self, trade: Trade):
        """Handle order status updates from IBKR."""
        order_id = trade.order.orderId
        status = trade.orderStatus.status
        
        self.logger.debug(f"Order {order_id} status: {status}")
        
        if status == "Filled":
            self._handle_fill(trade)
        elif status == "Cancelled":
            self._handle_cancel(trade)
    
    def _on_execution(self, trade: Trade, fill):
        """Handle execution details."""
        self.logger.info(f"Execution: {fill.execution.side} {fill.execution.shares} "
                        f"@ {fill.execution.price}")
    
    def _handle_fill(self, trade: Trade):
        """Process a filled order."""
        if not self.active_bracket:
            return
        
        order_id = trade.order.orderId
        fill_price = trade.orderStatus.avgFillPrice
        
        if order_id == self.active_bracket.entry_order_id:
            # Entry filled
            self.position = Position(
                side=PositionSide.LONG if self.active_bracket.side == "BUY" else PositionSide.SHORT,
                size=self.active_bracket.quantity,
                avg_entry=fill_price,
                entry_time=time.time()
            )
            self.active_bracket.status = "FILLED"
            self.logger.info(f"ENTRY FILLED @ {fill_price}")
            
        elif order_id in (self.active_bracket.stop_order_id, 
                          self.active_bracket.target_order_id):
            # Exit filled (stop or target)
            exit_price = fill_price
            entry_price = self.position.avg_entry
            
            if self.position.side == PositionSide.LONG:
                pnl = (exit_price - entry_price) * self.position.size
            else:
                pnl = (entry_price - exit_price) * self.position.size
            
            self.daily_pnl += pnl
            self.logger.info(f"EXIT FILLED @ {exit_price} | PnL: ${pnl:.2f}")
            
            # Reset position
            self.position = Position()
            self.active_bracket = None
            
            # Cancel remaining bracket leg
            self._cleanup_bracket_orders(order_id)
    
    def _handle_cancel(self, trade: Trade):
        """Handle cancelled order."""
        order_id = trade.order.orderId
        if order_id in self.pending_orders:
            del self.pending_orders[order_id]
    
    def _cleanup_bracket_orders(self, filled_order_id: int):
        """Cancel remaining bracket leg after one side fills."""
        if not self.active_bracket:
            return
        
        # If stop filled, cancel target. If target filled, cancel stop.
        other_order_id = None
        if filled_order_id == self.active_bracket.stop_order_id:
            other_order_id = self.active_bracket.target_order_id
        elif filled_order_id == self.active_bracket.target_order_id:
            other_order_id = self.active_bracket.stop_order_id
        
        if other_order_id and other_order_id in self.pending_orders:
            trade = self.pending_orders[other_order_id]
            self.ib.cancelOrder(trade.order)
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def _parse_command(self, command: str) -> tuple:
        """Parse DSL command into (name, args)."""
        match = re.match(r'([A-Z_]+)\((.*)\)', command)
        if not match:
            return (command, [])
        
        cmd_name = match.group(1)
        args_str = match.group(2).strip()
        
        if not args_str:
            return (cmd_name, [])
        
        # Split by comma, handle quoted strings
        args = []
        current = ""
        in_quotes = False
        
        for char in args_str:
            if char == '"':
                in_quotes = not in_quotes
            elif char == ',' and not in_quotes:
                args.append(current.strip().strip('"'))
                current = ""
            else:
                current += char
        
        if current:
            args.append(current.strip().strip('"'))
        
        return (cmd_name, args)
    
    def get_state(self) -> Dict[str, Any]:
        """Return current execution state for monitoring."""
        return {
            "position": {
                "side": self.position.side.value,
                "size": self.position.size,
                "avg_entry": self.position.avg_entry,
                "unrealized_pnl": self.position.unrealized_pnl
            },
            "daily_stats": {
                "pnl": round(self.daily_pnl, 2),
                "trades": self.daily_trades
            },
            "active_bracket": {
                "entry": self.active_bracket.entry_price if self.active_bracket else None,
                "stop": self.active_bracket.stop_price if self.active_bracket else None,
                "target": self.active_bracket.target_price if self.active_bracket else None,
                "status": self.active_bracket.status if self.active_bracket else None
            } if self.active_bracket else None,
            "limits": {
                "max_position": self.limits.max_position_size,
                "max_daily_loss": self.limits.max_daily_loss,
                "max_trades": self.limits.max_daily_trades
            }
        }
