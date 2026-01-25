#!/usr/bin/env python3
"""
FSDTrader Brain Module: Validation

Pre-execution validation for tool calls.
Ensures tool calls are valid before sending to the Executor.
"""
from typing import Dict, Any, Optional

from .types import ToolCall, ValidationResult
from .tools import (
    TOOL_ENTER_LONG,
    TOOL_ENTER_SHORT,
    TOOL_UPDATE_STOP,
    TOOL_UPDATE_TARGET,
    TOOL_EXIT_POSITION,
    TOOL_WAIT,
    is_valid_tool,
    requires_position,
    requires_flat,
)


# Validation constants
MIN_STOP_DISTANCE = 0.10  # 10 cents
MAX_STOP_DISTANCE = 0.30  # 30 cents
MAX_PRICE_DEVIATION_PCT = 0.01  # 1% from current price
MAX_POSITION_SIZE = 100
SPREAD_LIMIT_OPEN_DRIVE = 0.15
SPREAD_LIMIT_DEFAULT = 0.08


class ToolValidator:
    """
    Validates tool calls before execution.

    Checks:
    - Tool name is valid
    - Required parameters are present
    - Price values are reasonable
    - Stop distance is within limits
    - Position state is compatible
    - Conviction level is appropriate
    """

    def validate(
        self,
        tool_call: ToolCall,
        market_state: Dict[str, Any],
        account_state: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate a tool call.

        Args:
            tool_call: The tool call to validate
            market_state: Current market state
            account_state: Current account/position state

        Returns:
            ValidationResult with valid flag and any errors/warnings
        """
        result = ValidationResult(valid=True)

        # Check tool name
        if not is_valid_tool(tool_call.tool):
            result.add_error(f"INVALID_TOOL: Unknown tool '{tool_call.tool}'")
            return result

        # Check reasoning is provided
        if not tool_call.reasoning:
            result.add_error("MISSING_REASONING: Tool call must include reasoning")

        # Get market data
        mkt = market_state.get("MARKET_STATE", market_state)
        current_price = mkt.get("LAST", 0)
        spread = mkt.get("SPREAD", 0)
        time_session = mkt.get("TIME_SESSION", "")

        # Get position data
        position_side = account_state.get("POSITION_SIDE", "FLAT")
        position_size = account_state.get("POSITION", 0)
        is_flat = position_side == "FLAT" or position_size == 0

        # Check position requirements
        if requires_flat(tool_call.tool) and not is_flat:
            result.add_error(f"ALREADY_IN_POSITION: Cannot {tool_call.tool} while in position")

        if requires_position(tool_call.tool) and is_flat:
            result.add_error(f"NO_POSITION: Cannot {tool_call.tool} without active position")

        # Tool-specific validation
        if tool_call.tool == TOOL_ENTER_LONG:
            self._validate_enter_long(tool_call, result, current_price, spread, time_session)
        elif tool_call.tool == TOOL_ENTER_SHORT:
            self._validate_enter_short(tool_call, result, current_price, spread, time_session)
        elif tool_call.tool == TOOL_UPDATE_STOP:
            self._validate_update_stop(tool_call, result, position_side, account_state)
        elif tool_call.tool == TOOL_UPDATE_TARGET:
            self._validate_update_target(tool_call, result, position_side, account_state)
        elif tool_call.tool == TOOL_EXIT_POSITION:
            self._validate_exit_position(tool_call, result)
        elif tool_call.tool == TOOL_WAIT:
            self._validate_wait(tool_call, result)

        return result

    def _validate_enter_long(
        self,
        tool_call: ToolCall,
        result: ValidationResult,
        current_price: float,
        spread: float,
        time_session: str
    ):
        """Validate enter_long tool call."""
        args = tool_call.arguments

        # Check required fields
        limit_price = args.get("limit_price")
        stop_loss = args.get("stop_loss")
        profit_target = args.get("profit_target")
        conviction = args.get("conviction")

        if limit_price is None:
            result.add_error("MISSING_REQUIRED_FIELD: limit_price is required")
            return
        if stop_loss is None:
            result.add_error("MISSING_REQUIRED_FIELD: stop_loss is required")
            return
        if profit_target is None:
            result.add_error("MISSING_REQUIRED_FIELD: profit_target is required")
            return
        if conviction is None:
            result.add_error("MISSING_REQUIRED_FIELD: conviction is required")
            return

        # Check stop is below entry
        if stop_loss >= limit_price:
            result.add_error(f"INVALID_STOP_DIRECTION: stop_loss ({stop_loss}) must be below limit_price ({limit_price})")

        # Check target is above entry
        if profit_target <= limit_price:
            result.add_error(f"INVALID_TARGET_DIRECTION: profit_target ({profit_target}) must be above limit_price ({limit_price})")

        # Check stop distance
        stop_distance = abs(limit_price - stop_loss)
        if stop_distance < MIN_STOP_DISTANCE:
            result.add_error(f"STOP_TOO_TIGHT: Stop distance ({stop_distance:.2f}) is less than minimum ({MIN_STOP_DISTANCE})")
        if stop_distance > MAX_STOP_DISTANCE:
            result.add_error(f"STOP_TOO_WIDE: Stop distance ({stop_distance:.2f}) exceeds maximum ({MAX_STOP_DISTANCE})")

        # Check price is near market
        if current_price > 0:
            price_deviation = abs(limit_price - current_price) / current_price
            if price_deviation > MAX_PRICE_DEVIATION_PCT:
                result.add_error(f"PRICE_TOO_FAR_FROM_MARKET: limit_price ({limit_price}) is {price_deviation*100:.1f}% from current ({current_price})")

        # Check spread limit
        spread_limit = SPREAD_LIMIT_OPEN_DRIVE if time_session == "OPEN_DRIVE" else SPREAD_LIMIT_DEFAULT
        if spread > spread_limit:
            result.add_error(f"SPREAD_TOO_WIDE: Current spread ({spread}) exceeds limit ({spread_limit}) for {time_session}")

        # Check size
        size = args.get("size", 100)
        if size > MAX_POSITION_SIZE:
            result.add_error(f"SIZE_TOO_LARGE: Size ({size}) exceeds maximum ({MAX_POSITION_SIZE})")
        if size <= 0:
            result.add_error(f"INVALID_SIZE: Size must be positive, got {size}")

        # Check conviction
        if conviction not in ["HIGH", "MEDIUM", "LOW"]:
            result.add_error(f"INVALID_CONVICTION: Must be HIGH, MEDIUM, or LOW, got {conviction}")

        # Warning for low conviction
        if conviction == "LOW":
            result.add_warning("LOW_CONVICTION: Consider waiting for better setup")

    def _validate_enter_short(
        self,
        tool_call: ToolCall,
        result: ValidationResult,
        current_price: float,
        spread: float,
        time_session: str
    ):
        """Validate enter_short tool call."""
        args = tool_call.arguments

        # Check required fields
        limit_price = args.get("limit_price")
        stop_loss = args.get("stop_loss")
        profit_target = args.get("profit_target")
        conviction = args.get("conviction")

        if limit_price is None:
            result.add_error("MISSING_REQUIRED_FIELD: limit_price is required")
            return
        if stop_loss is None:
            result.add_error("MISSING_REQUIRED_FIELD: stop_loss is required")
            return
        if profit_target is None:
            result.add_error("MISSING_REQUIRED_FIELD: profit_target is required")
            return
        if conviction is None:
            result.add_error("MISSING_REQUIRED_FIELD: conviction is required")
            return

        # Check stop is above entry (for short)
        if stop_loss <= limit_price:
            result.add_error(f"INVALID_STOP_DIRECTION: stop_loss ({stop_loss}) must be above limit_price ({limit_price}) for short")

        # Check target is below entry (for short)
        if profit_target >= limit_price:
            result.add_error(f"INVALID_TARGET_DIRECTION: profit_target ({profit_target}) must be below limit_price ({limit_price}) for short")

        # Check stop distance
        stop_distance = abs(stop_loss - limit_price)
        if stop_distance < MIN_STOP_DISTANCE:
            result.add_error(f"STOP_TOO_TIGHT: Stop distance ({stop_distance:.2f}) is less than minimum ({MIN_STOP_DISTANCE})")
        if stop_distance > MAX_STOP_DISTANCE:
            result.add_error(f"STOP_TOO_WIDE: Stop distance ({stop_distance:.2f}) exceeds maximum ({MAX_STOP_DISTANCE})")

        # Check price is near market
        if current_price > 0:
            price_deviation = abs(limit_price - current_price) / current_price
            if price_deviation > MAX_PRICE_DEVIATION_PCT:
                result.add_error(f"PRICE_TOO_FAR_FROM_MARKET: limit_price ({limit_price}) is {price_deviation*100:.1f}% from current ({current_price})")

        # Check spread limit
        spread_limit = SPREAD_LIMIT_OPEN_DRIVE if time_session == "OPEN_DRIVE" else SPREAD_LIMIT_DEFAULT
        if spread > spread_limit:
            result.add_error(f"SPREAD_TOO_WIDE: Current spread ({spread}) exceeds limit ({spread_limit}) for {time_session}")

        # Check size
        size = args.get("size", 100)
        if size > MAX_POSITION_SIZE:
            result.add_error(f"SIZE_TOO_LARGE: Size ({size}) exceeds maximum ({MAX_POSITION_SIZE})")
        if size <= 0:
            result.add_error(f"INVALID_SIZE: Size must be positive, got {size}")

        # Check conviction
        if conviction not in ["HIGH", "MEDIUM", "LOW"]:
            result.add_error(f"INVALID_CONVICTION: Must be HIGH, MEDIUM, or LOW, got {conviction}")

        # Warning for low conviction
        if conviction == "LOW":
            result.add_warning("LOW_CONVICTION: Consider waiting for better setup")

    def _validate_update_stop(
        self,
        tool_call: ToolCall,
        result: ValidationResult,
        position_side: str,
        account_state: Dict[str, Any]
    ):
        """Validate update_stop tool call."""
        args = tool_call.arguments

        new_price = args.get("new_price")
        if new_price is None:
            result.add_error("MISSING_REQUIRED_FIELD: new_price is required")
            return

        if new_price <= 0:
            result.add_error(f"INVALID_PRICE: new_price must be positive, got {new_price}")

        # Check stop direction based on position
        avg_entry = account_state.get("AVG_ENTRY", 0)
        if avg_entry > 0:
            if position_side == "LONG" and new_price > avg_entry:
                # Trailing into profit - this is fine
                pass
            elif position_side == "SHORT" and new_price < avg_entry:
                # Trailing into profit for short - this is fine
                pass

    def _validate_update_target(
        self,
        tool_call: ToolCall,
        result: ValidationResult,
        position_side: str,
        account_state: Dict[str, Any]
    ):
        """Validate update_target tool call."""
        args = tool_call.arguments

        new_price = args.get("new_price")
        if new_price is None:
            result.add_error("MISSING_REQUIRED_FIELD: new_price is required")
            return

        if new_price <= 0:
            result.add_error(f"INVALID_PRICE: new_price must be positive, got {new_price}")

    def _validate_exit_position(
        self,
        tool_call: ToolCall,
        result: ValidationResult
    ):
        """Validate exit_position tool call."""
        # Just need reasoning, which is already checked
        pass

    def _validate_wait(
        self,
        tool_call: ToolCall,
        result: ValidationResult
    ):
        """Validate wait tool call."""
        # Just need reasoning, which is already checked
        pass


# Module-level instance
_validator = ToolValidator()


def validate_tool_call(
    tool_call: ToolCall,
    market_state: Dict[str, Any],
    account_state: Dict[str, Any]
) -> ValidationResult:
    """
    Convenience function to validate a tool call.

    Args:
        tool_call: The tool call to validate
        market_state: Current market state
        account_state: Current account/position state

    Returns:
        ValidationResult with valid flag and any errors/warnings
    """
    return _validator.validate(tool_call, market_state, account_state)
