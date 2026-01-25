#!/usr/bin/env python3
"""
FSDTrader Brain Module: TradingBrain

The main brain class that orchestrates LLM-based trading decisions.
"""
import time
import logging
from typing import Dict, Any, List, Optional
from collections import deque

from .types import ToolCall, Decision, LLMResponse, ValidationResult, DecisionLog
from .tools import TOOLS, TOOL_WAIT, is_valid_tool
from .prompts import get_system_prompt
from .context import ContextBuilder
from .validation import validate_tool_call
from .providers import get_provider, LLMProvider


logger = logging.getLogger(__name__)


# Configuration
DEFAULT_PROVIDER = "grok"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 1024
MAX_DECISION_HISTORY = 20


class TradingBrain:
    """
    The trading brain that makes decisions using LLM tool calling.

    Responsibilities:
    1. Build context from market state
    2. Call LLM with tools
    3. Validate tool calls
    4. Return executable commands

    Usage:
        brain = TradingBrain(api_key="...")
        command = brain.think(market_state)
        # command is a string like "ENTER_LONG|..." or "WAIT|..."
    """

    def __init__(
        self,
        api_key: str,
        provider: str = DEFAULT_PROVIDER,
        model: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        """
        Initialize the trading brain.

        Args:
            api_key: API key for the LLM provider
            provider: Provider name ("grok", "openai", "anthropic")
            model: Model override (uses provider default if None)
            temperature: LLM temperature (0.0 for deterministic)
            max_tokens: Maximum response tokens
        """
        self._provider: LLMProvider = get_provider(
            provider,
            api_key=api_key,
            model=model,
        )
        self._temperature = temperature
        self._max_tokens = max_tokens

        # Components
        self._context_builder = ContextBuilder()
        self._system_prompt = get_system_prompt()

        # State
        self._decision_history: deque = deque(maxlen=MAX_DECISION_HISTORY)
        self._total_calls = 0
        self._total_latency_ms = 0.0

        logger.info(
            f"TradingBrain initialized: provider={provider}, "
            f"model={self._provider.model_name}"
        )

    def think(self, state: Dict[str, Any]) -> str:
        """
        Make a trading decision based on current state.

        This is the main entry point. Takes raw market state,
        builds context, calls LLM, validates response, and
        returns an executable command string.

        Args:
            state: Complete market state dictionary containing:
                - MARKET_STATE: Market data (prices, L2, tape, etc.)
                - ACCOUNT_STATE: Position and P&L
                - ACTIVE_ORDERS: Current open orders

        Returns:
            Command string in format: "TOOL_NAME|arg1=val1|arg2=val2|reasoning=..."
            Examples:
                "WAIT|reasoning=Spread too wide"
                "ENTER_LONG|limit_price=245.50|stop_loss=245.20|..."
        """
        start_time = time.time()

        try:
            # Extract state components
            market_state = state.get("MARKET_STATE", {})
            account_state = state.get("ACCOUNT_STATE", {})
            active_orders = state.get("ACTIVE_ORDERS", [])

            # Build context
            context = self._context_builder.build(
                market_state=market_state,
                account_state=account_state,
                history=list(self._decision_history),
                active_orders=active_orders,
            )

            # Call LLM
            response = self._provider.call(
                system_prompt=self._system_prompt,
                user_message=context,
                tools=TOOLS,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )

            # Parse tool call
            tool_call = self._parse_response(response)

            # Validate
            market_state = state.get("MARKET_STATE", {})
            account_state = state.get("ACCOUNT_STATE", {})
            validation = validate_tool_call(tool_call, market_state, account_state)

            if not validation.valid:
                # Log validation errors and return WAIT
                for error in validation.errors:
                    logger.warning(f"Validation error: {error}")
                return self._format_wait(
                    f"Validation failed: {'; '.join(validation.errors)}"
                )

            # Log warnings
            for warning in validation.warnings:
                logger.info(f"Validation warning: {warning}")

            # Record decision
            decision = Decision(
                timestamp=time.time(),
                tool=tool_call.tool,
                arguments=tool_call.arguments,
                reasoning=tool_call.reasoning,
                conviction=tool_call.conviction,
                market_snapshot={
                    "last": market_state.get("LAST", 0),
                    "spread": market_state.get("SPREAD", 0),
                    "position": account_state.get("POSITION_SIDE", "FLAT"),
                },
            )
            self._decision_history.append(decision)

            # Update stats
            self._total_calls += 1
            self._total_latency_ms += response.latency_ms

            # Log
            latency = (time.time() - start_time) * 1000
            logger.debug(
                f"Brain decision: {tool_call.tool} "
                f"(latency={latency:.0f}ms, llm={response.latency_ms:.0f}ms)"
            )

            # Format command
            return self._format_command(tool_call)

        except Exception as e:
            logger.error(f"Brain error: {e}", exc_info=True)
            return self._format_wait(f"Error: {str(e)}")

    def _parse_response(self, response: LLMResponse) -> ToolCall:
        """
        Parse LLM response into a ToolCall.

        Args:
            response: LLM response with tool calls

        Returns:
            Parsed ToolCall

        Raises:
            ValueError: If no valid tool call in response
        """
        if not response.tool_calls:
            raise ValueError("No tool calls in LLM response")

        # Take first tool call (should only be one)
        call = response.tool_calls[0]
        name = call.get("name", "")
        arguments = call.get("arguments", {})

        # Extract reasoning from arguments
        reasoning = arguments.pop("reasoning", "No reasoning provided")

        # Extract conviction if present
        conviction = arguments.get("conviction")

        return ToolCall(
            tool=name,
            arguments=arguments,
            reasoning=reasoning,
            conviction=conviction,
            latency_ms=response.latency_ms,
        )

    def _format_command(self, tool_call: ToolCall) -> str:
        """
        Format a tool call into a command string.

        The executor expects commands in pipe-delimited format:
        TOOL_NAME|arg1=val1|arg2=val2|reasoning=...

        Args:
            tool_call: Validated tool call

        Returns:
            Formatted command string
        """
        parts = [tool_call.tool.upper()]

        # Add arguments
        for key, value in tool_call.arguments.items():
            parts.append(f"{key}={value}")

        # Add reasoning
        parts.append(f"reasoning={tool_call.reasoning}")

        return "|".join(parts)

    def _format_wait(self, reason: str) -> str:
        """Format a WAIT command."""
        return f"WAIT|reasoning={reason}"

    def get_stats(self) -> Dict[str, Any]:
        """
        Get brain statistics.

        Returns:
            Dictionary with:
                - total_calls: Number of LLM calls
                - avg_latency_ms: Average LLM latency
                - decision_count: Decisions in history
                - provider: Provider name
                - model: Model name
        """
        avg_latency = (
            self._total_latency_ms / self._total_calls
            if self._total_calls > 0
            else 0
        )

        return {
            "total_calls": self._total_calls,
            "avg_latency_ms": avg_latency,
            "decision_count": len(self._decision_history),
            "provider": self._provider.provider_name,
            "model": self._provider.model_name,
        }

    def get_decision_history(self) -> List[Decision]:
        """Get recent decision history."""
        return list(self._decision_history)

    def get_decision_log(self) -> DecisionLog:
        """Get complete decision log for reporting."""
        return DecisionLog(
            decisions=list(self._decision_history),
            provider=self._provider.provider_name,
            model=self._provider.model_name,
            total_calls=self._total_calls,
            avg_latency_ms=(
                self._total_latency_ms / self._total_calls
                if self._total_calls > 0
                else 0
            ),
        )

    def reset(self):
        """Reset brain state (clear history)."""
        self._decision_history.clear()
        self._total_calls = 0
        self._total_latency_ms = 0.0
        logger.info("Brain state reset")
