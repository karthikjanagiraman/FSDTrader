#!/usr/bin/env python3
"""
FSDTrader Brain Module: Type Definitions

Data classes for tool calls, decisions, and LLM responses.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime


@dataclass
class ToolCall:
    """
    Represents a parsed tool call from the LLM.

    This is the primary output of the Brain module, containing
    the action to take and all associated metadata.
    """
    tool: str                           # Tool name: "enter_long", "wait", etc.
    arguments: Dict[str, Any]           # Tool arguments
    reasoning: str                      # Extracted reasoning from arguments
    conviction: Optional[str] = None    # HIGH, MEDIUM, LOW, or None
    raw_response: Optional[str] = None  # Full LLM response for logging
    latency_ms: float = 0.0             # LLM call latency

    def is_entry(self) -> bool:
        """Check if this is an entry action."""
        return self.tool in ("enter_long", "enter_short")

    def is_exit(self) -> bool:
        """Check if this is an exit action."""
        return self.tool == "exit_position"

    def is_modification(self) -> bool:
        """Check if this modifies existing orders."""
        return self.tool in ("update_stop", "update_target")

    def is_wait(self) -> bool:
        """Check if this is a wait/no-action."""
        return self.tool == "wait"

    def get_limit_price(self) -> Optional[float]:
        """Get limit price for entry orders."""
        return self.arguments.get("limit_price")

    def get_stop_loss(self) -> Optional[float]:
        """Get stop loss price."""
        if self.is_entry():
            return self.arguments.get("stop_loss")
        elif self.tool == "update_stop":
            return self.arguments.get("new_price")
        return None

    def get_profit_target(self) -> Optional[float]:
        """Get profit target price."""
        if self.is_entry():
            return self.arguments.get("profit_target")
        elif self.tool == "update_target":
            return self.arguments.get("new_price")
        return None

    def get_size(self) -> int:
        """Get position size (default 100)."""
        return self.arguments.get("size", 100)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "tool": self.tool,
            "arguments": self.arguments,
            "reasoning": self.reasoning,
            "conviction": self.conviction,
            "latency_ms": self.latency_ms
        }

    @classmethod
    def wait(cls, reasoning: str, latency_ms: float = 0.0) -> "ToolCall":
        """Create a wait tool call (convenience method)."""
        return cls(
            tool="wait",
            arguments={"reasoning": reasoning},
            reasoning=reasoning,
            conviction=None,
            latency_ms=latency_ms
        )


@dataclass
class Decision:
    """
    Historical decision record for context continuity.

    Stored in the decision history and included in LLM context
    to maintain awareness of recent actions and reasoning.
    """
    timestamp: float                    # Unix timestamp
    tool: str                           # Tool called
    arguments: Dict[str, Any]           # Arguments used
    reasoning: str                      # LLM's reasoning
    conviction: Optional[str] = None    # Conviction level
    market_snapshot: Dict[str, Any] = field(default_factory=dict)  # Key market data
    result: Optional[Dict[str, Any]] = None  # Execution result (filled in later)

    def format_for_context(self) -> str:
        """Format this decision for inclusion in LLM context."""
        time_str = datetime.fromtimestamp(self.timestamp).strftime("%H:%M:%S")
        return f'[{time_str}] {self.tool}\n"{self.reasoning}"'

    @classmethod
    def from_tool_call(cls, tool_call: ToolCall, market_state: Dict[str, Any]) -> "Decision":
        """Create a Decision from a ToolCall."""
        # Extract key market data for the snapshot
        mkt = market_state.get("MARKET_STATE", {})
        snapshot = {
            "LAST": mkt.get("LAST"),
            "L2_IMBALANCE": mkt.get("L2_IMBALANCE"),
            "CVD_TREND": mkt.get("CVD_TREND"),
            "TAPE_VELOCITY": mkt.get("TAPE_VELOCITY"),
            "TAPE_SENTIMENT": mkt.get("TAPE_SENTIMENT"),
        }

        return cls(
            timestamp=datetime.now().timestamp(),
            tool=tool_call.tool,
            arguments=tool_call.arguments,
            reasoning=tool_call.reasoning,
            conviction=tool_call.conviction,
            market_snapshot=snapshot
        )


@dataclass
class LLMResponse:
    """
    Raw response from an LLM provider.

    Contains parsed tool calls and metadata about the API call.
    """
    tool_calls: List[Dict[str, Any]]    # Parsed tool calls from response
    content: Optional[str] = None       # Text content (if any)
    model: str = ""                     # Model used
    usage: Dict[str, int] = field(default_factory=dict)  # Token usage
    latency_ms: float = 0.0             # Request latency

    def has_tool_call(self) -> bool:
        """Check if response contains a tool call."""
        return len(self.tool_calls) > 0

    def get_first_tool_call(self) -> Optional[Dict[str, Any]]:
        """Get the first tool call (we expect exactly one)."""
        if self.tool_calls:
            return self.tool_calls[0]
        return None


@dataclass
class ValidationResult:
    """
    Result of validating a tool call.
    """
    valid: bool                         # Whether validation passed
    errors: List[str] = field(default_factory=list)  # List of error messages
    warnings: List[str] = field(default_factory=list)  # List of warnings

    def add_error(self, error: str):
        """Add an error and mark as invalid."""
        self.valid = False
        self.errors.append(error)

    def add_warning(self, warning: str):
        """Add a warning (doesn't invalidate)."""
        self.warnings.append(warning)


@dataclass
class DecisionLog:
    """
    Complete log entry for a decision.

    Used for detailed logging and post-session analysis.
    """
    timestamp: str
    market_snapshot: Dict[str, Any]     # Key market values at decision time
    context_sent: str                   # Full context string sent to LLM
    system_prompt_version: str          # System prompt version/hash
    provider: str                       # LLM provider used
    model: str                          # Model used
    tool_call: Dict[str, Any]           # Parsed tool call
    raw_response: str                   # Full LLM response
    latency_ms: float                   # LLM call latency
    validation_result: Dict[str, Any]   # Validation pass/fail and errors
    execution_result: Optional[Dict[str, Any]] = None  # Result from Executor

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "market_snapshot": self.market_snapshot,
            "provider": self.provider,
            "model": self.model,
            "tool_call": self.tool_call,
            "latency_ms": self.latency_ms,
            "validation": self.validation_result,
            "execution": self.execution_result
        }
