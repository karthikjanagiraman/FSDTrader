#!/usr/bin/env python3
"""
FSDTrader Brain Module

LLM-based trading decision engine with native tool calling.

Main Components:
- TradingBrain: Main class for making trading decisions
- ToolCall: Data class for parsed tool calls
- Decision: Record of a trading decision
- ContextBuilder: Builds LLM context from market state
- ToolValidator: Validates tool calls before execution

Usage:
    from brain import TradingBrain

    brain = TradingBrain(api_key="your-api-key")
    command = brain.think(market_state)
    # command: "ENTER_LONG|limit_price=245.50|stop_loss=245.20|..."
"""
from .brain import TradingBrain
from .types import ToolCall, Decision, LLMResponse, ValidationResult, DecisionLog
from .tools import (
    TOOLS,
    TOOL_ENTER_LONG,
    TOOL_ENTER_SHORT,
    TOOL_UPDATE_STOP,
    TOOL_UPDATE_TARGET,
    TOOL_EXIT_POSITION,
    TOOL_WAIT,
)
from .context import ContextBuilder
from .validation import ToolValidator, validate_tool_call
from .prompts import get_system_prompt, get_system_prompt_version


__all__ = [
    # Main class
    "TradingBrain",
    # Types
    "ToolCall",
    "Decision",
    "LLMResponse",
    "ValidationResult",
    "DecisionLog",
    # Tools
    "TOOLS",
    "TOOL_ENTER_LONG",
    "TOOL_ENTER_SHORT",
    "TOOL_UPDATE_STOP",
    "TOOL_UPDATE_TARGET",
    "TOOL_EXIT_POSITION",
    "TOOL_WAIT",
    # Components
    "ContextBuilder",
    "ToolValidator",
    "validate_tool_call",
    # Prompts
    "get_system_prompt",
    "get_system_prompt_version",
]
