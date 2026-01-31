#!/usr/bin/env python3
"""
FSDTrader Brain Module: Tool Definitions

Complete JSON schema definitions for all trading tools.
These are passed to the LLM for native function calling.
"""
from typing import List, Dict, Any


# Tool names as constants
TOOL_ENTER_LONG = "enter_long"
TOOL_ENTER_SHORT = "enter_short"
TOOL_UPDATE_STOP = "update_stop"
TOOL_UPDATE_TARGET = "update_target"
TOOL_EXIT_POSITION = "exit_position"
TOOL_WAIT = "wait"

# All valid tool names
VALID_TOOLS = [
    TOOL_ENTER_LONG,
    TOOL_ENTER_SHORT,
    TOOL_UPDATE_STOP,
    TOOL_UPDATE_TARGET,
    TOOL_EXIT_POSITION,
    TOOL_WAIT,
]

# Entry tools
ENTRY_TOOLS = [TOOL_ENTER_LONG, TOOL_ENTER_SHORT]

# Modification tools
MODIFICATION_TOOLS = [TOOL_UPDATE_STOP, TOOL_UPDATE_TARGET]

# Memo field definition (shared across all tools)
MEMO_FIELD = {
    "type": "string",
    "description": """Your self-notes for the next cycle. Format:
@DELTA|[time]|[position]|
same:[unchanged items]
changed:[what's new]
cumulative:[waits:N]
thesis:"[one sentence market read]"
watching:[specific triggers]
invalidates:[what flips thesis]
decision:[action(reason)]"""
}


TRADING_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "enter_long",
            "description": """Enter a LONG position with bracket orders (stop loss + profit target).

USE WHEN:
- Session trend is BULLISH (CVD rising, price above VWAP)
- L2 imbalance shows buyers in control (>1.5)
- Tape shows aggressive buying
- Velocity is at least MEDIUM
- Signal has persisted for 2+ snapshots
- Clear support level for stop placement

DO NOT USE WHEN:
- Already in a position
- Counter-trend without overwhelming evidence
- Spread exceeds session limit
- Low velocity / thin tape
- First signal without confirmation""",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit_price": {
                        "type": "number",
                        "description": "Entry limit price. Should be at or near current ask for immediate fill, or slightly below for better entry."
                    },
                    "stop_loss": {
                        "type": "number",
                        "description": "Stop loss price. MUST be below limit_price. Should be 10-30 cents below entry, placed at a logical support level."
                    },
                    "profit_target": {
                        "type": "number",
                        "description": "Take profit price. MUST be above limit_price. Typically 2x the stop distance for 1:2 risk/reward."
                    },
                    "size": {
                        "type": "integer",
                        "description": "Number of shares to trade. Default 100, max 100. Use smaller size for lower conviction.",
                        "default": 100,
                        "minimum": 1,
                        "maximum": 100
                    },
                    "conviction": {
                        "type": "string",
                        "enum": ["HIGH", "MEDIUM", "LOW"],
                        "description": "Your conviction level. HIGH = full confluence, MEDIUM = good setup with minor concerns, LOW = something's there but not clean."
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of why you're entering. Include: what signals you see, what confirms your thesis, what the stop is based on."
                    },
                    "memo": MEMO_FIELD
                },
                "required": ["limit_price", "stop_loss", "profit_target", "conviction", "reasoning", "memo"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "enter_short",
            "description": """Enter a SHORT position with bracket orders (stop loss + profit target).

USE WHEN:
- Session trend is BEARISH (CVD falling, price below VWAP)
- L2 imbalance shows sellers in control (<0.6)
- Tape shows aggressive selling
- Velocity is at least MEDIUM
- Signal has persisted for 2+ snapshots
- Clear resistance level for stop placement

DO NOT USE WHEN:
- Already in a position
- Counter-trend without overwhelming evidence
- Spread exceeds session limit
- Low velocity / thin tape
- First signal without confirmation""",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit_price": {
                        "type": "number",
                        "description": "Entry limit price. Should be at or near current bid for immediate fill, or slightly above for better entry."
                    },
                    "stop_loss": {
                        "type": "number",
                        "description": "Stop loss price. MUST be above limit_price. Should be 10-30 cents above entry, placed at a logical resistance level."
                    },
                    "profit_target": {
                        "type": "number",
                        "description": "Take profit price. MUST be below limit_price. Typically 2x the stop distance for 1:2 risk/reward."
                    },
                    "size": {
                        "type": "integer",
                        "description": "Number of shares to trade. Default 100, max 100.",
                        "default": 100,
                        "minimum": 1,
                        "maximum": 100
                    },
                    "conviction": {
                        "type": "string",
                        "enum": ["HIGH", "MEDIUM", "LOW"],
                        "description": "Your conviction level."
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of why you're entering."
                    },
                    "memo": MEMO_FIELD
                },
                "required": ["limit_price", "stop_loss", "profit_target", "conviction", "reasoning", "memo"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_stop",
            "description": """Move stop loss to a new price.

USE WHEN:
- Trade is in profit and you want to trail the stop to lock in gains
- New support/resistance level has formed
- Conditions have changed and you want to reduce risk
- Absorption detected at a level that provides new support/resistance

TRAILING STOP LOGIC:
- For LONG: Move stop UP to lock in profit
- For SHORT: Move stop DOWN to lock in profit
- Never move stop further away from entry (increases risk)""",
            "parameters": {
                "type": "object",
                "properties": {
                    "new_price": {
                        "type": "number",
                        "description": "New stop loss price."
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Why are you moving the stop? What level or condition justifies this?"
                    },
                    "memo": MEMO_FIELD
                },
                "required": ["new_price", "reasoning", "memo"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_target",
            "description": """Move profit target to a new price.

USE WHEN:
- A DOM wall has appeared that may block price
- New resistance/support level has formed
- Momentum is fading and you want to take profit sooner
- Conditions are stronger than expected and you want to extend target

TARGET ADJUSTMENT LOGIC:
- For LONG: Target should remain above current price
- For SHORT: Target should remain below current price""",
            "parameters": {
                "type": "object",
                "properties": {
                    "new_price": {
                        "type": "number",
                        "description": "New target price."
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Why are you adjusting the target?"
                    },
                    "memo": MEMO_FIELD
                },
                "required": ["new_price", "reasoning", "memo"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "exit_position",
            "description": """Exit current position immediately at market price.

USE WHEN:
- Momentum has reversed (CVD trend changed, L2 imbalance flipped)
- Your thesis is invalidated
- Conditions have deteriorated significantly
- You see absorption at your target level indicating the move is over

This sends a MARKET order to close the position immediately.
Stop and target orders will be cancelled automatically.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Why are you exiting? What changed?"
                    },
                    "memo": MEMO_FIELD
                },
                "required": ["reasoning", "memo"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wait",
            "description": """Take no action this cycle.

USE WHEN:
- No clear setup present
- Conditions are unclear or mixed
- Spread is too wide
- Velocity is too low for reliable signals
- First signal without confirmation (wait for persistence)
- Counter-trend without overwhelming evidence
- You're uncertain - when in doubt, wait

ALWAYS provide reasoning that includes:
- What you're seeing in the market
- What would change your mind
- What you're watching for""",
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Why are you waiting? What would make you act? Be specific about what you're watching for."
                    },
                    "memo": MEMO_FIELD
                },
                "required": ["reasoning", "memo"]
            }
        }
    }
]


def get_tools() -> List[Dict[str, Any]]:
    """Return the complete list of trading tools."""
    return TRADING_TOOLS


def get_tool_names() -> List[str]:
    """Return list of valid tool names."""
    return VALID_TOOLS


def is_valid_tool(tool_name: str) -> bool:
    """Check if a tool name is valid."""
    return tool_name in VALID_TOOLS


def is_entry_tool(tool_name: str) -> bool:
    """Check if tool is an entry action."""
    return tool_name in ENTRY_TOOLS


def is_modification_tool(tool_name: str) -> bool:
    """Check if tool modifies existing orders."""
    return tool_name in MODIFICATION_TOOLS


def requires_position(tool_name: str) -> bool:
    """Check if tool requires an active position."""
    return tool_name in [TOOL_UPDATE_STOP, TOOL_UPDATE_TARGET, TOOL_EXIT_POSITION]


def requires_flat(tool_name: str) -> bool:
    """Check if tool requires being flat (no position)."""
    return tool_name in ENTRY_TOOLS


# Alias for convenience
TOOLS = TRADING_TOOLS
