# FSDTrader: Technical Requirements Specification

> **Version**: 2.0.0
> **Last Updated**: 2026-01-24
> **Status**: Draft

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Trading Strategy](#3-trading-strategy)
4. [Brain Module (LLM)](#4-brain-module-llm)
5. [Market Data Module](#5-market-data-module)
6. [Execution Module](#6-execution-module)
7. [Main Loop](#7-main-loop)
8. [Configuration & Settings](#8-configuration--settings)
9. [Risk Management](#9-risk-management)
10. [Logging & Reporting](#10-logging--reporting)
11. [Operating Modes](#11-operating-modes)
12. [Dependencies & Infrastructure](#12-dependencies--infrastructure)
13. [Security Requirements](#13-security-requirements)
14. [Testing Requirements](#14-testing-requirements)
15. [Future Enhancements](#15-future-enhancements)

---

## 1. Project Overview

### 1.1 Purpose

FSDTrader is an autonomous momentum trading agent that uses Large Language Models (LLMs) with native tool-calling capabilities to make real-time trading decisions based on order flow analysis.

### 1.2 Philosophy

- **End-to-End Architecture**: Inspired by Tesla FSD - raw market data goes directly to an AI "brain" which outputs trading actions
- **Discretionary, Not Rule-Based**: The LLM acts as an experienced trader making judgment calls, not a rule engine checking boolean conditions
- **Trade Momentum, Not Noise**: Target quick $0.50-$2.00 moves over 1-5 minutes with full confluence
- **Trade What You See**: Make decisions based on current order flow, not predictions

### 1.3 Key Metrics

| Metric | Target |
|--------|--------|
| Decision Frequency | Every 5 seconds (0.2 Hz) |
| Maximum Latency | 2 seconds per decision |
| Target Win Rate | 55-65% |
| Risk/Reward | 1:2 minimum |
| Max Daily Loss | -$500 |
| Max Position Size | 100 shares |

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              FSDTrader                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐     │
│  │   DATA SOURCES  │    │      BRAIN      │    │    EXECUTION    │     │
│  │                 │    │                 │    │                 │     │
│  │  • IBKR (Live)  │───>│  • LLM Provider │───>│  • Order Mgmt   │     │
│  │  • MBO Replay   │    │  • Tool Calling │    │  • IBKR API     │     │
│  │  • Mock Data    │    │  • Validation   │    │  • Risk Checks  │     │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘     │
│           │                      │                      │              │
│           │                      │                      │              │
│           ▼                      ▼                      ▼              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                        MARKET DATA MODULE                        │   │
│  │                                                                  │   │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐       │   │
│  │  │ OrderBook │ │TapeStream │ │ Footprint │ │    CVD    │       │   │
│  │  │   (DOM)   │ │   (T&S)   │ │  Tracker  │ │  Tracker  │       │   │
│  │  └───────────┘ └───────────┘ └───────────┘ └───────────┘       │   │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐                     │   │
│  │  │  Volume   │ │Absorption │ │  Market   │                     │   │
│  │  │  Profile  │ │ Detector  │ │  Metrics  │                     │   │
│  │  └───────────┘ └───────────┘ └───────────┘                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Responsibilities

| Component | Responsibility |
|-----------|----------------|
| Main Loop | Orchestrates 5-second decision cycle, manages component lifecycle |
| Market Data Module | Aggregates real-time data into comprehensive state vector |
| Brain Module | Analyzes state, calls LLM with tools, returns structured action |
| Execution Module | Validates and executes orders via IBKR, manages positions |
| Reporting Module | Logs decisions, generates backtest reports |

### 2.3 File Structure

```
FSDTrader/
├── docs/
│   ├── REQUIREMENTS.md          # This document
│   └── PROMPTING_GUIDE.md       # LLM prompting best practices
├── src/
│   ├── main.py                  # Entry point, main loop
│   ├── market_data.py           # Market data analyzers
│   ├── execution.py             # Order management
│   ├── reporting.py             # Logging and reports
│   ├── data_replay.py           # MBO data replay for backtesting
│   └── brain/
│       ├── __init__.py
│       ├── brain.py             # Main TradingBrain class
│       ├── tools.py             # Tool definitions (JSON schemas)
│       ├── prompts.py           # System prompt
│       ├── context.py           # Context builder
│       └── providers/
│           ├── __init__.py
│           ├── base.py          # Abstract LLM provider
│           ├── grok.py          # Grok (xAI) implementation
│           ├── openai.py        # OpenAI implementation
│           └── anthropic.py     # Anthropic Claude implementation
├── config/
│   ├── settings.py              # Configuration management
│   └── .env.example             # Environment variable template
├── data/
│   ├── logs/                    # Daily agent logs
│   └── reports/                 # Backtest reports
├── BacktestData/                # MBO data files
│   └── TSLA-L3DATA/
├── tests/
│   ├── test_brain.py
│   ├── test_execution.py
│   └── test_market_data.py
├── requirements.txt
└── README.md
```

---

## 3. Trading Strategy

### 3.1 Core Philosophy

The system trades **momentum** using order flow analysis. Entry requires **full confluence** of multiple technical signals, with the LLM acting as a discretionary trader who weighs the evidence rather than checking rigid boolean conditions.

### 3.2 What We Look For

| Signal | Description | Interpretation |
|--------|-------------|----------------|
| Aggressive Tape | Buyers/sellers hitting the market, not resting | Urgency and conviction |
| Absorption | Large orders eating flow at key levels | Support/resistance confirmation |
| Wall Breaks | DOM walls being consumed | Breakout confirmation |
| Velocity | High tape speed | Active participation |
| Delta Confirmation | CVD trending in trade direction | Trend alignment |

### 3.3 What We Avoid

| Pattern | Description | Risk |
|---------|-------------|------|
| Chop | Two-sided tape, no clear direction | Whipsaw losses |
| Extended Moves | Price already moved significantly | Chasing |
| Counter-Trend | Trading against session trend without overwhelming evidence | Low probability |
| Low Velocity | Thin tape, slow prints | Unreliable signals |
| First Signal | Entering on first sign without confirmation | Traps |

### 3.4 Entry Framework

**LONG Entry Conditions** (Discretionary - LLM weighs these):
1. Session trend alignment (CVD rising, price above VWAP)
2. L2 Imbalance favoring bids (>1.5)
3. Tape showing aggressive buying
4. Velocity at least MEDIUM
5. No major resistance wall within 0.2%
6. Signal persistence (2+ snapshots)
7. Clear stop level defined

**SHORT Entry Conditions** (Mirror of LONG):
1. Session trend alignment (CVD falling, price below VWAP)
2. L2 Imbalance favoring asks (<0.6)
3. Tape showing aggressive selling
4. Velocity at least MEDIUM
5. No major support wall within 0.2%
6. Signal persistence
7. Clear stop level defined

### 3.5 Counter-Trend Rules

When trading against the session trend:
- Require HIGH conviction with multiple confirmations
- Reduce position size (50% of normal)
- Use tighter stops (max 20 cents)
- Expect lower win rate

### 3.6 Exit Rules

| Trigger | Action |
|---------|--------|
| L2 Imbalance reverses past 0.7/1.3 | Exit immediately |
| CVD Trend reverses | Exit |
| Price hits stop loss | Auto-exit via bracket |
| Price hits profit target | Auto-exit via bracket |
| Absorption detected at target level | Tighten stop |
| DOM wall appears in trade direction | Update target below/above wall |

---

## 4. Brain Module (LLM)

### 4.1 Overview

The Brain module is the decision-making core of FSDTrader. It uses **native LLM tool calling** (not text parsing) to generate structured trading actions. The LLM receives comprehensive market context and must call exactly one tool per decision cycle.

### 4.2 Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              BRAIN MODULE                                     │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         TRADING BRAIN                                   │ │
│  │  Main orchestrator class that:                                         │ │
│  │  • Receives raw market state from Main Loop                            │ │
│  │  • Builds rich context via Context Builder                             │ │
│  │  • Calls LLM via Provider abstraction                                  │ │
│  │  • Parses and validates tool call response                             │ │
│  │  • Returns structured action to Executor                               │ │
│  │  • Manages decision history                                            │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                              │                                               │
│          ┌───────────────────┼───────────────────┐                          │
│          ▼                   ▼                   ▼                          │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                    │
│  │   CONTEXT    │   │     LLM      │   │    TOOL      │                    │
│  │   BUILDER    │   │   PROVIDER   │   │  REGISTRY    │                    │
│  │              │   │              │   │              │                    │
│  │ Transforms   │   │ Abstraction  │   │ Defines 6    │                    │
│  │ raw state    │   │ for multiple │   │ action tools │                    │
│  │ into rich    │   │ LLM backends │   │ with schemas │                    │
│  │ context      │   │              │   │              │                    │
│  └──────────────┘   └──────────────┘   └──────────────┘                    │
│                              │                                               │
│                              ▼                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         VALIDATION LAYER                                │ │
│  │  Pre-execution checks before returning to Executor:                    │ │
│  │  • Tool call parsed successfully                                       │ │
│  │  • Required parameters present                                         │ │
│  │  • Price values are reasonable (within 1% of current price)           │ │
│  │  • Stop distance within limits (10-30 cents)                          │ │
│  │  • Position state compatible with action                               │ │
│  │  • Conviction level appropriate                                        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 File Structure

```
src/brain/
├── __init__.py              # Exports TradingBrain
├── brain.py                 # Main TradingBrain class
├── tools.py                 # Tool definitions and schemas
├── prompts.py               # System prompt template
├── context.py               # Context builder
├── validation.py            # Pre-execution validation
├── types.py                 # Data classes (ToolCall, Decision, etc.)
└── providers/
    ├── __init__.py          # Provider factory
    ├── base.py              # Abstract LLMProvider interface
    ├── grok.py              # Grok (xAI) implementation
    ├── openai_provider.py   # OpenAI implementation
    └── anthropic_provider.py # Anthropic Claude implementation
```

### 4.4 Core Classes

#### 4.4.1 TradingBrain

The main orchestrator class.

```python
class TradingBrain:
    """
    The decision-making core of FSDTrader.

    Responsibilities:
    - Build context from market state
    - Call LLM with tools
    - Parse and validate tool response
    - Manage decision history
    - Return structured action
    """

    def __init__(
        self,
        provider: str = "grok",
        model: str = None,
        api_key: str = None,
        temperature: float = 0.1,
        max_tokens: int = 1000
    ):
        """
        Initialize the trading brain.

        Args:
            provider: LLM provider ("grok", "openai", "anthropic", "ollama")
            model: Model name (defaults to provider's default)
            api_key: API key (falls back to environment variable)
            temperature: Response temperature (0.0-1.0)
            max_tokens: Maximum response tokens
        """
        pass

    def think(self, state: dict) -> ToolCall:
        """
        Analyze market state and return a trading action.

        This is the main entry point called by the Main Loop.

        Args:
            state: Complete market state dict with MARKET_STATE and ACCOUNT_STATE

        Returns:
            ToolCall with action name, arguments, and metadata

        Flow:
            1. Build context from state
            2. Call LLM with system prompt, context, and tools
            3. Parse tool call from response
            4. Validate tool call
            5. Update history
            6. Return ToolCall
        """
        pass

    def get_history(self) -> List[Decision]:
        """Return recent decision history."""
        pass

    def clear_history(self):
        """Clear decision history (on session reset or trade completion)."""
        pass

    def get_stats(self) -> dict:
        """Return brain statistics for monitoring."""
        pass
```

#### 4.4.2 ToolCall

Structured representation of an LLM tool call.

```python
@dataclass
class ToolCall:
    """Represents a parsed tool call from the LLM."""

    tool: str                    # Tool name: "enter_long", "wait", etc.
    arguments: dict              # Tool arguments
    reasoning: str               # Extracted reasoning
    conviction: Optional[str]    # HIGH, MEDIUM, LOW, or None
    raw_response: str            # Full LLM response for logging
    latency_ms: float            # LLM call latency

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
```

#### 4.4.3 Decision

Historical decision record.

```python
@dataclass
class Decision:
    """Historical decision record for context."""

    timestamp: float             # Unix timestamp
    tool: str                    # Tool called
    arguments: dict              # Arguments used
    reasoning: str               # LLM's reasoning
    conviction: Optional[str]    # Conviction level
    market_snapshot: dict        # Key market data at decision time
    result: Optional[dict]       # Execution result (filled in later)
```

### 4.5 Action Tools

The LLM has access to exactly **6 action tools**. Each tool uses native JSON schema for parameter validation.

#### 4.5.1 Complete Tool Definitions

```python
TRADING_TOOLS = [
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
                    }
                },
                "required": ["limit_price", "stop_loss", "profit_target", "conviction", "reasoning"]
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
                    }
                },
                "required": ["limit_price", "stop_loss", "profit_target", "conviction", "reasoning"]
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
                    }
                },
                "required": ["new_price", "reasoning"]
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
                    }
                },
                "required": ["new_price", "reasoning"]
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
                    }
                },
                "required": ["reasoning"]
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
                    }
                },
                "required": ["reasoning"]
            }
        }
    }
]
```

#### 4.5.2 Tool Validation Rules

| Tool | Validation Rule | Error Code |
|------|-----------------|------------|
| `enter_long` | Position must be FLAT | `ALREADY_IN_POSITION` |
| `enter_long` | stop_loss < limit_price | `INVALID_STOP_DIRECTION` |
| `enter_long` | profit_target > limit_price | `INVALID_TARGET_DIRECTION` |
| `enter_long` | Stop distance 10-30 cents | `STOP_DISTANCE_OUT_OF_RANGE` |
| `enter_long` | limit_price within 1% of current | `PRICE_TOO_FAR_FROM_MARKET` |
| `enter_short` | Position must be FLAT | `ALREADY_IN_POSITION` |
| `enter_short` | stop_loss > limit_price | `INVALID_STOP_DIRECTION` |
| `enter_short` | profit_target < limit_price | `INVALID_TARGET_DIRECTION` |
| `enter_short` | Stop distance 10-30 cents | `STOP_DISTANCE_OUT_OF_RANGE` |
| `update_stop` | Must have active position | `NO_POSITION` |
| `update_target` | Must have active position | `NO_POSITION` |
| `exit_position` | Must have active position | `NO_POSITION` |
| `wait` | Always valid | - |
| All | Required fields present | `MISSING_REQUIRED_FIELD` |
| All | Reasoning provided | `MISSING_REASONING` |

### 4.6 System Prompt

The system prompt defines the LLM's persona, trading style, and decision framework.

#### 4.6.1 Complete System Prompt Template

```python
SYSTEM_PROMPT = """You are FSDTrader, an experienced TSLA momentum trader with 10+ years of order flow trading experience. You read the tape like a book. You've seen every pattern, every trap, every squeeze.

## YOUR TRADING STYLE

You hunt for momentum trades - quick $0.50 to $2.00 moves over 1-5 minutes.
You trade what you SEE, not what you think should happen.

What you look for:
- Aggressive buyers/sellers HITTING the market (tape speed & direction)
- Absorption: big orders getting eaten at key levels
- Walls breaking after being tested
- Velocity and urgency - the tape should FEEL like something's happening
- Confirmation: signals that persist, not just blips

What you avoid:
- Chop and indecision (two-sided tape)
- Chasing moves that already happened
- Fighting strong walls without absorption
- Low velocity conditions (unreliable signals)
- First-signal entries (wait for persistence)
- Counter-trend trades without overwhelming evidence

## HOW YOU THINK

You read CONTEXT, not just numbers:
- "The tape feels heavy" vs "Buyers are stepping up"
- "This wall is holding strong" vs "They're absorbing into it"
- "Delta is positive but exhausted" vs "Momentum is building"

You think in probabilities:
- HIGH conviction: Everything aligns - trend, velocity, confirmation, clean levels
- MEDIUM conviction: Good setup with minor concerns
- LOW conviction: Something's there but not clean - usually means WAIT

You always ask: "What is this tape TELLING me?"

## TREND AWARENESS (Critical)

Before ANY entry, answer:
1. What is the SESSION trend? (CVD direction, price vs VWAP)
2. Am I trading WITH or AGAINST it?

COUNTER-TREND RULES:
If session CVD is falling AND you want to go LONG:
- This is a counter-trend trade - automatic caution
- Require HIGH conviction with multiple confirmations
- Use smaller size (50 shares instead of 100)
- Use tighter stops (20 cents max)
- Expect lower win rate

THE DEAD CAT BOUNCE TRAP:
A few aggressive buy prints in a downtrend is NOT a reversal.
It's usually trapped longs or shorts covering. Don't be the trapped buyer.
Wait for actual trend change: CVD turning, price reclaiming VWAP, etc.

## CONFIRMATION REQUIREMENTS

The FIRST signal is usually a trap. Wait for:
1. PERSISTENCE: Signal continues for 2+ snapshots (10+ seconds)
2. VELOCITY: Tape speed picks up, not just a blip
3. FOLLOW-THROUGH: Price moves in signal direction

EXCEPTION: Genuine absorption at a key level (wall getting eaten while price holds) - this alone can be high conviction. Absorption is the strongest signal in order flow.

## HARD LIMITS (Non-Negotiable)

SPREAD LIMITS:
- OPEN_DRIVE session (first 30 min): Max $0.15
- All other sessions: Max $0.08
- If spread exceeds limit: WAIT, no exceptions

STOP DISTANCE:
- Minimum: $0.10 (10 cents)
- Maximum: $0.30 (30 cents)
- Must be placed at logical level (support/resistance, wall, etc.)
- No logical stop level = no entry

RISK/REWARD:
- Minimum 1:2 risk/reward (e.g., 20 cent stop, 40 cent target)
- Target should be at logical level (resistance, wall, etc.)

CHASING:
- If price already moved significantly without you: WAIT for pullback
- If DISTANCE_TO_HOD_PCT > 0.5% and you're trying to go long: Be cautious

POSITION MANAGEMENT:
- One position at a time
- If already in a trade: manage it, don't enter new trades

UNCERTAINTY:
- If you're unsure: WAIT
- There's always another trade
- Missing a trade costs nothing
- Bad entry costs money

## TOOL USAGE

You MUST call exactly ONE tool for each decision. Available tools:

1. enter_long - Enter LONG with bracket orders
2. enter_short - Enter SHORT with bracket orders
3. update_stop - Move stop loss
4. update_target - Move profit target
5. exit_position - Exit immediately at market
6. wait - No action (with reasoning)

Every tool requires a 'reasoning' field. Be specific about:
- What you see in the market
- Why you're taking this action
- What would change your mind

## DECISION FLOW

For each market snapshot:

1. CHECK POSITION STATUS
   - If FLAT: Consider entries
   - If IN TRADE: Consider management (stop/target updates, exit)

2. ASSESS SESSION CONTEXT
   - What's the trend? (CVD direction + price vs VWAP)
   - Am I with or against it?
   - What session are we in? (affects spread limits)

3. READ THE TAPE
   - What's velocity telling me?
   - What's the sentiment?
   - Any absorption or walls?

4. CHECK FOR CONFIRMATION
   - Has signal persisted?
   - Is there follow-through?

5. MAKE DECISION
   - Full confluence → Enter with appropriate conviction
   - Partial signals → WAIT for more confirmation
   - Counter-trend → Extra caution, smaller size
   - Unclear → WAIT

Remember: It's better to miss a trade than to take a bad one."""
```

#### 4.6.2 System Prompt Sections

| Section | Purpose | Key Points |
|---------|---------|------------|
| Identity | Establish persona | 10+ years experience, reads tape like a book |
| Trading Style | Define what we trade | Momentum, $0.50-$2.00 moves, 1-5 minutes |
| How You Think | Reasoning framework | Context over numbers, probabilities, "what is tape telling me" |
| Trend Awareness | Counter-trend protection | Always check session trend, counter-trend = danger zone |
| Confirmation | Prevent first-signal traps | Persistence, velocity, follow-through |
| Hard Limits | Non-negotiable rules | Spread, stop distance, risk/reward, position limits |
| Tool Usage | How to respond | One tool per decision, reasoning required |
| Decision Flow | Step-by-step process | Position → Context → Tape → Confirmation → Decision |

### 4.7 Context Builder

The Context Builder transforms raw market state into a rich, human-readable format for the LLM.

#### 4.7.1 Context Structure

```python
def build_context(
    market_state: dict,
    account_state: dict,
    history: List[Decision]
) -> str:
    """
    Build comprehensive context for LLM.

    Returns a formatted string with all sections.
    """
    pass
```

#### 4.7.2 Context Sections

**Section 1: Position Status**

```markdown
## CURRENT POSITION

[If FLAT:]
FLAT - Looking for entry opportunity.

[If IN POSITION:]
LONG 100 shares @ $245.43
├── Current Price: $245.80
├── Unrealized P&L: +$0.37 (+$37.00)
├── Stop: $245.13 | Target: $246.43
├── Time in Trade: 2m 15s
└── Status: In profit, approaching target
```

**Section 2: Session Context**

```markdown
## SESSION CONTEXT

├── Time: 10:15:23 AM ET
├── Session: OPEN_DRIVE (first 30 min)
├── Spread Limit: $0.15
│
├── Session Trend: BULLISH
│   ├── CVD: +125,000 (RISING)
│   ├── Price vs VWAP: ABOVE ($245.50 vs $245.10)
│   └── Interpretation: Buyers in control
│
├── RVOL: 2.8x (High relative volume)
└── Current Spread: $0.02 ✓ (within limit)

[If counter-trend warning needed:]
⚠️ COUNTER-TREND WARNING
Going LONG here would fight the session trend (CVD falling).
Require HIGH conviction with multiple confirmations.
```

**Section 3: Key Levels**

```markdown
## KEY LEVELS

├── HOD: $245.80 (0.12% away) ← TESTING
├── LOD: $241.20 (1.75% away)
├── VWAP: $245.10
│
└── Volume Profile:
    ├── POC: $244.80
    ├── VAH: $246.20
    └── VAL: $243.50
```

**Section 4: Order Book**

```markdown
## ORDER BOOK (Level 2)

├── L2 Imbalance: 2.1 (BULLISH - bids > asks)
├── Spread: $0.02
│
├── Bid Stack:
│   ├── $245.48 x 150
│   ├── $245.46 x 200
│   └── $245.44 x 180
│
├── Ask Stack:
│   ├── $245.50 x 120
│   ├── $245.52 x 180
│   └── $245.54 x 250
│
└── Walls:
    ├── ASK $247.00 x 3,500 (MAJOR) - 0.61% away
    └── BID $244.50 x 2,100 (MINOR) - 0.41% away
```

**Section 5: Tape Analysis**

```markdown
## TAPE ANALYSIS

├── Velocity: HIGH (42.5 trades/sec)
├── Sentiment: AGGRESSIVE_BUYING
│
├── Delta:
│   ├── 1-second: +1,250
│   └── 5-second: +4,800
│
└── Large Prints (last 60s):
    └── BUY 500 @ $245.48 (12s ago)
```

**Section 6: Delta & Flow**

```markdown
## DELTA & FLOW

├── Footprint (Current Bar):
│   ├── OHLC: $245.30 / $245.55 / $245.25 / $245.50
│   ├── Delta: +2,800 (+18.7%)
│   ├── Volume: 15,000
│   └── POC: $245.45
│
├── CVD Session: +125,000
├── CVD Trend: RISING
└── CVD Slope (5m): +0.85
```

**Section 7: Absorption**

```markdown
## ABSORPTION

├── Detected: No
└── (No absorption pattern currently detected)

[If detected:]
├── Detected: YES
├── Side: BID (buyers absorbing selling)
├── Price: $244.50
└── Interpretation: Strong support forming at $244.50
```

**Section 8: Recent Decisions**

```markdown
## YOUR RECENT DECISIONS

[10:14:45] wait
"Velocity is low, waiting for tape to pick up. Watching for L2
imbalance to strengthen."

[10:14:50] wait
"Good imbalance now (1.8) but CVD was flat. Now seeing CVD start
to rise. Building conviction, need one more confirmation."

[10:14:55] wait
"CVD rising, imbalance holding at 1.9. Waiting for velocity to
confirm. Want to see MEDIUM or HIGH velocity before entry."
```

**Section 9: Raw Data**

```markdown
## RAW MARKET DATA

```json
{
  "TICKER": "TSLA",
  "LAST": 245.50,
  "VWAP": 245.10,
  ...
}
```
```

#### 4.7.3 Context Token Budget

| Section | Approximate Tokens |
|---------|-------------------|
| Position Status | 50-100 |
| Session Context | 100-150 |
| Key Levels | 80-100 |
| Order Book | 150-200 |
| Tape Analysis | 100-150 |
| Delta & Flow | 100-150 |
| Absorption | 30-80 |
| Recent Decisions (3) | 200-400 |
| Raw Data | 300-500 |
| **Total Context** | **~1,100-1,800** |
| System Prompt | ~1,500 |
| **Total Input** | **~2,600-3,300** |
| Response | ~200-500 |
| **Total per Call** | **~3,000-4,000** |

### 4.8 LLM Provider Abstraction

#### 4.8.1 Provider Interface

```python
from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class LLMResponse:
    """Raw response from LLM provider."""
    tool_calls: List[dict]       # Parsed tool calls
    content: Optional[str]       # Text content (if any)
    model: str                   # Model used
    usage: dict                  # Token usage
    latency_ms: float            # Request latency

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1000
    ):
        """Initialize provider with credentials and settings."""
        pass

    @abstractmethod
    def call_with_tools(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: List[dict],
        tool_choice: str = "required"
    ) -> LLMResponse:
        """
        Call LLM with tools.

        Args:
            system_prompt: System message
            user_prompt: User message (context)
            tools: List of tool definitions
            tool_choice: "required" (must call tool), "auto", or specific tool name

        Returns:
            LLMResponse with parsed tool calls
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return full model identifier."""
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return provider name."""
        pass

    def health_check(self) -> bool:
        """Check if provider is accessible."""
        pass
```

#### 4.8.2 Grok Provider Implementation

```python
class GrokProvider(LLMProvider):
    """Grok (xAI) LLM provider implementation."""

    DEFAULT_MODEL = "grok-3-mini-fast"
    BASE_URL = "https://api.x.ai/v1"

    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        base_url: str = None,
        temperature: float = 0.1,
        max_tokens: int = 1000
    ):
        self.api_key = api_key or os.environ.get("XAI_API_KEY")
        self.model = model or self.DEFAULT_MODEL
        self.base_url = base_url or self.BASE_URL
        self.temperature = temperature
        self.max_tokens = max_tokens

        if not self.api_key:
            raise ValueError("XAI_API_KEY not provided")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def call_with_tools(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: List[dict],
        tool_choice: str = "required"
    ) -> LLMResponse:
        start_time = time.time()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            tools=tools,
            tool_choice=tool_choice,  # Force tool use
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        latency_ms = (time.time() - start_time) * 1000

        # Parse tool calls
        tool_calls = []
        if response.choices[0].message.tool_calls:
            for tc in response.choices[0].message.tool_calls:
                tool_calls.append({
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments)
                })

        return LLMResponse(
            tool_calls=tool_calls,
            content=response.choices[0].message.content,
            model=self.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            latency_ms=latency_ms
        )
```

#### 4.8.3 Provider Factory

```python
def create_provider(
    provider: str,
    api_key: str = None,
    model: str = None,
    **kwargs
) -> LLMProvider:
    """
    Factory function to create LLM provider.

    Args:
        provider: "grok", "openai", "anthropic", "ollama"
        api_key: API key (falls back to environment variable)
        model: Model name (falls back to provider default)
        **kwargs: Additional provider-specific options

    Returns:
        LLMProvider instance
    """
    providers = {
        "grok": GrokProvider,
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "ollama": OllamaProvider,
    }

    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}")

    return providers[provider](api_key=api_key, model=model, **kwargs)
```

#### 4.8.4 Provider Configuration

| Provider | Environment Variable | Default Model | Base URL |
|----------|---------------------|---------------|----------|
| Grok | `XAI_API_KEY` | `grok-3-mini-fast` | `https://api.x.ai/v1` |
| OpenAI | `OPENAI_API_KEY` | `gpt-4o` | `https://api.openai.com/v1` |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-sonnet-4-20250514` | `https://api.anthropic.com/v1` |
| Ollama | N/A | `llama3.1:8b` | `http://localhost:11434/v1` |

### 4.9 History Management

#### 4.9.1 History Storage

```python
class DecisionHistory:
    """Manages decision history for context continuity."""

    def __init__(self, max_size: int = 3):
        self.max_size = max_size
        self.decisions: deque = deque(maxlen=max_size)

    def add(self, decision: Decision):
        """Add a decision to history."""
        self.decisions.append(decision)

    def get_recent(self, n: int = None) -> List[Decision]:
        """Get last n decisions."""
        n = n or self.max_size
        return list(self.decisions)[-n:]

    def clear(self):
        """Clear all history."""
        self.decisions.clear()

    def format_for_context(self) -> str:
        """Format history for LLM context."""
        if not self.decisions:
            return "(No recent decisions)"

        lines = []
        for d in self.decisions:
            timestamp = datetime.fromtimestamp(d.timestamp).strftime("%H:%M:%S")
            lines.append(f"[{timestamp}] {d.tool}")
            lines.append(f'"{d.reasoning}"')
            lines.append("")

        return "\n".join(lines)
```

#### 4.9.2 History Reset Conditions

| Condition | Action |
|-----------|--------|
| New trading session (market open) | Clear all history |
| Trade completed (win or loss) | Clear all history |
| Major trend change (CVD flip) | Clear all history |
| Manual reset requested | Clear all history |

### 4.10 Error Handling

#### 4.10.1 Error Types

```python
class BrainError(Exception):
    """Base exception for Brain module."""
    pass

class ProviderError(BrainError):
    """LLM provider error."""
    pass

class ToolParsingError(BrainError):
    """Failed to parse tool call from response."""
    pass

class ValidationError(BrainError):
    """Tool call validation failed."""
    pass

class ContextBuildError(BrainError):
    """Failed to build context."""
    pass
```

#### 4.10.2 Fallback Behavior

| Error Type | Fallback Action |
|------------|-----------------|
| Provider timeout | Return `wait(reasoning="API timeout")` |
| Provider rate limit | Return `wait(reasoning="API rate limited")` |
| Tool parsing failed | Return `wait(reasoning="Parse error")` |
| Validation failed | Return `wait(reasoning="{validation error}")` |
| Context build failed | Log error, return `wait(reasoning="Context error")` |

### 4.11 Logging Requirements

#### 4.11.1 Decision Logging

Every decision must be logged with:

```python
@dataclass
class DecisionLog:
    timestamp: str
    market_snapshot: dict        # Key market values at decision time
    context_sent: str            # Full context string
    system_prompt_version: str   # System prompt version/hash
    provider: str                # LLM provider used
    model: str                   # Model used
    tool_call: dict              # Parsed tool call
    raw_response: str            # Full LLM response
    latency_ms: float            # LLM call latency
    validation_result: dict      # Validation pass/fail and errors
    execution_result: dict       # Result from Executor (filled later)
```

#### 4.11.2 Log Files

| File | Content | Retention |
|------|---------|-----------|
| `brain_decisions_{date}.jsonl` | All decisions as JSON lines | 30 days |
| `brain_decisions_{date}.md` | Human-readable decision log | 30 days |
| `brain_errors_{date}.log` | Error logs | 30 days |

### 4.12 Performance Requirements

| Metric | Requirement |
|--------|-------------|
| LLM call latency | < 2000ms (p95) |
| Context build time | < 50ms |
| Tool parsing time | < 10ms |
| Validation time | < 10ms |
| Total think() time | < 2500ms (p95) |
| Memory usage | < 500MB |

### 4.13 Testing Requirements

#### 4.13.1 Unit Tests

| Component | Test Cases |
|-----------|------------|
| Context Builder | All section formatters, edge cases (empty data, missing fields) |
| Tool Parser | Valid tool calls, invalid JSON, missing fields, wrong types |
| Validation | All validation rules, boundary conditions |
| History | Add, get, clear, format, max size |
| Provider | Mock responses, error handling, timeout |

#### 4.13.2 Integration Tests

| Test | Description |
|------|-------------|
| End-to-end decision | Full flow from state to tool call |
| Provider switching | Verify all providers work |
| Error recovery | Verify fallback behavior |

#### 4.13.3 Scenario Tests

| Scenario | Expected Tool | Conviction |
|----------|---------------|------------|
| Full bullish confluence | `enter_long` | HIGH |
| Full bearish confluence | `enter_short` | HIGH |
| Good setup, minor concerns | Entry | MEDIUM |
| Counter-trend attempt | `wait` or cautious entry | - |
| Spread too wide | `wait` | - |
| Low velocity | `wait` | - |
| First signal only | `wait` | - |
| Already in position, good conditions | Management or `wait` | - |
| Position in profit, wall appearing | `update_target` or `update_stop` | - |
| Position, momentum reversing | `exit_position` | - |

---

## 5. Market Data Module

### 5.1 Overview

The Market Data Module aggregates real-time data from multiple sources into a comprehensive state vector for the LLM.

### 5.2 Components

#### 5.2.1 OrderBook (Level 2 / DOM)

**Purpose**: Track order book state, detect walls, calculate imbalance.

**Fields Provided**:
| Field | Type | Description |
|-------|------|-------------|
| `L2_IMBALANCE` | float | Bid Volume / Ask Volume (top 5 levels) |
| `SPREAD` | float | Best Ask - Best Bid |
| `DOM_WALLS` | array | Orders >3x average size |
| `BID_STACK` | array | Top 3 bid levels [price, size] |
| `ASK_STACK` | array | Top 3 ask levels [price, size] |

**Wall Detection**:
- MINOR: Size > 3x average
- MAJOR: Size > 5x average

#### 5.2.2 TapeStream (Time & Sales)

**Purpose**: Analyze trade flow, detect aggression, calculate velocity.

**Fields Provided**:
| Field | Type | Description |
|-------|------|-------------|
| `TAPE_VELOCITY` | string | LOW (<10 tps), MEDIUM (10-30), HIGH (>30) |
| `TAPE_VELOCITY_TPS` | float | Exact trades per second |
| `TAPE_SENTIMENT` | string | AGGRESSIVE_BUYING, AGGRESSIVE_SELLING, NEUTRAL |
| `TAPE_DELTA_1S` | int | Net delta over 1 second |
| `TAPE_DELTA_5S` | int | Net delta over 5 seconds |
| `LARGE_PRINTS_1M` | array | Trades >300 shares in last 60 seconds |

**Side Detection**:
- Price >= Ask → BUY (buyer lifted)
- Price <= Bid → SELL (seller hit)
- Between → Compare to midpoint

#### 5.2.3 FootprintTracker

**Purpose**: Track per-bar delta, POC, and imbalances.

**Fields Provided**:
| Field | Type | Description |
|-------|------|-------------|
| `FOOTPRINT_CURR_BAR.open` | float | Bar open price |
| `FOOTPRINT_CURR_BAR.high` | float | Bar high price |
| `FOOTPRINT_CURR_BAR.low` | float | Bar low price |
| `FOOTPRINT_CURR_BAR.close` | float | Bar close price |
| `FOOTPRINT_CURR_BAR.delta` | int | Buy vol - Sell vol |
| `FOOTPRINT_CURR_BAR.delta_pct` | float | Delta / Volume * 100 |
| `FOOTPRINT_CURR_BAR.volume` | int | Total bar volume |
| `FOOTPRINT_CURR_BAR.poc` | float | Price with most volume |
| `FOOTPRINT_CURR_BAR.imbalances` | array | Price levels with >3:1 ratio |

**Bar Duration**: 60 seconds (1-minute bars)

#### 5.2.4 CumulativeDelta (CVD)

**Purpose**: Track session-level delta accumulation and trend.

**Fields Provided**:
| Field | Type | Description |
|-------|------|-------------|
| `CVD_SESSION` | int | Cumulative delta since session open |
| `CVD_TREND` | string | RISING, FALLING, FLAT |
| `CVD_SLOPE_5M` | float | Normalized slope (-1 to +1) |

**Trend Calculation**:
- Linear regression over last 30 samples (30 seconds)
- Normalized slope > 0.3 → RISING
- Normalized slope < -0.3 → FALLING
- Otherwise → FLAT

#### 5.2.5 VolumeProfile

**Purpose**: Track session volume distribution.

**Fields Provided**:
| Field | Type | Description |
|-------|------|-------------|
| `VP_POC` | float | Point of Control (most volume price) |
| `VP_VAH` | float | Value Area High (70% volume top) |
| `VP_VAL` | float | Value Area Low (70% volume bottom) |
| `VP_DEVELOPING_POC` | float | Current session POC (moves) |
| `PRICE_VS_POC` | string | ABOVE, BELOW, AT_POC |

#### 5.2.6 AbsorptionDetector

**Purpose**: Detect absorption patterns (large orders absorbing flow without price movement).

**Fields Provided**:
| Field | Type | Description |
|-------|------|-------------|
| `ABSORPTION_DETECTED` | bool | True if pattern detected |
| `ABSORPTION_SIDE` | string | BID (support) or ASK (resistance) |
| `ABSORPTION_PRICE` | float | Price level of absorption |

**Detection Logic**:
- High volume (>500 shares in 20 trades)
- Low price movement (<$0.05 range)
- One side significantly heavier (>1.5x)

#### 5.2.7 MarketMetrics

**Purpose**: Track session-level metrics.

**Fields Provided**:
| Field | Type | Description |
|-------|------|-------------|
| `HOD` | float | High of Day |
| `LOD` | float | Low of Day |
| `HOD_LOD_LOC` | string | TESTING_HOD, TESTING_LOD, NEAR_HOD, NEAR_LOD, MID_RANGE |
| `DISTANCE_TO_HOD_PCT` | float | Percentage distance to HOD |
| `RVOL_DAY` | float | Relative volume vs 30-day average |
| `TIME_SESSION` | string | OPEN_DRIVE, OPEN_RANGE, MORNING, MIDDAY, CLOSE |
| `VWAP` | float | Volume-weighted average price |

**Session Phases**:
| Phase | Time (from open) |
|-------|------------------|
| OPEN_DRIVE | 0-30 min |
| OPEN_RANGE | 30-60 min |
| MORNING | 60-180 min |
| MIDDAY | 180-300 min |
| CLOSE | 300+ min |

### 5.3 State Vector

The complete state vector sent to the Brain:

```json
{
  "MARKET_STATE": {
    "TICKER": "TSLA",
    "LAST": 245.50,
    "VWAP": 245.10,
    "TIME_SESSION": "OPEN_DRIVE",

    "L2_IMBALANCE": 2.1,
    "SPREAD": 0.02,
    "DOM_WALLS": [...],
    "BID_STACK": [...],
    "ASK_STACK": [...],

    "TAPE_VELOCITY": "HIGH",
    "TAPE_VELOCITY_TPS": 42.5,
    "TAPE_SENTIMENT": "AGGRESSIVE_BUYING",
    "TAPE_DELTA_1S": 1250,
    "TAPE_DELTA_5S": 4800,
    "LARGE_PRINTS_1M": [...],

    "FOOTPRINT_CURR_BAR": {...},
    "CVD_SESSION": 125000,
    "CVD_TREND": "RISING",
    "CVD_SLOPE_5M": 0.85,

    "VP_POC": 244.80,
    "VP_VAH": 246.20,
    "VP_VAL": 243.50,
    "VP_DEVELOPING_POC": 245.30,
    "PRICE_VS_POC": "ABOVE",

    "HOD": 245.80,
    "LOD": 241.20,
    "HOD_LOD_LOC": "TESTING_HOD",
    "DISTANCE_TO_HOD_PCT": 0.12,
    "RVOL_DAY": 2.8,

    "ABSORPTION_DETECTED": false,
    "ABSORPTION_SIDE": null,
    "ABSORPTION_PRICE": null
  },
  "ACCOUNT_STATE": {
    "POSITION": 0,
    "POSITION_SIDE": "FLAT",
    "AVG_ENTRY": null,
    "UNREALIZED_PL": 0.0,
    "DAILY_PL": 150.00,
    "DAILY_TRADES": 3
  },
  "ACTIVE_ORDERS": []
}
```

---

## 6. Execution Module

### 6.1 Overview

The Execution Module handles order validation, submission, and position tracking via IBKR.

### 6.2 Order Types

#### 6.2.1 Bracket Order

Entry with attached stop loss and profit target:

```
Parent Order (Entry)
├── Take Profit Order (LMT)
└── Stop Loss Order (STP)
```

- Orders are linked via `parentId`
- When entry fills, stop and target become active
- When either stop or target fills, the other is cancelled

#### 6.2.2 Modification Orders

- Stop modification: Change `auxPrice` on existing stop order
- Target modification: Change `lmtPrice` on existing limit order

#### 6.2.3 Exit Order

Market order to close position immediately.

### 6.3 Position Tracking

```python
@dataclass
class Position:
    side: PositionSide  # FLAT, LONG, SHORT
    size: int
    avg_entry: float
    unrealized_pnl: float
    entry_time: float
```

### 6.4 Validation Rules

| Rule | Limit | Enforcement |
|------|-------|-------------|
| Max Position Size | 100 shares | Reject entry |
| Max Daily Loss | -$500 | Halt trading |
| Max Daily Trades | 10 | Reject entry |
| Max Spread | Session-dependent | Reject entry |
| Min Stop Distance | $0.10 | Reject entry |
| Max Stop Distance | $0.30 | Reject entry |
| Already in Position | 1 position max | Reject entry |

### 6.5 Spread Limits by Session

| Session | Max Spread |
|---------|------------|
| OPEN_DRIVE | $0.15 |
| All Others | $0.08 |

### 6.6 Execution Response

```python
{
    "success": bool,
    "action": str,           # Tool name executed
    "error": str | None,     # Error code if failed
    "details": {
        "entry_order_id": int,
        "stop_order_id": int,
        "target_order_id": int,
        ...
    }
}
```

### 6.7 Error Codes

| Code | Description |
|------|-------------|
| `ALREADY_IN_POSITION` | Cannot enter, already have position |
| `NO_POSITION` | Cannot modify, no active position |
| `STOP_TOO_TIGHT` | Stop distance < minimum |
| `STOP_TOO_WIDE` | Stop distance > maximum |
| `SPREAD_TOO_WIDE` | Current spread exceeds limit |
| `MAX_DAILY_LOSS_HIT` | Daily loss limit reached |
| `MAX_TRADES_HIT` | Daily trade limit reached |
| `INVALID_PRICE_FORMAT` | Could not parse price |
| `NO_CONNECTION` | IBKR not connected |

---

## 7. Main Loop

### 7.1 Loop Configuration

| Parameter | Value |
|-----------|-------|
| Decision Frequency | 0.2 Hz (every 5 seconds) |
| Loop Interval | 5000 ms |
| Max Latency Budget | 2000 ms |

### 7.2 Loop Flow

```python
while running:
    loop_start = time.time()

    # 1. Get market state
    state = connector.get_full_state(symbol)

    # 2. Check daily limits
    if daily_pnl <= max_daily_loss:
        break

    # 3. Inject account state
    state["ACCOUNT_STATE"] = executor.get_state()

    # 4. Call Brain (LLM with tools)
    action = brain.think(state)

    # 5. Execute action
    result = executor.execute(action)

    # 6. Log decision
    logger.log(state, action, result)

    # 7. Wait for next cycle
    elapsed = time.time() - loop_start
    await asyncio.sleep(max(0, LOOP_INTERVAL - elapsed))
```

### 7.3 Graceful Shutdown

On shutdown (SIGINT, SIGTERM):
1. Stop accepting new decisions
2. Cancel all pending orders
3. Close any open position at market
4. Generate session report
5. Log final P&L

---

## 8. Configuration & Settings

### 8.1 Environment Variables

```bash
# LLM Configuration
LLM_PROVIDER=grok                    # grok, openai, anthropic, ollama
LLM_API_KEY=xai-...                  # API key for provider
LLM_MODEL=grok-3-mini-fast           # Model to use
LLM_BASE_URL=https://api.x.ai/v1     # Optional base URL override
LLM_TEMPERATURE=0.1                  # Response temperature
LLM_MAX_TOKENS=1000                  # Max response tokens

# IBKR Configuration
IBKR_HOST=127.0.0.1
IBKR_PORT_PAPER=7497
IBKR_PORT_LIVE=7496
IBKR_CLIENT_ID=1

# Trading Configuration
TRADING_SYMBOL=TSLA
MAX_POSITION_SIZE=100
MAX_DAILY_LOSS=-500
MAX_DAILY_TRADES=10

# Logging
LOG_LEVEL=INFO
LOG_DIR=data/logs
```

### 8.2 Settings File

```python
# config/settings.py
from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class LLMSettings:
    provider: str = os.getenv("LLM_PROVIDER", "grok")
    api_key: str = os.getenv("LLM_API_KEY", "")
    model: str = os.getenv("LLM_MODEL", "grok-3-mini-fast")
    base_url: Optional[str] = os.getenv("LLM_BASE_URL")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1000"))

@dataclass
class TradingSettings:
    symbol: str = os.getenv("TRADING_SYMBOL", "TSLA")
    max_position_size: int = int(os.getenv("MAX_POSITION_SIZE", "100"))
    max_daily_loss: float = float(os.getenv("MAX_DAILY_LOSS", "-500"))
    max_daily_trades: int = int(os.getenv("MAX_DAILY_TRADES", "10"))
    decision_interval: float = 5.0  # seconds

@dataclass
class IBKRSettings:
    host: str = os.getenv("IBKR_HOST", "127.0.0.1")
    port_paper: int = int(os.getenv("IBKR_PORT_PAPER", "7497"))
    port_live: int = int(os.getenv("IBKR_PORT_LIVE", "7496"))
    client_id: int = int(os.getenv("IBKR_CLIENT_ID", "1"))
```

---

## 9. Risk Management

### 9.1 Pre-Trade Checks

| Check | Limit | Action on Fail |
|-------|-------|----------------|
| Daily P&L | > -$500 | Halt trading |
| Daily Trades | < 10 | Reject entry |
| Position Count | < 1 | Reject entry |
| Spread | < Session limit | Reject entry |
| Stop Distance | 10-30 cents | Reject entry |

### 9.2 Position-Level Risk

| Parameter | Limit |
|-----------|-------|
| Max Position Size | 100 shares |
| Max Risk per Trade | $30 (100 shares × $0.30 stop) |
| Typical Risk per Trade | $20 (100 shares × $0.20 stop) |

### 9.3 Session-Level Risk

| Parameter | Limit |
|-----------|-------|
| Max Daily Loss | -$500 |
| Max Daily Trades | 10 |
| Max Losing Streak | N/A (halt at daily loss) |

### 9.4 Conviction-Based Sizing

| Conviction | Size | Risk |
|------------|------|------|
| HIGH | 100 shares | $20-30 |
| MEDIUM | 100 shares | $20-30 |
| LOW | 50 shares (or reject) | $10-15 |

**Implementation**: LOW conviction entries should either be rejected or use reduced size.

---

## 10. Logging & Reporting

### 10.1 Log Levels

| Level | Usage |
|-------|-------|
| DEBUG | Detailed market data, LLM prompts/responses |
| INFO | Decisions, executions, P&L updates |
| WARNING | Risk limit warnings, parse failures |
| ERROR | Connection failures, execution errors |

### 10.2 Log Format

```
HH:MM:SS | COMPONENT | LEVEL | Message
```

Example:
```
10:15:03 | BRAIN      | INFO  | Tool call: enter_long(245.50, 245.20, 246.50)
10:15:03 | EXEC_TSLA  | INFO  | Bracket submitted: BUY 100 @ 245.50
10:15:04 | EXEC_TSLA  | INFO  | ENTRY FILLED @ 245.48
```

### 10.3 Decision Log File

Store full LLM interactions for review:

```
data/reports/brain_logs/grok_decisions_YYYYMMDD.md

## Decision at YYYY-MM-DD HH:MM:SS

### Context Sent
[Full context string]

### Tool Call
{
  "tool": "enter_long",
  "args": {...}
}

### Execution Result
{...}
```

### 10.4 Backtest Report

```
data/reports/YYYYMMDD_HHMMSS/
├── summary.json       # Trade statistics
├── equity_curve.json  # Equity progression
├── trades.json        # All trades with details
├── decisions.json     # All AI decisions
└── report.md          # Human-readable summary
```

### 10.5 Report Metrics

| Metric | Description |
|--------|-------------|
| Total Trades | Number of completed trades |
| Win Rate | Winners / Total |
| Profit Factor | Gross Profit / Gross Loss |
| Max Drawdown | Largest peak-to-trough decline |
| Sharpe Ratio | Risk-adjusted return |
| Average Win | Mean winning trade |
| Average Loss | Mean losing trade |
| Largest Win | Maximum single trade profit |
| Largest Loss | Maximum single trade loss |

---

## 11. Operating Modes

### 11.1 Mode Overview

| Mode | Data Source | Execution | Use Case |
|------|-------------|-----------|----------|
| `--live` | IBKR (7496) | Real orders | Live trading (DANGER) |
| `--paper` | IBKR (7497) | Paper orders | Paper trading |
| `--sim` | Mock random | Simulated | Development testing |
| `--backtest` | MBO replay | Simulated | Strategy validation |

### 11.2 Paper Trading Mode (Default)

```bash
python src/main.py --symbol TSLA
```

- Connects to IBKR paper trading port (7497)
- Uses real market data
- Submits paper orders
- Safe for testing

### 11.3 Live Trading Mode

```bash
python src/main.py --symbol TSLA --live
```

- Connects to IBKR live port (7496)
- **REAL MONEY AT RISK**
- Requires explicit flag
- All risk controls enforced

### 11.4 Simulation Mode

```bash
python src/main.py --symbol TSLA --sim
```

- Uses randomly generated mock data
- No IBKR connection required
- Fast iteration for development

### 11.5 Backtest Mode

```bash
python src/main.py --symbol TSLA --backtest --date 20251023 --speed 100
```

- Replays real MBO (Level 3) data from Databento
- Simulates market data events
- Speed multiplier for faster testing
- Generates detailed report

---

## 12. Dependencies & Infrastructure

### 12.1 Python Dependencies

```
# Core
python>=3.10

# LLM
openai>=1.0.0              # OpenAI-compatible client (works with Grok, OpenAI)

# Trading
ib_insync>=0.9.86          # IBKR API wrapper
websockets>=12.0           # Async websockets

# Data Processing
pandas>=2.0.0
numpy>=1.24.0

# Backtesting
databento>=0.30.0          # MBO data decoding

# Utilities
python-dotenv>=1.0.0       # Environment variable loading
colorama>=0.4.6            # Colored console output

# Development
pytest>=7.0.0
pytest-asyncio>=0.21.0
```

### 12.2 Infrastructure Requirements

| Component | Requirement |
|-----------|-------------|
| IBKR TWS/Gateway | Running on localhost:7496/7497 |
| Internet | Low latency connection |
| Python | 3.10+ |
| RAM | 4GB minimum |
| Storage | 10GB for backtest data |

### 12.3 API Requirements

| Provider | Endpoint | Auth |
|----------|----------|------|
| Grok (xAI) | api.x.ai/v1 | API Key |
| OpenAI | api.openai.com/v1 | API Key |
| Anthropic | api.anthropic.com/v1 | API Key |
| Databento | databento.com | API Key (for data download) |

---

## 13. Security Requirements

### 13.1 API Key Management

- **NEVER** hardcode API keys in source code
- Store in environment variables or `.env` file
- `.env` must be in `.gitignore`
- Rotate keys periodically

### 13.2 IBKR Security

- Use paper trading for development
- Implement kill switch for runaway losses
- Log all order submissions
- Rate limit order submissions

### 13.3 Audit Trail

All trading decisions must be logged with:
- Timestamp
- Market state at decision time
- LLM prompt and response
- Execution result
- P&L impact

---

## 14. Testing Requirements

### 14.1 Unit Tests

| Module | Tests |
|--------|-------|
| Brain | Tool parsing, validation, provider abstraction |
| Market Data | Each analyzer component |
| Execution | Order validation, position tracking |
| Context | Context building from state |

### 14.2 Integration Tests

| Test | Description |
|------|-------------|
| End-to-End Sim | Full loop with mock data |
| Backtest Run | Complete backtest with report |
| LLM Tool Calling | Verify tool calls parse correctly |

### 14.3 Manual Testing Scenarios

1. **Clear with-trend setup** → Should enter HIGH conviction
2. **Clear counter-trend setup** → Should WAIT or enter cautiously
3. **Spread too wide** → Must WAIT
4. **Low velocity with good signals** → Reduce conviction
5. **Already in a trade** → Should manage, not enter again
6. **First signal in downtrend** → Should WAIT for confirmation
7. **Absorption at key level** → Can be high conviction

---

## 15. Future Enhancements

### 15.1 Planned Features

| Feature | Priority | Description |
|---------|----------|-------------|
| Multi-symbol support | Medium | Trade multiple tickers |
| Scale in/out | Medium | Partial entries and exits |
| News integration | Low | Filter trades around news events |
| ML-based filtering | Low | Pre-filter setups before LLM |
| Web dashboard | Low | Real-time monitoring UI |

### 15.2 Performance Optimization

| Area | Optimization |
|------|--------------|
| Latency | Local LLM inference (Ollama) |
| Token usage | Compressed context format |
| Data | Pre-computed indicators |

### 15.3 Strategy Enhancements

| Enhancement | Description |
|-------------|-------------|
| Multi-timeframe | Add higher timeframe trend filter |
| Order flow patterns | Detect specific patterns (iceberg, sweeps) |
| Time-of-day filters | Adjust strategy by session |

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| CVD | Cumulative Volume Delta - running sum of buy volume minus sell volume |
| DOM | Depth of Market - Level 2 order book |
| HOD | High of Day |
| LOD | Low of Day |
| L2 | Level 2 market data (order book) |
| MBO | Market by Order - Level 3 data showing individual orders |
| POC | Point of Control - price with highest volume |
| RVOL | Relative Volume - current volume vs average |
| VAH | Value Area High |
| VAL | Value Area Low |
| VWAP | Volume-Weighted Average Price |

---

## Appendix B: Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-20 | Initial implementation with DSL parsing |
| 2.0.0 | 2026-01-24 | Tool calling architecture, provider abstraction |

---

*End of Requirements Document*
