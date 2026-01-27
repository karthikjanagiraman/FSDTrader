# FSDTrader: Comprehensive System Requirements & Design

**Version:** 3.0.0
**Date:** 2026-01-26
**Status:** Implementation Complete

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Trading Strategy & Philosophy](#3-trading-strategy--philosophy)
4. [Market Data Layer](#4-market-data-layer)
5. [Execution Layer](#5-execution-layer)
6. [Brain Module (LLM Decision Engine)](#6-brain-module-llm-decision-engine)
7. [Risk Management](#7-risk-management)
8. [Data Schemas](#8-data-schemas)
9. [Operating Modes](#9-operating-modes)
10. [Implementation Checklist](#10-implementation-checklist)
11. [File Structure](#11-file-structure)
12. [Appendices](#12-appendices)

---

## 1. Executive Summary

### 1.1 Purpose

FSDTrader is an autonomous day trading system for TSLA that uses Large Language Models (LLMs) to make real-time trading decisions based on order flow analysis. The system reads Level 2 market depth, time & sales (tape), cumulative delta, and volume profile to identify momentum opportunities.

### 1.2 Key Features

- **Order Flow Analysis**: Real-time L2/L3 market data processing
- **LLM Decision Engine**: Native function calling for structured trade decisions
- **Bracket Order Execution**: Automated stop loss and profit target management
- **Multi-Mode Operation**: Backtest, simulation, paper, and live trading
- **Data Parity**: Identical data structures across all modes

### 1.3 Trading Style

- **Type**: Momentum/Scalping
- **Target Moves**: $0.50 to $2.00 over 1-5 minutes
- **Position Size**: Maximum 100 shares per trade
- **Risk/Reward**: Minimum 1:2

---

## 2. System Architecture

### 2.1 High-Level Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FSDTrader Main                                  │
│                              (Orchestrator)                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────────┐    ┌───────────────────┐    ┌───────────────────┐
│ MarketDataProvider│    │    TradingBrain   │    │ ExecutionProvider │
│    (abstract)     │    │  (LLM Decision)   │    │    (abstract)     │
└───────────────────┘    └───────────────────┘    └───────────────────┘
        │                         │                         │
   ┌────┴────────┐          ┌─────┴─────┐            ┌──────┴──────┐
   ▼             ▼          ▼           ▼            ▼             ▼
┌──────────┐ ┌──────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌──────────┐ ┌──────────┐
│   IBKR   │ │   MBO    │ │  Grok   │ │  Groq   │ │ OpenAI  │ │Simulated │ │   IBKR   │
│ Connector│ │ Replay   │ │Provider │ │ Provider│ │ Provider│ │ Executor │ │ Executor │
└──────────┘ └──────────┘ └─────────┘ └─────────┘ └─────────┘ └──────────┘ └──────────┘
     │            │             │           │           │            │            │
     ▼            ▼             ▼           ▼           ▼            ▼            ▼
  TWS API   Databento DBN    xAI API   Groq API   OpenAI API  Virtual Acc   Real Orders
```

### 2.2 Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `main.py` | Orchestration, mode selection, main loop |
| `market_data.py` | Order book, tape, CVD, volume profile, absorption |
| `data_replay.py` | MBO file replay for backtesting |
| `brain/` | LLM interaction, prompt engineering, tool definitions |
| `execution/` | Order management, position tracking, P&L calculation |
| `reporting/` | Backtest reports, trade analysis |

### 2.3 Data Flow

```
┌──────────────────┐
│ Market Data Feed │  (IBKR or MBO Replay)
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Analysis Layer   │  OrderBook, TapeStream, CVD, VolumeProfile, etc.
└────────┬─────────┘
         ▼
┌──────────────────┐
│ State Builder    │  get_full_state() → MARKET_STATE
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Context Builder  │  MARKET_STATE + ACCOUNT_STATE + ACTIVE_ORDERS
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Trading Brain    │  LLM with function calling
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Command Parser   │  Tool call → Command string
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Executor         │  SimulatedExecutor or IBKRExecutor
└──────────────────┘
```

---

## 3. Trading Strategy & Philosophy

### 3.1 Core Philosophy

> "Trade what you SEE, not what you think should happen."

The system hunts for momentum trades - quick moves driven by aggressive order flow. It reads the tape like an experienced trader, looking for:

- **Velocity**: The speed and urgency of order flow
- **Direction**: Net buying vs selling pressure
- **Absorption**: Big orders getting eaten at key levels
- **Confirmation**: Signals that persist, not just blips

### 3.2 Entry Criteria

#### LONG Entry Requirements

1. **Trend Alignment**: CVD rising, price above VWAP
2. **L2 Imbalance**: Buyers in control (ratio > 1.5)
3. **Tape Confirmation**: Aggressive buying, MEDIUM+ velocity
4. **Signal Persistence**: Pattern held for 2+ snapshots (10+ seconds)
5. **Stop Placement**: Clear support level for stop loss
6. **Risk/Reward**: Minimum 1:2

#### SHORT Entry Requirements

1. **Trend Alignment**: CVD falling, price below VWAP
2. **L2 Imbalance**: Sellers in control (ratio < 0.6)
3. **Tape Confirmation**: Aggressive selling, MEDIUM+ velocity
4. **Signal Persistence**: Pattern held for 2+ snapshots
5. **Stop Placement**: Clear resistance level for stop loss
6. **Risk/Reward**: Minimum 1:2

### 3.3 Counter-Trend Rules

When trading against the session trend:

- Require HIGH conviction with multiple confirmations
- Use smaller size (50 shares instead of 100)
- Use tighter stops (20 cents max)
- Expect lower win rate

**Warning**: A few aggressive buy prints in a downtrend is NOT a reversal. Wait for actual trend change.

### 3.4 Exit Rules

| Exit Type | Trigger |
|-----------|---------|
| Stop Loss | Price hits stop price |
| Take Profit | Price hits target price |
| Manual Exit | Thesis invalidated (momentum reversal, CVD flip) |
| Time Exit | Session end (optional) |

### 3.5 Session Awareness

| Session | Time (ET) | Spread Limit | Characteristics |
|---------|-----------|--------------|-----------------|
| OPEN_DRIVE | 9:30-10:00 | $0.15 | High volatility, wide spreads |
| OPEN_RANGE | 10:00-10:30 | $0.08 | Range establishment |
| MORNING | 10:30-12:00 | $0.08 | Trend development |
| MIDDAY | 12:00-14:00 | $0.08 | Low volume, chop |
| CLOSE | 14:00-16:00 | $0.08 | Increasing activity |

---

## 4. Market Data Layer

### 4.1 Analysis Components

The system maintains seven analysis components, each providing specific market insights:

#### 4.1.1 OrderBook (Level 2 / DOM)

```python
class OrderBook:
    """Real-time Level 2 Order Book with Wall Detection."""

    bids: Dict[float, int]  # Price -> Size
    asks: Dict[float, int]

    def get_imbalance(self) -> float:
        """Bid/ask volume ratio (top 5 levels)."""

    def get_spread(self) -> float:
        """Best ask - best bid."""

    def get_walls(self, last_price: float) -> List[DOMWall]:
        """Detect significant orders (>3x average size)."""

    def get_stack(self, side: str, levels: int = 3) -> List[List]:
        """Top N price levels with sizes."""
```

**Wall Detection Tiers**:
- MAJOR: Size > 5x average
- MINOR: Size > 3x average

#### 4.1.2 TapeStream (Time & Sales)

```python
class TapeStream:
    """Tape Analysis with Delta Windows and Large Print Detection."""

    trades: deque  # Last 500 trades
    large_threshold: int = 300  # Shares

    def get_velocity(self, timestamp=None) -> tuple[str, float]:
        """Returns (label, trades_per_second).
        LOW: <10 tps, MEDIUM: 10-30 tps, HIGH: >30 tps"""

    def get_sentiment(self, timestamp=None) -> str:
        """AGGRESSIVE_BUYING | AGGRESSIVE_SELLING | NEUTRAL"""

    def get_delta(self, seconds: int, timestamp=None) -> int:
        """Net delta (buy_vol - sell_vol) over window."""

    def get_large_prints(self, seconds: int = 60, timestamp=None) -> List[LargePrint]:
        """Large trades (>300 shares) in time window."""
```

**Side Detection Logic**:
```python
if price >= last_ask:
    side = "BUY"    # Hit the ask = aggressive buyer
elif price <= last_bid:
    side = "SELL"   # Hit the bid = aggressive seller
else:
    side = "BUY" if price > mid else "SELL"
```

#### 4.1.3 FootprintTracker

```python
class FootprintTracker:
    """Per-bar delta analysis with imbalance detection."""

    def on_trade(self, price, size, side, timestamp=None):
        """Process trade into current bar."""

    def get_current_bar(self) -> Dict:
        """Returns: open, high, low, close, delta, volume,
                    delta_pct, poc, imbalances"""
```

#### 4.1.4 CumulativeDelta (CVD)

```python
class CumulativeDelta:
    """Session cumulative delta tracking."""

    session_delta: int  # Running total

    def add_trade(self, size, side, timestamp=None):
        """Add trade to cumulative delta."""

    def get_trend(self) -> str:
        """RISING | FALLING | FLAT (based on 5-min slope)"""

    def get_slope_5m(self) -> float:
        """Normalized slope (-1 to 1)."""
```

#### 4.1.5 VolumeProfile

```python
class VolumeProfile:
    """Session volume at price analysis."""

    def add_trade(self, price, size):
        """Add volume at price."""

    def get_poc(self) -> float:
        """Point of Control (price with highest volume)."""

    def get_value_area(self) -> tuple[float, float]:
        """Returns (VAH, VAL) - 70% of volume."""

    def get_price_location(self, price) -> str:
        """ABOVE | BELOW | AT_POC"""
```

#### 4.1.6 AbsorptionDetector

```python
class AbsorptionDetector:
    """Detects absorption at price levels."""

    def on_trade(self, price, size, side, timestamp=None):
        """Track trades at levels."""

    def detect(self) -> Dict:
        """Returns: {detected: bool, side: str, price: float}"""
```

**Absorption Signal**: Large volume absorbed at a price level while price holds (potential reversal).

#### 4.1.7 MarketMetrics

```python
class MarketMetrics:
    """Session-level metrics (VWAP, HOD, LOD, RVOL)."""

    vwap: float
    hod: float  # High of day
    lod: float  # Low of day

    def add_trade(self, price, size, timestamp=None):
        """Update metrics with trade."""

    def get_time_session(self, timestamp=None) -> str:
        """Current session: OPEN_DRIVE | OPEN_RANGE | MORNING | MIDDAY | CLOSE"""

    def get_rvol(self, timestamp=None) -> float:
        """Relative volume vs 30-day average."""

    def get_hod_lod_location(self) -> str:
        """TESTING_HOD | NEAR_HOD | MID_RANGE | NEAR_LOD | TESTING_LOD"""
```

### 4.2 Timestamp Handling

**Critical Requirement**: All components accept an optional `timestamp` parameter for backtest simulation time.

```python
# IBKR Mode: Uses real-time
tape.get_velocity()  # Uses time.time() internally

# Backtest Mode: Uses simulated time
tape.get_velocity(timestamp=self.current_simulation_time)
```

### 4.3 Data Providers

#### 4.3.1 IBKRConnector (Live/Paper)

- Connects to TWS via ib_insync
- Subscribes to Level 2 market depth
- Receives tick-by-tick trade data
- Updates all analysis components in real-time

#### 4.3.2 MBOReplayConnector (Backtest)

- Reads Databento MBO (Market-by-Order) files
- Reconstructs order book from ADD/CANCEL/MODIFY events
- Processes TRADE/FILL events for tape analysis
- Maintains simulation time for accurate timing

**MBO Event Mapping**:

| MBO Action | Description | Order Book Update |
|------------|-------------|-------------------|
| ADD | New order | Add size at price |
| CANCEL | Order cancelled | Remove size from price |
| MODIFY | Order modified | Remove old, add new |
| TRADE | Trade execution | Feed to all analyzers |
| FILL | Partial/full fill | Feed to all analyzers |

---

## 5. Execution Layer

### 5.1 ExecutionProvider Interface

```python
from abc import ABC, abstractmethod

class ExecutionProvider(ABC):
    """Abstract interface for order execution."""

    @abstractmethod
    def submit_bracket_order(
        self,
        side: str,              # "BUY" or "SELL"
        size: int,
        limit_price: float,
        stop_loss: float,
        profit_target: float,
        context: Optional[Dict] = None
    ) -> OrderResult:
        """Submit entry + stop + target as a unit."""

    @abstractmethod
    def modify_stop(self, new_price: float) -> OrderResult:
        """Move stop loss price."""

    @abstractmethod
    def modify_target(self, new_price: float) -> OrderResult:
        """Move profit target price."""

    @abstractmethod
    def exit_position(self, reason: str = "MANUAL") -> OrderResult:
        """Exit at market."""

    @abstractmethod
    def cancel_all(self) -> OrderResult:
        """Cancel all pending orders."""

    @abstractmethod
    def update(self, current_price: float, timestamp: float) -> None:
        """Update with current market price (for fill detection)."""

    @abstractmethod
    def get_position(self) -> Position:
        """Get current position state."""

    @abstractmethod
    def get_account_state(self) -> Dict[str, Any]:
        """Get account state for Brain context."""

    @abstractmethod
    def get_active_orders(self) -> List[Dict[str, Any]]:
        """Get list of active orders."""

    @abstractmethod
    def get_trade_history(self) -> List[TradeRecord]:
        """Get completed trades."""

    @abstractmethod
    def reset(self) -> None:
        """Reset for new session."""
```

### 5.2 Data Types

#### Position

```python
@dataclass
class Position:
    side: PositionSide = PositionSide.FLAT  # FLAT | LONG | SHORT
    size: int = 0
    avg_entry: float = 0.0
    unrealized_pnl: float = 0.0
    entry_time: float = 0.0

    def is_flat(self) -> bool:
        return self.side == PositionSide.FLAT or self.size == 0
```

#### BracketOrder

```python
@dataclass
class BracketOrder:
    """Entry + Stop + Target as a unit."""
    entry_order_id: int
    stop_order_id: int
    target_order_id: int
    entry_price: float
    stop_price: float
    target_price: float
    side: str           # "BUY" or "SELL"
    quantity: int
    status: str         # "PENDING" | "FILLED" | "CLOSED"
    fill_price: float = 0.0
    fill_time: float = 0.0
```

#### OrderResult

```python
@dataclass
class OrderResult:
    success: bool
    order_id: Optional[int] = None
    error: Optional[str] = None
    fill_price: Optional[float] = None
    fill_size: Optional[int] = None
    message: Optional[str] = None
```

#### TradeRecord

```python
@dataclass
class TradeRecord:
    """Completed trade for reporting."""
    entry_time: float
    exit_time: float
    duration_seconds: float
    side: str               # "LONG" | "SHORT"
    size: int
    entry_price: float
    exit_price: float
    pnl: float
    exit_reason: str        # "STOP" | "TARGET" | "MANUAL"

    # Market context at entry
    cvd_trend: str = ""
    l2_imbalance: float = 0.0
    tape_velocity: str = ""
    spread: float = 0.0
    time_session: str = ""
```

### 5.3 SimulatedExecutor

Virtual order execution for backtest/simulation modes.

> **Note (v3.0.0):** Execution-level validation (position size, stop distance, target direction) has been disabled in both `SimulatedExecutor` and `IBKRExecutor`. The LLM has full context and autonomously decides what's appropriate for each trade. The only remaining check is duplicate position prevention (`ALREADY_IN_POSITION`).

**Fill Logic**:

```python
def update(self, current_price: float, timestamp: float):
    """Check for fills based on current price."""

    # Check entry fill (limit order)
    if self.active_bracket.status == "PENDING":
        if side == "BUY":
            # Long entry: fill if price <= limit
            if current_price <= entry_price:
                fill_price = min(entry_price, current_price)
                self._fill_entry(fill_price, timestamp)
        else:
            # Short entry: fill if price >= limit
            if current_price >= entry_price:
                fill_price = max(entry_price, current_price)
                self._fill_entry(fill_price, timestamp)

    # Check stop/target fills (when in position)
    elif self.active_bracket.status == "FILLED":
        if position_side == "LONG":
            if current_price <= stop_price:
                self._close_position(stop_price, "STOP", timestamp)
            elif current_price >= target_price:
                self._close_position(target_price, "TARGET", timestamp)
        else:  # SHORT
            if current_price >= stop_price:
                self._close_position(stop_price, "STOP", timestamp)
            elif current_price <= target_price:
                self._close_position(target_price, "TARGET", timestamp)
```

**Fill Price Assumptions**:

| Order Type | LONG Fill Price | SHORT Fill Price |
|------------|-----------------|------------------|
| Limit Entry | min(limit, current) | max(limit, current) |
| Stop Loss | stop_price | stop_price |
| Take Profit | target_price | target_price |
| Market Exit | current_price | current_price |

### 5.4 IBKRExecutor

Real order execution via IBKR TWS API using ib_insync.

**Key Features**:
- Submits bracket orders as OCA (One-Cancels-All) groups
- Hooks into `orderStatusEvent` and `execDetailsEvent`
- Syncs position state from IBKR
- Fetches buying power from account values

**Bracket Order Submission**:
```python
# Parent order (entry)
parent = Order()
parent.action = side
parent.orderType = "LMT"
parent.lmtPrice = limit_price
parent.transmit = False

# Take profit (child)
take_profit = Order()
take_profit.parentId = parent.orderId
take_profit.orderType = "LMT"
take_profit.lmtPrice = profit_target
take_profit.transmit = False

# Stop loss (child)
stop_order = Order()
stop_order.parentId = parent.orderId
stop_order.orderType = "STP"
stop_order.auxPrice = stop_loss
stop_order.transmit = True  # Transmit all orders
```

### 5.5 Command Execution (DSL)

The executor accepts command strings from the Brain:

```python
def execute(self, command: str, current_spread: float = 0.0,
            context: Optional[Dict] = None) -> Dict:
    """
    Execute command string from Brain.
    Format: "TOOL_NAME|arg1=val1|arg2=val2|reasoning=..."
    """
```

**Supported Commands**:

| Command | Arguments | Description |
|---------|-----------|-------------|
| `ENTER_LONG` | limit_price, stop_loss, profit_target, size | Enter long position |
| `ENTER_SHORT` | limit_price, stop_loss, profit_target, size | Enter short position |
| `UPDATE_STOP` | new_price | Modify stop loss |
| `UPDATE_TARGET` | new_price | Modify profit target |
| `EXIT_POSITION` | reasoning | Close at market |
| `WAIT` | reasoning | No action |

---

## 6. Brain Module (LLM Decision Engine)

### 6.1 Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        TradingBrain                           │
├──────────────────────────────────────────────────────────────┤
│  System Prompt (persona, strategy, rules)                     │
│  Tool Definitions (enter_long, wait, etc.)                    │
│  Context Builder (shared instance for price history)          │
│  Decision History (last N decisions for continuity)          │
│  Session Logger (one file per backtest session)              │
│  Validator (DISABLED - LLM handles all validation)           │
└───────────────────────┬──────────────────────────────────────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │   Grok   │  │   Groq   │  │  OpenAI  │
    │ Provider │  │ Provider │  │ Provider │
    └──────────┘  └──────────┘  └──────────┘
```

### 6.2 System Prompt

The system prompt defines the LLM's:

1. **Persona**: Experienced TSLA momentum trader
2. **Trading Style**: Quick moves, order flow based
3. **Entry Criteria**: What to look for and avoid
4. **Trend Awareness**: With-trend vs counter-trend rules
5. **Confirmation Requirements**: Signal persistence
6. **Hard Limits**: Spread, stop distance, risk/reward
7. **Tool Usage**: Which tool for which situation

**Version**: 3.0.0 (tracked for reproducibility)

### 6.3 Tool Definitions

Six tools available via native function calling:

```python
TRADING_TOOLS = [
    {
        "name": "enter_long",
        "description": "Enter LONG with bracket orders...",
        "parameters": {
            "limit_price": {"type": "number"},
            "stop_loss": {"type": "number"},
            "profit_target": {"type": "number"},
            "size": {"type": "integer", "default": 100},
            "conviction": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
            "reasoning": {"type": "string"}
        },
        "required": ["limit_price", "stop_loss", "profit_target", "conviction", "reasoning"]
    },
    # ... enter_short, update_stop, update_target, exit_position, wait
]
```

### 6.4 Context Building

The Brain receives a structured snapshot of market conditions:

```python
def build_context(self, state: Dict) -> str:
    """Build LLM context from current state."""

    market = state["MARKET_STATE"]
    account = state["ACCOUNT_STATE"]
    orders = state["ACTIVE_ORDERS"]

    context = f"""
## Current Market State
LAST: ${market['LAST']:.2f} | VWAP: ${market['VWAP']:.2f}
Session: {market['TIME_SESSION']} | Spread: ${market['SPREAD']:.2f}

## Order Flow
L2 Imbalance: {market['L2_IMBALANCE']:.2f}
Tape Velocity: {market['TAPE_VELOCITY']} ({market['TAPE_VELOCITY_TPS']:.1f} tps)
Tape Sentiment: {market['TAPE_SENTIMENT']}
CVD Trend: {market['CVD_TREND']} (slope: {market['CVD_SLOPE_5M']:.2f})

## Position
Status: {account['POSITION_SIDE']}
...
"""
    return context
```

### 6.5 Decision Flow

1. **Receive State**: Market snapshot from connector
2. **Build Context**: Format state for LLM consumption (uses shared ContextBuilder with price history)
3. **Add History**: Include recent decisions for continuity
4. **Call LLM**: Send context with tool definitions
5. **Parse Response**: Extract tool call and arguments
6. **Log**: Append context + response to session log file
7. **Return Command**: Formatted command string for executor

> **Note (v3.0.0):** Brain-level validation (step 6 in v2.0.0) has been disabled. The LLM makes autonomous decisions without pre-flight checks on stop distance, position size, or risk/reward. The validation module exists in code but `validate_tool_call()` is not called.

### 6.6 Validation (DISABLED)

The validation framework exists but is **not active**. The LLM is trusted to make appropriate decisions based on the full context it receives including risk parameters in the system prompt.

```python
@dataclass
class ValidationResult:
    valid: bool
    errors: List[str]
    warnings: List[str]
```

### 6.7 Session Logging

LLM context logging uses one file per backtest session (not per call):

```python
# Session log file: data/logs/llm_contexts/session_YYYYMMDD_HHMMSS.txt
#
# Each call appends:
# - Call number and timestamp
# - System prompt (first call only)
# - User message (context)
# - LLM response (tool call, arguments, conviction, reasoning)
# - Usage stats (latency, token counts)
```

### 6.8 Shared Context Builder

The `ContextBuilder` is a shared module-level instance (via `get_context_builder()`) so that `record_price()` calls from `main.py` and the brain's context building use the same price history:

```python
# main.py records prices during backtest loop
from brain.context import record_price, clear_price_history
record_price(last_price, price_time, label)  # e.g., label="HOD test"

# brain.py uses the shared instance
self._context_builder = get_context_builder()  # Same instance
```

### 6.9 LLM Providers

| Provider | API | Default Model | Key Feature |
|----------|-----|---------------|-------------|
| Grok | xAI (`api.x.ai`) | `grok-3-mini-fast` | xAI native |
| Groq | Groq (`api.groq.com`) | `llama-3.3-70b-versatile` | Ultra-fast inference (10-30x) |
| OpenAI | OpenAI (`api.openai.com`) | `gpt-4o` | GPT-4 family |

All providers implement the `LLMProvider` interface with OpenAI-compatible chat completions + tool calling.

---

## 7. Risk Management

### 7.1 Philosophy (v3.0.0)

> **LLM-Driven Risk Management**: As of v3.0.0, all hard-coded validation has been disabled at both the brain and execution layers. The LLM receives full market context (including risk parameters in the system prompt) and makes autonomous decisions about position sizing, stop distance, and entry conditions. This allows the LLM to adapt its risk management to market conditions rather than being constrained by static rules.

### 7.2 Risk Limits (Reference Only)

The `RiskLimits` data class still exists but its validations are **not enforced**:

```python
@dataclass
class RiskLimits:
    # Position limits
    max_position_size: int = 100        # Max shares per trade

    # Daily limits
    max_daily_loss: float = -500.0      # Stop trading if hit
    max_daily_trades: int = 10          # Max round-trips

    # Entry conditions
    max_spread: float = 0.15            # Session-aware

    # Stop distance (NOT ENFORCED)
    min_stop_distance: float = 0.10     # $0.10 minimum
    max_stop_distance: float = 0.30     # $0.30 maximum
```

### 7.3 Enforcement Points

| Check | Location | Status | Action |
|-------|----------|--------|--------|
| Daily loss limit | Main loop | **ACTIVE** | Stop trading session |
| Daily trade limit | Executor.execute() | **ACTIVE** | Reject new entries |
| Spread limit | Executor.execute() | **ACTIVE** | Reject entry if too wide |
| Stop distance | submit_bracket_order() | **DISABLED** | ~~Reject if out of range~~ |
| Position size | submit_bracket_order() | **DISABLED** | ~~Reject if exceeds max~~ |
| Target direction | submit_bracket_order() | **DISABLED** | ~~Reject if wrong direction~~ |
| Already in position | submit_bracket_order() | **ACTIVE** | Reject duplicate entry |
| Brain validation | brain.py think() | **DISABLED** | ~~Pre-flight checks~~ |

### 7.4 What the LLM Controls

The LLM now has full autonomy over:
- **Position size** (no max limit enforced)
- **Stop distance** (no min/max enforced)
- **Target placement** (no direction check enforced)
- **Risk/reward ratio** (guidelines in prompt only)
- **Entry timing** (spread limits in prompt, not hard-coded)

### 7.5 Spread Limits by Session (Prompt-Based)

These are communicated to the LLM via the system prompt and context builder, but not hard-enforced at execution:

```python
SPREAD_LIMITS = {
    "OPEN_DRIVE": 0.15,   # 9:30-10:00
    "OPEN_RANGE": 0.08,   # 10:00-10:30
    "MORNING": 0.08,      # 10:30-12:00
    "MIDDAY": 0.08,       # 12:00-14:00
    "CLOSE": 0.08         # 14:00-16:00
}
```

---

## 8. Data Schemas

### 8.1 MARKET_STATE Schema

Both IBKRConnector and MBOReplayConnector output identical structure:

```python
{
    "MARKET_STATE": {
        # Identification
        "TICKER": str,              # "TSLA"
        "LAST": float,              # Last trade price
        "VWAP": float,              # Volume-weighted average price
        "TIME_SESSION": str,        # OPEN_DRIVE | OPEN_RANGE | MORNING | MIDDAY | CLOSE

        # Level 2 (DOM)
        "L2_IMBALANCE": float,      # Bid/ask volume ratio (top 5 levels)
        "SPREAD": float,            # Best ask - best bid
        "DOM_WALLS": List[{         # Significant orders
            "side": str,            # BID | ASK
            "price": float,
            "size": int,
            "tier": str,            # MAJOR | MINOR
            "distance_pct": float
        }],
        "BID_STACK": List[[price, size]],  # Top 3 bid levels
        "ASK_STACK": List[[price, size]],  # Top 3 ask levels

        # Tape (Time & Sales)
        "TAPE_VELOCITY": str,       # LOW | MEDIUM | HIGH
        "TAPE_VELOCITY_TPS": float, # Trades per second
        "TAPE_SENTIMENT": str,      # AGGRESSIVE_BUYING | AGGRESSIVE_SELLING | NEUTRAL
        "TAPE_DELTA_1S": int,       # Net delta last 1 second
        "TAPE_DELTA_5S": int,       # Net delta last 5 seconds
        "LARGE_PRINTS_1M": List[{   # Large trades in last 60s
            "price": float,
            "size": int,
            "side": str,
            "secs_ago": float
        }],

        # Footprint
        "FOOTPRINT_CURR_BAR": {
            "open": float,
            "high": float,
            "low": float,
            "close": float,
            "delta": int,
            "volume": int,
            "delta_pct": float,
            "poc": float,
            "imbalances": List[Dict]
        },

        # Cumulative Delta (CVD)
        "CVD_SESSION": int,         # Session cumulative delta
        "CVD_TREND": str,           # RISING | FALLING | FLAT
        "CVD_SLOPE_5M": float,      # Normalized slope (-1 to 1)

        # Volume Profile
        "VP_POC": float,            # Point of control
        "VP_VAH": float,            # Value area high
        "VP_VAL": float,            # Value area low
        "VP_DEVELOPING_POC": float,
        "PRICE_VS_POC": str,        # ABOVE | BELOW | AT_POC

        # Key Levels
        "HOD": float,               # High of day
        "LOD": float,               # Low of day
        "HOD_LOD_LOC": str,         # TESTING_HOD | NEAR_HOD | MID_RANGE | NEAR_LOD | TESTING_LOD
        "DISTANCE_TO_HOD_PCT": float,
        "RVOL_DAY": float,          # Relative volume

        # Absorption
        "ABSORPTION_DETECTED": bool,
        "ABSORPTION_SIDE": str | None,
        "ABSORPTION_PRICE": float | None
    }
}
```

### 8.2 ACCOUNT_STATE Schema

```python
{
    "ACCOUNT_STATE": {
        # Position
        "POSITION": int,            # Size (0 if flat)
        "POSITION_SIDE": str,       # FLAT | LONG | SHORT
        "AVG_ENTRY": float,         # Average entry price
        "UNREALIZED_PL": float,     # Current unrealized P&L

        # Daily Stats
        "DAILY_PL": float,          # Realized P&L for session
        "DAILY_TRADES": int,        # Round-trip count

        # Risk Status
        "BUYING_POWER": float,      # Available margin
        "DAILY_LOSS_REMAINING": float  # Distance to max loss
    }
}
```

### 8.3 ACTIVE_ORDERS Schema

```python
{
    "ACTIVE_ORDERS": [
        {
            "order_id": int,
            "type": str,            # LIMIT | STOP | MARKET
            "side": str,            # BUY | SELL
            "price": float,
            "size": int,
            "status": str,          # PENDING | PARTIAL | FILLED | CANCELLED
            "purpose": str          # ENTRY | STOP_LOSS | PROFIT_TARGET | EXIT
        }
    ]
}
```

---

## 9. Operating Modes

### 9.1 Mode Configuration

| Mode | CLI Flag | Data Source | Executor | Description |
|------|----------|-------------|----------|-------------|
| Backtest | `--backtest` | MBOReplayConnector | SimulatedExecutor | Historical replay |
| Simulation | `--sim` | MockDataGenerator | SimulatedExecutor | Random mock data |
| Paper | `--paper` | IBKRConnector (7497) | IBKRExecutor | IBKR paper account |
| Live | `--live` | IBKRConnector (7496) | IBKRExecutor | IBKR live account |

### 9.2 Backtest Mode

```python
async def _start_backtest(self):
    # Initialize MBO replay
    self.connector = MBOReplayConnector(
        data_dir="BacktestData/TSLA-L3DATA",
        date=self.backtest_date,
        symbol=self.symbol
    )

    # Initialize simulated executor
    self.executor = SimulatedExecutor(
        risk_limits=self.risk_limits,
        symbol=self.symbol
    )

    # Start replay with callback (uses CLI --start-time / --end-time)
    await self.connector.start_replay(
        speed=self.backtest_speed,
        on_state_update=self._on_backtest_state,
        start_time=self.start_time,    # Default: "09:30:00"
        end_time=self.end_time          # Default: "16:00:00"
    )
```

### 9.3 Live/Paper Mode

```python
async def _start_live(self):
    port = 7497 if self.mode == "paper" else 7496

    # Initialize IBKR connector
    self.connector = IBKRConnector(port=port)
    await self.connector.connect()

    # Initialize IBKR executor
    self.executor = IBKRExecutor(
        ib=self.connector.ib,
        symbol=self.symbol,
        risk_limits=self.risk_limits
    )

    await self._run_loop_live()
```

### 9.4 Main Loop Flow

```python
async def _on_state_update(self, state):
    # 1. Track price history for LLM context (with key level labels)
    from brain.context import record_price
    label = None
    if abs(last_price - hod) < 0.05:
        label = "HOD test"
    elif abs(last_price - lod) < 0.05:
        label = "LOD test"
    elif abs(last_price - vwap) < 0.05:
        label = "VWAP test"
    record_price(last_price, price_time, label)

    # 2. Update executor with current price (checks stop/target fills)
    last_price = state["MARKET_STATE"]["LAST"]
    self.executor.update(last_price, timestamp)

    # 3. Get account state from executor
    account_state = self.executor.get_account_state()

    # 4. Check daily loss limit
    if account_state["DAILY_PL"] <= self.risk_limits.max_daily_loss:
        self.running = False
        return

    # 5. Merge account into state
    state["ACCOUNT_STATE"] = account_state
    state["ACTIVE_ORDERS"] = self.executor.get_active_orders()

    # 6. Brain makes decision (LLM call with full context)
    command = self.brain.think(state)

    # 7. Execute command
    spread = state["MARKET_STATE"]["SPREAD"]
    context = state["MARKET_STATE"]
    result = self.executor.execute(command, current_spread=spread, context=context)
```

---

## 10. Implementation Checklist

### 10.1 Market Data Layer

- [x] OrderBook with wall detection
- [x] TapeStream with velocity and sentiment
- [x] FootprintTracker with delta analysis
- [x] CumulativeDelta with trend detection
- [x] VolumeProfile with POC/VAH/VAL
- [x] AbsorptionDetector
- [x] MarketMetrics (VWAP, HOD, LOD, RVOL)
- [x] IBKRConnector (live data)
- [x] MBOReplayConnector (backtest data)
- [x] All components accept `timestamp` parameter
- [x] get_full_state() returns identical schema

### 10.2 Execution Layer

- [x] ExecutionProvider abstract interface
- [x] SimulatedExecutor implementation
  - [x] Bracket order simulation
  - [x] Entry fill detection (limit orders)
  - [x] Stop/target fill detection
  - [x] P&L calculation
  - [x] Trade history tracking
  - [x] Context preservation for trade records
- [x] IBKRExecutor implementation
  - [x] Bracket order submission
  - [x] Order status event handling
  - [x] Fill event handling
  - [x] Buying power retrieval
- [x] RiskLimits enforcement
- [x] Command DSL parsing
- [x] Unit tests (69 tests passing)

### 10.3 Brain Module

- [x] System prompt (v3.0.0)
- [x] Tool definitions (6 tools)
- [x] Context builder (shared instance with price history)
- [x] Decision history tracking
- [x] Validation framework (exists but DISABLED)
- [x] GrokProvider (xAI)
- [x] GroqProvider (Llama 3.3 70B via Groq)
- [ ] OpenAI/Claude providers (optional)
- [x] Session-based LLM logging (one file per backtest)
- [x] Price history tracking with key level labels

### 10.4 Integration

- [x] main.py mode-based initialization
- [x] State merging (market + account + orders)
- [x] Backtest reporter integration
- [x] CLI --start-time / --end-time for backtest windows
- [x] CLI --provider / --model for LLM selection
- [x] .env file support (load_dotenv)
- [ ] End-to-end integration tests
- [ ] Performance benchmarks

---

## 11. File Structure

```
FSDTrader/
├── .env                              # API keys (GROQ_API_KEY, XAI_API_KEY, etc.)
├── .gitattributes                    # Git LFS tracking rules
│
├── requirements/
│   └── FSDTRADER_REQUIREMENTS.md     # This document
│
├── docs/
│   └── L2_DATA_PROCESSING.md        # L2 data flow documentation
│
├── src/
│   ├── main.py                       # Orchestrator (CLI, backtest/live loops)
│   │
│   ├── market_data.py                # Analysis components
│   │   ├── OrderBook
│   │   ├── TapeStream
│   │   ├── FootprintTracker
│   │   ├── CumulativeDelta
│   │   ├── VolumeProfile
│   │   ├── AbsorptionDetector
│   │   ├── MarketMetrics
│   │   └── IBKRConnector
│   │
│   ├── data_replay.py                # MBO replay
│   │   └── MBOReplayConnector
│   │
│   ├── reporting.py                  # BacktestReporter
│   │
│   ├── brain/
│   │   ├── __init__.py
│   │   ├── brain.py                  # TradingBrain (session logging)
│   │   ├── prompts.py                # System prompt
│   │   ├── tools.py                  # Tool definitions
│   │   ├── types.py                  # ToolCall, Decision, etc.
│   │   ├── context.py                # Context builder (shared instance, price history)
│   │   ├── validation.py             # Pre-flight checks (DISABLED)
│   │   └── providers/
│   │       ├── __init__.py           # Provider factory
│   │       ├── base.py               # LLMProvider ABC
│   │       ├── grok.py               # xAI Grok provider
│   │       └── groq.py               # Groq provider (Llama 3.3 70B)
│   │
│   └── execution/
│       ├── __init__.py               # Factory + exports
│       ├── base.py                   # ExecutionProvider ABC
│       ├── types.py                  # Position, BracketOrder, RiskLimits
│       ├── simulated.py              # SimulatedExecutor (validation DISABLED)
│       └── ibkr.py                   # IBKRExecutor (validation DISABLED)
│
├── data/
│   └── logs/
│       └── llm_contexts/             # Session log files
│           └── session_YYYYMMDD_HHMMSS.txt
│
├── tests/
│   └── test_execution.py             # 69 unit tests
│
└── BacktestData/                     # Git LFS tracked
    └── TSLA-L3DATA/                  # Databento MBO files (~2.7GB)
        ├── xnas-itch-YYYYMMDD.mbo.dbn.zst
        └── TSLA_MBO_Data_Documentation.md
```

---

## 12. Appendices

### A. Sample MBO Event (Databento)

```python
event = {
    "ts_event": 1729684200123456789,  # Nanoseconds since epoch
    "action": "TRADE",                 # ADD | CANCEL | MODIFY | TRADE | FILL
    "side": "ASK",                     # BID | ASK (which side was taken)
    "price": 245500000000,             # Fixed-point (divide by 1e9)
    "size": 100,                       # Shares
    "order_id": 12345678,              # Unique order ID
    "flags": 0                         # Event flags
}
```

### B. Sample Command String

```
ENTER_LONG|limit_price=245.50|stop_loss=245.30|profit_target=246.00|size=100|conviction=HIGH|reasoning=Aggressive buying at support, CVD rising, L2 imbalance 2.3
```

### C. Sample Trade Record

```python
TradeRecord(
    entry_time=1729684200.5,
    exit_time=1729684320.8,
    duration_seconds=120.3,
    side="LONG",
    size=100,
    entry_price=245.50,
    exit_price=246.00,
    pnl=50.00,
    exit_reason="TARGET",
    cvd_trend="RISING",
    l2_imbalance=2.3,
    tape_velocity="HIGH",
    spread=0.02,
    time_session="MORNING"
)
```

### D. Environment Variables

Stored in `.env` file (loaded via `python-dotenv`):

```bash
XAI_API_KEY=xai-xxxx        # Grok API key (xAI)
GROQ_API_KEY=gsk_xxxx       # Groq API key
OPENAI_API_KEY=sk-xxxx      # OpenAI API key (optional)
IBKR_PORT=7497              # TWS port (7497=paper, 7496=live)
```

### E. CLI Usage

```bash
# Backtest mode (full day)
python main.py --backtest --date 20251023 --speed 100

# Backtest with time window (30 minutes)
python main.py --backtest --start-time 09:30:00 --end-time 10:00:00

# Backtest with specific provider and model
python main.py --backtest --provider groq --model llama-3.3-70b-versatile

# Simulation mode
python main.py --sim

# Paper trading
python main.py --paper

# Live trading (CAUTION)
python main.py --live
```

**CLI Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--symbol` | `TSLA` | Trading symbol |
| `--backtest` | - | Backtest with MBO replay |
| `--sim` | - | Simulation with mock data |
| `--live` | - | Live trading (DANGER!) |
| `--date` | latest file | Backtest date (YYYYMMDD) |
| `--speed` | `100.0` | Backtest speed multiplier |
| `--start-time` | `09:30:00` | Backtest start time (HH:MM:SS) |
| `--end-time` | `16:00:00` | Backtest end time (HH:MM:SS) |
| `--provider` | `grok` | LLM provider (grok, groq, openai) |
| `--model` | provider default | LLM model override |

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-01-24 | Initial DATA_LAYER.md |
| 2.0.0 | 2025-01-24 | Comprehensive requirements combining all modules |
| 3.0.0 | 2026-01-26 | Disabled brain & execution validation (LLM-driven risk management), added Groq provider, session-based LLM logging, shared ContextBuilder with price history, CLI time args, .env support, L2 data processing docs |

---

*This document serves as the single source of truth for FSDTrader system requirements and design.*
