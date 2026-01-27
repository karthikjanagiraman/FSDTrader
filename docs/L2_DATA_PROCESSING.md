# L2 Data Processing in FSDTrader

This document provides a comprehensive technical reference for how Level 2 (order book) data flows through FSDTrader in both live trading and backtest modes.

---

## Table of Contents

1. [Overview](#overview)
2. [Data Sources](#data-sources)
3. [Order Book Data Structure](#order-book-data-structure)
4. [L2 Metrics Calculation](#l2-metrics-calculation)
5. [Related Analyzers](#related-analyzers)
6. [State Aggregation](#state-aggregation)
7. [Context Building for LLM](#context-building-for-llm)
8. [Main Event Loop](#main-event-loop)
9. [Data Flow Diagram](#data-flow-diagram)
10. [Live vs Backtest Comparison](#live-vs-backtest-comparison)
11. [File Reference](#file-reference)

---

## Overview

FSDTrader uses Level 2 market data (order book depth) as a primary input for LLM-based trading decisions. The system processes L2 data identically in both modes:

- **Live Mode**: Real-time data from Interactive Brokers (IBKR)
- **Backtest Mode**: Historical Level 3 (MBO) data from Databento, reconstructed into L2

The L2 data flows through several transformation stages before reaching the LLM:

```
Raw Events → OrderBook → Metrics → State Dict → Context String → LLM → Command
```

---

## Data Sources

### Live Mode: Interactive Brokers

**File:** `src/market_data.py` (lines 706-739)

The `IBKRConnector` subscribes to three IBKR data feeds:

```python
# L2 order book depth (20 levels, smart routing)
self.ib.reqMktDepth(contract, numRows=20, isSmartDepth=True)

# Time & Sales (every trade tick)
self.ib.reqTickByTickData(contract, 'AllLast', 0, False)

# Market snapshot (bid/ask/last/volume)
self.ib.reqMktData(contract, '', False, False)
```

**Event Callbacks:**

| Event | Callback | Purpose |
|-------|----------|---------|
| `updateMktDepthEvent` | `_on_depth()` | Order book changes |
| `tickByTickAllLastEvent` | `_on_tick()` | Trade executions |
| `pendingTickersEvent` | `_on_snapshot()` | Price/volume updates |

**How L2 updates arrive:**

```python
def _on_depth(self, item):
    """Called by IBKR when order book changes."""
    symbol = item.contract.symbol
    if symbol in self.books:
        self.books[symbol].update(item)  # Updates the OrderBook
```

### Backtest Mode: Databento MBO Replay

**File:** `src/data_replay.py` (lines 50-137)

The `MBOReplayConnector` replays historical Market-By-Order (Level 3) data:

```python
# Load compressed Databento file
store = db.DBNStore.from_file(filepath)
self.events = list(store)  # List of MBO events
```

**MBO Event Types:**

| Action | Description | Order Book Effect |
|--------|-------------|-------------------|
| `ADD` | New order placed | Add size at price level |
| `CANCEL` | Order cancelled | Remove size from level |
| `MODIFY` | Order changed | Update size/price |
| `TRADE` | Order executed | Remove size, update tape |

**Data Files Location:**
```
BacktestData/TSLA-L3DATA/
├── xnas-itch-20251023.mbo.dbn.zst
├── xnas-itch-20251024.mbo.dbn.zst
└── ...
```

**File Naming:** `xnas-itch-YYYYMMDD.mbo.dbn.zst`
- `xnas` = NASDAQ exchange
- `itch` = ITCH protocol format
- `mbo` = Market-By-Order (L3)
- `dbn` = Databento format
- `zst` = Zstandard compression

---

## Order Book Data Structure

**File:** `src/market_data.py` (lines 78-150)

### OrderBook Class

```python
@dataclass
class DOMWall:
    """Significant order in the book (wall)."""
    side: str           # "BID" or "ASK"
    price: float        # Price level
    size: int           # Order size
    tier: str           # "MAJOR" (>5x avg) or "MINOR" (>3x avg)
    distance_pct: float # % distance from current price

class OrderBook:
    def __init__(self, ticker: str):
        self.bids: Dict[float, int] = {}  # Price → Size (buy orders)
        self.asks: Dict[float, int] = {}  # Price → Size (sell orders)
        self.ticker = ticker
```

### Live Mode Update

```python
def update(self, dom_event):
    """Handle IBKR updateMktDepth event."""
    target = self.bids if dom_event.side == 1 else self.asks
    price = round(dom_event.price, 2)
    size = dom_event.size
    op = dom_event.operation

    if op in (0, 1):      # Insert or Update
        target[price] = size
    elif op == 2:         # Delete
        target.pop(price, None)
```

**IBKR Operation Codes:**
- `0` = Insert new level
- `1` = Update existing level
- `2` = Delete level

### Backtest Mode Reconstruction

**File:** `src/data_replay.py` (lines 267-381)

The backtest reconstructs L2 from L3 by tracking individual orders:

```python
# Track every order by ID
self.order_book_state: Dict[int, dict] = {}  # order_id → {price, size, side}

def _process_mbo_event(self, event):
    """Process MBO events to maintain order book state."""

    if action == 'ADD':
        # New order: store it and add to book
        self.order_book_state[order_id] = {
            'price': price,
            'size': size,
            'side': side
        }
        self._update_book(side, price, size, 'add')

    elif action == 'CANCEL':
        # Order cancelled: remove from book
        if order_id in self.order_book_state:
            old = self.order_book_state.pop(order_id)
            self._update_book(old['side'], old['price'], old['size'], 'remove')

    elif action == 'MODIFY':
        # Order modified: remove old, add new
        if order_id in self.order_book_state:
            old = self.order_book_state[order_id]
            self._update_book(old['side'], old['price'], old['size'], 'remove')
        self.order_book_state[order_id] = {'price': price, 'size': size, 'side': side}
        self._update_book(side, price, size, 'add')

    elif action == 'TRADE':
        # Trade: update tape analyzers, remove filled size
        self._on_trade(event)
```

**Aggregation into L2:**

```python
def _update_book(self, side: str, price: float, size: int, action: str):
    """Aggregate orders into L2 price levels."""
    book = self.books.get(self.symbol)
    target = book.bids if side == 'BID' else book.asks
    rounded_price = round(price, 2)

    if action == 'add':
        # Add size to existing level or create new
        target[rounded_price] = target.get(rounded_price, 0) + size
    elif action == 'remove':
        # Subtract size from level
        if rounded_price in target:
            target[rounded_price] -= size
            if target[rounded_price] <= 0:
                del target[rounded_price]
```

---

## L2 Metrics Calculation

**File:** `src/market_data.py` (lines 100-150)

### L2 Imbalance

Measures the ratio of bid volume to ask volume on the top 5 price levels:

```python
def get_imbalance(self) -> float:
    """Calculate bid/ask volume ratio."""
    sorted_bids = sorted(self.bids.items(), reverse=True)[:5]
    sorted_asks = sorted(self.asks.items())[:5]

    bid_vol = sum(size for _, size in sorted_bids)
    ask_vol = sum(size for _, size in sorted_asks)

    if ask_vol == 0:
        return 10.0 if bid_vol > 0 else 1.0
    return round(bid_vol / ask_vol, 2)
```

**Interpretation:**

| Imbalance | Label | Meaning |
|-----------|-------|---------|
| >= 1.5 | BULLISH | Bids overwhelm asks, buying pressure |
| <= 0.6 | BEARISH | Asks overwhelm bids, selling pressure |
| 0.6 - 1.5 | NEUTRAL | Balanced order flow |

### Spread

The difference between best ask and best bid:

```python
def get_spread(self) -> float:
    """Get current bid-ask spread."""
    if not self.bids or not self.asks:
        return 0.0
    best_bid = max(self.bids.keys())
    best_ask = min(self.asks.keys())
    return round(best_ask - best_bid, 2)
```

**Spread Limits (for entry validation):**
- Open Drive (9:30-9:45): Max $0.15
- Regular Session: Max $0.08

### DOM Walls

Detects significant orders that may act as support/resistance:

```python
def get_walls(self, last_price: float,
              threshold_minor: float = 3.0,
              threshold_major: float = 5.0) -> List[DOMWall]:
    """Detect significant orders (walls)."""
    all_sizes = list(self.bids.values()) + list(self.asks.values())
    avg_size = np.mean(all_sizes) if all_sizes else 0

    walls = []
    for price, size in self.bids.items():
        ratio = size / avg_size if avg_size > 0 else 0
        if ratio >= threshold_minor:
            tier = "MAJOR" if ratio >= threshold_major else "MINOR"
            distance = abs(price - last_price) / last_price * 100
            walls.append(DOMWall("BID", price, size, tier, round(distance, 2)))

    # Same for asks...

    return sorted(walls, key=lambda w: w.distance_pct)[:5]
```

**Wall Tiers:**
- **MAJOR**: Size >= 5x average (strong support/resistance)
- **MINOR**: Size >= 3x average (moderate support/resistance)

### Bid/Ask Stacks

Top N price levels for each side:

```python
def get_stack(self, side: str, levels: int = 3) -> List[List]:
    """Get top N levels for a side."""
    book = self.bids if side == "BID" else self.asks
    reverse = side == "BID"  # Bids sorted descending
    sorted_levels = sorted(book.items(), reverse=reverse)[:levels]
    return [[price, size] for price, size in sorted_levels]
```

**Example Output:**
```
BID_STACK: [[433.50, 250], [433.49, 180], [433.48, 320]]
ASK_STACK: [[433.52, 150], [433.53, 200], [433.54, 175]]
```

---

## Related Analyzers

**Initialization:** `src/market_data.py:714-721` & `src/data_replay.py:126-132`

For each symbol, the system initializes a suite of analyzers:

```python
self.books[symbol] = OrderBook(symbol)           # L2 order book
self.tapes[symbol] = TapeStream(symbol)          # Time & Sales analysis
self.footprints[symbol] = FootprintTracker(symbol)  # Volume by price/time
self.cvds[symbol] = CumulativeDelta(symbol)      # Cumulative Volume Delta
self.profiles[symbol] = VolumeProfile(symbol)   # Volume Profile (POC, VA)
self.absorbers[symbol] = AbsorptionDetector(symbol)  # Absorption detection
self.metrics[symbol] = MarketMetrics(symbol)    # HOD, LOD, VWAP, session
```

### TapeStream (Time & Sales)

**Purpose:** Analyze trade flow velocity and sentiment

| Method | Returns | Description |
|--------|---------|-------------|
| `get_velocity()` | (label, tps) | Trade velocity (SLOW/NORMAL/FAST/EXTREME) |
| `get_sentiment()` | str | Buyer/seller aggression (BULLISH/BEARISH/NEUTRAL) |
| `get_delta(seconds)` | float | Net buy-sell volume over N seconds |
| `get_large_prints()` | List | Recent large trades (>$50k notional) |

### CumulativeDelta (CVD)

**Purpose:** Track cumulative buy vs sell pressure

| Method | Returns | Description |
|--------|---------|-------------|
| `session_delta` | float | Net delta since session open |
| `get_trend()` | str | RISING/FALLING/FLAT |
| `get_slope_5m()` | float | 5-minute slope of CVD |

### VolumeProfile

**Purpose:** Track volume distribution by price

| Method | Returns | Description |
|--------|---------|-------------|
| `get_poc()` | float | Point of Control (highest volume price) |
| `get_value_area()` | (VAH, VAL) | Value Area High/Low (70% of volume) |
| `vwap` | float | Volume Weighted Average Price |

### AbsorptionDetector

**Purpose:** Detect institutional absorption (large orders absorbed with minimal price movement)

| Method | Returns | Description |
|--------|---------|-------------|
| `detect()` | Dict or None | Detected absorption event with side, volume, price |

### MarketMetrics

**Purpose:** Track session-level metrics

| Attribute | Type | Description |
|-----------|------|-------------|
| `hod` | float | High of Day |
| `lod` | float | Low of Day |
| `last_price` | float | Most recent trade price |
| `get_time_session()` | str | Current session phase |

**Session Phases:**
- `PRE_MARKET`: Before 9:30 ET
- `OPEN_DRIVE`: 9:30-9:45 ET (first 15 min)
- `MORNING`: 9:45-11:30 ET
- `MIDDAY`: 11:30-14:00 ET
- `AFTERNOON`: 14:00-15:30 ET
- `CLOSE`: 15:30-16:00 ET
- `AFTER_HOURS`: After 16:00 ET

---

## State Aggregation

**File:** `src/market_data.py` (lines 782-860)

The `get_full_state()` method aggregates all analyzers into a single dictionary:

```python
def get_full_state(self, symbol: str) -> Optional[Dict[str, Any]]:
    """Aggregate ALL data sources into the complete state."""

    book = self.books[symbol]
    tape = self.tapes[symbol]
    cvd = self.cvds[symbol]
    profile = self.profiles[symbol]
    metrics = self.metrics[symbol]
    absorber = self.absorbers[symbol]
    footprint = self.footprints[symbol]

    last_price = metrics.last_price
    vel_label, vel_tps = tape.get_velocity()
    vah, val = profile.get_value_area()
    absorption = absorber.detect()

    return {
        "MARKET_STATE": {
            # === L2 ORDER BOOK ===
            "L2_IMBALANCE": book.get_imbalance(),
            "SPREAD": book.get_spread(),
            "DOM_WALLS": [
                {
                    "side": w.side,
                    "price": w.price,
                    "size": w.size,
                    "tier": w.tier,
                    "distance_pct": w.distance_pct
                }
                for w in book.get_walls(last_price)
            ],
            "BID_STACK": book.get_stack("BID"),
            "ASK_STACK": book.get_stack("ASK"),

            # === TAPE (TIME & SALES) ===
            "TAPE_VELOCITY": vel_label,
            "TAPE_VELOCITY_TPS": vel_tps,
            "TAPE_SENTIMENT": tape.get_sentiment(),
            "TAPE_DELTA_1S": tape.get_delta(1),
            "TAPE_DELTA_5S": tape.get_delta(5),
            "LARGE_PRINTS_1M": tape.get_large_prints(60),

            # === CUMULATIVE DELTA ===
            "CVD_SESSION": cvd.session_delta,
            "CVD_TREND": cvd.get_trend(),
            "CVD_SLOPE_5M": cvd.get_slope_5m(),

            # === KEY LEVELS ===
            "HOD": metrics.hod,
            "LOD": metrics.lod,
            "VWAP": profile.vwap,
            "POC": profile.get_poc(),
            "VAH": vah,
            "VAL": val,

            # === PRICE ===
            "LAST": last_price,
            "BID": book.get_best_bid(),
            "ASK": book.get_best_ask(),

            # === SESSION ===
            "TIME_SESSION": metrics.get_time_session(),

            # === FOOTPRINT ===
            "FOOTPRINT_CURR_BAR": footprint.get_current_bar(),

            # === ABSORPTION ===
            "ABSORPTION_DETECTED": absorption,
        }
    }
```

---

## Context Building for LLM

**File:** `src/brain/context.py`

The `ContextBuilder` transforms raw state into human-readable markdown for the LLM.

### Build Method (lines 60-93)

```python
def build(self, market_state, account_state, history, active_orders) -> str:
    """Build complete context string for LLM."""
    sections = [
        self._build_position_section(account_state, active_orders),
        self._build_session_context(market_state),
        self._build_price_history(market_state),
        self._build_key_levels(market_state),
        self._build_order_book(market_state),     # L2 section
        self._build_tape_analysis(market_state),
        self._build_delta_flow(market_state),
        self._build_absorption(market_state),
        self._build_history(history),
        self._build_raw_data(market_state),
        self._build_closing_question(),
    ]
    return "\n\n".join(sections)
```

### Order Book Section (lines 333-389)

```python
def _build_order_book(self, market_state: Dict[str, Any]) -> str:
    """Build the order book (Level 2) section."""
    mkt = market_state.get("MARKET_STATE", market_state)

    l2_imbalance = mkt.get("L2_IMBALANCE", 1.0)
    spread = mkt.get("SPREAD", 0)
    bid_stack = mkt.get("BID_STACK", [])
    ask_stack = mkt.get("ASK_STACK", [])
    walls = mkt.get("DOM_WALLS", [])

    # Interpret imbalance
    if l2_imbalance >= 1.5:
        imbalance_label = "BULLISH - bids > asks"
    elif l2_imbalance <= 0.6:
        imbalance_label = "BEARISH - asks > bids"
    else:
        imbalance_label = "NEUTRAL"

    # Format stacks
    bid_lines = [f"│   ${p:.2f}: {s:,} shares" for p, s in bid_stack]
    ask_lines = [f"│   ${p:.2f}: {s:,} shares" for p, s in ask_stack]

    # Format walls
    wall_lines = []
    for w in walls:
        wall_lines.append(
            f"│   {w['tier']} {w['side']} wall: "
            f"${w['price']:.2f} ({w['size']:,} shares, {w['distance_pct']:.1f}% away)"
        )

    return f"""## ORDER BOOK (Level 2)

├── L2 Imbalance: {l2_imbalance:.1f} ({imbalance_label})
├── Spread: ${spread:.2f}
│
├── Bid Stack (top 3):
{chr(10).join(bid_lines)}
│
├── Ask Stack (top 3):
{chr(10).join(ask_lines)}
│
└── Walls:
{chr(10).join(wall_lines) if wall_lines else '│   No significant walls detected'}"""
```

### Example LLM Context Output

```markdown
## ORDER BOOK (Level 2)

├── L2 Imbalance: 1.8 (BULLISH - bids > asks)
├── Spread: $0.02
│
├── Bid Stack (top 3):
│   $433.50: 250 shares
│   $433.49: 180 shares
│   $433.48: 320 shares
│
├── Ask Stack (top 3):
│   $433.52: 150 shares
│   $433.53: 200 shares
│   $433.54: 175 shares
│
└── Walls:
│   MAJOR BID wall: $433.25 (2,500 shares, 0.06% away)
│   MINOR ASK wall: $434.00 (1,200 shares, 0.12% away)
```

---

## Main Event Loop

### Backtest Loop

**File:** `src/main.py` (lines 198-266)

```python
async def _on_backtest_state(self, state: Dict[str, Any]):
    """Called during backtest on each state update (every 5 sim seconds)."""

    # State already contains MARKET_STATE from get_full_state()
    market = state.get("MARKET_STATE", {})
    last_price = market.get("LAST", 0)
    sim_time = state.get("timestamp", 0)

    # Track price for context history
    record_price(last_price, datetime.fromtimestamp(sim_time), label)

    # Update executor (checks for stop/target fills)
    self.executor.update(last_price, sim_time)

    # Inject account state
    state["ACCOUNT_STATE"] = self.executor.get_account_state()
    state["ACTIVE_ORDERS"] = self.executor.get_active_orders()

    # Ask the Brain for a decision
    command = self.brain.think(state)  # L2 data consumed here

    # Execute the command
    spread = market.get("SPREAD", 0)
    result = self.executor.execute(command, current_spread=spread, context=market)

    # Log for reporting
    if hasattr(self, 'reporter'):
        self.reporter.log_decision(state, command, result)
```

### Live Loop

**File:** `src/main.py` (lines 302-346)

```python
async def _run_loop_live(self):
    """Live trading loop."""

    while self.running:
        # Get full state from IBKR connector
        state = self.connector.get_full_state(self.symbol)

        if state is None:
            await asyncio.sleep(LOOP_INTERVAL)
            continue

        # Extract price and update executor
        last_price = state.get("MARKET_STATE", {}).get("LAST", 0)
        self.executor.update(last_price, time.time())

        # Inject account state
        state["ACCOUNT_STATE"] = self.executor.get_account_state()
        state["ACTIVE_ORDERS"] = self.executor.get_active_orders()

        # Brain decision
        command = self.brain.think(state)

        # Execute
        spread = state.get("MARKET_STATE", {}).get("SPREAD", 0)
        result = self.executor.execute(command, current_spread=spread, context=state)

        await asyncio.sleep(LOOP_INTERVAL)
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA SOURCES                                    │
├─────────────────────────────────┬───────────────────────────────────────────┤
│         LIVE MODE               │            BACKTEST MODE                   │
│                                 │                                            │
│  IBKR TWS/Gateway               │  Databento MBO Files                       │
│  ├─ reqMktDepth (L2)            │  └─ BacktestData/TSLA-L3DATA/             │
│  ├─ reqTickByTickData (Tape)    │     └─ xnas-itch-YYYYMMDD.mbo.dbn.zst     │
│  └─ reqMktData (Snapshot)       │                                            │
│                                 │                                            │
│  Event-driven:                  │  Sequential replay:                        │
│  ├─ updateMktDepthEvent         │  ├─ ADD events                             │
│  ├─ tickByTickAllLastEvent      │  ├─ CANCEL events                          │
│  └─ pendingTickersEvent         │  ├─ MODIFY events                          │
│                                 │  └─ TRADE events                           │
└────────────────┬────────────────┴─────────────────┬─────────────────────────┘
                 │                                   │
                 ▼                                   ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                         CONNECTOR LAYER                                     │
│                                                                             │
│  IBKRConnector (market_data.py)    │    MBOReplayConnector (data_replay.py) │
│  ├─ _on_depth() → book.update()    │    ├─ _process_mbo_event()             │
│  ├─ _on_tick() → tape.on_trade()   │    ├─ _update_book()                   │
│  └─ _on_snapshot() → metrics       │    └─ _on_trade()                      │
└────────────────────────────────────┴───────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                      ANALYZERS (per symbol)                                 │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  OrderBook                    TapeStream                                    │
│  ├─ bids: Dict[float, int]    ├─ get_velocity() → (label, tps)             │
│  ├─ asks: Dict[float, int]    ├─ get_sentiment() → str                     │
│  ├─ get_imbalance() → float   ├─ get_delta(secs) → float                   │
│  ├─ get_spread() → float      └─ get_large_prints() → List                 │
│  ├─ get_walls() → List                                                      │
│  └─ get_stack() → List        CumulativeDelta                              │
│                               ├─ session_delta → float                      │
│  VolumeProfile                ├─ get_trend() → str                          │
│  ├─ vwap → float              └─ get_slope_5m() → float                     │
│  ├─ get_poc() → float                                                       │
│  └─ get_value_area() → tuple  AbsorptionDetector                           │
│                               └─ detect() → Dict or None                    │
│  MarketMetrics                                                              │
│  ├─ hod, lod → float          FootprintTracker                             │
│  ├─ last_price → float        └─ get_current_bar() → Dict                  │
│  └─ get_time_session() → str                                                │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                    STATE AGGREGATION                                        │
│                    get_full_state() → Dict                                  │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  {                                                                          │
│    "MARKET_STATE": {                                                        │
│      "L2_IMBALANCE": 1.8,              # OrderBook.get_imbalance()         │
│      "SPREAD": 0.02,                   # OrderBook.get_spread()            │
│      "DOM_WALLS": [...],               # OrderBook.get_walls()             │
│      "BID_STACK": [[433.50, 250],...], # OrderBook.get_stack("BID")        │
│      "ASK_STACK": [[433.52, 150],...], # OrderBook.get_stack("ASK")        │
│      "TAPE_VELOCITY": "FAST",          # TapeStream.get_velocity()         │
│      "TAPE_SENTIMENT": "BULLISH",      # TapeStream.get_sentiment()        │
│      "CVD_TREND": "RISING",            # CumulativeDelta.get_trend()       │
│      "HOD": 435.00,                    # MarketMetrics.hod                 │
│      "LOD": 430.50,                    # MarketMetrics.lod                 │
│      "VWAP": 433.25,                   # VolumeProfile.vwap                │
│      "LAST": 433.51,                   # MarketMetrics.last_price          │
│      ...                                                                    │
│    }                                                                        │
│  }                                                                          │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                    CONTEXT BUILDING                                         │
│                    ContextBuilder.build() → str                             │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Transforms Dict → Human-readable markdown:                                 │
│                                                                             │
│  ## CURRENT POSITION                                                        │
│  └── FLAT - No position                                                     │
│                                                                             │
│  ## SESSION CONTEXT                                                         │
│  ├── Phase: MORNING (9:45-11:30 ET)                                         │
│  └── Time: 10:15:32 ET                                                      │
│                                                                             │
│  ## ORDER BOOK (Level 2)                              ◄── L2 Section        │
│  ├── L2 Imbalance: 1.8 (BULLISH - bids > asks)                              │
│  ├── Spread: $0.02                                                          │
│  ├── Bid Stack: $433.50 (250), $433.49 (180), ...                           │
│  ├── Ask Stack: $433.52 (150), $433.53 (200), ...                           │
│  └── Walls: MAJOR BID @ $433.25 (2,500 shares)                              │
│                                                                             │
│  ## TAPE ANALYSIS                                                           │
│  ├── Velocity: FAST (45 tps)                                                │
│  └── Sentiment: BULLISH                                                     │
│                                                                             │
│  ... (other sections)                                                       │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                    TRADING BRAIN                                            │
│                    TradingBrain.think(state) → command                      │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. Build context string from state                                         │
│  2. Call LLM provider with:                                                 │
│     ├─ System prompt (trading rules)                                        │
│     ├─ User message (context string)                                        │
│     └─ Tools (enter_long, enter_short, wait, etc.)                          │
│  3. Parse LLM response (tool call)                                          │
│  4. Return command string:                                                  │
│     "ENTER_LONG|limit_price=433.50|stop_loss=433.20|..."                   │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                    EXECUTOR                                                 │
│                    Executor.execute(command) → result                       │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  SimulatedExecutor (backtest)    │    IBKRExecutor (live)                   │
│  ├─ submit_bracket_order()       │    ├─ submit_bracket_order()             │
│  ├─ modify_stop()                │    ├─ modify_stop()                      │
│  ├─ modify_target()              │    ├─ modify_target()                    │
│  ├─ exit_position()              │    ├─ exit_position()                    │
│  └─ update() [check fills]       │    └─ [IBKR callbacks for fills]         │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Live vs Backtest Comparison

| Aspect | Live Mode | Backtest Mode |
|--------|-----------|---------------|
| **Data Source** | IBKR TWS real-time | Databento MBO files |
| **Connector** | `IBKRConnector` | `MBOReplayConnector` |
| **Update Trigger** | Event-driven callbacks | Sequential event processing |
| **Timing** | Real wall-clock time | Simulated time (configurable speed) |
| **Order Book Build** | IBKR sends L2 deltas directly | Reconstruct L2 from L3 events |
| **State Update** | Continuous (~100ms loop) | Every 5 seconds of sim time |
| **Executor** | `IBKRExecutor` (real orders) | `SimulatedExecutor` (virtual) |
| **Fill Detection** | IBKR order status callbacks | Price-based simulation |

**Key Insight:** The L2 processing logic (`OrderBook`, `get_imbalance()`, etc.) is **identical** in both modes. Only the data source and executor differ.

---

## File Reference

| File | Purpose | Key Functions/Classes |
|------|---------|----------------------|
| `src/market_data.py` | Live data connector, analyzers | `IBKRConnector`, `OrderBook`, `TapeStream` |
| `src/data_replay.py` | Backtest data replay | `MBOReplayConnector`, `_process_mbo_event()` |
| `src/brain/context.py` | Context building for LLM | `ContextBuilder`, `_build_order_book()` |
| `src/brain/brain.py` | LLM decision making | `TradingBrain`, `think()` |
| `src/execution/simulated.py` | Virtual order execution | `SimulatedExecutor` |
| `src/execution/ibkr.py` | Real order execution | `IBKRExecutor` |
| `src/main.py` | Main event loops | `_on_backtest_state()`, `_run_loop_live()` |

---

## Appendix: MBO Event Structure (Databento)

```python
# Example MBO event from Databento
{
    "ts_event": 1698076200000000000,  # Nanosecond timestamp
    "action": "A",                    # A=Add, C=Cancel, M=Modify, T=Trade
    "side": "B",                      # B=Bid, A=Ask
    "price": 43350,                   # Price in fixed-point (÷100)
    "size": 100,                      # Order size
    "order_id": 123456789,            # Unique order ID
    "flags": 0,                       # Exchange-specific flags
}
```

**Action Codes:**
- `A` (65) = Add new order
- `C` (67) = Cancel order
- `M` (77) = Modify order
- `T` (84) = Trade execution
- `F` (70) = Fill (partial)

---

*Last updated: 2026-01-26*
