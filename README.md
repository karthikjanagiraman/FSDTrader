# FSDTrader: L2 Momentum Trading Agent

> An autonomous TSLA trading system using LLM-based decision making with real-time order flow analysis.

**Version:** 3.2.0

---

## Overview

FSDTrader is a momentum/scalping system that uses Large Language Models (LLMs) with native tool calling to make real-time trading decisions. The system analyzes Level 2 market depth, time & sales (tape), cumulative delta, and volume profile to identify short-term momentum opportunities.

### Key Features

- **Order Flow Analysis**: Real-time L2 market data processing with 7 specialized analyzers
- **LLM Decision Engine**: Native function calling for structured trade decisions (Anthropic, Groq, Grok)
- **Memo System**: Persistent LLM context via self-notes (@INIT/@DELTA chain)
- **Bracket Order Execution**: Automated stop loss and profit target management
- **Multi-Mode Operation**: Backtest, simulation, paper, and live trading
- **Data Parity**: Identical data structures across all modes

### Trading Style

| Attribute | Value |
|-----------|-------|
| Type | Momentum / Scalping |
| Target Moves | $0.50 - $2.00 |
| Duration | 1-5 minutes |
| Max Position | 100 shares |
| Risk/Reward | Minimum 1:2 |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      FSDTrader Main                          │
│                      (Orchestrator)                          │
└─────────────────────────────────────────────────────────────┘
                              │
      ┌───────────────────────┼───────────────────────┐
      ▼                       ▼                       ▼
┌─────────────┐      ┌─────────────────┐      ┌─────────────┐
│ MarketData  │      │  TradingBrain   │      │  Executor   │
│  Provider   │      │  (LLM Engine)   │      │  Provider   │
└─────────────┘      └─────────────────┘      └─────────────┘
      │                      │                       │
 ┌────┴────┐           ┌─────┴─────┐          ┌──────┴──────┐
 ▼         ▼           ▼           ▼          ▼             ▼
IBKR    MBP-10      Anthropic  Groq/Grok  Simulated      IBKR
Connector  Replay   Provider   Providers   Executor     Executor
```

### Modules

| Module | Description |
|--------|-------------|
| `main.py` | Orchestration, mode selection, main trading loop |
| `market_data.py` | Order book, tape, CVD, volume profile, absorption detection |
| `data_replay.py` | MBP-10 file replay for backtesting with Databento data |
| `brain/` | LLM interaction, prompt engineering, tool definitions, memo system |
| `execution/` | Order management, position tracking, P&L calculation |
| `reporting/` | Backtest reports and trade analysis |

---

## Market Data Analysis

The system maintains 7 real-time analysis components:

| Component | Purpose | Key Metrics |
|-----------|---------|-------------|
| **OrderBook** | Level 2 DOM analysis | L2_IMBALANCE, SPREAD, DOM_WALLS |
| **TapeStream** | Time & Sales analysis | TAPE_VELOCITY, TAPE_SENTIMENT, DELTA |
| **FootprintTracker** | Per-bar delta analysis | Delta, POC, Imbalances |
| **CumulativeDelta** | Session CVD tracking | CVD_SESSION, CVD_TREND, CVD_SLOPE |
| **VolumeProfile** | Volume at price | VP_POC, VP_VAH, VP_VAL |
| **AbsorptionDetector** | Large order absorption | ABSORPTION_DETECTED, SIDE, PRICE |
| **MarketMetrics** | Session metrics | HOD, LOD, VWAP, RVOL |

---

## Trading Strategy

### Entry Criteria (LONG)

1. **Trend Alignment**: CVD rising, price above VWAP
2. **L2 Imbalance**: Buyers in control (ratio > 1.5)
3. **Tape Confirmation**: Aggressive buying, MEDIUM+ velocity
4. **Signal Persistence**: Pattern held for 2+ snapshots
5. **Stop Placement**: Clear support level for stop loss
6. **Risk/Reward**: Minimum 1:2

### Entry Criteria (SHORT)

1. **Trend Alignment**: CVD falling, price below VWAP
2. **L2 Imbalance**: Sellers in control (ratio < 0.6)
3. **Tape Confirmation**: Aggressive selling, MEDIUM+ velocity
4. **Signal Persistence**: Pattern held for 2+ snapshots
5. **Stop Placement**: Clear resistance level for stop loss
6. **Risk/Reward**: Minimum 1:2

### Risk Limits

| Limit | Value | Enforcement |
|-------|-------|-------------|
| Max Position Size | 100 shares | Order submission |
| Max Daily Loss | -$500 | Main loop halts trading |
| Max Daily Trades | 10 | Order submission blocked |
| Max Spread | $0.15 (open), $0.08 (other) | Entry rejected |
| Stop Distance | $0.10 - $0.30 | Order validation |

---

## Installation

### Prerequisites

- Python 3.10+
- Interactive Brokers TWS or Gateway (for live/paper trading)
- API key for LLM provider (Grok or OpenAI)

### Setup

```bash
# Clone the repository
git clone https://github.com/karthikjanagiraman/FSDTrader.git
cd FSDTrader

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Set API key
export XAI_API_KEY="your-grok-api-key"
# or
export OPENAI_API_KEY="your-openai-api-key"
```

### Requirements

```
ib_insync>=0.9.86      # IBKR TWS API wrapper
databento>=0.39.0      # MBO data replay
numpy>=1.24.0          # Numerical operations
colorama>=0.4.6        # Console colors
httpx>=0.27.0          # HTTP client for LLM APIs
pytest>=8.0.0          # Testing
```

---

## Usage

### Operating Modes

| Mode | Command | Data Source | Executor |
|------|---------|-------------|----------|
| **Backtest** | `--backtest` | MBP-10 Replay | Simulated |
| **Simulation** | `--sim` | Mock Data | Simulated |
| **Paper** | (default) | IBKR (7497) | IBKR |
| **Live** | `--live` | IBKR (7496) | IBKR |

### Examples

```bash
# Backtest with real L2 data
python src/main.py --backtest --date 20260112 --speed 100

# Simulation with mock data
python src/main.py --sim

# Paper trading (requires TWS on port 7497)
python src/main.py --symbol TSLA

# Live trading (CAUTION - real money!)
python src/main.py --live --symbol TSLA

# Use different LLM provider
python src/main.py --provider anthropic --model claude-3-haiku-20240307
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--symbol` | TSLA | Trading symbol |
| `--backtest` | - | Enable backtest mode |
| `--date` | latest | Backtest date (YYYYMMDD) |
| `--speed` | 100 | Backtest speed multiplier |
| `--sim` | - | Enable simulation mode |
| `--live` | - | Enable live trading |
| `--provider` | groq | LLM provider (anthropic, groq, grok) |
| `--model` | - | Model override |

---

## Project Structure

```
FSDTrader/
├── src/
│   ├── main.py              # Main orchestrator
│   ├── market_data.py       # 7 order flow analyzers + IBKR connector
│   ├── data_replay.py       # MBP-10 file replay for backtesting
│   ├── brain/
│   │   ├── brain.py         # TradingBrain (LLM decision engine)
│   │   ├── prompts.py       # System prompt (v3.3.0)
│   │   ├── tools.py         # 6 tool definitions
│   │   ├── context.py       # State -> LLM context builder
│   │   ├── memo.py          # MemoManager (persistent LLM context)
│   │   ├── validation.py    # Pre-flight tool call validation
│   │   └── providers/
│   │       ├── base.py      # LLMProvider ABC
│   │       ├── anthropic.py # Anthropic Claude provider
│   │       ├── grok.py      # xAI Grok provider
│   │       └── groq.py      # Groq provider
│   ├── execution/
│   │   ├── base.py          # ExecutionProvider ABC
│   │   ├── types.py         # Position, BracketOrder, TradeRecord
│   │   ├── simulated.py     # SimulatedExecutor (backtest/sim)
│   │   └── ibkr.py          # IBKRExecutor (live/paper)
│   └── reporting/
│       └── backtest.py      # Report generation
├── tests/
│   └── test_execution.py    # Unit tests
├── docs/
│   └── REQUIREMENTS.md      # Comprehensive requirements doc (v3.2.0)
├── BacktestData/
│   └── TSLA-L2DATA/         # MBP-10 data files (Git LFS)
└── data/logs/               # Trading logs
```

---

## LLM Tools

The Brain module provides 6 tools via native function calling. All tools require `reasoning` and `memo` fields.

| Tool | Description | Arguments |
|------|-------------|-----------|
| `enter_long` | Enter long position with bracket | limit_price, stop_loss, profit_target, qty, conviction, reasoning, memo |
| `enter_short` | Enter short position with bracket | limit_price, stop_loss, profit_target, qty, conviction, reasoning, memo |
| `update_stop` | Modify stop loss | new_price, reasoning, memo |
| `update_target` | Modify profit target | new_price, reasoning, memo |
| `exit_position` | Exit at market | reasoning, memo |
| `wait` | No action | reasoning, memo |

### Memo System

The LLM maintains persistent context via self-notes across trading cycles:
- `@INIT` - Initial session assessment (first decision)
- `@DELTA` - What changed since last snapshot (subsequent decisions)
- Format: `@DELTA|[time]|[position]|same:[unchanged]|changed:[what's new]|thesis:"[market read]"|watching:[triggers]`

---

## Data Schemas

### MARKET_STATE (31 fields)

```python
{
    "TICKER": "TSLA",
    "LAST": 245.50,
    "VWAP": 245.10,
    "MARKET_TIME": "09:35:15",  # ET market time
    "TIME_SESSION": "MORNING",

    # Level 2
    "L2_IMBALANCE": 1.8,
    "SPREAD": 0.02,
    "DOM_WALLS": [...],
    "BID_STACK": [[245.48, 150], ...],
    "ASK_STACK": [[245.50, 120], ...],

    # Tape
    "TAPE_VELOCITY": "HIGH",
    "TAPE_SENTIMENT": "AGGRESSIVE_BUYING",
    "TAPE_DELTA_5S": 2500,

    # CVD
    "CVD_SESSION": 125000,
    "CVD_TREND": "RISING",

    # Volume Profile
    "VP_POC": 244.80,
    "VP_VAH": 246.20,
    "VP_VAL": 243.50,

    # Key Levels
    "HOD": 246.00,
    "LOD": 242.00,
    "RVOL_DAY": 1.8,

    # Absorption
    "ABSORPTION_DETECTED": false
}
```

### ACCOUNT_STATE (8 fields)

```python
{
    "POSITION": 100,
    "POSITION_SIDE": "LONG",
    "AVG_ENTRY": 245.50,
    "UNREALIZED_PL": 25.00,
    "DAILY_PL": 150.00,
    "DAILY_TRADES": 3,
    "BUYING_POWER": 100000.00,
    "DAILY_LOSS_REMAINING": 350.00
}
```

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run execution module tests
python -m pytest tests/test_execution.py -v

# Current: 69 tests, all passing
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Anthropic API key (for Claude models) |
| `XAI_API_KEY` or `GROK_API_KEY` | Grok API key |
| `GROQ_API_KEY` | Groq API key |
| `IBKR_PORT` | TWS port (default: 7497) |

---

## Documentation

- [Comprehensive Requirements](docs/REQUIREMENTS.md) - Full system specification (v3.2.0)

---

## Safety Notes

- **Paper trade first**: Always test with paper trading before going live
- **Risk limits**: Built-in limits prevent catastrophic losses
- **API costs**: LLM calls incur costs - monitor usage
- **Market hours**: System designed for regular trading hours (9:30-16:00 ET)

---

## License

Private repository. All rights reserved.

---

## Contributing

This is a private project. Contact the repository owner for collaboration.
