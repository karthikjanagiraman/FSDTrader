# FSDTrader: L2 Momentum Agent

> An autonomous trading agent inspired by Tesla FSD's "End-to-End" architecture.
> Uses **DeepSeek-R1-0528-Qwen3-8B** for local inference on Apple Silicon.

---

## 1. Trading Strategy: Order Flow Momentum

### Philosophy
We trade **Momentum**, not noise. We enter when:
1. **Order Flow Imbalance** shows clear buyer/seller dominance.
2. **Volume confirms** (RVOL > 1.5).
3. **Price is at a key level** (HOD/LOD breakout or S/R retest).

### Entry Rules (LONG)

| # | Condition | Metric | Threshold |
|---|-----------|--------|-----------|
| 1 | **Imbalance** | `L2_IMBALANCE` | > 1.5 (Buyers dominate) |
| 2 | **Volume** | `RVOL_DAY` | > 1.5 (Above average) |
| 3 | **Location** | `HOD_LOD_LOC` | `TESTING_HOD` or `BREAKOUT` |
| 4 | **Flow** | `TAPE_SENTIMENT` | `AGGRESSIVE_BUYING` |
| 5 | **No Wall** | `DOM_WALLS` | No resistance within 0.2% |
| 6 | **Delta Confirm** | `CVD_TREND` | `RISING` (Buyers in control) |

**ALL conditions must be true to enter.**

### Exit Rules

| Trigger | Action |
|---------|--------|
| `L2_IMBALANCE` drops below 0.7 | Exit immediately |
| `CVD_TREND` = `FALLING` while LONG | Exit (momentum fading) |
| Price hits `STOP_LOSS` | Auto-exit via bracket |
| Price hits `PROFIT_TARGET` | Auto-exit via bracket |
| Resistance wall appears ahead | `UPDATE_TARGET` |
| `ABSORPTION` detected at level | Trail stop tight |

---

## 2. The Enhanced State Vector

> [!IMPORTANT]
> This is the complete context sent to the LLM every 200ms (5Hz).

```json
{
  "MARKET_STATE": {
    "TICKER": "TSLA",
    "LAST": 245.50,
    "VWAP": 245.10,
    "TIME_SESSION": "OPEN_DRIVE",
    
    "// Level 2 (DOM) Analysis": "---",
    "L2_IMBALANCE": 2.1,
    "SPREAD": 0.02,
    "DOM_WALLS": [
      {"side": "ASK", "price": 247.00, "size": 3500, "tier": "MAJOR"},
      {"side": "BID", "price": 244.50, "size": 2100, "tier": "MINOR"}
    ],
    "BID_STACK": [[245.48, 150], [245.46, 200], [245.44, 180]],
    "ASK_STACK": [[245.50, 120], [245.52, 180], [245.54, 250]],
    
    "// Tape (Time & Sales) Analysis": "---",
    "TAPE_VELOCITY": "HIGH",
    "TAPE_VELOCITY_TPS": 42.5,
    "TAPE_SENTIMENT": "AGGRESSIVE_BUYING",
    "TAPE_DELTA_1S": 1250,
    "TAPE_DELTA_5S": 4800,
    "LARGE_PRINTS_1M": [
      {"price": 245.48, "size": 500, "side": "BUY", "secs_ago": 12}
    ],
    
    "// Footprint / Delta Analysis": "---",
    "FOOTPRINT_CURR_BAR": {
      "open": 245.30,
      "high": 245.55,
      "low": 245.25,
      "close": 245.50,
      "delta": 2800,
      "volume": 15000,
      "delta_pct": 18.7,
      "poc": 245.45,
      "imbalances": [
        {"price": 245.50, "bid_vol": 120, "ask_vol": 580, "ratio": 4.8}
      ]
    },
    "CVD_SESSION": 125000,
    "CVD_TREND": "RISING",
    "CVD_SLOPE_5M": 0.85,
    
    "// Volume Profile (Session)": "---",
    "VP_POC": 244.80,
    "VP_VAH": 246.20,
    "VP_VAL": 243.50,
    "VP_DEVELOPING_POC": 245.30,
    "PRICE_VS_POC": "ABOVE",
    
    "// Key Levels": "---",
    "HOD": 245.80,
    "LOD": 241.20,
    "HOD_LOD_LOC": "TESTING_HOD",
    "DISTANCE_TO_HOD_PCT": 0.12,
    "RVOL_DAY": 2.8,
    
    "// Absorption Detection": "---",
    "ABSORPTION_DETECTED": false,
    "ABSORPTION_SIDE": null,
    "ABSORPTION_PRICE": null
  },
  "ACCOUNT_STATE": {
    "POSITION": 0,
    "POSITION_SIDE": null,
    "AVG_ENTRY": null,
    "UNREALIZED_PL": 0.0,
    "DAILY_PL": 150.00,
    "DAILY_TRADES": 3
  },
  "ACTIVE_ORDERS": []
}
```

---

## 3. Field Definitions

### Level 2 (DOM) Fields

| Field | Type | Description |
|-------|------|-------------|
| `L2_IMBALANCE` | float | Bid Volume / Ask Volume (top 5 levels). >1.5 = Bullish |
| `SPREAD` | float | Ask[0] - Bid[0] in dollars |
| `DOM_WALLS` | array | Orders >3x average size. `tier`: MAJOR (>5x) or MINOR (>3x) |
| `BID_STACK` | array | Top 3 bid levels [price, size] |
| `ASK_STACK` | array | Top 3 ask levels [price, size] |

### Tape (Time & Sales) Fields

| Field | Type | Description |
|-------|------|-------------|
| `TAPE_VELOCITY` | string | LOW (<10 tps), MEDIUM (10-30), HIGH (>30) |
| `TAPE_VELOCITY_TPS` | float | Exact trades per second |
| `TAPE_SENTIMENT` | string | AGGRESSIVE_BUYING, AGGRESSIVE_SELLING, NEUTRAL |
| `TAPE_DELTA_1S` | int | Net delta over last 1 second (+ = buying) |
| `TAPE_DELTA_5S` | int | Net delta over last 5 seconds |
| `LARGE_PRINTS_1M` | array | Trades >300 shares in last 60 seconds |

### Footprint / Delta Fields

| Field | Type | Description |
|-------|------|-------------|
| `FOOTPRINT_CURR_BAR` | object | Current 1-min bar with delta, POC, imbalances |
| `delta` | int | Buy Volume - Sell Volume for bar |
| `delta_pct` | float | delta / volume * 100 |
| `poc` | float | Price level with most volume in bar |
| `imbalances` | array | Price levels with >3:1 bid/ask ratio |
| `CVD_SESSION` | int | Cumulative Delta since session open |
| `CVD_TREND` | string | RISING, FALLING, FLAT (based on 5 bar slope) |
| `CVD_SLOPE_5M` | float | -1 to +1 normalized slope |

### Volume Profile Fields

| Field | Type | Description |
|-------|------|-------------|
| `VP_POC` | float | Point of Control (most volume price) |
| `VP_VAH` | float | Value Area High (70% volume top) |
| `VP_VAL` | float | Value Area Low (70% volume bottom) |
| `VP_DEVELOPING_POC` | float | POC of current session (moves) |
| `PRICE_VS_POC` | string | ABOVE, BELOW, AT_POC |

### Key Levels Fields

| Field | Type | Description |
|-------|------|-------------|
| `HOD` | float | High of Day |
| `LOD` | float | Low of Day |
| `HOD_LOD_LOC` | string | TESTING_HOD, TESTING_LOD, NEAR_HOD, NEAR_LOD, MID_RANGE |
| `DISTANCE_TO_HOD_PCT` | float | % distance to HOD |
| `RVOL_DAY` | float | Relative Volume vs 30-day average |

### Absorption Fields

| Field | Type | Description |
|-------|------|-------------|
| `ABSORPTION_DETECTED` | bool | True if absorption pattern detected |
| `ABSORPTION_SIDE` | string | BID (support) or ASK (resistance) |
| `ABSORPTION_PRICE` | float | Price level where absorption is occurring |

---

## 4. The Complete System Prompt

```text
<system>
ROLE: You are FSDTrader, an autonomous execution algorithm.
OBJECTIVE: Capture momentum moves of $0.50-$2.00 over 1-5 minutes.

ENTRY RULES (LONG):
1. L2_IMBALANCE > 1.5
2. RVOL_DAY > 1.5
3. HOD_LOD_LOC = "TESTING_HOD" or "NEAR_HOD" or "BREAKOUT"
4. TAPE_SENTIMENT = "AGGRESSIVE_BUYING"
5. No wall in DOM_WALLS with side="ASK" within 0.20% of LAST
6. CVD_TREND = "RISING" (Delta confirms direction)
7. FOOTPRINT_CURR_BAR.delta > 0 (Current bar is bullish)

ENTRY RULES (SHORT):
1. L2_IMBALANCE < 0.6
2. RVOL_DAY > 1.5
3. HOD_LOD_LOC = "TESTING_LOD" or "NEAR_LOD"
4. TAPE_SENTIMENT = "AGGRESSIVE_SELLING"
5. No wall in DOM_WALLS with side="BID" within 0.20%
6. CVD_TREND = "FALLING"
7. FOOTPRINT_CURR_BAR.delta < 0

EXIT RULES:
- If L2_IMBALANCE reverses past 0.7/1.3: EXIT immediately.
- If CVD_TREND reverses (LONG: becomes FALLING): EXIT.
- If ABSORPTION_DETECTED at your target: Tighten stop.
- If a DOM_WALL appears in your direction: UPDATE_TARGET below it.

EDGE CASES:
- If price is ABOVE VP_VAH and testing HOD: Strong breakout setup.
- If price is BELOW VP_VAL and testing LOD: Strong breakdown setup.
- If ABSORPTION_DETECTED on BID with bullish CVD: Support confirmed, LONG.
- If LARGE_PRINTS_1M shows heavy buying: Institutional participation.

RISK RULES:
- Max risk per trade: $0.20 (Stop distance)
- Max profit target: $1.00 or next wall/level
- Never chase: If DISTANCE_TO_HOD_PCT > 0.5% and rising, WAIT for pullback.

OUTPUT FORMAT:
Respond with EXACTLY ONE DSL command:
- ENTER_LONG(limit_price, stop_loss, profit_target)
- ENTER_SHORT(limit_price, stop_loss, profit_target)
- UPDATE_STOP(new_price)
- UPDATE_TARGET(new_price)
- CANCEL_ALL()
- WAIT(reason)
</system>

<context>
HISTORY (Last 3 actions):
{history}

CURRENT MARKET STATE:
{market_state_json}

ACCOUNT STATE:
{account_state_json}
</context>

<instruction>
Analyze the state. Check ALL entry conditions. Output your command.
</instruction>
```

---

## 5. Output: DSL Command Examples

### Scenario A: Full Confluence Entry
**State**: Imbalance 2.1, CVD Rising, Testing HOD, Aggressive Buying, Wall at 247.

```
<think>
Entry checklist:
1. L2_IMBALANCE = 2.1 > 1.5 ✓
2. RVOL_DAY = 2.8 > 1.5 ✓
3. HOD_LOD_LOC = TESTING_HOD ✓
4. TAPE_SENTIMENT = AGGRESSIVE_BUYING ✓
5. Wall at 247.00 is 0.61% away > 0.20% ✓
6. CVD_TREND = RISING ✓
7. FOOTPRINT delta = 2800 > 0 ✓

Full confluence. Enter now.
Stop: 245.30 (20c risk). Target: 246.80 (below wall).
</think>
ENTER_LONG(245.50, 245.30, 246.80)
```

---

### Scenario B: CVD Divergence - No Entry
**State**: Imbalance 1.6, but CVD Falling.

```
<think>
1. L2_IMBALANCE = 1.6 > 1.5 ✓
2. CVD_TREND = FALLING ✗

CVD is falling despite bullish imbalance.
This is divergence - smart money selling into bids.
No entry.
</think>
WAIT("CVD divergence - trend not confirmed")
```

---

### Scenario C: Absorption Confirmed Support
**State**: Price at 244.50, Absorption on BID, CVD Rising.

```
<think>
ABSORPTION_DETECTED = true at BID 244.50.
Large orders absorbing selling pressure.
CVD_TREND = RISING confirms buyers in control.
This is a reversal setup.
</think>
ENTER_LONG(244.55, 244.30, 245.50)
```

---

## 6. Safety Controls

| Control | Rule | Enforcement |
|---------|------|-------------|
| Max Daily Loss | Halt if P&L < -$500 | `main.py` |
| Max Position | 100 shares | `execution.py` |
| Max Trades/Day | 10 trades | `main.py` |
| Slippage Guard | Reject if SPREAD > $0.05 | `brain.py` |
| Parse Failure | Default to WAIT | `brain.py` |

---

## 7. File Structure

```
FSDTrader/
├── README.md
├── requirements.txt
├── src/
│   ├── main.py          # 5Hz Loop
│   ├── market_data.py   # IBKR + All Analytics
│   ├── brain.py         # DeepSeek + Parser
│   └── execution.py     # Order Manager
├── config/
│   └── settings.py
└── data/logs/
```

---

## 8. Quickstart

```bash
pip install -r requirements.txt
ollama pull deepseek-r1:8b
# Start TWS on port 7497
python src/main.py --symbol TSLA
```
