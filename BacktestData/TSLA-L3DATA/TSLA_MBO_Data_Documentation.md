# TSLA Market-by-Order (MBO) Data Documentation

## Overview

This dataset contains **Level 3 / Market-by-Order (MBO)** data for Tesla (TSLA) from the NASDAQ exchange, sourced from Databento. Each file represents one trading day and captures every individual order book event with nanosecond precision.

**Data Source:** Databento XNAS.ITCH feed
**Symbol:** TSLA (Instrument ID: 16244)
**Date Range:** October 15, 2025 - November 14, 2025 (22 trading days)
**Format:** DBN (Databento Binary) with Zstandard compression (.dbn.zst)

---

## File Structure

### Naming Convention
```
xnas-itch-YYYYMMDD.mbo.dbn.zst
```
- `xnas` = NASDAQ exchange
- `itch` = ITCH protocol (NASDAQ's native data feed)
- `YYYYMMDD` = Trading date
- `mbo` = Market-by-Order schema
- `dbn.zst` = Databento format with Zstandard compression

### File Sizes
Each daily file is approximately **90-180 MB compressed**, containing **8-12 million order events** per day.

---

## Data Fields

Each record in an MBO file contains these fields:

| Field | Type | Description |
|-------|------|-------------|
| `ts_recv` | datetime64[ns, UTC] | Timestamp when the message was received by the exchange gateway |
| `ts_event` | datetime64[ns, UTC] | Timestamp of the actual market event |
| `rtype` | uint8 | Record type (160 = MBO message) |
| `publisher_id` | uint16 | Data publisher identifier (2 = NASDAQ) |
| `instrument_id` | uint32 | Unique instrument identifier (16244 for TSLA) |
| `action` | char | Order action type (see below) |
| `side` | char | Order side: B=Bid, A=Ask, N=None |
| `price` | float64 | Order price in USD |
| `size` | uint32 | Order quantity (shares) |
| `channel_id` | uint8 | Feed channel identifier |
| `order_id` | uint64 | Unique order identifier |
| `flags` | uint8 | Bit flags (see below) |
| `ts_in_delta` | int32 | Nanoseconds between ts_event and ts_recv |
| `sequence` | uint32 | Message sequence number |
| `symbol` | string | Symbol name (TSLA) |

---

## Action Types

The `action` field indicates what happened to an order:

| Action | Name | Description |
|--------|------|-------------|
| `A` | **Add** | New order added to the book |
| `C` | **Cancel** | Order removed from the book (cancelled or fully filled) |
| `T` | **Trade** | Non-displayed trade execution (no order_id) |
| `F` | **Fill** | Partial fill of a displayed order |
| `M` | **Modify** | Order modified (price or size change) |
| `R` | **Reset** | Order book clear/reset (system event) |

### Typical Distribution (per day)
```
Cancel (C):  ~3.8 million  (47%)
Add (A):     ~3.8 million  (47%)
Trade (T):   ~264,000      (3.3%)
Fill (F):    ~168,000      (2.1%)
Reset (R):   1             (0.01%)
```

---

## Side Values

| Side | Meaning |
|------|---------|
| `B` | Bid (buy order) |
| `A` | Ask (sell order) |
| `N` | None (used for trades, resets, or neutral events) |

---

## Flags Field

The `flags` field is a bitmask:

| Value | Meaning |
|-------|---------|
| 0 | Standard message |
| 8 | System event (e.g., book reset) |
| 128 | Last message in packet / timing boundary |

---

## Trading Sessions

The data covers the full NASDAQ trading day in UTC:

| Session | UTC Time | Eastern Time |
|---------|----------|--------------|
| Pre-market | 08:00-13:30 UTC | 4:00-9:30 AM ET |
| Regular Hours | 13:30-20:00 UTC | 9:30 AM-4:00 PM ET |
| After-hours | 20:00-24:00 UTC | 4:00-8:00 PM ET |

**Note:** ~96% of order activity occurs during regular market hours.

---

## Real Examples from October 15, 2025

### Example 1: Book Reset (Start of Day)
```
ts_event:      2025-10-15 07:05:01.987632623 UTC
action:        R (Reset)
side:          N (None)
price:         NaN
size:          0
order_id:      0
flags:         8 (System event)
```
*This clears any stale orders from the book before pre-market begins.*

### Example 2: New Order Added
```
ts_event:      2025-10-15 08:00:00.007504145 UTC
action:        A (Add)
side:          B (Bid)
price:         433.29
size:          100
order_id:      1504
flags:         128
```
*A buy order for 100 shares at $433.29 is placed on the book.*

### Example 3: Order Cancelled
```
ts_event:      2025-10-15 08:00:00.078813474 UTC
action:        C (Cancel)
side:          B (Bid)
price:         433.29
size:          100
order_id:      1504
flags:         0
```
*The same order (ID 1504) is cancelled 71 milliseconds later.*

### Example 4: Trade Execution (Non-Book)
```
ts_event:      2025-10-15 08:00:08.490989556 UTC
action:        T (Trade)
side:          N (None)
price:         433.88
size:          40
order_id:      0
flags:         0
```
*A trade of 40 shares executes at $433.88. No order_id because this is a crossing/internal execution.*

### Example 5: Partial Fill of Book Order
```
ts_event:      2025-10-15 08:00:11.557562542 UTC
action:        F (Fill)
side:          B (Bid)
price:         433.60
size:          1
order_id:      248064
flags:         0
```
*Order 248064 receives a partial fill of 1 share at $433.60.*

### Example 6: Complete Order Lifecycle
This shows order #248064 from creation to completion:
```
08:00:10.387 UTC | A (Add)    | Bid | $433.60 | 1 share  | Created
08:00:11.557 UTC | F (Fill)   | Bid | $433.60 | 1 share  | Filled
08:00:11.557 UTC | C (Cancel) | Bid | $433.60 | 1 share  | Removed from book
```
*Order placed, immediately filled 1.17 seconds later, then removed.*

---

## How to Read the Data (Python)

```python
import databento as db

# Load one day's data
data = db.DBNStore.from_file("xnas-itch-20251015.mbo.dbn.zst")

# Convert to DataFrame
df = data.to_df()

# Filter for trades only
trades = df[df['action'] == 'T']

# Filter for regular market hours (9:30 AM - 4:00 PM ET)
regular = df[(df['ts_event'].dt.hour >= 13) & (df['ts_event'].dt.hour < 20)]

# Calculate VWAP
trades_vwap = (trades['price'] * trades['size']).sum() / trades['size'].sum()
```

---

## Companion Files

| File | Purpose |
|------|---------|
| `metadata.json` | Query parameters and job details |
| `symbology.json` | Symbol-to-instrument ID mappings |
| `condition.json` | Data availability status by date |
| `manifest.json` | File list with checksums and URLs |

---

## Use Cases

1. **Order Book Reconstruction** - Replay Add/Cancel/Modify events to rebuild the full order book at any point in time
2. **Trade Analytics** - Analyze execution patterns, trade sizes, and timing
3. **Market Microstructure** - Study bid-ask spreads, queue positions, order flow imbalance
4. **Latency Analysis** - Use ts_in_delta to measure exchange processing latency
5. **HFT Research** - Nanosecond precision enables high-frequency strategy analysis

---

## Statistics for October 15, 2025

| Metric | Value |
|--------|-------|
| Total Events | 8,097,521 |
| Add Orders | 3,823,399 |
| Cancel Events | 3,842,205 |
| Trades | 263,597 |
| Partial Fills | 168,319 |
| Price Range | $427.41 - $440.12 |
| Average Order Size | 73 shares |
| Data Start | 07:05:01 UTC |
| Data End | 23:59:59 UTC |
