#!/usr/bin/env python3
"""
FSDTrader: Enhanced Market Data Module
Professional-grade Order Flow Analysis for LLM Context

Features:
- Level 2 (DOM) with Wall Detection
- Tape Analysis with Delta Windows
- Footprint (per-bar delta + imbalances)
- Cumulative Delta (CVD)
- Volume Profile (POC/VAH/VAL)
- Absorption Detection
"""
import logging
import asyncio
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import time

try:
    from ib_insync import IB, Stock, Crypto, util
except ImportError:
    IB = None
    Stock = None
    Crypto = None

# Known crypto symbols for contract type detection
CRYPTO_SYMBOLS = {"BTC", "ETH", "LTC", "BCH"}

import numpy as np

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Trade:
    """Single trade from tape."""
    price: float
    size: float  # Float for crypto fractional sizes
    time: float  # Unix timestamp (simulated or real)
    side: str    # "BUY" or "SELL"

@dataclass
class FootprintBar:
    """One-minute bar with delta analysis."""
    open: float = 0.0
    high: float = 0.0
    low: float = float('inf')
    close: float = 0.0
    volume: float = 0.0  # Float for crypto
    buy_volume: float = 0.0  # Float for crypto
    sell_volume: float = 0.0  # Float for crypto
    delta: float = 0.0  # buy_volume - sell_volume (float for crypto)
    delta_pct: float = 0.0  # delta / volume * 100
    poc: float = 0.0        # Price with most volume
    volume_at_price: Dict[float, float] = field(default_factory=dict)  # Float for crypto
    imbalances: List[Dict] = field(default_factory=list)

@dataclass
class DOMWall:
    """Significant order in the book."""
    side: str      # "BID" or "ASK"
    price: float
    size: float    # Float for crypto fractional sizes
    tier: str      # "MASSIVE" (>p95), "MAJOR" (>p90), or "MINOR" (>p75)
    distance_pct: float
    percentile: float = 0.0  # Which percentile this size is at

@dataclass
class LargePrint:
    """Large trade from tape."""
    price: float
    size: float  # Float for crypto fractional sizes
    side: str
    secs_ago: float

# =============================================================================
# ORDER BOOK (Level 2 / DOM)
# =============================================================================

class OrderBook:
    """
    Real-time Level 2 Order Book with Smart Wall Detection.

    Key features:
    - Cumulative sizes at each price level (aggregates multiple market makers)
    - Percentile-based wall detection (adapts to current market conditions)
    - Dynamic stack depth (shows levels up to first major wall)
    - Full-book imbalance calculation
    """
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.bids: Dict[float, int] = {}  # Price -> Cumulative Size
        self.asks: Dict[float, int] = {}
        self.logger = logging.getLogger(f"DOM_{ticker}")

        # Cache for percentile thresholds (recalculated on each update)
        self._p75: float = 0
        self._p90: float = 0
        self._p95: float = 0

    def update(self, dom_event):
        """Handle IBKR updateMktDepth event (legacy incremental)."""
        target = self.bids if dom_event.side == 1 else self.asks
        price = round(dom_event.price, 2)
        size = dom_event.size
        op = dom_event.operation

        if op in (0, 1):  # Insert or Update
            target[price] = size
        elif op == 2:     # Delete
            target.pop(price, None)

        self._update_percentiles()

    def update_from_dom_levels(self, dom_bids, dom_asks):
        """
        Update book from Ticker.domBids / Ticker.domAsks snapshots.
        Each is a list of DOMLevel(price, size, marketMaker).

        IMPORTANT: Multiple market makers can have orders at the same price.
        We ACCUMULATE sizes to show total liquidity at each level.
        This matches industry standard (what traders see on screens).
        """
        self.bids.clear()
        self.asks.clear()

        # Accumulate sizes from multiple market makers at same price
        # Use float for crypto fractional sizes, keep as-is for stocks
        for level in (dom_bids or []):
            price = round(level.price, 2)
            self.bids[price] = self.bids.get(price, 0) + level.size
        for level in (dom_asks or []):
            price = round(level.price, 2)
            self.asks[price] = self.asks.get(price, 0) + level.size

        self._update_percentiles()

    def _update_percentiles(self):
        """Calculate percentile thresholds for smart wall detection."""
        all_sizes = list(self.bids.values()) + list(self.asks.values())
        if len(all_sizes) < 5:
            self._p75 = self._p90 = self._p95 = 0
            return

        self._p75 = float(np.percentile(all_sizes, 75))
        self._p90 = float(np.percentile(all_sizes, 90))
        self._p95 = float(np.percentile(all_sizes, 95))

    def _get_wall_tier(self, size: float) -> Optional[str]:
        """Determine wall tier based on percentile thresholds."""
        if self._p95 > 0 and size >= self._p95:
            return "MASSIVE"
        elif self._p90 > 0 and size >= self._p90:
            return "MAJOR"
        elif self._p75 > 0 and size >= self._p75:
            return "MINOR"
        return None

    def _get_size_percentile(self, size: float) -> float:
        """Get approximate percentile for a size value."""
        all_sizes = list(self.bids.values()) + list(self.asks.values())
        if not all_sizes:
            return 0.0
        below = sum(1 for s in all_sizes if s < size)
        return round(below / len(all_sizes) * 100, 1)

    def get_imbalance(self) -> float:
        """
        Calculate bid/ask volume ratio using ALL levels.

        This gives a complete picture of order book pressure,
        not just the top few levels.
        """
        bid_vol = sum(self.bids.values())
        ask_vol = sum(self.asks.values())

        if ask_vol == 0:
            return 10.0 if bid_vol > 0 else 1.0
        return round(bid_vol / ask_vol, 2)

    def get_spread(self) -> float:
        """Get current spread."""
        if not self.bids or not self.asks:
            return 0.0
        best_bid = max(self.bids.keys())
        best_ask = min(self.asks.keys())
        return round(best_ask - best_bid, 2)

    def get_walls(self, last_price: float) -> List[DOMWall]:
        """
        Detect significant orders (walls) using percentile-based thresholds.

        Tiers:
        - MASSIVE: >= 95th percentile (very rare, significant barrier)
        - MAJOR: >= 90th percentile (notable resistance/support)
        - MINOR: >= 75th percentile (worth noting)

        Returns walls sorted by distance from current price, max 10.
        """
        walls = []

        if self._p75 == 0:
            return walls

        for price, size in self.bids.items():
            tier = self._get_wall_tier(size)
            if tier:
                dist = abs(price - last_price) / last_price * 100
                pct = self._get_size_percentile(size)
                walls.append(DOMWall("BID", price, size, tier, round(dist, 2), pct))

        for price, size in self.asks.items():
            tier = self._get_wall_tier(size)
            if tier:
                dist = abs(price - last_price) / last_price * 100
                pct = self._get_size_percentile(size)
                walls.append(DOMWall("ASK", price, size, tier, round(dist, 2), pct))

        # Sort by distance and return max 10
        return sorted(walls, key=lambda w: w.distance_pct)[:10]

    def get_stack(self, side: str, levels: int = 10) -> List[List]:
        """Get top N levels for a side (default 10 for more context)."""
        book = self.bids if side == "BID" else self.asks
        reverse = side == "BID"
        sorted_levels = sorted(book.items(), reverse=reverse)[:levels]
        return [[p, s] for p, s in sorted_levels]

    def get_stack_to_wall(self, side: str, max_levels: int = 15) -> List[Dict]:
        """
        Get all levels from best price UP TO first major/massive wall.

        This gives the LLM context on immediate liquidity AND where
        the first significant barrier is located.

        Returns list of dicts with price, size, cumulative_size, and wall_tier.
        """
        book = self.bids if side == "BID" else self.asks
        reverse = side == "BID"
        sorted_levels = sorted(book.items(), reverse=reverse)

        stack = []
        cumulative = 0
        for price, size in sorted_levels:
            cumulative += size
            tier = self._get_wall_tier(size)
            stack.append({
                "price": price,
                "size": size,
                "cumulative_size": cumulative,
                "wall_tier": tier  # None, "MINOR", "MAJOR", or "MASSIVE"
            })

            # Stop at first MAJOR or MASSIVE wall
            if tier in ("MAJOR", "MASSIVE"):
                break

            # Safety cap
            if len(stack) >= max_levels:
                break

        return stack

    def get_book_stats(self) -> Dict[str, Any]:
        """Get summary statistics about the order book."""
        all_sizes = list(self.bids.values()) + list(self.asks.values())
        return {
            "bid_levels": len(self.bids),
            "ask_levels": len(self.asks),
            "total_bid_volume": sum(self.bids.values()),
            "total_ask_volume": sum(self.asks.values()),
            "p75_threshold": self._p75,
            "p90_threshold": self._p90,
            "p95_threshold": self._p95,
            "avg_size": float(np.mean(all_sizes)) if all_sizes else 0,
        }

# =============================================================================
# TAPE STREAM (Time & Sales)
# =============================================================================

class TapeStream:
    """
    Enhanced Tape Analysis with Delta Windows and Large Print Detection.
    """
    def __init__(self, ticker: str, large_threshold: int = 300):
        self.ticker = ticker
        self.trades: deque = deque(maxlen=500)  # Last 500 trades
        self.large_threshold = large_threshold
        self.last_bid = 0.0
        self.last_ask = 0.0
        self.logger = logging.getLogger(f"TAPE_{ticker}")
        
    def update_quotes(self, bid: float, ask: float):
        """Update current bid/ask for side detection."""
        self.last_bid = bid
        self.last_ask = ask
        
    def on_tick(self, tick, timestamp: float = None):
        """Process incoming trade tick."""
        try:
            price = tick.price
            size = tick.size
            ts = timestamp if timestamp else time.time()
            
            # Determine side: if price >= ask, buyer hit ask (BUY)
            if price >= self.last_ask:
                side = "BUY"
            elif price <= self.last_bid:
                side = "SELL"
            else:
                side = "BUY" if price > (self.last_bid + self.last_ask) / 2 else "SELL"
            
            self.trades.append(Trade(price, size, ts, side))
        except Exception as e:
            self.logger.error(f"Tick error: {e}")
    
    def get_velocity(self, timestamp: float = None) -> tuple:
        """Returns (velocity_label, trades_per_second)."""
        now = timestamp if timestamp else time.time()
        recent = [t for t in self.trades if now - t.time <= 5]
        tps = len(recent) / 5.0 if recent else 0
        
        if tps < 10:
            label = "LOW"
        elif tps < 30:
            label = "MEDIUM"
        else:
            label = "HIGH"
        
        return label, round(tps, 1)
    
    def get_sentiment(self, timestamp: float = None) -> str:
        """Analyze tape for buy/sell aggression."""
        now = timestamp if timestamp else time.time()
        recent = [t for t in self.trades if now - t.time <= 10]
        
        if len(recent) < 5:
            return "NEUTRAL"
        
        buys = sum(t.size for t in recent if t.side == "BUY")
        sells = sum(t.size for t in recent if t.side == "SELL")
        
        if buys == 0 and sells == 0:
            return "NEUTRAL"
        
        ratio = buys / sells if sells > 0 else 10.0
        
        if ratio > 2.0:
            return "AGGRESSIVE_BUYING"
        elif ratio < 0.5:
            return "AGGRESSIVE_SELLING"
        else:
            return "NEUTRAL"
    
    def get_delta(self, seconds: float, timestamp: float = None) -> int:
        """Net delta over N seconds."""
        now = timestamp if timestamp else time.time()
        recent = [t for t in self.trades if now - t.time <= seconds]
        
        buy_vol = sum(t.size for t in recent if t.side == "BUY")
        sell_vol = sum(t.size for t in recent if t.side == "SELL")
        
        return buy_vol - sell_vol
    
    def get_large_prints(self, seconds: float = 60, timestamp: float = None) -> List[LargePrint]:
        """Get trades above threshold in last N seconds."""
        now = timestamp if timestamp else time.time()
        large = []
        
        for t in self.trades:
            age = now - t.time
            if age <= seconds and t.size >= self.large_threshold:
                large.append(LargePrint(t.price, t.size, t.side, round(age, 1)))
        
        return sorted(large, key=lambda x: x.secs_ago)[:5]

# =============================================================================
# FOOTPRINT TRACKER
# =============================================================================

class FootprintTracker:
    """
    Tracks Footprint data for current and recent bars.
    Calculates delta, POC, and imbalances per bar.
    """
    def __init__(self, ticker: str, bar_seconds: int = 60):
        self.ticker = ticker
        self.bar_seconds = bar_seconds
        self.current_bar: FootprintBar = FootprintBar()
        self.bar_start_time: float = 0
        self.recent_bars: deque = deque(maxlen=20)
        self.logger = logging.getLogger(f"FOOTPRINT_{ticker}")
    
    def on_trade(self, price: float, size: float, side: str, timestamp: float = None):
        """Process trade into footprint. Size can be float for crypto."""
        now = timestamp if timestamp else time.time()
        
        # Check for new bar
        if now - self.bar_start_time >= self.bar_seconds:
            if self.current_bar.volume > 0:
                self._finalize_bar()
            self._start_new_bar(price, now)
        
        # Update current bar
        bar = self.current_bar
        bar.high = max(bar.high, price)
        bar.low = min(bar.low, price)
        bar.close = price
        bar.volume += size
        
        if side == "BUY":
            bar.buy_volume += size
        else:
            bar.sell_volume += size
        
        bar.delta = bar.buy_volume - bar.sell_volume
        
        # Track volume at price
        rounded_price = round(price, 2)
        bar.volume_at_price[rounded_price] = bar.volume_at_price.get(rounded_price, 0) + size
    
    def _start_new_bar(self, price: float, ts: float):
        """Initialize new bar."""
        self.current_bar = FootprintBar(open=price, high=price, low=price, close=price)
        self.bar_start_time = ts
    
    def _finalize_bar(self):
        """Calculate final bar metrics and store."""
        bar = self.current_bar
        
        # Delta percent
        if bar.volume > 0:
            bar.delta_pct = round((bar.delta / bar.volume) * 100, 1)
        
        # Point of Control (price with most volume)
        if bar.volume_at_price:
            bar.poc = max(bar.volume_at_price.items(), key=lambda x: x[1])[0]
        
        # Detect imbalances (>3:1 ratio at price level)
        bar.imbalances = self._detect_imbalances(bar)
        
        self.recent_bars.append(bar)
    
    def _detect_imbalances(self, bar: FootprintBar) -> List[Dict]:
        """Find price levels with extreme buy/sell imbalance."""
        # Simplified: would need bid/ask volume per price level
        # For now, return empty (requires more granular data)
        return []
    
    def get_current_bar(self) -> Dict:
        """Get current bar as dict for LLM."""
        bar = self.current_bar
        if bar.volume == 0:
            return {}
        
        # Calculate POC for current bar
        poc = bar.close
        if bar.volume_at_price:
            poc = max(bar.volume_at_price.items(), key=lambda x: x[1])[0]
        
        return {
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "delta": bar.delta,
            "volume": bar.volume,
            "delta_pct": round((bar.delta / bar.volume) * 100, 1) if bar.volume else 0,
            "poc": poc,
            "imbalances": bar.imbalances
        }

# =============================================================================
# CUMULATIVE DELTA (CVD)
# =============================================================================

class CumulativeDelta:
    """
    Session Cumulative Delta with Trend Analysis.
    """
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.session_delta: float = 0.0  # Float for crypto fractional sizes
        self.delta_history: deque = deque(maxlen=300)  # 5 min at 1sec granularity
        self.last_update: float = 0
        self.logger = logging.getLogger(f"CVD_{ticker}")
    
    def add_trade(self, size: float, side: str, timestamp: float = None):
        """Accumulate delta from trade. Size can be float for crypto."""
        delta = size if side == "BUY" else -size
        self.session_delta += delta
        
        now = timestamp if timestamp else time.time()
        if now - self.last_update >= 1.0:  # Sample every second
            self.delta_history.append(self.session_delta)
            self.last_update = now
    
    def get_trend(self) -> str:
        """Determine CVD trend from slope."""
        if len(self.delta_history) < 10:
            return "FLAT"
        
        # Simple linear regression over last 30 samples (30 sec)
        recent = list(self.delta_history)[-30:]
        if len(recent) < 10:
            return "FLAT"
        
        x = np.arange(len(recent))
        y = np.array(recent)
        
        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]
        
        # Normalize slope
        y_range = max(y) - min(y) if max(y) != min(y) else 1
        norm_slope = slope / y_range * len(recent)
        
        if norm_slope > 0.3:
            return "RISING"
        elif norm_slope < -0.3:
            return "FALLING"
        else:
            return "FLAT"
    
    def get_slope_5m(self) -> float:
        """Get normalized slope over 5 minutes."""
        if len(self.delta_history) < 60:
            return 0.0
        
        recent = list(self.delta_history)[-300:]  # 5 min
        x = np.arange(len(recent))
        y = np.array(recent)
        
        slope = np.polyfit(x, y, 1)[0]
        y_range = max(y) - min(y) if max(y) != min(y) else 1
        
        return round(max(-1, min(1, slope / y_range * 100)), 2)

# =============================================================================
# VOLUME PROFILE
# =============================================================================

class VolumeProfile:
    """
    Session Volume Profile with POC, VAH, VAL.
    """
    def __init__(self, ticker: str, tick_size: float = 0.01):
        self.ticker = ticker
        self.tick_size = tick_size
        self.volume_at_price: Dict[float, float] = defaultdict(float)  # Float for crypto
        self.total_volume: float = 0.0  # Float for crypto fractional sizes
        self.logger = logging.getLogger(f"VP_{ticker}")
    
    def add_trade(self, price: float, size: float):
        """Add volume to profile. Size can be float for crypto."""
        rounded = round(price / self.tick_size) * self.tick_size
        rounded = round(rounded, 2)
        self.volume_at_price[rounded] += size
        self.total_volume += size
    
    def get_poc(self) -> float:
        """Point of Control - price with highest volume."""
        if not self.volume_at_price:
            return 0.0
        return max(self.volume_at_price.items(), key=lambda x: x[1])[0]
    
    def get_value_area(self, pct: float = 0.70) -> tuple:
        """Calculate Value Area High and Low (70% of volume)."""
        if not self.volume_at_price or self.total_volume == 0:
            return 0.0, 0.0
        
        sorted_levels = sorted(self.volume_at_price.items(), key=lambda x: x[1], reverse=True)
        target_volume = self.total_volume * pct
        
        accumulated = 0
        prices_in_va = []
        
        for price, vol in sorted_levels:
            accumulated += vol
            prices_in_va.append(price)
            if accumulated >= target_volume:
                break
        
        if not prices_in_va:
            return 0.0, 0.0
        
        return round(max(prices_in_va), 2), round(min(prices_in_va), 2)
    
    def get_developing_poc(self) -> float:
        """Same as POC, but named for clarity."""
        return self.get_poc()
    
    def get_price_location(self, current_price: float) -> str:
        """Where is price relative to POC."""
        poc = self.get_poc()
        if poc == 0:
            return "UNKNOWN"
        
        if current_price > poc * 1.001:
            return "ABOVE"
        elif current_price < poc * 0.999:
            return "BELOW"
        else:
            return "AT_POC"

# =============================================================================
# ABSORPTION DETECTOR
# =============================================================================

class AbsorptionDetector:
    """
    Detects absorption patterns - large orders absorbing flow without price moving.
    """
    def __init__(self, ticker: str, window_size: int = 50):
        self.ticker = ticker
        self.window_size = window_size
        self.recent_trades: deque = deque(maxlen=window_size)
        self.logger = logging.getLogger(f"ABSORB_{ticker}")
    
    def on_trade(self, price: float, size: float, side: str, timestamp: float = None):
        """Track trades for absorption detection. Size can be float for crypto."""
        self.recent_trades.append({
            "price": price,
            "size": size,
            "side": side,
            "time": timestamp if timestamp else time.time()
        })
    
    def detect(self) -> Dict:
        """
        Detect absorption: high volume at a price level with minimal price movement.
        Returns: {"detected": bool, "side": str|None, "price": float|None}
        """
        if len(self.recent_trades) < 20:
            return {"detected": False, "side": None, "price": None}
        
        trades = list(self.recent_trades)
        
        # Check price stability (low range despite volume)
        prices = [t["price"] for t in trades[-20:]]
        price_range = max(prices) - min(prices)
        total_volume = sum(t["size"] for t in trades[-20:])
        
        # Absorption = high volume, low price movement
        if price_range < 0.05 and total_volume > 500:  # Thresholds tunable
            # Determine which side is absorbing
            buys = sum(t["size"] for t in trades[-20:] if t["side"] == "BUY")
            sells = sum(t["size"] for t in trades[-20:] if t["side"] == "SELL")
            
            if sells > buys * 1.5:
                # Lots of selling but price stable = BID absorbing
                return {
                    "detected": True,
                    "side": "BID",
                    "price": round(np.mean(prices), 2)
                }
            elif buys > sells * 1.5:
                # Lots of buying but price stable = ASK absorbing
                return {
                    "detected": True,
                    "side": "ASK",
                    "price": round(np.mean(prices), 2)
                }
        
        return {"detected": False, "side": None, "price": None}

# =============================================================================
# MARKET METRICS
# =============================================================================

class MarketMetrics:
    """
    Session-level metrics: HOD, LOD, RVOL.
    """
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.hod: float = 0.0
        self.lod: float = float('inf')
        self.last_price: float = 0.0
        self.vwap: float = 0.0
        self.total_value: float = 0.0
        self.session_volume: float = 0.0  # Float for crypto fractional sizes
        self.avg_volume_30d: int = 10_000_000  # Default, should be fetched
        self.session_start: float = 0
        self.logger = logging.getLogger(f"METRICS_{ticker}")
    
    def update_price(self, price: float, timestamp: float = None):
        """Update HOD/LOD from price."""
        self.last_price = price
        if price > self.hod:
            self.hod = price
        if price < self.lod:
            self.lod = price
        
        if self.session_start == 0 and timestamp:
            self.session_start = timestamp
            
    def add_trade(self, price: float, size: float, timestamp: float = None):
        """Update VWAP and Volume from trade. Size can be float for crypto."""
        self.update_price(price, timestamp)

        self.session_volume += size
        self.total_value += price * size

        if self.session_volume > 0:
            self.vwap = round(self.total_value / self.session_volume, 2)
    
    def update_from_snapshot(self, ticker_obj):
        """Update from IBKR ticker snapshot.

        NOTE: We intentionally do NOT use ticker_obj.high/low here because
        IBKR provides 24-hour high/low, not intraday session high/low.
        HOD/LOD are tracked from actual trades via update_price() instead.
        """
        import math

        # Skip IBKR's high/low - they're 24h values, not intraday
        # HOD/LOD are tracked from trades in update_price()

        if hasattr(ticker_obj, 'last') and ticker_obj.last:
            # Update price (this also updates HOD/LOD from actual trades)
            self.update_price(ticker_obj.last)
        # Only use IBKR's volume/vwap if valid (not nan) - crypto often has nan values
        if hasattr(ticker_obj, 'volume') and ticker_obj.volume and not math.isnan(ticker_obj.volume):
            self.session_volume = ticker_obj.volume
        if hasattr(ticker_obj, 'vwap') and ticker_obj.vwap and not math.isnan(ticker_obj.vwap):
            self.vwap = ticker_obj.vwap
    
    def get_hod_lod_location(self) -> str:
        """Where is price relative to HOD/LOD."""
        if self.hod == 0 or self.lod == float('inf'):
            return "UNKNOWN"
        
        range_size = self.hod - self.lod
        if range_size == 0:
            return "AT_HOD" if self.last_price >= self.hod else "UNKNOWN"
        
        # Distance from LOD as percentage of range
        location_pct = (self.last_price - self.lod) / range_size
        
        if location_pct >= 0.98:
            return "TESTING_HOD"
        elif location_pct >= 0.85:
            return "NEAR_HOD"
        elif location_pct <= 0.02:
            return "TESTING_LOD"
        elif location_pct <= 0.15:
            return "NEAR_LOD"
        else:
            return "MID_RANGE"
    
    def get_distance_to_hod_pct(self) -> float:
        """Percentage distance to HOD."""
        if self.last_price == 0 or self.hod == 0:
            return 0.0
        return round(abs(self.hod - self.last_price) / self.last_price * 100, 2)
    
    def get_rvol(self, timestamp: float = None) -> float:
        """Relative volume vs 30-day average."""
        if self.avg_volume_30d == 0:
            return 1.0
        
        # Adjust for time of day
        now = timestamp if timestamp else time.time()
        if self.session_start == 0:
            elapsed_hours = 0.1 # avoid div by zero
        else:
            elapsed_hours = (now - self.session_start) / 3600
            
        # Avoid division by zero for very start of session
        if elapsed_hours < 0.01:
            elapsed_hours = 0.01
            
        expected_vol = (elapsed_hours / 6.5) * self.avg_volume_30d
        
        if expected_vol == 0:
            return 1.0
        
        return round(self.session_volume / expected_vol, 2)
    
    def get_time_session(self, timestamp: float = None) -> str:
        """Current session phase."""
        now = timestamp if timestamp else time.time()
        if self.session_start == 0:
            return "OPEN_DRIVE"
            
        elapsed_min = (now - self.session_start) / 60
        
        if elapsed_min < 30:
            return "OPEN_DRIVE"
        elif elapsed_min < 60:
            return "OPEN_RANGE"
        elif elapsed_min < 180:
            return "MORNING"
        elif elapsed_min < 300:
            return "MIDDAY"
        else:
            return "CLOSE"

# =============================================================================
# IBKR CONNECTOR (Orchestrator)
# =============================================================================

class IBKRConnector:
    """
    Main connector that orchestrates all data sources.
    """
    def __init__(self, host: str = '127.0.0.1', port: int = 7497, client_id: int = 10):
        self.ib = IB() if IB else None
        self.host = host
        self.port = port
        self.client_id = client_id
        
        # Data components per symbol
        self.books: Dict[str, OrderBook] = {}
        self.tapes: Dict[str, TapeStream] = {}
        self.footprints: Dict[str, FootprintTracker] = {}
        self.cvds: Dict[str, CumulativeDelta] = {}
        self.profiles: Dict[str, VolumeProfile] = {}
        self.absorbers: Dict[str, AbsorptionDetector] = {}
        self.metrics: Dict[str, MarketMetrics] = {}

        # Track last seen price/size for fallback trade detection (crypto)
        self._last_seen: Dict[str, tuple] = {}  # symbol -> (price, size)

        self.logger = logging.getLogger("IBKR_CONN")
    
    async def connect(self):
        """Connect to IBKR TWS/Gateway."""
        if not self.ib:
            self.logger.error("ib_insync not installed!")
            return
        
        self.logger.info(f"Connecting to TWS on {self.host}:{self.port}...")
        try:
            await self.ib.connectAsync(self.host, self.port, self.client_id)
            self.logger.info("Connected to IBKR!")
        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            raise
    
    async def subscribe_market_data(self, symbol: str):
        """Subscribe to all data feeds for a symbol."""
        if not self.ib:
            return

        # Determine contract type based on symbol
        if symbol.upper() in CRYPTO_SYMBOLS:
            contract = Crypto(symbol, 'ZEROHASH', 'USD')
            self.logger.info(f"Using Crypto contract for {symbol} (ZEROHASH)")
        else:
            contract = Stock(symbol, 'SMART', 'USD')
        await self.ib.qualifyContractsAsync(contract)

        # Initialize all components
        self.books[symbol] = OrderBook(symbol)
        self.tapes[symbol] = TapeStream(symbol)
        self.footprints[symbol] = FootprintTracker(symbol)
        self.cvds[symbol] = CumulativeDelta(symbol)
        self.profiles[symbol] = VolumeProfile(symbol)
        self.absorbers[symbol] = AbsorptionDetector(symbol)
        self.metrics[symbol] = MarketMetrics(symbol)

        # Subscribe to data feeds
        # reqMktDepth -> DOM data arrives on Ticker.domBids / Ticker.domAsks
        # Request 50 levels for deeper book analysis and better wall detection
        self.ib.reqMktDepth(contract, numRows=50, isSmartDepth=True)
        # reqTickByTickData -> trade ticks arrive on Ticker.tickByTicks
        self.ib.reqTickByTickData(contract, 'AllLast', 0, False)
        # reqMktData -> snapshot data (bid/ask/last/vwap) on Ticker
        # Generic tick types: 233=VWAP, 165=Misc Stats (avg volume)
        self.ib.reqMktData(contract, '233', False, False)

        # ib_insync 0.9.86: all updates come through pendingTickersEvent
        # (no separate updateMktDepthEvent or tickByTickAllLastEvent)
        self.ib.pendingTickersEvent += self._on_ticker_update

        self.logger.info(f"Subscribed to {symbol}")
    
    def _on_ticker_update(self, tickers):
        """
        Unified handler for all ticker updates (ib_insync 0.9.86).

        pendingTickersEvent fires with a set of Ticker objects.
        Each Ticker carries:
          - bid/ask/last (market data)
          - domBids/domAsks (L2 depth from reqMktDepth)
          - tickByTicks (trade ticks from reqTickByTickData)
        """
        for t in tickers:
            symbol = t.contract.symbol

            # --- Market data snapshot (bid/ask/last/volume) ---
            if symbol in self.metrics:
                self.metrics[symbol].update_from_snapshot(t)

            # --- L2 DOM updates ---
            # domTicks present means depth changed; sync from authoritative domBids/domAsks
            if symbol in self.books and t.domTicks:
                self.books[symbol].update_from_dom_levels(t.domBids, t.domAsks)

            # --- Trade ticks ---
            if symbol in self.tapes:
                # Update quote for side detection
                if t.bid and t.ask:
                    self.tapes[symbol].update_quotes(t.bid, t.ask)

                tape = self.tapes[symbol]

                if t.tickByTicks:
                    # Primary path: use tick-by-tick data (stocks)
                    for tick in t.tickByTicks:
                        tape.on_tick(tick)
                        self._process_trade(symbol, tick.price, tick.size, tape)
                elif t.last and t.lastSize:
                    # Fallback path: use last price/size (crypto via ZEROHASH)
                    # Only process if price or size changed (indicates new trade)
                    price = t.last
                    size = t.lastSize
                    prev = self._last_seen.get(symbol)
                    if prev is None or prev != (price, size):
                        self._last_seen[symbol] = (price, size)
                        if prev is not None:  # Skip first update (not a trade)
                            self._process_trade(symbol, price, size, tape)

    def _process_trade(self, symbol: str, price: float, size: float, tape: TapeStream):
        """Process a trade and feed all analyzers."""
        # Determine side based on quote
        if price >= tape.last_ask:
            side = "BUY"
        elif price <= tape.last_bid:
            side = "SELL"
        else:
            side = "BUY"  # Default

        # Feed all analyzers
        self.footprints[symbol].on_trade(price, size, side)
        self.cvds[symbol].add_trade(size, side)
        self.profiles[symbol].add_trade(price, size)
        self.absorbers[symbol].on_trade(price, size, side)
        self.metrics[symbol].add_trade(price, size)

    def get_full_state(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Aggregate ALL data sources into the complete LLM context.
        This is the State Vector sent to the Brain.
        """
        if symbol not in self.books:
            return None
        
        book = self.books[symbol]
        tape = self.tapes[symbol]
        footprint = self.footprints[symbol]
        cvd = self.cvds[symbol]
        profile = self.profiles[symbol]
        absorber = self.absorbers[symbol]
        metrics = self.metrics[symbol]
        
        last_price = metrics.last_price
        vel_label, vel_tps = tape.get_velocity()
        vah, val = profile.get_value_area()
        absorption = absorber.detect()
        
        return {
            "MARKET_STATE": {
                "TICKER": symbol,
                "LAST": last_price,
                "VWAP": metrics.vwap,
                "TIME_SESSION": metrics.get_time_session(),
                
                # Level 2 (DOM) - Smart processing with cumulative sizes
                "L2_IMBALANCE": book.get_imbalance(),
                "SPREAD": book.get_spread(),
                "DOM_WALLS": [
                    {"side": w.side, "price": w.price, "size": w.size,
                     "tier": w.tier, "distance_pct": w.distance_pct,
                     "percentile": w.percentile}
                    for w in book.get_walls(last_price)
                ],
                # Dynamic stacks - show levels up to first major wall
                "BID_STACK_TO_WALL": book.get_stack_to_wall("BID"),
                "ASK_STACK_TO_WALL": book.get_stack_to_wall("ASK"),
                # Also include simple top-10 for backwards compatibility
                "BID_STACK": book.get_stack("BID", levels=10),
                "ASK_STACK": book.get_stack("ASK", levels=10),
                # Book statistics for context
                "BOOK_STATS": book.get_book_stats(),
                
                # Tape
                "TAPE_VELOCITY": vel_label,
                "TAPE_VELOCITY_TPS": vel_tps,
                "TAPE_SENTIMENT": tape.get_sentiment(),
                "TAPE_DELTA_1S": tape.get_delta(1),
                "TAPE_DELTA_5S": tape.get_delta(5),
                "LARGE_PRINTS_1M": [
                    {"price": lp.price, "size": lp.size, "side": lp.side, 
                     "secs_ago": lp.secs_ago}
                    for lp in tape.get_large_prints()
                ],
                
                # Footprint
                "FOOTPRINT_CURR_BAR": footprint.get_current_bar(),
                
                # Cumulative Delta
                "CVD_SESSION": cvd.session_delta,
                "CVD_TREND": cvd.get_trend(),
                "CVD_SLOPE_5M": cvd.get_slope_5m(),
                
                # Volume Profile
                "VP_POC": profile.get_poc(),
                "VP_VAH": vah,
                "VP_VAL": val,
                "VP_DEVELOPING_POC": profile.get_developing_poc(),
                "PRICE_VS_POC": profile.get_price_location(last_price),
                
                # Key Levels
                "HOD": metrics.hod,
                "LOD": metrics.lod if metrics.lod != float('inf') else 0,
                "HOD_LOD_LOC": metrics.get_hod_lod_location(),
                "DISTANCE_TO_HOD_PCT": metrics.get_distance_to_hod_pct(),
                "RVOL_DAY": metrics.get_rvol(),
                
                # Absorption
                "ABSORPTION_DETECTED": absorption["detected"],
                "ABSORPTION_SIDE": absorption["side"],
                "ABSORPTION_PRICE": absorption["price"]
            }
        }
    
    def run(self):
        """Start the event loop."""
        if self.ib:
            self.ib.run()
