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
    from ib_insync import IB, Stock, util
except ImportError:
    IB = None
    Stock = None

import numpy as np

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Trade:
    """Single trade from tape."""
    price: float
    size: int
    time: float  # Unix timestamp (simulated or real)
    side: str    # "BUY" or "SELL"

@dataclass
class FootprintBar:
    """One-minute bar with delta analysis."""
    open: float = 0.0
    high: float = 0.0
    low: float = float('inf')
    close: float = 0.0
    volume: int = 0
    buy_volume: int = 0
    sell_volume: int = 0
    delta: int = 0          # buy_volume - sell_volume
    delta_pct: float = 0.0  # delta / volume * 100
    poc: float = 0.0        # Price with most volume
    volume_at_price: Dict[float, int] = field(default_factory=dict)
    imbalances: List[Dict] = field(default_factory=list)

@dataclass
class DOMWall:
    """Significant order in the book."""
    side: str      # "BID" or "ASK"
    price: float
    size: int
    tier: str      # "MAJOR" (>5x avg) or "MINOR" (>3x avg)
    distance_pct: float

@dataclass
class LargePrint:
    """Large trade from tape."""
    price: float
    size: int
    side: str
    secs_ago: float

# =============================================================================
# ORDER BOOK (Level 2 / DOM)
# =============================================================================

class OrderBook:
    """
    Real-time Level 2 Order Book with Wall Detection.
    """
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.bids: Dict[float, int] = {}  # Price -> Size
        self.asks: Dict[float, int] = {}
        self.logger = logging.getLogger(f"DOM_{ticker}")
        
    def update(self, dom_event):
        """Handle IBKR updateMktDepth event."""
        target = self.bids if dom_event.side == 1 else self.asks
        price = round(dom_event.price, 2)
        size = dom_event.size
        op = dom_event.operation
        
        if op in (0, 1):  # Insert or Update
            target[price] = size
        elif op == 2:     # Delete
            target.pop(price, None)
    
    def get_imbalance(self) -> float:
        """Calculate bid/ask volume ratio."""
        sorted_bids = sorted(self.bids.items(), reverse=True)[:5]
        sorted_asks = sorted(self.asks.items())[:5]
        
        bid_vol = sum(s for _, s in sorted_bids)
        ask_vol = sum(s for _, s in sorted_asks)
        
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
    
    def get_walls(self, last_price: float, threshold_minor: float = 3.0, 
                  threshold_major: float = 5.0) -> List[DOMWall]:
        """Detect significant orders (walls)."""
        walls = []
        all_sizes = list(self.bids.values()) + list(self.asks.values())
        if not all_sizes:
            return walls
        avg_size = np.mean(all_sizes)
        
        for price, size in self.bids.items():
            ratio = size / avg_size if avg_size > 0 else 0
            if ratio >= threshold_minor:
                tier = "MAJOR" if ratio >= threshold_major else "MINOR"
                dist = abs(price - last_price) / last_price * 100
                walls.append(DOMWall("BID", price, size, tier, round(dist, 2)))
                
        for price, size in self.asks.items():
            ratio = size / avg_size if avg_size > 0 else 0
            if ratio >= threshold_minor:
                tier = "MAJOR" if ratio >= threshold_major else "MINOR"
                dist = abs(price - last_price) / last_price * 100
                walls.append(DOMWall("ASK", price, size, tier, round(dist, 2)))
        
        return sorted(walls, key=lambda w: w.distance_pct)[:5]
    
    def get_stack(self, side: str, levels: int = 3) -> List[List]:
        """Get top N levels for a side."""
        book = self.bids if side == "BID" else self.asks
        reverse = side == "BID"
        sorted_levels = sorted(book.items(), reverse=reverse)[:levels]
        return [[p, s] for p, s in sorted_levels]

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
    
    def on_trade(self, price: float, size: int, side: str, timestamp: float = None):
        """Process trade into footprint."""
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
        self.session_delta: int = 0
        self.delta_history: deque = deque(maxlen=300)  # 5 min at 1sec granularity
        self.last_update: float = 0
        self.logger = logging.getLogger(f"CVD_{ticker}")
    
    def add_trade(self, size: int, side: str, timestamp: float = None):
        """Accumulate delta from trade."""
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
        self.volume_at_price: Dict[float, int] = defaultdict(int)
        self.total_volume: int = 0
        self.logger = logging.getLogger(f"VP_{ticker}")
    
    def add_trade(self, price: float, size: int):
        """Add volume to profile."""
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
    
    def on_trade(self, price: float, size: int, side: str, timestamp: float = None):
        """Track trades for absorption detection."""
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
        self.session_volume: int = 0
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
            
    def add_trade(self, price: float, size: int, timestamp: float = None):
        """Update VWAP and Volume from trade."""
        self.update_price(price, timestamp)
        
        self.session_volume += size
        self.total_value += price * size
        
        if self.session_volume > 0:
            self.vwap = round(self.total_value / self.session_volume, 2)
    
    def update_from_snapshot(self, ticker_obj):
        """Update from IBKR ticker snapshot."""
        if hasattr(ticker_obj, 'high') and ticker_obj.high:
            self.hod = ticker_obj.high
        if hasattr(ticker_obj, 'low') and ticker_obj.low:
            self.lod = ticker_obj.low
        if hasattr(ticker_obj, 'last') and ticker_obj.last:
            self.last_price = ticker_obj.last
        if hasattr(ticker_obj, 'volume') and ticker_obj.volume:
            self.session_volume = ticker_obj.volume
        if hasattr(ticker_obj, 'vwap') and ticker_obj.vwap:
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
    def __init__(self, host: str = '127.0.0.1', port: int = 7497, client_id: int = 1):
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
    
    def subscribe_market_data(self, symbol: str):
        """Subscribe to all data feeds for a symbol."""
        if not self.ib:
            return
        
        contract = Stock(symbol, 'SMART', 'USD')
        self.ib.qualifyContracts(contract)
        
        # Initialize all components
        self.books[symbol] = OrderBook(symbol)
        self.tapes[symbol] = TapeStream(symbol)
        self.footprints[symbol] = FootprintTracker(symbol)
        self.cvds[symbol] = CumulativeDelta(symbol)
        self.profiles[symbol] = VolumeProfile(symbol)
        self.absorbers[symbol] = AbsorptionDetector(symbol)
        self.metrics[symbol] = MarketMetrics(symbol)
        
        # Subscribe to data feeds
        self.ib.reqMktDepth(contract, numRows=20, isSmartDepth=True)
        self.ib.reqTickByTickData(contract, 'AllLast', 0, False)
        self.ib.reqMktData(contract, '', False, False)
        
        # Hook events
        self.ib.updateMktDepthEvent += self._on_depth
        self.ib.tickByTickAllLastEvent += self._on_tick
        self.ib.pendingTickersEvent += self._on_snapshot
        
        self.logger.info(f"Subscribed to {symbol}")
    
    def _on_depth(self, item):
        """Handle DOM update."""
        symbol = item.contract.symbol
        if symbol in self.books:
            self.books[symbol].update(item)
    
    def _on_tick(self, ticker, tick):
        """Handle trade tick - feeds all analyzers."""
        symbol = ticker.contract.symbol
        
        if symbol not in self.tapes:
            return
        
        # Update quote for side detection
        if ticker.bid and ticker.ask:
            self.tapes[symbol].update_quotes(ticker.bid, ticker.ask)
        
        # Feed tape
        self.tapes[symbol].on_tick(tick)
        
        # Get trade details
        price = tick.price
        size = tick.size
        
        # Determine side from tape
        tape = self.tapes[symbol]
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
        self.metrics[symbol].update_price(price)
    
    def _on_snapshot(self, tickers):
        """Handle market data snapshot."""
        for t in tickers:
            symbol = t.contract.symbol
            if symbol in self.metrics:
                self.metrics[symbol].update_from_snapshot(t)
    
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
                
                # Level 2 (DOM)
                "L2_IMBALANCE": book.get_imbalance(),
                "SPREAD": book.get_spread(),
                "DOM_WALLS": [
                    {"side": w.side, "price": w.price, "size": w.size, 
                     "tier": w.tier, "distance_pct": w.distance_pct}
                    for w in book.get_walls(last_price)
                ],
                "BID_STACK": book.get_stack("BID"),
                "ASK_STACK": book.get_stack("ASK"),
                
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
