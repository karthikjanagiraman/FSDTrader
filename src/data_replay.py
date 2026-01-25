#!/usr/bin/env python3
"""
FSDTrader: Data Replay Layer
Reads Databento MBO files and simulates IBKR-style market data events.

This allows backtesting with real L3 order-by-order data without a live connection.

Usage:
    replayer = MBOReplayConnector(data_dir="BacktestData/TSLA-L3DATA", date="20251023")
    replayer.subscribe_market_data("TSLA")
    await replayer.start_replay(speed=1.0)  # 1.0 = real-time, 10.0 = 10x speed
"""
import logging
import asyncio
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Callable, Any
from collections import deque
import json

import databento as db
import numpy as np

# Import our market data classes
from market_data import (
    OrderBook, TapeStream, FootprintTracker, 
    CumulativeDelta, VolumeProfile, AbsorptionDetector, MarketMetrics
)


@dataclass 
class MBOEvent:
    """Normalized MBO event for internal use."""
    ts_event: int       # Nanoseconds since epoch
    action: str         # "A" = Add, "C" = Cancel, "M" = Modify, "T" = Trade, "F" = Fill
    side: str           # "B" = Bid, "A" = Ask
    price: float        # Price in dollars
    size: int           # Size in shares
    order_id: int       # Unique order ID
    flags: int          # Event flags


class MBOReplayConnector:
    """
    Replays Databento MBO data as if it were live IBKR data.
    Implements the same interface as IBKRConnector.
    """
    
    def __init__(self, data_dir: str, date: str = None, symbol: str = "TSLA"):
        self.data_dir = Path(data_dir)
        self.date = date
        self.symbol = symbol
        self.logger = logging.getLogger("MBO_REPLAY")
        
        # Find available dates
        self.available_dates = self._find_available_dates()
        if date and date not in self.available_dates:
            raise ValueError(f"Date {date} not found. Available: {self.available_dates[:5]}...")
        
        # If no date specified, use first available
        if not date and self.available_dates:
            self.date = self.available_dates[0]
            self.logger.info(f"No date specified, using {self.date}")
        
        # Data components (same as IBKRConnector)
        self.books: Dict[str, OrderBook] = {}
        self.tapes: Dict[str, TapeStream] = {}
        self.footprints: Dict[str, FootprintTracker] = {}
        self.cvds: Dict[str, CumulativeDelta] = {}
        self.profiles: Dict[str, VolumeProfile] = {}
        self.absorbers: Dict[str, AbsorptionDetector] = {}
        self.metrics: Dict[str, MarketMetrics] = {}
        
        # Replay state
        self.events: List[Any] = []
        self.event_index: int = 0
        self.is_replaying: bool = False
        self.replay_speed: float = 1.0
        
        # Order book reconstruction
        self.order_book_state: Dict[int, dict] = {}  # order_id -> {price, size, side}
        
        # Stats
        self.events_processed: int = 0
        self.trades_count: int = 0
        
    def _find_available_dates(self) -> List[str]:
        """Find all available date files in the data directory."""
        dates = []
        for f in self.data_dir.glob("*.mbo.dbn.zst"):
            # Extract date from filename like "xnas-itch-20251023.mbo.dbn.zst"
            parts = f.stem.split("-")
            if len(parts) >= 3:
                date_str = parts[2].split(".")[0]
                if date_str.isdigit() and len(date_str) == 8:
                    dates.append(date_str)
        return sorted(dates)
    
    def _load_mbo_file(self, date: str):
        """Load MBO data for a specific date."""
        filename = f"xnas-itch-{date}.mbo.dbn.zst"
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"MBO file not found: {filepath}")
        
        self.logger.info(f"Loading MBO data from {filepath}...")
        
        # Read the DBN file
        store = db.DBNStore.from_file(filepath)
        
        # Convert to list of records
        self.events = list(store)
        self.event_index = 0
        
        self.logger.info(f"Loaded {len(self.events):,} MBO events for {date}")
        
        return len(self.events)
    
    def subscribe_market_data(self, symbol: str):
        """Initialize data structures for a symbol."""
        self.symbol = symbol
        
        # Initialize all components (same as IBKRConnector)
        self.books[symbol] = OrderBook(symbol)
        self.tapes[symbol] = TapeStream(symbol)
        self.footprints[symbol] = FootprintTracker(symbol)
        self.cvds[symbol] = CumulativeDelta(symbol)
        self.profiles[symbol] = VolumeProfile(symbol)
        self.absorbers[symbol] = AbsorptionDetector(symbol)
        self.metrics[symbol] = MarketMetrics(symbol)
        
        # Load the data
        self._load_mbo_file(self.date)
        
        self.logger.info(f"Subscribed to {symbol} replay data")
    
    async def start_replay(self, speed: float = 1.0, 
                           on_state_update: Callable = None,
                           start_time: str = "09:30:00",
                           end_time: str = "16:00:00"):
        """
        Start replaying the data.
        
        Args:
            speed: Playback speed (1.0 = real-time, 10.0 = 10x, 0 = as fast as possible)
            on_state_update: Callback called every N events with current state
            start_time: Start time in HH:MM:SS (default market open)
            end_time: End time in HH:MM:SS (default market close)
        """
        self.replay_speed = speed
        self.is_replaying = True
        
        # Filter events to market hours
        events_to_replay = self._filter_market_hours(start_time, end_time)
        total_events = len(events_to_replay)
        
        self.logger.info(f"Starting replay: {total_events:,} events at {speed}x speed")
        
        last_ts = None
        self.current_simulation_time = 0.0
        last_state_update = time.time()
        state_update_interval = 5.0  # Update state every 5 seconds for testing
        
        for i, event in enumerate(events_to_replay):
            if not self.is_replaying:
                break
            
            # Process the event
            self._process_mbo_event(event)
            self.events_processed += 1
            
            # Timing control
            if speed > 0 and last_ts is not None:
                event_ts = event.ts_event
                time_diff_ns = event_ts - last_ts
                sleep_time = (time_diff_ns / 1e9) / speed
                
                # Cap sleep time to avoid long waits
                if sleep_time > 0.001:  # > 1ms
                    await asyncio.sleep(min(sleep_time, 0.1))
            
            last_ts = event.ts_event
            self.current_simulation_time = event.ts_event / 1e9
            
            # Call state update callback periodically
            if on_state_update and time.time() - last_state_update > state_update_interval:
                state = self.get_full_state(self.symbol)
                if state:
                    await on_state_update(state)
                last_state_update = time.time()
            
            # Progress logging
            if i > 0 and i % 100000 == 0:
                pct = (i / total_events) * 100
                self.logger.info(f"Replay progress: {pct:.1f}% ({i:,}/{total_events:,})")
        
        self.is_replaying = False
        self.logger.info(f"Replay complete: {self.events_processed:,} events, {self.trades_count:,} trades")
    
    def _filter_market_hours(self, start_time: str, end_time: str) -> List:
        """Filter events to market hours only."""
        # Parse times (expected in Eastern Time)
        start_parts = [int(x) for x in start_time.split(":")]
        end_parts = [int(x) for x in end_time.split(":")]
        
        start_secs = start_parts[0] * 3600 + start_parts[1] * 60 + start_parts[2]
        end_secs = end_parts[0] * 3600 + end_parts[1] * 60 + end_parts[2]
        
        filtered = []
        for event in self.events:
            # Convert nanoseconds to time of day
            ts_ns = event.ts_event
            ts_secs = ts_ns / 1e9
            
            # Get time of day in UTC
            time_of_day_utc = ts_secs % 86400
            
            # Convert UTC to Eastern Time
            # DST (EDT = UTC-4): Mar-Nov (approx)
            # Standard (EST = UTC-5): Nov-Mar
            # For simplicity, check the month from timestamp
            from datetime import datetime
            dt = datetime.utcfromtimestamp(ts_secs)
            
            # DST is active from 2nd Sunday of March to 1st Sunday of November
            # Approximate: Use EDT (-4) for months 3-10, EST (-5) for 1-2, 11-12
            if 3 <= dt.month <= 10:
                et_offset = 4  # EDT
            else:
                et_offset = 5  # EST
            
            time_of_day_et = time_of_day_utc - et_offset * 3600
            if time_of_day_et < 0:
                time_of_day_et += 86400
            
            if start_secs <= time_of_day_et <= end_secs:
                filtered.append(event)
        
        return filtered
    
    def _process_mbo_event(self, event):
        """
        Process a single MBO event and update all analyzers.
        
        Databento MBO event fields:
        - ts_event: nanoseconds timestamp
        - action: 'A' (Add), 'C' (Cancel), 'M' (Modify), 'T' (Trade), 'F' (Fill)
        - side: 'B' (Bid) or 'A' (Ask)
        - price: price in fixed-point (divide by 1e9 for dollars)
        - size: number of shares
        - order_id: unique order identifier
        - flags: event flags
        """
        try:
            # Handle Databento enum types properly - they return full names like "TRADE", "ADD", etc.
            action = str(event.action.name) if hasattr(event.action, 'name') else str(event.action)
            side = str(event.side.name) if hasattr(event.side, 'name') else str(event.side)
            price = event.price / 1e9  # Convert fixed-point to dollars
            size = event.size
            order_id = event.order_id
            
            # Skip invalid prices
            if price <= 0 or price > 10000:
                return
            
            # Update order book state
            if action == 'ADD':  # Add order
                self.order_book_state[order_id] = {
                    'price': price,
                    'size': size,
                    'side': side
                }
                self._update_book(side, price, size, 'add')
                
            elif action == 'CANCEL':  # Cancel order
                if order_id in self.order_book_state:
                    old = self.order_book_state.pop(order_id)
                    self._update_book(old['side'], old['price'], old['size'], 'remove')
                    
            elif action == 'MODIFY':  # Modify order
                if order_id in self.order_book_state:
                    old = self.order_book_state[order_id]
                    self._update_book(old['side'], old['price'], old['size'], 'remove')
                    
                self.order_book_state[order_id] = {
                    'price': price,
                    'size': size,
                    'side': side
                }
                self._update_book(side, price, size, 'add')
                
            elif action in ('TRADE', 'FILL'):  # Trade or Fill
                self.trades_count += 1
                trade_side = "BUY" if side == 'ASK' else "SELL"  # Hit ask = buy
                
                # Update tape
                tape = self.tapes.get(self.symbol)
                if tape:
                    tape.update_quotes(
                        self._get_best_bid(),
                        self._get_best_ask()
                    )
                    # Create a mock tick object
                    class MockTick:
                        pass
                    tick = MockTick()
                    tick.price = price
                    tick.size = size
                    tick.time = event.ts_event / 1e9
                    tape.on_tick(tick, timestamp=tick.time)
                
                # Update footprint
                footprint = self.footprints.get(self.symbol)
                if footprint:
                    footprint.on_trade(price, size, trade_side, timestamp=event.ts_event / 1e9)
                
                # Update CVD
                cvd = self.cvds.get(self.symbol)
                if cvd:
                    cvd.add_trade(size, trade_side, timestamp=event.ts_event / 1e9)
                
                # Update volume profile
                profile = self.profiles.get(self.symbol)
                if profile:
                    profile.add_trade(price, size)
                
                # Update absorption detector
                absorber = self.absorbers.get(self.symbol)
                if absorber:
                    absorber.on_trade(price, size, trade_side, timestamp=event.ts_event / 1e9)
                
                # Update metrics (VWAP and RVOL)
                metrics = self.metrics.get(self.symbol)
                if metrics:
                    metrics.add_trade(price, size, timestamp=event.ts_event / 1e9)
                    
        except Exception as e:
            self.logger.debug(f"Event processing error: {e}")
    
    def _update_book(self, side: str, price: float, size: int, action: str):
        """Update the order book."""
        book = self.books.get(self.symbol)
        if not book:
            return
        
        target = book.bids if side == 'BID' else book.asks
        rounded_price = round(price, 2)
        
        if action == 'add':
            target[rounded_price] = target.get(rounded_price, 0) + size
        elif action == 'remove':
            if rounded_price in target:
                target[rounded_price] -= size
                if target[rounded_price] <= 0:
                    del target[rounded_price]
    
    def _get_best_bid(self) -> float:
        """Get current best bid price."""
        book = self.books.get(self.symbol)
        if book and book.bids:
            return max(book.bids.keys())
        return 0.0
    
    def _get_best_ask(self) -> float:
        """Get current best ask price."""
        book = self.books.get(self.symbol)
        if book and book.asks:
            return min(book.asks.keys())
        return 0.0
    
    def get_full_state(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get the complete market state (same format as IBKRConnector).
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
                "TIME_SESSION": metrics.get_time_session(timestamp=self.current_simulation_time),
                
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
                "TAPE_SENTIMENT": tape.get_sentiment(timestamp=self.current_simulation_time),
                "TAPE_DELTA_1S": tape.get_delta(1, timestamp=self.current_simulation_time),
                "TAPE_DELTA_5S": tape.get_delta(5, timestamp=self.current_simulation_time),
                "LARGE_PRINTS_1M": [
                    {"price": lp.price, "size": lp.size, "side": lp.side, 
                     "secs_ago": lp.secs_ago}
                    for lp in tape.get_large_prints(timestamp=self.current_simulation_time)
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
                "RVOL_DAY": metrics.get_rvol(timestamp=self.current_simulation_time),
                
                # Absorption
                "ABSORPTION_DETECTED": absorption["detected"],
                "ABSORPTION_SIDE": absorption["side"],
                "ABSORPTION_PRICE": absorption["price"]
            }
        }
    
    def stop(self):
        """Stop the replay."""
        self.is_replaying = False
        self.logger.info("Replay stopped")
    
    def get_stats(self) -> Dict:
        """Get replay statistics."""
        return {
            "date": self.date,
            "events_processed": self.events_processed,
            "trades_count": self.trades_count,
            "total_events": len(self.events),
            "available_dates": self.available_dates
        }


# Quick test function
async def test_replay():
    """Quick test of the replay functionality."""
    logging.basicConfig(level=logging.INFO)
    
    replayer = MBOReplayConnector(
        data_dir="BacktestData/TSLA-L3DATA",
        date="20251023"  # Pick a date
    )
    replayer.subscribe_market_data("TSLA")
    
    # Callback for state updates
    async def on_update(state):
        mkt = state.get("MARKET_STATE", {})
        print(f"LAST: {mkt.get('LAST', 0):.2f} | "
              f"IMB: {mkt.get('L2_IMBALANCE', 0):.2f} | "
              f"CVD: {mkt.get('CVD_TREND', '?')}")
    
    # Run replay at 100x speed
    await replayer.start_replay(speed=100, on_state_update=on_update)
    
    print(f"\nStats: {replayer.get_stats()}")


if __name__ == "__main__":
    asyncio.run(test_replay())
