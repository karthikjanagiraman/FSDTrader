#!/usr/bin/env python3
"""
FSDTrader: Data Replay Layer
Reads Databento MBP-10 (L2) files and simulates IBKR-style market data events.

MBP-10 provides 10 levels of aggregated bid/ask depth per update.

Usage:
    replayer = DataReplayConnector(data_dir="BacktestData/TSLA-L2DATA", date="20260112")
    replayer.subscribe_market_data("TSLA")
    await replayer.start_replay(speed=1.0)
"""
import logging
import asyncio
import time
from pathlib import Path
from typing import Optional, Dict, List, Callable, Any
from datetime import datetime

import databento as db

# Import our market data classes
from market_data import (
    OrderBook, TapeStream, FootprintTracker,
    CumulativeDelta, VolumeProfile, AbsorptionDetector, MarketMetrics
)


logger = logging.getLogger(__name__)


class DataReplayConnector:
    """
    Replays Databento MBP-10 (L2) data as if it were live IBKR data.
    Implements the same interface as IBKRConnector.
    """

    def __init__(self, data_dir: str, date: str = None, symbol: str = "TSLA"):
        self.data_dir = Path(data_dir)
        self.date = date
        self.symbol = symbol
        self.logger = logging.getLogger("DATA_REPLAY")

        # Find available dates
        self.available_dates = self._find_available_dates()
        if not self.available_dates:
            raise ValueError(f"No MBP-10 data files found in {self.data_dir}")

        if date and date not in self.available_dates:
            raise ValueError(f"Date {date} not found. Available: {self.available_dates[:5]}...")

        # If no date specified, use first available
        if not date:
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
        self.current_simulation_time: float = 0.0

        # Last known prices for trade detection
        self._last_price: float = 0.0
        self._last_best_bid: float = 0.0
        self._last_best_ask: float = 0.0

        # Stats
        self.events_processed: int = 0
        self.trades_count: int = 0

    def _find_available_dates(self) -> List[str]:
        """Find all available date files in the data directory."""
        dates = []
        for f in self.data_dir.glob("*.mbp-10.dbn.zst"):
            # Extract date from filename like "xnas-itch-20260112.mbp-10.dbn.zst"
            parts = f.stem.split("-")
            if len(parts) >= 3:
                date_str = parts[2].split(".")[0]
                if date_str.isdigit() and len(date_str) == 8:
                    dates.append(date_str)
        return sorted(dates)

    def _load_data_file(self, date: str):
        """Load MBP-10 data file for a specific date."""
        filename = f"xnas-itch-{date}.mbp-10.dbn.zst"
        filepath = self.data_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        self.logger.info(f"Loading MBP-10 data from {filepath}...")

        # Read the DBN file
        store = db.DBNStore.from_file(filepath)

        # Convert to list of records
        self.events = list(store)
        self.event_index = 0

        self.logger.info(f"Loaded {len(self.events):,} MBP-10 events for {date}")

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
        self._load_data_file(self.date)

        self.logger.info(f"Subscribed to {symbol} L2 replay data")

    async def start_replay(
        self,
        speed: float = 1.0,
        on_state_update: Callable = None,
        start_time: str = "09:30:00",
        end_time: str = "16:00:00",
        decision_interval: float = 5.0
    ):
        """
        Start replaying the data.

        Args:
            speed: Playback speed (1.0 = real-time, 10.0 = 10x, 0 = as fast as possible)
            on_state_update: Callback called every decision_interval of SIMULATION time
            start_time: Start time in HH:MM:SS (default market open)
            end_time: End time in HH:MM:SS (default market close)
            decision_interval: Seconds of simulation time between decisions (default 5.0)
        """
        self.replay_speed = speed
        self.is_replaying = True

        # Filter events to market hours
        events_to_replay = self._filter_market_hours(start_time, end_time)
        total_events = len(events_to_replay)

        if total_events == 0:
            self.logger.warning("No events to replay in the specified time range")
            return

        self.logger.info(f"Starting L2 replay: {total_events:,} events at {speed}x speed")
        self.logger.info(f"Decision interval: {decision_interval}s of simulation time")

        # Initialize simulation time tracking
        first_event_ts = events_to_replay[0].ts_event / 1e9
        self.current_simulation_time = first_event_ts
        last_decision_sim_time = first_event_ts

        last_ts = None
        last_progress_log = time.time()

        for i, event in enumerate(events_to_replay):
            if not self.is_replaying:
                break

            # Process the event
            self._process_mbp_event(event)
            self.events_processed += 1

            # Update simulation time
            self.current_simulation_time = event.ts_event / 1e9

            # Check if enough SIMULATION time has passed for a decision
            sim_time_elapsed = self.current_simulation_time - last_decision_sim_time

            if on_state_update and sim_time_elapsed >= decision_interval:
                # Get current market state
                state = self.get_full_state(self.symbol)
                if state:
                    state["SIM_TIMESTAMP"] = self.current_simulation_time
                    await on_state_update(state)

                last_decision_sim_time = self.current_simulation_time

            # Timing control for playback speed
            if speed > 0 and last_ts is not None:
                event_ts = event.ts_event
                time_diff_ns = event_ts - last_ts
                sleep_time = (time_diff_ns / 1e9) / speed

                if sleep_time > 0.001:
                    await asyncio.sleep(min(sleep_time, 0.1))

            last_ts = event.ts_event

            # Progress logging
            if time.time() - last_progress_log > 2.0:
                pct = (i / total_events) * 100
                sim_time_str = time.strftime('%H:%M:%S', time.gmtime(self.current_simulation_time % 86400))
                self.logger.info(
                    f"Replay: {pct:.1f}% | Sim time: {sim_time_str} | "
                    f"Events: {i:,}/{total_events:,} | Trades: {self.trades_count:,}"
                )
                last_progress_log = time.time()

        self.is_replaying = False
        self.logger.info(f"Replay complete: {self.events_processed:,} events, {self.trades_count:,} trades")

    def _filter_market_hours(self, start_time: str, end_time: str) -> List:
        """Filter events to market hours only."""
        start_parts = [int(x) for x in start_time.split(":")]
        end_parts = [int(x) for x in end_time.split(":")]

        start_secs = start_parts[0] * 3600 + start_parts[1] * 60 + start_parts[2]
        end_secs = end_parts[0] * 3600 + end_parts[1] * 60 + end_parts[2]

        filtered = []
        for event in self.events:
            ts_ns = event.ts_event
            ts_secs = ts_ns / 1e9

            # Get time of day in UTC
            time_of_day_utc = ts_secs % 86400

            # Convert UTC to Eastern Time
            dt = datetime.utcfromtimestamp(ts_secs)
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

    def _process_mbp_event(self, event):
        """
        Process a single MBP-10 event and update all analyzers.

        MBP-10 event fields:
        - ts_event: nanoseconds timestamp
        - action: Event action type (ADD, TRADE, etc.)
        - side: 'BID', 'ASK', or 'NONE' (both sides updated)
        - price: Top-of-book price in fixed-point
        - size: Top-of-book size
        - levels: Array of 10 BidAskPair with bid_px, ask_px, bid_sz, ask_sz
        """
        try:
            timestamp = event.ts_event / 1e9
            action = str(event.action.name) if hasattr(event.action, 'name') else str(event.action)

            # Extract the 10 levels of depth
            levels = event.levels if hasattr(event, 'levels') else []

            # Update the order book from levels
            book = self.books.get(self.symbol)
            if book and levels:
                book.bids.clear()
                book.asks.clear()

                for level in levels:
                    # Bid side
                    if hasattr(level, 'bid_px') and level.bid_px > 0:
                        bid_price = level.bid_px / 1e9 if level.bid_px > 1e6 else level.bid_px
                        if 0 < bid_price < 10000 and level.bid_sz > 0:
                            book.bids[round(bid_price, 2)] = level.bid_sz

                    # Ask side
                    if hasattr(level, 'ask_px') and level.ask_px > 0:
                        ask_price = level.ask_px / 1e9 if level.ask_px > 1e6 else level.ask_px
                        if 0 < ask_price < 10000 and level.ask_sz > 0:
                            book.asks[round(ask_price, 2)] = level.ask_sz

            # Get current best bid/ask
            best_bid = self._get_best_bid()
            best_ask = self._get_best_ask()

            # Detect trades from TRADE action
            if action == 'TRADE':
                price = event.price / 1e9 if event.price > 1e6 else event.price
                size = event.size

                if 0 < price < 10000 and size > 0:
                    # Determine trade side
                    if best_ask > 0 and abs(price - best_ask) < 0.01:
                        trade_side = "BUY"
                    elif best_bid > 0 and abs(price - best_bid) < 0.01:
                        trade_side = "SELL"
                    else:
                        trade_side = "BUY" if price > self._last_price else "SELL"

                    self._process_trade(price, size, timestamp, trade_side)

            # Update last known values
            if best_bid > 0:
                self._last_best_bid = best_bid
            if best_ask > 0:
                self._last_best_ask = best_ask

        except Exception as e:
            self.logger.debug(f"MBP event processing error: {e}")

    def _process_trade(self, price: float, size: int, timestamp: float, trade_side: str):
        """Process a detected trade and update all analyzers."""
        self.trades_count += 1
        self._last_price = price

        best_bid = self._get_best_bid()
        best_ask = self._get_best_ask()

        # Update tape
        tape = self.tapes.get(self.symbol)
        if tape:
            tape.update_quotes(best_bid, best_ask)

            class MockTick:
                pass
            tick = MockTick()
            tick.price = price
            tick.size = size
            tick.time = timestamp
            tape.on_tick(tick, timestamp=timestamp)

        # Update footprint
        footprint = self.footprints.get(self.symbol)
        if footprint:
            footprint.on_trade(price, size, trade_side, timestamp=timestamp)

        # Update CVD
        cvd = self.cvds.get(self.symbol)
        if cvd:
            cvd.add_trade(size, trade_side, timestamp=timestamp)

        # Update volume profile
        profile = self.profiles.get(self.symbol)
        if profile:
            profile.add_trade(price, size)

        # Update absorption detector
        absorber = self.absorbers.get(self.symbol)
        if absorber:
            absorber.on_trade(price, size, trade_side, timestamp=timestamp)

        # Update metrics (VWAP, HOD, LOD, RVOL)
        metrics = self.metrics.get(self.symbol)
        if metrics:
            metrics.add_trade(price, size, timestamp=timestamp)

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
        vel_label, vel_tps = tape.get_velocity(timestamp=self.current_simulation_time)
        vah, val = profile.get_value_area()
        absorption = absorber.detect()

        return {
            "MARKET_STATE": {
                "TICKER": symbol,
                "LAST": last_price,
                "VWAP": metrics.vwap,
                "TIME_SESSION": metrics.get_time_session(timestamp=self.current_simulation_time),

                # Level 2 (DOM) - 10-level depth from MBP-10
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
            "data_format": "MBP-10",
            "events_processed": self.events_processed,
            "trades_count": self.trades_count,
            "total_events": len(self.events),
            "available_dates": self.available_dates
        }


# Backward compatibility alias
MBOReplayConnector = DataReplayConnector


# Quick test function
async def test_replay():
    """Quick test of the replay functionality."""
    logging.basicConfig(level=logging.INFO)

    replayer = DataReplayConnector(
        data_dir="BacktestData/TSLA-L2DATA",
    )
    replayer.subscribe_market_data("TSLA")

    print(f"Available dates: {replayer.available_dates}")

    # Callback for state updates
    call_count = [0]

    async def on_update(state):
        call_count[0] += 1
        mkt = state.get("MARKET_STATE", {})
        if call_count[0] <= 5:
            print(
                f"Update {call_count[0]}: "
                f"LAST=${mkt.get('LAST', 0):.2f} | "
                f"Spread=${mkt.get('SPREAD', 0):.3f} | "
                f"IMB={mkt.get('L2_IMBALANCE', 0):.2f} | "
                f"CVD={mkt.get('CVD_TREND', '?')}"
            )

    # Run replay for first minute at max speed
    await replayer.start_replay(
        speed=0,
        on_state_update=on_update,
        start_time="09:30:00",
        end_time="09:31:00",
        decision_interval=5.0
    )

    print(f"\nStats: {replayer.get_stats()}")
    print(f"Total callbacks: {call_count[0]}")


if __name__ == "__main__":
    asyncio.run(test_replay())
