#!/usr/bin/env python3
"""
FSDTrader Brain Module: Context Builder

Transforms raw market state into rich, human-readable context for the LLM.
"""
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from collections import deque

from .types import Decision


# Type alias for price history entry: (timestamp, price, label)
PriceHistoryEntry = Tuple[datetime, float, Optional[str]]


def _fmt_size(val: float, signed: bool = False) -> str:
    """Format size for display - stocks use commas, crypto uses decimals.

    Args:
        val: The size value to format
        signed: If True, always show + or - prefix

    Returns:
        Formatted string appropriate for the value magnitude
    """
    prefix = "+" if signed and val > 0 else ""

    # Whole numbers (stocks): use comma format
    if val == int(val) and abs(val) >= 1:
        return f"{prefix}{int(val):,}"
    elif abs(val) >= 1:
        # Float >= 1 with fractional part (e.g., 1.5 BTC)
        return f"{prefix}{val:,.4f}"
    elif val != 0:
        # Small fractional values (crypto)
        return f"{prefix}{val:.8f}"
    else:
        return "0"


class ContextBuilder:
    """
    Builds comprehensive context for the LLM from raw market state.

    The context is formatted as human-readable markdown-style text
    that helps the LLM understand the current market situation.
    """

    def __init__(self, max_price_history: int = 24):
        """
        Initialize the context builder.

        Args:
            max_price_history: Maximum number of price history entries to keep
                              (24 entries at 5-second intervals = 2 minutes)
        """
        self._price_history: deque = deque(maxlen=max_price_history)

    def record_price(
        self,
        price: float,
        timestamp: Optional[datetime] = None,
        label: Optional[str] = None
    ) -> None:
        """
        Record a price point for the price history.

        Args:
            price: The price to record
            timestamp: When the price was recorded (defaults to now)
            label: Optional label for significant price events
                   (e.g., "Session high", "Broke VWAP", "Support test")
        """
        if timestamp is None:
            timestamp = datetime.now()
        self._price_history.append((timestamp, price, label))

    def clear_price_history(self) -> None:
        """Clear the price history (e.g., at start of new session)."""
        self._price_history.clear()

    def build(
        self,
        market_state: Dict[str, Any],
        account_state: Dict[str, Any],
        history: List[Decision],
        active_orders: Optional[List[Dict]] = None
    ) -> str:
        """
        Build the complete context string for the LLM.

        Args:
            market_state: Raw market state from MarketData module
            account_state: Current account/position state
            history: Recent decision history
            active_orders: Active orders (optional)

        Returns:
            Formatted context string
        """
        sections = [
            self._build_position_section(account_state, active_orders),
            self._build_session_context(market_state),
            self._build_price_history(market_state),
            self._build_key_levels(market_state),
            self._build_order_book(market_state),
            self._build_tape_analysis(market_state),
            self._build_delta_flow(market_state),
            self._build_absorption(market_state),
            self._build_history(history),
            self._build_raw_data(market_state),
            self._build_closing_question(),
        ]

        return "\n\n".join(sections)

    def _build_position_section(
        self,
        account_state: Dict[str, Any],
        active_orders: Optional[List[Dict]] = None
    ) -> str:
        """Build the current position section."""
        position_side = account_state.get("POSITION_SIDE", "FLAT")
        position_size = account_state.get("POSITION", 0)

        if position_side == "FLAT" or position_size == 0:
            return """## CURRENT POSITION

FLAT - Looking for entry opportunity."""

        avg_entry = account_state.get("AVG_ENTRY", 0)
        unrealized_pnl = account_state.get("UNREALIZED_PL", 0)
        daily_pnl = account_state.get("DAILY_PL", 0)
        daily_trades = account_state.get("DAILY_TRADES", 0)

        # Calculate P&L per share
        pnl_per_share = unrealized_pnl / position_size if position_size > 0 else 0

        # Get stop and target from active orders if available
        stop_price = "N/A"
        target_price = "N/A"
        if active_orders:
            for order in active_orders:
                if order.get("type") == "STOP":
                    stop_price = f"${order.get('price', 0):.2f}"
                elif order.get("type") == "LIMIT":
                    target_price = f"${order.get('price', 0):.2f}"

        pnl_sign = "+" if unrealized_pnl >= 0 else ""

        return f"""## CURRENT POSITION

{position_side} {position_size} shares @ ${avg_entry:.2f}
├── Unrealized P&L: {pnl_sign}${pnl_per_share:.2f} ({pnl_sign}${unrealized_pnl:.2f})
├── Stop: {stop_price} | Target: {target_price}
├── Daily P&L: ${daily_pnl:.2f}
└── Daily Trades: {daily_trades}"""

    def _build_session_context(self, market_state: Dict[str, Any]) -> str:
        """Build the session context section with trend analysis."""
        mkt = market_state.get("MARKET_STATE", market_state)

        time_session = mkt.get("TIME_SESSION", "UNKNOWN")
        last_price = mkt.get("LAST", 0)
        vwap = mkt.get("VWAP", 0)
        spread = mkt.get("SPREAD", 0)
        rvol = mkt.get("RVOL_DAY", 1.0)
        cvd_session = mkt.get("CVD_SESSION", 0)
        cvd_trend = mkt.get("CVD_TREND", "FLAT")

        # Determine spread limit based on session
        spread_limit = 0.15 if time_session == "OPEN_DRIVE" else 0.08
        spread_ok = spread <= spread_limit
        spread_status = "OK" if spread_ok else "TOO WIDE"

        # Determine session trend
        price_vs_vwap = "ABOVE" if last_price > vwap else "BELOW" if last_price < vwap else "AT"

        # Determine overall trend
        if cvd_trend == "RISING" and price_vs_vwap == "ABOVE":
            session_trend = "BULLISH"
            trend_interpretation = "Buyers in control"
        elif cvd_trend == "FALLING" and price_vs_vwap == "BELOW":
            session_trend = "BEARISH"
            trend_interpretation = "Sellers in control"
        else:
            session_trend = "MIXED"
            trend_interpretation = "No clear trend"

        # RVOL interpretation
        if rvol >= 2.0:
            rvol_label = "High activity"
        elif rvol >= 1.5:
            rvol_label = "Above average"
        elif rvol >= 1.0:
            rvol_label = "Average"
        else:
            rvol_label = "Below average"

        # Build counter-trend warning if applicable
        counter_trend_warning = ""
        # This would be enhanced based on what trade direction the model might consider

        # Format CVD using helper
        cvd_formatted = _fmt_size(cvd_session, signed=True)

        return f"""## SESSION CONTEXT

├── Time: {datetime.now().strftime("%H:%M:%S")} ET
├── Session: {time_session}
├── Spread Limit: ${spread_limit:.2f}
│
├── Session Trend: {session_trend}
│   ├── CVD: {cvd_formatted} ({cvd_trend})
│   ├── Price vs VWAP: {price_vs_vwap} (${last_price:.2f} vs ${vwap:.2f})
│   └── Interpretation: {trend_interpretation}
│
├── RVOL: {rvol:.1f}x ({rvol_label})
└── Current Spread: ${spread:.2f} ({spread_status})"""

    def _build_price_history(self, market_state: Dict[str, Any]) -> str:
        """
        Build the price history section showing recent price movement.

        Per prompt guide v3: Shows last 2 minutes of price action to help
        the LLM identify trends like "lower highs" or "higher lows".
        """
        if not self._price_history:
            return """## PRICE ACTION (Last 2 Minutes)

(No price history available yet)"""

        mkt = market_state.get("MARKET_STATE", market_state)
        current_price = mkt.get("LAST", 0)
        vwap = mkt.get("VWAP", 0)
        hod = mkt.get("HOD", 0)
        lod = mkt.get("LOD", 0)

        lines = ["## PRICE ACTION (Last 2 Minutes)", ""]

        # Get price history entries
        history_list = list(self._price_history)

        # Calculate pattern indicators
        if len(history_list) >= 4:
            # Check for lower highs / higher lows pattern
            prices = [entry[1] for entry in history_list]
            recent_prices = prices[-8:] if len(prices) >= 8 else prices

            # Find local highs and lows in recent prices
            highs = []
            lows = []
            for i in range(1, len(recent_prices) - 1):
                if recent_prices[i] > recent_prices[i-1] and recent_prices[i] > recent_prices[i+1]:
                    highs.append(recent_prices[i])
                if recent_prices[i] < recent_prices[i-1] and recent_prices[i] < recent_prices[i+1]:
                    lows.append(recent_prices[i])

            # Determine pattern
            pattern = None
            if len(highs) >= 2:
                if all(highs[i] < highs[i-1] for i in range(1, len(highs))):
                    pattern = "Lower highs forming - potential weakness"
                elif all(highs[i] > highs[i-1] for i in range(1, len(highs))):
                    pattern = "Higher highs forming - potential strength"
            if len(lows) >= 2:
                if all(lows[i] > lows[i-1] for i in range(1, len(lows))):
                    if pattern:
                        pattern += ", higher lows"
                    else:
                        pattern = "Higher lows forming - buyers stepping in"
                elif all(lows[i] < lows[i-1] for i in range(1, len(lows))):
                    if pattern:
                        pattern += ", lower lows"
                    else:
                        pattern = "Lower lows forming - sellers in control"

        # Format the price entries (show last 8-12 entries for readability)
        display_entries = history_list[-12:]
        for timestamp, price, label in display_entries:
            # Handle both datetime objects and Unix timestamps (float)
            if isinstance(timestamp, (int, float)):
                time_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
            else:
                time_str = timestamp.strftime("%H:%M:%S")

            # Add context labels
            context_parts = []
            if label:
                context_parts.append(label)
            if price == hod and hod > 0:
                context_parts.append("HOD")
            elif price == lod and lod > 0:
                context_parts.append("LOD")
            if abs(price - vwap) < 0.02 and vwap > 0:
                context_parts.append("at VWAP")

            context_str = f"  ({', '.join(context_parts)})" if context_parts else ""
            lines.append(f"{time_str}  ${price:.2f}{context_str}")

        # Add current price with NOW marker
        now_str = datetime.now().strftime("%H:%M:%S")
        lines.append(f"{now_str}  ${current_price:.2f}  <- NOW")

        # Add pattern interpretation if detected
        if len(history_list) >= 4:
            lines.append("")
            if pattern:
                lines.append(f"PATTERN: {pattern}")
            else:
                # Calculate simple trend from first to last
                first_price = history_list[0][1]
                price_change = current_price - first_price
                pct_change = (price_change / first_price * 100) if first_price > 0 else 0
                if abs(pct_change) < 0.05:
                    lines.append("PATTERN: Consolidating, no clear direction")
                elif price_change > 0:
                    lines.append(f"PATTERN: Uptrend (+${price_change:.2f}, +{pct_change:.2f}%)")
                else:
                    lines.append(f"PATTERN: Downtrend (${price_change:.2f}, {pct_change:.2f}%)")

        return "\n".join(lines)

    def _build_key_levels(self, market_state: Dict[str, Any]) -> str:
        """Build the key levels section."""
        mkt = market_state.get("MARKET_STATE", market_state)

        last_price = mkt.get("LAST", 0)
        hod = mkt.get("HOD", 0)
        lod = mkt.get("LOD", 0)
        vwap = mkt.get("VWAP", 0)
        hod_lod_loc = mkt.get("HOD_LOD_LOC", "UNKNOWN")
        distance_to_hod = mkt.get("DISTANCE_TO_HOD_PCT", 0)

        vp_poc = mkt.get("VP_POC", 0)
        vp_vah = mkt.get("VP_VAH", 0)
        vp_val = mkt.get("VP_VAL", 0)

        # Calculate distance to LOD
        if last_price > 0 and lod > 0:
            distance_to_lod = abs(last_price - lod) / last_price * 100
        else:
            distance_to_lod = 0

        # Location indicator
        hod_indicator = " <- TESTING" if hod_lod_loc == "TESTING_HOD" else ""
        lod_indicator = " <- TESTING" if hod_lod_loc == "TESTING_LOD" else ""

        return f"""## KEY LEVELS

├── HOD: ${hod:.2f} ({distance_to_hod:.2f}% away){hod_indicator}
├── LOD: ${lod:.2f} ({distance_to_lod:.2f}% away){lod_indicator}
├── VWAP: ${vwap:.2f}
│
└── Volume Profile:
    ├── POC: ${vp_poc:.2f}
    ├── VAH: ${vp_vah:.2f}
    └── VAL: ${vp_val:.2f}"""

    def _build_order_book(self, market_state: Dict[str, Any]) -> str:
        """
        Build the order book (Level 2) section.

        Uses smart L2 processing:
        - Dynamic stacks: Shows levels from best price up to first major wall
        - Percentile-based walls: MASSIVE (>p95), MAJOR (>p90), MINOR (>p75)
        - Cumulative sizes: Multiple market makers aggregated at each price
        """
        mkt = market_state.get("MARKET_STATE", market_state)

        l2_imbalance = mkt.get("L2_IMBALANCE", 1.0)
        spread = mkt.get("SPREAD", 0)
        book_stats = mkt.get("BOOK_STATS", {})

        # Prefer dynamic stacks (up to major wall), fallback to simple stacks
        bid_stack = mkt.get("BID_STACK_TO_WALL", mkt.get("BID_STACK", []))
        ask_stack = mkt.get("ASK_STACK_TO_WALL", mkt.get("ASK_STACK", []))
        walls = mkt.get("DOM_WALLS", [])

        # Interpret imbalance
        if l2_imbalance >= 1.5:
            imbalance_label = "BULLISH - strong bid support"
        elif l2_imbalance <= 0.6:
            imbalance_label = "BEARISH - heavy ask pressure"
        else:
            imbalance_label = "NEUTRAL"

        # Book statistics for context
        total_bid_vol = book_stats.get("total_bid_volume", 0)
        total_ask_vol = book_stats.get("total_ask_volume", 0)
        bid_levels = book_stats.get("bid_levels", 0)
        ask_levels = book_stats.get("ask_levels", 0)

        # Use module-level helper for size formatting
        fmt_size = _fmt_size

        # Format bid stack (dynamic depth - up to major wall)
        bid_lines = []
        for level in bid_stack[:10]:  # Cap at 10 for readability
            if isinstance(level, dict):
                price = level.get("price", 0)
                size = level.get("size", 0)
                cum_size = level.get("cumulative_size", size)
                bid_lines.append(f"│   ├── ${price:.2f} x {fmt_size(size)} (cum: {fmt_size(cum_size)})")
            elif isinstance(level, (list, tuple)) and len(level) >= 2:
                bid_lines.append(f"│   ├── ${level[0]:.2f} x {fmt_size(level[1])}")
        if bid_lines:
            bid_lines[-1] = bid_lines[-1].replace("├──", "└──")  # Last item
        bid_section = "\n".join(bid_lines) if bid_lines else "│   └── (No bids)"

        # Format ask stack (dynamic depth - up to major wall)
        ask_lines = []
        for level in ask_stack[:10]:  # Cap at 10 for readability
            if isinstance(level, dict):
                price = level.get("price", 0)
                size = level.get("size", 0)
                cum_size = level.get("cumulative_size", size)
                ask_lines.append(f"│   ├── ${price:.2f} x {fmt_size(size)} (cum: {fmt_size(cum_size)})")
            elif isinstance(level, (list, tuple)) and len(level) >= 2:
                ask_lines.append(f"│   ├── ${level[0]:.2f} x {fmt_size(level[1])}")
        if ask_lines:
            ask_lines[-1] = ask_lines[-1].replace("├──", "└──")  # Last item
        ask_section = "\n".join(ask_lines) if ask_lines else "│   └── (No asks)"

        # Format walls with tier indicators
        # Tier meanings: MASSIVE (>95th pct), MAJOR (>90th pct), MINOR (>75th pct)
        wall_lines = []
        for wall in walls[:5]:  # Show up to 5 most significant walls
            if isinstance(wall, dict):
                side = wall.get("side", "?")
                price = wall.get("price", 0)
                size = wall.get("size", 0)
                tier = wall.get("tier", "MINOR")
                dist = wall.get("distance_pct", 0)
                pctl = wall.get("percentile", 0)

                # Add visual indicator based on tier
                tier_icon = "🔴" if tier == "MASSIVE" else "🟠" if tier == "MAJOR" else "🟡"
                wall_lines.append(
                    f"    ├── {tier_icon} {side} ${price:.2f} x {fmt_size(size)} "
                    f"({tier}, p{pctl:.0f}) - {dist:.2f}% away"
                )
        if wall_lines:
            wall_lines[-1] = wall_lines[-1].replace("├──", "└──")  # Last item
        wall_section = "\n".join(wall_lines) if wall_lines else "    └── (No significant walls detected)"

        # Book depth summary
        depth_summary = ""
        if total_bid_vol > 0 or total_ask_vol > 0:
            depth_summary = f"""│
├── Book Depth: {bid_levels} bid levels, {ask_levels} ask levels
├── Total Volume: {fmt_size(total_bid_vol)} bid / {fmt_size(total_ask_vol)} ask"""

        return f"""## ORDER BOOK (Level 2)

├── L2 Imbalance: {l2_imbalance:.1f} ({imbalance_label})
├── Spread: ${spread:.2f}{depth_summary}
│
├── Bid Stack (to major wall):
{bid_section}
│
├── Ask Stack (to major wall):
{ask_section}
│
└── Walls (significant liquidity):
{wall_section}"""

    def _build_tape_analysis(self, market_state: Dict[str, Any]) -> str:
        """Build the tape analysis section."""
        mkt = market_state.get("MARKET_STATE", market_state)

        velocity = mkt.get("TAPE_VELOCITY", "UNKNOWN")
        velocity_tps = mkt.get("TAPE_VELOCITY_TPS", 0)
        sentiment = mkt.get("TAPE_SENTIMENT", "NEUTRAL")
        delta_1s = mkt.get("TAPE_DELTA_1S", 0)
        delta_5s = mkt.get("TAPE_DELTA_5S", 0)
        large_prints = mkt.get("LARGE_PRINTS_1M", [])

        # Format large prints - handle crypto fractional sizes
        print_lines = []
        for lp in large_prints[:3]:
            if isinstance(lp, dict):
                side = lp.get("side", "?")
                price = lp.get("price", 0)
                size = lp.get("size", 0)
                secs_ago = lp.get("secs_ago", 0)
                print_lines.append(f"    └── {side} {_fmt_size(size)} @ ${price:.2f} ({secs_ago:.0f}s ago)")
        prints_section = "\n".join(print_lines) if print_lines else "    └── (No large prints)"

        # Format deltas using helper
        delta_1s_str = _fmt_size(delta_1s, signed=True)
        delta_5s_str = _fmt_size(delta_5s, signed=True)

        return f"""## TAPE ANALYSIS

├── Velocity: {velocity} ({velocity_tps:.1f} trades/sec)
├── Sentiment: {sentiment}
│
├── Delta:
│   ├── 1-second: {delta_1s_str}
│   └── 5-second: {delta_5s_str}
│
└── Large Prints (last 60s):
{prints_section}"""

    def _build_delta_flow(self, market_state: Dict[str, Any]) -> str:
        """Build the delta and flow section."""
        mkt = market_state.get("MARKET_STATE", market_state)

        footprint = mkt.get("FOOTPRINT_CURR_BAR", {})
        cvd_session = mkt.get("CVD_SESSION", 0)
        cvd_trend = mkt.get("CVD_TREND", "FLAT")
        cvd_slope = mkt.get("CVD_SLOPE_5M", 0)

        # Format footprint
        if footprint:
            fp_open = footprint.get("open", 0)
            fp_high = footprint.get("high", 0)
            fp_low = footprint.get("low", 0)
            fp_close = footprint.get("close", 0)
            fp_delta = footprint.get("delta", 0)
            fp_delta_pct = footprint.get("delta_pct", 0)
            fp_volume = footprint.get("volume", 0)
            fp_poc = footprint.get("poc", 0)

            # Format delta/volume using helper
            delta_str = _fmt_size(fp_delta, signed=True)
            vol_str = _fmt_size(fp_volume)

            footprint_section = f"""├── Footprint (Current Bar):
│   ├── OHLC: ${fp_open:.2f} / ${fp_high:.2f} / ${fp_low:.2f} / ${fp_close:.2f}
│   ├── Delta: {delta_str} ({fp_delta_pct:+.1f}%)
│   ├── Volume: {vol_str}
│   └── POC: ${fp_poc:.2f}"""
        else:
            footprint_section = "├── Footprint: (No data)"

        # Format CVD using helper
        cvd_formatted = _fmt_size(cvd_session, signed=True)

        return f"""## DELTA & FLOW

{footprint_section}
│
├── CVD Session: {cvd_formatted}
├── CVD Trend: {cvd_trend}
└── CVD Slope (5m): {cvd_slope:+.2f}"""

    def _build_absorption(self, market_state: Dict[str, Any]) -> str:
        """Build the absorption section."""
        mkt = market_state.get("MARKET_STATE", market_state)

        detected = mkt.get("ABSORPTION_DETECTED", False)
        side = mkt.get("ABSORPTION_SIDE")
        price = mkt.get("ABSORPTION_PRICE")

        if not detected:
            return """## ABSORPTION

├── Detected: No
└── (No absorption pattern currently detected)"""

        # Interpret absorption
        if side == "BID":
            interpretation = "Strong support forming - buyers absorbing selling pressure"
        elif side == "ASK":
            interpretation = "Strong resistance forming - sellers absorbing buying pressure"
        else:
            interpretation = "Absorption detected"

        return f"""## ABSORPTION

├── Detected: YES
├── Side: {side}
├── Price: ${price:.2f}
└── Interpretation: {interpretation}"""

    def _build_history(self, history: List[Decision]) -> str:
        """Build the recent decisions section."""
        if not history:
            return """## YOUR RECENT DECISIONS

(No recent decisions)"""

        lines = ["## YOUR RECENT DECISIONS", ""]
        for decision in history[-3:]:  # Last 3 decisions
            lines.append(decision.format_for_context())
            lines.append("")

        return "\n".join(lines)

    def _build_raw_data(self, market_state: Dict[str, Any]) -> str:
        """Build the raw data section (JSON)."""
        mkt = market_state.get("MARKET_STATE", market_state)

        # Create a simplified version for context
        simplified = {
            "TICKER": mkt.get("TICKER"),
            "LAST": mkt.get("LAST"),
            "VWAP": mkt.get("VWAP"),
            "SPREAD": mkt.get("SPREAD"),
            "L2_IMBALANCE": mkt.get("L2_IMBALANCE"),
            "TAPE_VELOCITY": mkt.get("TAPE_VELOCITY"),
            "TAPE_SENTIMENT": mkt.get("TAPE_SENTIMENT"),
            "CVD_TREND": mkt.get("CVD_TREND"),
            "CVD_SESSION": mkt.get("CVD_SESSION"),
            "HOD": mkt.get("HOD"),
            "LOD": mkt.get("LOD"),
            "HOD_LOD_LOC": mkt.get("HOD_LOD_LOC"),
            "RVOL_DAY": mkt.get("RVOL_DAY"),
            "ABSORPTION_DETECTED": mkt.get("ABSORPTION_DETECTED"),
            "TIME_SESSION": mkt.get("TIME_SESSION"),
        }

        return f"""## RAW MARKET DATA

```json
{json.dumps(simplified, indent=2)}
```"""

    def _build_closing_question(self) -> str:
        """
        Build the closing question that prompts the LLM for a decision.

        Per prompt guide v3: End with a clear question that sets up
        the expected response format.
        """
        return """---

What's your read? Walk me through your analysis, then call the appropriate tool for your decision."""


# Module-level instance for convenience
_builder = ContextBuilder()


def build_context(
    market_state: Dict[str, Any],
    account_state: Dict[str, Any],
    history: List[Decision],
    active_orders: Optional[List[Dict]] = None
) -> str:
    """
    Convenience function to build context.

    Args:
        market_state: Raw market state
        account_state: Account/position state
        history: Recent decision history
        active_orders: Active orders (optional)

    Returns:
        Formatted context string
    """
    return _builder.build(market_state, account_state, history, active_orders)


def record_price(
    price: float,
    timestamp: Optional[datetime] = None,
    label: Optional[str] = None
) -> None:
    """
    Record a price point for the price history.

    Call this function periodically (every ~5 seconds) to build up
    price history that will be included in the LLM context.

    Args:
        price: The price to record
        timestamp: When the price was recorded (defaults to now)
        label: Optional label for significant price events
               (e.g., "Session high", "Broke VWAP", "Support test")
    """
    _builder.record_price(price, timestamp, label)


def clear_price_history() -> None:
    """Clear the price history (e.g., at start of new session)."""
    _builder.clear_price_history()


def get_context_builder() -> ContextBuilder:
    """Get the module-level ContextBuilder instance."""
    return _builder
