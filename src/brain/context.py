#!/usr/bin/env python3
"""
FSDTrader Brain Module: Context Builder

Transforms raw market state into rich, human-readable context for the LLM.
"""
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from .types import Decision


class ContextBuilder:
    """
    Builds comprehensive context for the LLM from raw market state.

    The context is formatted as human-readable markdown-style text
    that helps the LLM understand the current market situation.
    """

    def __init__(self):
        pass

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
            self._build_key_levels(market_state),
            self._build_order_book(market_state),
            self._build_tape_analysis(market_state),
            self._build_delta_flow(market_state),
            self._build_absorption(market_state),
            self._build_history(history),
            self._build_raw_data(market_state),
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

        cvd_formatted = f"{cvd_session:+,}" if cvd_session != 0 else "0"

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

        # Format bid stack
        bid_lines = []
        for level in bid_stack[:3]:
            if isinstance(level, (list, tuple)) and len(level) >= 2:
                bid_lines.append(f"│   ├── ${level[0]:.2f} x {level[1]:,}")
        bid_section = "\n".join(bid_lines) if bid_lines else "│   └── (No bids)"

        # Format ask stack
        ask_lines = []
        for level in ask_stack[:3]:
            if isinstance(level, (list, tuple)) and len(level) >= 2:
                ask_lines.append(f"│   ├── ${level[0]:.2f} x {level[1]:,}")
        ask_section = "\n".join(ask_lines) if ask_lines else "│   └── (No asks)"

        # Format walls
        wall_lines = []
        for wall in walls[:3]:
            if isinstance(wall, dict):
                side = wall.get("side", "?")
                price = wall.get("price", 0)
                size = wall.get("size", 0)
                tier = wall.get("tier", "MINOR")
                dist = wall.get("distance_pct", 0)
                wall_lines.append(f"    ├── {side} ${price:.2f} x {size:,} ({tier}) - {dist:.2f}% away")
        wall_section = "\n".join(wall_lines) if wall_lines else "    └── (No significant walls)"

        return f"""## ORDER BOOK (Level 2)

├── L2 Imbalance: {l2_imbalance:.1f} ({imbalance_label})
├── Spread: ${spread:.2f}
│
├── Bid Stack:
{bid_section}
│
├── Ask Stack:
{ask_section}
│
└── Walls:
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

        # Format large prints
        print_lines = []
        for lp in large_prints[:3]:
            if isinstance(lp, dict):
                side = lp.get("side", "?")
                price = lp.get("price", 0)
                size = lp.get("size", 0)
                secs_ago = lp.get("secs_ago", 0)
                print_lines.append(f"    └── {side} {size:,} @ ${price:.2f} ({secs_ago:.0f}s ago)")
        prints_section = "\n".join(print_lines) if print_lines else "    └── (No large prints)"

        return f"""## TAPE ANALYSIS

├── Velocity: {velocity} ({velocity_tps:.1f} trades/sec)
├── Sentiment: {sentiment}
│
├── Delta:
│   ├── 1-second: {delta_1s:+,}
│   └── 5-second: {delta_5s:+,}
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

            footprint_section = f"""├── Footprint (Current Bar):
│   ├── OHLC: ${fp_open:.2f} / ${fp_high:.2f} / ${fp_low:.2f} / ${fp_close:.2f}
│   ├── Delta: {fp_delta:+,} ({fp_delta_pct:+.1f}%)
│   ├── Volume: {fp_volume:,}
│   └── POC: ${fp_poc:.2f}"""
        else:
            footprint_section = "├── Footprint: (No data)"

        cvd_formatted = f"{cvd_session:+,}" if cvd_session != 0 else "0"

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
