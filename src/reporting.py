#!/usr/bin/env python3
"""
FSDTrader: Backtest Reporting Module
Comprehensive logging and reporting for backtest runs.

Generates:
- Decision log (JSON lines)
- Trade log (JSON lines) 
- Performance summary
- Equity curve data
"""
import json
import time
import os
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional
from pathlib import Path


@dataclass
class Decision:
    """Single decision record."""
    timestamp: float
    loop_count: int
    price: float
    command: str
    l2_imbalance: float
    cvd_trend: str
    hod_lod_loc: str
    tape_sentiment: str
    spread: float
    position_side: str
    position_size: int
    unrealized_pnl: float
    daily_pnl: float
    execution_result: str


@dataclass
class Trade:
    """Single trade record."""
    timestamp: float
    action: str  # ENTRY, EXIT, STOP_HIT, TARGET_HIT
    side: str    # LONG or SHORT
    price: float
    size: int
    stop: float
    target: float
    pnl: float = 0.0
    hold_time_sec: float = 0.0


@dataclass
class BacktestReport:
    """Complete backtest report."""
    date: str
    symbol: str
    start_time: str
    end_time: str
    duration_sec: float
    
    # Event stats
    total_mbo_events: int = 0
    market_events: int = 0
    trades_in_data: int = 0
    
    # AI stats  
    total_decisions: int = 0
    enter_long_count: int = 0
    enter_short_count: int = 0
    update_stop_count: int = 0
    update_target_count: int = 0
    cancel_count: int = 0
    wait_count: int = 0
    
    # Trading stats
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    # P&L stats
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    max_drawdown: float = 0.0
    max_profit: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    
    # Risk stats
    max_position_held: int = 0
    avg_hold_time_sec: float = 0.0


class BacktestReporter:
    """
    Comprehensive backtest reporting.
    """
    
    def __init__(self, symbol: str, date: str, output_dir: str = "data/reports"):
        self.symbol = symbol
        self.date = date
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create run-specific dir
        self.run_id = f"{date}_{time.strftime('%H%M%S')}"
        self.run_dir = self.output_dir / self.run_id
        self.run_dir.mkdir(exist_ok=True)
        
        # Data storage
        self.decisions: List[Decision] = []
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []
        
        # State
        self.start_time = time.time()
        self.current_equity = 0.0
        self.peak_equity = 0.0
        self.max_drawdown = 0.0
        
        # File handles
        self.decision_file = open(self.run_dir / "decisions.jsonl", "w")
        self.trade_file = open(self.run_dir / "trades.jsonl", "w")
        
        print(f"📊 Report will be saved to: {self.run_dir}")
    
    def log_decision(self, state: Dict, command: str, result: Dict):
        """Log a single decision."""
        mkt = state.get("MARKET_STATE", {})
        acct = state.get("ACCOUNT_STATE", {})
        
        decision = Decision(
            timestamp=time.time(),
            loop_count=len(self.decisions) + 1,
            price=mkt.get("LAST", 0),
            command=command,
            l2_imbalance=mkt.get("L2_IMBALANCE", 0),
            cvd_trend=mkt.get("CVD_TREND", "?"),
            hod_lod_loc=mkt.get("HOD_LOD_LOC", "?"),
            tape_sentiment=mkt.get("TAPE_SENTIMENT", "?"),
            spread=mkt.get("SPREAD", 0),
            position_side=acct.get("POSITION_SIDE", "FLAT"),
            position_size=acct.get("POSITION", 0),
            unrealized_pnl=acct.get("UNREALIZED_PL", 0),
            daily_pnl=acct.get("DAILY_PL", 0),
            execution_result="OK" if result.get("success") else result.get("error", "UNKNOWN")
        )
        
        self.decisions.append(decision)
        
        # Write to file immediately
        self.decision_file.write(json.dumps(asdict(decision)) + "\n")
        self.decision_file.flush()
        
        # Update equity curve
        self._update_equity(acct.get("DAILY_PL", 0))
    
    def log_trade(self, action: str, side: str, price: float, size: int,
                  stop: float = 0, target: float = 0, pnl: float = 0,
                  hold_time: float = 0):
        """Log a trade execution."""
        trade = Trade(
            timestamp=time.time(),
            action=action,
            side=side,
            price=price,
            size=size,
            stop=stop,
            target=target,
            pnl=pnl,
            hold_time_sec=hold_time
        )
        
        self.trades.append(trade)
        
        # Write to file
        self.trade_file.write(json.dumps(asdict(trade)) + "\n")
        self.trade_file.flush()
    
    def _update_equity(self, current_pnl: float):
        """Update equity curve and drawdown."""
        self.current_equity = current_pnl
        
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity
        
        drawdown = self.peak_equity - self.current_equity
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
        
        self.equity_curve.append({
            "time": time.time(),
            "equity": self.current_equity,
            "drawdown": drawdown
        })
    
    def generate_report(self, mbo_stats: Dict = None) -> BacktestReport:
        """Generate final backtest report."""
        end_time = time.time()
        
        # Count command types
        enter_long = sum(1 for d in self.decisions if "ENTER_LONG" in d.command)
        enter_short = sum(1 for d in self.decisions if "ENTER_SHORT" in d.command)
        update_stop = sum(1 for d in self.decisions if "UPDATE_STOP" in d.command)
        update_target = sum(1 for d in self.decisions if "UPDATE_TARGET" in d.command)
        cancel = sum(1 for d in self.decisions if "CANCEL" in d.command)
        wait = sum(1 for d in self.decisions if "WAIT" in d.command)
        
        # Calculate trade stats
        winning = [t for t in self.trades if t.pnl > 0]
        losing = [t for t in self.trades if t.pnl < 0]
        
        gross_profit = sum(t.pnl for t in winning)
        gross_loss = abs(sum(t.pnl for t in losing))
        
        report = BacktestReport(
            date=self.date,
            symbol=self.symbol,
            start_time=datetime.fromtimestamp(self.start_time).isoformat(),
            end_time=datetime.fromtimestamp(end_time).isoformat(),
            duration_sec=round(end_time - self.start_time, 2),
            
            total_mbo_events=mbo_stats.get("events_processed", 0) if mbo_stats else 0,
            market_events=mbo_stats.get("events_processed", 0) if mbo_stats else 0,
            trades_in_data=mbo_stats.get("trades_count", 0) if mbo_stats else 0,
            
            total_decisions=len(self.decisions),
            enter_long_count=enter_long,
            enter_short_count=enter_short,
            update_stop_count=update_stop,
            update_target_count=update_target,
            cancel_count=cancel,
            wait_count=wait,
            
            total_trades=len(self.trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=len(winning) / len(self.trades) * 100 if self.trades else 0,
            
            gross_pnl=round(gross_profit - gross_loss, 2),
            net_pnl=round(self.current_equity, 2),
            max_drawdown=round(self.max_drawdown, 2),
            max_profit=round(self.peak_equity, 2),
            avg_win=round(gross_profit / len(winning), 2) if winning else 0,
            avg_loss=round(gross_loss / len(losing), 2) if losing else 0,
            profit_factor=round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0
        )
        
        # Save report
        self._save_report(report)
        
        return report
    
    def _save_report(self, report: BacktestReport):
        """Save all report files."""
        # Close file handles
        self.decision_file.close()
        self.trade_file.close()
        
        # Save summary JSON
        with open(self.run_dir / "summary.json", "w") as f:
            json.dump(asdict(report), f, indent=2)
        
        # Save equity curve
        with open(self.run_dir / "equity_curve.json", "w") as f:
            json.dump(self.equity_curve, f)
        
        # Generate markdown report
        self._generate_markdown_report(report)
        
        print(f"\n📈 Report saved to: {self.run_dir}")
    
    def _generate_markdown_report(self, report: BacktestReport):
        """Generate a human-readable markdown report."""
        md = f"""# FSDTrader Backtest Report

**Date**: {report.date}
**Symbol**: {report.symbol}
**Duration**: {report.duration_sec:.0f} seconds

---

## Performance Summary

| Metric | Value |
|--------|-------|
| Net P&L | **${report.net_pnl:,.2f}** |
| Max Drawdown | ${report.max_drawdown:,.2f} |
| Max Profit | ${report.max_profit:,.2f} |
| Win Rate | {report.win_rate:.1f}% |
| Profit Factor | {report.profit_factor:.2f} |

---

## Trade Statistics

| Metric | Value |
|--------|-------|
| Total Trades | {report.total_trades} |
| Winning Trades | {report.winning_trades} |
| Losing Trades | {report.losing_trades} |
| Avg Win | ${report.avg_win:,.2f} |
| Avg Loss | ${report.avg_loss:,.2f} |

---

## AI Decision Statistics

| Command | Count |
|---------|-------|
| ENTER_LONG | {report.enter_long_count} |
| ENTER_SHORT | {report.enter_short_count} |
| UPDATE_STOP | {report.update_stop_count} |
| UPDATE_TARGET | {report.update_target_count} |
| CANCEL_ALL | {report.cancel_count} |
| WAIT | {report.wait_count} |
| **Total Decisions** | **{report.total_decisions}** |

---

## Data Statistics

| Metric | Value |
|--------|-------|
| MBO Events Processed | {report.total_mbo_events:,} |
| Trades in Market Data | {report.trades_in_data:,} |

---

*Generated at {report.end_time}*
"""
        
        with open(self.run_dir / "report.md", "w") as f:
            f.write(md)
    
    def print_summary(self, report: BacktestReport):
        """Print a summary to console."""
        print("\n" + "=" * 60)
        print(f"  BACKTEST REPORT: {report.symbol} {report.date}")
        print("=" * 60)
        print(f"  Duration: {report.duration_sec:.0f}s")
        print(f"  Events: {report.total_mbo_events:,} MBO | {report.trades_in_data:,} trades")
        print("-" * 60)
        print(f"  AI Decisions: {report.total_decisions}")
        print(f"    ENTER_LONG:  {report.enter_long_count}")
        print(f"    ENTER_SHORT: {report.enter_short_count}")
        print(f"    WAIT:        {report.wait_count}")
        print("-" * 60)
        print(f"  Net P&L:      ${report.net_pnl:,.2f}")
        print(f"  Win Rate:     {report.win_rate:.1f}%")
        print(f"  Max Drawdown: ${report.max_drawdown:,.2f}")
        print("=" * 60)
        print(f"  Report: {self.run_dir}")
        print()
