#!/usr/bin/env python3
"""
FSDTrader: L2 Momentum Agent
Main Loop - Connects Market Data -> Brain -> Execution

Architecture:
  Market Data -> Brain (LLM with tool calling) -> Executor -> Orders

Modes:
  --sim      : Simulation with random mock data (no dependencies)
  --backtest : Replay real TSLA L3 data from BacktestData folder
  (default)  : Live paper trading via IBKR TWS
  --live     : Live real money trading via IBKR
"""
import os
import sys
import time
import asyncio
import logging
import argparse
import warnings
from pathlib import Path
from colorama import Fore, Style, init

# Suppress deprecation warnings from databento
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load environment variables from .env file
def load_dotenv():
    """Load environment variables from .env file if it exists."""
    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())

load_dotenv()

# Local imports
from market_data import IBKRConnector
from brain import TradingBrain
from execution import SimulatedExecutor, IBKRExecutor, RiskLimits
from reporting import BacktestReporter

# Initialize Colorama
init(autoreset=True)

# Configuration
VERSION = "0.4.0-alpha"
AGENT_NAME = "FSD-TRADER"
LOOP_INTERVAL = 5.0  # Decision interval in seconds (0.2 Hz)


def setup_logging(log_dir: str = "data/logs"):
    """Configure logging to file and console."""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)-12s | %(levelname)-5s | %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(f"{log_dir}/agent_{time.strftime('%Y%m%d')}.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(AGENT_NAME)


class FSDTrader:
    """
    The main trading agent that orchestrates:
    1. Market Data (from IBKR or Replay)
    2. Brain (LLM with native tool calling)
    3. Execution (Order Management)
    """

    def __init__(self, symbol: str, mode: str = "paper",
                 backtest_date: str = None, backtest_speed: float = 10.0,
                 backtest_start_time: str = "09:30:00", backtest_end_time: str = "16:00:00",
                 provider: str = "grok", model: str = None):
        self.symbol = symbol
        self.mode = mode  # "live", "paper", "sim", "backtest"
        self.backtest_date = backtest_date
        self.backtest_speed = backtest_speed
        self.backtest_start_time = backtest_start_time
        self.backtest_end_time = backtest_end_time
        self.logger = setup_logging()

        # Risk limits
        self.risk_limits = RiskLimits(
            max_position_size=100,
            max_daily_loss=-500.0,
            max_daily_trades=10,
            max_spread=0.05
        )

        # Get API key from environment based on provider
        api_key = self._get_api_key(provider)

        # Components (initialized in start())
        self.connector = None
        self.brain = TradingBrain(
            api_key=api_key,
            provider=provider,
            model=model,
        )
        self.executor = None

        # State
        self.running = False
        self.loop_count = 0
        self.decisions_made = 0

    def _get_api_key(self, provider: str) -> str:
        """Get API key from environment for the specified provider."""
        env_vars = {
            "grok": ["GROK_API_KEY", "XAI_API_KEY"],
            "openai": ["OPENAI_API_KEY"],
            "anthropic": ["ANTHROPIC_API_KEY"],
        }

        keys_to_try = env_vars.get(provider, [f"{provider.upper()}_API_KEY"])
        for key in keys_to_try:
            value = os.environ.get(key)
            if value:
                return value

        raise ValueError(
            f"Missing API key for {provider}. "
            f"Set one of: {', '.join(keys_to_try)}"
        )
        
    async def start(self):
        """Initialize connections and start the loop."""
        self._print_banner()
        
        if self.mode == "backtest":
            await self._start_backtest()
        elif self.mode == "sim":
            await self._start_simulation()
        else:
            await self._start_live()
    
    async def _start_backtest(self):
        """Start backtest mode with real MBO data."""
        from data_replay import MBOReplayConnector

        self.logger.info(f"{Fore.CYAN}Starting BACKTEST mode")

        # Initialize replay connector (path relative to project root)
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(project_root, "BacktestData", "TSLA-L3DATA")
        self.connector = MBOReplayConnector(
            data_dir=data_dir,
            date=self.backtest_date,
            symbol=self.symbol
        )
        self.connector.subscribe_market_data(self.symbol)

        # Initialize simulated executor
        self.executor = SimulatedExecutor(
            risk_limits=self.risk_limits,
            symbol=self.symbol,
        )

        # Initialize reporter
        self.reporter = BacktestReporter(
            symbol=self.symbol,
            date=self.connector.date
        )

        # Show available dates
        dates = self.connector.available_dates
        self.logger.info(f"Available dates: {dates[:5]}... ({len(dates)} total)")
        self.logger.info(f"Using date: {self.connector.date}")

        # Start replay
        self.running = True
        self.logger.info(f"Time range: {self.backtest_start_time} - {self.backtest_end_time}")
        await self.connector.start_replay(
            speed=self.backtest_speed,
            on_state_update=self._on_backtest_state,
            start_time=self.backtest_start_time,
            end_time=self.backtest_end_time
        )

        # Generate report
        self._print_backtest_results()
    
    async def _on_backtest_state(self, state):
        """Called during backtest on each state update."""
        if not self.running:
            self.connector.stop()
            return

        self.loop_count += 1

        try:
            # Get current simulation time and price
            sim_time = self.connector.current_simulation_time
            last_price = state.get("MARKET_STATE", {}).get("LAST", 0)

            # Skip if no price data yet
            if last_price == 0:
                return

            # Update executor with current price (checks for stop/target fills)
            self.executor.update(last_price, sim_time)

            # Get account state from executor
            account_state = self.executor.get_account_state()
            daily_pnl = account_state["DAILY_PL"]

            # Safety check
            if daily_pnl <= self.risk_limits.max_daily_loss:
                self.logger.warning(f"{Fore.RED}MAX DAILY LOSS HIT: ${daily_pnl:.2f}")
                self.running = False
                return

            # Inject account state and active orders
            state["ACCOUNT_STATE"] = account_state
            state["ACTIVE_ORDERS"] = self.executor.get_active_orders()

            # Ask the Brain
            command = self.brain.think(state)
            self.decisions_made += 1

            # Execute Command with market context for trade records
            spread = state.get("MARKET_STATE", {}).get("SPREAD", 0)
            market_context = state.get("MARKET_STATE", {})
            result = self.executor.execute(command, current_spread=spread, context=market_context)

            # Log to reporter
            if hasattr(self, 'reporter'):
                self.reporter.log_decision(state, command, result)

            # Log Decision to console
            self._log_decision(state, command, result)

        except Exception as e:
            self.logger.error(f"Backtest Error: {e}")
    
    async def _start_simulation(self):
        """Start simulation with random mock data."""
        self.logger.info(f"{Fore.YELLOW}Running in SIMULATION mode (mock data)")

        # Initialize simulated executor
        self.executor = SimulatedExecutor(
            risk_limits=self.risk_limits,
            symbol=self.symbol,
        )

        self.running = True
        await self._run_loop_mock()
    
    async def _start_live(self):
        """Start live/paper trading via IBKR."""
        port = 7497 if self.mode == "paper" else 7496
        self.connector = IBKRConnector(port=port)

        await self.connector.connect()
        self.connector.subscribe_market_data(self.symbol)

        # Initialize IBKR executor
        self.executor = IBKRExecutor(
            ib=self.connector.ib,
            symbol=self.symbol,
            risk_limits=self.risk_limits,
        )

        self.running = True
        await self._run_loop_live()
    
    async def _run_loop_live(self):
        """Live trading loop."""
        self.logger.info(f"{Fore.GREEN}Live loop STARTED")

        while self.running:
            loop_start = time.time()
            self.loop_count += 1

            try:
                state = self.connector.get_full_state(self.symbol)
                if state is None:
                    await asyncio.sleep(LOOP_INTERVAL)
                    continue

                # Get current price and update executor
                last_price = state.get("MARKET_STATE", {}).get("LAST", 0)
                self.executor.update(last_price, time.time())

                # Get account state from executor
                account_state = self.executor.get_account_state()
                daily_pnl = account_state["DAILY_PL"]

                if daily_pnl <= self.risk_limits.max_daily_loss:
                    self.logger.warning(f"{Fore.RED}MAX DAILY LOSS HIT")
                    break

                # Inject account state and active orders
                state["ACCOUNT_STATE"] = account_state
                state["ACTIVE_ORDERS"] = self.executor.get_active_orders()

                # Brain makes decision
                command = self.brain.think(state)

                # Execute with market context
                spread = state.get("MARKET_STATE", {}).get("SPREAD", 0)
                market_context = state.get("MARKET_STATE", {})
                result = self.executor.execute(command, current_spread=spread, context=market_context)

                self._log_decision(state, command, result)

            except Exception as e:
                self.logger.error(f"Loop Error: {e}")

            elapsed = time.time() - loop_start
            await asyncio.sleep(max(0, LOOP_INTERVAL - elapsed))
    
    async def _run_loop_mock(self):
        """Simulation loop with mock data."""
        self.logger.info(f"{Fore.GREEN}Simulation loop STARTED")

        import random
        base_price = 245.50
        sim_time = time.time()

        for i in range(100):  # 100 iterations
            if not self.running:
                break

            self.loop_count += 1
            sim_time += 5.0  # 5 second intervals

            current_price = base_price + random.uniform(-0.5, 0.5)

            # Update executor with current price (checks for fills)
            self.executor.update(current_price, sim_time)

            # Get account state from executor
            account_state = self.executor.get_account_state()

            state = {
                "MARKET_STATE": {
                    "TICKER": self.symbol,
                    "LAST": current_price,
                    "VWAP": base_price - 0.40,
                    "TIME_SESSION": "OPEN_DRIVE",
                    "L2_IMBALANCE": random.uniform(0.8, 2.5),
                    "SPREAD": random.uniform(0.01, 0.03),
                    "DOM_WALLS": [],
                    "BID_STACK": [[base_price - 0.01, 150]],
                    "ASK_STACK": [[base_price + 0.01, 120]],
                    "TAPE_VELOCITY": random.choice(["LOW", "MEDIUM", "HIGH"]),
                    "TAPE_VELOCITY_TPS": random.uniform(5, 50),
                    "TAPE_SENTIMENT": random.choice(["NEUTRAL", "AGGRESSIVE_BUYING"]),
                    "TAPE_DELTA_1S": random.randint(-500, 500),
                    "TAPE_DELTA_5S": random.randint(-2000, 2000),
                    "LARGE_PRINTS_1M": [],
                    "FOOTPRINT_CURR_BAR": {"delta": random.randint(-1000, 1000)},
                    "CVD_SESSION": 100000,
                    "CVD_TREND": random.choice(["RISING", "FALLING", "FLAT"]),
                    "CVD_SLOPE_5M": random.uniform(-1, 1),
                    "VP_POC": base_price - 0.70,
                    "VP_VAH": base_price + 1.50,
                    "VP_VAL": base_price - 2.00,
                    "VP_DEVELOPING_POC": base_price - 0.20,
                    "PRICE_VS_POC": "ABOVE",
                    "HOD": base_price + 0.30,
                    "LOD": base_price - 4.30,
                    "HOD_LOD_LOC": random.choice(["TESTING_HOD", "NEAR_HOD", "MID_RANGE"]),
                    "DISTANCE_TO_HOD_PCT": random.uniform(0, 0.5),
                    "RVOL_DAY": random.uniform(1.0, 3.0),
                    "ABSORPTION_DETECTED": False,
                    "ABSORPTION_SIDE": None,
                    "ABSORPTION_PRICE": None
                },
                "ACCOUNT_STATE": account_state,
                "ACTIVE_ORDERS": self.executor.get_active_orders()
            }

            command = self.brain.think(state)
            market_context = state.get("MARKET_STATE", {})
            result = self.executor.execute(command, context=market_context)
            self._log_decision(state, command, result)

            await asyncio.sleep(0.5)  # Slow down for visibility
    
    def _print_banner(self):
        """Print startup banner."""
        mode_str = {
            "live": f"{Fore.RED}LIVE TRADING",
            "paper": f"{Fore.YELLOW}PAPER TRADING",
            "sim": f"{Fore.CYAN}SIMULATION",
            "backtest": f"{Fore.MAGENTA}BACKTEST"
        }.get(self.mode, self.mode)
        
        self.logger.info("=" * 50)
        self.logger.info(f"{Fore.CYAN}  {AGENT_NAME} v{VERSION}")
        self.logger.info(f"  Symbol: {self.symbol}")
        self.logger.info(f"  Mode: {mode_str}{Style.RESET_ALL}")
        self.logger.info("=" * 50)
    
    def _print_backtest_results(self):
        """Print backtest summary and generate report."""
        stats = self.connector.get_stats()
        exec_state = self.executor.get_state()
        
        self.logger.info("=" * 50)
        self.logger.info(f"{Fore.CYAN}BACKTEST RESULTS")
        self.logger.info(f"  Date: {stats['date']}")
        self.logger.info(f"  Events Processed: {stats['events_processed']:,}")
        self.logger.info(f"  Trades in Data: {stats['trades_count']:,}")
        self.logger.info(f"  AI Decisions: {self.decisions_made}")
        self.logger.info(f"  Trades Executed: {exec_state['daily_stats']['trades']}")
        self.logger.info(f"  Final P&L: ${exec_state['daily_stats']['pnl']:.2f}")
        self.logger.info("=" * 50)
        
        # Generate detailed report
        if hasattr(self, 'reporter'):
            report = self.reporter.generate_report()
            self.logger.info(f"\n{Fore.GREEN}Report generated: {report}")
            self.reporter.print_summary(report)
        
    def _log_decision(self, state, command, result):
        """Pretty print the decision."""
        mkt = state.get("MARKET_STATE", {})
        last = mkt.get("LAST", 0)
        imbalance = mkt.get("L2_IMBALANCE", 0)
        cvd_trend = mkt.get("CVD_TREND", "?")
        hod_loc = mkt.get("HOD_LOD_LOC", "?")
        
        if "ENTER" in command:
            color = Fore.GREEN
        elif "UPDATE" in command:
            color = Fore.BLUE
        elif "CANCEL" in command:
            color = Fore.RED
        else:
            color = Fore.YELLOW
        
        # Log all non-WAIT, or every 50th WAIT
        if "WAIT" not in command or self.loop_count % 50 == 0:
            status = "✓" if result.get("success") else f"✗ {result.get('error', '')}"
            self.logger.info(
                f"${last:7.2f} | IMB:{imbalance:4.1f} | CVD:{cvd_trend:7} | "
                f"{color}{command[:35]:<35}{Style.RESET_ALL} | {status}"
            )
    
    def stop(self):
        """Graceful shutdown."""
        self.running = False
        self.logger.info(f"{Fore.RED}Shutting down...")
        
        if self.executor:
            state = self.executor.get_state()
            self.logger.info(f"Final P&L: ${state['daily_stats']['pnl']:.2f}")


async def main():
    parser = argparse.ArgumentParser(description="FSDTrader L2 Momentum Agent")
    parser.add_argument("--symbol", type=str, default="TSLA", help="Symbol to trade")
    parser.add_argument("--live", action="store_true", help="Live trading (DANGER!)")
    parser.add_argument("--sim", action="store_true", help="Simulation with mock data")
    parser.add_argument("--backtest", action="store_true", help="Backtest with real L3 data")
    parser.add_argument("--date", type=str, default=None, help="Backtest date (YYYYMMDD)")
    parser.add_argument("--speed", type=float, default=100.0, help="Backtest speed multiplier")
    parser.add_argument("--start-time", type=str, default="09:30:00", help="Backtest start time (HH:MM:SS)")
    parser.add_argument("--end-time", type=str, default="16:00:00", help="Backtest end time (HH:MM:SS)")
    parser.add_argument("--provider", type=str, default="grok",
                        help="LLM provider (grok, openai, anthropic)")
    parser.add_argument("--model", type=str, default=None,
                        help="LLM model override")
    args = parser.parse_args()

    # Determine mode
    if args.live:
        mode = "live"
    elif args.sim:
        mode = "sim"
    elif args.backtest:
        mode = "backtest"
    else:
        mode = "paper"

    trader = FSDTrader(
        symbol=args.symbol,
        mode=mode,
        backtest_date=args.date,
        backtest_speed=args.speed,
        backtest_start_time=args.start_time,
        backtest_end_time=args.end_time,
        provider=args.provider,
        model=args.model,
    )
    
    try:
        await trader.start()
    except KeyboardInterrupt:
        trader.stop()
    except Exception as e:
        logging.error(f"Fatal: {e}", exc_info=True)
        trader.stop()


if __name__ == "__main__":
    asyncio.run(main())
