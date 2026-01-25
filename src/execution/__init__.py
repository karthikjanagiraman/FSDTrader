#!/usr/bin/env python3
"""
FSDTrader Execution Module

Order execution and position management with provider abstraction.

Components:
- ExecutionProvider: Abstract interface for order execution
- SimulatedExecutor: Virtual execution for backtest/sim modes
- IBKRExecutor: Real execution via IBKR TWS API

Usage:
    from execution import get_executor, RiskLimits

    # For backtest/sim
    executor = get_executor("simulated", risk_limits=RiskLimits())

    # For live/paper
    executor = get_executor("ibkr", ib=ib_connection, symbol="TSLA", risk_limits=RiskLimits())
"""
from .types import (
    PositionSide,
    Position,
    BracketOrder,
    OrderResult,
    TradeRecord,
    RiskLimits,
)
from .base import ExecutionProvider
from .simulated import SimulatedExecutor
from .ibkr import IBKRExecutor


def get_executor(
    provider: str,
    **kwargs
) -> ExecutionProvider:
    """
    Factory function to create an execution provider.

    Args:
        provider: "simulated" or "ibkr"
        **kwargs: Provider-specific arguments

    Returns:
        ExecutionProvider instance
    """
    providers = {
        "simulated": SimulatedExecutor,
        "ibkr": IBKRExecutor,
    }

    provider_class = providers.get(provider.lower())
    if provider_class is None:
        raise ValueError(f"Unknown provider '{provider}'. Use 'simulated' or 'ibkr'")

    return provider_class(**kwargs)


__all__ = [
    # Types
    "PositionSide",
    "Position",
    "BracketOrder",
    "OrderResult",
    "TradeRecord",
    "RiskLimits",
    # Providers
    "ExecutionProvider",
    "SimulatedExecutor",
    "IBKRExecutor",
    # Factory
    "get_executor",
]
