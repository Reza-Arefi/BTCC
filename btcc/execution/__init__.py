"""Execution architecture: Strategy → Intent → Risk → OMS → Broker → Exchange.

Stage 1–2: PAPER broker adapts existing paper fills. REAL broker trading is fail-closed.
Stage 3: MEXC Spot authenticated READ-ONLY under btcc.execution.mexc (no order POST).
"""

from __future__ import annotations

from btcc.execution.factory import build_execution_stack, resolve_execution_mode
from btcc.execution.modes import ExecutionMode
from btcc.execution.oms import OrderManagementSystem
from btcc.execution.types import AccountState, FillReport, OrderRequest, TradeIntent

__all__ = [
    "AccountState",
    "ExecutionMode",
    "FillReport",
    "OrderManagementSystem",
    "OrderRequest",
    "TradeIntent",
    "build_execution_stack",
    "resolve_execution_mode",
]
