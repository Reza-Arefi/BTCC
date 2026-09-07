"""Typed contracts for trade intents and broker I/O (Stage 1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TradeIntent:
    """Strategy decision to open a LONG ALT/BTC opportunity (pre-broker)."""

    opportunity_id: str
    symbol: str
    side: str  # LONG_ALT_BTC
    decision_ts: Any
    position_btc: float
    requested_allocation_pct: float
    actual_allocation_pct: float
    remaining_exposure_pct: float
    exposure_limited: bool
    s_value: float
    regime: str | None = None
    selected_strategy_key: str | None = None
    selected_arm_label: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OrderRequest:
    """OMS → Broker request to open (or protect) a position."""

    client_order_id: str
    symbol: str
    side: str
    notional_usd: float
    alt_btc_mid: float
    btc_usdt: float
    entry_ts: Any
    intent_id: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SpotMarketOrderRequest:
    """REAL Spot MARKET order (ALT/BTC). Quantity is base asset; never use as final inventory."""

    client_order_id: str
    symbol: str
    side: str  # BUY / SELL
    quantity: float
    intent_id: str
    order_type: str = "MARKET"
    quote_order_qty: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class FillReport:
    """Broker fill result used by OMS / strategy engine."""

    ok: bool
    legs: list[Any] = field(default_factory=list)
    entry_ts: Any = None
    entry_mid: float | None = None
    notional_usd: float | None = None
    rejection_reason: str | None = None
    broker_name: str = ""
    raw: dict[str, Any] = field(default_factory=dict)
    # Stage 7 real-path fields (optional; paper leaves unset)
    exchange_order_id: str | None = None
    client_order_id: str | None = None
    requested_quantity: float | None = None
    executed_quantity: float | None = None
    average_price: float | None = None
    fee: float | None = None
    order_status: str | None = None


@dataclass(frozen=True)
class AccountState:
    """Broker-reported account snapshot (paper or real)."""

    mode: str
    equity: float
    available: float
    reserved: float
    currency: str = "BTC"
    open_positions: int = 0
    open_orders: int = 0
    meta: dict[str, Any] = field(default_factory=dict)
