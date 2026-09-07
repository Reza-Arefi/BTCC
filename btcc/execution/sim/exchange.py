"""Independent simulated exchange state (not PaperBroker, not MEXC)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SimFill:
    trade_id: str
    order_id: str
    client_order_id: str
    symbol: str
    quantity: float
    price: float
    fee: float
    fee_asset: str
    timestamp: int


@dataclass
class SimOrder:
    exchange_order_id: str
    client_order_id: str
    symbol: str
    side: str
    order_type: str
    status: str  # NEW / PARTIALLY_FILLED / FILLED / REJECTED / CANCELLED
    requested_quantity: float
    executed_quantity: float = 0.0
    price: float | None = None
    average_price: float | None = None
    fees: float = 0.0
    fee_asset: str = "BTC"
    fills: list[SimFill] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def remaining_quantity(self) -> float:
        return max(0.0, float(self.requested_quantity) - float(self.executed_quantity))


@dataclass
class SimulatedExchange:
    """Exchange-side truth for Stage-4 tests. Independent of local OMS journals."""

    orders_by_id: dict[str, SimOrder] = field(default_factory=dict)
    orders_by_client_id: dict[str, SimOrder] = field(default_factory=dict)
    fills: list[SimFill] = field(default_factory=list)
    # asset -> free balance (inventory)
    balances: dict[str, float] = field(default_factory=lambda: {"BTC": 1.0})
    # symbol -> base inventory from fills (simplified spot long)
    positions: dict[str, float] = field(default_factory=dict)
    _seq: int = 0
    lost: bool = False  # EXCHANGE_STATE_LOST

    def next_id(self, prefix: str = "sim") -> str:
        self._seq += 1
        return f"{prefix}-{self._seq:06d}"

    def get_by_client_order_id(self, client_order_id: str) -> SimOrder | None:
        if self.lost:
            return None
        return self.orders_by_client_id.get(client_order_id)

    def get_by_order_id(self, order_id: str) -> SimOrder | None:
        if self.lost:
            return None
        return self.orders_by_id.get(order_id)

    def open_orders(self, symbol: str | None = None) -> list[SimOrder]:
        if self.lost:
            return []
        out = []
        for o in self.orders_by_id.values():
            if o.status in ("FILLED", "CANCELLED", "REJECTED"):
                continue
            if symbol and o.symbol != symbol:
                continue
            out.append(o)
        return out

    def record_order(self, order: SimOrder) -> SimOrder:
        self.orders_by_id[order.exchange_order_id] = order
        self.orders_by_client_id[order.client_order_id] = order
        return order

    def apply_fill(self, order: SimOrder, *, qty: float, price: float, fee: float, fee_asset: str = "BTC") -> SimFill:
        if qty <= 0:
            raise ValueError("fill qty must be > 0")
        if order.executed_quantity + qty > order.requested_quantity + 1e-12:
            raise ValueError("fill would exceed requested quantity")
        fill = SimFill(
            trade_id=self.next_id("fill"),
            order_id=order.exchange_order_id,
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            quantity=float(qty),
            price=float(price),
            fee=float(fee),
            fee_asset=fee_asset,
            timestamp=self._seq + 1_700_000_000_000,
        )
        order.fills.append(fill)
        self.fills.append(fill)
        prev_exe = order.executed_quantity
        order.executed_quantity = prev_exe + qty
        # VWAP
        if order.average_price is None or prev_exe <= 0:
            order.average_price = price
        else:
            order.average_price = (order.average_price * prev_exe + price * qty) / order.executed_quantity
        order.fees += fee
        if abs(order.executed_quantity - order.requested_quantity) <= 1e-12:
            order.status = "FILLED"
            order.executed_quantity = order.requested_quantity
        else:
            order.status = "PARTIALLY_FILLED"

        # Inventory: BUY increases base position
        base = order.symbol.replace("BTC", "").replace("USDT", "") or order.symbol
        if order.side.upper() in ("BUY", "LONG", "LONG_ALT_BTC"):
            self.positions[order.symbol] = self.positions.get(order.symbol, 0.0) + qty
        else:
            self.positions[order.symbol] = self.positions.get(order.symbol, 0.0) - qty
        return fill
