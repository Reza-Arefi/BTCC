"""Dry-run Binance Spot order simulator — full lifecycle without real writes.

Used when live.enabled=false / dry_run=true so Layer 2 can exercise:
  BUY → fill → OCO accept/reject → exit → cancel other leg
"""

from __future__ import annotations

import itertools
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SimOrder:
    order_id: str
    client_order_id: str | None
    symbol: str
    side: str
    order_type: str
    status: str
    quantity: float
    executed_qty: float = 0.0
    cumulative_quote_qty: float = 0.0
    avg_price: float = 0.0
    fills: list[dict[str, Any]] = field(default_factory=list)
    order_list_id: str | None = None
    created_at: float = field(default_factory=time.time)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimOco:
    order_list_id: str
    list_client_order_id: str | None
    symbol: str
    status: str  # EXECUTING / ALL_DONE / REJECT
    list_order_status: str  # EXECUTING / ALL_DONE / REJECT
    quantity: float
    above_order_id: str
    below_order_id: str
    entry_ref_price: float
    activation_price: float
    stop_price: float
    trailing_delta: int
    created_at: float = field(default_factory=time.time)
    exit_fill: dict[str, Any] | None = None


class DryRunBroker:
    """In-memory Binance-like broker for dry-run lifecycle tests."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._orders: dict[str, SimOrder] = {}
        self._by_client: dict[str, str] = {}
        self._ocos: dict[str, SimOco] = {}
        self._seq = itertools.count(1)
        self.buy_behavior: str = "FILL"  # FILL | PARTIAL | REJECT | CANCEL | EXPIRE
        self.oco_behavior: str = "ACCEPT"  # ACCEPT | REJECT
        self.allow_emergency_protect: bool = True
        self.slippage_bps: float = 1.0  # slight adverse slip on market buys
        self.default_price: dict[str, float] = {}
        self.events: list[dict[str, Any]] = []

    def _next_id(self, prefix: str) -> str:
        return f"{prefix}{next(self._seq)}"

    def set_price(self, symbol: str, price: float) -> None:
        self.default_price[symbol.upper()] = float(price)

    def lookup_by_client_order_id(self, client_order_id: str) -> SimOrder | None:
        with self._lock:
            oid = self._by_client.get(client_order_id)
            return self._orders.get(oid) if oid else None

    def place_market_buy(
        self,
        *,
        symbol: str,
        quantity: float,
        client_order_id: str | None,
        ref_price: float,
    ) -> SimOrder:
        with self._lock:
            if client_order_id and client_order_id in self._by_client:
                # Idempotent: return existing order
                return self._orders[self._by_client[client_order_id]]

            oid = self._next_id("drybuy")
            slip = 1.0 + (self.slippage_bps / 10_000.0)
            px = float(ref_price) * slip
            behavior = self.buy_behavior.upper()

            if behavior == "REJECT":
                order = SimOrder(
                    order_id=oid,
                    client_order_id=client_order_id,
                    symbol=symbol.upper(),
                    side="BUY",
                    order_type="MARKET",
                    status="REJECTED",
                    quantity=float(quantity),
                )
            elif behavior == "CANCEL":
                order = SimOrder(
                    order_id=oid,
                    client_order_id=client_order_id,
                    symbol=symbol.upper(),
                    side="BUY",
                    order_type="MARKET",
                    status="CANCELED",
                    quantity=float(quantity),
                )
            elif behavior == "EXPIRE":
                order = SimOrder(
                    order_id=oid,
                    client_order_id=client_order_id,
                    symbol=symbol.upper(),
                    side="BUY",
                    order_type="MARKET",
                    status="EXPIRED",
                    quantity=float(quantity),
                )
            elif behavior == "PARTIAL":
                filled = float(quantity) * 0.5
                order = SimOrder(
                    order_id=oid,
                    client_order_id=client_order_id,
                    symbol=symbol.upper(),
                    side="BUY",
                    order_type="MARKET",
                    status="PARTIALLY_FILLED",
                    quantity=float(quantity),
                    executed_qty=filled,
                    cumulative_quote_qty=filled * px,
                    avg_price=px,
                    fills=[
                        {
                            "price": str(px),
                            "qty": str(filled),
                            "commission": str(filled * px * 0.001),
                            "commissionAsset": "BTC",
                            "id": self._next_id("t"),
                        }
                    ],
                )
            else:  # FILL — optionally two legs for weighted-average tests
                half = float(quantity) * 0.4
                rest = float(quantity) - half
                px2 = px * 1.0002
                fills = [
                    {
                        "price": str(px),
                        "qty": str(half),
                        "commission": str(half * px * 0.001),
                        "commissionAsset": "BTC",
                        "id": self._next_id("t"),
                    },
                    {
                        "price": str(px2),
                        "qty": str(rest),
                        "commission": str(rest * px2 * 0.001),
                        "commissionAsset": "BTC",
                        "id": self._next_id("t"),
                    },
                ]
                quote = half * px + rest * px2
                order = SimOrder(
                    order_id=oid,
                    client_order_id=client_order_id,
                    symbol=symbol.upper(),
                    side="BUY",
                    order_type="MARKET",
                    status="FILLED",
                    quantity=float(quantity),
                    executed_qty=float(quantity),
                    cumulative_quote_qty=quote,
                    avg_price=quote / float(quantity),
                    fills=fills,
                )

            self._orders[oid] = order
            if client_order_id:
                self._by_client[client_order_id] = oid
            self.events.append({"type": "executionReport", "order": order.order_id, "status": order.status})
            return order

    def place_oco(
        self,
        *,
        symbol: str,
        quantity: float,
        list_client_order_id: str | None,
        above_stop_price: float,
        above_trailing_delta: int,
        below_stop_price: float,
        entry_ref_price: float,
    ) -> SimOco | None:
        with self._lock:
            if list_client_order_id and list_client_order_id in self._by_client:
                key = self._by_client[list_client_order_id]
                if key in self._ocos:
                    return self._ocos[key]
            if self.oco_behavior.upper() == "REJECT":
                self.events.append({"type": "ocoReject", "symbol": symbol})
                return None
            lid = self._next_id("dryoco")
            above_id = self._next_id("dryabv")
            below_id = self._next_id("dryblw")
            oco = SimOco(
                order_list_id=lid,
                list_client_order_id=list_client_order_id,
                symbol=symbol.upper(),
                status="EXECUTING",
                list_order_status="EXECUTING",
                quantity=float(quantity),
                above_order_id=above_id,
                below_order_id=below_id,
                entry_ref_price=float(entry_ref_price),
                activation_price=float(above_stop_price),
                stop_price=float(below_stop_price),
                trailing_delta=int(above_trailing_delta),
            )
            self._ocos[lid] = oco
            # Track legs as open orders
            for oid, side_type in ((above_id, "TAKE_PROFIT"), (below_id, "STOP_LOSS")):
                self._orders[oid] = SimOrder(
                    order_id=oid,
                    client_order_id=None,
                    symbol=symbol.upper(),
                    side="SELL",
                    order_type=side_type,
                    status="NEW",
                    quantity=float(quantity),
                    order_list_id=lid,
                )
            if list_client_order_id:
                self._by_client[list_client_order_id] = lid
            self.events.append({"type": "listStatus", "orderListId": lid, "listStatusType": "EXECUTING"})
            return oco

    def get_order(self, order_id: str | None = None, client_order_id: str | None = None) -> SimOrder | None:
        with self._lock:
            if client_order_id and client_order_id in self._by_client:
                key = self._by_client[client_order_id]
                return self._orders.get(key) or None
            if order_id:
                return self._orders.get(order_id)
            return None

    def get_oco(self, order_list_id: str) -> SimOco | None:
        with self._lock:
            return self._ocos.get(str(order_list_id))

    def open_order_lists(self, symbol: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            out = []
            for oco in self._ocos.values():
                if oco.list_order_status != "EXECUTING":
                    continue
                if symbol and oco.symbol != symbol.upper():
                    continue
                out.append(
                    {
                        "orderListId": oco.order_list_id,
                        "contingencyType": "OCO",
                        "listStatusType": oco.status,
                        "listOrderStatus": oco.list_order_status,
                        "symbol": oco.symbol,
                        "orders": [
                            {"orderId": oco.above_order_id},
                            {"orderId": oco.below_order_id},
                        ],
                    }
                )
            return out

    def trigger_exit(self, order_list_id: str, *, exit_price: float, leg: str = "TRAIL") -> dict[str, Any]:
        """Simulate one OCO leg filling and the other cancelling."""
        with self._lock:
            oco = self._ocos.get(str(order_list_id))
            if not oco:
                raise KeyError(order_list_id)
            filled_id = oco.above_order_id if leg.upper() == "TRAIL" else oco.below_order_id
            other_id = oco.below_order_id if leg.upper() == "TRAIL" else oco.above_order_id
            qty = oco.quantity
            fill = {
                "price": str(exit_price),
                "qty": str(qty),
                "commission": str(qty * exit_price * 0.001),
                "commissionAsset": "BTC",
            }
            filled = self._orders[filled_id]
            filled.status = "FILLED"
            filled.executed_qty = qty
            filled.avg_price = float(exit_price)
            filled.cumulative_quote_qty = qty * float(exit_price)
            filled.fills = [fill]
            other = self._orders[other_id]
            other.status = "CANCELED"
            oco.list_order_status = "ALL_DONE"
            oco.status = "ALL_DONE"
            oco.exit_fill = {
                "order_id": filled_id,
                "price": float(exit_price),
                "qty": qty,
                "leg": leg.upper(),
                "other_cancelled": other_id,
            }
            self.events.append(
                {
                    "type": "executionReport",
                    "orderListId": order_list_id,
                    "status": "FILLED",
                    "leg": leg,
                }
            )
            return oco.exit_fill

    def mark_trailing_active(self, order_list_id: str) -> None:
        """Simulate activation of Binance-native trailing (price crossed activation)."""
        with self._lock:
            oco = self._ocos.get(str(order_list_id))
            if not oco:
                raise KeyError(order_list_id)
            oco.status = "TRAILING_ACTIVE"
            self.events.append(
                {
                    "type": "listStatus",
                    "orderListId": order_list_id,
                    "listStatusType": "TRAILING_ACTIVE",
                    "event_id": f"trail:{order_list_id}",
                }
            )

    def open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            out = []
            for o in self._orders.values():
                if o.status not in {"NEW", "PARTIALLY_FILLED"}:
                    continue
                if symbol and o.symbol != symbol.upper():
                    continue
                out.append(
                    {
                        "orderId": o.order_id,
                        "clientOrderId": o.client_order_id,
                        "symbol": o.symbol,
                        "side": o.side,
                        "type": o.order_type,
                        "status": o.status,
                        "orderListId": o.order_list_id,
                    }
                )
            return out

    def all_ocos(self) -> list[SimOco]:
        with self._lock:
            return list(self._ocos.values())
