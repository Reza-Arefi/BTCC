"""SimulatedBroker — TEST-mode unreliable exchange. Never calls PaperBroker or MEXC.

Available only when explicitly constructed for tests (ExecutionMode.TEST).
"""

from __future__ import annotations

from typing import Any, Callable

from btcc.execution.modes import ExecutionMode
from btcc.execution.sim.exchange import SimOrder, SimulatedExchange
from btcc.execution.sim.scenarios import (
    SimBrokerError,
    SimNetworkError,
    SimRejectedError,
    SimScenario,
    SimTimeoutError,
)
from btcc.execution.types import AccountState, FillReport, OrderRequest
from btcc.safety.no_trading import TradingForbiddenError


def _qty_from_request(request: OrderRequest) -> float:
    meta = request.meta or {}
    if "quantity" in meta:
        return float(meta["quantity"])
    if "requested_quantity" in meta:
        return float(meta["requested_quantity"])
    # Fall back to notional as quantity units for simple tests
    return float(request.notional_usd)


def _price_from_request(request: OrderRequest) -> float:
    meta = request.meta or {}
    if "price" in meta:
        return float(meta["price"])
    return float(request.alt_btc_mid or 1.0)


class SimulatedBroker:
    """Deterministic failure-injection broker with independent exchange state."""

    def __init__(
        self,
        *,
        exchange: SimulatedExchange | None = None,
        default_scenario: SimScenario = SimScenario.NORMAL,
        fee_rate: float = 0.001,
        scenario_for_client_id: dict[str, SimScenario] | None = None,
        scenario_queue: list[SimScenario] | None = None,
        clock: Callable[[], int] | None = None,
    ) -> None:
        self.exchange = exchange or SimulatedExchange()
        self.default_scenario = SimScenario(default_scenario)
        self.fee_rate = float(fee_rate)
        self._scenario_for_coid = dict(scenario_for_client_id or {})
        self._scenario_queue = list(scenario_queue or [])
        self._clock = clock or (lambda: 0)
        self._submit_attempts: list[str] = []

    @property
    def mode(self) -> ExecutionMode:
        return ExecutionMode.TEST

    @property
    def name(self) -> str:
        return "SimulatedBroker(TEST)"

    def set_scenario(self, scenario: SimScenario) -> None:
        self.default_scenario = SimScenario(scenario)

    def queue_scenario(self, scenario: SimScenario) -> None:
        self._scenario_queue.append(SimScenario(scenario))

    def set_scenario_for_client_order_id(self, client_order_id: str, scenario: SimScenario) -> None:
        self._scenario_for_coid[client_order_id] = SimScenario(scenario)

    def _next_scenario(self, client_order_id: str) -> SimScenario:
        if client_order_id in self._scenario_for_coid:
            return self._scenario_for_coid[client_order_id]
        if self._scenario_queue:
            return self._scenario_queue.pop(0)
        return self.default_scenario

    def submit_order(self, request: OrderRequest) -> FillReport:
        coid = request.client_order_id
        self._submit_attempts.append(coid)
        scenario = self._next_scenario(coid)
        qty = _qty_from_request(request)
        price = _price_from_request(request)

        # Idempotency: same client_order_id already on exchange → return existing (no new order)
        existing = self.exchange.get_by_client_order_id(coid)
        if existing is not None and scenario != SimScenario.DUPLICATE_REQUEST:
            return self._report_from_order(existing, request, note="IDEMPOTENT_REPLAY")

        if scenario == SimScenario.NETWORK_FAILURE:
            raise SimNetworkError("simulated network failure before acceptance unknown")

        if scenario == SimScenario.TIMEOUT_BEFORE_ACCEPTANCE:
            # Do NOT record on exchange
            raise SimTimeoutError(
                "timeout before acceptance — exchange outcome unknown",
                accepted=False,
                client_order_id=coid,
            )

        if scenario == SimScenario.REJECTED:
            oid = self.exchange.next_id("ord")
            order = SimOrder(
                exchange_order_id=oid,
                client_order_id=coid,
                symbol=request.symbol,
                side=request.side,
                order_type="MARKET",
                status="REJECTED",
                requested_quantity=qty,
                price=price,
            )
            self.exchange.record_order(order)
            raise SimRejectedError(f"simulated reject {oid}")

        if scenario == SimScenario.EXCHANGE_STATE_LOST:
            self.exchange.lost = True
            raise SimNetworkError("exchange state lost")

        # Accept on exchange first (for timeout-after and normal paths)
        oid = self.exchange.next_id("ord")
        order = SimOrder(
            exchange_order_id=oid,
            client_order_id=coid,
            symbol=request.symbol,
            side=request.side,
            order_type="MARKET",
            status="NEW",
            requested_quantity=qty,
            price=price,
        )
        self.exchange.record_order(order)

        if scenario == SimScenario.TIMEOUT_AFTER_ACCEPTANCE:
            raise SimTimeoutError(
                "timeout after acceptance — response lost; order may exist on exchange",
                accepted=True,
                exchange_order_id=oid,
                client_order_id=coid,
            )

        if scenario == SimScenario.DUPLICATE_REQUEST:
            # Force a second independent order id (anti-pattern) — used to prove OMS must prevent this
            oid2 = self.exchange.next_id("ord")
            order2 = SimOrder(
                exchange_order_id=oid2,
                client_order_id=f"{coid}#DUP",
                symbol=request.symbol,
                side=request.side,
                order_type="MARKET",
                status="NEW",
                requested_quantity=qty,
                price=price,
            )
            self.exchange.record_order(order2)
            # Still fill the first; OMS should never have called twice
            self._fill_normal(order, price)
            return self._report_from_order(order, request, note="DUP_SCENARIO_FIRST")

        if scenario == SimScenario.PARTIAL_FILL:
            fill_qty = float((request.meta or {}).get("partial_qty", qty * 0.3))
            fill_qty = min(fill_qty, qty)
            fee = fill_qty * price * self.fee_rate
            self.exchange.apply_fill(order, qty=fill_qty, price=price, fee=fee)
            return self._report_from_order(order, request, note="PARTIAL")

        if scenario == SimScenario.MULTI_FILL:
            chunks = (request.meta or {}).get("fill_chunks") or [0.3, 0.2, 0.5]
            chunks = [float(c) for c in chunks]
            # interpret as fractions if sum≈1 else absolute qty
            if abs(sum(chunks) - 1.0) < 1e-9:
                abs_chunks = [qty * c for c in chunks]
            else:
                abs_chunks = chunks
            for cq in abs_chunks:
                if cq <= 0:
                    continue
                fee = cq * price * self.fee_rate
                self.exchange.apply_fill(order, qty=cq, price=price, fee=fee)
            return self._report_from_order(order, request, note="MULTI")

        if scenario == SimScenario.DELAYED_FILL:
            # Accepted, not filled yet — resting
            return self._report_from_order(order, request, note="DELAYED_ACK_ONLY")

        # NORMAL
        self._fill_normal(order, price)
        return self._report_from_order(order, request, note="NORMAL")

    def _fill_normal(self, order: SimOrder, price: float) -> None:
        fee = order.requested_quantity * price * self.fee_rate
        self.exchange.apply_fill(order, qty=order.requested_quantity, price=price, fee=fee)

    def advance_delayed_fill(self, client_order_id: str, *, qty: float | None = None) -> SimOrder:
        order = self.exchange.get_by_client_order_id(client_order_id)
        if order is None:
            raise SimBrokerError(f"no order for {client_order_id}")
        rem = order.remaining_quantity
        fill_qty = rem if qty is None else min(float(qty), rem)
        if fill_qty <= 0:
            return order
        price = float(order.price or 1.0)
        self.exchange.apply_fill(order, qty=fill_qty, price=price, fee=fill_qty * price * self.fee_rate)
        return order

    def _report_from_order(self, order: SimOrder, request: OrderRequest, *, note: str) -> FillReport:
        fully = order.status == "FILLED"
        return FillReport(
            ok=order.status not in ("REJECTED",),
            legs=[],
            entry_ts=request.entry_ts,
            entry_mid=order.average_price or order.price,
            notional_usd=request.notional_usd,
            rejection_reason=None if order.status != "REJECTED" else "SIM_REJECTED",
            broker_name=self.name,
            raw={
                "scenario_note": note,
                "exchange_order_id": order.exchange_order_id,
                "client_order_id": order.client_order_id,
                "status": order.status,
                "requested_quantity": order.requested_quantity,
                "executed_quantity": order.executed_quantity,
                "remaining_quantity": order.remaining_quantity,
                "average_price": order.average_price,
                "fees": order.fees,
                "fills": [
                    {
                        "trade_id": f.trade_id,
                        "quantity": f.quantity,
                        "price": f.price,
                        "fee": f.fee,
                        "fee_asset": f.fee_asset,
                    }
                    for f in order.fills
                ],
                "acknowledged": True,
                "fully_filled": fully,
            },
        )

    def get_account_state(self) -> AccountState:
        btc = float(self.exchange.balances.get("BTC", 0.0))
        return AccountState(
            mode=self.mode.value,
            equity=btc,
            available=btc,
            reserved=0.0,
            currency="BTC",
            open_positions=sum(1 for v in self.exchange.positions.values() if abs(v) > 1e-12),
            open_orders=len(self.exchange.open_orders()),
            meta={"source": "simulated_exchange"},
        )

    def get_balances(self) -> dict[str, float]:
        return dict(self.exchange.balances)

    def get_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        return [self._order_dict(o) for o in self.exchange.open_orders(symbol)]

    def get_open_positions(self, symbol: str | None = None) -> list[dict[str, Any]]:
        out = []
        for sym, qty in self.exchange.positions.items():
            if abs(qty) <= 1e-15:
                continue
            if symbol and sym != symbol:
                continue
            out.append({"symbol": sym, "quantity": qty})
        return out

    def get_order(self, order_id: str) -> dict[str, Any]:
        o = self.exchange.get_by_order_id(order_id)
        if o is None:
            # also allow lookup by client id
            o = self.exchange.get_by_client_order_id(order_id)
        if o is None:
            raise SimBrokerError(f"order not found: {order_id}")
        return self._order_dict(o)

    def get_order_by_client_order_id(self, client_order_id: str) -> dict[str, Any] | None:
        o = self.exchange.get_by_client_order_id(client_order_id)
        return None if o is None else self._order_dict(o)

    def get_recent_fills(self, symbol: str | None = None, *, limit: int = 50) -> list[dict[str, Any]]:
        fills = list(reversed(self.exchange.fills))
        out = []
        for f in fills:
            if symbol and f.symbol != symbol:
                continue
            out.append(
                {
                    "trade_id": f.trade_id,
                    "order_id": f.order_id,
                    "client_order_id": f.client_order_id,
                    "quantity": f.quantity,
                    "price": f.price,
                    "fee": f.fee,
                    "fee_asset": f.fee_asset,
                    "timestamp": f.timestamp,
                }
            )
            if len(out) >= limit:
                break
        return out

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        # Stage 4: allow cancel on sim only for lifecycle tests — not production
        o = self.exchange.get_by_order_id(order_id) or self.exchange.get_by_client_order_id(order_id)
        if o is None:
            raise SimBrokerError(f"cannot cancel unknown {order_id}")
        if o.status == "FILLED":
            raise SimBrokerError("cannot cancel filled order")
        o.status = "CANCELLED"
        return self._order_dict(o)

    def place_protective_stop(
        self, position_id: str, *, stop_price: float, meta: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        # Not implementing real stops; Stage 4 only tracks intended protective qty in meta tests.
        raise TradingForbiddenError(
            "SimulatedBroker does not place protective stops in Stage 4 "
            "(representation-only; no exchange-native protection)"
        )

    def open_long_legs(self, **kwargs: Any) -> list[Any]:
        raise TradingForbiddenError(
            "SimulatedBroker does not implement open_long_legs — use submit_order in TEST mode"
        )

    @staticmethod
    def _order_dict(o: SimOrder) -> dict[str, Any]:
        return {
            "order_id": o.exchange_order_id,
            "exchange_order_id": o.exchange_order_id,
            "client_order_id": o.client_order_id,
            "symbol": o.symbol,
            "side": o.side,
            "type": o.order_type,
            "status": o.status,
            "original_quantity": o.requested_quantity,
            "requested_quantity": o.requested_quantity,
            "executed_quantity": o.executed_quantity,
            "remaining_quantity": o.remaining_quantity,
            "price": o.price,
            "average_price": o.average_price,
            "fees": o.fees,
            "fills": [
                {
                    "trade_id": f.trade_id,
                    "quantity": f.quantity,
                    "price": f.price,
                    "fee": f.fee,
                    "fee_asset": f.fee_asset,
                }
                for f in o.fills
            ],
        }

    @property
    def submit_attempt_count(self) -> int:
        return len(self._submit_attempts)
