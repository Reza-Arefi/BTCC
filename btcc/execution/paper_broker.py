"""PAPER broker — adapts existing paper accounting/exits. No exchange trading calls."""

from __future__ import annotations

from typing import Any

from btcc.execution.modes import ExecutionMode
from btcc.execution.types import AccountState, FillReport, OrderRequest
from btcc.sim.accounting import CostModel
from btcc.sim.exits import StrategySpec, open_opportunity_legs
from btcc.sim.paper_equity import PaperEquityTracker


class PaperBroker:
    """Simulated broker backed by CostModel + open_opportunity_legs (unchanged economics)."""

    def __init__(
        self,
        costs: CostModel,
        *,
        equity: PaperEquityTracker | None = None,
        open_opps_provider: Any | None = None,
    ) -> None:
        self._costs = costs
        self._equity = equity
        self._open_opps_provider = open_opps_provider
        self._submitted: list[OrderRequest] = []
        self._fills: list[FillReport] = []

    @property
    def mode(self) -> ExecutionMode:
        return ExecutionMode.PAPER

    @property
    def name(self) -> str:
        return "PaperBroker"

    def get_account_state(self) -> AccountState:
        eq = self._equity
        opens = self._open_opps()
        if eq is None:
            return AccountState(
                mode=self.mode.value,
                equity=0.0,
                available=0.0,
                reserved=0.0,
                open_positions=len(opens),
            )
        reserved = eq.reserved_btc(opens)
        return AccountState(
            mode=self.mode.value,
            equity=float(eq.equity_btc),
            available=float(eq.available_btc(opens)),
            reserved=float(reserved),
            currency="BTC",
            open_positions=sum(1 for o in opens if o.get("status") in ("OPEN", "PENDING_ENTRY")),
            open_orders=0,
            meta={"allocation_pct": eq.allocation_pct, "starting_equity_btc": eq.starting_equity_btc},
        )

    def get_balances(self) -> dict[str, float]:
        st = self.get_account_state()
        return {"BTC_equity": st.equity, "BTC_available": st.available, "BTC_reserved": st.reserved}

    def get_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        # Paper fills are immediate; no resting exchange orders.
        return []

    def get_open_positions(self, symbol: str | None = None) -> list[dict[str, Any]]:
        out = []
        for opp in self._open_opps():
            if opp.get("status") not in ("OPEN", "PENDING_ENTRY"):
                continue
            if symbol and opp.get("symbol") != symbol:
                continue
            out.append({
                "opportunity_id": opp.get("opportunity_id"),
                "symbol": opp.get("symbol"),
                "status": opp.get("status"),
                "position_btc": opp.get("position_btc"),
                "mode": self.mode.value,
            })
        return out

    def submit_order(self, request: OrderRequest) -> FillReport:
        """Record a paper order intent. Leg creation uses open_long_legs (identical to prior engine)."""
        self._submitted.append(request)
        specs = list((request.meta or {}).get("specs") or [])
        if not specs:
            report = FillReport(
                ok=False,
                rejection_reason="PAPER_MISSING_SPECS",
                broker_name=self.name,
            )
            self._fills.append(report)
            return report
        legs = self.open_long_legs(
            alt_btc_entry_mid=float(request.alt_btc_mid),
            btc_usdt=float(request.btc_usdt),
            notional_usd=float(request.notional_usd),
            specs=specs,
            entry_ts=request.entry_ts,
        )
        report = FillReport(
            ok=True,
            legs=legs,
            entry_ts=request.entry_ts,
            entry_mid=float(request.alt_btc_mid),
            notional_usd=float(request.notional_usd),
            broker_name=self.name,
            raw={"client_order_id": request.client_order_id, "mode": self.mode.value},
        )
        self._fills.append(report)
        return report

    def get_order(self, order_id: str) -> dict[str, Any]:
        for req in self._submitted:
            if req.client_order_id == order_id:
                return {"order_id": order_id, "status": "FILLED", "mode": self.mode.value}
        return {"order_id": order_id, "status": "NOT_FOUND", "mode": self.mode.value}

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        # Paper has no resting orders; cancellation is a no-op acknowledgment.
        return {"order_id": order_id, "status": "NO_OPEN_ORDER", "mode": self.mode.value, "cancelled": False}

    def get_recent_fills(self, symbol: str | None = None, *, limit: int = 50) -> list[dict[str, Any]]:
        rows = []
        for fr in self._fills[-limit:]:
            rows.append({
                "ok": fr.ok,
                "entry_ts": fr.entry_ts,
                "entry_mid": fr.entry_mid,
                "notional_usd": fr.notional_usd,
                "broker": fr.broker_name,
                "mode": self.mode.value,
            })
        return rows

    def place_protective_stop(
        self, position_id: str, *, stop_price: float, meta: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Paper stops remain Python-evaluated (existing exits.py). No exchange stop."""
        return {
            "position_id": position_id,
            "stop_price": float(stop_price),
            "status": "PAPER_LOCAL_STOP",
            "mode": self.mode.value,
            "exchange_native": False,
            "meta": meta or {},
        }

    def open_long_legs(
        self,
        *,
        alt_btc_entry_mid: float,
        btc_usdt: float,
        notional_usd: float,
        specs: list[StrategySpec],
        entry_ts: Any,
    ) -> list[Any]:
        """Identical paper fill path as SelectorLiveEngine historically used."""
        return open_opportunity_legs(
            alt_btc_entry_mid=alt_btc_entry_mid,
            btc_usdt=btc_usdt,
            notional_usd=notional_usd,
            costs=self._costs,
            specs=list(specs),
            entry_ts=entry_ts,
        )

    def _open_opps(self) -> list[dict[str, Any]]:
        if self._open_opps_provider is None:
            return []
        try:
            return list(self._open_opps_provider() or [])
        except Exception:
            return []
