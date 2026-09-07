"""Broker protocol — only this layer may eventually talk to exchange trading APIs.

Stage 1: interface + PAPER adapter + REAL fail-closed stub.
No MEXC private endpoints are called from any implementation in this package.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from btcc.execution.modes import ExecutionMode
from btcc.execution.types import AccountState, FillReport, OrderRequest


@runtime_checkable
class Broker(Protocol):
    """Exchange-facing trading port (paper or real)."""

    @property
    def mode(self) -> ExecutionMode: ...

    @property
    def name(self) -> str: ...

    def get_account_state(self) -> AccountState: ...

    def get_balances(self) -> dict[str, float]: ...

    def get_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]: ...

    def get_open_positions(self, symbol: str | None = None) -> list[dict[str, Any]]: ...

    def submit_order(self, request: OrderRequest) -> FillReport: ...

    def get_order(self, order_id: str) -> dict[str, Any]: ...

    def cancel_order(self, order_id: str) -> dict[str, Any]: ...

    def get_recent_fills(self, symbol: str | None = None, *, limit: int = 50) -> list[dict[str, Any]]: ...

    def place_protective_stop(self, position_id: str, *, stop_price: float, meta: dict[str, Any] | None = None) -> dict[str, Any]: ...

    def open_long_legs(
        self,
        *,
        alt_btc_entry_mid: float,
        btc_usdt: float,
        notional_usd: float,
        specs: list[Any],
        entry_ts: Any,
    ) -> list[Any]:
        """Open strategy legs for an approved intent (paper adapter uses existing sim fills)."""
        ...
