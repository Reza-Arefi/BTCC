"""REAL_SHADOW broker — MEXC Spot authenticated reads only.

Uses Stage-3 MexcReadOnlyClient. Never submits/cancels/places stops.
mode is always REAL_SHADOW (never REAL) so naming cannot imply write capability.
"""

from __future__ import annotations

from typing import Any

from btcc.execution.modes import ExecutionMode
from btcc.execution.mexc.client import MexcReadOnlyClient
from btcc.execution.mexc.credentials import MexcCredentials, load_mexc_credentials
from btcc.execution.mexc.errors import AccountDataUnavailable, MexcReadError, OrderDataUnavailable
from btcc.execution.mexc.models import AssetBalance, ExchangeFill, ExchangeOrder
from btcc.execution.safety_state import ExecutionSafetyState, REAL_TRADING_DISABLED
from btcc.execution.symbols import SymbolMeta
from btcc.execution.types import AccountState, FillReport, OrderRequest
from btcc.safety.no_trading import TradingForbiddenError


_NO_WRITE = (
    "ShadowBroker is REAL_SHADOW read-only. "
    "No MEXC order POST/DELETE/stop placement. "
    "Refusing to fall back to PAPER or REAL write path."
)


class ShadowBroker:
    """Observation-only broker for Stage 5A shadow reconciliation."""

    def __init__(
        self,
        *,
        client: MexcReadOnlyClient | None = None,
        credentials: MexcCredentials | None = None,
        load_env_credentials: bool = False,
        safety: ExecutionSafetyState | None = None,
    ) -> None:
        self._client = client
        if self._client is None and credentials is not None:
            self._client = MexcReadOnlyClient(credentials)
        if self._client is None and load_env_credentials:
            creds = load_mexc_credentials(require=True)
            assert creds is not None
            self._client = MexcReadOnlyClient(creds)
        self.safety = safety or ExecutionSafetyState(
            reason_code=REAL_TRADING_DISABLED,
            detail="REAL_SHADOW: trading impossible; observation only",
            real_entries_blocked=True,
        )

    @property
    def mode(self) -> ExecutionMode:
        return ExecutionMode.REAL_SHADOW

    @property
    def name(self) -> str:
        if self._client is not None:
            return "ShadowBroker(REAL_SHADOW_READ_ONLY)"
        return "ShadowBroker(UNCONFIGURED)"

    @property
    def client(self) -> MexcReadOnlyClient | None:
        return self._client

    def _require_client(self) -> MexcReadOnlyClient:
        if self._client is None:
            raise AccountDataUnavailable(
                "REAL_SHADOW credentials/client not configured "
                "(set MEXC_API_KEY/MEXC_API_SECRET for read-only key)"
            )
        return self._client

    def _deny_write(self, action: str) -> None:
        raise TradingForbiddenError(f"{_NO_WRITE} action={action}")

    # --- reads ----------------------------------------------------------

    def get_account_state(self) -> AccountState:
        client = self._require_client()
        try:
            snap = client.get_account_snapshot(quote_asset="BTC")
            opens = client.get_open_orders()
        except MexcReadError as e:
            self.safety.apply_read_error(e)
            raise
        return AccountState(
            mode=ExecutionMode.REAL_SHADOW.value,
            equity=float(snap.total_balance),
            available=float(snap.available_balance),
            reserved=float(snap.locked_balance),
            currency=snap.currency,
            open_positions=0,
            open_orders=len(opens),
            meta={"source": "mexc_shadow", "assets_returned": snap.meta.get("assets_returned")},
        )

    def get_balances(self) -> dict[str, float]:
        return {b.asset: float(b.free) for b in self.get_balances_detailed()}

    def get_balances_detailed(self) -> list[AssetBalance]:
        client = self._require_client()
        try:
            return client.get_balances()
        except MexcReadError as e:
            self.safety.apply_read_error(e)
            raise

    def get_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        return [self._order_dict(o) for o in self.get_open_orders_typed(symbol=symbol)]

    def get_open_orders_typed(self, symbol: str | None = None) -> list[ExchangeOrder]:
        client = self._require_client()
        try:
            return client.get_open_orders(symbol=symbol)
        except MexcReadError as e:
            self.safety.apply_read_error(e)
            raise

    def get_open_positions(self, symbol: str | None = None) -> list[dict[str, Any]]:
        bals = self.get_balances_detailed()
        out = []
        for b in bals:
            if b.total == 0:
                continue
            if symbol and b.asset.upper() not in str(symbol).upper():
                continue
            out.append(
                {"asset": b.asset, "free": str(b.free), "locked": str(b.locked), "total": str(b.total)}
            )
        return out

    def get_order(self, order_id: str, *, symbol: str | None = None) -> dict[str, Any]:
        if not symbol:
            raise OrderDataUnavailable("symbol required for MEXC order query")
        o = self.get_order_status(symbol=symbol, order_id=order_id)
        return self._order_dict(o)

    def get_order_status(
        self,
        *,
        symbol: str,
        order_id: str | None = None,
        orig_client_order_id: str | None = None,
    ) -> ExchangeOrder:
        client = self._require_client()
        try:
            return client.get_order_status(
                symbol=symbol,
                order_id=order_id,
                orig_client_order_id=orig_client_order_id,
            )
        except MexcReadError as e:
            self.safety.apply_read_error(e)
            raise

    def get_recent_fills(self, symbol: str | None = None, *, limit: int = 50) -> list[dict[str, Any]]:
        if not symbol:
            raise OrderDataUnavailable("symbol required for MEXC myTrades")
        return [self._fill_dict(f) for f in self.get_recent_fills_typed(symbol, limit=limit)]

    def get_recent_fills_typed(self, symbol: str, *, limit: int = 50) -> list[ExchangeFill]:
        client = self._require_client()
        try:
            return client.get_recent_fills(symbol, limit=limit)
        except MexcReadError as e:
            self.safety.apply_read_error(e)
            raise

    def get_symbol_metadata(self, symbol: str) -> SymbolMeta:
        client = self._require_client()
        try:
            return client.get_symbol_metadata(symbol)
        except MexcReadError as e:
            self.safety.apply_read_error(e)
            raise

    def get_server_time(self) -> int:
        return self._require_client().get_server_time()

    # --- writes always denied -------------------------------------------

    def submit_order(self, request: OrderRequest) -> FillReport:
        self._deny_write("submit_order")
        raise AssertionError("unreachable")

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        self._deny_write("cancel_order")
        raise AssertionError("unreachable")

    def place_protective_stop(
        self, position_id: str, *, stop_price: float, meta: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        self._deny_write("place_protective_stop")
        raise AssertionError("unreachable")

    def open_long_legs(self, **kwargs: Any) -> list[Any]:
        self._deny_write("open_long_legs")
        raise AssertionError("unreachable")

    @staticmethod
    def _order_dict(o: ExchangeOrder) -> dict[str, Any]:
        return {
            "order_id": o.order_id,
            "client_order_id": o.client_order_id,
            "symbol": o.symbol,
            "side": o.side,
            "type": o.type,
            "status": o.status,
            "original_quantity": str(o.original_quantity),
            "executed_quantity": str(o.executed_quantity),
            "remaining_quantity": str(o.remaining_quantity),
            "price": str(o.price) if o.price is not None else None,
            "average_price": str(o.average_price) if o.average_price is not None else None,
            "time": o.time,
            "update_time": o.update_time,
        }

    @staticmethod
    def _fill_dict(f: ExchangeFill) -> dict[str, Any]:
        return {
            "trade_id": f.trade_id,
            "order_id": f.order_id,
            "symbol": f.symbol,
            "quantity": str(f.quantity),
            "price": str(f.price),
            "fee": str(f.fee) if f.fee is not None else None,
            "fee_asset": f.fee_asset,
            "timestamp": f.timestamp,
        }
