"""Exchange-neutral adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Balance:
    asset: str
    free: float
    locked: float

    @property
    def total(self) -> float:
        return float(self.free) + float(self.locked)


@dataclass(frozen=True)
class AccountSnapshot:
    balances: dict[str, Balance]
    raw: dict[str, Any] = field(default_factory=dict)

    def free(self, asset: str) -> float:
        bal = self.balances.get(asset.upper())
        return float(bal.free) if bal else 0.0


@dataclass(frozen=True)
class SymbolInfo:
    symbol: str
    status: str
    base_asset: str
    quote_asset: str
    quantity_step: float
    min_quantity: float
    max_quantity: float | None
    price_tick: float
    min_notional: float
    order_types: tuple[str, ...]
    oco_allowed: bool
    min_trailing_above_delta: int | None
    max_trailing_above_delta: int | None
    min_trailing_below_delta: int | None
    max_trailing_below_delta: int | None
    raw_filters: dict[str, Any] = field(default_factory=dict)

    @property
    def is_trading(self) -> bool:
        return str(self.status).upper() == "TRADING"


@dataclass(frozen=True)
class OrderRequest:
    symbol: str
    side: str
    order_type: str
    quantity: float | None = None
    quote_order_qty: float | None = None
    price: float | None = None
    stop_price: float | None = None
    trailing_delta: int | None = None
    time_in_force: str | None = None
    client_order_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrailingOcoRequest:
    """Binance-native OCO exit: trailing TAKE_PROFIT above + STOP_LOSS below."""

    symbol: str
    side: str
    quantity: float
    above_type: str
    above_stop_price: float
    above_trailing_delta: int
    below_type: str
    below_stop_price: float
    list_client_order_id: str | None = None
    above_price: float | None = None
    below_price: float | None = None
    above_time_in_force: str | None = None
    below_time_in_force: str | None = None
    new_order_resp_type: str = "RESULT"


@dataclass(frozen=True)
class OrderResult:
    ok: bool
    order_id: str | None = None
    client_order_id: str | None = None
    status: str | None = None
    symbol: str | None = None
    side: str | None = None
    order_type: str | None = None
    price: float | None = None
    quantity: float | None = None
    executed_qty: float | None = None
    cumulative_quote_qty: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None
    dry_run: bool = False


class ExchangeAdapter(ABC):
    """Strategy-facing exchange interface (no Binance SDK leakage upward)."""

    @abstractmethod
    def ping(self) -> bool: ...

    @abstractmethod
    def get_account(self) -> AccountSnapshot: ...

    @abstractmethod
    def get_balance(self, asset: str) -> Balance: ...

    @abstractmethod
    def get_price(self, symbol: str) -> float: ...

    @abstractmethod
    def get_prices(self, symbols: list[str]) -> dict[str, float]: ...

    @abstractmethod
    def get_symbol_info(self, symbol: str) -> SymbolInfo: ...

    @abstractmethod
    def place_entry(self, request: OrderRequest) -> OrderResult:
        """Place an entry order (typically MARKET BUY). Implemented via ``place_order``."""

    @abstractmethod
    def place_order(self, request: OrderRequest) -> OrderResult:
        """Low-level authenticated Spot order placement (``POST /api/v3/order``).

        No portfolio / signal / allocation semantics — params only.
        """

    @abstractmethod
    def place_protective_sell(self, request: OrderRequest) -> OrderResult:
        """Place a protective SELL (STOP_LOSS / contingency). Refuses BUY and MARKET."""

    @abstractmethod
    def place_trailing_exit(self, request: TrailingOcoRequest) -> OrderResult: ...

    @abstractmethod
    def cancel_order(self, symbol: str, order_id: str | None = None, client_order_id: str | None = None) -> OrderResult: ...

    @abstractmethod
    def cancel_order_list(self, symbol: str, order_list_id: str | None = None, list_client_order_id: str | None = None) -> OrderResult: ...

    @abstractmethod
    def get_order(self, symbol: str, order_id: str | None = None, client_order_id: str | None = None) -> OrderResult: ...

    @abstractmethod
    def get_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]: ...

    @abstractmethod
    def get_open_order_lists(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """Return open OCO/order lists.

        ``symbol`` is an optional *client-side* filter only. Binance
        ``GET /api/v3/openOrderList`` does not accept ``symbol``.
        """

    @abstractmethod
    def get_order_list(
        self,
        *,
        order_list_id: str | None = None,
        orig_client_order_id: str | None = None,
    ) -> dict[str, Any]:
        """Authoritative single order-list query (``GET /api/v3/orderList``)."""

    @abstractmethod
    def get_my_trades(self, symbol: str, limit: int = 50) -> list[dict[str, Any]]: ...
