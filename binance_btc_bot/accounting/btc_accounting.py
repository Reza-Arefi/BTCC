"""BTC-centric trade accounting."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class TradeAccounting:
    trade_id: str
    symbol: str
    strategy: str
    selector: str | None
    entry_time: str | None
    entry_price: float
    quantity: float
    entry_btc_value: float
    entry_usdt_value: float | None
    configured_activation: float
    configured_trailing_distance: float
    binance_entry_order_id: str | None = None
    binance_oco_list_id: str | None = None
    exit_time: str | None = None
    exit_price: float | None = None
    exit_btc_value: float | None = None
    exit_usdt_value: float | None = None
    fees_btc: float = 0.0
    fees_usdt: float = 0.0
    realized_pnl_btc: float | None = None
    realized_pnl_usdt: float | None = None
    realized_pnl_btc_equivalent: float | None = None

    def mark_exit(
        self,
        *,
        exit_time: str,
        exit_price: float,
        btc_usdt: float | None = None,
        fees_btc: float = 0.0,
        fees_usdt: float = 0.0,
    ) -> None:
        self.exit_time = exit_time
        self.exit_price = float(exit_price)
        self.exit_btc_value = float(self.quantity) * float(exit_price)
        self.fees_btc += float(fees_btc)
        self.fees_usdt += float(fees_usdt)
        self.realized_pnl_btc = self.exit_btc_value - self.entry_btc_value - self.fees_btc
        if btc_usdt and btc_usdt > 0:
            self.exit_usdt_value = self.exit_btc_value * float(btc_usdt)
            if self.entry_usdt_value is not None:
                self.realized_pnl_usdt = self.exit_usdt_value - self.entry_usdt_value - self.fees_usdt
            self.realized_pnl_btc_equivalent = self.realized_pnl_btc
        else:
            self.realized_pnl_btc_equivalent = self.realized_pnl_btc

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def equity_btc_equivalent(
    *,
    btc_free: float,
    btc_locked: float,
    usdt_free: float,
    btc_usdt: float,
    alt_balances_btc_value: float = 0.0,
) -> float:
    usdt_as_btc = (float(usdt_free) / float(btc_usdt)) if btc_usdt > 0 else 0.0
    return float(btc_free) + float(btc_locked) + usdt_as_btc + float(alt_balances_btc_value)
