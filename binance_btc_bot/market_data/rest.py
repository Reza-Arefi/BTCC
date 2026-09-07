"""REST market-data helpers (sync + reconciliation)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from binance_btc_bot.exchange.base import ExchangeAdapter
from binance_btc_bot.strategy.relative_price import (
    compute_relative_price,
    required_usdt_markets,
    base_from_btc_pair,
)


@dataclass
class PriceBook:
    prices: dict[str, float] = field(default_factory=dict)
    updated_at: float = 0.0

    def age_sec(self) -> float:
        if self.updated_at <= 0:
            return float("inf")
        return time.time() - self.updated_at

    def get(self, symbol: str) -> float | None:
        return self.prices.get(symbol.upper())


class RestMarketData:
    def __init__(self, exchange: ExchangeAdapter, stale_after_sec: float = 30.0) -> None:
        self.exchange = exchange
        self.stale_after_sec = float(stale_after_sec)
        self.book = PriceBook()

    def sync(self, btc_pairs: list[str]) -> PriceBook:
        usdt = required_usdt_markets(btc_pairs)
        symbols = sorted(set([p.upper() for p in btc_pairs]) | set(usdt))
        # Chunk to keep URL reasonable
        merged: dict[str, float] = {}
        chunk = 50
        for i in range(0, len(symbols), chunk):
            part = symbols[i : i + chunk]
            merged.update(self.exchange.get_prices(part))
        self.book = PriceBook(prices=merged, updated_at=time.time())
        return self.book

    def is_stale(self) -> bool:
        return self.book.age_sec() > self.stale_after_sec

    def relative_for(self, btc_pair: str) -> dict[str, Any]:
        base = base_from_btc_pair(btc_pair)
        usdt = f"{base}USDT"
        base_px = self.book.get(usdt)
        btc_px = self.book.get("BTCUSDT")
        native = self.book.get(btc_pair.upper())
        if base_px is None or btc_px is None:
            raise KeyError(f"missing USDT legs for {btc_pair}")
        rel = compute_relative_price(
            btc_pair=btc_pair,
            base_usdt=base_px,
            btc_usdt=btc_px,
            native_btc_price=native,
        )
        return {
            "symbol": btc_pair.upper(),
            "relative_price": rel.relative_price,
            "base_usdt": rel.base_usdt,
            "btc_usdt": rel.btc_usdt,
            "native_btc_price": rel.native_btc_price,
            "native_vs_relative_bps": rel.native_vs_relative_bps,
        }
