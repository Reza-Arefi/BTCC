"""Relative-price construction: BASEUSDT / BTCUSDT (not native *BTC as signal)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RelativePrice:
    base: str
    btc_pair: str
    usdt_pair: str
    base_usdt: float
    btc_usdt: float
    relative_price: float
    native_btc_price: float | None = None

    @property
    def native_vs_relative_bps(self) -> float | None:
        if self.native_btc_price is None or self.native_btc_price <= 0:
            return None
        return (self.relative_price / self.native_btc_price - 1.0) * 10_000


def base_from_btc_pair(btc_pair: str) -> str:
    sym = btc_pair.upper()
    if not sym.endswith("BTC"):
        raise ValueError(f"expected *BTC pair, got {btc_pair}")
    if sym == "BTCUSDT":
        raise ValueError("BTCUSDT is the reference market, not an alt/BTC pair")
    return sym[:-3]


def usdt_pair_for_base(base: str) -> str:
    return f"{base.upper()}USDT"


def compute_relative_price(
    *,
    btc_pair: str,
    base_usdt: float,
    btc_usdt: float,
    native_btc_price: float | None = None,
) -> RelativePrice:
    if base_usdt <= 0 or btc_usdt <= 0:
        raise ValueError("USDT legs must be > 0")
    base = base_from_btc_pair(btc_pair)
    return RelativePrice(
        base=base,
        btc_pair=btc_pair.upper(),
        usdt_pair=usdt_pair_for_base(base),
        base_usdt=float(base_usdt),
        btc_usdt=float(btc_usdt),
        relative_price=float(base_usdt) / float(btc_usdt),
        native_btc_price=float(native_btc_price) if native_btc_price is not None else None,
    )


def required_usdt_markets(btc_pairs: list[str]) -> list[str]:
    bases = [base_from_btc_pair(p) for p in btc_pairs]
    return sorted({usdt_pair_for_base(b) for b in bases} | {"BTCUSDT"})
