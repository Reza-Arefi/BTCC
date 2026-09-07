"""Universe helpers + symbol validation against Binance exchangeInfo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from binance_btc_bot.exchange.base import ExchangeAdapter, SymbolInfo
from binance_btc_bot.strategy.relative_price import base_from_btc_pair, required_usdt_markets
from binance_btc_bot.strategy.trails import TrailStrategy, get_strategy


@dataclass(frozen=True)
class SymbolValidation:
    symbol: str
    ok: bool
    info: SymbolInfo | None
    reasons: tuple[str, ...]


def validate_universe(
    exchange: ExchangeAdapter,
    btc_pairs: list[str],
    *,
    strategy: TrailStrategy | None = None,
) -> list[SymbolValidation]:
    strategy = strategy or get_strategy("T1")
    out: list[SymbolValidation] = []
    infos: dict[str, SymbolInfo] = {}
    get_many = getattr(exchange, "get_symbol_infos", None)
    if callable(get_many):
        try:
            infos = get_many(list(btc_pairs))
        except Exception:
            infos = {}
    for pair in btc_pairs:
        reasons: list[str] = []
        info: SymbolInfo | None = infos.get(pair.upper())
        if info is None:
            try:
                info = exchange.get_symbol_info(pair)
            except Exception as e:  # noqa: BLE001 — collect per-symbol failures
                out.append(SymbolValidation(pair, False, None, (f"exchangeInfo_error:{e}",)))
                continue
        if not info.is_trading:
            reasons.append(f"status={info.status}")
        if info.quote_asset != "BTC":
            reasons.append(f"quote={info.quote_asset}")
        if not info.oco_allowed:
            reasons.append("oco_not_allowed")
        for need in ("MARKET", "TAKE_PROFIT", "STOP_LOSS"):
            if need not in info.order_types:
                reasons.append(f"missing_order_type:{need}")
        trail_bips = strategy.trail_distance_bips
        if info.min_trailing_above_delta is not None and trail_bips < info.min_trailing_above_delta:
            reasons.append(
                f"{strategy.key}_trail_{trail_bips}_bips_below_min_{info.min_trailing_above_delta}"
            )
        if info.max_trailing_above_delta is not None and trail_bips > info.max_trailing_above_delta:
            reasons.append(
                f"{strategy.key}_trail_{trail_bips}_bips_above_max_{info.max_trailing_above_delta}"
            )
        out.append(SymbolValidation(pair, not reasons, info, tuple(reasons)))
    return out


def validate_usdt_legs(exchange: ExchangeAdapter, btc_pairs: list[str]) -> dict[str, Any]:
    markets = required_usdt_markets(btc_pairs)
    ok: list[str] = []
    bad: dict[str, str] = {}
    prices: dict[str, float] = {}
    for m in markets:
        try:
            px = exchange.get_price(m)
            if px <= 0:
                bad[m] = "nonpositive_price"
            else:
                ok.append(m)
                prices[m] = px
        except Exception as e:  # noqa: BLE001
            bad[m] = str(e)
    return {"ok": ok, "bad": bad, "prices": prices, "bases": [base_from_btc_pair(p) for p in btc_pairs]}
