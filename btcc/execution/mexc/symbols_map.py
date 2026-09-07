"""Map MEXC Spot exchangeInfo → SymbolMeta (no invented LOT_SIZE)."""

from __future__ import annotations

from typing import Any

from btcc.execution.mexc.errors import SymbolMetadataUnavailable
from btcc.execution.symbols import SymbolMeta


def _status_label(raw: dict[str, Any]) -> str:
    if raw.get("isSpotTradingAllowed") is True and "SPOT" in (raw.get("permissions") or []):
        # MEXC uses numeric status "1" for active; normalize.
        return "TRADING"
    return str(raw.get("status", "UNKNOWN"))


def map_exchange_info_symbol(raw: dict[str, Any]) -> SymbolMeta:
    """Map one exchangeInfo symbol object.

    MEXC Spot often omits Binance-style LOT_SIZE filters and instead exposes:
      baseSizePrecision, quoteAmountPrecision, baseAssetPrecision, quoteAssetPrecision,
      maxQuoteAmount / maxQuoteAmountMarket

    If we cannot derive a positive quantity_step, raise SymbolMetadataUnavailable
    so REAL readiness stays BLOCKED (no silent paper substitution).
    """
    if not isinstance(raw, dict) or "symbol" not in raw:
        raise SymbolMetadataUnavailable("malformed exchangeInfo symbol")

    symbol = str(raw["symbol"])
    base = str(raw.get("baseAsset") or "")
    quote = str(raw.get("quoteAsset") or "")
    if not base or not quote:
        raise SymbolMetadataUnavailable(f"{symbol}: missing base/quote asset")

    # Quantity step
    step = None
    bsp = raw.get("baseSizePrecision")
    if bsp is not None and str(bsp) not in ("",):
        try:
            step_f = float(bsp)
            if step_f > 0:
                step = step_f
            elif step_f == 0:
                # Integer base lots (observed for some alts)
                step = 1.0
        except (TypeError, ValueError) as e:
            raise SymbolMetadataUnavailable(f"{symbol}: invalid baseSizePrecision") from e
    if step is None:
        bap = raw.get("baseAssetPrecision")
        if bap is not None:
            try:
                exp = int(bap)
                step = 10 ** (-exp) if exp > 0 else 1.0
            except (TypeError, ValueError) as e:
                raise SymbolMetadataUnavailable(f"{symbol}: invalid baseAssetPrecision") from e
    if step is None or step <= 0:
        raise SymbolMetadataUnavailable(f"{symbol}: quantity step unavailable")

    # Price tick
    tick = None
    qap = raw.get("quoteAssetPrecision", raw.get("quotePrecision"))
    if qap is not None:
        try:
            exp = int(qap)
            tick = 10 ** (-exp) if exp > 0 else 1.0
        except (TypeError, ValueError) as e:
            raise SymbolMetadataUnavailable(f"{symbol}: invalid quote precision") from e
    if tick is None or tick <= 0:
        raise SymbolMetadataUnavailable(f"{symbol}: price tick unavailable")

    # Min notional — use quoteAmountPrecision as minimum quote increment / notional proxy
    min_notional = 0.0
    qnp = raw.get("quoteAmountPrecision")
    if qnp is not None and str(qnp) not in ("",):
        try:
            min_notional = float(qnp)
        except (TypeError, ValueError) as e:
            raise SymbolMetadataUnavailable(f"{symbol}: invalid quoteAmountPrecision") from e

    min_qty = float(step)
    max_qty = None
    # No reliable max base qty on MEXC info; leave None rather than invent.

    raw_types = raw.get("orderTypes") or raw.get("order_types") or []
    order_types: tuple[str, ...] | None
    if isinstance(raw_types, (list, tuple)):
        order_types = tuple(str(t).upper() for t in raw_types)
    else:
        order_types = None

    return SymbolMeta(
        symbol=symbol,
        status=_status_label(raw),
        base_asset=base,
        quote_asset=quote,
        quantity_step=float(step),
        min_quantity=min_qty,
        price_tick=float(tick),
        min_notional=float(min_notional),
        max_quantity=max_qty,
        source="mexc_exchangeInfo",
        order_types=order_types,
    )
