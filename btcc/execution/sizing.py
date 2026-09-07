"""Production sizing: 25% of available free BTC, floor to precision, never round up."""

from __future__ import annotations

from dataclasses import dataclass

from btcc.execution.symbols import SymbolMeta, normalize_order_quantity


@dataclass(frozen=True)
class SizeDecision:
    ok: bool
    quantity: float = 0.0
    notional_btc: float = 0.0
    reason: str = ""
    # Pre-floor allocation quantity vs exchange-submitted (normalized) quantity.
    raw_quantity: float = 0.0
    quantity_serialized: str = ""


def size_alt_btc_position(
    *,
    available_btc: float,
    price_alt_btc: float,
    meta: SymbolMeta,
    allocation_pct: float = 0.25,
    open_exposure_pct: float = 0.0,
    max_aggregate_exposure: float = 1.0,
) -> SizeDecision:
    """Size LONG ALT/BTC using free BTC only. Never round quantity upward."""
    if available_btc <= 0:
        return SizeDecision(ok=False, reason="INSUFFICIENT_AVAILABLE_BTC")
    if price_alt_btc <= 0:
        return SizeDecision(ok=False, reason="INVALID_PRICE")
    if not meta.is_trading:
        return SizeDecision(ok=False, reason="SYMBOL_NOT_TRADING")
    if not meta.quote_asset or meta.quote_asset.upper() != "BTC":
        return SizeDecision(ok=False, reason="NOT_ALT_BTC_MARKET")
    if meta.quantity_step <= 0 or meta.min_quantity < 0:
        return SizeDecision(ok=False, reason="INCOMPLETE_SYMBOL_METADATA")
    if allocation_pct <= 0 or allocation_pct > 0.25 + 1e-12:
        return SizeDecision(ok=False, reason="ALLOCATION_EXCEEDS_25PCT")

    remaining_cap = max_aggregate_exposure - open_exposure_pct
    if remaining_cap <= 1e-12:
        return SizeDecision(ok=False, reason="MAX_AGGREGATE_EXPOSURE")

    target_frac = min(float(allocation_pct), float(remaining_cap))
    notional = float(available_btc) * target_frac
    if notional <= 0:
        return SizeDecision(ok=False, reason="ZERO_NOTIONAL")

    raw_qty = notional / float(price_alt_btc)
    norm = normalize_order_quantity(raw_qty, meta, price_alt_btc=float(price_alt_btc))
    if not norm.ok:
        return SizeDecision(
            ok=False,
            quantity=norm.quantity,
            raw_quantity=raw_qty,
            quantity_serialized=norm.serialized,
            reason=norm.reason or "NORMALIZE_FAILED",
        )

    qty = float(norm.quantity)
    floored_notional = qty * float(price_alt_btc)
    if floored_notional > notional + 1e-12:
        return SizeDecision(
            ok=False,
            quantity=qty,
            raw_quantity=raw_qty,
            quantity_serialized=norm.serialized,
            reason="NOTIONAL_EXCEEDS_TARGET",
        )
    if floored_notional > float(available_btc) + 1e-12:
        return SizeDecision(
            ok=False,
            quantity=qty,
            raw_quantity=raw_qty,
            quantity_serialized=norm.serialized,
            reason="INSUFFICIENT_AVAILABLE_BTC",
        )

    return SizeDecision(
        ok=True,
        quantity=qty,
        notional_btc=floored_notional,
        reason="OK",
        raw_quantity=raw_qty,
        quantity_serialized=norm.serialized,
    )
