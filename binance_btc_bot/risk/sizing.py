"""Position sizing — allocation ≤ portfolio cap AND planned loss ≤ 0.5% equity.

Rounding (LOT_SIZE / step) must never push actual allocation above the requested
allocation fraction of trading equity.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN

from binance_btc_bot.exchange.base import SymbolInfo
from binance_btc_bot.strategy.trails import TrailStrategy


@dataclass(frozen=True)
class SizeDecision:
    ok: bool
    quantity: float = 0.0
    notional_btc: float = 0.0
    risk_budget_btc: float = 0.0
    planned_loss_btc: float = 0.0
    reason: str = ""
    raw_quantity: float = 0.0
    quantity_serialized: str = ""
    requested_allocation_pct: float = 0.0
    actual_allocation_pct: float = 0.0
    requested_notional_btc: float = 0.0


def floor_to_step(quantity: float, step: float) -> float:
    if quantity < 0:
        raise ValueError("quantity must be >= 0")
    if step <= 0:
        raise ValueError("step must be > 0")
    q = Decimal(str(quantity))
    s = Decimal(str(step))
    units = (q / s).to_integral_value(rounding=ROUND_DOWN)
    return float(units * s)


def serialize_quantity(quantity: float, step: float) -> str:
    q = Decimal(str(quantity)).quantize(Decimal(str(step)), rounding=ROUND_DOWN)
    text = format(q, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def size_position(
    *,
    equity_btc: float,
    available_btc: float,
    price_alt_btc: float,
    strategy: TrailStrategy,
    meta: SymbolInfo,
    max_loss_per_trade: float = 0.005,
    max_allocation_pct: float = 0.125,
    open_exposure_pct: float = 0.0,
    max_aggregate_exposure: float = 1.0,
    fee_buffer_pct: float = 0.002,
) -> SizeDecision:
    """Size LONG ALT/BTC from risk budget at the strategy hard SL.

    Both constraints must hold:
      actual_allocation_pct <= max_allocation_pct
      planned_loss_btc <= max_loss_per_trade * equity_btc
    """
    requested_alloc = float(max_allocation_pct)
    if equity_btc <= 0:
        return SizeDecision(ok=False, reason="NONPOSITIVE_EQUITY", requested_allocation_pct=requested_alloc)
    if available_btc <= 0:
        return SizeDecision(ok=False, reason="INSUFFICIENT_AVAILABLE_BTC", requested_allocation_pct=requested_alloc)
    if price_alt_btc <= 0:
        return SizeDecision(ok=False, reason="INVALID_PRICE", requested_allocation_pct=requested_alloc)
    if not meta.is_trading:
        return SizeDecision(ok=False, reason="SYMBOL_NOT_TRADING", requested_allocation_pct=requested_alloc)
    if meta.quote_asset.upper() != "BTC":
        return SizeDecision(ok=False, reason="NOT_ALT_BTC_MARKET", requested_allocation_pct=requested_alloc)
    if strategy.arm_sl_activation_trail <= 0:
        return SizeDecision(ok=False, reason="INVALID_STOP_LOSS", requested_allocation_pct=requested_alloc)
    if meta.quantity_step <= 0:
        return SizeDecision(ok=False, reason="INCOMPLETE_SYMBOL_METADATA", requested_allocation_pct=requested_alloc)

    remaining_cap = float(max_aggregate_exposure) - float(open_exposure_pct)
    if remaining_cap <= 1e-12:
        return SizeDecision(ok=False, reason="MAX_AGGREGATE_EXPOSURE", requested_allocation_pct=requested_alloc)

    risk_budget = float(equity_btc) * float(max_loss_per_trade)
    notional_risk = risk_budget / float(strategy.arm_sl_activation_trail)

    # Allocation is of trading equity (not free balance alone).
    fee_buf = max(0.0, float(fee_buffer_pct))
    alloc_headroom = max(0.0, requested_alloc * (1.0 - fee_buf))
    requested_notional = float(equity_btc) * alloc_headroom
    notional_alloc = min(requested_notional, float(available_btc), float(equity_btc) * remaining_cap)
    notional = min(notional_risk, notional_alloc, float(available_btc))
    if notional <= 0:
        return SizeDecision(
            ok=False,
            reason="ZERO_NOTIONAL",
            risk_budget_btc=risk_budget,
            requested_allocation_pct=requested_alloc,
            requested_notional_btc=requested_notional,
        )

    raw_qty = notional / float(price_alt_btc)
    qty = floor_to_step(raw_qty, meta.quantity_step)
    if qty <= 0 or qty < meta.min_quantity:
        return SizeDecision(
            ok=False,
            quantity=qty,
            raw_quantity=raw_qty,
            risk_budget_btc=risk_budget,
            reason="BELOW_MIN_QUANTITY",
            requested_allocation_pct=requested_alloc,
            requested_notional_btc=requested_notional,
        )
    if meta.max_quantity is not None and qty > meta.max_quantity:
        qty = floor_to_step(float(meta.max_quantity), meta.quantity_step)

    # Ensure floored notional cannot exceed requested allocation of equity.
    max_notional_by_alloc = float(equity_btc) * requested_alloc
    notional_btc = qty * float(price_alt_btc)
    if notional_btc > max_notional_by_alloc + 1e-12:
        # Step down one lot until within allocation (or fail).
        step = float(meta.quantity_step)
        while qty > 0 and qty * float(price_alt_btc) > max_notional_by_alloc + 1e-12:
            qty = floor_to_step(qty - step, step)
        notional_btc = qty * float(price_alt_btc)
        if qty <= 0 or notional_btc > max_notional_by_alloc + 1e-12:
            return SizeDecision(
                ok=False,
                quantity=qty,
                notional_btc=notional_btc,
                raw_quantity=raw_qty,
                risk_budget_btc=risk_budget,
                reason="ALLOCATION_ROUNDING_EXCEEDED",
                requested_allocation_pct=requested_alloc,
                requested_notional_btc=requested_notional,
                actual_allocation_pct=(notional_btc / float(equity_btc)) if equity_btc else 0.0,
            )

    if notional_btc + 1e-12 < meta.min_notional:
        return SizeDecision(
            ok=False,
            quantity=qty,
            notional_btc=notional_btc,
            raw_quantity=raw_qty,
            risk_budget_btc=risk_budget,
            reason="BELOW_MIN_NOTIONAL",
            requested_allocation_pct=requested_alloc,
            requested_notional_btc=requested_notional,
            actual_allocation_pct=notional_btc / float(equity_btc),
        )
    if notional_btc > float(available_btc) + 1e-12:
        return SizeDecision(
            ok=False,
            reason="INSUFFICIENT_AVAILABLE_BTC",
            risk_budget_btc=risk_budget,
            requested_allocation_pct=requested_alloc,
            requested_notional_btc=requested_notional,
        )

    planned_loss = notional_btc * float(strategy.arm_sl_activation_trail)
    actual_alloc_pct = notional_btc / float(equity_btc)
    if planned_loss > risk_budget + 1e-12:
        return SizeDecision(
            ok=False,
            quantity=qty,
            notional_btc=notional_btc,
            risk_budget_btc=risk_budget,
            planned_loss_btc=planned_loss,
            reason="PLANNED_LOSS_EXCEEDS_BUDGET",
            requested_allocation_pct=requested_alloc,
            actual_allocation_pct=actual_alloc_pct,
            requested_notional_btc=requested_notional,
        )
    if actual_alloc_pct > requested_alloc + 1e-12:
        return SizeDecision(
            ok=False,
            quantity=qty,
            notional_btc=notional_btc,
            risk_budget_btc=risk_budget,
            planned_loss_btc=planned_loss,
            reason="ALLOCATION_EXCEEDED",
            requested_allocation_pct=requested_alloc,
            actual_allocation_pct=actual_alloc_pct,
            requested_notional_btc=requested_notional,
        )

    return SizeDecision(
        ok=True,
        quantity=qty,
        notional_btc=notional_btc,
        risk_budget_btc=risk_budget,
        planned_loss_btc=planned_loss,
        reason="OK",
        raw_quantity=raw_qty,
        quantity_serialized=serialize_quantity(qty, meta.quantity_step),
        requested_allocation_pct=requested_alloc,
        actual_allocation_pct=actual_alloc_pct,
        requested_notional_btc=requested_notional,
    )
