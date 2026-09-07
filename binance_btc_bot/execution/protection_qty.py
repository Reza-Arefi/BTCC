"""Protectable sell quantity after BUY fills (fee-aware, balance-aware).

Binance remains authoritative for fees. This module never assumes commissionAsset
and never manually deducts BNB — it only subtracts base-asset commission when
Binance explicitly reports commissionAsset == BASE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from binance_btc_bot.exchange.base import SymbolInfo
from binance_btc_bot.execution.fills import AggregatedFill, FillLeg
from binance_btc_bot.risk.sizing import floor_to_step


@dataclass(frozen=True)
class ProtectableQuantity:
    ok: bool
    quantity: float
    sellable_from_fills: float
    free_base: float
    reason: str
    notes: tuple[str, ...] = ()
    commissions_by_asset: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "quantity": self.quantity,
            "sellable_from_fills": self.sellable_from_fills,
            "free_base": self.free_base,
            "reason": self.reason,
            "notes": list(self.notes),
            "commissions_by_asset": dict(self.commissions_by_asset),
        }


def commissions_by_asset(fills: AggregatedFill | Mapping[str, Any] | None) -> dict[str, float]:
    """Sum commissions keyed by commissionAsset (empty asset → UNKNOWN, never assumed base)."""
    out: dict[str, float] = {}
    legs: tuple[FillLeg, ...] | list[FillLeg] = ()
    if isinstance(fills, AggregatedFill):
        legs = fills.fills
    elif isinstance(fills, Mapping):
        raw = fills.get("fills") or []
        for row in raw:
            if isinstance(row, FillLeg):
                legs = list(legs) + [row]  # type: ignore[assignment]
            elif isinstance(row, Mapping):
                asset = str(row.get("commissionAsset") or row.get("commission_asset") or "").upper() or "UNKNOWN"
                c = float(row.get("commission") or row.get("c") or 0)
                out[asset] = out.get(asset, 0.0) + c
        return out
    for leg in legs:
        asset = str(leg.commission_asset or "").upper() or "UNKNOWN"
        out[asset] = out.get(asset, 0.0) + float(leg.commission or 0)
    return out


def sellable_qty_from_fills(fill: AggregatedFill, base_asset: str) -> float:
    """Fill-derived max sellable BASE qty.

    If commissionAsset == BASE: subtract that commission from executed qty.
    If commissionAsset is anything else (BNB, quote, UNKNOWN): do NOT reduce base.
    Never assume a missing commissionAsset is the base asset.
    """
    base = str(base_asset or "").upper()
    total_qty = float(fill.executed_qty or 0)
    if total_qty <= 0:
        return 0.0
    base_fee = 0.0
    for leg in fill.fills:
        asset = str(leg.commission_asset or "").upper()
        if not asset:
            # Missing asset — do not assume BASE; leave qty unreduced for this leg.
            continue
        if asset == base:
            base_fee += float(leg.commission or 0)
    sellable = total_qty - base_fee
    return max(0.0, sellable)


def compute_protectable_quantity(
    *,
    fill: AggregatedFill,
    base_asset: str,
    free_base: float,
    meta: SymbolInfo,
    ref_price: float,
) -> ProtectableQuantity:
    """min(fill-sellable, free balance), FLOOR to LOT_SIZE step; never round up."""
    notes: list[str] = []
    by_asset = commissions_by_asset(fill)
    sellable = sellable_qty_from_fills(fill, base_asset)
    notes.append(f"sellable_from_fills={sellable}")
    notes.append(f"commissions_by_asset={by_asset}")
    free = max(0.0, float(free_base))
    notes.append(f"free_base={free}")
    capped = min(sellable, free)
    notes.append(f"min_cap={capped}")

    step = float(meta.quantity_step or 0)
    if step > 0:
        qty = floor_to_step(capped, step)
        notes.append(f"floored_step={step}→{qty}")
    else:
        qty = capped
        notes.append("no_step_floor")

    if qty <= 0:
        return ProtectableQuantity(
            False, 0.0, sellable, free, "PROTECT_QTY_ZERO", tuple(notes), by_asset
        )
    if meta.min_quantity > 0 and qty + 1e-15 < float(meta.min_quantity):
        return ProtectableQuantity(
            False, 0.0, sellable, free, "MIN_QTY", tuple(notes), by_asset
        )
    px = float(ref_price or fill.avg_price or 0)
    notion = qty * px if px > 0 else 0.0
    if meta.min_notional > 0 and notion + 1e-15 < float(meta.min_notional):
        return ProtectableQuantity(
            False, 0.0, sellable, free, "MIN_NOTIONAL", tuple(notes), by_asset
        )
    if qty > free + 1e-12:
        # Defensive — should be impossible after min+floor.
        return ProtectableQuantity(
            False, 0.0, sellable, free, "EXCEEDS_FREE_BALANCE", tuple(notes), by_asset
        )
    return ProtectableQuantity(True, qty, sellable, free, "OK", tuple(notes), by_asset)


def is_insufficient_balance_reason(reason: str | None) -> bool:
    r = str(reason or "").lower()
    return "-2010" in r or "insufficient balance" in r or "insufficient_balance" in r


@dataclass(frozen=True)
class ResidualBaseClassification:
    """Distinguish protected qty vs unprotected dust vs fully closed inventory."""

    sellable_from_fills: float
    free_base: float
    protectable_qty: float
    residual_unprotected: float
    category: str
    counts_as_trading_position: bool
    reason: str
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "sellable_from_fills": self.sellable_from_fills,
            "free_base": self.free_base,
            "protectable_qty": self.protectable_qty,
            "residual_unprotected": self.residual_unprotected,
            "category": self.category,
            "counts_as_trading_position": self.counts_as_trading_position,
            "reason": self.reason,
            "notes": list(self.notes),
        }


def classify_residual_base(
    *,
    fill: AggregatedFill,
    base_asset: str,
    free_base: float,
    meta: SymbolInfo,
    ref_price: float,
    protected_qty: float = 0.0,
) -> ResidualBaseClassification:
    """Classify leftover BASE after fees / LOT_SIZE flooring / protection.

    Dust below Binance minQty/minNotional must not be treated as a protected
    trading position and must not trigger invalid sell orders.
    """
    notes: list[str] = []
    pq = compute_protectable_quantity(
        fill=fill,
        base_asset=base_asset,
        free_base=free_base,
        meta=meta,
        ref_price=ref_price,
    )
    sellable = pq.sellable_from_fills
    free = max(0.0, float(free_base))
    prot = max(0.0, float(protected_qty))
    notes.extend(pq.notes)

    if free <= 1e-15 and sellable <= 1e-15:
        return ResidualBaseClassification(
            sellable, free, 0.0, 0.0, "FULLY_CLOSED", False, "FULLY_CLOSED", tuple(notes)
        )

    if prot > 0:
        residual = max(0.0, free - prot)
        notes.append(f"protected_qty={prot}")
        notes.append(f"residual_after_protect={residual}")
        if residual <= 1e-15:
            return ResidualBaseClassification(
                sellable, free, prot, 0.0, "PROTECTED_NO_DUST", True, "OK", tuple(notes)
            )
        # Residual after flooring / fee that cannot form another valid order.
        step = float(meta.quantity_step or 0)
        floored = floor_to_step(residual, step) if step > 0 else residual
        if floored <= 0 or (meta.min_quantity > 0 and floored + 1e-15 < meta.min_quantity):
            return ResidualBaseClassification(
                sellable,
                free,
                prot,
                residual,
                "LOT_SIZE_DUST",
                True,  # position is protected; dust is ancillary
                "LOT_SIZE_DUST",
                tuple(notes + [f"dust={residual}"]),
            )
        px = float(ref_price or 0)
        if meta.min_notional > 0 and floored * px + 1e-15 < meta.min_notional:
            return ResidualBaseClassification(
                sellable,
                free,
                prot,
                residual,
                "LOT_SIZE_DUST",
                True,
                "MIN_NOTIONAL_DUST",
                tuple(notes + [f"dust={residual}"]),
            )
        return ResidualBaseClassification(
            sellable,
            free,
            prot,
            residual,
            "UNPROTECTED_REMAINDER",
            True,
            "UNPROTECTED_REMAINDER",
            tuple(notes),
        )

    # No protection placed — classify free inventory.
    if pq.ok and pq.quantity > 0:
        residual = max(0.0, free - pq.quantity)
        cat = "LOT_SIZE_DUST" if residual > 1e-15 else "PROTECTABLE"
        return ResidualBaseClassification(
            sellable,
            free,
            pq.quantity,
            residual,
            cat,
            True,
            pq.reason,
            tuple(notes),
        )

    if pq.reason == "MIN_QTY":
        return ResidualBaseClassification(
            sellable, free, 0.0, free, "DUST_BELOW_MIN_QTY", False, "MIN_QTY", tuple(notes)
        )
    if pq.reason == "MIN_NOTIONAL":
        return ResidualBaseClassification(
            sellable,
            free,
            0.0,
            free,
            "DUST_BELOW_MIN_NOTIONAL",
            False,
            "MIN_NOTIONAL",
            tuple(notes),
        )
    if pq.reason == "PROTECT_QTY_ZERO" and free > 1e-15:
        # Floored to zero by LOT_SIZE or dust below step/min.
        if meta.min_quantity > 0 and free + 1e-15 < float(meta.min_quantity):
            return ResidualBaseClassification(
                sellable, free, 0.0, free, "DUST_BELOW_MIN_QTY", False, "MIN_QTY", tuple(notes)
            )
        return ResidualBaseClassification(
            sellable, free, 0.0, free, "LOT_SIZE_DUST", False, "PROTECT_QTY_ZERO", tuple(notes)
        )
    if free <= 1e-15:
        return ResidualBaseClassification(
            sellable, free, 0.0, 0.0, "FULLY_CLOSED", False, "FULLY_CLOSED", tuple(notes)
        )
    return ResidualBaseClassification(
        sellable,
        free,
        0.0,
        free,
        "UNPROTECTED_INVENTORY",
        True,
        pq.reason or "UNPROTECTED",
        tuple(notes),
    )


def is_economically_flat_base(
    *,
    free: float,
    locked: float,
    meta: SymbolInfo,
    ref_price: float,
) -> tuple[bool, str]:
    """True when base inventory cannot form a valid protective/exit SELL.

    Used by reconciliation so residual LOT_SIZE / minNotional dust after a filled
    OCO exit is treated as POSITION_CLOSED rather than PROTECTION_MISSING.
    Locked inventory always counts as non-flat (open working orders).
    """
    free_f = max(0.0, float(free or 0.0))
    locked_f = max(0.0, float(locked or 0.0))
    total = free_f + locked_f
    if total <= 1e-15:
        return True, "ZERO"
    if locked_f > 1e-15:
        return False, "LOCKED"
    step = float(getattr(meta, "quantity_step", 0) or 0)
    floored = floor_to_step(free_f, step) if step > 0 else free_f
    if floored <= 1e-15:
        return True, "LOT_SIZE_DUST"
    min_qty = float(getattr(meta, "min_quantity", 0) or 0)
    if min_qty > 0 and floored + 1e-15 < min_qty:
        return True, "DUST_BELOW_MIN_QTY"
    min_notional = float(getattr(meta, "min_notional", 0) or 0)
    px = float(ref_price or 0)
    if min_notional > 0 and px > 0 and floored * px + 1e-15 < min_notional:
        return True, "DUST_BELOW_MIN_NOTIONAL"
    return False, "SELLABLE"
