"""Exchange-neutral symbol metadata + quantity floor rounding / wire serialization.

CRITICAL: rounding must NEVER increase intended order size.
Wire serialization must respect base-size precision (e.g. precision 0 → "1343", not "1343.0").
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN, InvalidOperation
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class SymbolMeta:
    symbol: str
    status: str  # e.g. TRADING / BREAK / UNKNOWN
    base_asset: str
    quote_asset: str
    quantity_step: float
    min_quantity: float
    price_tick: float
    min_notional: float
    max_quantity: float | None = None
    source: str = "unknown"  # paper_stub | exchange | unknown
    # From exchangeInfo.orderTypes when known (e.g. LIMIT, MARKET, LIMIT_MAKER).
    # None = unknown (unit-test stubs); production MEXC map always sets a tuple.
    order_types: tuple[str, ...] | None = None

    @property
    def is_trading(self) -> bool:
        return str(self.status).upper() == "TRADING"

    @property
    def quantity_decimals(self) -> int:
        """Allowed decimal places for base quantity, derived from quantity_step."""
        return quantity_decimal_places(self.quantity_step)

    @property
    def supports_market_orders(self) -> bool | None:
        """True/False when order_types known; None when unknown."""
        if self.order_types is None:
            return None
        return any(str(t).upper() == "MARKET" for t in self.order_types)


def market_orders_supported(meta: SymbolMeta) -> bool | None:
    return meta.supports_market_orders


def t1_market_exit_capability(meta: SymbolMeta) -> tuple[bool, str]:
    """Production T1 requires MARKET for immediate bot-managed exits.

    Policy: no verified MARKET capability → no entry / no MARKET POST.
    Returns (ok, reason_code).
    """
    if meta is None:
        return False, "SYMBOL_META_MISSING"
    if not meta.is_trading:
        return False, "SYMBOL_NOT_TRADING"
    supported = meta.supports_market_orders
    if supported is False:
        types = ",".join(meta.order_types or ())
        return False, f"MARKET_ORDER_TYPE_UNSUPPORTED:orderTypes=[{types}]"
    if supported is None and str(meta.source).startswith("mexc"):
        # Exchange-sourced meta without orderTypes is fail-closed for REAL T1.
        return False, "MARKET_ORDER_TYPES_UNKNOWN"
    # Test stubs (order_types=None, non-mexc source) remain allowed.
    return True, "OK"


@dataclass(frozen=True)
class NormalizedQuantity:
    """Result of floor + exchange wire formatting for a base quantity."""

    ok: bool
    raw_quantity: float = 0.0
    quantity: float = 0.0
    serialized: str = ""
    reason: str = ""
    decimals: int = 0


@runtime_checkable
class SymbolMetaProvider(Protocol):
    def get_symbol(self, symbol: str) -> SymbolMeta | None: ...


class UnknownSymbolMetaProvider:
    """Fail-closed for REAL: unknown metadata returns None (caller must reject)."""

    def get_symbol(self, symbol: str) -> SymbolMeta | None:
        return None


class StaticSymbolMetaProvider:
    """Test/paper helper with explicit metadata (never invents exchange values)."""

    def __init__(self, rows: dict[str, SymbolMeta] | None = None) -> None:
        self._rows = {k.upper(): v for k, v in (rows or {}).items()}

    def get_symbol(self, symbol: str) -> SymbolMeta | None:
        return self._rows.get(str(symbol).upper())


def quantity_decimal_places(step: float) -> int:
    """Decimal places implied by a positive lot step (1 → 0, 0.001 → 3)."""
    if step <= 0:
        raise ValueError("step must be > 0")
    s = Decimal(str(step)).normalize()
    exp = int(s.as_tuple().exponent)
    return 0 if exp >= 0 else -exp


def floor_to_step(quantity: float, step: float) -> float:
    """Floor quantity to step size. Never rounds up."""
    if quantity < 0:
        raise ValueError("quantity must be >= 0")
    if step <= 0:
        raise ValueError("step must be > 0")
    q = Decimal(str(quantity))
    s = Decimal(str(step))
    try:
        units = (q / s).to_integral_value(rounding=ROUND_DOWN)
    except InvalidOperation as e:
        raise ValueError(f"invalid quantity/step: {quantity}/{step}") from e
    return float(units * s)


def floor_quantity_for_symbol(quantity: float, meta: SymbolMeta) -> float:
    """Apply lot-size floor. Result never exceeds requested quantity."""
    floored = floor_to_step(quantity, meta.quantity_step)
    if floored > quantity + 1e-15:
        raise RuntimeError("floor_quantity_for_symbol increased size — invariant broken")
    if meta.max_quantity is not None:
        floored = min(floored, float(meta.max_quantity))
        floored = floor_to_step(floored, meta.quantity_step)
    return floored


def format_quantity_for_exchange(quantity: float, meta: SymbolMeta) -> str:
    """Serialize a *already floored* base quantity for exchange query params.

    - precision 0 → integer string (\"1343\"), never \"1343.0\"
    - precision N → fixed-point with at most N decimals, no scientific notation
    """
    if meta.quantity_step <= 0:
        raise ValueError("quantity_step must be > 0")
    decimals = quantity_decimal_places(meta.quantity_step)
    q = Decimal(str(quantity))
    step = Decimal(str(meta.quantity_step))
    # Re-floor in Decimal space so serialization cannot invent extra scale.
    units = (q / step).to_integral_value(rounding=ROUND_DOWN)
    q = units * step
    if decimals == 0:
        as_int = int(q.to_integral_value(rounding=ROUND_DOWN))
        if Decimal(as_int) > Decimal(str(quantity)) + Decimal("1e-15"):
            raise RuntimeError("format_quantity_for_exchange increased size")
        out = str(as_int)
    else:
        quant = Decimal(1).scaleb(-decimals)  # 10**-decimals
        q = q.quantize(quant, rounding=ROUND_DOWN)
        out = format(q, "f")
        if "." in out:
            out = out.rstrip("0").rstrip(".")
        if out in {"", "-0"}:
            out = "0"
    if "e" in out.lower() or "E" in out:
        raise RuntimeError(f"scientific notation forbidden in quantity wire form: {out!r}")
    if decimals == 0 and ("." in out or "e" in out.lower()):
        raise RuntimeError(f"integer-lot quantity must not contain a decimal point: {out!r}")
    # Scale check: decimal places in wire form must not exceed allowed.
    if "." in out:
        frac = out.split(".", 1)[1]
        if len(frac) > decimals:
            raise RuntimeError(f"serialized quantity exceeds precision: {out!r} decimals={decimals}")
    return out


def normalize_order_quantity(
    quantity: float,
    meta: SymbolMeta,
    *,
    price_alt_btc: float | None = None,
) -> NormalizedQuantity:
    """Floor to symbol step, validate mins, and produce exchange wire string.

    Never rounds up. Returns ok=False (does not raise) for zero / below-min results
    so callers can HALT before POST.
    """
    raw = float(quantity)
    decimals = 0
    try:
        if raw < 0:
            return NormalizedQuantity(False, raw_quantity=raw, reason="NEGATIVE_QUANTITY")
        if meta.quantity_step <= 0:
            return NormalizedQuantity(False, raw_quantity=raw, reason="INCOMPLETE_SYMBOL_METADATA")
        decimals = quantity_decimal_places(meta.quantity_step)
        floored = floor_quantity_for_symbol(raw, meta)
        if floored > raw + 1e-15:
            return NormalizedQuantity(
                False, raw_quantity=raw, reason="FLOOR_INCREASED_SIZE", decimals=decimals
            )
        if floored <= 0:
            return NormalizedQuantity(
                False,
                raw_quantity=raw,
                quantity=0.0,
                reason="QTY_BELOW_STEP_AFTER_FLOOR",
                decimals=decimals,
            )
        if floored + 1e-15 < float(meta.min_quantity):
            return NormalizedQuantity(
                False,
                raw_quantity=raw,
                quantity=floored,
                reason="BELOW_MIN_QTY",
                decimals=decimals,
            )
        if price_alt_btc is not None:
            if float(price_alt_btc) <= 0:
                return NormalizedQuantity(
                    False, raw_quantity=raw, quantity=floored, reason="INVALID_PRICE", decimals=decimals
                )
            notional = floored * float(price_alt_btc)
            if float(meta.min_notional) > 0 and notional + 1e-15 < float(meta.min_notional):
                return NormalizedQuantity(
                    False,
                    raw_quantity=raw,
                    quantity=floored,
                    reason="BELOW_MIN_NOTIONAL",
                    decimals=decimals,
                )
        serialized = format_quantity_for_exchange(floored, meta)
        # Round-trip: parsed wire value must not exceed floored/raw.
        wire_val = float(Decimal(serialized))
        if wire_val > floored + 1e-15 or wire_val > raw + 1e-15:
            return NormalizedQuantity(
                False, raw_quantity=raw, quantity=floored, reason="SERIALIZE_INCREASED_SIZE", decimals=decimals
            )
        if wire_val <= 0:
            return NormalizedQuantity(
                False, raw_quantity=raw, quantity=floored, reason="SERIALIZE_ZERO", decimals=decimals
            )
        return NormalizedQuantity(
            True,
            raw_quantity=raw,
            quantity=floored,
            serialized=serialized,
            reason="OK",
            decimals=decimals,
        )
    except (ValueError, RuntimeError, InvalidOperation) as e:
        return NormalizedQuantity(
            False, raw_quantity=raw, reason=f"NORMALIZE_FAILED:{e}", decimals=decimals
        )
