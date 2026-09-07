"""Production live-price invariants for bot-managed T1.

Hard rules:
  1. SequencePriceSource / test stubs are forbidden on armed REAL paths.
  2. A mark may never become T1 input unless it is a positive finite price for
     the exact position symbol and within an entry-relative sanity band.
  3. Invalid / stale / fabricated prices must not be converted into protection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from btcc.execution.price_monitor import MexcPublicTickerSource, PriceTick, SequencePriceSource

# Defense-in-depth vs wrong instrument / stub 0.05 on ~0.00x BTC pairs.
MARK_ENTRY_RATIO_MAX = 10.0
MARK_ENTRY_RATIO_MIN = 0.1
DEFAULT_LIVE_CONFIRM_TIMEOUT_S = 5.0
DEFAULT_MAX_TICK_AGE_S = 5.0


class ProductionPriceSourceError(RuntimeError):
    """Raised when a non-live / forbidden price source reaches production."""


@dataclass(frozen=True)
class MarkValidation:
    ok: bool
    reason: str
    ratio: float | None = None


def normalize_symbol(symbol: str) -> str:
    return str(symbol).upper().replace("/", "").replace("-", "").replace("_", "")


def is_forbidden_production_source(source: Any) -> bool:
    if source is None:
        return True
    if isinstance(source, SequencePriceSource):
        return True
    if getattr(source, "is_production_forbidden", False):
        return True
    return False


def is_live_mexc_ticker(source: Any) -> bool:
    if isinstance(source, MexcPublicTickerSource):
        return True
    return bool(getattr(source, "is_live_mexc_ticker", False))


def assert_production_price_source(source: Any, *, context: str = "") -> None:
    ctx = f" ({context})" if context else ""
    if is_forbidden_production_source(source):
        raise ProductionPriceSourceError(
            f"FORBIDDEN_PRICE_SOURCE{ctx}: {type(source).__name__} cannot protect real positions"
        )
    if not is_live_mexc_ticker(source):
        raise ProductionPriceSourceError(
            f"NON_LIVE_PRICE_SOURCE{ctx}: {type(source).__name__}; require MexcPublicTickerSource"
        )


def validate_tick_basic(tick: PriceTick, *, expected_symbol: str) -> MarkValidation:
    if tick is None:
        return MarkValidation(False, "TICK_NONE")
    exp = normalize_symbol(expected_symbol)
    got = normalize_symbol(tick.symbol or "")
    if not got:
        return MarkValidation(False, "TICK_SYMBOL_MISSING")
    if got != exp:
        return MarkValidation(False, f"TICK_SYMBOL_MISMATCH:expected={exp}:got={got}")
    px = float(tick.price)
    if not math.isfinite(px):
        return MarkValidation(False, "TICK_PRICE_NOT_FINITE")
    if px <= 0:
        return MarkValidation(False, "TICK_PRICE_NON_POSITIVE")
    return MarkValidation(True, "OK")


def validate_mark_vs_entry(mark: float, entry: float) -> MarkValidation:
    if not math.isfinite(mark) or mark <= 0:
        return MarkValidation(False, "MARK_NON_POSITIVE")
    if not math.isfinite(entry) or entry <= 0:
        return MarkValidation(False, "ENTRY_NON_POSITIVE")
    ratio = float(mark) / float(entry)
    if ratio > MARK_ENTRY_RATIO_MAX or ratio < MARK_ENTRY_RATIO_MIN:
        return MarkValidation(
            False,
            f"MARK_ABSURD_VS_ENTRY:mark={mark}:entry={entry}:ratio={ratio:.6g}",
            ratio=ratio,
        )
    return MarkValidation(True, "OK", ratio=ratio)


def validate_protection_mark(
    *,
    mark: float,
    entry: float,
    tick: PriceTick | None = None,
    expected_symbol: str | None = None,
) -> MarkValidation:
    if tick is not None and expected_symbol is not None:
        basic = validate_tick_basic(tick, expected_symbol=expected_symbol)
        if not basic.ok:
            return basic
    return validate_mark_vs_entry(mark, entry)
