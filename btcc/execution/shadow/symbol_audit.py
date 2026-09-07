"""Symbol metadata audit for REAL_SHADOW — no invented values."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from btcc.execution.mexc.client import MexcReadOnlyClient
from btcc.execution.mexc.errors import MexcReadError, SymbolMetadataUnavailable
from btcc.execution.symbols import SymbolMeta


@dataclass
class SymbolAuditRow:
    symbol: str
    status: str | None = None
    base_asset: str | None = None
    quote_asset: str | None = None
    quantity_step: float | None = None
    min_quantity: float | None = None
    price_tick: float | None = None
    min_notional: float | None = None
    max_quantity: float | None = None
    blocked: bool = True
    block_reason: str | None = "NOT_FETCHED"
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "status": self.status,
            "base_asset": self.base_asset,
            "quote_asset": self.quote_asset,
            "quantity_step": self.quantity_step,
            "min_quantity": self.min_quantity,
            "price_tick": self.price_tick,
            "min_notional": self.min_notional,
            "max_quantity": self.max_quantity,
            "blocked": self.blocked,
            "block_reason": self.block_reason,
            "source": self.source,
        }


@dataclass
class SymbolAuditReport:
    rows: list[SymbolAuditRow] = field(default_factory=list)
    any_blocked: bool = True
    real_execution_blocked: bool = True  # always true in Stage 5A + until protection exists

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": [r.to_dict() for r in self.rows],
            "any_blocked": self.any_blocked,
            "real_execution_blocked": self.real_execution_blocked,
            "note": "Missing/invalid metadata → BLOCKED; Stage 5A never enables trading",
        }


def audit_symbols(client: MexcReadOnlyClient, symbols: list[str]) -> SymbolAuditReport:
    rows: list[SymbolAuditRow] = []
    any_blocked = False
    for sym in symbols:
        s = str(sym).upper().strip()
        if not s:
            continue
        try:
            meta: SymbolMeta = client.get_symbol_metadata(s, use_cache=False)
            blocked = False
            reason = None
            if not meta.is_trading:
                blocked = True
                reason = "NOT_TRADING"
            elif meta.quantity_step <= 0 or meta.price_tick <= 0:
                blocked = True
                reason = "INVALID_PRECISION"
            elif meta.min_quantity <= 0:
                blocked = True
                reason = "INVALID_MIN_QUANTITY"
            row = SymbolAuditRow(
                symbol=meta.symbol,
                status=meta.status,
                base_asset=meta.base_asset,
                quote_asset=meta.quote_asset,
                quantity_step=meta.quantity_step,
                min_quantity=meta.min_quantity,
                price_tick=meta.price_tick,
                min_notional=meta.min_notional,
                max_quantity=meta.max_quantity,
                blocked=blocked,
                block_reason=reason,
                source=meta.source,
            )
        except SymbolMetadataUnavailable as e:
            any_blocked = True
            row = SymbolAuditRow(symbol=s, blocked=True, block_reason=f"METADATA_UNAVAILABLE:{e}")
        except MexcReadError as e:
            any_blocked = True
            row = SymbolAuditRow(symbol=s, blocked=True, block_reason=f"READ_ERROR:{type(e).__name__}")
        if row.blocked:
            any_blocked = True
        rows.append(row)

    return SymbolAuditReport(
        rows=rows,
        any_blocked=any_blocked or not rows,
        real_execution_blocked=True,
    )
