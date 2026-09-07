"""Timestamped REAL_SHADOW account snapshots — no secrets, no paper mutation."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from btcc.execution.mexc.models import AssetBalance, ExchangeFill, ExchangeOrder
from btcc.execution.symbols import SymbolMeta


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class BalanceRow:
    asset: str
    free: str
    locked: str
    total: str

    @classmethod
    def from_balance(cls, b: AssetBalance) -> BalanceRow:
        return cls(asset=b.asset, free=str(b.free), locked=str(b.locked), total=str(b.total))


@dataclass
class ShadowAccountSnapshot:
    """Exchange truth observation for REAL_SHADOW (read-only)."""

    snapshot_id: str
    mode: str  # REAL_SHADOW
    local_ts: str
    exchange_server_time_ms: int | None
    clock_skew_ms: int | None
    age_ms: int | None
    fresh: bool
    balances: list[BalanceRow] = field(default_factory=list)
    open_orders: list[dict[str, Any]] = field(default_factory=list)
    fills: list[dict[str, Any]] = field(default_factory=list)
    symbols: list[dict[str, Any]] = field(default_factory=list)
    checks_performed: list[str] = field(default_factory=list)
    empty_account: bool = False
    successfully_fetched: bool = False
    notes: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def persist(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, path)
        return path


def build_snapshot(
    *,
    balances: list[AssetBalance],
    open_orders: list[ExchangeOrder],
    fills: list[ExchangeFill],
    symbols: list[SymbolMeta],
    exchange_server_time_ms: int | None,
    local_ts: str | None = None,
    max_skew_ms: int = 5000,
    max_age_ms: int = 30_000,
    dust: Decimal = Decimal("0"),
) -> ShadowAccountSnapshot:
    local = local_ts or _now_iso()
    local_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    skew = None
    age = None
    fresh = True
    checks: list[str] = [
        "BALANCES_FETCHED",
        "OPEN_ORDERS_FETCHED",
        "FILLS_FETCHED",
        "SYMBOLS_FETCHED",
    ]
    notes: list[str] = []

    if exchange_server_time_ms is not None:
        checks.append("SERVER_TIME_FETCHED")
        skew = local_ms - int(exchange_server_time_ms)
        age = abs(skew)
        if abs(skew) > max_skew_ms:
            fresh = False
            notes.append(f"clock_skew_ms={skew} exceeds max_skew_ms={max_skew_ms}")
            checks.append("CLOCK_SKEW_EXCEEDED")
        else:
            checks.append("CLOCK_SKEW_OK")
        if age > max_age_ms:
            fresh = False
            notes.append(f"data_age_ms={age} exceeds max_age_ms={max_age_ms}")
            checks.append("DATA_STALE")
        else:
            checks.append("DATA_FRESH")
    else:
        fresh = False
        notes.append("exchange_server_time unavailable")
        checks.append("SERVER_TIME_MISSING")

    non_dust = [b for b in balances if b.total > dust]
    empty = len(non_dust) == 0 and len(open_orders) == 0
    if empty:
        checks.append("EMPTY_ACCOUNT_OBSERVED")
        notes.append(
            "Empty account is NOT the same as successfully reconciled readiness for trading"
        )
    else:
        checks.append("NON_EMPTY_ACCOUNT_OBSERVED")

    # Explicit: never equate successful fetch with trade readiness
    checks.append("TRADE_READINESS_NOT_IMPLIED")
    notes.append("REAL production execution remains BLOCKED (no write path; protective-stop unresolved)")

    snap_id = f"shadow_{local_ms}"
    return ShadowAccountSnapshot(
        snapshot_id=snap_id,
        mode="REAL_SHADOW",
        local_ts=local,
        exchange_server_time_ms=exchange_server_time_ms,
        clock_skew_ms=skew,
        age_ms=age,
        fresh=fresh,
        balances=[BalanceRow.from_balance(b) for b in balances],
        open_orders=[
            {
                "order_id": o.order_id,
                "client_order_id": o.client_order_id,
                "symbol": o.symbol,
                "side": o.side,
                "type": o.type,
                "status": o.status,
                "original_quantity": str(o.original_quantity),
                "executed_quantity": str(o.executed_quantity),
                "remaining_quantity": str(o.remaining_quantity),
            }
            for o in open_orders
        ],
        fills=[
            {
                "trade_id": f.trade_id,
                "order_id": f.order_id,
                "symbol": f.symbol,
                "quantity": str(f.quantity),
                "price": str(f.price),
                "fee": str(f.fee) if f.fee is not None else None,
                "fee_asset": f.fee_asset,
                "timestamp": f.timestamp,
            }
            for f in fills
        ],
        symbols=[
            {
                "symbol": s.symbol,
                "status": s.status,
                "base_asset": s.base_asset,
                "quote_asset": s.quote_asset,
                "quantity_step": s.quantity_step,
                "min_quantity": s.min_quantity,
                "price_tick": s.price_tick,
                "min_notional": s.min_notional,
                "max_quantity": s.max_quantity,
                "source": s.source,
                "blocked": not s.is_trading or s.quantity_step <= 0 or s.price_tick <= 0,
            }
            for s in symbols
        ],
        checks_performed=checks,
        empty_account=empty,
        successfully_fetched=True,
        notes=notes,
        meta={"non_dust_assets": len(non_dust), "open_order_count": len(open_orders)},
    )
