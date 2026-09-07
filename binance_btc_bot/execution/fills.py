"""Fill aggregation — actual Binance execution qty/price/fees (no theoretical prices)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FillLeg:
    price: float
    qty: float
    commission: float = 0.0
    # Empty means Binance did not report an asset — NEVER assume BASE/BTC.
    commission_asset: str = ""
    trade_id: str | None = None


@dataclass(frozen=True)
class AggregatedFill:
    ok: bool
    status: str
    symbol: str
    side: str
    order_id: str | None
    client_order_id: str | None
    executed_qty: float
    avg_price: float
    cumulative_quote_qty: float
    commission_btc: float
    commission_usdt: float
    fills: tuple[FillLeg, ...] = ()
    reason: str = ""
    raw: dict[str, Any] = field(default_factory=dict)
    # Optional rollup of all commission assets (BNB, BASE, quote, …).
    commissions_by_asset: dict[str, float] = field(default_factory=dict)

    @property
    def fully_filled(self) -> bool:
        return self.status.upper() in {"FILLED"}

    @property
    def partially_filled(self) -> bool:
        return self.status.upper() in {"PARTIALLY_FILLED"} and self.executed_qty > 0

    def sellable_base_qty(self, base_asset: str) -> float:
        from binance_btc_bot.execution.protection_qty import sellable_qty_from_fills

        return sellable_qty_from_fills(self, base_asset)


def weighted_average_fill(fills: list[FillLeg] | tuple[FillLeg, ...]) -> tuple[float, float, float]:
    """Return (avg_price, total_qty, quote_spent)."""
    total_qty = 0.0
    quote = 0.0
    for f in fills:
        q = float(f.qty)
        p = float(f.price)
        if q <= 0 or p <= 0:
            continue
        total_qty += q
        quote += q * p
    if total_qty <= 0:
        return 0.0, 0.0, 0.0
    return quote / total_qty, total_qty, quote


def aggregate_fills_from_order(
    *,
    symbol: str,
    side: str,
    order_id: str | None,
    client_order_id: str | None,
    status: str,
    executed_qty: float | None,
    cumulative_quote_qty: float | None,
    fills_raw: list[dict[str, Any]] | None,
    raw: dict[str, Any] | None = None,
) -> AggregatedFill:
    """Build AggregatedFill from Binance order response / myTrades-like fills."""
    legs: list[FillLeg] = []
    commission_btc = 0.0
    commission_usdt = 0.0
    by_asset: dict[str, float] = {}
    for row in fills_raw or []:
        px = float(row.get("price") or row.get("p") or 0)
        qty = float(row.get("qty") or row.get("q") or row.get("quantity") or 0)
        if px <= 0 or qty <= 0:
            continue
        commission = float(row.get("commission") or row.get("c") or 0)
        # Do not default commissionAsset — empty means unknown (not assumed BASE/BTC).
        raw_asset = row.get("commissionAsset")
        if raw_asset is None:
            raw_asset = row.get("commission_asset")
        asset = str(raw_asset).upper() if raw_asset not in (None, "") else ""
        legs.append(
            FillLeg(
                price=px,
                qty=qty,
                commission=commission,
                commission_asset=asset,
                trade_id=str(row.get("id") or row.get("tradeId") or "") or None,
            )
        )
        key = asset or "UNKNOWN"
        by_asset[key] = by_asset.get(key, 0.0) + commission
        if asset == "BTC":
            commission_btc += commission
        elif asset in {"USDT", "USDC", "BUSD", "FDUSD"}:
            commission_usdt += commission

    if legs:
        avg, qty, quote = weighted_average_fill(legs)
    else:
        qty = float(executed_qty or 0)
        quote = float(cumulative_quote_qty or 0)
        avg = (quote / qty) if qty > 0 and quote > 0 else 0.0
        if qty > 0 and avg <= 0 and raw:
            # Last resort: single price field (still prefer avg from quote/qty).
            avg = float(raw.get("price") or 0) or 0.0
            if avg > 0 and quote <= 0:
                quote = avg * qty

    st = str(status or "").upper()
    ok = st in {"FILLED", "PARTIALLY_FILLED"} and qty > 0
    reason = "OK" if ok else (st or "NO_FILL")
    return AggregatedFill(
        ok=ok,
        status=st or "UNKNOWN",
        symbol=symbol.upper(),
        side=side.upper(),
        order_id=order_id,
        client_order_id=client_order_id,
        executed_qty=qty,
        avg_price=avg,
        cumulative_quote_qty=quote,
        commission_btc=commission_btc,
        commission_usdt=commission_usdt,
        fills=tuple(legs),
        reason=reason,
        raw=dict(raw or {}),
        commissions_by_asset=by_asset,
    )
