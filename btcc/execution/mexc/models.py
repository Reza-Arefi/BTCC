"""Exchange-neutral read models mapped from MEXC Spot responses."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any


def _dec(value: Any, *, field_name: str) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as e:
        raise ValueError(f"invalid decimal for {field_name}: {value!r}") from e


@dataclass(frozen=True)
class AssetBalance:
    asset: str
    free: Decimal
    locked: Decimal

    @property
    def total(self) -> Decimal:
        return self.free + self.locked

    @classmethod
    def from_mexc(cls, raw: dict[str, Any]) -> AssetBalance:
        if not isinstance(raw, dict) or "asset" not in raw or "free" not in raw or "locked" not in raw:
            raise ValueError("malformed balance row")
        return cls(
            asset=str(raw["asset"]),
            free=_dec(raw["free"], field_name="free"),
            locked=_dec(raw["locked"], field_name="locked"),
        )


@dataclass(frozen=True)
class ExchangeOrder:
    order_id: str
    client_order_id: str | None
    symbol: str
    side: str
    type: str
    status: str
    original_quantity: Decimal
    executed_quantity: Decimal
    remaining_quantity: Decimal
    price: Decimal | None
    average_price: Decimal | None
    stop_price: Decimal | None
    time: int | None
    update_time: int | None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_mexc(cls, raw: dict[str, Any]) -> ExchangeOrder:
        if not isinstance(raw, dict):
            raise ValueError("malformed order")
        for req in ("symbol", "orderId", "status", "side", "type"):
            if req not in raw:
                raise ValueError(f"order missing {req}")
        orig = raw.get("origQty", raw.get("Qty", raw.get("origQuantity")))
        executed = raw.get("executedQty", "0")
        if orig is None:
            raise ValueError("order missing origQty")
        orig_d = _dec(orig, field_name="origQty")
        exe_d = _dec(executed, field_name="executedQty")
        remaining = orig_d - exe_d
        if remaining < 0:
            remaining = Decimal("0")
        avg = raw.get("avgPrice") or raw.get("averagePrice")
        stop = raw.get("stopPrice")
        avg_d = _dec(avg, field_name="avgPrice") if avg not in (None, "", "0", "0.0") else None
        if (avg_d is None or avg_d <= 0) and exe_d > 0:
            cq = raw.get("cummulativeQuoteQty") or raw.get("cumulativeQuoteQty")
            if cq not in (None, "", "0", "0.0"):
                try:
                    avg_d = _dec(cq, field_name="cummulativeQuoteQty") / exe_d
                except Exception:
                    avg_d = None
        return cls(
            order_id=str(raw["orderId"]),
            client_order_id=(str(raw["clientOrderId"]) if raw.get("clientOrderId") not in (None, "") else None),
            symbol=str(raw["symbol"]),
            side=str(raw["side"]),
            type=str(raw["type"]),
            status=str(raw["status"]),
            original_quantity=orig_d,
            executed_quantity=exe_d,
            remaining_quantity=remaining,
            price=_dec(raw["price"], field_name="price") if raw.get("price") not in (None, "") else None,
            average_price=avg_d,
            stop_price=_dec(stop, field_name="stopPrice") if stop not in (None, "") else None,
            time=int(raw["time"]) if raw.get("time") is not None else None,
            update_time=int(raw["updateTime"]) if raw.get("updateTime") is not None else None,
            raw=dict(raw),
        )


@dataclass(frozen=True)
class ExchangeFill:
    trade_id: str
    order_id: str
    symbol: str
    quantity: Decimal
    price: Decimal
    fee: Decimal | None
    fee_asset: str | None
    timestamp: int | None
    is_buyer: bool | None = None
    client_order_id: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_mexc(cls, raw: dict[str, Any]) -> ExchangeFill:
        if not isinstance(raw, dict):
            raise ValueError("malformed fill")
        for req in ("symbol", "id", "orderId", "price", "qty"):
            if req not in raw:
                raise ValueError(f"fill missing {req}")
        return cls(
            trade_id=str(raw["id"]),
            order_id=str(raw["orderId"]),
            symbol=str(raw["symbol"]),
            quantity=_dec(raw["qty"], field_name="qty"),
            price=_dec(raw["price"], field_name="price"),
            fee=_dec(raw["commission"], field_name="commission") if raw.get("commission") is not None else None,
            fee_asset=str(raw["commissionAsset"]) if raw.get("commissionAsset") is not None else None,
            timestamp=int(raw["time"]) if raw.get("time") is not None else None,
            is_buyer=bool(raw["isBuyer"]) if "isBuyer" in raw else None,
            client_order_id=str(raw["clientOrderId"]) if raw.get("clientOrderId") not in (None, "") else None,
            raw=dict(raw),
        )
