"""Helpers to construct emergency/OCO request payloads without submitting."""

from __future__ import annotations

from typing import Any

from binance_btc_bot.exchange.base import OrderRequest, SymbolInfo, TrailingOcoRequest
from binance_btc_bot.exchange.binance import _fmt_decimal
from binance_btc_bot.strategy.trails import TrailStrategy, map_trail_to_binance_oco


def build_t1_oco_request_params(
    *,
    strategy: TrailStrategy,
    symbol: str,
    entry_price: float,
    quantity: float,
    symbol_info: SymbolInfo,
    list_client_order_id: str,
) -> dict[str, Any]:
    """Exact signed params that ``place_trailing_exit`` would send (no HTTP)."""
    mapping = map_trail_to_binance_oco(
        strategy=strategy,
        symbol=symbol,
        entry_price=entry_price,
        quantity=quantity,
        symbol_info=symbol_info,
        list_client_order_id=list_client_order_id,
    )
    if not mapping.allowed or mapping.request is None:
        return {
            "ok": False,
            "endpoint": "POST /api/v3/orderList/oco",
            "reason": ";".join(mapping.constraint_notes),
            "notes": list(mapping.constraint_notes),
        }
    req: TrailingOcoRequest = mapping.request
    params: dict[str, Any] = {
        "symbol": req.symbol.upper(),
        "side": req.side.upper(),
        "quantity": _fmt_decimal(req.quantity),
        "aboveType": req.above_type.upper(),
        "aboveStopPrice": _fmt_decimal(req.above_stop_price),
        "aboveTrailingDelta": int(req.above_trailing_delta),
        "belowType": req.below_type.upper(),
        "belowStopPrice": _fmt_decimal(req.below_stop_price),
        "newOrderRespType": req.new_order_resp_type,
    }
    if req.list_client_order_id:
        params["listClientOrderId"] = req.list_client_order_id
    return {
        "ok": True,
        "endpoint": "POST /api/v3/orderList/oco",
        "method": "POST",
        "params": params,
        "mapping": {
            "activation_price": mapping.activation_price,
            "initial_stop_price": mapping.initial_stop_price,
            "trailing_delta_bips": mapping.trailing_delta_bips,
            "notes": list(mapping.constraint_notes),
        },
        "filters": {
            "lot_step": symbol_info.quantity_step,
            "min_qty": symbol_info.min_quantity,
            "min_notional": symbol_info.min_notional,
            "price_tick": symbol_info.price_tick,
            "oco_allowed": symbol_info.oco_allowed,
            "order_types": list(symbol_info.order_types),
        },
    }


def build_emergency_stop_request(
    *,
    symbol: str,
    quantity: float,
    entry_price: float,
    strategy: TrailStrategy,
    symbol_info: SymbolInfo,
    client_order_id: str,
) -> dict[str, Any]:
    """Exact STOP_LOSS SELL params for emergency fallback (no HTTP)."""
    from binance_btc_bot.strategy.trails import _floor_to_tick

    stop_px = float(entry_price) * (1.0 - float(strategy.arm_sl_activation_trail))
    if symbol_info.price_tick > 0:
        stop_px = _floor_to_tick(stop_px, symbol_info.price_tick)
    req = OrderRequest(
        symbol=symbol,
        side="SELL",
        order_type="STOP_LOSS",
        quantity=quantity,
        stop_price=stop_px,
        client_order_id=client_order_id,
    )
    params = {
        "symbol": req.symbol.upper(),
        "side": req.side.upper(),
        "type": req.order_type.upper(),
        "quantity": _fmt_decimal(req.quantity),
        "stopPrice": _fmt_decimal(req.stop_price) if req.stop_price is not None else None,
        "newClientOrderId": req.client_order_id,
    }
    notion = float(quantity) * float(entry_price)
    filter_ok = True
    reasons: list[str] = []
    if symbol_info.min_quantity > 0 and quantity + 1e-15 < symbol_info.min_quantity:
        filter_ok = False
        reasons.append("MIN_QTY")
    if symbol_info.min_notional > 0 and notion + 1e-15 < symbol_info.min_notional:
        filter_ok = False
        reasons.append("MIN_NOTIONAL")
    if "STOP_LOSS" not in {str(x).upper() for x in symbol_info.order_types}:
        filter_ok = False
        reasons.append("STOP_LOSS_NOT_IN_ORDER_TYPES")
    return {
        "ok": filter_ok,
        "endpoint": "POST /api/v3/order",
        "method": "POST",
        "adapter": "place_protective_sell",
        "params": params,
        "request": req,
        "reasons": reasons,
        "filters": {
            "lot_step": symbol_info.quantity_step,
            "min_qty": symbol_info.min_quantity,
            "min_notional": symbol_info.min_notional,
            "price_tick": symbol_info.price_tick,
            "notional_at_entry": notion,
        },
    }
