"""Native Binance trailing exit submission + monitoring (no local trail engine)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from binance_btc_bot.exchange.base import ExchangeAdapter, OrderResult
from binance_btc_bot.notifications.manager import NotificationManager
from binance_btc_bot.secrets import scrub_exception
from binance_btc_bot.storage.database import BotDatabase
from binance_btc_bot.strategy.trails import TrailStrategy, map_trail_to_binance_oco

logger = logging.getLogger(__name__)


@dataclass
class TrailingSubmitResult:
    ok: bool
    mapping_notes: tuple[str, ...]
    order: OrderResult | None
    reason: str
    activation_price: float | None = None
    trail_bips: int | None = None
    initial_stop: float | None = None


class TrailingExecutor:
    """Submit Binance-native OCO trailing exit; monitor fills via exchange queries."""

    def __init__(
        self,
        exchange: ExchangeAdapter,
        db: BotDatabase,
        *,
        notifications: NotificationManager | None = None,
    ) -> None:
        self.exchange = exchange
        self.db = db
        self.notifications = notifications

    def _notify(self, method: str, event: str, message: str, **kwargs: Any) -> None:
        if not self.notifications:
            return
        try:
            getattr(self.notifications, method)(event, message, **kwargs)
        except Exception:  # noqa: BLE001
            logger.warning("trailing notification failed (ignored)")
    def submit_native_trailing(
        self,
        *,
        trade_id: str,
        symbol: str,
        strategy: TrailStrategy,
        entry_price: float,
        quantity: float,
        list_client_order_id: str | None = None,
        prefer_market_contingent: bool = True,
    ) -> TrailingSubmitResult:
        info = self.exchange.get_symbol_info(symbol)
        mapping = map_trail_to_binance_oco(
            strategy=strategy,
            symbol=symbol,
            entry_price=entry_price,
            quantity=quantity,
            symbol_info=info,
            prefer_market_contingent=prefer_market_contingent,
            list_client_order_id=list_client_order_id or f"t_{trade_id[:20]}",
        )
        logger.info(
            "TRAILING ORDER symbol=%s activation=%s trail_distance=%s BIPS=%s mode=BINANCE_NATIVE notes=%s",
            symbol,
            mapping.activation_price,
            strategy.trail_distance,
            mapping.trailing_delta_bips,
            mapping.constraint_notes,
        )
        if not mapping.allowed or mapping.request is None:
            reason = ";".join(mapping.constraint_notes) or "MAPPING_REJECTED"
            self.db.insert_event(
                "ERROR",
                symbol=symbol,
                trade_id=trade_id,
                reason=reason,
                payload={"mapping": mapping.__dict__},
            )
            self._notify(
                "notify_critical",
                "OCO_FAILURE",
                f"OCO mapping rejected for {symbol}: {reason}",
                symbol=symbol,
                trade_id=trade_id,
                reason=reason,
            )
            return TrailingSubmitResult(False, mapping.constraint_notes, None, reason)

        try:
            order = self.exchange.place_trailing_exit(mapping.request)
        except Exception as e:  # noqa: BLE001
            err = scrub_exception(e)
            self.db.insert_event("ERROR", symbol=symbol, trade_id=trade_id, reason=err)
            self._notify(
                "notify_critical",
                "OCO_FAILURE",
                f"OCO submit failed for {symbol}: {err}",
                symbol=symbol,
                trade_id=trade_id,
                reason=err,
            )
            return TrailingSubmitResult(False, mapping.constraint_notes, None, f"API_FAILURE:{err}")

        self.db.insert_order(
            trade_id=trade_id,
            symbol=symbol,
            order_id=order.order_id,
            client_order_id=order.client_order_id,
            order_list_id=order.order_id,
            side="SELL",
            order_type="OCO_TRAILING",
            status=order.status,
            payload={
                "raw": order.raw,
                "activation": mapping.activation_price,
                "trailing_delta_bips": mapping.trailing_delta_bips,
                "initial_stop": mapping.initial_stop_price,
                "mode": "BINANCE_NATIVE",
            },
        )
        self.db.update_trade(trade_id, binance_oco_list_id=order.order_id)
        event = "TRAILING_ORDER_SUBMITTED" if not order.dry_run else "TRAILING_ORDER_SUBMITTED"
        self.db.insert_event(
            event,
            symbol=symbol,
            trade_id=trade_id,
            order_id=order.order_id,
            reason="DRY_RUN" if order.dry_run else "OK",
            payload={"mode": "BINANCE_NATIVE", "params": (order.raw or {}).get("params")},
        )
        if order.dry_run:
            logger.info("DRY RUN — ORDER NOT SUBMITTED trailing trade_id=%s", trade_id)
        self._notify(
            "notify_info",
            "TRAILING_OCO_SUBMITTED",
            f"Native OCO trailing submitted {symbol} delta={mapping.trailing_delta_bips}BIPS dry_run={order.dry_run}",
            symbol=symbol,
            trade_id=trade_id,
            order_id=order.order_id,
        )
        return TrailingSubmitResult(
            True,
            mapping.constraint_notes,
            order,
            "OK",
            activation_price=mapping.activation_price,
            trail_bips=mapping.trailing_delta_bips,
            initial_stop=mapping.initial_stop_price,
        )

    def poll_oco_status(self, symbol: str, order_list_id: str | None) -> dict[str, Any]:
        """Query Binance for open order lists; detect fill/cancel. No local trail math.

        Uses client-side symbol filter (openOrderList has no symbol param).
        API / ambiguous failures fail closed — no silent empty success.
        """
        from binance_btc_bot.execution.order_lists import find_matching_open_list, is_binance_param_error

        try:
            open_lists = self.exchange.get_open_order_lists(symbol)
        except Exception as e:  # noqa: BLE001
            err = scrub_exception(e)
            if is_binance_param_error(err):
                # Authoritative retry: unfiltered open lists only.
                try:
                    open_lists = self.exchange.get_open_order_lists()
                except Exception as e2:  # noqa: BLE001
                    return {
                        "still_open": False,
                        "matched": None,
                        "open_count": 0,
                        "ok": False,
                        "error": scrub_exception(e2),
                        "fail_closed": True,
                    }
            else:
                return {
                    "still_open": False,
                    "matched": None,
                    "open_count": 0,
                    "ok": False,
                    "error": err,
                    "fail_closed": True,
                }
        matched_view = find_matching_open_list(
            open_lists, symbol=symbol, order_list_id=order_list_id
        )
        return {
            "still_open": matched_view is not None and matched_view.is_open,
            "matched": matched_view.raw if matched_view else None,
            "open_count": len(open_lists),
            "ok": True,
            "fail_closed": False,
        }
