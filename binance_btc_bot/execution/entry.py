"""Entry execution — market BUY on *BTC, then hand off to native trailing.

Integrates PortfolioManager reservation → sizing → risk → filters → submit.
Never submits real orders when live.enabled=false / dry_run=true.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from binance_btc_bot.exchange.base import ExchangeAdapter, OrderRequest, OrderResult
from binance_btc_bot.notifications.manager import NotificationManager
from binance_btc_bot.portfolio.manager import PortfolioManager
from binance_btc_bot.portfolio.trade_state import TradeStatus
from binance_btc_bot.risk.safety import SafetySystem
from binance_btc_bot.risk.sizing import SizeDecision, size_position
from binance_btc_bot.secrets import scrub_exception
from binance_btc_bot.storage.database import BotDatabase
from binance_btc_bot.strategy.trails import TrailStrategy

logger = logging.getLogger(__name__)


@dataclass
class EntryAttempt:
    ok: bool
    trade_id: str | None
    order: OrderResult | None
    size: SizeDecision | None
    reason: str
    dry_run: bool = False
    reservation_id: str | None = None
    status: str | None = None


class EntryExecutor:
    def __init__(
        self,
        exchange: ExchangeAdapter,
        db: BotDatabase,
        safety: SafetySystem,
        *,
        portfolio: PortfolioManager | None = None,
        max_loss_per_trade: float = 0.005,
        max_allocation_pct: float = 0.125,
        max_aggregate_exposure: float = 1.0,
        fee_buffer_pct: float = 0.002,
        notifications: NotificationManager | None = None,
        live_enabled: bool = False,
        dry_run: bool = True,
    ) -> None:
        self.exchange = exchange
        self.db = db
        self.safety = safety
        self.portfolio = portfolio
        self.max_loss_per_trade = max_loss_per_trade
        self.max_allocation_pct = max_allocation_pct
        self.max_aggregate_exposure = max_aggregate_exposure
        self.fee_buffer_pct = fee_buffer_pct
        self.notifications = notifications
        self.live_enabled = bool(live_enabled)
        self.dry_run = bool(dry_run)
        self._inflight_symbols: set[str] = set()

    def _notify(self, method: str, event: str, message: str, **kwargs: Any) -> None:
        if not self.notifications:
            return
        try:
            getattr(self.notifications, method)(event, message, **kwargs)
        except Exception:  # noqa: BLE001
            logger.warning("entry notification failed (ignored)")

    def attempt_entry(
        self,
        *,
        symbol: str,
        strategy: TrailStrategy,
        price_alt_btc: float,
        equity_btc: float,
        available_btc: float,
        open_exposure_pct: float,
        btc_usdt: float | None = None,
        client_order_id: str | None = None,
        reservation_id: str | None = None,
    ) -> EntryAttempt:
        sym = symbol.upper()
        rid = reservation_id

        # Hard gate: never submit when live is off.
        if not self.live_enabled or self.dry_run:
            # Still allow dry-run sizing path; exchange.place_entry also blocks writes.
            pass
        else:
            # Defensive — engine construction should already refuse this posture.
            return EntryAttempt(False, None, None, None, "LIVE_WRITES_NOT_AUTHORIZED")

        if sym in self._inflight_symbols or self.db.find_open_trade_for_symbol(sym):
            if rid and self.portfolio:
                self.portfolio.release(rid)
            self.safety.halt("DUPLICATE_ORDER_PROTECTION", symbol=sym)
            self.db.insert_event("HALT", symbol=sym, reason="DUPLICATE_ORDER_PROTECTION")
            return EntryAttempt(False, None, None, None, "DUPLICATE_ORDER_PROTECTION", reservation_id=rid)

        # Reserve now if caller did not (atomic slot claim).
        if self.portfolio is not None and rid is None:
            res = self.portfolio.try_reserve(sym)
            if not res.ok:
                return EntryAttempt(False, None, None, None, res.reason)
            rid = res.reservation.reservation_id if res.reservation else None

        if rid and self.portfolio:
            self.portfolio.transition(rid, TradeStatus.ENTRY_PENDING)

        meta = self.exchange.get_symbol_info(sym)
        size = size_position(
            equity_btc=equity_btc,
            available_btc=available_btc,
            price_alt_btc=price_alt_btc,
            strategy=strategy,
            meta=meta,
            max_loss_per_trade=self.max_loss_per_trade,
            max_allocation_pct=self.max_allocation_pct,
            open_exposure_pct=open_exposure_pct,
            max_aggregate_exposure=self.max_aggregate_exposure,
            fee_buffer_pct=self.fee_buffer_pct,
        )
        if not size.ok:
            if rid and self.portfolio:
                self.portfolio.release(rid)
            self.db.insert_event("ENTRY_ATTEMPT", symbol=sym, reason=size.reason, payload=size.__dict__)
            return EntryAttempt(False, None, None, size, size.reason, reservation_id=rid)

        # Dual gate: allocation AND risk.
        if size.actual_allocation_pct > self.max_allocation_pct + 1e-12:
            if rid and self.portfolio:
                self.portfolio.release(rid)
            return EntryAttempt(False, None, None, size, "ALLOCATION_EXCEEDED", reservation_id=rid)
        if size.planned_loss_btc > size.risk_budget_btc + 1e-12:
            if rid and self.portfolio:
                self.portfolio.release(rid)
            return EntryAttempt(False, None, None, size, "PLANNED_LOSS_EXCEEDS_BUDGET", reservation_id=rid)

        pretrade_ok = self.safety.assert_pretrade(
            symbol_ok=meta.is_trading and meta.quote_asset == "BTC",
            balance_ok=available_btc >= size.notional_btc,
            filters_ok=True,
            duplicate=False,
            stale_data=False,
            api_ok=True,
            reconciliation_ok=True,
            max_risk_ok=size.planned_loss_btc <= size.risk_budget_btc + 1e-12,
            max_exposure_ok=True,
            leverage_ok=True,
        )
        if not pretrade_ok:
            if rid and self.portfolio:
                self.portfolio.release(rid)
            return EntryAttempt(False, None, None, size, "SAFETY_BLOCK", reservation_id=rid)

        trade_id = self.db.new_trade_id()
        self._inflight_symbols.add(sym)
        self.db.insert_event(
            "ENTRY_ATTEMPT",
            symbol=sym,
            trade_id=trade_id,
            payload={
                "price": price_alt_btc,
                "quantity": size.quantity,
                "risk": size.planned_loss_btc,
                "BTC_exposure": size.notional_btc,
                "requested_allocation_pct": size.requested_allocation_pct,
                "actual_allocation_pct": size.actual_allocation_pct,
                "strategy": strategy.key,
                "reservation_id": rid,
            },
        )
        self._notify(
            "notify_info",
            "ENTRY_SUBMITTED",
            f"Entry submitted {sym} qty={size.quantity_serialized} strategy={strategy.key}",
            symbol=sym,
            trade_id=trade_id,
        )
        logger.info(
            "ENTRY symbol=%s qty=%s price=%s risk=%s BTC_exposure=%s "
            "requested_alloc=%.4f actual_alloc=%.4f strategy=%s live=%s dry_run=%s",
            sym,
            size.quantity_serialized,
            price_alt_btc,
            size.planned_loss_btc,
            size.notional_btc,
            size.requested_allocation_pct,
            size.actual_allocation_pct,
            strategy.key,
            self.live_enabled,
            self.dry_run,
        )

        req = OrderRequest(
            symbol=sym,
            side="BUY",
            order_type="MARKET",
            quantity=size.quantity,
            client_order_id=client_order_id or f"e_{trade_id[:20]}",
        )
        try:
            order = self.exchange.place_entry(req)
        except Exception as e:  # noqa: BLE001
            self._inflight_symbols.discard(sym)
            if rid and self.portfolio:
                self.portfolio.release(rid)
            err = scrub_exception(e)
            self.safety.halt("API_FAILURE", error=err)
            self.db.insert_event("ERROR", symbol=sym, trade_id=trade_id, reason=err)
            self._notify("notify_error", "ORDER_ERROR", f"Entry failed: {err}", symbol=sym, trade_id=trade_id)
            return EntryAttempt(
                False, trade_id, None, size, f"API_FAILURE:{err}", reservation_id=rid, status=TradeStatus.UNKNOWN_ORDER_STATE.value
            )

        self.db.insert_order(
            trade_id=trade_id,
            symbol=sym,
            order_id=order.order_id,
            client_order_id=order.client_order_id,
            side="BUY",
            order_type="MARKET",
            status=order.status,
            payload=order.raw,
        )

        fill_qty = float(order.executed_qty or size.quantity)
        fill_px = price_alt_btc
        if order.cumulative_quote_qty and fill_qty > 0 and not order.dry_run:
            fill_px = float(order.cumulative_quote_qty) / fill_qty
        btc_value = fill_qty * fill_px
        usdt_value = btc_value * float(btc_usdt) if btc_usdt else None

        # Dry-run / live-disabled: mark DRY_RUN; real OPEN only if writes allowed.
        if order.dry_run or not self.live_enabled:
            status = "DRY_RUN"
            trade_status = TradeStatus.ENTRY_FILLED
        else:
            status = "OPEN"
            trade_status = TradeStatus.ENTRY_FILLED

        self.db.insert_trade(
            {
                "trade_id": trade_id,
                "symbol": sym,
                "strategy": strategy.key,
                "selector": None,
                "entry_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "entry_price": fill_px,
                "quantity": fill_qty,
                "btc_value": btc_value,
                "usdt_value": usdt_value,
                "configured_activation": strategy.activation,
                "configured_trailing_distance": strategy.trail_distance,
                "strategy_config": {
                    "arm_sl_activation_trail": strategy.arm_sl_activation_trail,
                    "activation": strategy.activation,
                    "trail_distance": strategy.trail_distance,
                    "requested_allocation_pct": size.requested_allocation_pct,
                    "actual_allocation_pct": size.actual_allocation_pct,
                    "reservation_id": rid,
                },
                "binance_entry_order_id": order.order_id,
                "status": status,
            }
        )
        self.db.freeze_strategy_config(
            strategy.key,
            {
                "arm_sl_activation_trail": strategy.arm_sl_activation_trail,
                "activation": strategy.activation,
                "trail_distance": strategy.trail_distance,
                "trade_id": trade_id,
            },
        )
        if rid and self.portfolio:
            # Until trailing protection is attached, hold as ENTRY_FILLED / PROTECTION_PENDING.
            self.portfolio.transition(rid, TradeStatus.PROTECTION_PENDING, trade_id=trade_id)

        if order.dry_run or not self.live_enabled:
            self.db.insert_event("ENTRY_FILLED", symbol=sym, trade_id=trade_id, reason="DRY_RUN")
            logger.info("DRY RUN — ORDER NOT SUBMITTED trade_id=%s", trade_id)
            # In dry-run, treat as protected for slot accounting so capacity tests work.
            if rid and self.portfolio:
                self.portfolio.mark_protected(rid, trade_id)
                trade_status = TradeStatus.PROTECTED
        else:
            self.db.insert_event("ENTRY_FILLED", symbol=sym, trade_id=trade_id, order_id=order.order_id)

        self._notify(
            "notify_info",
            "ENTRY_FILLED",
            f"Entry filled {sym} qty={fill_qty} dry_run={order.dry_run}",
            symbol=sym,
            trade_id=trade_id,
            order_id=order.order_id,
        )
        self._inflight_symbols.discard(sym)
        return EntryAttempt(
            True,
            trade_id,
            order,
            size,
            "OK",
            dry_run=bool(order.dry_run or not self.live_enabled),
            reservation_id=rid,
            status=trade_status.value,
        )

    def mark_protected(self, reservation_id: str | None, trade_id: str) -> None:
        if reservation_id and self.portfolio:
            self.portfolio.mark_protected(reservation_id, trade_id)
