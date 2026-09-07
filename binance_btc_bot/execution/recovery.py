"""Startup / restart recovery — Binance is source of truth for live orders."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from binance_btc_bot.exchange.base import ExchangeAdapter
from binance_btc_bot.execution.dry_broker import DryRunBroker
from binance_btc_bot.notifications.manager import NotificationManager
from binance_btc_bot.portfolio.manager import PortfolioManager
from binance_btc_bot.secrets import scrub_exception
from binance_btc_bot.risk.safety import SafetySystem
from binance_btc_bot.storage.database import BotDatabase

logger = logging.getLogger(__name__)


@dataclass
class RecoveryReport:
    ok: bool
    open_orders: list[dict[str, Any]] = field(default_factory=list)
    open_order_lists: list[dict[str, Any]] = field(default_factory=list)
    local_open_trades: list[dict[str, Any]] = field(default_factory=list)
    recovered_trade_ids: list[str] = field(default_factory=list)
    orphan_exchange_orders: list[dict[str, Any]] = field(default_factory=list)
    orphan_local_trades: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    cases: list[str] = field(default_factory=list)


class RecoveryManager:
    def __init__(
        self,
        exchange: ExchangeAdapter,
        db: BotDatabase,
        safety: SafetySystem,
        *,
        notifications: NotificationManager | None = None,
        dry_broker: DryRunBroker | None = None,
        portfolio: PortfolioManager | None = None,
    ) -> None:
        self.exchange = exchange
        self.db = db
        self.safety = safety
        self.notifications = notifications
        self.dry_broker = dry_broker
        self.portfolio = portfolio

    def _notify(self, method: str, event: str, message: str, **kwargs: Any) -> None:
        if not self.notifications:
            return
        try:
            getattr(self.notifications, method)(event, message, **kwargs)
        except Exception:  # noqa: BLE001
            logger.warning("recovery notification failed (ignored)")

    def recover(self, universe: list[str], *, dry_run: bool = True) -> RecoveryReport:
        """Reconcile local OPEN trades with Binance open orders / OCO lists.

        HALT blocks new entries but never skips recovery / protection reconciliation.
        """
        report = RecoveryReport(ok=True)
        local_open = self.db.open_trades()
        report.local_open_trades = local_open

        if dry_run and self.dry_broker is not None:
            report = self._recover_dry_broker(universe, report)
        elif dry_run:
            report.notes.append("dry_run_or_unsigned: skipped private openOrders reconciliation")
            self.db.insert_event("RECOVERY", reason="DRY_SKIP_PRIVATE", payload={"local_open": len(local_open)})
            logger.info("RECOVERY dry/unsigned — local_open=%d", len(local_open))
        else:
            try:
                open_orders = self.exchange.get_open_orders()
                open_lists = self.exchange.get_open_order_lists()
            except Exception as e:  # noqa: BLE001
                err = scrub_exception(e)
                self.safety.halt("API_FAILURE", stage="recovery", error=err)
                report.ok = False
                report.notes.append(f"API_FAILURE:{err}")
                self.db.insert_event("HALT", reason=f"RECOVERY_API_FAILURE:{err}")
                self._notify(
                    "notify_critical",
                    "RECOVERY_FAILURE",
                    f"Recovery API failure: {err}",
                    reason=err,
                )
                return report

            report.open_orders = open_orders
            report.open_order_lists = open_lists
            report = self._reconcile_lists(universe, report, open_lists)

        if self.portfolio is not None:
            rh = self.portfolio.rehydrate_from_trades(self.db.open_trades())
            report.notes.append(f"portfolio_rehydrate:{rh}")
        return report

    def _recover_dry_broker(self, universe: list[str], report: RecoveryReport) -> RecoveryReport:
        assert self.dry_broker is not None
        open_orders = self.dry_broker.open_orders()
        open_lists = self.dry_broker.open_order_lists()
        report.open_orders = open_orders
        report.open_order_lists = open_lists
        report.notes.append("dry_broker_source_of_truth")

        # Offline exits (Case D)
        for oco in self.dry_broker.all_ocos():
            if oco.list_order_status == "ALL_DONE" and oco.exit_fill:
                for t in list(report.local_open_trades):
                    if str(t.get("binance_oco_list_id")) == str(oco.order_list_id):
                        self.db.update_trade(
                            t["trade_id"],
                            status="CLOSED",
                            exit_price=float(oco.exit_fill["price"]),
                            exit_time=__import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
                        )
                        report.recovered_trade_ids.append(t["trade_id"])
                        report.cases.append("D")
                        report.notes.append(f"D:offline_exit:{t['trade_id']}")
                        self.db.insert_event(
                            "RECOVERY",
                            trade_id=t["trade_id"],
                            reason="OFFLINE_EXIT",
                            order_id=oco.order_list_id,
                        )

        # Refresh local after closes
        report.local_open_trades = self.db.open_trades()
        return self._reconcile_lists(universe, report, open_lists)

    def _reconcile_lists(
        self,
        universe: list[str],
        report: RecoveryReport,
        open_lists: list[dict[str, Any]],
    ) -> RecoveryReport:
        lists_by_symbol: dict[str, list[dict[str, Any]]] = {}
        for row in open_lists:
            sym = str(row.get("symbol") or "").upper()
            lists_by_symbol.setdefault(sym, []).append(row)

        local_by_symbol = {str(t["symbol"]).upper(): t for t in report.local_open_trades}

        for sym, trade in local_by_symbol.items():
            lists = lists_by_symbol.get(sym) or []
            oco_id = trade.get("binance_oco_list_id")
            entry_id = trade.get("binance_entry_order_id")
            st = str(trade.get("status") or "").upper()

            # Case A: BUY not filled / no protection and no entry
            if not entry_id and not oco_id and st in {"ENTRY_PENDING", "OPEN", "DRY_RUN"}:
                report.cases.append("A")
                report.notes.append(f"A:buy_not_filled:{trade['trade_id']}")
                self.db.insert_event("RECOVERY", trade_id=trade["trade_id"], reason="BUY_NOT_FILLED")
                continue

            if oco_id and any(str(x.get("orderListId")) == str(oco_id) for x in lists):
                report.recovered_trade_ids.append(trade["trade_id"])
                # Distinguish trailing-active vs submitted (Cases B/C)
                matched = next(x for x in lists if str(x.get("orderListId")) == str(oco_id))
                if str(matched.get("listStatusType") or "") == "TRAILING_ACTIVE":
                    report.cases.append("C")
                    reason = "TRAILING_ACTIVE"
                else:
                    report.cases.append("B")
                    reason = "MATCHED_OPEN_OCO"
                self.db.insert_event(
                    "RECOVERY",
                    symbol=sym,
                    trade_id=trade["trade_id"],
                    order_id=str(oco_id),
                    reason=reason,
                )
            elif lists:
                chosen = lists[0]
                self.db.update_trade(trade["trade_id"], binance_oco_list_id=str(chosen.get("orderListId")))
                report.recovered_trade_ids.append(trade["trade_id"])
                report.cases.append("B")
                self.db.insert_event(
                    "RECOVERY",
                    symbol=sym,
                    trade_id=trade["trade_id"],
                    order_id=str(chosen.get("orderListId")),
                    reason="REBOUND_OCO",
                )
            else:
                report.orphan_local_trades.append(trade)
                self.safety.warn("LOCAL_OPEN_WITHOUT_EXCHANGE_OCO", symbol=sym, trade_id=trade["trade_id"])
                self.db.insert_event(
                    "WARNING",
                    symbol=sym,
                    trade_id=trade["trade_id"],
                    reason="LOCAL_OPEN_WITHOUT_EXCHANGE_OCO",
                )

        for sym, lists in lists_by_symbol.items():
            if sym not in local_by_symbol and sym in {u.upper() for u in universe}:
                for row in lists:
                    report.orphan_exchange_orders.append(row)
                    report.cases.append("E")
                    self.safety.warn("EXCHANGE_OCO_WITHOUT_LOCAL_TRADE", symbol=sym)
                    self.db.insert_event(
                        "WARNING",
                        symbol=sym,
                        order_id=str(row.get("orderListId")),
                        reason="EXCHANGE_OCO_WITHOUT_LOCAL_TRADE",
                    )

        for sym, lists in lists_by_symbol.items():
            if len(lists) > 1:
                self.safety.halt("DUPLICATE_ORDER_PROTECTION", symbol=sym, count=len(lists))
                report.ok = False
                report.notes.append(f"duplicate_oco:{sym}:{len(lists)}")

        report.cases = sorted(set(report.cases))
        self.db.insert_event(
            "RECOVERY",
            reason="COMPLETE",
            payload={
                "recovered": report.recovered_trade_ids,
                "orphan_local": len(report.orphan_local_trades),
                "orphan_exchange": len(report.orphan_exchange_orders),
                "cases": report.cases,
            },
        )
        self._notify(
            "notify_error" if not report.ok else "notify_info",
            "RECOVERY",
            f"Recovery complete ok={report.ok} recovered={len(report.recovered_trade_ids)}",
        )
        logger.info(
            "RECOVERY complete recovered=%d orphan_local=%d orphan_exchange=%d cases=%s",
            len(report.recovered_trade_ids),
            len(report.orphan_local_trades),
            len(report.orphan_exchange_orders),
            report.cases,
        )
        return report
