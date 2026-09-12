"""Layer 2 — complete Spot order lifecycle (dry-run safe).

NEW CROSS → reserve → size → risk → filters → MARKET BUY → wait fill →
native T1 OCO → verify accepted → PROTECTED → exit → CLOSE → accounting → release slot

CRITICAL: BUY filled + OCO fail → PROTECTION_FAILED → halt new entries → emergency → HALT
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from binance_btc_bot.accounting.btc_accounting import TradeAccounting
from binance_btc_bot.exchange.base import ExchangeAdapter, OrderRequest, OrderResult
from binance_btc_bot.execution.dry_broker import DryRunBroker
from binance_btc_bot.execution.fills import AggregatedFill, aggregate_fills_from_order
from binance_btc_bot.execution.protection_qty import (
    ProtectableQuantity,
    compute_protectable_quantity,
    is_insufficient_balance_reason,
)
from binance_btc_bot.execution.trailing import TrailingExecutor, TrailingSubmitResult
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
class LifecycleResult:
    ok: bool
    trade_id: str | None = None
    reservation_id: str | None = None
    status: str | None = None
    reason: str = ""
    fill: AggregatedFill | None = None
    size: SizeDecision | None = None
    oco: TrailingSubmitResult | None = None
    accounting: TradeAccounting | None = None
    dry_run: bool = True
    events: list[str] = field(default_factory=list)


class OrderLifecycle:
    """Production-safe order lifecycle controller (works under dry-run)."""

    def __init__(
        self,
        exchange: ExchangeAdapter,
        db: BotDatabase,
        safety: SafetySystem,
        *,
        portfolio: PortfolioManager | None = None,
        trailing: TrailingExecutor | None = None,
        notifications: NotificationManager | None = None,
        dry_broker: DryRunBroker | None = None,
        live_enabled: bool = False,
        dry_run: bool = True,
        max_loss_per_trade: float = 0.005,
        max_allocation_pct: float = 0.125,
        max_aggregate_exposure: float = 1.0,
        fee_buffer_pct: float = 0.002,
        fill_poll_attempts: int = 5,
        fill_poll_delay_sec: float = 0.0,
    ) -> None:
        self.exchange = exchange
        self.db = db
        self.safety = safety
        self.portfolio = portfolio
        self.trailing = trailing or TrailingExecutor(exchange, db, notifications=notifications)
        self.notifications = notifications
        self.dry_broker = dry_broker if dry_run else None
        self.live_enabled = bool(live_enabled)
        self.dry_run = bool(dry_run)
        self.max_loss_per_trade = max_loss_per_trade
        self.max_allocation_pct = max_allocation_pct
        self.max_aggregate_exposure = max_aggregate_exposure
        self.fee_buffer_pct = fee_buffer_pct
        self.fill_poll_attempts = fill_poll_attempts
        self.fill_poll_delay_sec = fill_poll_delay_sec
        self._client_order_index: dict[str, str] = {}  # client_order_id -> trade_id
        self._seen_notifications: set[str] = set()
        # Idempotency: one emergency submission attempt per trade_id in-process.
        self._emergency_submitted: set[str] = set()

    def _notify(self, method: str, event: str, message: str, *, dedupe_key: str | None = None, **kwargs: Any) -> None:
        if dedupe_key:
            if dedupe_key in self._seen_notifications:
                return
            self._seen_notifications.add(dedupe_key)
        if not self.notifications:
            return
        try:
            getattr(self.notifications, method)(event, message, **kwargs)
        except Exception:  # noqa: BLE001
            logger.warning("lifecycle notification failed (ignored)")

    def _client_buy_id(self, trade_id: str) -> str:
        return f"e_{trade_id[:20]}"

    def _client_oco_id(self, trade_id: str) -> str:
        return f"t_{trade_id[:20]}"

    def run_entry(
        self,
        *,
        symbol: str,
        strategy: TrailStrategy,
        price_alt_btc: float,
        equity_btc: float,
        available_btc: float,
        open_exposure_pct: float = 0.0,
        btc_usdt: float | None = None,
        reservation_id: str | None = None,
        signal: dict[str, Any] | None = None,
        selector: str | None = None,
        portfolio_before: dict[str, Any] | None = None,
        portfolio_after: dict[str, Any] | None = None,
    ) -> LifecycleResult:
        events: list[str] = []
        sym = symbol.upper()
        rid = reservation_id
        signal_snippet = dict(signal) if isinstance(signal, dict) else None
        selector_key = str(selector) if selector else None
        port_before = dict(portfolio_before) if isinstance(portfolio_before, dict) else None
        port_after = dict(portfolio_after) if isinstance(portfolio_after, dict) else None

        if not self.safety.allow_new_entries():
            return LifecycleResult(False, reason="SAFETY_HALT", dry_run=self.dry_run, events=events)

        if self.portfolio is not None and rid is None:
            res = self.portfolio.try_reserve(sym)
            if not res.ok:
                return LifecycleResult(False, reason=res.reason, dry_run=self.dry_run, events=events)
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
            self._release(rid)
            return LifecycleResult(False, reservation_id=rid, size=size, reason=size.reason, dry_run=self.dry_run, events=events)
        events.append("SIZED")

        # Dual gate
        if size.actual_allocation_pct > self.max_allocation_pct + 1e-12:
            self._release(rid)
            return LifecycleResult(False, reservation_id=rid, size=size, reason="ALLOCATION_EXCEEDED", dry_run=self.dry_run)
        if size.planned_loss_btc > size.risk_budget_btc + 1e-12:
            self._release(rid)
            return LifecycleResult(False, reservation_id=rid, size=size, reason="PLANNED_LOSS_EXCEEDS_BUDGET", dry_run=self.dry_run)

        if not meta.is_trading or meta.quote_asset.upper() != "BTC":
            self._release(rid)
            return LifecycleResult(False, reservation_id=rid, size=size, reason="BINANCE_FILTER_FAILURE", dry_run=self.dry_run)
        events.append("FILTERS_OK")

        trade_id = self.db.new_trade_id()
        client_id = self._client_buy_id(trade_id)

        # Idempotency: if client id already known, reuse.
        existing = self._lookup_existing_buy(client_id, sym)
        if existing is not None:
            events.append("IDEMPOTENT_REUSE")
            order = existing
        else:
            order = self._submit_buy(
                symbol=sym,
                quantity=size.quantity,
                client_order_id=client_id,
                ref_price=price_alt_btc,
            )
            events.append("BUY_SUBMITTED")

        if not order.ok and not order.dry_run:
            self._release(rid)
            return LifecycleResult(False, trade_id=trade_id, reservation_id=rid, size=size, reason=order.reason or "BUY_REJECTED", dry_run=self.dry_run, events=events)

        fill = self._wait_for_buy_fill(order, symbol=sym, client_order_id=client_id, ref_price=price_alt_btc, planned_qty=size.quantity)
        events.append(f"BUY_STATUS:{fill.status}")

        if fill.status in {"REJECTED", "CANCELED", "CANCELLED", "EXPIRED"} or not fill.ok:
            self._release(rid)
            self.db.insert_event("ENTRY_FAILED", symbol=sym, trade_id=trade_id, reason=fill.status)
            return LifecycleResult(
                False,
                trade_id=trade_id,
                reservation_id=rid,
                size=size,
                fill=fill,
                reason=f"BUY_{fill.status}",
                dry_run=self.dry_run,
                events=events,
                status=TradeStatus.CLOSED.value,
            )

        if fill.partially_filled and not fill.fully_filled:
            # Safe state: do not protect assumed quantity.
            self._release(rid)
            self.safety.halt("PARTIAL_FILL_UNSAFE", symbol=sym, trade_id=trade_id)
            self._notify(
                "notify_critical",
                "PARTIAL_FILL_UNSAFE",
                f"Partial BUY fill on {sym}; refusing assumed protection",
                symbol=sym,
                trade_id=trade_id,
                dedupe_key=f"partial:{trade_id}",
            )
            return LifecycleResult(
                False,
                trade_id=trade_id,
                reservation_id=rid,
                size=size,
                fill=fill,
                reason="PARTIAL_FILL_UNSAFE",
                dry_run=self.dry_run,
                events=events,
                status=TradeStatus.HALTED.value,
            )

        events.append("BUY_FILLED")
        if rid and self.portfolio:
            self.portfolio.transition(rid, TradeStatus.ENTRY_FILLED, trade_id=trade_id)

        # Persist trade with ACTUAL fill (never request price).
        entry_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        btc_value = float(fill.executed_qty) * float(fill.avg_price)
        usdt_value = btc_value * float(btc_usdt) if btc_usdt else None
        strategy_cfg: dict[str, Any] = {
            "arm_sl_activation_trail": strategy.arm_sl_activation_trail,
            "activation": strategy.activation,
            "trail_distance": strategy.trail_distance,
            "requested_allocation_pct": size.requested_allocation_pct,
            "actual_allocation_pct": btc_value / equity_btc if equity_btc else None,
            "reservation_id": rid,
            "fill": {
                "avg_price": fill.avg_price,
                "executed_qty": fill.executed_qty,
                "commission_btc": fill.commission_btc,
                "commission_usdt": fill.commission_usdt,
                "order_id": fill.order_id,
            },
        }
        if signal_snippet:
            strategy_cfg["signal"] = signal_snippet
        if selector_key:
            strategy_cfg["selector"] = selector_key
        self.db.insert_trade(
            {
                "trade_id": trade_id,
                "symbol": sym,
                "strategy": strategy.key,
                "selector": selector_key,
                "entry_time": entry_time,
                "entry_price": fill.avg_price,
                "quantity": fill.executed_qty,
                "btc_value": btc_value,
                "usdt_value": usdt_value,
                "configured_activation": strategy.activation,
                "configured_trailing_distance": strategy.trail_distance,
                "strategy_config": strategy_cfg,
                "binance_entry_order_id": fill.order_id,
                "status": "ENTRY_FILLED" if not self.dry_run else "DRY_RUN",
                "fees_btc": fill.commission_btc,
                "fees_usdt": fill.commission_usdt,
            }
        )
        # Persist full entry factor snapshot (if caller attached it to signal).
        try:
            snap = None
            if isinstance(signal_snippet, dict):
                snap = signal_snippet.get("entry_snapshot")
            if isinstance(snap, dict) and snap:
                self.db.insert_signal(
                    symbol=sym,
                    score=signal_snippet.get("current_s") if signal_snippet else None,
                    relative_price=snap.get("relative_price"),
                    strategy=strategy.key,
                    classification="ENTRY_SNAPSHOT",
                    payload={
                        "trade_id": trade_id,
                        "previous_s": (signal_snippet or {}).get("previous_s"),
                        "current_s": (signal_snippet or {}).get("current_s"),
                        "threshold": (signal_snippet or {}).get("threshold"),
                        "entry_snapshot": snap,
                    },
                )
                self.db.insert_event(
                    "ENTRY_SNAPSHOT",
                    symbol=sym,
                    trade_id=trade_id,
                    reason="FACTORS_AT_ENTRY",
                    payload={"entry_snapshot": snap, "signal": signal_snippet},
                )
                snap_dir = Path(self.db.path).resolve().parent / "entry_snapshots"
                snap_dir.mkdir(parents=True, exist_ok=True)
                (snap_dir / f"{trade_id}.json").write_text(
                    json.dumps(
                        {
                            "trade_id": trade_id,
                            "symbol": sym,
                            "strategy": strategy.key,
                            "entry_time": entry_time,
                            "signal": signal_snippet,
                            "entry_snapshot": snap,
                        },
                        indent=2,
                        default=str,
                    ),
                    encoding="utf-8",
                )
                events.append("ENTRY_SNAPSHOT_SAVED")
        except Exception as e:  # noqa: BLE001
            logger.warning("entry snapshot persist failed: %s", scrub_exception(e))
            events.append("ENTRY_SNAPSHOT_FAILED")
        self.db.insert_order(
            trade_id=trade_id,
            symbol=sym,
            order_id=fill.order_id,
            client_order_id=client_id,
            side="BUY",
            order_type="MARKET",
            status=fill.status,
            payload=fill.raw,
        )
        self._client_order_index[client_id] = trade_id

        if rid and self.portfolio:
            self.portfolio.transition(rid, TradeStatus.PROTECTION_PENDING, trade_id=trade_id)

        protect_qty, protect_info, oco, verified, verified_list_id, events = self._submit_protection(
            trade_id=trade_id,
            symbol=sym,
            strategy=strategy,
            fill=fill,
            events=events,
        )
        if protect_qty is None or protect_qty <= 0:
            return self._handle_protection_failure(
                trade_id=trade_id,
                reservation_id=rid,
                symbol=sym,
                fill=fill,
                size=size,
                oco=oco,
                events=events,
                reason=(protect_info.reason if protect_info else "PROTECT_QTY_INVALID"),
                strategy=strategy,
            )

        if not (oco and oco.ok and verified):
            return self._handle_protection_failure(
                trade_id=trade_id,
                reservation_id=rid,
                symbol=sym,
                fill=fill,
                size=size,
                oco=oco,
                events=events,
                reason=oco.reason if oco and not oco.ok else "OCO_NOT_VERIFIED",
                strategy=strategy,
            )

        # PROTECTED only after acceptance verified.
        list_id = verified_list_id or (oco.order.order_id if oco.order else None)
        if rid and self.portfolio:
            self.portfolio.mark_protected(rid, trade_id)
        self.db.update_trade(
            trade_id,
            status="PROTECTED" if not self.dry_run else "DRY_RUN_PROTECTED",
            binance_oco_list_id=list_id,
            quantity=protect_qty,
        )
        self.db.insert_event(
            "TRADE_PROTECTED",
            symbol=sym,
            trade_id=trade_id,
            order_id=list_id,
            payload={
                "protect_qty": protect_qty,
                "protect": protect_info.to_dict() if protect_info else None,
            },
        )
        events.append("PROTECTED")
        if port_after is None and self.portfolio is not None:
            try:
                port_after = {
                    "equity_btc": equity_btc,
                    "btc_free": available_btc,
                    "open_trades": self.portfolio.slots_used(),
                }
            except Exception:  # noqa: BLE001
                port_after = None
        try:
            from binance_btc_bot.notifications.report_payloads import protection_payload, trade_open_payload

            base = meta.base_asset if meta else sym.replace("BTC", "")
            open_details = trade_open_payload(
                symbol=sym,
                strategy=strategy.key,
                selector=selector_key,
                signal=signal_snippet,
                entry={
                    "side": "BUY",
                    "quantity": protect_qty,
                    "base_asset": base,
                    "avg_price": fill.avg_price,
                    "btc_invested": float(protect_qty) * float(fill.avg_price),
                    "actual_allocation_pct": (
                        float(size.actual_allocation_pct) * 100.0
                        if size and size.actual_allocation_pct is not None
                        else None
                    ),
                    "order_id": fill.order_id,
                    "client_order_id": client_id,
                },
                fees={
                    "fees_btc": fill.commission_btc,
                    "fees_usdt": fill.commission_usdt,
                },
                protection={
                    "strategy": strategy.key,
                    "entry_price": fill.avg_price,
                    "activation_price": (
                        oco.activation_price
                        if oco and oco.activation_price is not None
                        else float(fill.avg_price) * (1.0 + float(strategy.activation))
                    ),
                    "stop_loss_price": (
                        oco.initial_stop
                        if oco and oco.initial_stop is not None
                        else float(fill.avg_price)
                        * (1.0 - float(strategy.arm_sl_activation_trail))
                    ),
                    "activation_display": f"{strategy.activation * 100:.2f}%",
                    "trail_display": f"{strategy.trail_distance * 100:.2f}%",
                    "hard_sl_display": f"{strategy.arm_sl_activation_trail * 100:.2f}%",
                    "oco_id": list_id,
                    "oco_status": "ACCEPTED",
                    "protected_qty": protect_qty,
                },
                portfolio_before=port_before,
                portfolio_after=port_after,
            )
            self._notify(
                "notify_info",
                "TRADE_OPENED",
                f"Trade opened {sym}",
                symbol=sym,
                trade_id=trade_id,
                dedupe_key=f"opened:{trade_id}",
                details=open_details,
            )
            self._notify(
                "notify_info",
                "TRADE_PROTECTED",
                f"Trade protected {sym} qty={protect_qty} entry={fill.avg_price}",
                symbol=sym,
                trade_id=trade_id,
                dedupe_key=f"protected:{trade_id}",
                details=protection_payload(
                    kind="OCO_ACCEPTED",
                    symbol=sym,
                    quantity=protect_qty,
                    reason="OCO_ACCEPTED",
                    protection_state="PROTECTED" if not self.dry_run else "DRY_RUN_PROTECTED",
                    binance_verified=True,
                    new_entries_status="ALLOWED" if self.safety.allow_new_entries() else "BLOCKED",
                    oco_id=list_id,
                ),
            )
        except Exception:  # noqa: BLE001
            self._notify(
                "notify_info",
                "TRADE_PROTECTED",
                f"Trade protected {sym} qty={protect_qty} entry={fill.avg_price}",
                symbol=sym,
                trade_id=trade_id,
                dedupe_key=f"protected:{trade_id}",
            )

        accounting = TradeAccounting(
            trade_id=trade_id,
            symbol=sym,
            strategy=strategy.key,
            selector=selector_key,
            entry_time=entry_time,
            entry_price=fill.avg_price,
            quantity=protect_qty,
            entry_btc_value=float(protect_qty) * float(fill.avg_price),
            entry_usdt_value=(
                (float(protect_qty) * float(fill.avg_price) * float(btc_usdt)) if btc_usdt else None
            ),
            configured_activation=strategy.activation,
            configured_trailing_distance=strategy.trail_distance,
            binance_entry_order_id=fill.order_id,
            binance_oco_list_id=list_id,
            fees_btc=fill.commission_btc,
            fees_usdt=fill.commission_usdt,
        )
        return LifecycleResult(
            ok=True,
            trade_id=trade_id,
            reservation_id=rid,
            status=TradeStatus.PROTECTED.value,
            reason="OK",
            fill=fill,
            size=size,
            oco=oco,
            accounting=accounting,
            dry_run=self.dry_run,
            events=events,
        )

    def close_from_exit_fill(
        self,
        *,
        trade_id: str,
        reservation_id: str | None,
        exit_price: float | None,
        exit_qty: float | None = None,
        fees_btc: float = 0.0,
        fees_usdt: float = 0.0,
        btc_usdt: float | None = None,
        other_leg_cancelled: bool = True,
        close_reason: str = "UNKNOWN",
        exit_order_id: str | None = None,
        exit_time: str | None = None,
        commission_asset: str | None = None,
        portfolio_before: dict[str, Any] | None = None,
        portfolio_after: dict[str, Any] | None = None,
        reconciliation: dict[str, Any] | None = None,
    ) -> LifecycleResult:
        """Handle OCO exit fill → accounting → CLOSED → release slot."""
        events = ["EXIT_FILL"]
        if not other_leg_cancelled:
            events.append("OTHER_LEG_NOT_CANCELLED")
            self.safety.warn("OCO_OTHER_LEG_OPEN", trade_id=trade_id)

        # Load trade row
        row = None
        try:
            cur = self.db._conn.execute("SELECT * FROM trades WHERE trade_id=?", (trade_id,))
            row = cur.fetchone()
        except Exception:  # noqa: BLE001
            row = None
        if row is None:
            return LifecycleResult(False, trade_id=trade_id, reason="TRADE_NOT_FOUND", events=events)

        tr = dict(row)
        if str(tr.get("status") or "").upper() == "CLOSED":
            return LifecycleResult(
                ok=True,
                trade_id=trade_id,
                reservation_id=reservation_id,
                status=TradeStatus.CLOSED.value,
                reason="ALREADY_CLOSED",
                dry_run=self.dry_run,
                events=["ALREADY_CLOSED"],
            )

        qty = float(exit_qty if exit_qty is not None else tr.get("quantity") or 0)
        entry_price = float(tr.get("entry_price") or 0)
        entry_btc = float(tr.get("btc_value") or (qty * entry_price))
        rid = reservation_id if reservation_id is not None else self._reservation_id_from_trade(tr)
        accounting = TradeAccounting(
            trade_id=trade_id,
            symbol=str(tr.get("symbol")),
            strategy=str(tr.get("strategy")),
            selector=tr.get("selector"),
            entry_time=tr.get("entry_time"),
            entry_price=entry_price,
            quantity=qty,
            entry_btc_value=entry_btc,
            entry_usdt_value=tr.get("usdt_value"),
            configured_activation=float(tr.get("configured_activation") or 0),
            configured_trailing_distance=float(tr.get("configured_trailing_distance") or 0),
            binance_entry_order_id=tr.get("binance_entry_order_id"),
            binance_oco_list_id=tr.get("binance_oco_list_id"),
            fees_btc=float(tr.get("fees_btc") or 0),
            fees_usdt=float(tr.get("fees_usdt") or 0),
        )
        resolved_exit_time = exit_time or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        exit_px: float | None = None
        if exit_price is not None:
            try:
                exit_px = float(exit_price)
            except (TypeError, ValueError):
                exit_px = None

        update_fields: dict[str, Any] = {
            "status": "CLOSED",
            "exit_time": resolved_exit_time,
        }
        if exit_px is not None:
            accounting.mark_exit(
                exit_time=resolved_exit_time,
                exit_price=exit_px,
                btc_usdt=btc_usdt,
                fees_btc=fees_btc,
                fees_usdt=fees_usdt,
            )
            update_fields.update(
                {
                    "exit_price": exit_px,
                    "fees_btc": accounting.fees_btc,
                    "fees_usdt": accounting.fees_usdt,
                    "realized_pnl_btc": accounting.realized_pnl_btc,
                    "realized_pnl_usdt": accounting.realized_pnl_usdt,
                    "realized_pnl_btc_equivalent": accounting.realized_pnl_btc_equivalent,
                }
            )
        else:
            # Flat/ALL_DONE without authoritative fill price — do not invent PnL.
            accounting.exit_time = resolved_exit_time
            accounting.fees_btc += float(fees_btc or 0)
            accounting.fees_usdt += float(fees_usdt or 0)
            update_fields["fees_btc"] = accounting.fees_btc
            update_fields["fees_usdt"] = accounting.fees_usdt

        self.db.update_trade(trade_id, **update_fields)
        self.db.insert_event(
            "EXIT_FILLED",
            symbol=tr.get("symbol"),
            trade_id=trade_id,
            order_id=str(exit_order_id) if exit_order_id is not None else None,
            payload=accounting.to_dict(),
        )
        events.append("CLOSED")
        events.append("ACCOUNTING_DONE")
        if rid and self.portfolio:
            self.portfolio.release(rid)
        if self.portfolio:
            self.portfolio.release_symbol(str(tr.get("symbol")))
        events.append("SLOT_RELEASED")

        price_pnl_pct = None
        if exit_px is not None and entry_price > 0:
            price_pnl_pct = ((exit_px - entry_price) / entry_price) * 100.0
        net_pnl = accounting.realized_pnl_btc
        gross_pnl = None
        if net_pnl is not None:
            gross_pnl = float(net_pnl) + float(accounting.fees_btc or 0)

        duration = None
        try:
            if tr.get("entry_time") and resolved_exit_time:
                # Best-effort ISO duration label; ignore parse failures.
                from datetime import datetime

                et = datetime.fromisoformat(str(tr["entry_time"]).replace("Z", "+00:00"))
                xt = datetime.fromisoformat(str(resolved_exit_time).replace("Z", "+00:00"))
                secs = max(0, int((xt - et).total_seconds()))
                if secs < 60:
                    duration = f"{secs}s"
                elif secs < 3600:
                    duration = f"{secs // 60}m"
                else:
                    duration = f"{secs // 3600}h{(secs % 3600) // 60:02d}m"
        except Exception:  # noqa: BLE001
            duration = None

        recon = dict(reconciliation or {})
        if not recon:
            recon = {"binance": "N/A", "local_db": "PASS"}

        try:
            from binance_btc_bot.notifications.report_payloads import trade_close_payload

            close_details = trade_close_payload(
                symbol=str(tr.get("symbol")),
                strategy=str(tr.get("strategy")),
                selector=tr.get("selector"),
                close_reason=str(close_reason or "UNKNOWN"),
                duration=duration,
                entry={
                    "timestamp": tr.get("entry_time"),
                    "quantity": qty,
                    "avg_price": entry_price,
                    "btc_invested": entry_btc,
                    "order_id": tr.get("binance_entry_order_id"),
                },
                exit={
                    "timestamp": resolved_exit_time,
                    "quantity": qty,
                    "avg_price": exit_px,
                    "order_id": exit_order_id,
                },
                result={
                    "price_pnl_pct": price_pnl_pct,
                    "gross_pnl_btc": gross_pnl,
                    "fees_btc": accounting.fees_btc,
                    "net_realized_pnl_btc": net_pnl,
                    "commission_assets": commission_asset,
                },
                trailing={
                    "activation": tr.get("configured_activation"),
                    "trail_distance": tr.get("configured_trailing_distance"),
                },
                portfolio_before=portfolio_before,
                portfolio_after=portfolio_after,
                reconciliation=recon,
            )
            self._notify(
                "notify_info",
                "TRADE_CLOSED",
                f"Exit filled {tr.get('symbol')} pnl_btc={accounting.realized_pnl_btc}",
                symbol=tr.get("symbol"),
                trade_id=trade_id,
                dedupe_key=f"closed:{trade_id}",
                details=close_details,
            )
        except Exception:  # noqa: BLE001
            self._notify(
                "notify_info",
                "EXIT_FILLED",
                f"Exit filled {tr.get('symbol')} pnl_btc={accounting.realized_pnl_btc}",
                symbol=tr.get("symbol"),
                trade_id=trade_id,
                dedupe_key=f"closed:{trade_id}",
            )
        return LifecycleResult(
            ok=True,
            trade_id=trade_id,
            reservation_id=rid,
            status=TradeStatus.CLOSED.value,
            reason="OK",
            accounting=accounting,
            dry_run=self.dry_run,
            events=events,
        )

    # --- internals ---------------------------------------------------------

    def _release(self, reservation_id: str | None) -> None:
        if reservation_id and self.portfolio:
            self.portfolio.release(reservation_id)

    def _lookup_existing_buy(self, client_order_id: str, symbol: str) -> OrderResult | None:
        if self.dry_broker:
            sim = self.dry_broker.lookup_by_client_order_id(client_order_id)
            if sim:
                return OrderResult(
                    ok=True,
                    order_id=sim.order_id,
                    client_order_id=sim.client_order_id,
                    status=sim.status,
                    symbol=sim.symbol,
                    side=sim.side,
                    order_type=sim.order_type,
                    quantity=sim.quantity,
                    executed_qty=sim.executed_qty,
                    cumulative_quote_qty=sim.cumulative_quote_qty,
                    price=sim.avg_price,
                    dry_run=True,
                    raw={"fills": sim.fills, "sim": True},
                )
            # Dry-run: never consult a live/mock exchange for idempotency.
            if self.dry_run or not self.live_enabled:
                return None
        # Live path: ask Binance whether this client order already exists.
        try:
            got = self.exchange.get_order(symbol, client_order_id=client_order_id)
            if got and got.order_id:
                return got
        except Exception:  # noqa: BLE001
            return None
        return None

    def _submit_buy(
        self,
        *,
        symbol: str,
        quantity: float,
        client_order_id: str,
        ref_price: float,
    ) -> OrderResult:
        if self.dry_run or not self.live_enabled:
            if self.dry_broker is None:
                self.dry_broker = DryRunBroker()
            self.dry_broker.set_price(symbol, ref_price)
            sim = self.dry_broker.place_market_buy(
                symbol=symbol,
                quantity=quantity,
                client_order_id=client_order_id,
                ref_price=ref_price,
            )
            return OrderResult(
                ok=sim.status not in {"REJECTED"},
                order_id=sim.order_id,
                client_order_id=sim.client_order_id,
                status=sim.status,
                symbol=sim.symbol,
                side="BUY",
                order_type="MARKET",
                quantity=sim.quantity,
                executed_qty=sim.executed_qty,
                cumulative_quote_qty=sim.cumulative_quote_qty,
                price=sim.avg_price,
                dry_run=True,
                reason="DRY_RUN",
                raw={"fills": sim.fills, "sim": True, "blocked_reason": "DRY_RUN"},
            )

        req = OrderRequest(
            symbol=symbol,
            side="BUY",
            order_type="MARKET",
            quantity=quantity,
            client_order_id=client_order_id,
        )
        return self.exchange.place_entry(req)

    def _wait_for_buy_fill(
        self,
        order: OrderResult,
        *,
        symbol: str,
        client_order_id: str,
        ref_price: float,
        planned_qty: float,
    ) -> AggregatedFill:
        last = order
        for attempt in range(max(1, self.fill_poll_attempts)):
            if self.dry_broker and (order.dry_run or not self.live_enabled):
                sim = self.dry_broker.get_order(order_id=order.order_id, client_order_id=client_order_id)
                if sim:
                    return aggregate_fills_from_order(
                        symbol=symbol,
                        side="BUY",
                        order_id=sim.order_id,
                        client_order_id=sim.client_order_id,
                        status=sim.status,
                        executed_qty=sim.executed_qty,
                        cumulative_quote_qty=sim.cumulative_quote_qty,
                        fills_raw=sim.fills,
                        raw={"sim": True, "avg_price": sim.avg_price},
                    )
            status = str(last.status or "").upper()
            if status in {"FILLED", "PARTIALLY_FILLED", "REJECTED", "CANCELED", "CANCELLED", "EXPIRED"}:
                return aggregate_fills_from_order(
                    symbol=symbol,
                    side="BUY",
                    order_id=last.order_id,
                    client_order_id=last.client_order_id,
                    status=status,
                    executed_qty=last.executed_qty,
                    cumulative_quote_qty=last.cumulative_quote_qty,
                    fills_raw=(last.raw or {}).get("fills"),
                    raw=last.raw,
                )
            if self.fill_poll_delay_sec > 0:
                time.sleep(self.fill_poll_delay_sec)
            try:
                last = self.exchange.get_order(symbol, order_id=order.order_id, client_order_id=client_order_id)
            except Exception as e:  # noqa: BLE001
                logger.warning("fill poll failed: %s", scrub_exception(e))
        # Exhausted
        return aggregate_fills_from_order(
            symbol=symbol,
            side="BUY",
            order_id=order.order_id,
            client_order_id=client_order_id,
            status=str(last.status or "UNKNOWN"),
            executed_qty=last.executed_qty,
            cumulative_quote_qty=last.cumulative_quote_qty,
            fills_raw=(last.raw or {}).get("fills"),
            raw=last.raw,
        )

    def _verify_oco_accepted(
        self,
        *,
        symbol: str,
        trade_id: str,
        order: OrderResult | None,
        list_client_order_id: str,
        quantity: float,
        entry_price: float,
        activation: float,
        stop: float,
        trail_bips: int,
    ) -> tuple[bool, str | None]:
        if order is None:
            return False, None
        if self.dry_run or not self.live_enabled:
            if self.dry_broker is None:
                self.dry_broker = DryRunBroker()
            # If trailing already returned dry without creating sim OCO, create/verify here.
            existing = None
            if order.order_id:
                existing = self.dry_broker.get_oco(str(order.order_id))
            if existing is None and list_client_order_id in self.dry_broker._by_client:
                key = self.dry_broker._by_client[list_client_order_id]
                existing = self.dry_broker.get_oco(key)
            if existing is None:
                created = self.dry_broker.place_oco(
                    symbol=symbol,
                    quantity=quantity,
                    list_client_order_id=list_client_order_id,
                    above_stop_price=activation,
                    above_trailing_delta=trail_bips,
                    below_stop_price=stop,
                    entry_ref_price=entry_price,
                )
                if created is None:
                    return False, None
                self.db.update_trade(trade_id, binance_oco_list_id=created.order_list_id)
                order_id = created.order_list_id
            else:
                order_id = existing.order_list_id
            lists = self.dry_broker.open_order_lists(symbol)
            ok = any(str(x.get("orderListId")) == str(order_id) for x in lists)
            return ok, order_id if ok else None

        # Live: confirm list appears in open order lists; else authoritative get_order_list.
        if not order.order_id and not list_client_order_id:
            return False, None
        from binance_btc_bot.execution.order_lists import (
            find_matching_open_list,
            is_binance_param_error,
            parse_order_list,
            protection_state_from_list,
        )

        try:
            lists = self.exchange.get_open_order_lists(symbol)
        except Exception as e:  # noqa: BLE001
            err = scrub_exception(e)
            if is_binance_param_error(err):
                try:
                    lists = self.exchange.get_open_order_lists()
                except Exception as e2:  # noqa: BLE001
                    self.db.insert_event(
                        "OCO_VERIFY_FAILED",
                        symbol=symbol,
                        trade_id=trade_id,
                        reason=scrub_exception(e2),
                        payload={"fail_closed": True, "param_error": err},
                    )
                    return False, None
            else:
                self.db.insert_event(
                    "OCO_VERIFY_FAILED",
                    symbol=symbol,
                    trade_id=trade_id,
                    reason=err,
                    payload={"fail_closed": True},
                )
                return False, None

        matched = find_matching_open_list(
            lists,
            symbol=symbol,
            order_list_id=order.order_id,
            list_client_order_id=list_client_order_id,
        )
        if matched is not None and matched.is_open:
            return True, matched.order_list_id or order.order_id

        # Authoritative single-list lookup when open list miss (race / filter).
        if order.order_id and hasattr(self.exchange, "get_order_list"):
            try:
                detail = self.exchange.get_order_list(order_list_id=str(order.order_id))
                view = parse_order_list(detail)
                state = protection_state_from_list(view)
                self.db.insert_event(
                    "OCO_VERIFY_ORDER_LIST",
                    symbol=symbol,
                    trade_id=trade_id,
                    order_id=view.order_list_id,
                    reason=state,
                    payload={"listOrderStatus": view.list_order_status, "listStatusType": view.list_status_type},
                )
                if view.is_open and (
                    not symbol or view.symbol == str(symbol).upper() or not view.symbol
                ):
                    return True, view.order_list_id or order.order_id
            except Exception as e:  # noqa: BLE001
                self.db.insert_event(
                    "OCO_VERIFY_FAILED",
                    symbol=symbol,
                    trade_id=trade_id,
                    reason=scrub_exception(e),
                    payload={"fail_closed": True, "stage": "get_order_list"},
                )
                return False, None
        return False, None

    def _free_base_balance(self, base_asset: str, *, fallback: float | None = None) -> float:
        """Query Binance free BASE balance.

        On failure: dry-run may use ``fallback`` (fill-derived sellable); live fails closed to 0.
        """
        asset = str(base_asset or "").upper()
        try:
            acct = self.exchange.get_account()
            return max(0.0, float(acct.free(asset)))
        except Exception as e:  # noqa: BLE001
            logger.warning("free base balance query failed: %s", scrub_exception(e))
            if (self.dry_run or not self.live_enabled) and fallback is not None:
                return max(0.0, float(fallback))
            return 0.0

    def _submit_protection(
        self,
        *,
        trade_id: str,
        symbol: str,
        strategy: TrailStrategy,
        fill: AggregatedFill,
        events: list[str],
    ) -> tuple[
        float | None,
        ProtectableQuantity | None,
        TrailingSubmitResult | None,
        bool,
        str | None,
        list[str],
    ]:
        """Submit native OCO using fee/balance-aware qty; one corrected retry on -2010."""
        from binance_btc_bot.execution.protection_qty import sellable_qty_from_fills

        meta = self.exchange.get_symbol_info(symbol)
        base = str(meta.base_asset or "").upper()
        sellable = sellable_qty_from_fills(fill, base)
        free = self._free_base_balance(base, fallback=sellable)
        info = compute_protectable_quantity(
            fill=fill,
            base_asset=base,
            free_base=free,
            meta=meta,
            ref_price=float(fill.avg_price or 0),
        )
        self.db.insert_event(
            "PROTECT_QTY",
            symbol=symbol,
            trade_id=trade_id,
            reason=info.reason,
            payload=info.to_dict(),
        )
        events.append(f"PROTECT_QTY:{info.reason}:{info.quantity}")
        if not info.ok or info.quantity <= 0:
            return None, info, None, False, None, events

        def _one_attempt(qty: float, *, attempt: int) -> tuple[TrailingSubmitResult, bool, str | None]:
            # Distinct list client ids so a retry cannot duplicate the first OCO.
            list_cid = self._client_oco_id(trade_id) if attempt == 1 else f"r{attempt}_{trade_id[:16]}"
            oco_local = self.trailing.submit_native_trailing(
                trade_id=trade_id,
                symbol=symbol,
                strategy=strategy,
                entry_price=fill.avg_price,
                quantity=qty,
                list_client_order_id=list_cid,
            )
            events.append("OCO_SUBMITTED" if oco_local.ok else "OCO_SUBMIT_FAILED")
            verified_local = False
            list_id_local: str | None = None
            if oco_local.ok:
                verified_local, list_id_local = self._verify_oco_accepted(
                    symbol=symbol,
                    trade_id=trade_id,
                    order=oco_local.order,
                    list_client_order_id=list_cid,
                    quantity=qty,
                    entry_price=fill.avg_price,
                    activation=oco_local.activation_price
                    or (fill.avg_price * (1.0 + strategy.activation)),
                    stop=oco_local.initial_stop
                    or (fill.avg_price * (1.0 - strategy.arm_sl_activation_trail)),
                    trail_bips=oco_local.trail_bips
                    or int(round(strategy.trail_distance * 10000)),
                )
                events.append("OCO_ACCEPTED" if verified_local else "OCO_NOT_ACCEPTED")
            return oco_local, verified_local, list_id_local

        oco, verified, list_id = _one_attempt(info.quantity, attempt=1)
        if oco.ok and verified:
            return info.quantity, info, oco, True, list_id, events

        # Retry once only when Binance rejects for insufficient balance — never same qty.
        if is_insufficient_balance_reason(oco.reason):
            events.append("OCO_INSUFFICIENT_BALANCE_RETRY")
            free2 = self._free_base_balance(base, fallback=None)
            # Re-read commissions from authoritative fill object (execution events).
            info2 = compute_protectable_quantity(
                fill=fill,
                base_asset=base,
                free_base=free2,
                meta=meta,
                ref_price=float(fill.avg_price or 0),
            )
            self.db.insert_event(
                "PROTECT_QTY_RETRY",
                symbol=symbol,
                trade_id=trade_id,
                reason=info2.reason,
                payload={"first": info.to_dict(), "retry": info2.to_dict()},
            )
            if (
                info2.ok
                and info2.quantity > 0
                and abs(info2.quantity - info.quantity) > 1e-15
                and info2.quantity <= free2 + 1e-12
            ):
                oco2, verified2, list_id2 = _one_attempt(info2.quantity, attempt=2)
                if oco2.ok and verified2:
                    events.append("OCO_RETRY_OK")
                    return info2.quantity, info2, oco2, True, list_id2, events
                events.append("OCO_RETRY_FAILED")
                return info2.quantity, info2, oco2, verified2, list_id2, events
            events.append("OCO_RETRY_SKIPPED_SAME_OR_INVALID_QTY")

        return info.quantity, info, oco, verified, list_id, events

    def _handle_protection_failure(
        self,
        *,
        trade_id: str,
        reservation_id: str | None,
        symbol: str,
        fill: AggregatedFill,
        size: SizeDecision | None,
        oco: TrailingSubmitResult | None,
        events: list[str],
        reason: str,
        strategy: TrailStrategy | None = None,
    ) -> LifecycleResult:
        events.append("PROTECTION_FAILED")
        self.db.update_trade(trade_id, status=TradeStatus.PROTECTION_FAILED.value)
        self.db.insert_event(
            "PROTECTION_FAILED",
            symbol=symbol,
            trade_id=trade_id,
            reason=reason,
            payload={"fill": fill.__dict__ if fill else None, "oco_reason": oco.reason if oco else None},
        )
        # Stop new trades immediately.
        self.safety.halt("PROTECTION_FAILED", symbol=symbol, trade_id=trade_id, detail=reason)
        self._notify(
            "notify_critical",
            "PROTECTION_FAILED",
            f"BUY filled but OCO failed for {symbol}: {reason}",
            symbol=symbol,
            trade_id=trade_id,
            dedupe_key=f"protfail:{trade_id}",
        )

        from binance_btc_bot.execution.protection_qty import (
            classify_residual_base,
            sellable_qty_from_fills,
        )
        from binance_btc_bot.strategy.trails import get_strategy

        strat = strategy or get_strategy("T1")
        dust_payload: dict[str, Any] | None = None
        emerg_qty = 0.0
        try:
            meta = self.exchange.get_symbol_info(symbol)
            base = str(meta.base_asset or "").upper()
            sellable = sellable_qty_from_fills(fill, base)
            free = self._free_base_balance(base, fallback=sellable)
            pq = compute_protectable_quantity(
                fill=fill,
                base_asset=base,
                free_base=free,
                meta=meta,
                ref_price=float(fill.avg_price or 0),
            )
            emerg_qty = pq.quantity if pq.ok else 0.0
            dust = classify_residual_base(
                fill=fill,
                base_asset=base,
                free_base=free,
                meta=meta,
                ref_price=float(fill.avg_price or 0),
                protected_qty=0.0,
            )
            dust_payload = dust.to_dict()
            self.db.insert_event(
                "DUST_RESIDUAL",
                symbol=symbol,
                trade_id=trade_id,
                reason=dust.category,
                payload=dust_payload,
            )
            events.append(f"DUST:{dust.category}")
        except Exception as e:  # noqa: BLE001
            self.db.insert_event(
                "DUST_RESIDUAL",
                symbol=symbol,
                trade_id=trade_id,
                reason=scrub_exception(e),
            )

        recovered = False
        if emerg_qty > 0:
            recovered = self._emergency_protect(
                symbol=symbol,
                trade_id=trade_id,
                qty=emerg_qty,
                entry=fill.avg_price,
                strategy=strat,
                fill=fill,
            )
        events.append("EMERGENCY_OK" if recovered else "EMERGENCY_FAILED")

        # Reconcile after emergency attempt (Binance = source of truth).
        try:
            recon = self.reconcile_rest(universe=[symbol])
            events.append("RECONCILE_AFTER_EMERGENCY")
            self.db.insert_event(
                "RECONCILE_AFTER_EMERGENCY",
                symbol=symbol,
                trade_id=trade_id,
                reason="OK" if recon.get("ok") else "FAIL",
                payload=recon,
            )
        except Exception as e:  # noqa: BLE001
            events.append("RECONCILE_AFTER_EMERGENCY_FAILED")
            self.db.insert_event(
                "RECONCILE_AFTER_EMERGENCY",
                symbol=symbol,
                trade_id=trade_id,
                reason=scrub_exception(e),
            )

        if recovered:
            if reservation_id and self.portfolio:
                self.portfolio.transition(
                    reservation_id, TradeStatus.PROTECTED_EMERGENCY, trade_id=trade_id
                )
            self.db.update_trade(
                trade_id,
                status="PROTECTED_EMERGENCY",
                quantity=emerg_qty,
            )
            # Remain HALTED for new entries — existing position is protected.
            # Never mark plain PROTECTED on the emergency path.
            return LifecycleResult(
                ok=False,
                trade_id=trade_id,
                reservation_id=reservation_id,
                status="PROTECTED_EMERGENCY",
                reason="PROTECTION_FAILED_RECOVERED",
                fill=fill,
                size=size,
                oco=oco,
                dry_run=self.dry_run,
                events=events,
            )

        # Cannot protect → global HALT stays; never falsely mark PROTECTED.
        self._notify(
            "notify_critical",
            "UNPROTECTED_POSITION",
            f"UNPROTECTED POSITION {symbol} trade={trade_id} — HALTED"
            + (f" dust={dust_payload.get('category')}" if dust_payload else ""),
            symbol=symbol,
            trade_id=trade_id,
            dedupe_key=f"unprot:{trade_id}",
        )
        if reservation_id and self.portfolio:
            # Keep slot occupied for the unprotected inventory (incl. dust-only).
            self.portfolio.transition(reservation_id, TradeStatus.PROTECTION_FAILED, trade_id=trade_id)
        return LifecycleResult(
            ok=False,
            trade_id=trade_id,
            reservation_id=reservation_id,
            status=TradeStatus.PROTECTION_FAILED.value,
            reason="PROTECTION_FAILED",
            fill=fill,
            size=size,
            oco=oco,
            dry_run=self.dry_run,
            events=events,
        )

    def _emergency_client_oco_id(self, trade_id: str) -> str:
        return f"em_{trade_id[:18]}"

    def _emergency_client_stop_id(self, trade_id: str) -> str:
        return f"es_{trade_id[:18]}"

    def _find_existing_emergency_protection(
        self, *, symbol: str, trade_id: str
    ) -> tuple[bool, str | None, str]:
        """Return (found, order_id, kind) if an open protective order already exists."""
        from binance_btc_bot.execution.order_lists import find_matching_open_list

        em_oco = self._emergency_client_oco_id(trade_id)
        em_stop = self._emergency_client_stop_id(trade_id)
        try:
            lists = self.exchange.get_open_order_lists(symbol)
            hit = find_matching_open_list(
                lists, symbol=symbol, list_client_order_id=em_oco
            )
            if hit and hit.is_open:
                return True, hit.order_list_id, "OCO"
            # Also match primary / retry OCO client ids already on the book.
            for cid in (self._client_oco_id(trade_id), f"r2_{trade_id[:16]}"):
                hit2 = find_matching_open_list(
                    lists, symbol=symbol, list_client_order_id=cid
                )
                if hit2 and hit2.is_open:
                    return True, hit2.order_list_id, "OCO"
        except Exception:  # noqa: BLE001
            pass
        try:
            opens = self.exchange.get_open_orders(symbol)
            for row in opens or []:
                cid = str(row.get("clientOrderId") or "")
                if cid in {em_stop, em_oco} and str(row.get("side") or "").upper() == "SELL":
                    return True, str(row.get("orderId")), "STOP_LOSS"
        except Exception:  # noqa: BLE001
            pass
        return False, None, ""

    def _emergency_protect(
        self,
        *,
        symbol: str,
        trade_id: str,
        qty: float,
        entry: float,
        strategy: TrailStrategy | None = None,
        fill: AggregatedFill | None = None,
    ) -> bool:
        """Production emergency protection after OCO failure.

        Order of preference (no new strategy, no improvisation):
        1. If an open protective list/order already exists → succeed (no duplicate).
        2. Native trail OCO via existing TrailingExecutor (same T1–T10 geometry).
        3. Contingent STOP_LOSS SELL via ``place_protective_sell`` at strategy hard SL
           (explicit SELL/protection semantics — never MARKET, never entry path).

        Never marks the trade PROTECTED (caller uses PROTECTED_EMERGENCY only).
        Does not reserve slots, allocate equity, evaluate signals, or place BUY.
        """
        from binance_btc_bot.execution.protection_qty import (
            classify_residual_base,
            sellable_qty_from_fills,
        )
        from binance_btc_bot.strategy.trails import get_strategy

        strat = strategy or get_strategy("T1")
        try:
            # Idempotency: do not submit a second emergency order for same trade.
            if trade_id in self._emergency_submitted:
                found, oid, kind = self._find_existing_emergency_protection(
                    symbol=symbol, trade_id=trade_id
                )
                self.db.insert_event(
                    "EMERGENCY_DUPLICATE_SKIPPED",
                    symbol=symbol,
                    trade_id=trade_id,
                    order_id=oid,
                    reason=kind or "ALREADY_ATTEMPTED",
                )
                return bool(found)

            found, oid, kind = self._find_existing_emergency_protection(
                symbol=symbol, trade_id=trade_id
            )
            if found:
                self.db.update_trade(trade_id, binance_oco_list_id=oid if kind == "OCO" else None)
                self.db.insert_event(
                    "EMERGENCY_PROTECTION",
                    symbol=symbol,
                    trade_id=trade_id,
                    order_id=oid,
                    reason=f"EXISTING_{kind}",
                )
                return True

            # Refresh account + commission-aware qty (same logic as primary protect).
            meta = self.exchange.get_symbol_info(symbol)
            base = str(meta.base_asset or "").upper()
            if fill is None:
                self.db.insert_event(
                    "EMERGENCY_PROTECTION_FAILED",
                    symbol=symbol,
                    trade_id=trade_id,
                    reason="MISSING_FILL",
                )
                return False
            sellable = sellable_qty_from_fills(fill, base)
            free = self._free_base_balance(base, fallback=sellable)
            pq = compute_protectable_quantity(
                fill=fill,
                base_asset=base,
                free_base=free,
                meta=meta,
                ref_price=float(entry or fill.avg_price or 0),
            )
            if not pq.ok or pq.quantity <= 0:
                dust = classify_residual_base(
                    fill=fill,
                    base_asset=base,
                    free_base=free,
                    meta=meta,
                    ref_price=float(entry or fill.avg_price or 0),
                )
                self.db.insert_event(
                    "EMERGENCY_PROTECTION_FAILED",
                    symbol=symbol,
                    trade_id=trade_id,
                    reason=pq.reason,
                    payload={"dust": dust.to_dict()},
                )
                return False

            use_qty = pq.quantity

            if self.dry_run or not self.live_enabled:
                if self.dry_broker is None:
                    self.dry_broker = DryRunBroker()
                if not self.dry_broker.allow_emergency_protect:
                    self.db.insert_event(
                        "EMERGENCY_PROTECTION_FAILED",
                        symbol=symbol,
                        trade_id=trade_id,
                        reason="EMERGENCY_DISABLED",
                    )
                    return False
                self._emergency_submitted.add(trade_id)
                prev = self.dry_broker.oco_behavior
                self.dry_broker.oco_behavior = "ACCEPT"
                oco = self.dry_broker.place_oco(
                    symbol=symbol,
                    quantity=use_qty,
                    list_client_order_id=self._emergency_client_oco_id(trade_id),
                    above_stop_price=entry * (1.0 + strat.activation),
                    above_trailing_delta=int(round(strat.trail_distance * 10000)),
                    below_stop_price=entry * (1.0 - strat.arm_sl_activation_trail),
                    entry_ref_price=entry,
                )
                self.dry_broker.oco_behavior = prev
                ok = oco is not None
                if ok:
                    self.db.update_trade(trade_id, binance_oco_list_id=oco.order_list_id, quantity=use_qty)
                    self.db.insert_event(
                        "EMERGENCY_PROTECTION",
                        symbol=symbol,
                        trade_id=trade_id,
                        order_id=oco.order_list_id,
                        reason="DRY_OCO",
                        payload={"qty": use_qty},
                    )
                    if fill is not None:
                        dust = classify_residual_base(
                            fill=fill,
                            base_asset=base,
                            free_base=free,
                            meta=meta,
                            ref_price=float(entry),
                            protected_qty=use_qty,
                        )
                        self.db.insert_event(
                            "DUST_RESIDUAL",
                            symbol=symbol,
                            trade_id=trade_id,
                            reason=dust.category,
                            payload=dust.to_dict(),
                        )
                return ok

            # ---- Live path ----
            self._emergency_submitted.add(trade_id)

            # Prefer native T1 OCO via existing TrailingExecutor.
            list_cid = self._emergency_client_oco_id(trade_id)
            oco_res = self.trailing.submit_native_trailing(
                trade_id=trade_id,
                symbol=symbol,
                strategy=strat,
                entry_price=float(entry),
                quantity=use_qty,
                list_client_order_id=list_cid,
            )
            if oco_res.ok and oco_res.order:
                verified, list_id = self._verify_oco_accepted(
                    symbol=symbol,
                    trade_id=trade_id,
                    order=oco_res.order,
                    list_client_order_id=list_cid,
                    quantity=use_qty,
                    entry_price=float(entry),
                    activation=oco_res.activation_price
                    or (float(entry) * (1.0 + strat.activation)),
                    stop=oco_res.initial_stop
                    or (float(entry) * (1.0 - strat.arm_sl_activation_trail)),
                    trail_bips=oco_res.trail_bips
                    or int(round(strat.trail_distance * 10000)),
                )
                if verified:
                    self.db.update_trade(
                        trade_id, binance_oco_list_id=list_id, quantity=use_qty
                    )
                    self.db.insert_event(
                        "EMERGENCY_PROTECTION",
                        symbol=symbol,
                        trade_id=trade_id,
                        order_id=list_id,
                        reason="EMERGENCY_OCO",
                        payload={"qty": use_qty},
                    )
                    if fill is not None:
                        dust = classify_residual_base(
                            fill=fill,
                            base_asset=base,
                            free_base=free,
                            meta=meta,
                            ref_price=float(entry),
                            protected_qty=use_qty,
                        )
                        self.db.insert_event(
                            "DUST_RESIDUAL",
                            symbol=symbol,
                            trade_id=trade_id,
                            reason=dust.category,
                            payload=dust.to_dict(),
                        )
                    return True

            # Fallback: contingent STOP_LOSS via place_protective_sell (not entry path).
            # Do NOT market-sell — only STOP_LOSS if the symbol supports it.
            if "STOP_LOSS" not in {str(x).upper() for x in (meta.order_types or ())}:
                self.db.insert_event(
                    "EMERGENCY_PROTECTION_FAILED",
                    symbol=symbol,
                    trade_id=trade_id,
                    reason="STOP_LOSS_NOT_SUPPORTED",
                    payload={"oco_reason": oco_res.reason if oco_res else None},
                )
                return False

            from binance_btc_bot.strategy.trails import _floor_to_tick

            stop_px = float(entry) * (1.0 - float(strat.arm_sl_activation_trail))
            if meta.price_tick > 0:
                stop_px = _floor_to_tick(stop_px, meta.price_tick)
            stop_cid = self._emergency_client_stop_id(trade_id)
            # Idempotent: existing client order?
            try:
                existing = self.exchange.get_order(symbol, client_order_id=stop_cid)
                if (
                    isinstance(existing, OrderResult)
                    and existing.order_id
                    and str(existing.status or "").upper()
                    not in {
                        "CANCELED",
                        "CANCELLED",
                        "REJECTED",
                        "EXPIRED",
                        "UNKNOWN",
                        "",
                    }
                ):
                    self.db.insert_event(
                        "EMERGENCY_PROTECTION",
                        symbol=symbol,
                        trade_id=trade_id,
                        order_id=existing.order_id,
                        reason="EXISTING_STOP_LOSS",
                    )
                    return True
            except Exception:  # noqa: BLE001
                pass

            req = OrderRequest(
                symbol=symbol,
                side="SELL",
                order_type="STOP_LOSS",
                quantity=use_qty,
                stop_price=stop_px,
                client_order_id=stop_cid,
            )
            place_fn = getattr(self.exchange, "place_protective_sell", None)
            if callable(place_fn):
                placed = place_fn(req)
            else:
                # Fail closed if adapter lacks explicit protective API.
                self.db.insert_event(
                    "EMERGENCY_PROTECTION_FAILED",
                    symbol=symbol,
                    trade_id=trade_id,
                    reason="NO_PROTECTIVE_SELL_ADAPTER",
                )
                return False
            if not isinstance(placed, OrderResult) or not placed.ok or placed.dry_run:
                self.db.insert_event(
                    "EMERGENCY_PROTECTION_FAILED",
                    symbol=symbol,
                    trade_id=trade_id,
                    reason=getattr(placed, "reason", None) or "STOP_SUBMIT_FAILED",
                    payload={"raw": getattr(placed, "raw", None)},
                )
                return False

            # Verify Binance accepted the protective STOP.
            verified_stop = False
            try:
                got = self.exchange.get_order(
                    symbol, order_id=placed.order_id, client_order_id=stop_cid
                )
                if isinstance(got, OrderResult) and got.order_id:
                    st = str(got.status or "").upper()
                    verified_stop = st not in {
                        "REJECTED",
                        "CANCELED",
                        "CANCELLED",
                        "EXPIRED",
                    }
            except Exception:  # noqa: BLE001
                verified_stop = False
            if not verified_stop:
                # Fail closed on ambiguous verify — also check open orders.
                try:
                    opens = self.exchange.get_open_orders(symbol)
                    verified_stop = any(
                        str(o.get("orderId")) == str(placed.order_id)
                        or str(o.get("clientOrderId") or "") == stop_cid
                        for o in (opens or [])
                        if isinstance(o, dict)
                    )
                except Exception:  # noqa: BLE001
                    verified_stop = False

            if not verified_stop:
                self.db.insert_event(
                    "EMERGENCY_PROTECTION_FAILED",
                    symbol=symbol,
                    trade_id=trade_id,
                    order_id=placed.order_id,
                    reason="STOP_NOT_VERIFIED",
                )
                return False

            self.db.insert_order(
                trade_id=trade_id,
                symbol=symbol,
                order_id=placed.order_id,
                client_order_id=stop_cid,
                side="SELL",
                order_type="STOP_LOSS",
                status=placed.status,
                payload={"raw": placed.raw, "mode": "EMERGENCY_STOP_LOSS", "stop_price": stop_px},
            )
            self.db.update_trade(trade_id, quantity=use_qty)
            self.db.insert_event(
                "EMERGENCY_PROTECTION",
                symbol=symbol,
                trade_id=trade_id,
                order_id=placed.order_id,
                reason="EMERGENCY_STOP_LOSS",
                payload={"qty": use_qty, "stop_price": stop_px},
            )
            if fill is not None:
                dust = classify_residual_base(
                    fill=fill,
                    base_asset=base,
                    free_base=free,
                    meta=meta,
                    ref_price=float(entry),
                    protected_qty=use_qty,
                )
                self.db.insert_event(
                    "DUST_RESIDUAL",
                    symbol=symbol,
                    trade_id=trade_id,
                    reason=dust.category,
                    payload=dust.to_dict(),
                )
            return True
        except Exception as e:  # noqa: BLE001
            self.db.insert_event(
                "EMERGENCY_PROTECTION_FAILED",
                symbol=symbol,
                trade_id=trade_id,
                reason=scrub_exception(e),
            )
            return False

    def _reservation_id_from_trade(self, tr: Mapping[str, Any] | dict[str, Any]) -> str | None:
        try:
            cfg = tr.get("strategy_config_json")
            sc = json.loads(cfg) if isinstance(cfg, str) else (cfg or {})
            rid = sc.get("reservation_id")
            return str(rid) if rid else None
        except Exception:  # noqa: BLE001
            return None

    def _base_inventory(self, symbol: str) -> tuple[float, float, float] | None:
        """Return (free, locked, total) for the symbol base asset, or None on API fail."""
        try:
            meta = self.exchange.get_symbol_info(symbol)
            base = str(getattr(meta, "base_asset", None) or "").upper()
            if not base:
                return None
            acct = self.exchange.get_account()
            bal = acct.balances.get(base) if getattr(acct, "balances", None) else None
            free = float(bal.free) if bal else 0.0
            locked = float(bal.locked) if bal else 0.0
            return (free, locked, free + locked)
        except Exception as e:  # noqa: BLE001
            logger.warning("base inventory check failed: %s", scrub_exception(e))
            return None

    def _inventory_economically_flat(
        self, symbol: str, *, inv: tuple[float, float, float] | None = None
    ) -> tuple[bool | None, str]:
        """Flat for reconcile: zero OR unsellable dust (minQty/minNotional/LOT_SIZE).

        Returns (flat?, reason). None flat = ambiguous API failure (fail closed).
        """
        from binance_btc_bot.execution.protection_qty import is_economically_flat_base

        try:
            meta = self.exchange.get_symbol_info(symbol)
            if inv is None:
                inv = self._base_inventory(symbol)
            if inv is None:
                return None, "INVENTORY_UNKNOWN"
            free, locked, _total = inv
            ref = 0.0
            try:
                ref = float(self.exchange.get_price(symbol) or 0)
            except Exception:  # noqa: BLE001
                try:
                    ref = float(getattr(meta, "price_tick", 0) or 0)
                except Exception:  # noqa: BLE001
                    ref = 0.0
            flat, reason = is_economically_flat_base(
                free=float(free), locked=float(locked), meta=meta, ref_price=ref
            )
            return flat, reason
        except Exception as e:  # noqa: BLE001
            logger.warning("economic flat check failed: %s", scrub_exception(e))
            return None, scrub_exception(e)
    @staticmethod
    def _map_exit_close_reason(
        order_type: str | None,
        *,
        trailing_delta: Any = None,
        strategy_has_trail: bool = False,
    ) -> str:
        ot = str(order_type or "").upper()
        if ot in {"TAKE_PROFIT", "TAKE_PROFIT_LIMIT"}:
            if trailing_delta not in (None, "", 0, "0") or strategy_has_trail:
                return "TRAILING_EXIT"
            return "TAKE_PROFIT"
        if ot in {"STOP_LOSS", "STOP_LOSS_LIMIT"}:
            return "HARD_SL"
        return "UNKNOWN"

    @staticmethod
    def _ms_to_iso_z(ms: Any) -> str | None:
        try:
            t_ms = int(ms)
            return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_ms / 1000.0))
        except Exception:  # noqa: BLE001
            return None

    def _discover_oco_exit_evidence(self, tr: dict[str, Any]) -> dict[str, Any] | None:
        """Aggregate authoritative OCO exit evidence from orderList + myTrades.

        Returns None when exit cannot be confirmed (still open / no sells / incomplete).
        """
        from binance_btc_bot.execution.order_lists import parse_order_list

        symbol = str(tr.get("symbol") or "").upper()
        trade_id = str(tr.get("trade_id") or "")
        oco_id = tr.get("binance_oco_list_id")
        try:
            quantity = float(tr.get("quantity") or 0)
        except (TypeError, ValueError):
            quantity = 0.0
        try:
            entry = float(tr.get("entry_price") or 0)
        except (TypeError, ValueError):
            entry = 0.0
        if not symbol or not trade_id or not oco_id or quantity <= 0 or entry <= 0:
            return None

        entry_order_id = str(tr.get("binance_entry_order_id") or "")
        strategy_has_trail = float(tr.get("configured_trailing_distance") or 0) > 0

        try:
            raw_list = self.exchange.get_order_list(order_list_id=str(oco_id))
        except Exception as e:  # noqa: BLE001
            logger.warning("get_order_list failed for %s: %s", oco_id, scrub_exception(e))
            return None

        if not isinstance(raw_list, dict):
            return None
        view = parse_order_list(raw_list)
        los = str(view.list_order_status or "").upper()
        lst = str(view.list_status_type or "").upper()
        list_done = (not view.is_open) or los == "ALL_DONE" or lst == "ALL_DONE"
        if not list_done:
            return None

        child_ids: set[str] = set()
        for c in view.child_orders:
            if c.get("orderId") is not None:
                child_ids.add(str(c["orderId"]))

        # orderReports often carry type/status/trailingDelta for completed lists.
        reports: list[dict[str, Any]] = []
        raw_reports = raw_list.get("orderReports") or ()
        if isinstance(raw_reports, (list, tuple)):
            for r in raw_reports:
                if isinstance(r, dict):
                    reports.append(r)
                    if r.get("orderId") is not None:
                        child_ids.add(str(r["orderId"]))

        try:
            trades = self.exchange.get_my_trades(symbol, limit=100) or []
        except Exception as e:  # noqa: BLE001
            logger.warning("get_my_trades failed for %s: %s", symbol, scrub_exception(e))
            trades = []

        sells: list[dict[str, Any]] = []
        for t in trades:
            if not isinstance(t, dict):
                continue
            if t.get("isBuyer") is True:
                continue
            oid = str(t.get("orderId") or "")
            if entry_order_id and oid == entry_order_id:
                continue
            sells.append(t)

        preferred = [t for t in sells if str(t.get("orderId") or "") in child_ids] if child_ids else []
        use = preferred if preferred else sells
        if not use:
            # Flat + ALL_DONE confirmed but no sell fills — caller may close without inventing PnL.
            return {
                "exit_price": None,
                "exit_qty": None,
                "fees_btc": 0.0,
                "fees_usdt": 0.0,
                "close_reason": "UNKNOWN",
                "exit_order_id": None,
                "exit_time": None,
                "commission_asset": None,
                "commission_amount": None,
                "reconciliation_source": "REST_ORDER_LIST",
                "list_all_done": True,
                "incomplete_fills": True,
            }

        qty_sum = 0.0
        notional = 0.0
        fees_btc = 0.0
        fees_usdt = 0.0
        commission_asset: str | None = None
        commission_amount = 0.0
        exit_order_id: str | None = None
        exit_time_ms: int | None = None
        for t in use:
            try:
                q = float(t.get("qty") or 0)
                px = float(t.get("price") or 0)
            except (TypeError, ValueError):
                continue
            if q <= 0 or px <= 0:
                continue
            qty_sum += q
            notional += q * px
            try:
                c = float(t.get("commission") or 0)
            except (TypeError, ValueError):
                c = 0.0
            asset = str(t.get("commissionAsset") or "").upper()
            if asset == "BTC":
                fees_btc += c
            elif asset == "USDT":
                fees_usdt += c
            if c and asset:
                commission_asset = asset
                commission_amount += c
            if t.get("orderId") is not None:
                exit_order_id = str(t["orderId"])
            try:
                tm = int(t.get("time") or 0)
                if tm and (exit_time_ms is None or tm > exit_time_ms):
                    exit_time_ms = tm
            except (TypeError, ValueError):
                pass

        if qty_sum <= 0 or notional <= 0:
            return None
        avg_price = notional / qty_sum

        # Infer close reason from filled child report / get_order.
        close_reason = "UNKNOWN"
        filled_type: str | None = None
        trailing_delta = None
        for r in reports:
            if str(r.get("status") or "").upper() != "FILLED":
                continue
            if str(r.get("side") or "").upper() not in {"", "SELL"}:
                continue
            filled_type = str(r.get("type") or r.get("orderType") or "")
            trailing_delta = r.get("trailingDelta")
            break
        if not filled_type and exit_order_id:
            try:
                got = self.exchange.get_order(symbol, order_id=str(exit_order_id))
                filled_type = getattr(got, "order_type", None) if got else None
                raw = getattr(got, "raw", None) or {}
                if isinstance(raw, dict):
                    trailing_delta = raw.get("trailingDelta")
            except Exception:  # noqa: BLE001
                filled_type = None
        close_reason = self._map_exit_close_reason(
            filled_type,
            trailing_delta=trailing_delta,
            strategy_has_trail=strategy_has_trail and str(filled_type or "").upper().startswith("TAKE_PROFIT"),
        )

        source = "REST_MY_TRADES" if preferred or sells else "REST_ORDER_LIST"
        return {
            "exit_price": avg_price,
            "exit_qty": qty_sum,
            "fees_btc": fees_btc,
            "fees_usdt": fees_usdt,
            "close_reason": close_reason,
            "exit_order_id": exit_order_id,
            "exit_time": self._ms_to_iso_z(exit_time_ms) if exit_time_ms else None,
            "commission_asset": commission_asset,
            "commission_amount": commission_amount if commission_asset else None,
            "reconciliation_source": source,
            "list_all_done": True,
            "incomplete_fills": False,
        }

    def reconcile_protected_flat_exits(self, universe: list[str] | None = None) -> dict[str, Any]:
        """Close local PROTECTED* trades that are flat on Binance with OCO exit evidence."""
        notes: list[str] = []
        closed: list[str] = []
        protected = {"PROTECTED", "PROTECTED_EMERGENCY", "DRY_RUN_PROTECTED"}
        for tr in list(self.db.open_trades()):
            local_status = str(tr.get("status") or "").upper()
            if local_status not in protected:
                continue
            sym = str(tr.get("symbol") or "").upper()
            tid = str(tr.get("trade_id") or "")
            if universe and sym not in {u.upper() for u in universe}:
                continue

            inv = self._base_inventory(sym)
            flat_ex = self._exchange_base_is_flat(sym)
            econ_flat, econ_reason = self._inventory_economically_flat(sym, inv=inv)
            is_flat = bool(flat_ex is True) or (econ_flat is True)
            if not is_flat:
                notes.append(
                    f"STILL_HAS_INVENTORY:{tid}:{sym}:econ={econ_reason}:inv={inv}"
                )
                continue
            if econ_flat is True and econ_reason not in {"ZERO", ""}:
                notes.append(f"ECONOMIC_FLAT_DUST:{tid}:{sym}:{econ_reason}")

            evidence = self._discover_oco_exit_evidence(tr)
            if evidence is None:
                # Inventory flat but cannot confirm OCO done / exit — leave open, no warn here.
                notes.append(f"FLAT_NO_OCO_EVIDENCE:{tid}:{sym}")
                continue

            rid = self._reservation_id_from_trade(tr)
            incomplete = bool(evidence.get("incomplete_fills"))
            exit_px = evidence.get("exit_price")
            if incomplete and exit_px is None and evidence.get("list_all_done"):
                # Flat + ALL_DONE, no sell fills found — close without inventing PnL.
                result = self.close_from_exit_fill(
                    trade_id=tid,
                    reservation_id=rid,
                    exit_price=None,
                    exit_qty=None,
                    fees_btc=0.0,
                    fees_usdt=0.0,
                    close_reason=str(evidence.get("close_reason") or "UNKNOWN"),
                    exit_order_id=evidence.get("exit_order_id"),
                    exit_time=evidence.get("exit_time"),
                    commission_asset=evidence.get("commission_asset"),
                    reconciliation={
                        "source": evidence.get("reconciliation_source") or "REST_ORDER_LIST",
                        "binance": "ALL_DONE_FLAT_NO_FILLS",
                        "local_db": "PASS",
                    },
                )
            elif exit_px is None:
                notes.append(f"FLAT_INCOMPLETE_EXIT:{tid}:{sym}")
                continue
            else:
                result = self.close_from_exit_fill(
                    trade_id=tid,
                    reservation_id=rid,
                    exit_price=float(exit_px),
                    exit_qty=float(evidence["exit_qty"]) if evidence.get("exit_qty") is not None else None,
                    fees_btc=float(evidence.get("fees_btc") or 0),
                    fees_usdt=float(evidence.get("fees_usdt") or 0),
                    close_reason=str(evidence.get("close_reason") or "UNKNOWN"),
                    exit_order_id=evidence.get("exit_order_id"),
                    exit_time=evidence.get("exit_time"),
                    commission_asset=evidence.get("commission_asset"),
                    reconciliation={
                        "source": evidence.get("reconciliation_source") or "REST_MY_TRADES",
                        "binance": "PASS",
                        "local_db": "PASS",
                    },
                )
            notes.append(f"FLAT_EXIT_CLOSED:{tid}:{sym}:{result.reason}")
            if result.ok and result.reason != "ALREADY_CLOSED":
                closed.append(tid)
            elif result.ok and result.reason == "ALREADY_CLOSED":
                notes.append(f"ALREADY_CLOSED:{tid}")
        return {"ok": True, "closed": closed, "notes": notes, "dry_run": self.dry_run}

    def handle_user_data_event(self, event: dict[str, Any]) -> dict[str, Any] | None:
        """Process Binance user-data WS events → close local trades. Never places orders."""
        if not isinstance(event, dict):
            return None
        et = str(event.get("e") or event.get("type") or "")

        if et == "executionReport":
            side = str(event.get("S") or event.get("side") or "").upper()
            status = str(event.get("X") or event.get("status") or "").upper()
            if side != "SELL":
                return {"ok": True, "handled": False, "reason": "NOT_SELL"}
            if status != "FILLED":
                return {"ok": True, "handled": False, "reason": "NOT_FILLED"}
            symbol = str(event.get("s") or event.get("symbol") or "").upper()
            if not symbol:
                return {"ok": False, "handled": False, "reason": "NO_SYMBOL"}
            list_id = event.get("g") if event.get("g") not in (None, -1, "-1") else None
            tr = None
            if list_id is not None:
                for row in self.db.open_trades():
                    if str(row.get("binance_oco_list_id") or "") == str(list_id):
                        tr = row
                        break
            if tr is None:
                tr = self.db.find_open_trade_for_symbol(symbol)
            if tr is None and list_id is not None:
                # Idempotent path: trade may already be CLOSED.
                try:
                    cur = self.db._conn.execute(
                        "SELECT * FROM trades WHERE binance_oco_list_id=? ORDER BY updated_at DESC LIMIT 1",
                        (str(list_id),),
                    )
                    row = cur.fetchone()
                    if row:
                        tr = dict(row)
                except Exception:  # noqa: BLE001
                    tr = None
            if not tr:
                return {"ok": True, "handled": False, "reason": "NO_OPEN_TRADE", "symbol": symbol}

            try:
                last_px = float(event.get("L") or event.get("p") or event.get("price") or 0)
            except (TypeError, ValueError):
                last_px = 0.0
            try:
                last_qty = float(event.get("l") or event.get("z") or event.get("q") or tr.get("quantity") or 0)
            except (TypeError, ValueError):
                last_qty = float(tr.get("quantity") or 0)
            if last_px <= 0:
                return {"ok": False, "handled": False, "reason": "NO_EXIT_PRICE"}

            fees_btc = 0.0
            fees_usdt = 0.0
            commission_asset = event.get("N") or event.get("commissionAsset")
            try:
                commission = float(event.get("n") or event.get("commission") or 0)
            except (TypeError, ValueError):
                commission = 0.0
            asset_u = str(commission_asset or "").upper()
            if asset_u == "BTC":
                fees_btc = commission
            elif asset_u == "USDT":
                fees_usdt = commission

            exit_order_id = event.get("i") or event.get("orderId")
            exit_time = self._ms_to_iso_z(event.get("T") or event.get("E") or event.get("time"))
            otype = str(event.get("o") or event.get("orderType") or "")
            trailing_delta = event.get("d") or event.get("trailingDelta")
            close_reason = self._map_exit_close_reason(
                otype,
                trailing_delta=trailing_delta,
                strategy_has_trail=float(tr.get("configured_trailing_distance") or 0) > 0
                and str(otype).upper().startswith("TAKE_PROFIT"),
            )
            result = self.close_from_exit_fill(
                trade_id=str(tr["trade_id"]),
                reservation_id=self._reservation_id_from_trade(tr),
                exit_price=last_px,
                exit_qty=last_qty if last_qty > 0 else None,
                fees_btc=fees_btc,
                fees_usdt=fees_usdt,
                close_reason=close_reason,
                exit_order_id=str(exit_order_id) if exit_order_id is not None else None,
                exit_time=exit_time,
                commission_asset=str(commission_asset) if commission_asset else None,
                reconciliation={"source": "USER_DATA_WS", "binance": "PASS", "local_db": "PASS"},
            )
            return {
                "ok": result.ok,
                "handled": True,
                "event": et,
                "trade_id": result.trade_id,
                "reason": result.reason,
                "events": result.events,
            }

        if et == "listStatus":
            # Binance: "l" = listStatusType, "L" = listOrderStatus
            list_order_status = str(event.get("L") or event.get("listOrderStatus") or "").upper()
            list_status_type = str(event.get("l") or event.get("listStatusType") or "").upper()
            done = "ALL_DONE" in {list_order_status, list_status_type}
            if not done:
                return {"ok": True, "handled": False, "reason": "LIST_NOT_DONE"}
            list_id = event.get("g") or event.get("orderListId")
            symbol = str(event.get("s") or event.get("symbol") or "").upper()
            tr = None
            if list_id is not None:
                for row in self.db.open_trades():
                    if str(row.get("binance_oco_list_id") or "") == str(list_id):
                        tr = row
                        break
            if tr is None and symbol:
                tr = self.db.find_open_trade_for_symbol(symbol)
            if not tr:
                return {"ok": True, "handled": False, "reason": "NO_OPEN_TRADE"}
            out = self.reconcile_protected_flat_exits(universe=[str(tr.get("symbol") or symbol)])
            return {
                "ok": True,
                "handled": True,
                "event": et,
                "trade_id": tr.get("trade_id"),
                "closed": out.get("closed") or [],
                "notes": out.get("notes") or [],
            }

        return {"ok": True, "handled": False, "reason": f"IGNORED:{et or 'EMPTY'}"}

    def _has_reconciled_flat_event(self, trade_id: str) -> bool:
        cur = self.db._conn.execute(
            "SELECT 1 FROM bot_events WHERE trade_id=? AND event=? LIMIT 1",
            (trade_id, "RECONCILED_FLAT"),
        )
        return cur.fetchone() is not None

    def _exchange_base_is_flat(self, symbol: str) -> bool | None:
        """Return True if base free+locked=0 and no open orders/lists for symbol.

        None = ambiguous / API failure (fail closed — do not close local).
        """
        try:
            meta = self.exchange.get_symbol_info(symbol)
            base = str(meta.base_asset or "").upper()
            if not base:
                return None
            acct = self.exchange.get_account()
            bal = acct.balances.get(base)
            free = float(bal.free) if bal else 0.0
            locked = float(bal.locked) if bal else 0.0
            if free > 1e-12 or locked > 1e-12:
                return False
            opens = self.exchange.get_open_orders(symbol) or []
            if any(str(o.get("symbol") or "").upper() == str(symbol).upper() for o in opens if isinstance(o, dict)):
                return False
            # get_open_orders(symbol) already filters; empty is fine.
            if opens:
                return False
            lists = self.exchange.get_open_order_lists(symbol) or []
            if lists:
                return False
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("exchange flat check failed: %s", scrub_exception(e))
            return None

    def close_reconciled_flat(
        self,
        *,
        trade_id: str,
        evidence: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Close a local PROTECTION_FAILED trade when Binance inventory is already flat.

        Does not place orders, fabricate exits, or invent realized P/L.
        Idempotent: second call is a no-op when already CLOSED + RECONCILED_FLAT.
        """
        row = self.db.get_trade(trade_id)
        if not row:
            return {"ok": False, "changed": False, "reason": "TRADE_NOT_FOUND"}
        status = str(row.get("status") or "").upper()
        if status == "CLOSED" and self._has_reconciled_flat_event(trade_id):
            return {
                "ok": True,
                "changed": False,
                "reason": "ALREADY_RECONCILED_FLAT",
                "trade_id": trade_id,
                "status": "CLOSED",
            }
        if status != "PROTECTION_FAILED":
            return {
                "ok": False,
                "changed": False,
                "reason": f"UNEXPECTED_STATUS:{status}",
                "trade_id": trade_id,
            }
        if self._has_reconciled_flat_event(trade_id):
            # Event exists but status not CLOSED — repair status only once.
            now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self.db.update_trade(trade_id, status="CLOSED", exit_time=now)
            if self.portfolio:
                self.portfolio.release_symbol(str(row.get("symbol") or ""))
            return {
                "ok": True,
                "changed": True,
                "reason": "STATUS_REPAIRED_RECONCILED_FLAT",
                "trade_id": trade_id,
            }

        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        # Preserve historical entry/qty/fees; do not invent exit_price or realized PnL.
        self.db.update_trade(
            trade_id,
            status="CLOSED",
            exit_time=now,
        )
        payload = {
            "prior_status": "PROTECTION_FAILED",
            "reconcile_reason": "EXCHANGE_FLAT_AFTER_PROTECTION_FAILED",
            "no_fabricated_exit": True,
            "no_binance_order": True,
            "historical_quantity": row.get("quantity"),
            "historical_entry_price": row.get("entry_price"),
            "historical_btc_value": row.get("btc_value"),
            "historical_fees_btc": row.get("fees_btc"),
            "binance_entry_order_id": row.get("binance_entry_order_id"),
            "binance_oco_list_id": row.get("binance_oco_list_id"),
            "evidence": evidence or {},
        }
        self.db.insert_event(
            "RECONCILED_FLAT",
            symbol=row.get("symbol"),
            trade_id=trade_id,
            reason="EXCHANGE_FLAT_AFTER_PROTECTION_FAILED",
            payload=payload,
        )
        # Keep PROTECTION_FAILED incident visible in event stream.
        self.db.insert_event(
            "INCIDENT_PRESERVED",
            symbol=row.get("symbol"),
            trade_id=trade_id,
            reason="PRIOR_PROTECTION_FAILED_RECONCILED",
            payload={
                "original_status": "PROTECTION_FAILED",
                "quantity": row.get("quantity"),
                "entry_price": row.get("entry_price"),
            },
        )
        if self.portfolio:
            self.portfolio.release_symbol(str(row.get("symbol") or ""))
        return {
            "ok": True,
            "changed": True,
            "reason": "RECONCILED_FLAT",
            "trade_id": trade_id,
            "status": "CLOSED",
            "payload": payload,
        }

    def reconcile_stale_protection_failed(
        self, universe: list[str] | None = None
    ) -> dict[str, Any]:
        """Close local PROTECTION_FAILED trades when Binance base inventory is flat."""
        notes: list[str] = []
        closed: list[str] = []
        skipped: list[str] = []
        for tr in self.db.open_trades():
            if str(tr.get("status") or "").upper() != "PROTECTION_FAILED":
                continue
            sym = str(tr.get("symbol") or "").upper()
            tid = str(tr.get("trade_id") or "")
            if universe and sym not in {u.upper() for u in universe}:
                continue
            flat = self._exchange_base_is_flat(sym)
            if flat is None:
                notes.append(f"FAIL_CLOSED_FLAT_CHECK:{tid}:{sym}")
                skipped.append(tid)
                continue
            if not flat:
                notes.append(f"STILL_HAS_INVENTORY:{tid}:{sym}")
                skipped.append(tid)
                continue
            evidence = {
                "symbol": sym,
                "exchange_flat": True,
                "checked_open_orders": True,
                "checked_open_lists": True,
            }
            result = self.close_reconciled_flat(trade_id=tid, evidence=evidence)
            notes.append(f"{result.get('reason')}:{tid}:{sym}")
            if result.get("changed"):
                closed.append(tid)
            elif result.get("reason") == "ALREADY_RECONCILED_FLAT":
                skipped.append(tid)
        return {
            "ok": True,
            "closed": closed,
            "skipped": skipped,
            "notes": notes,
            "dry_run": self.dry_run,
        }

    def reconcile_rest(self, universe: list[str] | None = None) -> dict[str, Any]:
        """REST reconciliation after WS disconnect / restart (Binance = source of truth)."""
        notes: list[str] = []
        closed: list[str] = []
        if self.dry_run or not self.live_enabled:
            if self.dry_broker:
                lists = self.dry_broker.open_order_lists()
                notes.append(f"dry_open_ocos={len(lists)}")
                # Detect exits that completed while disconnected.
                for oco in self.dry_broker.all_ocos():
                    if oco.list_order_status != "ALL_DONE" or not oco.exit_fill:
                        continue
                    cur = self.db._conn.execute(
                        "SELECT * FROM trades WHERE binance_oco_list_id=? AND status != 'CLOSED'",
                        (oco.order_list_id,),
                    )
                    row = cur.fetchone()
                    if not row:
                        continue
                    tr = dict(row)
                    rid = None
                    try:
                        cfg = tr.get("strategy_config_json")
                        sc = json.loads(cfg) if isinstance(cfg, str) else (cfg or {})
                        rid = sc.get("reservation_id")
                    except Exception:  # noqa: BLE001
                        rid = None
                    self.close_from_exit_fill(
                        trade_id=tr["trade_id"],
                        reservation_id=rid,
                        exit_price=float(oco.exit_fill["price"]),
                        exit_qty=float(oco.exit_fill["qty"]),
                        fees_btc=float(oco.exit_fill.get("qty", 0))
                        * float(oco.exit_fill["price"])
                        * 0.001,
                        other_leg_cancelled=True,
                    )
                    closed.append(tr["trade_id"])
                    notes.append(f"closed_offline_exit:{tr['trade_id']}")
            # Stale PROTECTION_FAILED vs real/mocked exchange flat inventory.
            stale = self.reconcile_stale_protection_failed(universe=universe)
            notes.extend(stale.get("notes") or [])
            closed.extend(stale.get("closed") or [])
            notes.append("REST_RECONCILE_DRY")
            return {"ok": True, "notes": notes, "closed": closed, "dry_run": True}
        # Live path: open orders + lists (no unsupported symbol param on openOrderList).
        # INVARIANT: never upgrade local status to PROTECTED / PROTECTED_EMERGENCY here.
        # Presence of exchange protection is recorded; absence fails closed (warn), not silent OK.
        try:
            from binance_btc_bot.execution.order_lists import (
                filter_order_lists_by_symbol,
                parse_order_list,
                protection_state_from_list,
            )

            # Close PROTECTED trades that are already flat + ALL_DONE before protection-missing checks.
            flat_exits = self.reconcile_protected_flat_exits(universe=universe)
            notes.extend(flat_exits.get("notes") or [])
            closed.extend(flat_exits.get("closed") or [])
            closed_set = {str(x) for x in closed}

            open_orders = self.exchange.get_open_orders()
            open_lists = self.exchange.get_open_order_lists()
            notes.append(f"open_orders={len(open_orders)}")
            notes.append(f"open_lists={len(open_lists)}")
            fail_closed = False
            for tr in self.db.open_trades():
                sym = str(tr.get("symbol") or "").upper()
                if universe and sym not in {u.upper() for u in universe}:
                    continue
                trade_id = tr.get("trade_id")
                if trade_id is not None and str(trade_id) in closed_set:
                    continue
                oco_id = tr.get("binance_oco_list_id")
                local_status = str(tr.get("status") or "").upper()
                sym_lists = filter_order_lists_by_symbol(open_lists, sym)
                matched = None
                for row in sym_lists:
                    view = parse_order_list(row)
                    if oco_id and str(view.order_list_id) == str(oco_id):
                        matched = view
                        break
                    # Do not rebound arbitrary symbol OCO onto this trade (no silent rebind).
                state = protection_state_from_list(matched)

                # Detect emergency STOP_LOSS SELL for this trade (es_ / em_ client ids).
                em_stop = self._emergency_client_stop_id(str(trade_id or ""))
                em_oco = self._emergency_client_oco_id(str(trade_id or ""))
                stop_hit = None
                for o in open_orders or []:
                    if not isinstance(o, dict):
                        continue
                    if str(o.get("symbol") or "").upper() != sym:
                        continue
                    if str(o.get("side") or "").upper() != "SELL":
                        continue
                    cid = str(o.get("clientOrderId") or "")
                    otype = str(o.get("type") or o.get("orderType") or "").upper()
                    emergency_cid = cid.startswith(("es_", "es2_", "es3_", "em_")) or cid.startswith(
                        "es_"
                    )
                    if cid in {em_stop, em_oco} or (
                        otype in {"STOP_LOSS", "STOP_LOSS_LIMIT"} and emergency_cid
                    ):
                        stop_hit = o
                        break
                if stop_hit and state == "NO_OPEN_LIST":
                    state = "PROTECTED_EMERGENCY_STOP"

                # Operator-marked manual protection (e.g. web UI replaced bot OCO).
                manual_protection = False
                try:
                    raw_cfg = tr.get("strategy_config_json")
                    scfg = json.loads(raw_cfg) if isinstance(raw_cfg, str) else (raw_cfg or {})
                    manual_protection = bool(scfg.get("manual_protection"))
                except Exception:  # noqa: BLE001
                    manual_protection = False
                if manual_protection and local_status in {
                    "PROTECTED",
                    "PROTECTED_EMERGENCY",
                    "DRY_RUN_PROTECTED",
                }:
                    state = "PROTECTED_MANUAL"
                    notes.append(f"PROTECTED_MANUAL:{trade_id}:{sym}")
                    try:
                        self.safety.clear_warn(
                            "RECONCILE_PROTECTION_MISSING",
                            symbol=sym,
                            trade_id=trade_id,
                        )
                    except Exception:  # noqa: BLE001
                        pass

                # Accept standalone exchange protective sells (manual TP/SL) as coverage
                # when the stored OCO list id no longer matches / was cancelled.
                if state in {"NO_OPEN_LIST", "LIST_NOT_OPEN"} and local_status in {
                    "PROTECTED",
                    "PROTECTED_EMERGENCY",
                    "DRY_RUN_PROTECTED",
                }:
                    protective_types = {
                        "STOP_LOSS",
                        "STOP_LOSS_LIMIT",
                        "TAKE_PROFIT",
                        "TAKE_PROFIT_LIMIT",
                        "LIMIT_MAKER",
                        "TRAILING_STOP_MARKET",
                    }
                    for o in open_orders or []:
                        if not isinstance(o, dict):
                            continue
                        if str(o.get("symbol") or "").upper() != sym:
                            continue
                        if str(o.get("side") or "").upper() != "SELL":
                            continue
                        otype = str(o.get("type") or o.get("orderType") or "").upper()
                        if otype in protective_types:
                            state = "PROTECTED_EXTERNAL"
                            notes.append(
                                f"PROTECTED_EXTERNAL:{trade_id}:{sym}:orderId={o.get('orderId')}:{otype}"
                            )
                            try:
                                self.safety.clear_warn(
                                    "RECONCILE_PROTECTION_MISSING",
                                    symbol=sym,
                                    trade_id=trade_id,
                                )
                            except Exception:  # noqa: BLE001
                                pass
                            break

                # Fail closed: local claims protection but exchange shows none — only if
                # sellable inventory remains. Unsellable dust after OCO exit = POSITION_CLOSED path.
                if local_status in {
                    "PROTECTED",
                    "PROTECTED_EMERGENCY",
                    "DRY_RUN_PROTECTED",
                } and state in {"NO_OPEN_LIST", "LIST_NOT_OPEN"}:
                    inv = self._base_inventory(sym)
                    econ_flat, econ_reason = self._inventory_economically_flat(sym, inv=inv)
                    diag = {
                        "local_status": local_status,
                        "protection_state": state,
                        "oco_list_id": oco_id,
                        "inventory_free": inv[0] if inv else None,
                        "inventory_locked": inv[1] if inv else None,
                        "inventory_total": inv[2] if inv else None,
                        "economic_flat": econ_flat,
                        "economic_flat_reason": econ_reason,
                        "local_qty": tr.get("quantity"),
                        "entry_order_id": tr.get("binance_entry_order_id"),
                        "entry_time": tr.get("entry_time"),
                    }
                    logger.warning(
                        "reconcile protection check symbol=%s trade_id=%s state=%s econ_flat=%s/%s inv=%s oco=%s",
                        sym,
                        trade_id,
                        state,
                        econ_flat,
                        econ_reason,
                        inv,
                        oco_id,
                    )
                    if econ_flat is True:
                        # Dust or zero — treat as POSITION_CLOSED candidate, not missing protection.
                        state = "POSITION_CLOSED_PENDING"
                        retry = self.reconcile_protected_flat_exits(universe=[sym])
                        notes.extend(retry.get("notes") or [])
                        for cid in retry.get("closed") or []:
                            if cid not in closed_set:
                                closed.append(cid)
                                closed_set.add(str(cid))
                        notes.append(f"POSITION_CLOSED_DUST:{trade_id}:{sym}:{econ_reason}")
                        try:
                            self.safety.clear_warn(
                                "RECONCILE_PROTECTION_MISSING",
                                symbol=sym,
                                trade_id=trade_id,
                            )
                        except Exception:  # noqa: BLE001
                            pass
                    elif econ_flat is None:
                        state = "PROTECTION_MISSING"
                        fail_closed = True
                        self.safety.warn(
                            "RECONCILE_PROTECTION_MISSING",
                            symbol=sym,
                            trade_id=trade_id,
                            **diag,
                        )
                        notes.append(f"FAIL_CLOSED:{trade_id}:{sym}:inventory_unknown")
                    else:
                        # Sellable inventory with no open protection — genuine missing.
                        state = "PROTECTION_MISSING"
                        fail_closed = True
                        diag["protection_state"] = "PROTECTION_MISSING"
                        self.safety.warn(
                            "RECONCILE_PROTECTION_MISSING",
                            symbol=sym,
                            trade_id=trade_id,
                            **diag,
                        )
                        notes.append(f"FAIL_CLOSED:{trade_id}:{sym}:unverified_protection")

                if state in {"PROTECTED_OPEN_LIST", "PROTECTED_EXTERNAL", "PROTECTED_MANUAL"}:
                    try:
                        self.safety.clear_warn(
                            "RECONCILE_PROTECTION_MISSING",
                            symbol=sym,
                            trade_id=trade_id,
                        )
                    except Exception:  # noqa: BLE001
                        pass

                # Never rewrite unprotected / failed into PROTECTED.
                if local_status in {"PROTECTION_FAILED", "ENTRY_FILLED", "PROTECTION_PENDING"}:
                    if state == "NO_OPEN_LIST":
                        state = "PROTECTION_PENDING" if local_status == "PROTECTION_PENDING" else "OPEN_UNPROTECTED"
                    # Even if an OCO appears, do not auto-flip status here — lifecycle owns that.

                notes.append(f"trade:{trade_id}:{sym}:{local_status}->{state}")
                self.db.insert_event(
                    "REST_RECONCILE",
                    symbol=sym,
                    trade_id=trade_id,
                    order_id=str(oco_id) if oco_id else None,
                    reason=state,
                    payload={
                        "local_status": local_status,
                        "list": matched.raw if matched else None,
                        "child_orders": list(matched.child_orders) if matched else [],
                        "quantity": matched.quantity if matched else None,
                        "stop": stop_hit,
                        # Explicit: reconcile_rest does not mutate protection status upward.
                        "status_mutated": False,
                    },
                )
            stale = self.reconcile_stale_protection_failed(universe=universe)
            notes.extend(stale.get("notes") or [])
            closed.extend(stale.get("closed") or [])
            return {
                "ok": not fail_closed,
                "notes": notes,
                "closed": closed,
                "dry_run": False,
                "fail_closed": fail_closed,
            }
        except Exception as e:  # noqa: BLE001
            err = scrub_exception(e)
            notes.append(err)
            return {"ok": False, "notes": notes, "closed": closed, "dry_run": False, "fail_closed": True}

    def restart_recover(self, *, case_hint: str | None = None) -> dict[str, Any]:
        """Restart recovery treating Binance (dry broker) as source of truth.

        Cases:
          A — BUY not filled
          B — BUY filled, OCO submitted
          C — BUY filled, OCO submitted, trailing active
          D — exit occurred while offline
          E — position on exchange, local DB missing/incomplete
        """
        report: dict[str, Any] = {"ok": True, "cases": [], "notes": []}
        if not self.dry_broker:
            report["notes"].append("no_dry_broker")
            return report

        # Case D first: completed OCOs → close local
        d = self.reconcile_rest()
        if d.get("closed"):
            report["cases"].append("D")
            report["notes"].extend(d.get("notes") or [])

        open_lists = {str(x["orderListId"]): x for x in self.dry_broker.open_order_lists()}
        local = self.db.open_trades()
        local_by_oco = {str(t.get("binance_oco_list_id")): t for t in local if t.get("binance_oco_list_id")}
        local_by_sym = {str(t["symbol"]).upper(): t for t in local}

        # Case A: local ENTRY without fill / buy still open / cancelled
        for t in local:
            st = str(t.get("status") or "").upper()
            if st in {"ENTRY_PENDING", "OPEN"} and not t.get("binance_entry_order_id"):
                self.db.update_trade(t["trade_id"], status="CLOSED")
                self.db.insert_event("RECOVERY", trade_id=t["trade_id"], reason="RESTART_BUY_NOT_FILLED")
                report["cases"].append("A")
                report["notes"].append(f"A:{t['trade_id']}")

        # Cases B/C: OCO present on exchange → ensure PROTECTED
        for lid, row in open_lists.items():
            sym = str(row.get("symbol") or "").upper()
            trade = local_by_oco.get(lid) or local_by_sym.get(sym)
            if trade:
                status = "PROTECTED" if not self.dry_run else "DRY_RUN_PROTECTED"
                self.db.update_trade(trade["trade_id"], status=status, binance_oco_list_id=lid)
                oco = self.dry_broker.get_oco(lid)
                if oco and oco.status == "TRAILING_ACTIVE":
                    report["cases"].append("C")
                    self.db.insert_event("RECOVERY", trade_id=trade["trade_id"], reason="TRAILING_ACTIVE", order_id=lid)
                else:
                    report["cases"].append("B")
                    self.db.insert_event("RECOVERY", trade_id=trade["trade_id"], reason="OCO_ACTIVE", order_id=lid)
            else:
                # Case E: exchange position without local trade
                report["cases"].append("E")
                report["notes"].append(f"E:orphan_oco:{lid}:{sym}")
                self.safety.warn("EXCHANGE_OCO_WITHOUT_LOCAL_TRADE", symbol=sym, order_list_id=lid)
                self.db.insert_event(
                    "WARNING",
                    symbol=sym,
                    order_id=lid,
                    reason="EXCHANGE_OCO_WITHOUT_LOCAL_TRADE",
                )

        # Deduplicate case labels
        report["cases"] = sorted(set(report["cases"]))
        if case_hint:
            report["hint"] = case_hint
        return report
