"""Canary / T1 production execution runtime (isolated from paper engine).

Orchestrates: size → durable intent → risk → MARKET BUY → reconcile →
bot-managed T1 + T1PriceMonitor → MARKET SELL on stop cross → CANARY_COMPLETE.

BOT-MANAGED T1 — does NOT call place_protective_stop / require
protection_api_confirmed. Default WriteGate remains CLOSED.
Does NOT wire into btcc.service.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from btcc.execution.canary import CanaryPhase, CanaryState
from btcc.execution.client_order_id import OrderKind, build_client_order_id, intent_fingerprint
from btcc.execution.intent import DurableTradeIntent, IntentJournal, IntentStatus
from btcc.execution.kill_switch import ExecutionKillSwitch
from btcc.execution.modes import ExecutionMode
from btcc.execution.mexc.write_client import MexcWriteTimeout
from btcc.execution.order_state import OrderJournal, OrderLifecycleStatus, OrderRecord
from btcc.execution.price_monitor import (
    SIGNAL_EVALUATION_INTERVAL_S,
    PriceMonitorState,
    PriceTick,
    T1PriceMonitor,
)
from btcc.execution.protection import ProtectionState, ProtectiveExitManager
from btcc.execution.risk import ExecutionRiskGate, RiskContext
from btcc.execution.sizing import size_alt_btc_position
from btcc.execution.symbols import SymbolMeta
from btcc.execution.t1 import T1, T1State
from btcc.execution.types import SpotMarketOrderRequest
from btcc.execution.write_gate import WriteGate, CLOSED_WRITE_GATE
from btcc.safety.no_trading import TradingForbiddenError

logger = logging.getLogger(__name__)

PRODUCTION_STRATEGY_VERSION = "T1-ONLY-PROD"
CANARY_STRATEGY_VERSION = "T1-ONLY-CANARY"

_QUOTE_ASSETS = frozenset({"BTC", "USDT", "USDC"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AuditEvent:
    ts: str
    event: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"ts": self.ts, "event": self.event, "payload": self.payload}


class ExecutionAuditLog:
    """Append-only JSONL audit (never logs secrets)."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.events: list[AuditEvent] = []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: str, **payload: Any) -> None:
        clean = {
            k: v
            for k, v in payload.items()
            if "secret" not in str(k).lower() and "api_key" not in str(k).lower()
        }
        ev = AuditEvent(ts=_utc_now(), event=event, payload=clean)
        self.events.append(ev)
        logger.info("EXEC_AUDIT %s %s", event, list(dict(list(clean.items())[:12])))
        if self.path:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(ev.to_dict(), default=str) + "\n")


@dataclass
class CanaryEntryResult:
    ok: bool
    reason: str | None = None
    position_id: str | None = None
    executed_quantity: float | None = None
    average_price: float | None = None
    protection_id: str | None = None
    halt: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class CanaryExitResult:
    ok: bool
    reason: str | None = None
    position_id: str | None = None
    executed_quantity: float | None = None
    average_price: float | None = None
    trigger_price: float | None = None
    halt: bool = False
    already_exiting: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


class CanaryRuntime:
    """Single-shot REAL canary orchestration with T1 protection.

    Live T1 stop monitoring is driven by ``T1PriceMonitor`` (near-real-time),
    independent of the 15-minute signal evaluation loop.
    """

    def __init__(
        self,
        *,
        broker: Any,
        risk: ExecutionRiskGate,
        protection_manager: ProtectiveExitManager,
        canary: CanaryState | None = None,
        write_gate: WriteGate | None = None,
        kill_switch: ExecutionKillSwitch | None = None,
        intent_journal: IntentJournal | None = None,
        order_journal: OrderJournal | None = None,
        audit: ExecutionAuditLog | None = None,
        strategy_version: str = CANARY_STRATEGY_VERSION,
        max_positions: int = 1,
        allocation_pct: float = 0.25,
        require_s_min: float = 0.6,
        price_monitor: T1PriceMonitor | None = None,
        trade_notifier: Any | None = None,
    ) -> None:
        if risk.mode != ExecutionMode.REAL_CANARY_SINGLE_SHOT:
            raise TradingForbiddenError("CanaryRuntime requires REAL_CANARY_SINGLE_SHOT risk mode")
        if getattr(broker, "mode", None) != ExecutionMode.REAL_CANARY_SINGLE_SHOT:
            raise TradingForbiddenError(
                "CanaryRuntime requires RealBroker in REAL_CANARY_SINGLE_SHOT mode"
            )
        self.broker = broker
        self.risk = risk
        self.protection = protection_manager
        self.canary = canary or CanaryState(max_entries=1, max_positions=1)
        self.gate = write_gate or CLOSED_WRITE_GATE
        self.kill_switch = kill_switch or risk.kill_switch
        self.intent_journal = intent_journal
        self.order_journal = order_journal
        self.audit = audit or ExecutionAuditLog()
        self.strategy_version = strategy_version
        self.max_positions = max_positions
        self.allocation_pct = allocation_pct
        self.require_s_min = require_s_min
        self.halted = False
        self.halt_reason: str | None = None
        self.price_stale = False
        self.price_monitor = price_monitor
        self.trade_notifier = trade_notifier  # observability only; never gates orders
        self._open_position_meta: dict[str, dict[str, Any]] = {}
        self._exit_in_flight: set[str] = set()
        self._last_exit: CanaryExitResult | None = None
        self._started_monotonic = time.monotonic()
        self._entries_count = 0
        self._closes_count = 0
        self._realized_pnl_btc = 0.0
        self._btc_balance_at_start: float | None = None
        self._last_hourly_telegram_bucket: str | None = None

    def halt(self, reason: str) -> None:
        self.halted = True
        self.halt_reason = reason
        self.canary.halt(reason)
        self.audit.record("HALT", reason=reason)
        if self.kill_switch is not None:
            self.kill_switch.halt_submissions(reason)
        if self.price_monitor is not None:
            self.price_monitor.halt(reason)

    def attach_price_monitor(self, monitor: T1PriceMonitor) -> None:
        """Attach near-real-time monitor (must be << 15m signal interval)."""
        if not monitor.is_independent_of_signal_loop:
            raise TradingForbiddenError(
                f"price monitor must be independent of {SIGNAL_EVALUATION_INTERVAL_S}s signal loop"
            )
        monitor.on_tick = self._on_monitor_tick
        monitor.on_stale = self._on_monitor_stale
        monitor.on_reconnect = self._on_monitor_reconnect
        self.price_monitor = monitor
        self.audit.record(
            "PRICE_MONITOR_ATTACHED",
            symbol=monitor.symbol,
            poll_interval_s=monitor.poll_interval_s,
            max_stale_age_s=monitor.max_stale_age_s,
            signal_loop_s=SIGNAL_EVALUATION_INTERVAL_S,
        )

    def start_price_monitor(self) -> None:
        if self.price_monitor is None:
            raise RuntimeError("no price monitor attached")
        self.price_monitor.start()
        self.audit.record(
            "PRICE_MONITOR_STARTED",
            symbol=self.price_monitor.symbol,
            poll_interval_s=self.price_monitor.poll_interval_s,
        )

    def stop_price_monitor(self) -> None:
        if self.price_monitor is None:
            return
        self.price_monitor.stop()
        self.audit.record("PRICE_MONITOR_STOPPED", symbol=self.price_monitor.symbol)

    def _on_monitor_tick(self, tick: PriceTick) -> None:
        self.price_stale = False
        pid = self.canary.position_id
        if not pid:
            return
        self.on_mark(pid, tick.price, source="price_monitor", tick_ts=tick.ts_wall)

    def _on_monitor_stale(self, symbol: str, age_s: float) -> None:
        self.price_stale = True
        self.audit.record("PRICE_STALE", symbol=symbol, age_s=age_s)
        # Halt new entries; exits remain allowed.
        if self.kill_switch is not None and self.kill_switch.can_submit_new_order():
            self.kill_switch.halt_submissions(f"PRICE_STALE:{symbol}:age={age_s:.3f}s")

    def _on_monitor_reconnect(self, symbol: str, attempts: int) -> None:
        self.audit.record("PRICE_RECONNECTED", symbol=symbol, attempts=attempts)

    def _peek_btc_available(self) -> float | None:
        """Best-effort BTC available balance. Never raises; never places orders."""
        try:
            if hasattr(self.broker, "get_account_state"):
                st = self.broker.get_account_state()
                return float(getattr(st, "available", None) or getattr(st, "available_balance", 0.0) or 0.0)
            if hasattr(self.broker, "get_balances"):
                bals = self.broker.get_balances()
                if isinstance(bals, dict):
                    return float(bals.get("BTC") or bals.get("btc") or 0.0)
        except Exception as e:  # noqa: BLE001
            logger.warning("btc balance peek failed (non-blocking): %s", e)
        return None

    def _notify_real_open(
        self,
        *,
        position_id: str,
        symbol: str,
        coid: str,
        fill_oid: str | None,
        fill_exec: float,
        fill_avg: float,
        fill_fee: float,
        available_btc_before: float,
        prot: Any,
    ) -> None:
        """Post-fill OPEN Telegram. Failures must never affect the trade lifecycle."""
        if self.trade_notifier is None:
            return
        try:
            t1 = prot.t1 if getattr(prot, "t1", None) is not None else None
            btc_alloc = float(fill_exec) * float(fill_avg)
            btc_after = self._peek_btc_available()
            payload = {
                "symbol": symbol,
                "entry_time": _utc_now(),
                "executed_quantity": fill_exec,
                "entry_vwap": fill_avg,
                "btc_allocated": btc_alloc,
                "btc_balance": btc_after if btc_after is not None else available_btc_before,
                "btc_balance_before_entry": available_btc_before,
                "entry_fee_btc": fill_fee,
                "initial_stop": getattr(t1, "initial_stop_price_value", None)
                or getattr(prot, "stop_price", None),
                "activation_price": getattr(t1, "activation_price_value", None),
                "t1": {
                    "stop_loss_pct": getattr(getattr(t1, "config", None), "stop_loss_pct", T1.stop_loss_pct),
                    "activation_pct": getattr(getattr(t1, "config", None), "activation_pct", T1.activation_pct),
                    "trailing_pct": getattr(getattr(t1, "config", None), "trailing_pct", T1.trailing_pct),
                },
                "client_order_id": coid,
                "exchange_order_id": fill_oid,
                "position_id": position_id,
                "strategy_version": self.strategy_version,
                "asof_utc": _utc_now(),
            }
            ok = bool(self.trade_notifier.trade_opened(payload))
            self.audit.record("TELEGRAM_OPEN", sent=ok, client_order_id=coid)
        except Exception as e:  # noqa: BLE001
            logger.error("TELEGRAM_OPEN failed (non-blocking): %s", e)
            self.audit.record("TELEGRAM_OPEN_FAILED", error=str(e), client_order_id=coid)

    def _notify_real_close(
        self,
        *,
        position_id: str,
        symbol: str,
        entry_coid: str,
        entry_oid: str | None,
        exit_coid: str,
        exit_oid: str | None,
        entry_vwap: float,
        exit_vwap: float,
        qty: float,
        entry_fee: float,
        exit_fee: float,
        btc_before: float | None,
        exit_reason: str,
        trigger_price: float | None = None,
        stop_price: float | None = None,
        activated: bool | None = None,
    ) -> None:
        """Post-fill CLOSE Telegram. Failures must never affect the trade lifecycle."""
        if self.trade_notifier is None:
            return
        try:
            gross = (float(exit_vwap) - float(entry_vwap)) * float(qty)
            fees = float(entry_fee or 0.0) + float(exit_fee or 0.0)
            net = gross - fees
            btc_after = self._peek_btc_available()
            stop_px = stop_price if stop_price is not None else trigger_price
            payload = {
                "symbol": symbol,
                "entry_vwap": entry_vwap,
                "exit_vwap": exit_vwap,
                "executed_quantity": qty,
                "gross_pnl_btc": gross,
                "fees_btc": fees,
                "net_pnl_btc": net,
                "btc_before": btc_before,
                "btc_after": btc_after,
                "exit_reason": exit_reason,
                "activated": activated,
                "trigger_price": stop_px,
                "stop_price": stop_px,
                "execution_price": exit_vwap,
                "t1": {
                    "stop_loss_pct": T1.stop_loss_pct,
                    "activation_pct": T1.activation_pct,
                    "trailing_pct": T1.trailing_pct,
                },
                "entry_client_order_id": entry_coid,
                "entry_exchange_order_id": entry_oid,
                "exit_client_order_id": exit_coid,
                "exit_exchange_order_id": exit_oid,
                "position_id": position_id,
                "strategy_version": self.strategy_version,
                "asof_utc": _utc_now(),
            }
            ok = bool(self.trade_notifier.trade_closed(payload))
            self._realized_pnl_btc += float(net)
            self.audit.record(
                "TELEGRAM_CLOSE",
                sent=ok,
                exit_client_order_id=exit_coid,
                net_pnl_btc=net,
            )
        except Exception as e:  # noqa: BLE001
            logger.error("TELEGRAM_CLOSE failed (non-blocking): %s", e)
            self.audit.record("TELEGRAM_CLOSE_FAILED", error=str(e), exit_client_order_id=exit_coid)

    def maybe_send_hourly_report(self) -> bool:
        """Hourly LIVE digest at BRT hour boundaries. Never affects trading.

        Seeds the current BRT hour on first call without sending — avoids an
        immediate post-OPEN duplicate that is not a true hourly report.
        """
        if self.trade_notifier is None:
            return False
        try:
            from btcc.telegram.real_trades import brt_hour_bucket, compute_hourly_real_stats

            now = datetime.now(timezone.utc)
            bucket = brt_hour_bucket(now)
            # First observation: seed only (do not emit for the in-progress hour).
            if self._last_hourly_telegram_bucket is None:
                self._last_hourly_telegram_bucket = bucket
                self.audit.record("TELEGRAM_HOURLY_SEEDED", bucket=bucket, sent=False)
                return False
            if self._last_hourly_telegram_bucket == bucket:
                return False
            open_pos = None
            pid = self.canary.position_id
            if pid and self.canary.phase == CanaryPhase.OPEN:
                meta = self._open_position_meta.get(pid) or {}
                rec = self.protection.get(pid) if hasattr(self.protection, "get") else None
                t1 = getattr(rec, "t1", None) if rec is not None else None
                cfg = getattr(t1, "config", None) if t1 is not None else None
                open_pos = {
                    "symbol": meta.get("symbol"),
                    "executed_quantity": meta.get("executed_quantity"),
                    "entry_vwap": meta.get("entry_vwap"),
                    "initial_stop": getattr(t1, "initial_stop_price_value", None) if t1 else None,
                    "activation_price": getattr(t1, "activation_price_value", None) if t1 else None,
                    "stop_price": getattr(rec, "stop_price", None) if rec else None,
                    "trailing_stop": getattr(t1, "stop_price", None) if t1 else None,
                    "highest_price": getattr(t1, "highest_price", None) if t1 else None,
                    "activated": bool(getattr(t1, "activated", False)) if t1 else False,
                    "t1_state": getattr(getattr(t1, "state", None), "value", None) if t1 else None,
                    "stop_loss_pct": getattr(cfg, "stop_loss_pct", T1.stop_loss_pct),
                    "activation_pct": getattr(cfg, "activation_pct", T1.activation_pct),
                    "trailing_pct": getattr(cfg, "trailing_pct", T1.trailing_pct),
                }
            status = "HALTED" if self.halted else (
                "FLAT" if self.canary.phase == CanaryPhase.CANARY_COMPLETE else "MONITORING"
            )
            stats = compute_hourly_real_stats(
                btc_balance=self._peek_btc_available(),
                n_entered=self._entries_count,
                n_closed=self._closes_count,
                realized_pnl_btc=self._realized_pnl_btc,
                open_position=open_pos,
                uptime_s=time.monotonic() - self._started_monotonic,
                execution_status=status,
                canary_phase=self.canary.phase.value,
                write_gate_armed=bool(getattr(self.gate, "market_writes_allowed", False)),
                now=now,
            )
            ok = bool(
                self.trade_notifier.hourly_summary(
                    stats,
                    version={"strategy_version": self.strategy_version},
                    bucket_key=bucket,
                )
            )
            self._last_hourly_telegram_bucket = bucket
            self.audit.record("TELEGRAM_HOURLY", sent=ok, bucket=bucket)
            return ok
        except Exception as e:  # noqa: BLE001
            logger.error("TELEGRAM_HOURLY failed (non-blocking): %s", e)
            self.audit.record("TELEGRAM_HOURLY_FAILED", error=str(e))
            return False

    def _normalize_symbol(self, symbol: str) -> str:
        return (
            str(symbol)
            .upper()
            .replace("/", "")
            .replace("-", "")
            .replace("_", "")
        )

    def _emergency_market_sell(
        self,
        *,
        symbol: str,
        position_id: str,
        quantity: float,
        entry_coid: str,
        symbol_meta: SymbolMeta | None = None,
    ) -> None:
        try:
            self.gate.assert_market_write("emergency_market_sell")
            meta = symbol_meta or (self._open_position_meta.get(position_id) or {}).get("symbol_meta")
            req_meta: dict = {"emergency": True}
            if meta is not None:
                req_meta["symbol_meta"] = meta
            self.broker.submit_order(
                SpotMarketOrderRequest(
                    client_order_id=f"{entry_coid}-EMRG",
                    symbol=symbol,
                    side="SELL",
                    quantity=float(quantity),
                    intent_id=position_id,
                    meta=req_meta,
                )
            )
            self.audit.record("EMERGENCY_MARKET_SELL", intent_id=position_id, quantity=quantity)
        except Exception as ee:  # noqa: BLE001
            self.audit.record("EMERGENCY_SELL_FAILED", error=str(ee))

    def try_canary_entry(
        self,
        *,
        symbol: str,
        price_alt_btc: float,
        s_value: float,
        signal_ts: str,
        candle_ts: str,
        available_btc: float,
        symbol_meta: SymbolMeta,
        open_exposure_pct: float = 0.0,
        open_position_count: int = 0,
        account_ctx: RiskContext | None = None,
        intent_nonce: str = "canary1",
    ) -> CanaryEntryResult:
        if self.halted or self.canary.phase == CanaryPhase.HALTED:
            return CanaryEntryResult(False, "HALTED", halt=True)
        if self.canary.phase == CanaryPhase.CANARY_COMPLETE:
            return CanaryEntryResult(False, "CANARY_COMPLETE")
        if not self.canary.allow_new_entry():
            return CanaryEntryResult(False, f"CANARY_PHASE_{self.canary.phase.value}")
        if not self.kill_switch.can_submit_new_order():
            return CanaryEntryResult(False, "EXECUTION_KILL_SWITCH")
        # Crash mid-submit leaves SUBMIT_PENDING / RECONCILE_UNKNOWN under a prior
        # clientOrderId. A new signal_ts would mint a new id — never place a second BUY.
        if self.order_journal is not None and self.order_journal.any_unresolved():
            unresolved = self.order_journal.unresolved_orders()
            detail = ",".join(f"{o.client_order_id}:{o.status}" for o in unresolved[:4])
            self.halt(f"UNRESOLVED_PRIOR_ORDER:{detail}")
            return CanaryEntryResult(False, self.halt_reason, halt=True)
        if float(s_value) < self.require_s_min:
            return CanaryEntryResult(False, "S_BELOW_THRESHOLD")
        if open_position_count >= self.max_positions:
            return CanaryEntryResult(False, "MAX_POSITIONS")

        from btcc.execution.symbols import t1_market_exit_capability

        ok_mkt, mkt_reason = t1_market_exit_capability(symbol_meta)
        if not ok_mkt:
            self.audit.record(
                "ENTRY_REJECTED_MARKET_UNSUPPORTED",
                symbol=symbol,
                reason=mkt_reason,
            )
            return CanaryEntryResult(False, mkt_reason)

        size = size_alt_btc_position(
            available_btc=available_btc,
            price_alt_btc=price_alt_btc,
            meta=symbol_meta,
            allocation_pct=self.allocation_pct,
            open_exposure_pct=open_exposure_pct,
            max_aggregate_exposure=1.0,
        )
        if not size.ok:
            self.audit.record("SIZE_REJECT", reason=size.reason, symbol=symbol)
            return CanaryEntryResult(False, size.reason)

        position_id = intent_fingerprint(
            strategy_version=self.strategy_version,
            symbol=symbol,
            side="BUY",
            signal_ts=signal_ts,
            candle_ts=candle_ts,
            selected_exit=T1.key,
            kind=OrderKind.ENTRY,
            intent_nonce=intent_nonce,
        )
        coid = build_client_order_id(
            strategy_version=self.strategy_version,
            symbol=symbol,
            side="BUY",
            signal_ts=signal_ts,
            candle_ts=candle_ts,
            selected_exit=T1.key,
            kind=OrderKind.ENTRY,
            attempt=1,
            intent_nonce=intent_nonce,
        )
        intent = DurableTradeIntent(
            intent_id=position_id,
            client_order_id=coid,
            strategy_version=self.strategy_version,
            symbol=symbol,
            side="BUY",
            requested_allocation_pct=self.allocation_pct,
            requested_quantity=size.quantity,
            signal_ts=signal_ts,
            candle_ts=candle_ts,
            selected_exit=T1.key,
            execution_mode=ExecutionMode.REAL_CANARY_SINGLE_SHOT.value,
            status=IntentStatus.CREATED.value,
            order_kind=OrderKind.ENTRY.value,
            current_strategy_state={"s_value": s_value, "exit": T1.key},
            meta={
                "notional_btc": size.notional_btc,
                "production_path": "T1_ONLY",
                "raw_quantity": size.raw_quantity,
                "submitted_quantity": size.quantity,
                "submitted_quantity_serialized": size.quantity_serialized,
            },
        )
        ctx = account_ctx or RiskContext(
            symbol_meta=symbol_meta,
            open_position_count=open_position_count,
            current_exposure_pct=open_exposure_pct,
            market_data_stale=False,
            account_data_stale=False,
        )
        if ctx.symbol_meta is None:
            ctx.symbol_meta = symbol_meta

        decision = self.risk.validate_intent(intent, ctx)
        if not decision.approved:
            self.audit.record("RISK_REJECT", reason=decision.reason, intent_id=position_id)
            return CanaryEntryResult(False, decision.reason)

        intent.status = IntentStatus.PERSISTED.value
        if self.intent_journal is not None:
            self.intent_journal.append(intent)

        self.canary.begin_entry(symbol=symbol, position_id=position_id)
        self.audit.record(
            "ENTRY_SUBMIT",
            intent_id=position_id,
            client_order_id=coid,
            symbol=symbol,
            requested_quantity=size.quantity,
            raw_quantity=size.raw_quantity,
            submitted_quantity=size.quantity,
            submitted_quantity_serialized=size.quantity_serialized,
        )

        order_rec = OrderRecord(
            order_local_id=f"{coid}:NEW",
            intent_id=position_id,
            client_order_id=coid,
            symbol=symbol,
            side="BUY",
            order_kind=OrderKind.ENTRY.value,
            status=OrderLifecycleStatus.SUBMIT_PENDING.value,
            requested_quantity=size.quantity,
        )
        if self.order_journal is not None:
            self.order_journal.append(order_rec)

        if not size.quantity_serialized:
            self.halt("ENTRY_QTY_SERIALIZE_MISSING")
            return CanaryEntryResult(False, self.halt_reason, halt=True)

        req = SpotMarketOrderRequest(
            client_order_id=coid,
            symbol=symbol,
            side="BUY",
            quantity=size.quantity,
            intent_id=position_id,
            meta={
                "symbol_meta": symbol_meta,
                "raw_quantity": size.raw_quantity,
                "submitted_quantity_serialized": size.quantity_serialized,
            },
        )

        fill_ok = False
        fill_exec = 0.0
        fill_avg = 0.0
        fill_fee = 0.0
        fill_oid: str | None = None
        fill_status: str | None = None
        fill_reject: str | None = None

        try:
            fill = self.broker.submit_order(req)
            fill_ok = bool(fill.ok)
            fill_exec = float(fill.executed_quantity or 0.0)
            fill_avg = float(fill.average_price or 0.0)
            fill_fee = float(fill.fee or 0.0)
            fill_oid = fill.exchange_order_id
            fill_status = str(fill.order_status or "")
            fill_reject = fill.rejection_reason
        except MexcWriteTimeout as e:
            self.audit.record("ENTRY_TIMEOUT", intent_id=position_id, error=str(e))
            if self.order_journal is not None:
                self.order_journal.transition(
                    coid, OrderLifecycleStatus.RECONCILE_UNKNOWN.value
                )
            # Timeout without blind retry — reconcile once by client_order_id.
            try:
                ex = self.broker.get_order_status(symbol=symbol, orig_client_order_id=coid)
                executed = float(ex.executed_quantity)
                avg = float(ex.average_price or 0.0)
                if avg <= 0 and executed > 0 and hasattr(self.broker, "_vwap_from_order"):
                    avg = float(self.broker._vwap_from_order(ex))
                if executed <= 0:
                    # Accepted-or-unknown after timeout must HALT — never re-arm for another BUY.
                    self.halt("ENTRY_TIMEOUT_FILL_UNCONFIRMED")
                    return CanaryEntryResult(False, self.halt_reason, halt=True)
                fill_ok = True
                fill_exec = executed
                fill_avg = avg
                fill_fee = 0.0
                fill_oid = ex.order_id
                fill_status = str(ex.status or "FILLED")
            except Exception as re:  # noqa: BLE001
                self.halt(f"TIMEOUT_RECONCILE_FAILED:{re}")
                return CanaryEntryResult(False, self.halt_reason, halt=True)
        except TradingForbiddenError as e:
            self.canary.phase = CanaryPhase.ARMED
            self.canary.position_id = None
            return CanaryEntryResult(False, f"WRITE_GATE:{e}")
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "reject" in msg.lower() or "Insufficient" in msg:
                self.canary.phase = CanaryPhase.ARMED
                self.canary.position_id = None
                return CanaryEntryResult(False, f"ENTRY_REJECTED:{e}")
            self.audit.record("ENTRY_ERROR", error=str(e))
            self.halt(f"ENTRY_UNKNOWN_ERROR:{e}")
            return CanaryEntryResult(False, self.halt_reason, halt=True)

        # Incomplete ACK / unconfirmed fill ≠ ZERO_FILL. HALT — never second BUY.
        ambiguous = (
            (fill_reject or "").startswith("ACK_INCOMPLETE_RECONCILE_FAILED")
            or fill_reject
            in {
                "ORDER_ACCEPTED_FILL_UNCONFIRMED",
                "FILL_QTY_OK_VWAP_UNCONFIRMED",
            }
            or str(fill_status).upper() in {"RECONCILE_UNKNOWN", "NEW", "SUBMITTED", "PENDING", "ACK"}
        )
        if (not fill_ok or fill_exec <= 0 or fill_avg <= 0) and ambiguous:
            if self.order_journal is not None:
                self.order_journal.transition(
                    coid, OrderLifecycleStatus.RECONCILE_UNKNOWN.value
                )
            self.halt(f"ENTRY_RECONCILE_UNKNOWN:{fill_reject or fill_status}")
            return CanaryEntryResult(False, self.halt_reason, halt=True)

        if (not fill_ok) or fill_exec <= 0:
            # Conclusive unfilled/canceled/rejected only.
            self.canary.phase = CanaryPhase.ARMED
            self.canary.position_id = None
            if fill_reject and (
                "MARKET_ORDER" in fill_reject
                or fill_reject.startswith("SYMBOL_META_UNAVAILABLE")
            ):
                return CanaryEntryResult(False, fill_reject)
            return CanaryEntryResult(False, "ZERO_FILL")

        if fill_avg <= 0:
            if self.order_journal is not None:
                self.order_journal.transition(
                    coid, OrderLifecycleStatus.RECONCILE_UNKNOWN.value
                )
            self.halt("ENTRY_VWAP_UNCONFIRMED")
            return CanaryEntryResult(False, self.halt_reason, halt=True)

        if self.order_journal is not None:
            self.order_journal.transition(
                coid,
                OrderLifecycleStatus.FILLED.value
                if fill_exec + 1e-15 >= float(size.quantity)
                else OrderLifecycleStatus.PARTIALLY_FILLED.value,
                executed_quantity=fill_exec,
                average_fill_price=fill_avg,
                exchange_order_id=fill_oid,
                remaining_quantity=max(0.0, float(size.quantity) - fill_exec),
            )

        self.canary.mark_open()
        self.audit.record(
            "ENTRY_FILLED",
            intent_id=position_id,
            client_order_id=coid,
            exchange_order_id=fill_oid,
            executed_quantity=fill_exec,
            average_price=fill_avg,
            fee=fill_fee,
            requested_quantity=size.quantity,
        )

        entry_price = fill_avg
        try:
            # BOT_MANAGED: establish() never calls place_protective_stop.
            prot = self.protection.establish(
                position_id=position_id,
                symbol=symbol,
                quantity=fill_exec,
                entry_price=entry_price,
                client_order_id=f"{coid}-SL",
                requested_quantity=size.quantity,
            )
        except Exception as e:  # noqa: BLE001
            self.audit.record("PROTECTION_FAILED", intent_id=position_id, error=str(e))
            self.halt(f"ENTRY_OK_PROTECTION_FAILED:{e}")
            self._emergency_market_sell(
                symbol=symbol,
                position_id=position_id,
                quantity=fill_exec,
                entry_coid=coid,
                symbol_meta=symbol_meta,
            )
            return CanaryEntryResult(
                False,
                self.halt_reason,
                position_id=position_id,
                executed_quantity=fill_exec,
                average_price=fill_avg,
                halt=True,
            )

        t1_fields = prot.t1.log_fields() if prot.t1 is not None else {}
        self.audit.record(
            "PROTECTION_ESTABLISHED",
            **{
                "intent_id": position_id,
                "client_order_id": coid,
                "protective_order_id": prot.exchange_protection_id,
                "stop_price": prot.stop_price,
                "quantity": fill_exec,
                "requested_quantity": size.quantity,
                "raw_quantity": size.raw_quantity,
                "submitted_quantity_serialized": size.quantity_serialized,
                "t1_state": prot.t1_state.value if prot.t1_state else None,
                **{k: v for k, v in t1_fields.items() if k not in {"intent_id", "client_order_id"}},
            },
        )

        self._open_position_meta[position_id] = {
            "symbol": symbol,
            "signal_ts": signal_ts,
            "candle_ts": candle_ts,
            "entry_client_order_id": coid,
            "entry_exchange_order_id": fill_oid,
            "requested_quantity": size.quantity,
            "raw_quantity": size.raw_quantity,
            "submitted_quantity": size.quantity,
            "submitted_quantity_serialized": size.quantity_serialized,
            "executed_quantity": fill_exec,
            "entry_vwap": fill_avg,
            "entry_fee_btc": fill_fee,
            "btc_balance_before_entry": float(available_btc),
            "btc_allocated": float(fill_exec) * float(fill_avg),
            "symbol_meta": symbol_meta,
        }
        if prot.t1 is not None:
            prot.t1.meta.update(self._open_position_meta[position_id])

        # Position is NOT fully initialized unless monitor is running/armed.
        try:
            if self.price_monitor is None:
                raise RuntimeError("price_monitor required after fill")
            if self.price_monitor.symbol.upper() != self._normalize_symbol(symbol):
                self.price_monitor.symbol = self._normalize_symbol(symbol)
            # Arm before start so the first tick cannot trip MARK_WITHOUT_ARMED_MONITOR.
            self.protection.mark_monitor_armed(position_id)
            self.start_price_monitor()
            if self.price_monitor.state != PriceMonitorState.RUNNING:
                raise RuntimeError(
                    f"monitor state={self.price_monitor.state.value} (expected RUNNING)"
                )
            # Require a validated mark before PROTECTED (same invariant as bounded 6h).
            tick = self.price_monitor.poll_once()
            if tick is None:
                # Allow brief async tick from monitor thread.
                import time as _time

                for _ in range(20):
                    _time.sleep(0.05)
                    rec_wait = self.protection.get(position_id)
                    if rec_wait is not None and rec_wait.state == ProtectionState.PROTECTED:
                        break
                    tick = self.price_monitor.poll_once()
                    if tick is not None:
                        break
            if tick is not None:
                from btcc.execution.live_price_guard import validate_mark_vs_entry

                vv = validate_mark_vs_entry(float(tick.price), float(fill_avg))
                if not vv.ok:
                    raise RuntimeError(vv.reason)
                self.protection.confirm_live_price(
                    position_id, mark=float(tick.price), source="canary_live_confirm"
                )
            else:
                rec_wait = self.protection.get(position_id)
                if rec_wait is None or rec_wait.state != ProtectionState.PROTECTED:
                    raise RuntimeError("LIVE_PRICE_CONFIRM_FAILED")
        except Exception as e:  # noqa: BLE001
            self.halt(f"PRICE_MONITOR_START_FAILED:{e}")
            self._emergency_market_sell(
                symbol=symbol,
                position_id=position_id,
                quantity=fill_exec,
                entry_coid=coid,
                symbol_meta=symbol_meta,
            )
            return CanaryEntryResult(
                False,
                self.halt_reason,
                position_id=position_id,
                executed_quantity=fill_exec,
                average_price=fill_avg,
                protection_id=prot.exchange_protection_id,
                halt=True,
            )

        # LIVE after confirmed fill + T1 established + monitor running + live mark.
        # Telegram is AFTER execution — never retries BUY on failure.
        prot = self.protection.get(position_id) or prot
        self._entries_count += 1
        if self._btc_balance_at_start is None:
            self._btc_balance_at_start = float(available_btc)
        self._notify_real_open(
            position_id=position_id,
            symbol=symbol,
            coid=coid,
            fill_oid=fill_oid,
            fill_exec=fill_exec,
            fill_avg=fill_avg,
            fill_fee=fill_fee,
            available_btc_before=float(available_btc),
            prot=prot,
        )

        return CanaryEntryResult(
            True,
            None,
            position_id=position_id,
            executed_quantity=fill_exec,
            average_price=fill_avg,
            protection_id=prot.exchange_protection_id,
        )

    def on_mark(
        self,
        position_id: str,
        mark: float,
        *,
        source: str = "manual",
        tick_ts: float | None = None,
    ) -> CanaryExitResult | None:
        """Apply live mark to T1. On stop breach → immediate protected MARKET SELL.

        Called by ``T1PriceMonitor`` at near-real-time frequency, not by the
        15-minute signal evaluation loop.
        """
        if self.halted:
            return None
        if self.price_stale and source != "price_monitor":
            self.audit.record(
                "MARK_IGNORED_STALE", position_id=position_id, mark=mark, source=source
            )
            return None

        rec0 = self.protection.get(position_id)
        if rec0 is not None and rec0.state == ProtectionState.AWAITING_LIVE_PRICE:
            self.protection.confirm_live_price(
                position_id, mark=float(mark), source=source
            )

        rec = self.protection.on_mark_price(position_id, mark)
        if rec.state == ProtectionState.HALTED:
            self.halt(rec.halt_reason or "PROTECTION_HALT")
            return CanaryExitResult(
                False, self.halt_reason, position_id=position_id, halt=True
            )

        if rec.t1 is not None:
            fields = rec.t1.log_fields(current_price=mark)
        else:
            fields = {
                "stop_price": rec.stop_price,
                "activated": rec.activated,
                "highest_price": rec.high_water,
                "state": rec.state.value,
                "current_price": mark,
            }
        self.audit.record(
            "PROTECTION_UPDATE",
            **{
                "position_id": position_id,
                "protection_state": rec.state.value,
                "source": source,
                "tick_ts": tick_ts,
                **fields,
            },
        )

        if position_id in self._exit_in_flight:
            return CanaryExitResult(
                False,
                "EXIT_IN_FLIGHT",
                position_id=position_id,
                trigger_price=float(mark),
                already_exiting=True,
            )
        if self.canary.phase == CanaryPhase.CANARY_COMPLETE:
            return CanaryExitResult(False, "CANARY_COMPLETE", position_id=position_id)

        if rec.state == ProtectionState.EXITING or (
            rec.t1 is not None and rec.t1.state == T1State.EXIT_TRIGGERED
        ):
            self.audit.record(
                "T1_EXIT_TRIGGERED",
                position_id=position_id,
                trigger_price=mark,
                stop_price=rec.stop_price,
                activated=rec.activated,
                source=source,
            )
            # Capture exit reason before MARKET SELL (initial SL vs trailing).
            meta = self._open_position_meta.get(position_id)
            if meta is not None:
                activated = bool(
                    (rec.t1 is not None and bool(rec.t1.activated)) or bool(rec.activated)
                )
                if activated:
                    meta["exit_reason"] = "T1_TRAILING_STOP (bot-managed trailing exit)"
                else:
                    meta["exit_reason"] = "T1_INITIAL_STOP (bot-managed initial SL)"
                meta["exit_activated"] = activated
                meta["exit_stop_price"] = float(rec.stop_price) if rec.stop_price is not None else None
                meta["exit_trigger_mark"] = float(mark)
            return self.execute_t1_exit(position_id, trigger_price=float(mark))
        return None

    def execute_t1_exit(
        self, position_id: str, *, trigger_price: float
    ) -> CanaryExitResult:
        """Protected MARKET SELL path after T1 stop breach (initial or trailing)."""
        if position_id in self._exit_in_flight:
            return CanaryExitResult(
                False,
                "EXIT_IN_FLIGHT",
                position_id=position_id,
                trigger_price=trigger_price,
                already_exiting=True,
            )
        if self.canary.phase == CanaryPhase.CANARY_COMPLETE:
            return CanaryExitResult(False, "CANARY_COMPLETE", position_id=position_id)

        if self.kill_switch is not None and not self.kill_switch.can_exit():
            self.halt("EXIT_BLOCKED_BY_KILL_SWITCH")
            return CanaryExitResult(
                False, self.halt_reason, position_id=position_id, halt=True
            )

        rec = self.protection.get(position_id)
        if rec is None:
            self.halt(f"EXIT_UNKNOWN_POSITION:{position_id}")
            return CanaryExitResult(
                False, self.halt_reason, position_id=position_id, halt=True
            )

        qty = float(rec.quantity)
        if qty <= 0:
            self.halt("EXIT_ZERO_QUANTITY")
            return CanaryExitResult(
                False, self.halt_reason, position_id=position_id, halt=True
            )

        meta = self._open_position_meta.get(position_id) or {}
        signal_ts = str(meta.get("signal_ts") or "t1-exit")
        candle_ts = str(meta.get("candle_ts") or "t1-exit")
        sym_meta = meta.get("symbol_meta")
        if sym_meta is None:
            try:
                sym_meta = self.broker.get_symbol_metadata(rec.symbol)
            except Exception as e:  # noqa: BLE001
                self.halt(f"EXIT_SYMBOL_META_UNAVAILABLE:{e}")
                return CanaryExitResult(
                    False, self.halt_reason, position_id=position_id, halt=True
                )
        from btcc.execution.symbols import normalize_order_quantity

        exit_norm = normalize_order_quantity(qty, sym_meta)
        if not exit_norm.ok:
            self.halt(f"EXIT_QTY_NORMALIZE:{exit_norm.reason}")
            return CanaryExitResult(
                False, self.halt_reason, position_id=position_id, halt=True
            )
        qty = float(exit_norm.quantity)

        coid = build_client_order_id(
            strategy_version=self.strategy_version,
            symbol=rec.symbol,
            side="SELL",
            signal_ts=signal_ts,
            candle_ts=candle_ts,
            selected_exit=T1.key,
            kind=OrderKind.EXIT,
            attempt=1,
            intent_nonce=position_id[:8],
        )

        self._exit_in_flight.add(position_id)
        self.audit.record(
            "EXIT_SUBMIT",
            position_id=position_id,
            client_order_id=coid,
            symbol=rec.symbol,
            quantity=qty,
            raw_quantity=exit_norm.raw_quantity,
            submitted_quantity=qty,
            submitted_quantity_serialized=exit_norm.serialized,
            trigger_price=trigger_price,
            stop_price=rec.stop_price,
            path="PROTECTED_MARKET_SELL",
        )

        if self.order_journal is not None:
            self.order_journal.append(
                OrderRecord(
                    order_local_id=f"{coid}:NEW",
                    intent_id=position_id,
                    client_order_id=coid,
                    symbol=rec.symbol,
                    side="SELL",
                    order_kind=OrderKind.EXIT.value,
                    status=OrderLifecycleStatus.SUBMIT_PENDING.value,
                    requested_quantity=qty,
                )
            )

        req = SpotMarketOrderRequest(
            client_order_id=coid,
            symbol=rec.symbol,
            side="SELL",
            quantity=qty,
            intent_id=position_id,
            meta={
                "t1_exit": True,
                "trigger_price": trigger_price,
                "symbol_meta": sym_meta,
                "submitted_quantity_serialized": exit_norm.serialized,
            },
        )

        fill_ok = False
        fill_exec = 0.0
        fill_avg = 0.0
        fill_oid: str | None = None
        fill_status: str | None = None
        fill_reject: str | None = None
        fill_fee = 0.0

        try:
            fill = self.broker.submit_order(req)
            fill_ok = bool(fill.ok)
            fill_exec = float(fill.executed_quantity or 0.0)
            fill_avg = float(fill.average_price or 0.0)
            fill_oid = fill.exchange_order_id
            fill_status = str(fill.order_status or "")
            fill_reject = fill.rejection_reason
            fill_fee = float(fill.fee or 0.0)
        except MexcWriteTimeout as e:
            self.audit.record("EXIT_TIMEOUT", position_id=position_id, error=str(e))
            if self.order_journal is not None:
                self.order_journal.transition(
                    coid, OrderLifecycleStatus.RECONCILE_UNKNOWN.value
                )
            try:
                ex = self.broker.get_order_status(
                    symbol=rec.symbol, orig_client_order_id=coid
                )
                executed = float(ex.executed_quantity)
                avg = float(ex.average_price or 0.0)
                if avg <= 0 and executed > 0 and hasattr(self.broker, "_vwap_from_order"):
                    avg = float(self.broker._vwap_from_order(ex))
                if executed <= 0:
                    self.halt("EXIT_TIMEOUT_FILL_UNCONFIRMED")
                    self._exit_in_flight.discard(position_id)
                    return CanaryExitResult(
                        False,
                        self.halt_reason,
                        position_id=position_id,
                        trigger_price=trigger_price,
                        halt=True,
                    )
                fill_ok = True
                fill_exec = executed
                fill_avg = avg
                fill_oid = ex.order_id
                fill_status = str(ex.status or "FILLED")
            except Exception as re:  # noqa: BLE001
                self.halt(f"EXIT_TIMEOUT_RECONCILE_FAILED:{re}")
                self._exit_in_flight.discard(position_id)
                return CanaryExitResult(
                    False, self.halt_reason, position_id=position_id, halt=True
                )
        except TradingForbiddenError as e:
            self.halt(f"EXIT_WRITE_GATE:{e}")
            self._exit_in_flight.discard(position_id)
            return CanaryExitResult(
                False, self.halt_reason, position_id=position_id, halt=True
            )
        except Exception as e:  # noqa: BLE001
            self.halt(f"EXIT_SUBMIT_FAILED:{e}")
            self._exit_in_flight.discard(position_id)
            return CanaryExitResult(
                False, self.halt_reason, position_id=position_id, halt=True
            )

        ambiguous = (
            (fill_reject or "").startswith("ACK_INCOMPLETE_RECONCILE_FAILED")
            or fill_reject
            in {
                "ORDER_ACCEPTED_FILL_UNCONFIRMED",
                "FILL_QTY_OK_VWAP_UNCONFIRMED",
            }
            or str(fill_status).upper() in {"RECONCILE_UNKNOWN", "NEW", "SUBMITTED", "PENDING", "ACK"}
        )
        if (not fill_ok or fill_exec <= 0 or fill_avg <= 0) and ambiguous:
            if self.order_journal is not None:
                self.order_journal.transition(
                    coid, OrderLifecycleStatus.RECONCILE_UNKNOWN.value
                )
            self.halt(f"EXIT_RECONCILE_UNKNOWN:{fill_reject or fill_status}")
            self._exit_in_flight.discard(position_id)
            return CanaryExitResult(
                False,
                self.halt_reason,
                position_id=position_id,
                trigger_price=trigger_price,
                halt=True,
            )

        if (not fill_ok) or fill_exec <= 0:
            self.halt("EXIT_ZERO_FILL")
            self._exit_in_flight.discard(position_id)
            return CanaryExitResult(
                False,
                self.halt_reason,
                position_id=position_id,
                trigger_price=trigger_price,
                halt=True,
            )

        if fill_avg <= 0:
            self.halt("EXIT_VWAP_UNCONFIRMED")
            self._exit_in_flight.discard(position_id)
            return CanaryExitResult(
                False,
                self.halt_reason,
                position_id=position_id,
                trigger_price=trigger_price,
                halt=True,
            )

        if self.order_journal is not None:
            self.order_journal.transition(
                coid,
                OrderLifecycleStatus.FILLED.value,
                executed_quantity=fill_exec,
                average_fill_price=fill_avg,
                exchange_order_id=fill_oid,
                remaining_quantity=max(0.0, qty - fill_exec),
            )

        self.audit.record(
            "EXIT_FILLED",
            position_id=position_id,
            client_order_id=coid,
            exchange_order_id=fill_oid,
            executed_quantity=fill_exec,
            average_price=fill_avg,
            fee=fill_fee,
            trigger_price=trigger_price,
        )
        # Telegram AFTER confirmed SELL — never blocks close / never retries SELL.
        meta_snap = dict(self._open_position_meta.get(position_id) or {})
        exit_reason = str(
            meta_snap.get("exit_reason")
            or "T1_EXIT (bot-managed)"
        )
        self._closes_count += 1
        self._notify_real_close(
            position_id=position_id,
            symbol=str(meta_snap.get("symbol") or rec.symbol),
            entry_coid=str(meta_snap.get("entry_client_order_id") or ""),
            entry_oid=meta_snap.get("entry_exchange_order_id"),
            exit_coid=coid,
            exit_oid=fill_oid,
            entry_vwap=float(meta_snap.get("entry_vwap") or 0.0),
            exit_vwap=float(fill_avg),
            qty=float(fill_exec),
            entry_fee=float(meta_snap.get("entry_fee_btc") or 0.0),
            exit_fee=float(fill_fee),
            btc_before=meta_snap.get("btc_balance_before_entry"),
            exit_reason=exit_reason,
            trigger_price=float(trigger_price),
            stop_price=(
                float(meta_snap["exit_stop_price"])
                if meta_snap.get("exit_stop_price") is not None
                else float(rec.stop_price)
                if getattr(rec, "stop_price", None) is not None
                else float(trigger_price)
            ),
            activated=bool(meta_snap.get("exit_activated")),
        )
        self.mark_canary_closed(position_id)
        result = CanaryExitResult(
            True,
            None,
            position_id=position_id,
            executed_quantity=fill_exec,
            average_price=fill_avg,
            trigger_price=trigger_price,
        )
        self._last_exit = result
        self._exit_in_flight.discard(position_id)
        return result

    def mark_canary_closed(self, position_id: str) -> None:
        self.protection.mark_closed(position_id)
        self.canary.mark_complete()
        self.stop_price_monitor()
        self._open_position_meta.pop(position_id, None)
        self.audit.record("CANARY_COMPLETE", position_id=position_id)

    def reconcile_startup(
        self,
        *,
        local_positions: list[dict[str, Any]],
        exchange_inventory: list[dict[str, Any]],
        open_protections: dict[str, Any],
    ) -> str:
        """Return NORMAL or HALTED. Never allows entries before this succeeds.

        For bot-managed T1, ``open_protections`` maps position_id →
        ``\"BOT_MANAGED:...\"`` or any truthy protection id.
        """
        self.audit.record(
            "STARTUP_RECONCILE",
            local=len(local_positions),
            inventory=len(exchange_inventory),
            protections=len(open_protections),
        )

        for inv in exchange_inventory:
            asset = str(inv.get("asset") or "")
            total = float(inv.get("total") or 0)
            if asset.upper() in _QUOTE_ASSETS or total <= 0:
                continue
            matched = next(
                (
                    p
                    for p in local_positions
                    if asset.upper() in str(p.get("symbol") or "").upper()
                ),
                None,
            )
            if matched is None:
                self.halt(f"UNEXPECTED_EXCHANGE_INVENTORY:{asset}")
                return "HALTED"
            pid = str(matched.get("position_id") or "")
            # Bot-managed: any truthy protection id (e.g. BOT_MANAGED:...) is OK.
            if not open_protections.get(pid):
                self.halt(f"UNPROTECTED_INVENTORY:{asset}")
                return "HALTED"

        for p in local_positions:
            pid = str(p.get("position_id") or "")
            if not pid:
                continue
            if not open_protections.get(pid):
                self.halt(f"LOCAL_POSITION_WITHOUT_PROTECTION:{pid}")
                return "HALTED"

        return "NORMAL"
