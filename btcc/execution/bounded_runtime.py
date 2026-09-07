"""Bounded 6-hour REAL multi-position runtime (isolated from paper + canary).

REAL_BOUNDED_6H: continuous signal eval, ≤4 positions × 25%, 100% exposure,
independent bot-managed T1 per position, hard entry cutoff at session deadline,
drain → flat reconcile → disarm → COMPLETE.

Does NOT weaken REAL_CANARY_SINGLE_SHOT. Does NOT enable unrestricted REAL.
Default WriteGate CLOSED — requires allow_trading ∧ bounded_6h_writes_armed.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from btcc.execution.bounded_session import (
    BoundedSession,
    BoundedSessionConfig,
    SessionPhase,
)
from btcc.execution.canary_runtime import (
    CanaryEntryResult,
    CanaryExitResult,
    ExecutionAuditLog,
)
from btcc.execution.client_order_id import OrderKind, build_client_order_id, intent_fingerprint
from btcc.execution.intent import DurableTradeIntent, IntentJournal, IntentStatus
from btcc.execution.kill_switch import ExecutionKillSwitch
from btcc.execution.live_price_guard import (
    DEFAULT_LIVE_CONFIRM_TIMEOUT_S,
    assert_production_price_source,
    is_forbidden_production_source,
    validate_mark_vs_entry,
    validate_protection_mark,
)
from btcc.execution.modes import ExecutionMode
from btcc.execution.mexc.write_client import MexcWriteTimeout
from btcc.execution.multi_monitor import MultiPositionMonitorRegistry
from btcc.execution.order_state import OrderJournal, OrderLifecycleStatus, OrderRecord
from btcc.execution.price_monitor import PriceTick
from btcc.execution.protection import ProtectionState, ProtectiveExitManager
from btcc.execution.risk import ExecutionRiskGate, RiskContext
from btcc.execution.sizing import size_alt_btc_position
from btcc.execution.symbols import SymbolMeta
from btcc.execution.t1 import T1, T1State
from btcc.execution.types import SpotMarketOrderRequest
from btcc.execution.write_gate import WriteGate, CLOSED_WRITE_GATE
from btcc.safety.no_trading import TradingForbiddenError

logger = logging.getLogger(__name__)

BOUNDED_STRATEGY_VERSION = "T1-ONLY-BOUNDED-6H"
_QUOTE_ASSETS = frozenset({"BTC", "USDT", "USDC"})
_ALLOWED_DUST = frozenset({"MX"})
# Reject marks that cannot be the same instrument as entry (stub 0.05 vs ~0.0009 BTC pairs).
MARK_ENTRY_RATIO_MAX = 10.0
MARK_ENTRY_RATIO_MIN = 0.1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def closed_bounded_write_gate() -> WriteGate:
    return WriteGate(
        allow_trading=False,
        canary_writes_armed=False,
        protection_api_confirmed=False,
        bounded_6h_writes_armed=False,
    )


@dataclass
class BoundedEntryResult(CanaryEntryResult):
    """Alias fields from canary entry result for shared test helpers."""


@dataclass
class BoundedExitResult(CanaryExitResult):
    pass


@dataclass
class EmergencyFlattenResult:
    """Outcome of exchange-inventory emergency flatten (never inferred flat)."""

    ok: bool
    incomplete: bool
    reason: str
    remaining: list[dict[str, Any]] = field(default_factory=list)
    sold: list[dict[str, Any]] = field(default_factory=list)
    cancelled_orders: int = 0
    local_exit_results: list[BoundedExitResult] = field(default_factory=list)


class BoundedSessionRuntime:
    """Multi-position REAL_BOUNDED_6H orchestration with independent T1 monitors."""

    def __init__(
        self,
        *,
        broker: Any,
        risk: ExecutionRiskGate,
        protection_manager: ProtectiveExitManager,
        session: BoundedSession | None = None,
        session_config: BoundedSessionConfig | None = None,
        write_gate: WriteGate | None = None,
        kill_switch: ExecutionKillSwitch | None = None,
        intent_journal: IntentJournal | None = None,
        order_journal: OrderJournal | None = None,
        audit: ExecutionAuditLog | None = None,
        monitor_registry: MultiPositionMonitorRegistry | None = None,
        trade_notifier: Any | None = None,
        strategy_version: str = BOUNDED_STRATEGY_VERSION,
    ) -> None:
        if risk.mode != ExecutionMode.REAL_BOUNDED_6H:
            raise TradingForbiddenError("BoundedSessionRuntime requires REAL_BOUNDED_6H risk mode")
        if getattr(broker, "mode", None) != ExecutionMode.REAL_BOUNDED_6H:
            raise TradingForbiddenError(
                "BoundedSessionRuntime requires RealBroker in REAL_BOUNDED_6H mode"
            )
        self.broker = broker
        self.risk = risk
        self.protection = protection_manager
        cfg = session_config or BoundedSessionConfig()
        self.session = session or BoundedSession(config=cfg)
        self.gate = write_gate or CLOSED_WRITE_GATE
        self.kill_switch = kill_switch or risk.kill_switch
        self.intent_journal = intent_journal
        self.order_journal = order_journal
        self.audit = audit or ExecutionAuditLog()
        self.strategy_version = strategy_version or self.session.config.strategy_version
        self.trade_notifier = trade_notifier
        self.halted = False
        self.halt_reason: str | None = None
        self.price_stale = False
        self._open_position_meta: dict[str, dict[str, Any]] = {}
        self._exit_in_flight: set[str] = set()
        self._entry_lock = threading.RLock()
        self._started_monotonic = time.monotonic()
        self._realized_pnl_btc = 0.0
        self._btc_balance_at_start: float | None = None
        self._last_hourly_telegram_bucket: str | None = None
        self._last_exit: BoundedExitResult | None = None
        self.stack_ref: dict[str, Any] | None = None
        self.universe_bases: set[str] = set()
        self._last_flatten_result: EmergencyFlattenResult | None = None
        self.flatten_incomplete: bool = False

        if monitor_registry is None:
            from btcc.execution.price_monitor import MexcPublicTickerSource, SequencePriceSource

            # Never default an armed live gate to the 0.05 SequencePriceSource stub.
            if self.gate.market_writes_allowed:

                def _live_src():
                    return MexcPublicTickerSource()

                _factory = _live_src
            else:
                _factory = lambda: SequencePriceSource(lambda: 0.05)  # noqa: E731

            monitor_registry = MultiPositionMonitorRegistry(
                source_factory=_factory,
                on_position_tick=self._on_position_tick,
                on_stale=self._on_monitor_stale,
                on_reconnect=self._on_monitor_reconnect,
                require_live_ticker=bool(self.gate.market_writes_allowed),
            )
        else:
            # Rebind callbacks to this runtime instance.
            monitor_registry._on_position_tick = self._on_position_tick
            monitor_registry._on_stale = self._on_monitor_stale
            monitor_registry._on_reconnect = self._on_monitor_reconnect
        self.monitors = monitor_registry

    # --- lifecycle / safety -------------------------------------------------

    def halt(self, reason: str) -> None:
        """Fail-closed: block new entries; leave exits to operator/emergency paths."""
        self.halted = True
        self.halt_reason = reason
        self.session.halt(reason)
        self.audit.record("HALT", reason=reason, phase=self.session.phase.value)
        if self.kill_switch is not None:
            self.kill_switch.halt_submissions(reason)
        self.monitors.halt_all(reason)

    def emergency_flatten_all_open(self, *, reason: str) -> EmergencyFlattenResult:
        """Exchange-inventory-complete emergency flatten for configured BTC universe.

        Steps:
          1) Attempt local tracked-position exits (best-effort).
          2) Query balances + open orders.
          3) Cancel open orders on relevant *BTC symbols.
          4) MARKET SELL every non-zero free balance for universe ALT bases.
          5) Re-query; FLAT only if no remaining relevant inventory/orders.
          6) Never sell BTC / stables / MX dust / non-universe assets.

        If flatten cannot complete → incomplete=True, never claim FLAT/COMPLETE.
        """
        from btcc.execution.symbols import normalize_order_quantity

        sold: list[dict[str, Any]] = []
        local_results: list[BoundedExitResult] = []
        cancelled = 0
        universe = {b.upper() for b in (self.universe_bases or set())}
        # Include bases from currently tracked positions even if universe unset.
        for meta in self._open_position_meta.values():
            sym = str(meta.get("symbol") or "")
            if sym.upper().endswith("BTC") and len(sym) > 3:
                universe.add(sym.upper()[:-3])

        self.audit.record(
            "EMERGENCY_FLATTEN_BEGIN",
            reason=reason,
            universe_bases=sorted(universe),
            local_open=list(self._open_position_meta.keys()),
        )

        # 1) Local tracked exits (does not alone define FLAT).
        for position_id in list(self._open_position_meta.keys()):
            meta = self._open_position_meta.get(position_id) or {}
            symbol = str(meta.get("symbol") or "")
            entry = float(meta.get("entry_vwap") or 0.0)
            trigger = entry
            if symbol:
                try:
                    from btcc.execution.price_monitor import MexcPublicTickerSource

                    trigger = float(MexcPublicTickerSource().fetch_price(symbol).price)
                except Exception as e:  # noqa: BLE001
                    self.audit.record(
                        "EMERGENCY_FLATTEN_TICKER_FALLBACK",
                        position_id=position_id,
                        error=str(e),
                    )
            if meta.get("exit_reason") is None:
                meta["exit_reason"] = f"EMERGENCY_FLATTEN ({reason})"
            try:
                local_results.append(
                    self.execute_t1_exit(position_id, trigger_price=float(trigger or 0.0))
                )
            except Exception as e:  # noqa: BLE001
                self.audit.record(
                    "EMERGENCY_FLATTEN_LOCAL_EXIT_FAILED",
                    position_id=position_id,
                    error=str(e),
                )
                local_results.append(
                    BoundedExitResult(
                        False, f"EMERGENCY_FLATTEN_FAILED:{e}", position_id=position_id
                    )
                )

        # 2–5) Exchange inventory loop (up to 2 passes).
        remaining: list[dict[str, Any]] = []
        for _pass in range(2):
            remaining = []
            bals: list[Any] = []
            try:
                if hasattr(self.broker, "get_balances_detailed"):
                    bals = list(self.broker.get_balances_detailed() or [])
                elif hasattr(self.broker, "get_balances"):
                    raw = self.broker.get_balances()
                    if isinstance(raw, dict):
                        class _B:
                            def __init__(self, asset: str, free: float):
                                self.asset = asset
                                self.free = free
                                self.locked = 0.0
                                self.total = free

                        bals = [_B(k, float(v)) for k, v in raw.items()]
            except Exception as e:  # noqa: BLE001
                res = EmergencyFlattenResult(
                    ok=False,
                    incomplete=True,
                    reason=f"EMERGENCY_FLATTEN_INCOMPLETE:BALANCE_READ_FAILED:{e}",
                    remaining=[],
                    sold=sold,
                    cancelled_orders=cancelled,
                    local_exit_results=local_results,
                )
                self._last_flatten_result = res
                self.flatten_incomplete = True
                self.audit.record("EMERGENCY_FLATTEN_INCOMPLETE", **res.__dict__)
                return res

            # Cancel open orders on universe BTC pairs.
            try:
                orders = []
                if hasattr(self.broker, "get_open_orders"):
                    orders = list(self.broker.get_open_orders() or [])
                for o in orders:
                    sym = str(
                        getattr(o, "symbol", None)
                        or (o.get("symbol") if isinstance(o, dict) else "")
                        or ""
                    ).upper()
                    if not sym.endswith("BTC"):
                        continue
                    base = sym[:-3]
                    if universe and base not in universe:
                        continue
                    oid = getattr(o, "order_id", None) or (
                        o.get("orderId") if isinstance(o, dict) else None
                    )
                    coid = getattr(o, "client_order_id", None) or (
                        o.get("clientOrderId") if isinstance(o, dict) else None
                    )
                    try:
                        if hasattr(self.broker, "cancel_order"):
                            self.broker.cancel_order(symbol=sym, order_id=oid, client_order_id=coid)
                            cancelled += 1
                    except Exception as ce:  # noqa: BLE001
                        self.audit.record(
                            "EMERGENCY_CANCEL_FAILED", symbol=sym, error=str(ce)
                        )
            except Exception as e:  # noqa: BLE001
                self.audit.record("EMERGENCY_OPEN_ORDERS_READ_FAILED", error=str(e))

            for b in bals:
                asset = str(getattr(b, "asset", "") or "").upper()
                free = float(getattr(b, "free", 0) or 0)
                locked = float(getattr(b, "locked", 0) or 0)
                total = float(getattr(b, "total", free + locked) or (free + locked))
                if total <= 0:
                    continue
                if asset in _QUOTE_ASSETS or asset in _ALLOWED_DUST:
                    continue
                if universe and asset not in universe:
                    # Non-universe ALT: do not sell; report as remaining unexpected.
                    remaining.append(
                        {
                            "asset": asset,
                            "free": free,
                            "locked": locked,
                            "total": total,
                            "reason": "NON_UNIVERSE_ALT",
                        }
                    )
                    continue
                symbol = f"{asset}BTC"
                try:
                    meta = self.broker.get_symbol_metadata(symbol)
                except Exception as e:  # noqa: BLE001
                    remaining.append(
                        {
                            "asset": asset,
                            "symbol": symbol,
                            "free": free,
                            "locked": locked,
                            "total": total,
                            "reason": f"META_UNAVAILABLE:{e}",
                        }
                    )
                    continue
                if meta.supports_market_orders is False:
                    self.audit.record(
                        "EMERGENCY_MARKET_SELL_SKIPPED",
                        symbol=symbol,
                        reason="MARKET_ORDER_TYPE_UNSUPPORTED",
                        order_types=list(meta.order_types or ()),
                        free=free,
                        locked=locked,
                    )
                    remaining.append(
                        {
                            "asset": asset,
                            "symbol": symbol,
                            "free": free,
                            "locked": locked,
                            "total": total,
                            "reason": "MARKET_ORDER_TYPE_UNSUPPORTED",
                            "order_types": list(meta.order_types or ()),
                        }
                    )
                    continue
                if free <= 0:
                    if locked > 0:
                        remaining.append(
                            {
                                "asset": asset,
                                "symbol": symbol,
                                "free": free,
                                "locked": locked,
                                "total": total,
                                "reason": "LOCKED_AFTER_CANCEL",
                            }
                        )
                    continue
                norm = normalize_order_quantity(free, meta)
                if not norm.ok or float(norm.quantity) <= 0:
                    remaining.append(
                        {
                            "asset": asset,
                            "symbol": symbol,
                            "free": free,
                            "locked": locked,
                            "total": total,
                            "reason": f"QTY_NORMALIZE:{norm.reason}",
                        }
                    )
                    continue
                try:
                    self.gate.assert_market_write("emergency_exchange_flatten")
                    fill = self.broker.submit_order(
                        SpotMarketOrderRequest(
                            client_order_id=f"BTCC-EMRG-{asset}-{int(time.time())}",
                            symbol=symbol,
                            side="SELL",
                            quantity=float(norm.quantity),
                            intent_id=f"emergency-{asset}",
                            meta={
                                "emergency": True,
                                "symbol_meta": meta,
                                "submitted_quantity_serialized": norm.serialized,
                            },
                        )
                    )
                    sold.append(
                        {
                            "symbol": symbol,
                            "qty": float(norm.quantity),
                            "ok": bool(fill.ok),
                            "executed": float(fill.executed_quantity or 0),
                            "avg": float(fill.average_price or 0) if fill.average_price else None,
                            "reject": fill.rejection_reason,
                        }
                    )
                    self.audit.record(
                        "EMERGENCY_MARKET_SELL",
                        symbol=symbol,
                        quantity=float(norm.quantity),
                        fill_ok=bool(fill.ok),
                    )
                    if not fill.ok or float(fill.executed_quantity or 0) <= 0:
                        remaining.append(
                            {
                                "asset": asset,
                                "symbol": symbol,
                                "free": free,
                                "reason": f"SELL_FAILED:{fill.rejection_reason or fill.order_status}",
                            }
                        )
                except Exception as e:  # noqa: BLE001
                    self.audit.record("EMERGENCY_SELL_FAILED", symbol=symbol, error=str(e))
                    remaining.append(
                        {
                            "asset": asset,
                            "symbol": symbol,
                            "free": free,
                            "reason": f"SELL_EXCEPTION:{e}",
                        }
                    )

            # Re-check open orders for universe symbols.
            try:
                if hasattr(self.broker, "get_open_orders"):
                    for o in list(self.broker.get_open_orders() or []):
                        sym = str(
                            getattr(o, "symbol", None)
                            or (o.get("symbol") if isinstance(o, dict) else "")
                            or ""
                        ).upper()
                        if sym.endswith("BTC") and (
                            not universe or sym[:-3] in universe
                        ):
                            remaining.append(
                                {
                                    "symbol": sym,
                                    "reason": "OPEN_ORDER_REMAINS",
                                }
                            )
            except Exception as e:  # noqa: BLE001
                remaining.append({"reason": f"OPEN_ORDERS_RECHECK_FAILED:{e}"})

            if not remaining:
                break

        incomplete = bool(remaining)
        # Relevant flat = no remaining universe ALTs / orders. Non-universe leftovers
        # also block FLAT (fail closed — operator must clear unexpected inventory).
        res = EmergencyFlattenResult(
            ok=not incomplete,
            incomplete=incomplete,
            reason=(
                "FLAT"
                if not incomplete
                else "EMERGENCY_FLATTEN_INCOMPLETE"
            ),
            remaining=remaining,
            sold=sold,
            cancelled_orders=cancelled,
            local_exit_results=local_results,
        )
        self._last_flatten_result = res
        self.flatten_incomplete = incomplete
        self.audit.record(
            "EMERGENCY_FLATTEN_END" if not incomplete else "EMERGENCY_FLATTEN_INCOMPLETE",
            reason=reason,
            ok=res.ok,
            incomplete=incomplete,
            remaining=remaining,
            sold=sold,
            cancelled_orders=cancelled,
            local_open_remaining=self.session.open_count,
        )
        return res

    def close_market_write_gate(
        self, *, reason: str, stack: dict[str, Any] | None = None
    ) -> WriteGate:
        """Canary-grade disarm: replace all WriteGate refs with CLOSED."""
        closed = closed_bounded_write_gate()
        self.gate = closed
        broker = self.broker
        if broker is not None:
            if hasattr(broker, "gate"):
                broker.gate = closed
            write = getattr(broker, "_write", None)
            if write is not None and hasattr(write, "gate"):
                write.gate = closed
        if self.kill_switch is not None:
            try:
                self.kill_switch.halt_submissions(reason)
            except Exception:  # noqa: BLE001
                pass
        if stack is not None:
            stack["gate"] = closed
            stack["armed"] = False
            stack["runtime"] = self
        elif self.stack_ref is not None:
            self.stack_ref["gate"] = closed
            self.stack_ref["armed"] = False
        self.audit.record(
            "BOUNDED_WRITES_DISARMED",
            reason=reason,
            market_writes_allowed=False,
        )
        return closed

    def reconcile_exchange_flat(
        self, *, inventory_flat_eps: float = 1e-12
    ) -> dict[str, Any]:
        """Fresh exchange reconciliation required before COMPLETE.

        Fail-closed: unexpected ALT inventory, open orders, or unresolved
        journal → not OK (caller HALTs). Allowed: BTC capital + MX dust +
        optional USDT/USDC stables (non-trading).
        """
        out: dict[str, Any] = {
            "ok": False,
            "reason": None,
            "open_orders": [],
            "inventory": None,
            "unresolved": False,
        }
        if self.session.open_count > 0:
            out["reason"] = "LOCAL_POSITIONS_REMAIN"
            return out
        if self.order_journal is not None and self.order_journal.any_unresolved():
            out["unresolved"] = True
            out["reason"] = "UNRESOLVED_JOURNAL_ORDERS"
            return out

        try:
            if hasattr(self.broker, "get_open_orders"):
                orders = list(self.broker.get_open_orders() or [])
            else:
                orders = []
        except Exception as e:  # noqa: BLE001
            out["reason"] = f"OPEN_ORDERS_UNAVAILABLE:{e}"
            return out
        out["open_orders"] = orders
        if orders:
            out["reason"] = "OPEN_EXCHANGE_ORDERS"
            return out

        try:
            if hasattr(self.broker, "get_balances_detailed"):
                bals = list(self.broker.get_balances_detailed() or [])
            else:
                out["reason"] = "INVENTORY_UNAVAILABLE"
                return out
        except Exception as e:  # noqa: BLE001
            out["reason"] = f"INVENTORY_UNAVAILABLE:{e}"
            return out

        from btcc.execution.canary_launcher import classify_preflight_inventory

        # Flat session: no base ALT inventory allowed (positions must be sold).
        inv = classify_preflight_inventory(bals, allow_base_inventory=False)
        out["inventory"] = inv
        if inv.get("unexpected"):
            out["reason"] = f"UNEXPECTED_EXCHANGE_INVENTORY:{inv['unexpected']}"
            return out
        # Explicit dust/capital sanity: MX may remain; any unexpected already caught.
        _ = inventory_flat_eps
        _ = _ALLOWED_DUST
        out["ok"] = True
        out["reason"] = "FLAT_OK"
        return out

    def advance_cutoff_if_due(self, *, now: datetime | None = None) -> SessionPhase:
        """Transition RUNNING → ENTRY_CUTOFF → DRAIN when deadline passes."""
        if self.halted:
            return SessionPhase.HALTED
        self.session.mark_entry_cutoff(now=now)
        if self.session.phase == SessionPhase.ENTRY_CUTOFF:
            self.audit.record("ENTRY_CUTOFF", deadline=self.session.deadline_utc)
            self.session.begin_drain()
            self.audit.record("DRAIN_BEGIN", open=self.session.open_count)
        return self.session.phase

    def try_complete_if_flat(self, *, stack: dict[str, Any] | None = None) -> bool:
        """Exchange flat reconcile → close WriteGate → DISARM → COMPLETE.

        WriteGate is closed before COMPLETE. Any exception after a successful
        exchange flat check still closes the WriteGate (fail-closed).
        """
        stack = stack if stack is not None else self.stack_ref
        if self.halted:
            return False
        if self.flatten_incomplete:
            self.audit.record(
                "COMPLETE_BLOCKED_FLATTEN_INCOMPLETE",
                remaining=(
                    None
                    if self._last_flatten_result is None
                    else self._last_flatten_result.remaining
                ),
            )
            return False
        phase = self.advance_cutoff_if_due()
        if phase not in (
            SessionPhase.ENTRY_CUTOFF,
            SessionPhase.DRAIN,
            SessionPhase.FLAT_RECONCILE,
            SessionPhase.RUNNING,
        ):
            if phase == SessionPhase.COMPLETE:
                return True
            return False
        # Only complete after cutoff (or drain) and flat — not mid-running with zero pos.
        if phase == SessionPhase.RUNNING and not self.session.past_entry_cutoff():
            return False
        if self.session.open_count > 0:
            return False

        recon = self.reconcile_exchange_flat()
        self.audit.record("FLAT_EXCHANGE_RECONCILE", **{k: v for k, v in recon.items() if k != "inventory"})
        if not recon.get("ok"):
            self.halt(f"FLAT_RECONCILE_FAILED:{recon.get('reason')}")
            return False

        self.session.begin_flat_reconcile()
        self.audit.record("FLAT_RECONCILE_OK", reason=recon.get("reason"))
        # From here: terminalization intended — gate must end CLOSED even on exception.
        try:
            self.close_market_write_gate(reason="SESSION_COMPLETE", stack=stack)
            assert self.gate.market_writes_allowed is False
            self.session.disarm()
            self.audit.record("DISARM")
            self.session.complete()
            self.monitors.stop_all()
            self.audit.record(
                "SESSION_COMPLETE",
                market_writes_allowed=bool(self.gate.market_writes_allowed),
                **self.session.to_dict(),
            )
            return True
        except Exception as e:  # noqa: BLE001
            self.close_market_write_gate(reason=f"FINALIZE_EXCEPTION:{e}", stack=stack)
            self.halt(f"FINALIZE_EXCEPTION:{e}")
            return False

    # --- monitors -----------------------------------------------------------

    def _on_position_tick(self, position_id: str, tick: PriceTick) -> None:
        self.price_stale = False
        self.on_mark(
            position_id,
            tick.price,
            source="price_monitor",
            tick_ts=tick.ts_wall,
            tick=tick,
        )

    def _on_monitor_stale(self, symbol: str, age_s: float) -> None:
        self.price_stale = True
        self.audit.record("PRICE_STALE", symbol=symbol, age_s=age_s)
        if self.kill_switch is not None and self.kill_switch.can_submit_new_order():
            self.kill_switch.halt_submissions(f"PRICE_STALE:{symbol}:age={age_s:.3f}s")
        # No valid live protection without fresh marks — flatten open inventory.
        if not self.halted and self.session.open_count > 0:
            self.halt(f"PRICE_FEED_STALE:{symbol}:age={age_s:.3f}s")
            self.emergency_flatten_all_open(reason=f"PRICE_FEED_STALE:{symbol}")

    def _on_monitor_reconnect(self, symbol: str, attempts: int) -> None:
        self.audit.record("PRICE_RECONNECTED", symbol=symbol, attempts=attempts)

    # --- telegram (observability only) --------------------------------------

    def _peek_btc_available(self) -> float | None:
        try:
            if hasattr(self.broker, "get_account_state"):
                st = self.broker.get_account_state()
                return float(
                    getattr(st, "available", None)
                    or getattr(st, "available_balance", 0.0)
                    or 0.0
                )
            if hasattr(self.broker, "get_balances"):
                bals = self.broker.get_balances()
                if isinstance(bals, dict):
                    return float(bals.get("BTC") or bals.get("btc") or 0.0)
        except Exception as e:  # noqa: BLE001
            logger.warning("btc balance peek failed (non-blocking): %s", e)
        return None

    def _notify_real_open(self, **kwargs: Any) -> None:
        if self.trade_notifier is None:
            return
        try:
            position_id = kwargs["position_id"]
            symbol = kwargs["symbol"]
            coid = kwargs["coid"]
            fill_oid = kwargs.get("fill_oid")
            fill_exec = kwargs["fill_exec"]
            fill_avg = kwargs["fill_avg"]
            fill_fee = kwargs.get("fill_fee", 0.0)
            available_btc_before = kwargs["available_btc_before"]
            prot = kwargs["prot"]
            t1 = prot.t1 if getattr(prot, "t1", None) is not None else None
            btc_after = self._peek_btc_available()
            protection_verified = bool(kwargs.get("protection_verified"))
            if getattr(prot, "meta", None):
                protection_verified = protection_verified or bool(
                    prot.meta.get("live_price_verified")
                )
            if getattr(prot, "state", None) == ProtectionState.PROTECTED:
                protection_verified = protection_verified or bool(
                    (getattr(prot, "meta", None) or {}).get("live_price_verified")
                )
            payload = {
                "symbol": symbol,
                "entry_time": _utc_now(),
                "executed_quantity": fill_exec,
                "entry_vwap": fill_avg,
                "btc_allocated": float(fill_exec) * float(fill_avg),
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
                "protection_verified": protection_verified,
                "protection_state": getattr(getattr(prot, "state", None), "value", None),
            }
            ok = bool(self.trade_notifier.trade_opened(payload))
            self.audit.record("TELEGRAM_OPEN", sent=ok, client_order_id=coid)
        except Exception as e:  # noqa: BLE001
            logger.error("TELEGRAM_OPEN failed (non-blocking): %s", e)
            self.audit.record("TELEGRAM_OPEN_FAILED", error=str(e))

    def _notify_real_close(self, **kwargs: Any) -> None:
        if self.trade_notifier is None:
            return
        try:
            entry_vwap = float(kwargs["entry_vwap"])
            exit_vwap = float(kwargs["exit_vwap"])
            qty = float(kwargs["qty"])
            entry_fee = float(kwargs.get("entry_fee") or 0.0)
            exit_fee = float(kwargs.get("exit_fee") or 0.0)
            gross = (exit_vwap - entry_vwap) * qty
            fees = entry_fee + exit_fee
            net = gross - fees
            btc_after = self._peek_btc_available()
            stop_px = kwargs.get("stop_price")
            if stop_px is None:
                stop_px = kwargs.get("trigger_price")
            payload = {
                "symbol": kwargs["symbol"],
                "entry_vwap": entry_vwap,
                "exit_vwap": exit_vwap,
                "executed_quantity": qty,
                "gross_pnl_btc": gross,
                "fees_btc": fees,
                "net_pnl_btc": net,
                "btc_before": kwargs.get("btc_before"),
                "btc_after": btc_after,
                "exit_reason": kwargs.get("exit_reason"),
                "activated": kwargs.get("activated"),
                "trigger_price": stop_px,
                "stop_price": stop_px,
                "execution_price": exit_vwap,
                "t1": {
                    "stop_loss_pct": T1.stop_loss_pct,
                    "activation_pct": T1.activation_pct,
                    "trailing_pct": T1.trailing_pct,
                },
                "entry_client_order_id": kwargs.get("entry_coid"),
                "entry_exchange_order_id": kwargs.get("entry_oid"),
                "exit_client_order_id": kwargs.get("exit_coid"),
                "exit_exchange_order_id": kwargs.get("exit_oid"),
                "position_id": kwargs.get("position_id"),
                "strategy_version": self.strategy_version,
                "asof_utc": _utc_now(),
            }
            ok = bool(self.trade_notifier.trade_closed(payload))
            self._realized_pnl_btc += float(net)
            self.audit.record("TELEGRAM_CLOSE", sent=ok, net_pnl_btc=net)
        except Exception as e:  # noqa: BLE001
            logger.error("TELEGRAM_CLOSE failed (non-blocking): %s", e)
            self.audit.record("TELEGRAM_CLOSE_FAILED", error=str(e))

    def _open_positions_for_hourly(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for pid in list(self.session.open_position_ids):
            meta = self._open_position_meta.get(pid) or {}
            rec = self.protection.get(pid) if hasattr(self.protection, "get") else None
            t1 = getattr(rec, "t1", None) if rec is not None else None
            cfg = getattr(t1, "config", None) if t1 is not None else None
            out.append(
                {
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
            )
        return out

    def maybe_send_hourly_report(self) -> bool:
        if self.trade_notifier is None:
            return False
        try:
            from btcc.telegram.real_trades import brt_hour_bucket, compute_hourly_real_stats

            now = datetime.now(timezone.utc)
            bucket = brt_hour_bucket(now)
            if self._last_hourly_telegram_bucket is None:
                self._last_hourly_telegram_bucket = bucket
                self.audit.record("TELEGRAM_HOURLY_SEEDED", bucket=bucket, sent=False)
                return False
            if self._last_hourly_telegram_bucket == bucket:
                return False
            opens = self._open_positions_for_hourly()
            status = "HALTED" if self.halted else (
                "COMPLETE" if self.session.phase == SessionPhase.COMPLETE else "MONITORING"
            )
            stats = compute_hourly_real_stats(
                btc_balance=self._peek_btc_available(),
                n_entered=self.session.entries_count,
                n_closed=self.session.closes_count,
                realized_pnl_btc=self._realized_pnl_btc,
                open_position=opens[0] if len(opens) == 1 else None,
                open_positions=opens,
                uptime_s=time.monotonic() - self._started_monotonic,
                execution_status=status,
                canary_phase=self.session.phase.value,
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

    # --- emergency / helpers ------------------------------------------------

    def _normalize_symbol(self, symbol: str) -> str:
        return str(symbol).upper().replace("/", "").replace("-", "").replace("_", "")

    def _confirm_live_price_or_fail(
        self,
        *,
        position_id: str,
        symbol: str,
        entry: float,
        timeout_s: float = DEFAULT_LIVE_CONFIRM_TIMEOUT_S,
    ) -> bool:
        """Block until a validated mark for this symbol confirms PROTECTED, or fail."""
        deadline = time.monotonic() + float(timeout_s)
        mon = self.monitors.monitor_for(position_id)
        while time.monotonic() < deadline:
            rec = self.protection.get(position_id)
            if rec is not None and rec.state == ProtectionState.PROTECTED and bool(
                (rec.meta or {}).get("live_price_verified")
            ):
                return True
            if self.halted:
                return False
            tick = None
            if mon is not None:
                try:
                    tick = mon.poll_once()
                except Exception as e:  # noqa: BLE001
                    self.audit.record(
                        "LIVE_PRICE_CONFIRM_POLL_ERROR",
                        position_id=position_id,
                        error=str(e),
                    )
            if tick is not None:
                # on_mark via poll path: poll_once already called on_tick → on_mark.
                # Also apply direct confirmation if still awaiting (race with thread).
                rec2 = self.protection.get(position_id)
                if rec2 is not None and rec2.state == ProtectionState.AWAITING_LIVE_PRICE:
                    self.on_mark(
                        position_id,
                        tick.price,
                        source="live_confirm",
                        tick_ts=tick.ts_wall,
                        tick=tick,
                    )
            time.sleep(0.05)
        rec = self.protection.get(position_id)
        if rec is not None and rec.state == ProtectionState.PROTECTED:
            return True
        self.halt(f"LIVE_PRICE_CONFIRM_TIMEOUT:{symbol}")
        self.audit.record(
            "LIVE_PRICE_CONFIRM_FAILED",
            position_id=position_id,
            symbol=symbol,
            entry=entry,
            timeout_s=timeout_s,
            state=None if rec is None else rec.state.value,
        )
        return False

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

    # --- entry --------------------------------------------------------------

    def try_entry(
        self,
        *,
        symbol: str,
        price_alt_btc: float,
        s_value: float,
        signal_ts: str,
        candle_ts: str,
        available_btc: float,
        symbol_meta: SymbolMeta,
        locked_btc: float = 0.0,
        account_ctx: RiskContext | None = None,
        intent_nonce: str = "",
        now: datetime | None = None,
    ) -> BoundedEntryResult:
        """Attempt a BUY. Serialized under entry lock (race-safe vs max 4 / 100%)."""
        with self._entry_lock:
            return self._try_entry_locked(
                symbol=symbol,
                price_alt_btc=price_alt_btc,
                s_value=s_value,
                signal_ts=signal_ts,
                candle_ts=candle_ts,
                available_btc=available_btc,
                symbol_meta=symbol_meta,
                locked_btc=locked_btc,
                account_ctx=account_ctx,
                intent_nonce=intent_nonce,
                now=now,
            )

    def _try_entry_locked(
        self,
        *,
        symbol: str,
        price_alt_btc: float,
        s_value: float,
        signal_ts: str,
        candle_ts: str,
        available_btc: float,
        symbol_meta: SymbolMeta,
        locked_btc: float,
        account_ctx: RiskContext | None,
        intent_nonce: str,
        now: datetime | None,
    ) -> BoundedEntryResult:
        self.advance_cutoff_if_due(now=now)
        if self.halted or self.session.phase == SessionPhase.HALTED:
            return BoundedEntryResult(False, "HALTED", halt=True)
        if self.session.phase == SessionPhase.COMPLETE:
            return BoundedEntryResult(False, "SESSION_COMPLETE")
        if self.session.phase != SessionPhase.RUNNING:
            return BoundedEntryResult(
                False, f"NO_ENTRY_PHASE_{self.session.phase.value}"
            )
        if self.session.past_entry_cutoff(now=now):
            self.session.mark_entry_cutoff(now=now)
            self.session.begin_drain()
            return BoundedEntryResult(False, "ENTRY_CUTOFF")
        if not self.kill_switch.can_submit_new_order():
            return BoundedEntryResult(False, "EXECUTION_KILL_SWITCH")
        if self.order_journal is not None and self.order_journal.any_unresolved():
            unresolved = self.order_journal.unresolved_orders()
            detail = ",".join(f"{o.client_order_id}:{o.status}" for o in unresolved[:4])
            self.halt(f"UNRESOLVED_PRIOR_ORDER:{detail}")
            return BoundedEntryResult(False, self.halt_reason, halt=True)
        if float(s_value) < float(self.session.config.require_s_min):
            return BoundedEntryResult(False, "S_BELOW_THRESHOLD")

        # Pre-submit gate: fresh exchangeInfo MARKET capability (not stale caller meta).
        from btcc.execution.symbols import t1_market_exit_capability

        live_meta = symbol_meta
        broker_client = getattr(self.broker, "_client", None)
        if hasattr(self.broker, "get_symbol_metadata"):
            try:
                live_meta = self.broker.get_symbol_metadata(symbol, use_cache=False)
            except TypeError:
                try:
                    live_meta = self.broker.get_symbol_metadata(symbol)
                except Exception as e:  # noqa: BLE001
                    if broker_client is None:
                        # Unit-test stacks without a read client keep caller meta.
                        live_meta = symbol_meta
                    else:
                        self.audit.record(
                            "ENTRY_REJECTED_META_UNAVAILABLE",
                            symbol=symbol,
                            error=str(e),
                        )
                        return BoundedEntryResult(False, f"SYMBOL_META_UNAVAILABLE:{e}")
            except Exception as e:  # noqa: BLE001
                if broker_client is None:
                    live_meta = symbol_meta
                else:
                    self.audit.record(
                        "ENTRY_REJECTED_META_UNAVAILABLE",
                        symbol=symbol,
                        error=str(e),
                    )
                    return BoundedEntryResult(False, f"SYMBOL_META_UNAVAILABLE:{e}")
        symbol_meta = live_meta

        ok_mkt, mkt_reason = t1_market_exit_capability(symbol_meta)
        if not ok_mkt:
            self.audit.record(
                "ENTRY_REJECTED_MARKET_UNSUPPORTED",
                symbol=symbol,
                reason=mkt_reason,
                order_types=list(symbol_meta.order_types or ()),
            )
            return BoundedEntryResult(False, mkt_reason)

        open_count = self.session.open_count
        if open_count >= int(self.session.config.max_positions):
            return BoundedEntryResult(False, "MAX_POSITIONS")

        # Available for sizing: free BTC only (locked funds excluded).
        free_btc = max(0.0, float(available_btc))
        _ = locked_btc  # explicit: locked must not inflate allocation
        open_exposure = float(open_count) * float(self.session.config.allocation_pct)
        size = size_alt_btc_position(
            available_btc=free_btc,
            price_alt_btc=price_alt_btc,
            meta=symbol_meta,
            allocation_pct=self.session.config.allocation_pct,
            open_exposure_pct=open_exposure,
            max_aggregate_exposure=self.session.config.max_total_exposure,
        )
        if not size.ok:
            self.audit.record("SIZE_REJECT", reason=size.reason, symbol=symbol)
            return BoundedEntryResult(False, size.reason)

        nonce = intent_nonce or f"b6h-{self.session.entries_count + 1}-{signal_ts}"
        position_id = intent_fingerprint(
            strategy_version=self.strategy_version,
            symbol=symbol,
            side="BUY",
            signal_ts=signal_ts,
            candle_ts=candle_ts,
            selected_exit=T1.key,
            kind=OrderKind.ENTRY,
            intent_nonce=nonce,
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
            intent_nonce=nonce,
        )
        intent = DurableTradeIntent(
            intent_id=position_id,
            client_order_id=coid,
            strategy_version=self.strategy_version,
            symbol=symbol,
            side="BUY",
            requested_allocation_pct=self.session.config.allocation_pct,
            requested_quantity=size.quantity,
            signal_ts=signal_ts,
            candle_ts=candle_ts,
            selected_exit=T1.key,
            execution_mode=ExecutionMode.REAL_BOUNDED_6H.value,
            status=IntentStatus.CREATED.value,
            order_kind=OrderKind.ENTRY.value,
            current_strategy_state={"s_value": s_value, "exit": T1.key},
            meta={
                "notional_btc": size.notional_btc,
                "production_path": "T1_ONLY",
                "raw_quantity": size.raw_quantity,
                "submitted_quantity": size.quantity,
                "submitted_quantity_serialized": size.quantity_serialized,
                "session_phase": self.session.phase.value,
            },
        )
        ctx = account_ctx or RiskContext(
            symbol_meta=symbol_meta,
            open_position_count=open_count,
            current_exposure_pct=open_exposure,
            market_data_stale=False,
            account_data_stale=False,
        )
        if ctx.symbol_meta is None:
            ctx.symbol_meta = symbol_meta
        ctx.open_position_count = open_count
        ctx.current_exposure_pct = open_exposure

        decision = self.risk.validate_intent(intent, ctx)
        if not decision.approved:
            self.audit.record("RISK_REJECT", reason=decision.reason, intent_id=position_id)
            return BoundedEntryResult(False, decision.reason)

        # Double-check caps under lock after risk (race defense).
        if self.session.open_count >= int(self.session.config.max_positions):
            return BoundedEntryResult(False, "MAX_POSITIONS")
        projected = float(self.session.open_count) * float(self.session.config.allocation_pct)
        projected += float(self.session.config.allocation_pct)
        if projected > float(self.session.config.max_total_exposure) + 1e-12:
            return BoundedEntryResult(False, "MAX_TOTAL_EXPOSURE")

        intent.status = IntentStatus.PERSISTED.value
        if self.intent_journal is not None:
            self.intent_journal.append(intent)

        self.audit.record(
            "ENTRY_SUBMIT",
            intent_id=position_id,
            client_order_id=coid,
            symbol=symbol,
            requested_quantity=size.quantity,
            open_before=open_count,
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
            return BoundedEntryResult(False, self.halt_reason, halt=True)

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
            try:
                ex = self.broker.get_order_status(symbol=symbol, orig_client_order_id=coid)
                executed = float(ex.executed_quantity)
                avg = float(ex.average_price or 0.0)
                if avg <= 0 and executed > 0 and hasattr(self.broker, "_vwap_from_order"):
                    avg = float(self.broker._vwap_from_order(ex))
                if executed <= 0:
                    self.halt("ENTRY_TIMEOUT_FILL_UNCONFIRMED")
                    return BoundedEntryResult(False, self.halt_reason, halt=True)
                fill_ok = True
                fill_exec = executed
                fill_avg = avg
                fill_oid = ex.order_id
                fill_status = str(ex.status or "FILLED")
            except Exception as re:  # noqa: BLE001
                self.halt(f"TIMEOUT_RECONCILE_FAILED:{re}")
                return BoundedEntryResult(False, self.halt_reason, halt=True)
        except TradingForbiddenError as e:
            return BoundedEntryResult(False, f"WRITE_GATE:{e}")
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "reject" in msg.lower() or "Insufficient" in msg:
                return BoundedEntryResult(False, f"ENTRY_REJECTED:{e}")
            self.audit.record("ENTRY_ERROR", error=str(e))
            self.halt(f"ENTRY_UNKNOWN_ERROR:{e}")
            return BoundedEntryResult(False, self.halt_reason, halt=True)

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
            return BoundedEntryResult(False, self.halt_reason, halt=True)

        if (not fill_ok) or fill_exec <= 0:
            return BoundedEntryResult(False, "ZERO_FILL")

        if fill_avg <= 0:
            if self.order_journal is not None:
                self.order_journal.transition(
                    coid, OrderLifecycleStatus.RECONCILE_UNKNOWN.value
                )
            self.halt("ENTRY_VWAP_UNCONFIRMED")
            return BoundedEntryResult(False, self.halt_reason, halt=True)

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

        self.audit.record(
            "ENTRY_FILLED",
            intent_id=position_id,
            client_order_id=coid,
            exchange_order_id=fill_oid,
            executed_quantity=fill_exec,
            average_price=fill_avg,
            fee=fill_fee,
        )

        try:
            prot = self.protection.establish(
                position_id=position_id,
                symbol=symbol,
                quantity=fill_exec,
                entry_price=fill_avg,
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
            return BoundedEntryResult(
                False,
                self.halt_reason,
                position_id=position_id,
                executed_quantity=fill_exec,
                average_price=fill_avg,
                halt=True,
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
            "allocation_pct": self.session.config.allocation_pct,
        }
        if prot.t1 is not None:
            prot.t1.meta.update(self._open_position_meta[position_id])

        try:
            self.protection.mark_monitor_armed(position_id)
            # Reject forbidden stubs before the monitor thread can poison T1.
            if getattr(self.monitors, "require_live_ticker", False):
                probe = self.monitors._source_factory()
                assert_production_price_source(probe, context=f"pre_attach:{symbol}")
            mon = self.monitors.attach_position(position_id, symbol)
            from btcc.execution.price_monitor import PriceMonitorState

            if mon.state != PriceMonitorState.RUNNING:
                raise RuntimeError(f"monitor state={mon.state.value}")
            if getattr(self.monitors, "require_live_ticker", False) and is_forbidden_production_source(
                mon.source
            ):
                raise RuntimeError(
                    f"FORBIDDEN_PRICE_SOURCE_ATTACHED:{type(mon.source).__name__}"
                )
        except Exception as e:  # noqa: BLE001
            self.halt(f"PRICE_MONITOR_START_FAILED:{e}")
            self._emergency_market_sell(
                symbol=symbol,
                position_id=position_id,
                quantity=fill_exec,
                entry_coid=coid,
                symbol_meta=symbol_meta,
            )
            return BoundedEntryResult(
                False,
                self.halt_reason,
                position_id=position_id,
                executed_quantity=fill_exec,
                average_price=fill_avg,
                protection_id=prot.exchange_protection_id,
                halt=True,
            )

        # Hard invariant: never advertise PROTECTED without a validated live mark.
        live_ok = self._confirm_live_price_or_fail(
            position_id=position_id,
            symbol=symbol,
            entry=fill_avg,
            timeout_s=DEFAULT_LIVE_CONFIRM_TIMEOUT_S,
        )
        prot = self.protection.get(position_id) or prot
        if not live_ok:
            self.halt(self.halt_reason or "LIVE_PRICE_CONFIRM_FAILED")
            self._emergency_market_sell(
                symbol=symbol,
                position_id=position_id,
                quantity=fill_exec,
                entry_coid=coid,
                symbol_meta=symbol_meta,
            )
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
                protection_verified=False,
            )
            return BoundedEntryResult(
                False,
                self.halt_reason,
                position_id=position_id,
                executed_quantity=fill_exec,
                average_price=fill_avg,
                protection_id=prot.exchange_protection_id,
                halt=True,
            )

        self.session.register_open(position_id)
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
            protection_verified=True,
        )
        self.audit.record(
            "PROTECTION_ESTABLISHED",
            intent_id=position_id,
            stop_price=prot.stop_price,
            open_count=self.session.open_count,
            live_price_verified=True,
            protection_state=prot.state.value,
        )
        return BoundedEntryResult(
            True,
            None,
            position_id=position_id,
            executed_quantity=fill_exec,
            average_price=fill_avg,
            protection_id=prot.exchange_protection_id,
        )

    # --- mark / exit --------------------------------------------------------

    def on_mark(
        self,
        position_id: str,
        mark: float,
        *,
        source: str = "manual",
        tick_ts: float | None = None,
        tick: PriceTick | None = None,
    ) -> BoundedExitResult | None:
        if self.halted:
            return None
        if self.price_stale and source != "price_monitor" and source != "live_confirm":
            self.audit.record(
                "MARK_IGNORED_STALE", position_id=position_id, mark=mark, source=source
            )
            return None

        meta = self._open_position_meta.get(position_id) or {}
        entry = float(meta.get("entry_vwap") or 0.0)
        expected_symbol = str(meta.get("symbol") or "")
        v = validate_protection_mark(
            mark=float(mark),
            entry=entry if entry > 0 else float(mark),
            tick=tick,
            expected_symbol=expected_symbol or None,
        )
        if entry > 0:
            v2 = validate_mark_vs_entry(float(mark), entry)
            if not v2.ok:
                v = v2
        if not v.ok:
            self.audit.record(
                "MARK_REJECTED",
                position_id=position_id,
                mark=mark,
                entry_vwap=entry,
                reason=v.reason,
                source=source,
            )
            self.halt(v.reason)
            return BoundedExitResult(
                False, self.halt_reason, position_id=position_id, halt=True
            )

        rec0 = self.protection.get(position_id)
        if rec0 is not None and rec0.state == ProtectionState.AWAITING_LIVE_PRICE:
            # First validated mark promotes to PROTECTED, then T1 may consume it.
            self.protection.confirm_live_price(
                position_id, mark=float(mark), source=source
            )
            self.audit.record(
                "LIVE_PRICE_VERIFIED",
                position_id=position_id,
                mark=mark,
                source=source,
                tick_ts=tick_ts,
            )

        rec = self.protection.on_mark_price(position_id, mark)
        if rec.state == ProtectionState.HALTED:
            self.halt(rec.halt_reason or "PROTECTION_HALT")
            return BoundedExitResult(
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
            return BoundedExitResult(
                False,
                "EXIT_IN_FLIGHT",
                position_id=position_id,
                trigger_price=float(mark),
                already_exiting=True,
            )
        if self.session.phase == SessionPhase.COMPLETE:
            return BoundedExitResult(False, "SESSION_COMPLETE", position_id=position_id)

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
    ) -> BoundedExitResult:
        if position_id in self._exit_in_flight:
            return BoundedExitResult(
                False,
                "EXIT_IN_FLIGHT",
                position_id=position_id,
                trigger_price=trigger_price,
                already_exiting=True,
            )
        if self.session.phase == SessionPhase.COMPLETE:
            return BoundedExitResult(False, "SESSION_COMPLETE", position_id=position_id)

        if self.kill_switch is not None and not self.kill_switch.can_exit():
            self.halt("EXIT_BLOCKED_BY_KILL_SWITCH")
            return BoundedExitResult(
                False, self.halt_reason, position_id=position_id, halt=True
            )

        rec = self.protection.get(position_id)
        if rec is None:
            self.halt(f"EXIT_UNKNOWN_POSITION:{position_id}")
            return BoundedExitResult(
                False, self.halt_reason, position_id=position_id, halt=True
            )

        qty = float(rec.quantity)
        if qty <= 0:
            self.halt("EXIT_ZERO_QUANTITY")
            return BoundedExitResult(
                False, self.halt_reason, position_id=position_id, halt=True
            )

        meta = self._open_position_meta.get(position_id) or {}
        signal_ts = str(meta.get("signal_ts") or "t1-exit")
        candle_ts = str(meta.get("candle_ts") or "t1-exit")
        sym_meta = meta.get("symbol_meta")
        broker_client = getattr(self.broker, "_client", None)
        if broker_client is not None and hasattr(self.broker, "get_symbol_metadata"):
            try:
                try:
                    sym_meta = self.broker.get_symbol_metadata(rec.symbol, use_cache=False)
                except TypeError:
                    sym_meta = self.broker.get_symbol_metadata(rec.symbol)
            except Exception as e:  # noqa: BLE001
                self.halt(f"EXIT_SYMBOL_META_UNAVAILABLE:{e}")
                return BoundedExitResult(
                    False, self.halt_reason, position_id=position_id, halt=True
                )
        if sym_meta is None:
            self.halt("EXIT_SYMBOL_META_UNAVAILABLE")
            return BoundedExitResult(
                False, self.halt_reason, position_id=position_id, halt=True
            )
        from btcc.execution.symbols import normalize_order_quantity, t1_market_exit_capability

        ok_mkt, mkt_reason = t1_market_exit_capability(sym_meta)
        if not ok_mkt:
            # Must never POST MARKET for LTCBTC-class symbols.
            self.audit.record(
                "EXIT_MARKET_UNSUPPORTED",
                position_id=position_id,
                symbol=rec.symbol,
                reason=mkt_reason,
                order_types=list(getattr(sym_meta, "order_types", None) or ()),
            )
            self.flatten_incomplete = True
            self.halt(f"EXIT_MARKET_UNSUPPORTED:{rec.symbol}:{mkt_reason}")
            return BoundedExitResult(
                False, self.halt_reason, position_id=position_id, halt=True
            )

        exit_norm = normalize_order_quantity(qty, sym_meta)
        if not exit_norm.ok:
            self.halt(f"EXIT_QTY_NORMALIZE:{exit_norm.reason}")
            return BoundedExitResult(
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
                    return BoundedExitResult(
                        False, self.halt_reason, position_id=position_id, halt=True
                    )
                fill_ok = True
                fill_exec = executed
                fill_avg = avg
                fill_oid = ex.order_id
                fill_status = str(ex.status or "FILLED")
            except Exception as re:  # noqa: BLE001
                self.halt(f"EXIT_TIMEOUT_RECONCILE_FAILED:{re}")
                self._exit_in_flight.discard(position_id)
                return BoundedExitResult(
                    False, self.halt_reason, position_id=position_id, halt=True
                )
        except TradingForbiddenError as e:
            self.halt(f"EXIT_WRITE_GATE:{e}")
            self._exit_in_flight.discard(position_id)
            return BoundedExitResult(
                False, self.halt_reason, position_id=position_id, halt=True
            )
        except Exception as e:  # noqa: BLE001
            self.halt(f"EXIT_SUBMIT_FAILED:{e}")
            self._exit_in_flight.discard(position_id)
            return BoundedExitResult(
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
            return BoundedExitResult(
                False, self.halt_reason, position_id=position_id, halt=True
            )

        if (not fill_ok) or fill_exec <= 0:
            self.halt("EXIT_ZERO_FILL")
            self._exit_in_flight.discard(position_id)
            return BoundedExitResult(
                False, self.halt_reason, position_id=position_id, halt=True
            )

        if fill_avg <= 0:
            self.halt("EXIT_VWAP_UNCONFIRMED")
            self._exit_in_flight.discard(position_id)
            return BoundedExitResult(
                False, self.halt_reason, position_id=position_id, halt=True
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
        meta_snap = dict(self._open_position_meta.get(position_id) or {})
        exit_reason = str(meta_snap.get("exit_reason") or "T1_EXIT (bot-managed)")
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
        self.mark_position_closed(position_id)
        result = BoundedExitResult(
            True,
            None,
            position_id=position_id,
            executed_quantity=fill_exec,
            average_price=fill_avg,
            trigger_price=trigger_price,
        )
        self._last_exit = result
        self._exit_in_flight.discard(position_id)
        self.try_complete_if_flat()
        return result

    def mark_position_closed(self, position_id: str) -> None:
        self.protection.mark_closed(position_id)
        self.session.register_close(position_id)
        self.monitors.detach_position(position_id)
        self._open_position_meta.pop(position_id, None)
        self.audit.record(
            "POSITION_CLOSED",
            position_id=position_id,
            open_remaining=self.session.open_count,
            phase=self.session.phase.value,
        )

    def reconcile_startup(
        self,
        *,
        local_positions: list[dict[str, Any]],
        exchange_inventory: list[dict[str, Any]],
        open_protections: dict[str, Any],
        open_orders: list[dict[str, Any]] | None = None,
        unresolved_journal: bool | None = None,
    ) -> str:
        """Return NORMAL or HALTED. Never blind re-entry on missing local state."""
        self.audit.record(
            "STARTUP_RECONCILE",
            local=len(local_positions),
            inventory=len(exchange_inventory),
            protections=len(open_protections),
            open_orders=len(open_orders or []),
        )
        if unresolved_journal or (
            self.order_journal is not None and self.order_journal.any_unresolved()
        ):
            self.halt("STARTUP_UNRESOLVED_ORDERS")
            return "HALTED"

        for o in open_orders or []:
            # Any unexpected open write order at startup → operator reconcile.
            self.halt(f"STARTUP_OPEN_ORDER:{o.get('client_order_id') or o.get('order_id')}")
            return "HALTED"

        for inv in exchange_inventory:
            asset = str(inv.get("asset") or "")
            total = float(inv.get("total") or 0)
            if asset.upper() in _QUOTE_ASSETS or asset.upper() in _ALLOWED_DUST or total <= 0:
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
            if not open_protections.get(pid):
                self.halt(f"UNPROTECTED_INVENTORY:{asset}")
                return "HALTED"

        for p in local_positions:
            pid = str(p.get("position_id") or "")
            if not open_protections.get(pid):
                self.halt(f"LOCAL_POSITION_UNPROTECTED:{pid}")
                return "HALTED"

        return "NORMAL"
