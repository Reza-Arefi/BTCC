"""Order Management System — persist intent BEFORE any submission attempt.

Stage 4 adds TEST-mode lifecycle against SimulatedBroker:
  SUBMISSION_ATTEMPTED ≠ ORDER_ACKNOWLEDGED ≠ FILLED
Timeout after acceptance → RECONCILE_REQUIRED (no immediate resubmit).
"""

from __future__ import annotations

from typing import Any

from btcc.execution.broker import Broker
from btcc.execution.client_order_id import OrderKind, build_client_order_id, intent_fingerprint
from btcc.execution.intent import (
    DurableTradeIntent,
    IntentJournal,
    IntentPersistenceError,
    IntentStatus,
)
from btcc.execution.kill_switch import ExecutionKillSwitch
from btcc.execution.modes import ExecutionMode
from btcc.execution.order_fsm import ReconciliationState
from btcc.execution.order_state import OrderJournal, OrderLifecycleStatus, OrderRecord
from btcc.execution.risk import ExecutionRiskGate, RiskContext
from btcc.execution.safety_state import (
    RECONCILIATION_MISMATCH,
    ExecutionSafetyState,
    ExecutionSafetyStatus,
)
from btcc.execution.sim.scenarios import SimNetworkError, SimRejectedError, SimTimeoutError
from btcc.execution.types import FillReport, OrderRequest, TradeIntent
from btcc.safety.no_trading import TradingForbiddenError
from btcc.sim.exits import StrategySpec


class OrderManagementSystem:
    """Routes approved intents to the configured broker. No silent REAL→PAPER fallback."""

    def __init__(
        self,
        broker: Broker,
        risk: ExecutionRiskGate,
        *,
        intent_journal: IntentJournal | None = None,
        order_journal: OrderJournal | None = None,
        kill_switch: ExecutionKillSwitch | None = None,
        strategy_version: str = "E-v1-25pct-PAPER",
        safety: ExecutionSafetyState | None = None,
    ) -> None:
        if broker.mode != risk.mode:
            raise TradingForbiddenError(
                f"OMS/broker mode mismatch: broker={broker.mode.value} risk={risk.mode.value}"
            )
        # Simulated broker must only pair with TEST mode
        if broker.mode == ExecutionMode.TEST and risk.mode != ExecutionMode.TEST:
            raise TradingForbiddenError("SimulatedBroker requires ExecutionMode.TEST")
        if risk.mode == ExecutionMode.TEST and broker.mode != ExecutionMode.TEST:
            raise TradingForbiddenError("TEST mode requires SimulatedBroker")
        if broker.mode == ExecutionMode.REAL_SHADOW and risk.mode != ExecutionMode.REAL_SHADOW:
            raise TradingForbiddenError("ShadowBroker requires ExecutionMode.REAL_SHADOW")
        if risk.mode == ExecutionMode.REAL_SHADOW and broker.mode != ExecutionMode.REAL_SHADOW:
            raise TradingForbiddenError("REAL_SHADOW requires ShadowBroker")
        if broker.mode == ExecutionMode.REAL_CANARY_SINGLE_SHOT and risk.mode != ExecutionMode.REAL_CANARY_SINGLE_SHOT:
            raise TradingForbiddenError("Canary RealBroker requires REAL_CANARY_SINGLE_SHOT risk mode")
        if risk.mode == ExecutionMode.REAL_CANARY_SINGLE_SHOT and broker.mode != ExecutionMode.REAL_CANARY_SINGLE_SHOT:
            raise TradingForbiddenError("REAL_CANARY_SINGLE_SHOT requires canary RealBroker")
        if broker.mode == ExecutionMode.REAL_BOUNDED_6H and risk.mode != ExecutionMode.REAL_BOUNDED_6H:
            raise TradingForbiddenError("Bounded-6h RealBroker requires REAL_BOUNDED_6H risk mode")
        if risk.mode == ExecutionMode.REAL_BOUNDED_6H and broker.mode != ExecutionMode.REAL_BOUNDED_6H:
            raise TradingForbiddenError("REAL_BOUNDED_6H requires bounded-6h RealBroker")
        self.broker = broker
        self.risk = risk
        self.intent_journal = intent_journal
        self.order_journal = order_journal
        self.kill_switch = kill_switch or risk.kill_switch
        self.strategy_version = strategy_version
        self.safety = safety or ExecutionSafetyState(
            status=ExecutionSafetyStatus.NORMAL,
            reason_code="OK",
            detail="OMS initialized",
            real_entries_blocked=True,
        )

    @property
    def mode(self) -> ExecutionMode:
        return self.broker.mode

    def build_durable_intent(
        self,
        *,
        symbol: str,
        side: str,
        requested_allocation_pct: float,
        requested_quantity: float,
        signal_ts: str,
        candle_ts: str,
        selected_exit: str,
        order_kind: OrderKind | str = OrderKind.ENTRY,
        strategy_state: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
        attempt: int = 1,
        intent_nonce: str = "",
    ) -> DurableTradeIntent:
        kind = OrderKind(order_kind) if not isinstance(order_kind, OrderKind) else order_kind
        coid = build_client_order_id(
            strategy_version=self.strategy_version,
            symbol=symbol,
            side=side,
            signal_ts=str(signal_ts),
            candle_ts=str(candle_ts),
            selected_exit=str(selected_exit),
            kind=kind,
            attempt=attempt,
            intent_nonce=intent_nonce,
        )
        intent_id = intent_fingerprint(
            strategy_version=self.strategy_version,
            symbol=symbol,
            side=side,
            signal_ts=str(signal_ts),
            candle_ts=str(candle_ts),
            selected_exit=str(selected_exit),
            kind=kind,
            intent_nonce=intent_nonce,
        )
        return DurableTradeIntent(
            intent_id=intent_id,
            client_order_id=coid,
            strategy_version=self.strategy_version,
            symbol=symbol,
            side=side,
            requested_allocation_pct=float(requested_allocation_pct),
            requested_quantity=float(requested_quantity),
            signal_ts=str(signal_ts),
            candle_ts=str(candle_ts),
            selected_exit=str(selected_exit),
            execution_mode=self.mode.value,
            order_kind=kind.value,
            current_strategy_state=dict(strategy_state or {}),
            meta=dict(meta or {}),
        )

    def persist_intent(self, intent: DurableTradeIntent) -> DurableTradeIntent:
        if self.intent_journal is None:
            raise IntentPersistenceError("IntentJournal not configured — cannot submit")
        return self.intent_journal.append(intent)

    def _mark_submit_pending(self, intent: DurableTradeIntent) -> None:
        if self.order_journal is None:
            return
        self.order_journal.append(
            OrderRecord(
                order_local_id=f"ord_{intent.intent_id}_pending",
                intent_id=intent.intent_id,
                client_order_id=intent.client_order_id,
                symbol=intent.symbol,
                side=intent.side,
                order_kind=intent.order_kind,
                status=OrderLifecycleStatus.SUBMIT_PENDING.value,
                requested_quantity=intent.requested_quantity,
                remaining_quantity=intent.requested_quantity,
                reconciliation_state=ReconciliationState.AWAITING_ACK.value,
                meta={"allocation_pct": intent.requested_allocation_pct},
            )
        )

    def _apply_ack_and_fills(self, intent: DurableTradeIntent, report: FillReport) -> OrderRecord | None:
        if self.order_journal is None:
            return None
        raw = report.raw or {}
        acknowledged = bool(raw.get("acknowledged", report.ok))
        exe = float(raw.get("executed_quantity", 0.0) or 0.0)
        req = float(raw.get("requested_quantity", intent.requested_quantity) or intent.requested_quantity)
        rem = float(raw.get("remaining_quantity", max(0.0, req - exe)))
        exch_status = str(raw.get("status", "")).upper()
        fully = bool(raw.get("fully_filled", False)) or exch_status == "FILLED" or (
            exe > 0 and abs(exe - req) <= 1e-12
        )

        if not report.ok or exch_status == "REJECTED":
            status = OrderLifecycleStatus.REJECTED.value
        elif fully and exe > 0:
            status = OrderLifecycleStatus.FILLED.value
        elif exe > 0:
            status = OrderLifecycleStatus.PARTIALLY_FILLED.value
        elif acknowledged:
            status = OrderLifecycleStatus.SUBMITTED.value
        else:
            status = OrderLifecycleStatus.SUBMIT_PENDING.value

        # Never jump to FILLED without executed quantity
        if status == OrderLifecycleStatus.FILLED.value and exe <= 0:
            status = OrderLifecycleStatus.SUBMITTED.value if acknowledged else OrderLifecycleStatus.SUBMIT_PENDING.value

        rec = OrderRecord(
            order_local_id=f"ord_{intent.intent_id}_update",
            intent_id=intent.intent_id,
            client_order_id=intent.client_order_id,
            symbol=intent.symbol,
            side=intent.side,
            order_kind=intent.order_kind,
            status=status,
            requested_quantity=req,
            executed_quantity=exe,
            remaining_quantity=rem,
            average_fill_price=report.entry_mid,
            fees=float(raw["fees"]) if raw.get("fees") is not None else None,
            exchange_order_id=raw.get("exchange_order_id"),
            reconciliation_state=(
                ReconciliationState.ACKNOWLEDGED.value
                if acknowledged
                else ReconciliationState.AWAITING_ACK.value
            ),
            error=None if report.ok else report.rejection_reason,
            meta={
                "allocation_pct": intent.requested_allocation_pct,
                "fills": raw.get("fills") or [],
                "protective_quantity": exe,
            },
        )
        return self.order_journal.append(rec)

    def submit_with_intent(
        self,
        intent: DurableTradeIntent,
        request: OrderRequest,
        *,
        ctx: RiskContext | None = None,
        allow_resubmit: bool = False,
    ) -> FillReport:
        """Persist → risk → kill switch → broker.

        REAL always refused.
        PAPER may call PaperBroker after persistence.
        TEST uses SimulatedBroker with timeout/reconcile semantics.
        """
        # Idempotency: unresolved prior order for same client_order_id blocks resubmit
        if (
            not allow_resubmit
            and self.order_journal is not None
            and self.order_journal.has_unresolved(intent.client_order_id)
        ):
            raise TradingForbiddenError(
                f"OMS refused duplicate submit — unresolved order for {intent.client_order_id}; "
                "reconcile required"
            )

        # 1) Persist FIRST
        try:
            persisted = self.persist_intent(intent)
        except IntentPersistenceError:
            raise

        # 2) Risk / kill switch
        decision = self.risk.validate_intent(persisted, ctx)
        if not decision.approved:
            if self.intent_journal is not None:
                self.intent_journal.update_status(
                    persisted.intent_id,
                    IntentStatus.RISK_REJECTED.value,
                    reason=decision.reason,
                    details=decision.details,
                )
            raise TradingForbiddenError(f"OMS risk rejected intent: {decision.reason}")

        if not self.kill_switch.can_submit_new_order():
            if self.intent_journal is not None:
                self.intent_journal.update_status(
                    persisted.intent_id, IntentStatus.SUBMISSION_BLOCKED.value
                )
            raise TradingForbiddenError("OMS blocked by execution kill switch")

        # 3) REAL / REAL_SHADOW fail-closed (no paper / sim fallback)
        if self.mode == ExecutionMode.REAL:
            if self.intent_journal is not None:
                self.intent_journal.update_status(
                    persisted.intent_id, IntentStatus.SUBMISSION_BLOCKED.value, reason="REAL_DISABLED"
                )
            raise TradingForbiddenError(
                "OMS refused REAL submit_with_intent — fail-closed; no PAPER/TEST fallback"
            )
        if self.mode == ExecutionMode.REAL_SHADOW:
            if self.intent_journal is not None:
                self.intent_journal.update_status(
                    persisted.intent_id,
                    IntentStatus.SUBMISSION_BLOCKED.value,
                    reason="REAL_SHADOW_READ_ONLY",
                )
            raise TradingForbiddenError(
                "OMS refused REAL_SHADOW submit_with_intent — observation only; no write path"
            )

        # Ensure request carries durable ids
        if request.client_order_id != persisted.client_order_id:
            raise TradingForbiddenError("OrderRequest.client_order_id must match durable intent")

        # 4) Record SUBMIT_PENDING before broker call (SUBMISSION_ATTEMPTED)
        self._mark_submit_pending(persisted)
        if self.intent_journal is not None:
            self.intent_journal.update_status(
                persisted.intent_id, IntentStatus.SUBMIT_ATTEMPTED.value
            )

        if self.mode == ExecutionMode.TEST:
            return self._submit_test(persisted, request)

        # PAPER path (existing economics)
        report = self.broker.submit_order(request)
        if self.order_journal is not None:
            status = (
                OrderLifecycleStatus.FILLED.value
                if report.ok
                else OrderLifecycleStatus.REJECTED.value
            )
            self.order_journal.append(
                OrderRecord(
                    order_local_id=f"ord_{persisted.intent_id}_ack",
                    intent_id=persisted.intent_id,
                    client_order_id=persisted.client_order_id,
                    symbol=persisted.symbol,
                    side=persisted.side,
                    order_kind=persisted.order_kind,
                    status=status,
                    requested_quantity=persisted.requested_quantity,
                    executed_quantity=persisted.requested_quantity if report.ok else 0.0,
                    remaining_quantity=0.0 if report.ok else persisted.requested_quantity,
                    average_fill_price=report.entry_mid,
                    reconciliation_state=ReconciliationState.PAPER_LOCAL.value,
                    error=None if report.ok else report.rejection_reason,
                    meta={"allocation_pct": persisted.requested_allocation_pct},
                )
            )
        return report

    def _submit_test(self, intent: DurableTradeIntent, request: OrderRequest) -> FillReport:
        try:
            report = self.broker.submit_order(request)
        except SimTimeoutError as e:
            # Critical: do NOT resubmit. Mark reconcile required.
            status = OrderLifecycleStatus.RECONCILE_UNKNOWN.value
            recon = ReconciliationState.RECONCILE_REQUIRED.value
            detail = (
                "TIMEOUT_AFTER_ACCEPTANCE"
                if e.accepted
                else "TIMEOUT_BEFORE_ACCEPTANCE_OUTCOME_UNKNOWN"
            )
            if self.order_journal is not None:
                self.order_journal.append(
                    OrderRecord(
                        order_local_id=f"ord_{intent.intent_id}_timeout",
                        intent_id=intent.intent_id,
                        client_order_id=intent.client_order_id,
                        symbol=intent.symbol,
                        side=intent.side,
                        order_kind=intent.order_kind,
                        status=status,
                        requested_quantity=intent.requested_quantity,
                        remaining_quantity=intent.requested_quantity,
                        exchange_order_id=e.exchange_order_id,
                        reconciliation_state=recon,
                        error=str(e),
                        meta={
                            "allocation_pct": intent.requested_allocation_pct,
                            "timeout_accepted": e.accepted,
                            "uncertainty": detail,
                        },
                    )
                )
            if self.intent_journal is not None:
                self.intent_journal.update_status(
                    intent.intent_id, IntentStatus.FAILED.value, reason=detail
                )
            return FillReport(
                ok=False,
                rejection_reason=detail,
                broker_name=getattr(self.broker, "name", "SimulatedBroker"),
                raw={
                    "timeout": True,
                    "accepted": e.accepted,
                    "exchange_order_id": e.exchange_order_id,
                    "client_order_id": intent.client_order_id,
                    "reconcile_required": True,
                },
            )
        except SimRejectedError as e:
            if self.order_journal is not None:
                self.order_journal.append(
                    OrderRecord(
                        order_local_id=f"ord_{intent.intent_id}_rej",
                        intent_id=intent.intent_id,
                        client_order_id=intent.client_order_id,
                        symbol=intent.symbol,
                        side=intent.side,
                        order_kind=intent.order_kind,
                        status=OrderLifecycleStatus.REJECTED.value,
                        requested_quantity=intent.requested_quantity,
                        remaining_quantity=intent.requested_quantity,
                        reconciliation_state=ReconciliationState.ACKNOWLEDGED.value,
                        error=str(e),
                        meta={"allocation_pct": intent.requested_allocation_pct},
                    )
                )
            if self.intent_journal is not None:
                self.intent_journal.update_status(
                    intent.intent_id, IntentStatus.FAILED.value, reason="REJECTED"
                )
            return FillReport(
                ok=False,
                rejection_reason="SIM_REJECTED",
                broker_name=getattr(self.broker, "name", "SimulatedBroker"),
                raw={"rejected": True},
            )
        except SimNetworkError as e:
            if self.order_journal is not None:
                self.order_journal.append(
                    OrderRecord(
                        order_local_id=f"ord_{intent.intent_id}_net",
                        intent_id=intent.intent_id,
                        client_order_id=intent.client_order_id,
                        symbol=intent.symbol,
                        side=intent.side,
                        order_kind=intent.order_kind,
                        status=OrderLifecycleStatus.RECONCILE_UNKNOWN.value,
                        requested_quantity=intent.requested_quantity,
                        remaining_quantity=intent.requested_quantity,
                        reconciliation_state=ReconciliationState.RECONCILE_REQUIRED.value,
                        error=str(e),
                        meta={
                            "allocation_pct": intent.requested_allocation_pct,
                            "uncertainty": "NETWORK_FAILURE_OUTCOME_UNKNOWN",
                        },
                    )
                )
            return FillReport(
                ok=False,
                rejection_reason="NETWORK_FAILURE",
                broker_name=getattr(self.broker, "name", "SimulatedBroker"),
                raw={"network_failure": True, "reconcile_required": True},
            )

        rec = self._apply_ack_and_fills(intent, report)
        if self.intent_journal is not None and rec is not None:
            if rec.status == OrderLifecycleStatus.FILLED.value:
                self.intent_journal.update_status(intent.intent_id, IntentStatus.FILLED.value)
            elif rec.status == OrderLifecycleStatus.PARTIALLY_FILLED.value:
                self.intent_journal.update_status(
                    intent.intent_id, IntentStatus.PARTIALLY_FILLED.value
                )
            elif rec.status == OrderLifecycleStatus.SUBMITTED.value:
                self.intent_journal.update_status(
                    intent.intent_id, IntentStatus.ACKNOWLEDGED.value
                )
        return report

    def reconcile_order_by_client_id(self, client_order_id: str) -> OrderRecord:
        """Query exchange by client_order_id and update local state.

        After timeout-after-acceptance: if order exists → continue that order (no new submit).
        If absent → only then may a future resubmit be considered (caller decides).
        """
        if not self.kill_switch.can_reconcile():
            raise TradingForbiddenError("OMS reconciliation blocked by kill switch")
        if self.order_journal is None:
            raise TradingForbiddenError("OrderJournal required for reconcile")

        local = self.order_journal.latest_by_client_order_id(client_order_id)
        if local is None:
            raise TradingForbiddenError(f"no local order for {client_order_id}")

        getter = getattr(self.broker, "get_order_by_client_order_id", None)
        if getter is None:
            raise TradingForbiddenError("broker cannot query by client_order_id")

        remote = getter(client_order_id)
        if remote is None:
            # Exchange has no order — unresolved local must stay RECONCILE_UNKNOWN until policy allows retry
            return self.order_journal.append(
                OrderRecord(
                    order_local_id=f"{local.order_local_id}:recon_miss",
                    intent_id=local.intent_id,
                    client_order_id=client_order_id,
                    symbol=local.symbol,
                    side=local.side,
                    order_kind=local.order_kind,
                    status=OrderLifecycleStatus.RECONCILE_UNKNOWN.value,
                    requested_quantity=local.requested_quantity,
                    executed_quantity=local.executed_quantity,
                    remaining_quantity=local.remaining_quantity,
                    reconciliation_state=ReconciliationState.MISMATCH.value,
                    error="LOCAL_ORDER_MISSING_ON_EXCHANGE",
                    meta={**local.meta, "exchange_found": False, "resubmit_may_be_safe": True},
                )
            )

        exe = float(remote.get("executed_quantity") or 0.0)
        req = float(remote.get("requested_quantity") or local.requested_quantity)
        rem = float(remote.get("remaining_quantity") or max(0.0, req - exe))
        st = str(remote.get("status", "")).upper()
        if st == "FILLED" or (exe > 0 and abs(exe - req) <= 1e-12):
            new_status = OrderLifecycleStatus.FILLED.value
        elif exe > 0:
            new_status = OrderLifecycleStatus.PARTIALLY_FILLED.value
        elif st == "REJECTED":
            new_status = OrderLifecycleStatus.REJECTED.value
        elif st == "CANCELLED":
            new_status = OrderLifecycleStatus.CANCELLED.value
        else:
            new_status = OrderLifecycleStatus.SUBMITTED.value

        return self.order_journal.append(
            OrderRecord(
                order_local_id=f"{local.order_local_id}:recon",
                intent_id=local.intent_id,
                client_order_id=client_order_id,
                symbol=local.symbol,
                side=local.side,
                order_kind=local.order_kind,
                status=new_status,
                requested_quantity=req,
                executed_quantity=exe,
                remaining_quantity=rem,
                average_fill_price=remote.get("average_price"),
                fees=remote.get("fees"),
                exchange_order_id=remote.get("exchange_order_id") or remote.get("order_id"),
                reconciliation_state=ReconciliationState.MATCHED.value,
                meta={
                    **local.meta,
                    "exchange_found": True,
                    "fills": remote.get("fills") or [],
                    "protective_quantity": exe,
                    "resubmit_may_be_safe": False,
                },
            )
        )

    def apply_reconciliation_report(self, *, matched: bool, detail: str = "") -> ExecutionSafetyState:
        if matched:
            self.safety.mark_normal(detail=detail or "reconcile matched")
            self.safety.real_entries_blocked = True
        else:
            self.safety.mark_halted(RECONCILIATION_MISMATCH, detail or "mismatch")
        return self.safety

    def resume_submit_persisted_intent(
        self,
        intent: DurableTradeIntent,
        request: OrderRequest,
        *,
        ctx: RiskContext | None = None,
    ) -> FillReport:
        """Restart recovery: intent already persisted; do not create a second intent.

        Still blocked if an unresolved local order exists for the client_order_id.
        """
        if self.intent_journal is None:
            raise IntentPersistenceError("IntentJournal required")
        existing = self.intent_journal.find_by_client_order_id(intent.client_order_id)
        if existing is None:
            raise IntentPersistenceError("cannot resume — intent not found in journal")
        if (
            self.order_journal is not None
            and self.order_journal.has_unresolved(intent.client_order_id)
        ):
            raise TradingForbiddenError(
                "resume blocked — unresolved order; call reconcile_order_by_client_id first"
            )

        decision = self.risk.validate_intent(existing, ctx)
        if not decision.approved:
            raise TradingForbiddenError(f"OMS risk rejected intent: {decision.reason}")
        if not self.kill_switch.can_submit_new_order():
            raise TradingForbiddenError("OMS blocked by execution kill switch")
        if self.mode == ExecutionMode.REAL:
            raise TradingForbiddenError("OMS refused REAL resume_submit — fail-closed")
        if self.mode == ExecutionMode.REAL_SHADOW:
            raise TradingForbiddenError("OMS refused REAL_SHADOW resume_submit — observation only")

        self._mark_submit_pending(existing)
        self.intent_journal.update_status(existing.intent_id, IntentStatus.SUBMIT_ATTEMPTED.value)
        if self.mode == ExecutionMode.TEST:
            return self._submit_test(existing, request)
        report = self.broker.submit_order(request)
        if self.order_journal is not None:
            self.order_journal.append(
                OrderRecord(
                    order_local_id=f"ord_{existing.intent_id}_ack",
                    intent_id=existing.intent_id,
                    client_order_id=existing.client_order_id,
                    symbol=existing.symbol,
                    side=existing.side,
                    order_kind=existing.order_kind,
                    status=(
                        OrderLifecycleStatus.FILLED.value
                        if report.ok
                        else OrderLifecycleStatus.REJECTED.value
                    ),
                    requested_quantity=existing.requested_quantity,
                    executed_quantity=existing.requested_quantity if report.ok else 0.0,
                    remaining_quantity=0.0 if report.ok else existing.requested_quantity,
                    average_fill_price=report.entry_mid,
                    reconciliation_state=ReconciliationState.PAPER_LOCAL.value,
                    meta={"allocation_pct": existing.requested_allocation_pct},
                )
            )
        return report

    def open_entry_legs(
        self,
        *,
        intent: TradeIntent | DurableTradeIntent | None = None,
        alt_btc_entry_mid: float,
        btc_usdt: float,
        notional_usd: float,
        cf_specs: list[StrategySpec],
        selected_spec: StrategySpec,
        entry_ts: Any,
        risk_ctx: RiskContext | None = None,
        persist: bool = True,
    ) -> tuple[list[Any], list[Any]]:
        """Open CF + selected E legs via the broker (paper: identical fill economics)."""
        if self.mode == ExecutionMode.REAL:
            raise TradingForbiddenError(
                "OMS refused REAL open_entry_legs — fail-closed; no PAPER fallback"
            )
        if self.mode == ExecutionMode.REAL_SHADOW:
            raise TradingForbiddenError(
                "OMS refused REAL_SHADOW open_entry_legs — observation only; no write path"
            )
        if self.mode == ExecutionMode.TEST:
            raise TradingForbiddenError(
                "OMS open_entry_legs is paper-path only; use submit_with_intent in TEST mode"
            )

        durable: DurableTradeIntent | None = None
        if persist and self.intent_journal is not None:
            if isinstance(intent, DurableTradeIntent):
                durable = intent
            elif isinstance(intent, TradeIntent):
                durable = self.build_durable_intent(
                    symbol=intent.symbol,
                    side=intent.side,
                    requested_allocation_pct=intent.requested_allocation_pct,
                    requested_quantity=intent.position_btc,
                    signal_ts=str(intent.decision_ts),
                    candle_ts=str(entry_ts),
                    selected_exit=str(
                        intent.selected_arm_label or intent.selected_strategy_key or selected_spec.key
                    ),
                    strategy_state={"S": intent.s_value, "regime": intent.regime},
                    meta={"opportunity_id": intent.opportunity_id},
                )

        if durable is not None:
            try:
                durable = self.persist_intent(durable)
            except IntentPersistenceError:
                raise

            decision = self.risk.validate_intent(durable, risk_ctx)
            if not decision.approved:
                self.intent_journal.update_status(
                    durable.intent_id, IntentStatus.RISK_REJECTED.value, reason=decision.reason
                )
                raise TradingForbiddenError(f"OMS risk rejected intent: {decision.reason}")
        elif intent is not None:
            decision = self.risk.validate_intent(intent, risk_ctx)
            if not decision.approved:
                raise TradingForbiddenError(f"OMS risk rejected intent: {decision.reason}")

        if not self.kill_switch.can_submit_new_order():
            if durable is not None and self.intent_journal is not None:
                self.intent_journal.update_status(
                    durable.intent_id, IntentStatus.SUBMISSION_BLOCKED.value
                )
            raise TradingForbiddenError("OMS blocked by execution kill switch")

        cf_legs = self.broker.open_long_legs(
            alt_btc_entry_mid=alt_btc_entry_mid,
            btc_usdt=btc_usdt,
            notional_usd=notional_usd,
            specs=list(cf_specs),
            entry_ts=entry_ts,
        )
        sel_legs = self.broker.open_long_legs(
            alt_btc_entry_mid=alt_btc_entry_mid,
            btc_usdt=btc_usdt,
            notional_usd=notional_usd,
            specs=[selected_spec],
            entry_ts=entry_ts,
        )
        if durable is not None and self.intent_journal is not None:
            self.intent_journal.update_status(durable.intent_id, IntentStatus.FILLED.value, paper=True)
        return cf_legs, sel_legs

    def submit_order(self, request: OrderRequest) -> FillReport:
        if self.mode == ExecutionMode.REAL:
            raise TradingForbiddenError(
                "OMS refused REAL submit_order — fail-closed; no PAPER fallback"
            )
        if self.mode == ExecutionMode.REAL_SHADOW:
            raise TradingForbiddenError(
                "OMS refused REAL_SHADOW submit_order — observation only; no write path"
            )
        if not self.kill_switch.can_submit_new_order():
            raise TradingForbiddenError("OMS blocked by execution kill switch")
        return self.broker.submit_order(request)

    def can_reconcile(self) -> bool:
        return self.kill_switch.can_reconcile()

    def can_exit(self) -> bool:
        return self.kill_switch.can_exit()

    def can_monitor(self) -> bool:
        """Existing position monitoring is always allowed (not gated by entry kill)."""
        return True
