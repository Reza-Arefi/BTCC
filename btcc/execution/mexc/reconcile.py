"""READ-ONLY reconciliation: local execution state vs MEXC Spot truth.

Mismatch → EXECUTION_RECONCILIATION_FAILURE and execution HALT.
Never auto-trades to correct mismatches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from btcc.execution.mexc.client import MexcReadOnlyClient
from btcc.execution.mexc.errors import MexcReadError
from btcc.execution.mexc.models import AssetBalance, ExchangeFill, ExchangeOrder
from btcc.execution.order_state import OrderLifecycleStatus, OrderRecord
from btcc.execution.safety_state import RECONCILIATION_MISMATCH, ExecutionSafetyState

TERMINAL_LOCAL = {
    OrderLifecycleStatus.FILLED.value,
    OrderLifecycleStatus.CANCELLED.value,
    OrderLifecycleStatus.REJECTED.value,
    OrderLifecycleStatus.ERROR.value,
}


@dataclass(frozen=True)
class LocalBalanceExpectation:
    asset: str
    free: Decimal | None = None
    locked: Decimal | None = None
    total: Decimal | None = None
    # If True, any non-dust balance for this asset on exchange is unexpected when local expects ~0
    expect_zero: bool = False


@dataclass
class LocalExecutionView:
    """Local OMS/order journal view used for reconcile (read-only inputs)."""

    open_orders: list[OrderRecord] = field(default_factory=list)
    balance_expectations: list[LocalBalanceExpectation] = field(default_factory=list)
    # Assets that should not appear with inventory on the exchange (beyond dust)
    forbidden_inventory_assets: list[str] = field(default_factory=list)
    known_fill_ids: set[str] = field(default_factory=set)
    max_account_age_ms: int | None = None  # unused until exchange provides updateTime consistently
    dust: Decimal = Decimal("0")


@dataclass(frozen=True)
class ReconcileMismatch:
    code: str
    detail: str
    local: Any = None
    exchange: Any = None


@dataclass
class ReconcileReport:
    matched: bool
    mismatches: list[ReconcileMismatch] = field(default_factory=list)
    exchange_balances: list[AssetBalance] = field(default_factory=list)
    exchange_open_orders: list[ExchangeOrder] = field(default_factory=list)
    reason_code: str = "OK"

    @property
    def failure_code(self) -> str:
        return RECONCILIATION_MISMATCH if not self.matched else "OK"


def _qty_close(a: Decimal | float | None, b: Decimal | float | None, *, tol: Decimal = Decimal("1e-12")) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(Decimal(str(a)) - Decimal(str(b))) <= tol


def reconcile_local_vs_exchange(
    local: LocalExecutionView,
    *,
    exchange_balances: list[AssetBalance],
    exchange_open_orders: list[ExchangeOrder],
    exchange_fills: list[ExchangeFill] | None = None,
) -> ReconcileReport:
    """Pure comparison — no HTTP, no side effects."""
    mismatches: list[ReconcileMismatch] = []

    bal_by_asset = {b.asset.upper(): b for b in exchange_balances}

    # Unexpected / forbidden inventory
    for asset in local.forbidden_inventory_assets:
        b = bal_by_asset.get(asset.upper())
        if b is not None and b.total > local.dust:
            mismatches.append(
                ReconcileMismatch(
                    code="UNEXPECTED_EXCHANGE_POSITION",
                    detail=f"unexpected inventory for {asset}: total={b.total}",
                    exchange=b,
                )
            )

    for exp in local.balance_expectations:
        b = bal_by_asset.get(exp.asset.upper())
        if exp.expect_zero:
            if b is not None and b.total > local.dust:
                mismatches.append(
                    ReconcileMismatch(
                        code="UNEXPECTED_REAL_BALANCE",
                        detail=f"expected zero {exp.asset}, got total={b.total}",
                        local=exp,
                        exchange=b,
                    )
                )
            continue
        if b is None:
            mismatches.append(
                ReconcileMismatch(
                    code="UNEXPECTED_REAL_BALANCE",
                    detail=f"missing exchange balance for {exp.asset}",
                    local=exp,
                )
            )
            continue
        if exp.free is not None and not _qty_close(exp.free, b.free):
            mismatches.append(
                ReconcileMismatch(
                    code="QUANTITY_MISMATCH",
                    detail=f"{exp.asset} free local={exp.free} exchange={b.free}",
                    local=exp,
                    exchange=b,
                )
            )
        if exp.locked is not None and not _qty_close(exp.locked, b.locked):
            mismatches.append(
                ReconcileMismatch(
                    code="QUANTITY_MISMATCH",
                    detail=f"{exp.asset} locked local={exp.locked} exchange={b.locked}",
                    local=exp,
                    exchange=b,
                )
            )
        if exp.total is not None and not _qty_close(exp.total, b.total):
            mismatches.append(
                ReconcileMismatch(
                    code="QUANTITY_MISMATCH",
                    detail=f"{exp.asset} total local={exp.total} exchange={b.total}",
                    local=exp,
                    exchange=b,
                )
            )

    local_open = [o for o in local.open_orders if o.status not in TERMINAL_LOCAL]
    local_by_exch: dict[str, OrderRecord] = {}
    local_by_client: dict[str, OrderRecord] = {}
    for o in local_open:
        if o.exchange_order_id:
            local_by_exch[str(o.exchange_order_id)] = o
        local_by_client[o.client_order_id] = o

    exch_by_id = {o.order_id: o for o in exchange_open_orders}
    exch_by_client = {o.client_order_id: o for o in exchange_open_orders if o.client_order_id}

    # Unexpected open orders on exchange
    for eo in exchange_open_orders:
        if eo.order_id in local_by_exch:
            continue
        if eo.client_order_id and eo.client_order_id in local_by_client:
            continue
        mismatches.append(
            ReconcileMismatch(
                code="UNEXPECTED_OPEN_ORDER",
                detail=f"exchange open order {eo.order_id} not in local state",
                exchange=eo,
            )
        )

    # Local orders missing / unknown / qty / status
    for lo in local_open:
        eo = None
        if lo.exchange_order_id and lo.exchange_order_id in exch_by_id:
            eo = exch_by_id[lo.exchange_order_id]
        elif lo.client_order_id in exch_by_client:
            eo = exch_by_client[lo.client_order_id]

        if eo is None:
            # SUBMIT_PENDING with no exchange id may be "unknown" rather than missing
            if lo.status == OrderLifecycleStatus.SUBMIT_PENDING.value and not lo.exchange_order_id:
                mismatches.append(
                    ReconcileMismatch(
                        code="UNKNOWN_LOCAL_ORDER",
                        detail=f"local SUBMIT_PENDING {lo.client_order_id} has no exchange ack",
                        local=lo,
                    )
                )
            else:
                mismatches.append(
                    ReconcileMismatch(
                        code="LOCAL_ORDER_MISSING_ON_EXCHANGE",
                        detail=f"local open order {lo.client_order_id} not on exchange",
                        local=lo,
                    )
                )
            continue

        if not _qty_close(Decimal(str(lo.requested_quantity)), eo.original_quantity) and lo.requested_quantity > 0:
            # Compare remaining/executed when available
            if not _qty_close(Decimal(str(lo.executed_quantity)), eo.executed_quantity):
                mismatches.append(
                    ReconcileMismatch(
                        code="QUANTITY_MISMATCH",
                        detail=(
                            f"order {lo.client_order_id} executed "
                            f"local={lo.executed_quantity} exchange={eo.executed_quantity}"
                        ),
                        local=lo,
                        exchange=eo,
                    )
                )

        # Status coarse map
        local_status = lo.status
        exch_status = eo.status.upper()
        if local_status == OrderLifecycleStatus.PARTIALLY_FILLED.value and exch_status not in (
            "PARTIALLY_FILLED",
            "NEW",
            "LIVE",
        ):
            mismatches.append(
                ReconcileMismatch(
                    code="STATUS_MISMATCH",
                    detail=f"order {lo.client_order_id} local={local_status} exchange={exch_status}",
                    local=lo,
                    exchange=eo,
                )
            )

    if exchange_fills is not None and local.known_fill_ids:
        for f in exchange_fills:
            if f.trade_id not in local.known_fill_ids and f.order_id in local_by_exch:
                mismatches.append(
                    ReconcileMismatch(
                        code="FILL_MISMATCH",
                        detail=f"exchange fill {f.trade_id} for known order not in local fills",
                        exchange=f,
                    )
                )

    matched = len(mismatches) == 0
    return ReconcileReport(
        matched=matched,
        mismatches=mismatches,
        exchange_balances=list(exchange_balances),
        exchange_open_orders=list(exchange_open_orders),
        reason_code="OK" if matched else RECONCILIATION_MISMATCH,
    )


def apply_reconcile_to_safety(report: ReconcileReport, safety: ExecutionSafetyState) -> ExecutionSafetyState:
    if report.matched:
        safety.mark_normal(detail="startup reconcile matched")
        # Stage 3: still no trading path — keep real entries conceptually blocked at broker layer
        safety.real_entries_blocked = True
        safety.meta["stage3_note"] = "match_ok_but_trading_not_implemented"
    else:
        detail = "; ".join(f"{m.code}:{m.detail}" for m in report.mismatches[:8])
        safety.mark_halted(RECONCILIATION_MISMATCH, detail)
    return safety


def startup_recovery_reconcile(
    client: MexcReadOnlyClient,
    local: LocalExecutionView,
    safety: ExecutionSafetyState,
    *,
    open_order_symbol: str | None = None,
) -> ReconcileReport:
    """Foundation for startup recovery:

    load local → query MEXC → reconcile → NORMAL or HALT.

    Does not enable order submission.
    """
    try:
        balances = client.get_balances()
        opens = client.get_open_orders(symbol=open_order_symbol)
    except MexcReadError as e:
        safety.apply_read_error(e)
        return ReconcileReport(
            matched=False,
            mismatches=[ReconcileMismatch(code=type(e).__name__, detail=str(e))],
            reason_code=safety.reason_code,
        )

    report = reconcile_local_vs_exchange(
        local,
        exchange_balances=balances,
        exchange_open_orders=opens,
    )
    apply_reconcile_to_safety(report, safety)
    return report
