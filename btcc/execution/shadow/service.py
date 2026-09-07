"""REAL_SHADOW reconciliation cycle — observe MEXC, never write, never touch paper."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

from btcc.execution.mexc.errors import (
    AccountDataUnavailable,
    MexcAuthError,
    MexcMalformedResponseError,
    MexcRateLimitError,
    MexcReadError,
    MexcTimeoutError,
)
from btcc.execution.mexc.models import AssetBalance, ExchangeFill, ExchangeOrder
from btcc.execution.mexc.reconcile import (
    LocalExecutionView,
    ReconcileReport,
    apply_reconcile_to_safety,
    reconcile_local_vs_exchange,
)
from btcc.execution.safety_state import (
    ACCOUNT_UNAVAILABLE,
    MALFORMED_EXCHANGE_RESPONSE,
    RATE_LIMIT_FAILURE,
    RECONCILIATION_MISMATCH,
    REAL_TRADING_DISABLED,
    STALE_ACCOUNT,
    SYMBOL_METADATA_UNAVAILABLE,
    TIMEOUT,
    ExecutionSafetyState,
    ExecutionSafetyStatus,
)
from btcc.execution.shadow.broker import ShadowBroker
from btcc.execution.shadow.snapshot import ShadowAccountSnapshot, build_snapshot
from btcc.execution.shadow.symbol_audit import SymbolAuditReport, audit_symbols
from btcc.execution.symbols import SymbolMeta


class ShadowResultClass(str, Enum):
    MATCH = "MATCH"
    DEGRADED = "DEGRADED"
    MISMATCH = "MISMATCH"
    UNAVAILABLE = "UNAVAILABLE"


# Shadow-specific reason codes
SHADOW_UNAVAILABLE = "SHADOW_UNAVAILABLE"
SHADOW_AUTH_FAILURE = "SHADOW_AUTH_FAILURE"
UNEXPECTED_EXCHANGE_STATE = "UNEXPECTED_EXCHANGE_STATE"
REAL_EXECUTION_BLOCKED = "REAL_EXECUTION_BLOCKED"
EMPTY_ACCOUNT_OBSERVED = "EMPTY_ACCOUNT_OBSERVED"


@dataclass
class ShadowCycleResult:
    result_class: ShadowResultClass
    reason_code: str
    detail: str
    snapshot: ShadowAccountSnapshot | None = None
    reconcile: ReconcileReport | None = None
    symbol_audit: SymbolAuditReport | None = None
    checks_performed: list[str] = field(default_factory=list)
    empty_account: bool = False
    successfully_reconciled: bool = False
    unexpected_exchange_state: bool = False
    real_execution_blocked: bool = True
    safety_status: str = ExecutionSafetyStatus.HALTED.value
    local_ts: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.local_ts:
            self.local_ts = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "result_class": self.result_class.value,
            "reason_code": self.reason_code,
            "detail": self.detail,
            "checks_performed": list(self.checks_performed),
            "empty_account": self.empty_account,
            "successfully_reconciled": self.successfully_reconciled,
            "unexpected_exchange_state": self.unexpected_exchange_state,
            "real_execution_blocked": self.real_execution_blocked,
            "safety_status": self.safety_status,
            "local_ts": self.local_ts,
            "meta": dict(self.meta),
            "snapshot": self.snapshot.to_dict() if self.snapshot else None,
            "reconcile_matched": None if self.reconcile is None else self.reconcile.matched,
            "reconcile_mismatches": (
                None
                if self.reconcile is None
                else [{"code": m.code, "detail": m.detail} for m in self.reconcile.mismatches]
            ),
            "symbol_audit": self.symbol_audit.to_dict() if self.symbol_audit else None,
        }
        return d

    def persist(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n"
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, path)
        return path


class ShadowReconciliationService:
    """Pull MEXC truth → snapshot → reconcile → update safety. Never writes orders."""

    def __init__(
        self,
        broker: ShadowBroker,
        *,
        safety: ExecutionSafetyState | None = None,
        local_view: LocalExecutionView | None = None,
        symbols: list[str] | None = None,
        fill_symbols: list[str] | None = None,
        max_skew_ms: int = 5000,
        max_age_ms: int = 30_000,
        dust: Decimal = Decimal("0"),
        state_dir: Path | None = None,
    ) -> None:
        self.broker = broker
        self.safety = safety or broker.safety
        self.local_view = local_view or LocalExecutionView()
        self.symbols = list(symbols or [])
        self.fill_symbols = list(fill_symbols or symbols or [])
        self.max_skew_ms = int(max_skew_ms)
        self.max_age_ms = int(max_age_ms)
        self.dust = dust
        self.state_dir = Path(state_dir) if state_dir else None
        # Always block real entries in Stage 5A
        self.safety.real_entries_blocked = True

    def run_cycle(self) -> ShadowCycleResult:
        checks: list[str] = ["SHADOW_CYCLE_START", "WRITE_PATH_NOT_INVOKED"]
        client = self.broker.client
        if client is None:
            self.safety.mark_halted(SHADOW_UNAVAILABLE, "no MEXC read client configured")
            return self._finish(
                ShadowResultClass.UNAVAILABLE,
                SHADOW_UNAVAILABLE,
                "credentials/client missing",
                checks=checks + ["CLIENT_MISSING"],
            )

        try:
            server_time = client.get_server_time()
            checks.append("SERVER_TIME_OK")
        except MexcAuthError as e:
            self.safety.mark_halted(SHADOW_AUTH_FAILURE, str(e))
            return self._finish(
                ShadowResultClass.UNAVAILABLE,
                SHADOW_AUTH_FAILURE,
                str(e),
                checks=checks + ["AUTH_FAILURE"],
            )
        except MexcTimeoutError as e:
            self.safety.mark_halted(TIMEOUT, str(e))
            return self._finish(
                ShadowResultClass.UNAVAILABLE, TIMEOUT, str(e), checks=checks + ["TIMEOUT"]
            )
        except MexcRateLimitError as e:
            self.safety.mark_halted(RATE_LIMIT_FAILURE, str(e))
            return self._finish(
                ShadowResultClass.UNAVAILABLE,
                RATE_LIMIT_FAILURE,
                str(e),
                checks=checks + ["RATE_LIMIT"],
            )
        except MexcReadError as e:
            code = SHADOW_UNAVAILABLE
            if isinstance(e, MexcMalformedResponseError):
                code = MALFORMED_EXCHANGE_RESPONSE
                self.safety.mark_halted(code, str(e))
            else:
                self.safety.mark_halted(ACCOUNT_UNAVAILABLE, str(e))
            return self._finish(
                ShadowResultClass.UNAVAILABLE, code, str(e), checks=checks + ["READ_FAILURE"]
            )

        try:
            balances = client.get_balances()
            checks.append("BALANCES_OK")
            opens = client.get_open_orders()
            checks.append("OPEN_ORDERS_OK")
        except MexcAuthError as e:
            self.safety.mark_halted(SHADOW_AUTH_FAILURE, str(e))
            return self._finish(
                ShadowResultClass.UNAVAILABLE, SHADOW_AUTH_FAILURE, str(e), checks=checks
            )
        except MexcTimeoutError as e:
            self.safety.mark_halted(TIMEOUT, str(e))
            return self._finish(ShadowResultClass.UNAVAILABLE, TIMEOUT, str(e), checks=checks)
        except MexcRateLimitError as e:
            self.safety.mark_halted(RATE_LIMIT_FAILURE, str(e))
            return self._finish(
                ShadowResultClass.UNAVAILABLE, RATE_LIMIT_FAILURE, str(e), checks=checks
            )
        except (AccountDataUnavailable, MexcMalformedResponseError) as e:
            self.safety.mark_halted(
                MALFORMED_EXCHANGE_RESPONSE
                if isinstance(e, MexcMalformedResponseError)
                else ACCOUNT_UNAVAILABLE,
                str(e),
            )
            return self._finish(
                ShadowResultClass.UNAVAILABLE, type(e).__name__, str(e), checks=checks
            )
        except MexcReadError as e:
            # Never convert to empty account
            self.safety.mark_halted(ACCOUNT_UNAVAILABLE, str(e))
            return self._finish(
                ShadowResultClass.UNAVAILABLE, SHADOW_UNAVAILABLE, str(e), checks=checks
            )

        fills: list[ExchangeFill] = []
        for sym in self.fill_symbols:
            try:
                fills.extend(client.get_recent_fills(sym, limit=50))
                checks.append(f"FILLS_OK:{sym}")
            except MexcReadError as e:
                self.safety.mark_degraded(ACCOUNT_UNAVAILABLE, f"fills unavailable for {sym}: {e}")
                checks.append(f"FILLS_DEGRADED:{sym}")
                # Continue — fills optional for cycle classification, but note degraded

        metas: list[SymbolMeta] = []
        audit: SymbolAuditReport | None = None
        if self.symbols:
            audit = audit_symbols(client, self.symbols)
            checks.append("SYMBOL_AUDIT_DONE")
            for row in audit.rows:
                if not row.blocked and row.quantity_step and row.price_tick:
                    metas.append(
                        SymbolMeta(
                            symbol=row.symbol,
                            status=row.status or "UNKNOWN",
                            base_asset=row.base_asset or "",
                            quote_asset=row.quote_asset or "",
                            quantity_step=float(row.quantity_step),
                            min_quantity=float(row.min_quantity or 0),
                            price_tick=float(row.price_tick),
                            min_notional=float(row.min_notional or 0),
                            max_quantity=row.max_quantity,
                            source=row.source or "mexc_exchangeInfo",
                        )
                    )
            if audit.any_blocked:
                checks.append("SYMBOL_METADATA_BLOCKED")
                self.safety.mark_degraded(
                    SYMBOL_METADATA_UNAVAILABLE,
                    "one or more symbols lack verified metadata → REAL_EXECUTION_BLOCKED",
                )

        snapshot = build_snapshot(
            balances=balances,
            open_orders=opens,
            fills=fills,
            symbols=metas,
            exchange_server_time_ms=server_time,
            max_skew_ms=self.max_skew_ms,
            max_age_ms=self.max_age_ms,
            dust=self.dust,
        )
        checks.extend(snapshot.checks_performed)

        if not snapshot.fresh:
            self.safety.mark_halted(STALE_ACCOUNT, "; ".join(snapshot.notes) or "stale")
            return self._finish(
                ShadowResultClass.DEGRADED if snapshot.successfully_fetched else ShadowResultClass.UNAVAILABLE,
                STALE_ACCOUNT,
                "; ".join(snapshot.notes),
                checks=checks,
                snapshot=snapshot,
                audit=audit,
                empty=snapshot.empty_account,
            )

        report = reconcile_local_vs_exchange(
            self.local_view,
            exchange_balances=balances,
            exchange_open_orders=opens,
            exchange_fills=fills if fills else None,
        )
        checks.append("RECONCILE_COMPARE_DONE")

        unexpected = any(
            m.code
            in {
                "UNEXPECTED_OPEN_ORDER",
                "UNEXPECTED_REAL_BALANCE",
                "UNEXPECTED_EXCHANGE_POSITION",
            }
            for m in report.mismatches
        )

        if not report.matched:
            apply_reconcile_to_safety(report, self.safety)
            # Always keep real entries blocked
            self.safety.real_entries_blocked = True
            code = UNEXPECTED_EXCHANGE_STATE if unexpected else RECONCILIATION_MISMATCH
            if unexpected:
                self.safety.mark_halted(code, self.safety.detail)
            return self._finish(
                ShadowResultClass.MISMATCH,
                code,
                self.safety.detail,
                checks=checks + ["RECONCILE_MISMATCH"],
                snapshot=snapshot,
                reconcile=report,
                audit=audit,
                empty=snapshot.empty_account,
                unexpected=unexpected,
                reconciled=False,
            )

        # MATCH path — empty vs reconciled distinction
        apply_reconcile_to_safety(report, self.safety)
        self.safety.real_entries_blocked = True
        self.safety.meta["stage5a"] = "shadow_match_trading_still_impossible"
        self.safety.meta["protective_stop"] = "UNVERIFIED_BLOCKER"
        detail = "reconcile matched"
        reason = "OK"
        if snapshot.empty_account:
            reason = EMPTY_ACCOUNT_OBSERVED
            detail = (
                "EMPTY ACCOUNT observed and compared — NOT trade-ready; "
                "checks passed for emptiness vs local expectations only"
            )
            checks.append("EMPTY_DISTINCT_FROM_TRADE_READY")
        if audit is not None and audit.any_blocked:
            self.safety.mark_degraded(REAL_EXECUTION_BLOCKED, "symbol metadata blocked")
            return self._finish(
                ShadowResultClass.DEGRADED,
                REAL_EXECUTION_BLOCKED,
                "matched account but symbol metadata incomplete",
                checks=checks,
                snapshot=snapshot,
                reconcile=report,
                audit=audit,
                empty=snapshot.empty_account,
                reconciled=True,
            )

        # Even on perfect match, Stage 5A never enables trading
        self.safety.mark_normal(detail=detail + " (REAL_SHADOW observation only)")
        self.safety.real_entries_blocked = True
        self.safety.reason_code = REAL_TRADING_DISABLED if reason == "OK" else reason
        return self._finish(
            ShadowResultClass.MATCH,
            reason,
            detail,
            checks=checks + ["RECONCILE_MATCH", "TRADING_STILL_IMPOSSIBLE"],
            snapshot=snapshot,
            reconcile=report,
            audit=audit,
            empty=snapshot.empty_account,
            reconciled=True,
        )

    def _finish(
        self,
        result_class: ShadowResultClass,
        reason_code: str,
        detail: str,
        *,
        checks: list[str],
        snapshot: ShadowAccountSnapshot | None = None,
        reconcile: ReconcileReport | None = None,
        audit: SymbolAuditReport | None = None,
        empty: bool = False,
        unexpected: bool = False,
        reconciled: bool = False,
    ) -> ShadowCycleResult:
        result = ShadowCycleResult(
            result_class=result_class,
            reason_code=reason_code,
            detail=detail,
            snapshot=snapshot,
            reconcile=reconcile,
            symbol_audit=audit,
            checks_performed=list(checks),
            empty_account=empty,
            successfully_reconciled=reconciled and result_class == ShadowResultClass.MATCH,
            unexpected_exchange_state=unexpected,
            real_execution_blocked=True,
            safety_status=self.safety.status.value,
            meta={
                "broker": self.broker.name,
                "mode": "REAL_SHADOW",
                "allow_trading": False,
            },
        )
        if self.state_dir is not None:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            result.persist(self.state_dir / "shadow_last_cycle.json")
            if snapshot is not None:
                snapshot.persist(self.state_dir / "shadow_last_snapshot.json")
        return result
