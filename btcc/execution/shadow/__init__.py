"""REAL_SHADOW package — Stage 5A MEXC read-only observation + reconciliation.

Never submits/cancels/places stops. Never mutates paper equity or strategy state.
"""

from btcc.execution.shadow.broker import ShadowBroker
from btcc.execution.shadow.service import (
    EMPTY_ACCOUNT_OBSERVED,
    REAL_EXECUTION_BLOCKED,
    SHADOW_AUTH_FAILURE,
    SHADOW_UNAVAILABLE,
    UNEXPECTED_EXCHANGE_STATE,
    ShadowCycleResult,
    ShadowReconciliationService,
    ShadowResultClass,
)
from btcc.execution.shadow.snapshot import ShadowAccountSnapshot, build_snapshot
from btcc.execution.shadow.symbol_audit import SymbolAuditReport, audit_symbols

__all__ = [
    "ShadowBroker",
    "ShadowReconciliationService",
    "ShadowCycleResult",
    "ShadowResultClass",
    "ShadowAccountSnapshot",
    "build_snapshot",
    "SymbolAuditReport",
    "audit_symbols",
    "SHADOW_UNAVAILABLE",
    "SHADOW_AUTH_FAILURE",
    "UNEXPECTED_EXCHANGE_STATE",
    "REAL_EXECUTION_BLOCKED",
    "EMPTY_ACCOUNT_OBSERVED",
]
