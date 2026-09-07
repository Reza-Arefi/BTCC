"""Execution safety state — distinct from strategy HALT.

NORMAL   — exchange truth established; REAL readiness may proceed later
DEGRADED — partial issues; new REAL entries must not proceed
HALTED   — safety cannot be established; REAL entries blocked
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ExecutionSafetyStatus(str, Enum):
    NORMAL = "NORMAL"
    DEGRADED = "DEGRADED"
    HALTED = "HALTED"


# Canonical halt reason codes (execution layer, not strategy)
AUTH_FAILURE = "AUTH_FAILURE"
ACCOUNT_UNAVAILABLE = "ACCOUNT_UNAVAILABLE"
SYMBOL_METADATA_UNAVAILABLE = "SYMBOL_METADATA_UNAVAILABLE"
RECONCILIATION_MISMATCH = "EXECUTION_RECONCILIATION_FAILURE"
STALE_ACCOUNT = "STALE_ACCOUNT"
RATE_LIMIT_FAILURE = "RATE_LIMIT_FAILURE"
MALFORMED_EXCHANGE_RESPONSE = "MALFORMED_EXCHANGE_RESPONSE"
TIMEOUT = "TIMEOUT"
MISSING_CREDENTIALS = "MISSING_CREDENTIALS"
REAL_TRADING_DISABLED = "REAL_TRADING_DISABLED"


@dataclass
class ExecutionSafetyState:
    """Mutable execution-layer safety. Independent of strategy halt flags."""

    status: ExecutionSafetyStatus = ExecutionSafetyStatus.HALTED
    reason_code: str = REAL_TRADING_DISABLED
    detail: str = "Stage 3 default: REAL not enabled; fail closed until reconcile succeeds"
    real_entries_blocked: bool = True
    updated_ts: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.updated_ts:
            self.updated_ts = datetime.now(timezone.utc).isoformat()

    @property
    def is_halted(self) -> bool:
        return self.status == ExecutionSafetyStatus.HALTED

    @property
    def is_normal(self) -> bool:
        return self.status == ExecutionSafetyStatus.NORMAL

    def mark_normal(self, detail: str = "reconcile matched") -> None:
        self.status = ExecutionSafetyStatus.NORMAL
        self.reason_code = "OK"
        self.detail = detail
        self.real_entries_blocked = False  # Stage 3: still no submit path; flag for future
        self.updated_ts = datetime.now(timezone.utc).isoformat()

    def mark_degraded(self, reason_code: str, detail: str) -> None:
        self.status = ExecutionSafetyStatus.DEGRADED
        self.reason_code = reason_code
        self.detail = detail
        self.real_entries_blocked = True
        self.updated_ts = datetime.now(timezone.utc).isoformat()

    def mark_halted(self, reason_code: str, detail: str) -> None:
        self.status = ExecutionSafetyStatus.HALTED
        self.reason_code = reason_code
        self.detail = detail
        self.real_entries_blocked = True
        self.updated_ts = datetime.now(timezone.utc).isoformat()

    def apply_read_error(self, exc: BaseException) -> None:
        """Map MEXC read failures onto HALTED (fail closed)."""
        name = type(exc).__name__
        msg = str(exc)
        if "Auth" in name or "MissingCredentials" in name:
            self.mark_halted(AUTH_FAILURE if "Auth" in name else MISSING_CREDENTIALS, msg)
        elif "RateLimit" in name:
            self.mark_halted(RATE_LIMIT_FAILURE, msg)
        elif "Timeout" in name:
            self.mark_halted(TIMEOUT, msg)
        elif "Malformed" in name:
            self.mark_halted(MALFORMED_EXCHANGE_RESPONSE, msg)
        elif "SymbolMetadata" in name:
            self.mark_halted(SYMBOL_METADATA_UNAVAILABLE, msg)
        elif "Account" in name:
            self.mark_halted(ACCOUNT_UNAVAILABLE, msg)
        else:
            self.mark_halted(ACCOUNT_UNAVAILABLE, f"{name}: {msg}")
