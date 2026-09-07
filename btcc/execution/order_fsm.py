"""Deterministic order lifecycle state machine.

Illegal transitions raise OrderTransitionError.
Do NOT infer FILLED from SUBMITTED / ACK alone.

Status string constants mirror OrderLifecycleStatus values (kept here to avoid
circular imports with order_state).
"""

from __future__ import annotations

from enum import Enum


class OrderTransitionError(ValueError):
    """Raised when an illegal lifecycle transition is attempted."""


# Mirror of OrderLifecycleStatus values
NEW = "NEW"
SUBMIT_PENDING = "SUBMIT_PENDING"
SUBMITTED = "SUBMITTED"
PARTIALLY_FILLED = "PARTIALLY_FILLED"
FILLED = "FILLED"
CANCEL_PENDING = "CANCEL_PENDING"
CANCELLED = "CANCELLED"
REJECTED = "REJECTED"
ERROR = "ERROR"
RECONCILE_UNKNOWN = "RECONCILE_UNKNOWN"


_LEGAL: dict[str, frozenset[str]] = {
    NEW: frozenset({SUBMIT_PENDING, REJECTED, CANCELLED, ERROR}),
    SUBMIT_PENDING: frozenset(
        {
            SUBMITTED,
            RECONCILE_UNKNOWN,
            REJECTED,
            ERROR,
            CANCELLED,
            PARTIALLY_FILLED,
            FILLED,
        }
    ),
    SUBMITTED: frozenset(
        {
            PARTIALLY_FILLED,
            FILLED,
            CANCEL_PENDING,
            CANCELLED,
            RECONCILE_UNKNOWN,
            ERROR,
            REJECTED,
        }
    ),
    PARTIALLY_FILLED: frozenset(
        {
            PARTIALLY_FILLED,
            FILLED,
            CANCEL_PENDING,
            CANCELLED,
            RECONCILE_UNKNOWN,
            ERROR,
        }
    ),
    FILLED: frozenset({RECONCILE_UNKNOWN}),
    CANCEL_PENDING: frozenset(
        {CANCELLED, PARTIALLY_FILLED, FILLED, RECONCILE_UNKNOWN, ERROR}
    ),
    CANCELLED: frozenset({RECONCILE_UNKNOWN, FILLED, PARTIALLY_FILLED}),
    REJECTED: frozenset({RECONCILE_UNKNOWN}),
    ERROR: frozenset({RECONCILE_UNKNOWN, SUBMIT_PENDING}),
    RECONCILE_UNKNOWN: frozenset(
        {
            SUBMIT_PENDING,
            SUBMITTED,
            PARTIALLY_FILLED,
            FILLED,
            CANCELLED,
            REJECTED,
            ERROR,
        }
    ),
}


def legal_transitions() -> dict[str, frozenset[str]]:
    return {k: frozenset(v) for k, v in _LEGAL.items()}


def can_transition(current: str, new: str) -> bool:
    if current == new and current == PARTIALLY_FILLED:
        return True
    return new in _LEGAL.get(current, frozenset())


def assert_transition(current: str, new: str) -> str:
    if not can_transition(current, new):
        raise OrderTransitionError(f"illegal order transition {current} → {new}")
    return new


class ReconciliationState(str, Enum):
    LOCAL_ONLY = "LOCAL_ONLY"
    AWAITING_ACK = "AWAITING_ACK"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RECONCILE_REQUIRED = "RECONCILE_REQUIRED"
    MATCHED = "MATCHED"
    MISMATCH = "MISMATCH"
    PAPER_LOCAL = "PAPER_LOCAL"
    UNCONFIRMED = "UNCONFIRMED"
