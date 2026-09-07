"""Explicit live trade lifecycle states."""

from __future__ import annotations

from enum import Enum


class TradeStatus(str, Enum):
    SIGNAL = "SIGNAL"
    RESERVED = "RESERVED"
    ENTRY_PENDING = "ENTRY_PENDING"
    ENTRY_FILLED = "ENTRY_FILLED"
    PROTECTION_PENDING = "PROTECTION_PENDING"
    PROTECTED = "PROTECTED"
    PROTECTED_EMERGENCY = "PROTECTED_EMERGENCY"
    EXIT_PENDING = "EXIT_PENDING"
    CLOSED = "CLOSED"
    # Failure / halt
    PROTECTION_FAILED = "PROTECTION_FAILED"
    RECONCILIATION_FAILED = "RECONCILIATION_FAILED"
    UNKNOWN_ORDER_STATE = "UNKNOWN_ORDER_STATE"
    HALTED = "HALTED"


# States that occupy a portfolio slot (count toward max simultaneous).
# PROTECTION_FAILED keeps the slot — inventory still exists / must be reconciled.
ACTIVE_SLOT_STATES: frozenset[TradeStatus] = frozenset(
    {
        TradeStatus.RESERVED,
        TradeStatus.ENTRY_PENDING,
        TradeStatus.ENTRY_FILLED,
        TradeStatus.PROTECTION_PENDING,
        TradeStatus.PROTECTED,
        TradeStatus.PROTECTED_EMERGENCY,
        TradeStatus.EXIT_PENDING,
        TradeStatus.PROTECTION_FAILED,
    }
)

# Only fully protected positions are "normal" active trades for ops reporting.
PROTECTED_ACTIVE_STATES: frozenset[TradeStatus] = frozenset(
    {TradeStatus.PROTECTED, TradeStatus.PROTECTED_EMERGENCY}
)
