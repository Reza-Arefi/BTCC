"""Single-shot real canary state machine.

REAL_CANARY_SINGLE_SHOT is NOT unrestricted live trading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CanaryPhase(str, Enum):
    ARMED = "ARMED"  # may take at most one entry when writes enabled
    ENTRY_IN_FLIGHT = "ENTRY_IN_FLIGHT"
    OPEN = "OPEN"  # one position open
    CANARY_COMPLETE = "CANARY_COMPLETE"  # closed; no further entries
    HALTED = "HALTED"


@dataclass
class CanaryState:
    phase: CanaryPhase = CanaryPhase.ARMED
    entries_opened: int = 0
    max_entries: int = 1
    max_positions: int = 1
    position_id: str | None = None
    symbol: str | None = None
    halt_reason: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def allow_new_entry(self) -> bool:
        return (
            self.phase == CanaryPhase.ARMED
            and self.entries_opened < self.max_entries
        )

    def begin_entry(self, *, symbol: str, position_id: str) -> None:
        if not self.allow_new_entry():
            raise RuntimeError(
                f"canary blocks entry: phase={self.phase.value} entries={self.entries_opened}"
            )
        self.phase = CanaryPhase.ENTRY_IN_FLIGHT
        self.symbol = symbol
        self.position_id = position_id

    def mark_open(self) -> None:
        if self.phase != CanaryPhase.ENTRY_IN_FLIGHT:
            raise RuntimeError(f"mark_open invalid from {self.phase}")
        self.entries_opened += 1
        self.phase = CanaryPhase.OPEN

    def mark_complete(self) -> None:
        self.phase = CanaryPhase.CANARY_COMPLETE

    def halt(self, reason: str) -> None:
        self.phase = CanaryPhase.HALTED
        self.halt_reason = reason

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "entries_opened": self.entries_opened,
            "max_entries": self.max_entries,
            "max_positions": self.max_positions,
            "position_id": self.position_id,
            "symbol": self.symbol,
            "halt_reason": self.halt_reason,
            "meta": dict(self.meta),
        }
