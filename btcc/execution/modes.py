"""Execution mode enum — PAPER / REAL / REAL_SHADOW / TEST / canary / bounded 6h.

REAL                     — write-capable name; gated by WriteGate (default closed)
REAL_SHADOW              — authenticated MEXC reads + reconcile only
TEST                     — Stage 4 simulated exchange
REAL_CANARY_SINGLE_SHOT  — Stage 7: at most one real entry; still gated until enablement
REAL_BOUNDED_6H          — bounded multi-position session (≤6h); distinct arming from canary
"""

from __future__ import annotations

from enum import Enum


class ExecutionMode(str, Enum):
    PAPER = "PAPER"
    REAL = "REAL"
    REAL_SHADOW = "REAL_SHADOW"
    TEST = "TEST"
    REAL_CANARY_SINGLE_SHOT = "REAL_CANARY_SINGLE_SHOT"
    REAL_BOUNDED_6H = "REAL_BOUNDED_6H"

    @classmethod
    def parse(cls, value: str | None) -> ExecutionMode:
        if value is None:
            return cls.PAPER
        raw = str(value).strip().upper()
        if raw in ("PAPER", "SIM", "SIMULATION"):
            return cls.PAPER
        if raw in ("TEST", "SIM_EXCHANGE"):
            return cls.TEST
        if raw in ("REAL_SHADOW", "SHADOW", "MEXC_SHADOW", "READ_ONLY_SHADOW"):
            return cls.REAL_SHADOW
        if raw in ("REAL_CANARY_SINGLE_SHOT", "REAL_CANARY", "CANARY", "CANARY_SINGLE_SHOT"):
            return cls.REAL_CANARY_SINGLE_SHOT
        if raw in ("REAL_BOUNDED_6H", "BOUNDED_6H", "REAL_6H", "SIX_HOUR"):
            return cls.REAL_BOUNDED_6H
        if raw in ("REAL", "LIVE", "MEXC_REAL"):
            return cls.REAL
        raise ValueError(
            f"Unknown execution_mode={value!r}; expected PAPER, REAL, REAL_SHADOW, "
            "TEST, REAL_CANARY_SINGLE_SHOT, or REAL_BOUNDED_6H"
        )

    @property
    def allows_order_submission(self) -> bool:
        """Modes that may submit when their respective gates are open.

        PAPER/TEST: paper/sim paths.
        REAL_CANARY_SINGLE_SHOT / REAL_BOUNDED_6H: only when WriteGate open (default false).
        REAL / REAL_SHADOW: never auto-allow unrestricted trading.
        """
        return self in (ExecutionMode.PAPER, ExecutionMode.TEST)

    @property
    def is_real_write_mode(self) -> bool:
        return self in (
            ExecutionMode.REAL,
            ExecutionMode.REAL_CANARY_SINGLE_SHOT,
            ExecutionMode.REAL_BOUNDED_6H,
        )
