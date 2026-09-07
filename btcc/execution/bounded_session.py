"""Bounded 6-hour REAL session state machine.

Hard limits (immutable; cannot be raised by CLI/config after construction):
  duration ≤ 6h, max_positions ≤ 4, allocation ≤ 25%, exposure ≤ 100%

Lifecycle:
  PRECHECK → ARM → RUNNING → ENTRY_CUTOFF → DRAIN → FLAT_RECONCILE → DISARM → COMPLETE
Critical failure → HALTED (fail-closed).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from btcc.safety.no_trading import TradingForbiddenError


# Absolute safety ceiling — not overridable.
MAX_SESSION_DURATION_S = 6 * 3600
MAX_POSITIONS_HARD = 4
MAX_ALLOCATION_HARD = 0.25
MAX_EXPOSURE_HARD = 1.0


class SessionPhase(str, Enum):
    PRECHECK = "PRECHECK"
    ARMED = "ARMED"
    RUNNING = "RUNNING"
    ENTRY_CUTOFF = "ENTRY_CUTOFF"
    DRAIN = "DRAIN"
    FLAT_RECONCILE = "FLAT_RECONCILE"
    DISARMED = "DISARMED"
    COMPLETE = "COMPLETE"
    HALTED = "HALTED"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(ts: str) -> datetime:
    raw = str(ts).strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass
class BoundedSessionConfig:
    """Immutable session policy. Construction rejects unsafe values."""

    duration_s: float = float(MAX_SESSION_DURATION_S)
    max_positions: int = MAX_POSITIONS_HARD
    allocation_pct: float = MAX_ALLOCATION_HARD
    max_total_exposure: float = MAX_EXPOSURE_HARD
    strategy_version: str = "T1-ONLY-BOUNDED-6H"
    require_s_min: float = 0.60

    def __post_init__(self) -> None:
        if float(self.duration_s) <= 0:
            raise TradingForbiddenError("session duration_s must be > 0")
        if float(self.duration_s) > MAX_SESSION_DURATION_S + 1e-9:
            raise TradingForbiddenError(
                f"session duration cannot exceed {MAX_SESSION_DURATION_S}s (6h hard limit)"
            )
        if int(self.max_positions) > MAX_POSITIONS_HARD:
            raise TradingForbiddenError(
                f"max_positions cannot exceed {MAX_POSITIONS_HARD}"
            )
        if int(self.max_positions) < 1:
            raise TradingForbiddenError("max_positions must be >= 1")
        if float(self.allocation_pct) > MAX_ALLOCATION_HARD + 1e-12:
            raise TradingForbiddenError(
                f"allocation_pct cannot exceed {MAX_ALLOCATION_HARD}"
            )
        if float(self.allocation_pct) <= 0:
            raise TradingForbiddenError("allocation_pct must be > 0")
        if float(self.max_total_exposure) > MAX_EXPOSURE_HARD + 1e-12:
            raise TradingForbiddenError(
                f"max_total_exposure cannot exceed {MAX_EXPOSURE_HARD}"
            )
        if float(self.max_total_exposure) <= 0:
            raise TradingForbiddenError("max_total_exposure must be > 0")
        # Freeze numeric fields as plain types
        object.__setattr__(self, "duration_s", float(self.duration_s))
        object.__setattr__(self, "max_positions", int(self.max_positions))
        object.__setattr__(self, "allocation_pct", float(self.allocation_pct))
        object.__setattr__(self, "max_total_exposure", float(self.max_total_exposure))
        object.__setattr__(self, "require_s_min", float(self.require_s_min))

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_s": self.duration_s,
            "max_positions": self.max_positions,
            "allocation_pct": self.allocation_pct,
            "max_total_exposure": self.max_total_exposure,
            "strategy_version": self.strategy_version,
            "require_s_min": self.require_s_min,
            "hard_caps": {
                "max_session_duration_s": MAX_SESSION_DURATION_S,
                "max_positions": MAX_POSITIONS_HARD,
                "max_allocation": MAX_ALLOCATION_HARD,
                "max_exposure": MAX_EXPOSURE_HARD,
            },
        }


@dataclass
class BoundedSession:
    """Explicit session with immutable start + deadline. Cannot extend itself."""

    config: BoundedSessionConfig
    phase: SessionPhase = SessionPhase.PRECHECK
    started_at_utc: str | None = None
    deadline_utc: str | None = None
    entry_cutoff_at_utc: str | None = None
    completed_at_utc: str | None = None
    halt_reason: str | None = None
    open_position_ids: list[str] = field(default_factory=list)
    entries_count: int = 0
    closes_count: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def arm(self, *, now: datetime | None = None) -> None:
        if self.phase not in (SessionPhase.PRECHECK, SessionPhase.ARMED):
            raise TradingForbiddenError(f"cannot arm from phase={self.phase.value}")
        now = now or _utc_now()
        start = now.astimezone(timezone.utc)
        deadline = start + timedelta(seconds=float(self.config.duration_s))
        # Guard against any accidental extension beyond hard ceiling from clock skew.
        max_deadline = start + timedelta(seconds=MAX_SESSION_DURATION_S)
        if deadline > max_deadline:
            deadline = max_deadline
        self.started_at_utc = start.isoformat()
        self.deadline_utc = deadline.isoformat()
        self.entry_cutoff_at_utc = self.deadline_utc
        self.phase = SessionPhase.ARMED

    def start_running(self) -> None:
        if self.phase != SessionPhase.ARMED:
            raise TradingForbiddenError(f"cannot start RUNNING from {self.phase.value}")
        if not self.started_at_utc or not self.deadline_utc:
            raise TradingForbiddenError("session not armed with immutable deadline")
        self.phase = SessionPhase.RUNNING

    def extend_deadline(self, *_a: Any, **_k: Any) -> None:
        """Explicitly forbidden — session cannot extend itself."""
        raise TradingForbiddenError("BOUNDED_6H_SESSION_CANNOT_EXTEND")

    def set_deadline(self, *_a: Any, **_k: Any) -> None:
        raise TradingForbiddenError("BOUNDED_6H_DEADLINE_IMMUTABLE")

    @property
    def deadline(self) -> datetime | None:
        if not self.deadline_utc:
            return None
        return _parse_iso(self.deadline_utc)

    def seconds_until_cutoff(self, *, now: datetime | None = None) -> float | None:
        dl = self.deadline
        if dl is None:
            return None
        now = (now or _utc_now()).astimezone(timezone.utc)
        return (dl - now).total_seconds()

    def past_entry_cutoff(self, *, now: datetime | None = None) -> bool:
        rem = self.seconds_until_cutoff(now=now)
        if rem is None:
            return True  # fail-closed: unknown deadline → no new entries
        return rem <= 0

    def allow_new_entry(self, *, now: datetime | None = None) -> bool:
        if self.phase != SessionPhase.RUNNING:
            return False
        if self.halt_reason:
            return False
        if self.past_entry_cutoff(now=now):
            return False
        return len(self.open_position_ids) < int(self.config.max_positions)

    def mark_entry_cutoff(self, *, now: datetime | None = None) -> None:
        if self.phase in (
            SessionPhase.ENTRY_CUTOFF,
            SessionPhase.DRAIN,
            SessionPhase.FLAT_RECONCILE,
            SessionPhase.DISARMED,
            SessionPhase.COMPLETE,
            SessionPhase.HALTED,
        ):
            return
        if self.phase not in (SessionPhase.RUNNING, SessionPhase.ARMED):
            return
        if not self.past_entry_cutoff(now=now):
            return
        self.phase = SessionPhase.ENTRY_CUTOFF

    def begin_drain(self) -> None:
        if self.phase in (SessionPhase.DRAIN, SessionPhase.FLAT_RECONCILE, SessionPhase.COMPLETE):
            return
        if self.phase not in (SessionPhase.ENTRY_CUTOFF, SessionPhase.RUNNING):
            if self.phase == SessionPhase.HALTED:
                return
        self.phase = SessionPhase.DRAIN

    def begin_flat_reconcile(self) -> None:
        if self.phase == SessionPhase.HALTED:
            return
        self.phase = SessionPhase.FLAT_RECONCILE

    def disarm(self) -> None:
        if self.phase == SessionPhase.HALTED:
            return
        self.phase = SessionPhase.DISARMED

    def complete(self, *, now: datetime | None = None) -> None:
        if self.phase == SessionPhase.HALTED:
            raise TradingForbiddenError("cannot COMPLETE a HALTED session")
        if self.open_position_ids:
            raise TradingForbiddenError("cannot COMPLETE while positions remain open")
        now = now or _utc_now()
        self.completed_at_utc = now.astimezone(timezone.utc).isoformat()
        self.phase = SessionPhase.COMPLETE

    def halt(self, reason: str) -> None:
        self.halt_reason = str(reason)
        self.phase = SessionPhase.HALTED

    def register_open(self, position_id: str) -> None:
        if position_id not in self.open_position_ids:
            self.open_position_ids.append(position_id)
        self.entries_count += 1

    def register_close(self, position_id: str) -> None:
        self.open_position_ids = [p for p in self.open_position_ids if p != position_id]
        self.closes_count += 1

    @property
    def open_count(self) -> int:
        return len(self.open_position_ids)

    @property
    def exposure_pct(self) -> float:
        return float(self.open_count) * float(self.config.allocation_pct)

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "started_at_utc": self.started_at_utc,
            "deadline_utc": self.deadline_utc,
            "entry_cutoff_at_utc": self.entry_cutoff_at_utc,
            "completed_at_utc": self.completed_at_utc,
            "halt_reason": self.halt_reason,
            "open_position_ids": list(self.open_position_ids),
            "entries_count": self.entries_count,
            "closes_count": self.closes_count,
            "config": self.config.to_dict(),
            "meta": dict(self.meta),
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path, *, config: BoundedSessionConfig | None = None) -> "BoundedSession":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg_raw = raw.get("config") or {}
        cfg = config or BoundedSessionConfig(
            duration_s=float(cfg_raw.get("duration_s", MAX_SESSION_DURATION_S)),
            max_positions=int(cfg_raw.get("max_positions", MAX_POSITIONS_HARD)),
            allocation_pct=float(cfg_raw.get("allocation_pct", MAX_ALLOCATION_HARD)),
            max_total_exposure=float(cfg_raw.get("max_total_exposure", MAX_EXPOSURE_HARD)),
            strategy_version=str(cfg_raw.get("strategy_version", "T1-ONLY-BOUNDED-6H")),
            require_s_min=float(cfg_raw.get("require_s_min", 0.60)),
        )
        # Loaded deadline must not exceed hard ceiling from started_at.
        sess = cls(
            config=cfg,
            phase=SessionPhase(str(raw.get("phase", "PRECHECK"))),
            started_at_utc=raw.get("started_at_utc"),
            deadline_utc=raw.get("deadline_utc"),
            entry_cutoff_at_utc=raw.get("entry_cutoff_at_utc"),
            completed_at_utc=raw.get("completed_at_utc"),
            halt_reason=raw.get("halt_reason"),
            open_position_ids=list(raw.get("open_position_ids") or []),
            entries_count=int(raw.get("entries_count") or 0),
            closes_count=int(raw.get("closes_count") or 0),
            meta=dict(raw.get("meta") or {}),
        )
        if sess.started_at_utc and sess.deadline_utc:
            start = _parse_iso(sess.started_at_utc)
            dl = _parse_iso(sess.deadline_utc)
            if (dl - start).total_seconds() > MAX_SESSION_DURATION_S + 1.0:
                raise TradingForbiddenError("loaded session deadline exceeds 6h hard limit — HALT required")
        return sess
