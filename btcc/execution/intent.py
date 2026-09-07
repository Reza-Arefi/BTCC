"""Durable trade-intent journal (append-only JSONL).

Invariant:
  INTENT PERSISTED  →  only then may order submission be attempted.
  If persistence fails → DO NOT submit.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable


class IntentStatus(str, Enum):
    CREATED = "CREATED"
    PERSISTED = "PERSISTED"
    RISK_REJECTED = "RISK_REJECTED"
    SUBMISSION_BLOCKED = "SUBMISSION_BLOCKED"
    SUBMIT_ATTEMPTED = "SUBMIT_ATTEMPTED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    DUPLICATE = "DUPLICATE"


@dataclass
class DurableTradeIntent:
    intent_id: str
    client_order_id: str
    strategy_version: str
    symbol: str
    side: str
    requested_allocation_pct: float
    requested_quantity: float
    signal_ts: str
    candle_ts: str
    selected_exit: str
    execution_mode: str
    status: str = IntentStatus.CREATED.value
    created_ts: str = ""
    order_kind: str = "ENT"
    current_strategy_state: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.created_ts:
            self.created_ts = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DurableTradeIntent:
        return cls(**{k: raw[k] for k in cls.__dataclass_fields__ if k in raw})  # type: ignore[arg-type]


class IntentPersistenceError(RuntimeError):
    """Raised when durable intent cannot be persisted — submission must not proceed."""


class IntentJournal:
    """Append-only JSONL intent log with duplicate detection by intent_id / client_order_id."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def _read_all(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if not self.path.exists() or self.path.stat().st_size == 0:
            return rows
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    def find_by_intent_id(self, intent_id: str) -> DurableTradeIntent | None:
        for row in reversed(self._read_all()):
            if row.get("intent_id") == intent_id:
                return DurableTradeIntent.from_dict(row)
        return None

    def find_by_client_order_id(self, client_order_id: str) -> DurableTradeIntent | None:
        for row in reversed(self._read_all()):
            if row.get("client_order_id") == client_order_id:
                return DurableTradeIntent.from_dict(row)
        return None

    def has_active_or_submitted(self, client_order_id: str) -> bool:
        hit = self.find_by_client_order_id(client_order_id)
        if hit is None:
            return False
        return hit.status not in {
            IntentStatus.RISK_REJECTED.value,
            IntentStatus.FAILED.value,
            IntentStatus.CANCELLED.value,
            IntentStatus.DUPLICATE.value,
        }

    def append(self, intent: DurableTradeIntent) -> DurableTradeIntent:
        """Persist intent. On failure raises IntentPersistenceError (fail closed)."""
        existing = self.find_by_client_order_id(intent.client_order_id)
        if existing is not None and existing.status not in {
            IntentStatus.RISK_REJECTED.value,
            IntentStatus.FAILED.value,
            IntentStatus.CANCELLED.value,
            IntentStatus.DUPLICATE.value,
        }:
            dup = DurableTradeIntent.from_dict(existing.to_dict())
            dup.status = IntentStatus.DUPLICATE.value
            dup.meta = {**dup.meta, "duplicate_of": existing.intent_id}
            # Record duplicate marker but do not allow submission path to proceed.
            self._append_raw(dup.to_dict())
            raise IntentPersistenceError(
                f"Duplicate client_order_id already persisted: {intent.client_order_id}"
            )

        intent.status = IntentStatus.PERSISTED.value
        try:
            self._append_raw(intent.to_dict())
        except Exception as e:  # noqa: BLE001 — must fail closed
            raise IntentPersistenceError(f"Failed to persist trade intent: {e}") from e
        return intent

    def update_status(self, intent_id: str, status: str, **extra: Any) -> DurableTradeIntent:
        cur = self.find_by_intent_id(intent_id)
        if cur is None:
            raise IntentPersistenceError(f"Unknown intent_id={intent_id}")
        cur.status = status
        if extra:
            cur.meta = {**cur.meta, **extra}
        self._append_raw(cur.to_dict())
        return cur

    def _append_raw(self, row: dict[str, Any]) -> None:
        line = json.dumps(row, sort_keys=True, default=str) + "\n"
        # Append with fsync for durability.
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

    def iter_intents(self) -> Iterable[DurableTradeIntent]:
        for row in self._read_all():
            yield DurableTradeIntent.from_dict(row)
