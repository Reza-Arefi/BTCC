"""Persistent local order lifecycle records.

Do NOT assume FILLED merely because submission returned successfully.
Transitions go through order_fsm.assert_transition.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from btcc.execution.order_fsm import OrderTransitionError, assert_transition


class OrderLifecycleStatus(str, Enum):
    NEW = "NEW"
    SUBMIT_PENDING = "SUBMIT_PENDING"  # submission attempted; ack unknown
    SUBMITTED = "SUBMITTED"  # exchange ack received
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCEL_PENDING = "CANCEL_PENDING"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    ERROR = "ERROR"
    RECONCILE_UNKNOWN = "RECONCILE_UNKNOWN"


UNRESOLVED_STATUSES = frozenset(
    {
        OrderLifecycleStatus.NEW.value,
        OrderLifecycleStatus.SUBMIT_PENDING.value,
        OrderLifecycleStatus.SUBMITTED.value,
        OrderLifecycleStatus.PARTIALLY_FILLED.value,
        OrderLifecycleStatus.CANCEL_PENDING.value,
        OrderLifecycleStatus.RECONCILE_UNKNOWN.value,
        OrderLifecycleStatus.ERROR.value,
    }
)


@dataclass
class OrderRecord:
    order_local_id: str
    intent_id: str
    client_order_id: str
    symbol: str
    side: str
    order_kind: str
    status: str = OrderLifecycleStatus.NEW.value
    requested_quantity: float = 0.0
    executed_quantity: float = 0.0
    remaining_quantity: float = 0.0
    average_fill_price: float | None = None
    fees: float | None = None
    exchange_order_id: str | None = None
    created_ts: str = ""
    updated_ts: str = ""
    error: str | None = None
    reconciliation_state: str = "LOCAL_ONLY"
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        if not self.created_ts:
            self.created_ts = now
        if not self.updated_ts:
            self.updated_ts = now
        if self.remaining_quantity <= 0 and self.requested_quantity > 0 and self.executed_quantity == 0:
            self.remaining_quantity = self.requested_quantity

    @property
    def protective_quantity(self) -> float:
        """Exit/protection must use executed qty, never unfilled requested qty."""
        return float(self.executed_quantity)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> OrderRecord:
        return cls(**{k: raw[k] for k in cls.__dataclass_fields__ if k in raw})  # type: ignore[arg-type]


class OrderJournal:
    """Append-only JSONL order lifecycle log."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def append(self, record: OrderRecord, *, previous_status: str | None = None) -> OrderRecord:
        """Append a new lifecycle snapshot.

        If previous_status is provided (or a prior record exists for the same
        client_order_id), validate the transition via the FSM.
        """
        prior = self.latest_by_client_order_id(record.client_order_id)
        from_status = previous_status
        if from_status is None and prior is not None:
            from_status = prior.status
        if from_status is not None and from_status != record.status:
            assert_transition(from_status, record.status)

        record.updated_ts = datetime.now(timezone.utc).isoformat()
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), sort_keys=True, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return record

    def transition(
        self,
        client_order_id: str,
        new_status: str,
        **updates: Any,
    ) -> OrderRecord:
        """Create next snapshot from latest record with FSM-checked transition."""
        prior = self.latest_by_client_order_id(client_order_id)
        if prior is None:
            raise OrderTransitionError(f"no prior order for {client_order_id}")
        assert_transition(prior.status, new_status)
        data = prior.to_dict()
        data.update(updates)
        data["status"] = new_status
        data["order_local_id"] = f"{prior.order_local_id}:{new_status}"
        rec = OrderRecord.from_dict(data)
        return self.append(rec, previous_status=prior.status)

    def latest_by_client_order_id(self, client_order_id: str) -> OrderRecord | None:
        hit = None
        for row in self._read_all():
            if row.get("client_order_id") == client_order_id:
                hit = OrderRecord.from_dict(row)
        return hit

    def has_unresolved(self, client_order_id: str) -> bool:
        rec = self.latest_by_client_order_id(client_order_id)
        if rec is None:
            return False
        return rec.status in UNRESOLVED_STATUSES

    def unresolved_orders(self) -> list[OrderRecord]:
        """Latest snapshot for every client_order_id still in an unresolved state."""
        return [r for r in self.all_latest().values() if r.status in UNRESOLVED_STATUSES]

    def any_unresolved(self) -> bool:
        return bool(self.unresolved_orders())

    def all_latest(self) -> dict[str, OrderRecord]:
        """Latest record per client_order_id."""
        out: dict[str, OrderRecord] = {}
        for row in self._read_all():
            rec = OrderRecord.from_dict(row)
            out[rec.client_order_id] = rec
        return out

    def actual_exposure_fraction(self, *, allocation_per_full_fill: float = 0.25) -> float:
        """Sum executed/requested * allocation for non-terminal cancelled/rejected orders.

        Uses actual fills for exposure, not requested size alone.
        """
        total = 0.0
        for rec in self.all_latest().values():
            if rec.status in {
                OrderLifecycleStatus.CANCELLED.value,
                OrderLifecycleStatus.REJECTED.value,
                OrderLifecycleStatus.ERROR.value,
            }:
                continue
            if rec.requested_quantity <= 0:
                continue
            filled_frac = min(1.0, max(0.0, rec.executed_quantity / rec.requested_quantity))
            # Prefer explicit allocation in meta when present
            alloc = float(rec.meta.get("allocation_pct", allocation_per_full_fill))
            total += alloc * filled_frac
        return total

    def _read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return []
        out = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def iter_records(self) -> Iterable[OrderRecord]:
        for row in self._read_all():
            yield OrderRecord.from_dict(row)
