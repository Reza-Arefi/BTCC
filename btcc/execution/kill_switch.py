"""Execution-level kill switch (distinct from strategy safety HALT).

STRATEGY HALT  → blocks new strategy entries
EXECUTION HALT → blocks new real order submissions

Does NOT block (when designed correctly):
  - monitoring existing positions
  - reconciliation
  - order/fill status retrieval
  - safe exits of existing real positions
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ExecutionPermissions:
    allow_new_entries: bool = True
    allow_order_submission: bool = True
    allow_reconciliation: bool = True
    allow_exits: bool = True
    reason: str = ""
    updated_ts: str = ""

    def __post_init__(self) -> None:
        if not self.updated_ts:
            self.updated_ts = datetime.now(timezone.utc).isoformat()


class ExecutionKillSwitch:
    """Persistent execution control plane. Fail-closed for REAL submissions."""

    def __init__(self, path: Path | None = None, *, permissions: ExecutionPermissions | None = None) -> None:
        self.path = Path(path) if path is not None else None
        if permissions is not None:
            self._perm = permissions
        elif self.path is not None and self.path.exists():
            self._perm = self._load()
        else:
            # Default: paper-friendly; REAL still globally disabled elsewhere.
            self._perm = ExecutionPermissions()

    @property
    def permissions(self) -> ExecutionPermissions:
        return self._perm

    def halt_submissions(self, reason: str = "EXECUTION_HALT") -> ExecutionPermissions:
        self._perm = ExecutionPermissions(
            allow_new_entries=False,
            allow_order_submission=False,
            allow_reconciliation=True,
            allow_exits=True,
            reason=reason,
        )
        self._save()
        return self._perm

    def halt_all_trading_actions(self, reason: str = "EXECUTION_FULL_HALT") -> ExecutionPermissions:
        """Extreme halt: no new entries/submissions/exits; reconciliation still allowed."""
        self._perm = ExecutionPermissions(
            allow_new_entries=False,
            allow_order_submission=False,
            allow_reconciliation=True,
            allow_exits=False,
            reason=reason,
        )
        self._save()
        return self._perm

    def resume(self, reason: str = "RESUMED") -> ExecutionPermissions:
        self._perm = ExecutionPermissions(reason=reason)
        self._save()
        return self._perm

    def can_submit_new_order(self) -> bool:
        return bool(self._perm.allow_order_submission and self._perm.allow_new_entries)

    def can_reconcile(self) -> bool:
        return bool(self._perm.allow_reconciliation)

    def can_exit(self) -> bool:
        return bool(self._perm.allow_exits)

    def _load(self) -> ExecutionPermissions:
        assert self.path is not None
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        return ExecutionPermissions(**{k: raw[k] for k in ExecutionPermissions.__dataclass_fields__ if k in raw})

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(asdict(self._perm), indent=2, sort_keys=True) + "\n"
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, self.path)
