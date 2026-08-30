"""Runtime persistence for continuous live process (crash / reboot recovery).

Stores last successful decision candle and cycle metadata so a restarted
process can skip duplicate work and resume from the next closed bar.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class RuntimeState:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, Any] = {}
        self.load()

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            self._data = {}
            return self._data
        try:
            self._data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Runtime state unreadable (%s) — starting empty", e)
            self._data = {}
        return self._data

    def save(self) -> None:
        import os
        import tempfile

        text = json.dumps(self._data, indent=2, default=str)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), prefix=self.path.name + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def update(self, **kwargs: Any) -> None:
        self._data.update(kwargs)
        self._data["updated_utc"] = datetime.now(timezone.utc).isoformat()
        self.save()

    def last_decision_candle_ts(self) -> str | None:
        return self._data.get("last_decision_candle_ts")

    def mark_cycle_ok(
        self,
        *,
        decision_candle_ts: str,
        n_ranked: int,
        weights_version: str | None = None,
        health_ok: bool | None = None,
    ) -> None:
        self.update(
            last_decision_candle_ts=str(decision_candle_ts),
            last_successful_cycle_utc=datetime.now(timezone.utc).isoformat(),
            last_n_ranked=int(n_ranked),
            last_weights_version=weights_version,
            last_health_ok=health_ok,
            consecutive_failures=0,
        )

    def mark_cycle_failure(self, error: str) -> int:
        n = int(self._data.get("consecutive_failures") or 0) + 1
        self.update(
            consecutive_failures=n,
            last_failure_utc=datetime.now(timezone.utc).isoformat(),
            last_failure=str(error)[:2000],
        )
        return n

    def already_processed(self, decision_candle_ts: str) -> bool:
        last = self.last_decision_candle_ts()
        return last is not None and str(last) == str(decision_candle_ts)
