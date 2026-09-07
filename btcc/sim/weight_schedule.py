"""Weight schedule: calculated_at vs effective_from (next eligible prediction)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from btcc.sim.score import normalize_weights


@dataclass
class WeightVersion:
    version_id: str
    weights: dict[str, float]
    calculated_at: str
    effective_from: str
    learning_window_start: str | None = None
    learning_window_end: str | None = None
    n_samples: int | None = None
    phase: str = "daily"  # init | daily
    update_number: int = 0
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "weights": dict(self.weights),
            "calculated_at": self.calculated_at,
            "effective_from": self.effective_from,
            "learning_window_start": self.learning_window_start,
            "learning_window_end": self.learning_window_end,
            "n_samples": self.n_samples,
            "phase": self.phase,
            "update_number": self.update_number,
            "stats": self.stats,
        }


class WeightSchedule:
    """Active weights are those with effective_from <= decision_ts (UTC)."""

    def __init__(self, initial: dict[str, float], *, version_id: str = "config_initial"):
        w = normalize_weights(initial)
        self._versions: list[WeightVersion] = [
            WeightVersion(
                version_id=version_id,
                weights=w,
                calculated_at="epoch",
                effective_from="1970-01-01T00:00:00+00:00",
                phase="config",
                update_number=0,
            )
        ]

    def add(self, version: WeightVersion) -> None:
        version.weights = normalize_weights(version.weights)
        self._versions.append(version)
        self._versions.sort(key=lambda v: pd.Timestamp(v.effective_from))

    def active_at(self, decision_ts) -> WeightVersion:
        t = pd.Timestamp(decision_ts)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        else:
            t = t.tz_convert("UTC")
        active = self._versions[0]
        for v in self._versions:
            ef = pd.Timestamp(v.effective_from)
            if ef.tzinfo is None:
                ef = ef.tz_localize("UTC")
            if ef <= t:
                active = v
        return active

    def history(self) -> list[dict[str, Any]]:
        return [v.to_dict() for v in self._versions]
