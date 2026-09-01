"""Signal crossing state machine — LONG ALT/BTC opportunities.

State per pair:

  FLAT  --(S >= threshold)--> IN_SIGNAL (open opportunity if eligible)
  IN_SIGNAL --(S < threshold)--> FLAT
  While IN_SIGNAL: subsequent bars with S >= threshold do NOT open new trades.

Documented rules:
- First transition into the trade zone creates one opportunity.
- New opportunity only after exit from zone AND a subsequent re-entry.
- One active opportunity per pair while previous is still open.
- Max N open opportunities globally; excess signals are recorded as rejected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PairSignalState(str, Enum):
    FLAT = "FLAT"
    IN_SIGNAL = "IN_SIGNAL"


@dataclass
class CrossingStateMachine:
    long_threshold: float = 0.60
    upper_threshold: float | None = None  # exclusive upper band; None = no cap
    max_open: int = 10
    one_per_pair: bool = True
    # pair_key -> PairSignalState (zone membership, independent of open trade)
    zone_state: dict[str, PairSignalState] = field(default_factory=dict)
    # pair_key -> opportunity_id if an open opportunity exists for that pair
    open_by_pair: dict[str, str] = field(default_factory=dict)
    # opportunity_id set
    open_ids: set[str] = field(default_factory=set)

    def n_open(self) -> int:
        return len(self.open_ids)

    def slots_remaining(self) -> int:
        return max(0, self.max_open - self.n_open())

    def _in_band(self, s: float) -> bool:
        if s < self.long_threshold:
            return False
        if self.upper_threshold is not None and s >= float(self.upper_threshold):
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "long_threshold": self.long_threshold,
            "upper_threshold": self.upper_threshold,
            "max_open": self.max_open,
            "one_per_pair": self.one_per_pair,
            "zone_state": {k: v.value for k, v in self.zone_state.items()},
            "open_by_pair": dict(self.open_by_pair),
            "open_ids": sorted(self.open_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CrossingStateMachine":
        sm = cls(
            long_threshold=float(data.get("long_threshold", 0.60)),
            upper_threshold=(
                float(data["upper_threshold"]) if data.get("upper_threshold") is not None else None
            ),
            max_open=int(data.get("max_open", 10)),
            one_per_pair=bool(data.get("one_per_pair", True)),
        )
        for k, v in (data.get("zone_state") or {}).items():
            sm.zone_state[k] = PairSignalState(v)
        sm.open_by_pair = dict(data.get("open_by_pair") or {})
        sm.open_ids = set(data.get("open_ids") or [])
        return sm

    def register_open(self, pair: str, opportunity_id: str) -> None:
        self.open_by_pair[pair] = opportunity_id
        self.open_ids.add(opportunity_id)

    def register_close(self, opportunity_id: str, pair: str | None = None) -> None:
        self.open_ids.discard(opportunity_id)
        if pair and self.open_by_pair.get(pair) == opportunity_id:
            del self.open_by_pair[pair]
        else:
            for p, oid in list(self.open_by_pair.items()):
                if oid == opportunity_id:
                    del self.open_by_pair[p]

    def evaluate(
        self,
        pair: str,
        S: float,
    ) -> dict[str, Any]:
        """Evaluate one bar for a pair. Does not mutate open trade book except via caller.

        Returns decision dict:
          signal_generated, crossed_into, still_in_zone, crossed_out,
          trade_opened (suggested), rejection_reason
        """
        prev = self.zone_state.get(pair, PairSignalState.FLAT)
        in_zone = self._in_band(S)
        crossed_into = (prev == PairSignalState.FLAT) and in_zone
        crossed_out = (prev == PairSignalState.IN_SIGNAL) and (not in_zone)
        still_in_zone = (prev == PairSignalState.IN_SIGNAL) and in_zone

        if in_zone:
            self.zone_state[pair] = PairSignalState.IN_SIGNAL
        else:
            self.zone_state[pair] = PairSignalState.FLAT

        signal_generated = in_zone
        trade_suggested = False
        rejection: str | None = None

        if not in_zone:
            if S >= self.long_threshold and self.upper_threshold is not None and S >= self.upper_threshold:
                rejection = "ABOVE_THRESHOLD"
            else:
                rejection = "BELOW_THRESHOLD"
        elif still_in_zone:
            # Already in qualifying zone — do not open another
            rejection = "SIGNAL_CONTINUATION"
        elif crossed_into:
            if self.one_per_pair and pair in self.open_by_pair:
                rejection = "SAME_PAIR_ALREADY_OPEN"
            elif self.n_open() >= self.max_open:
                rejection = "MAX_OPEN_TRADES"
            else:
                trade_suggested = True
                rejection = None
        else:
            rejection = "BELOW_THRESHOLD"

        return {
            "pair": pair,
            "S": S,
            "threshold": self.long_threshold,
            "upper_threshold": self.upper_threshold,
            "prev_zone": prev.value,
            "zone": self.zone_state[pair].value,
            "signal_generated": signal_generated,
            "crossed_into": crossed_into,
            "crossed_out": crossed_out,
            "still_in_zone": still_in_zone,
            "trade_suggested": trade_suggested,
            "rejection_reason": rejection,
            "n_open": self.n_open(),
            "slots_remaining": self.slots_remaining(),
        }
