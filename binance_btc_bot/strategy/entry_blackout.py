"""Weekly UTC entry blackout window (blocks NEW entries only).

Open positions keep their exchange-side OCO protection; the blackout only
suppresses opening new trades. Windows may wrap across the week boundary,
e.g. ``FRI 12:00`` → ``MON 00:00``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

WEEKDAYS = ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")
_MIN_PER_WEEK = 7 * 24 * 60


def parse_week_time(text: str) -> int:
    """``'FRI 12:00'`` → minutes since Monday 00:00 UTC."""
    parts = str(text).strip().upper().split()
    if len(parts) != 2 or parts[0] not in WEEKDAYS:
        raise ValueError(f"entry blackout time must look like 'FRI 12:00' (got {text!r})")
    hh, _, mm = parts[1].partition(":")
    h, m = int(hh), int(mm or 0)
    if not (0 <= h < 24 and 0 <= m < 60):
        raise ValueError(f"entry blackout time out of range: {text!r}")
    return WEEKDAYS.index(parts[0]) * 1440 + h * 60 + m


def format_week_time(minutes: int) -> str:
    d, rem = divmod(int(minutes) % _MIN_PER_WEEK, 1440)
    return f"{WEEKDAYS[d]} {rem // 60:02d}:{rem % 60:02d}"


@dataclass(frozen=True)
class EntryBlackout:
    start_min: int
    end_min: int

    @classmethod
    def from_config(cls, raw: Mapping[str, Any] | None) -> "EntryBlackout | None":
        if not raw or not bool(raw.get("enabled", False)):
            return None
        tz = str(raw.get("timezone") or "UTC").upper()
        if tz != "UTC":
            raise ValueError(f"entry.blackout.timezone must be UTC (got {tz!r})")
        start = parse_week_time(str(raw.get("start") or ""))
        end = parse_week_time(str(raw.get("end") or ""))
        if start == end:
            raise ValueError("entry.blackout start and end must differ")
        return cls(start_min=start, end_min=end)

    def label(self) -> str:
        return f"{format_week_time(self.start_min)} -> {format_week_time(self.end_min)} UTC"

    def is_blocked(self, when: datetime | None = None) -> bool:
        ts = when or datetime.now(timezone.utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        ts = ts.astimezone(timezone.utc)
        m = ts.weekday() * 1440 + ts.hour * 60 + ts.minute
        if self.start_min < self.end_min:
            return self.start_min <= m < self.end_min
        return m >= self.start_min or m < self.end_min
