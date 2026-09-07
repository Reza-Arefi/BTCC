"""Brazil display timezone for Telegram presentation only.

Internal/exchange timestamps remain UTC. Convert only at display time via
zoneinfo America/Sao_Paulo (never manual ±3h arithmetic).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

DISPLAY_TZ = ZoneInfo("America/Sao_Paulo")
DISPLAY_TZ_NAME = "BRT"  # label; actual offset comes from zoneinfo


def _display_tz_label(local: datetime) -> str:
    """Prefer 'BRT'; if offset ≠ UTC-03:00, append the real zone offset."""
    off = local.utcoffset()
    if off is None:
        return DISPLAY_TZ_NAME
    total = int(off.total_seconds())
    if total == -3 * 3600:
        return DISPLAY_TZ_NAME
    sign = "+" if total >= 0 else "-"
    total = abs(total)
    hours, rem = divmod(total, 3600)
    mins = rem // 60
    return f"{DISPLAY_TZ_NAME} (UTC{sign}{hours:02d}:{mins:02d})"


def parse_utc(value: Any) -> datetime | None:
    """Parse authoritative UTC timestamps (ISO / epoch). Returns aware UTC dt."""
    if value is None or value == "" or value == "n/a" or value == "N/A":
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    s = str(value).strip()
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt
    except Exception:  # noqa: BLE001
        return None


def to_brt(dt_utc: datetime) -> datetime:
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    return dt_utc.astimezone(DISPLAY_TZ)


def format_brt(
    value: Any,
    *,
    with_date: bool = True,
    fallback: str = "N/A",
) -> str:
    """Format a UTC/internal timestamp for Telegram as local BRT."""
    dt = parse_utc(value)
    if dt is None:
        return fallback
    local = to_brt(dt)
    abbr = _display_tz_label(local)
    if with_date:
        return local.strftime(f"%Y-%m-%d %H:%M:%S {abbr}")
    return local.strftime(f"%H:%M:%S {abbr}")


def format_brt_hm(value: Any, *, fallback: str = "N/A") -> str:
    dt = parse_utc(value)
    if dt is None:
        return fallback
    local = to_brt(dt)
    abbr = _display_tz_label(local)
    return local.strftime(f"%H:%M {abbr}")


def format_hourly_window(start_utc: Any, end_utc: Any) -> str:
    """e.g. 15:00 → 16:00 BRT"""
    a = parse_utc(start_utc)
    b = parse_utc(end_utc)
    if a is None or b is None:
        return "N/A → N/A BRT"
    la, lb = to_brt(a), to_brt(b)
    abbr = _display_tz_label(lb)
    return f"{la.strftime('%H:%M')} → {lb.strftime('%H:%M')} {abbr}"
