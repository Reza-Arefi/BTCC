"""Simulated-day indexing for walk-forward backtests (Day 1..N)."""

from __future__ import annotations

import pandas as pd


def day_number_at(ts, eval_start) -> int:
    """1-indexed day number relative to eval_start (usable simulation start)."""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    e = pd.Timestamp(eval_start)
    if e.tzinfo is None:
        e = e.tz_localize("UTC")
    else:
        e = e.tz_convert("UTC")
    return int((t.normalize() - e.normalize()).days) + 1


def attach_day_number(df: pd.DataFrame, ts_col: str, eval_start) -> pd.DataFrame:
    """Add day_number column from timestamps (no lookahead)."""
    if df is None or df.empty or ts_col not in df.columns:
        out = df.copy() if df is not None else pd.DataFrame()
        if "day_number" not in out.columns:
            out["day_number"] = pd.Series(dtype="Int64")
        return out
    out = df.copy()
    ts = pd.to_datetime(out[ts_col], utc=True, errors="coerce")
    e = pd.Timestamp(eval_start)
    if e.tzinfo is None:
        e = e.tz_localize("UTC")
    else:
        e = e.tz_convert("UTC")
    e_norm = e.normalize()
    out["day_number"] = ((ts.dt.normalize() - e_norm).dt.days + 1).astype("Int64")
    return out
