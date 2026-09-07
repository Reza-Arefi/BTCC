"""Outcome maturity helpers — prevent premature use of future labels in learning."""

from __future__ import annotations

import pandas as pd


def outcome_mature_at(
    prediction_ts,
    *,
    horizon_hours: int = 4,
    asof_ts,
) -> bool:
    """True iff prediction_ts + horizon_hours <= asof_ts (UTC-aware)."""
    p = pd.Timestamp(prediction_ts)
    a = pd.Timestamp(asof_ts)
    if p.tzinfo is None:
        p = p.tz_localize("UTC")
    else:
        p = p.tz_convert("UTC")
    if a.tzinfo is None:
        a = a.tz_localize("UTC")
    else:
        a = a.tz_convert("UTC")
    return (p + pd.Timedelta(hours=int(horizon_hours))) <= a


def filter_matured_for_learning(
    rows: list[dict] | pd.DataFrame,
    *,
    asof_ts,
    horizon_hours: int = 4,
    outcome_col: str = "future_return_4h",
) -> pd.DataFrame:
    """Return only rows whose outcome label is allowed to influence learning at asof_ts.

    Requires:
      - outcome_col is non-null
      - prediction_timestamp + horizon_hours <= asof_ts
    """
    if isinstance(rows, list):
        df = pd.DataFrame(rows)
    else:
        df = rows.copy()
    if df.empty or "timestamp" not in df.columns:
        return df.iloc[0:0].copy() if not df.empty else df
    if outcome_col not in df.columns:
        return df.iloc[0:0].copy()

    ts = pd.to_datetime(df["timestamp"], utc=True)
    asof = pd.Timestamp(asof_ts)
    if asof.tzinfo is None:
        asof = asof.tz_localize("UTC")
    else:
        asof = asof.tz_convert("UTC")
    mature = (ts + pd.Timedelta(hours=int(horizon_hours))) <= asof
    has_y = df[outcome_col].notna()
    return df.loc[mature & has_y].copy()
