"""Independent $1,000 starting-capital equity curves (per policy × strategy).

Trade sizing still uses ``notional_usd`` (default $100) per opportunity.
This module tracks a separate standardized portfolio starting at
``starting_capital_usd`` (default $1000) that compounds from closed-trade PnL.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def _pnl_usd(legs: pd.DataFrame) -> pd.Series:
    pnl_btc = pd.to_numeric(legs.get("pnl_btc"), errors="coerce").fillna(0.0)
    px = None
    for col in ("entry_btc_usdt", "exit_btc_usdt", "btc_usdt"):
        if col in legs.columns:
            px = pd.to_numeric(legs[col], errors="coerce")
            break
    if px is None:
        px = pd.Series(0.0, index=legs.index)
    px = px.fillna(0.0)
    # If price missing, leave USD pnl as 0 (do not invent FX)
    return pnl_btc * px


def capital_trade_series(
    legs: pd.DataFrame,
    *,
    starting_capital_usd: float = 1000.0,
) -> pd.DataFrame:
    """Per-trade equity path for each independent (entry_policy, strategy_key)."""
    cols = [
        "entry_policy", "strategy_key", "exit_ts", "day_number",
        "pnl_btc", "pnl_usd", "portfolio_value_usd", "cumulative_pnl_usd",
        "cumulative_return_pct",
    ]
    if legs is None or legs.empty or "strategy_key" not in legs.columns:
        return pd.DataFrame(columns=cols)

    df = legs.copy()
    if "closed" in df.columns:
        df = df[df["closed"] == True]  # noqa: E712
    if df.empty:
        return pd.DataFrame(columns=cols)

    if "entry_policy" not in df.columns:
        df["entry_policy"] = "ALL"
    df["exit_ts"] = pd.to_datetime(df.get("exit_ts"), utc=True, errors="coerce")
    df = df.dropna(subset=["exit_ts"])
    df["pnl_usd"] = _pnl_usd(df)

    rows: list[dict[str, Any]] = []
    for (pol, sk), g in df.groupby(["entry_policy", "strategy_key"], dropna=False):
        g = g.sort_values("exit_ts")
        cum = g["pnl_usd"].cumsum()
        value = float(starting_capital_usd) + cum
        for ts, pnl_u, c, v, dn, pb in zip(
            g["exit_ts"], g["pnl_usd"], cum, value,
            g["day_number"] if "day_number" in g.columns else [None] * len(g),
            pd.to_numeric(g.get("pnl_btc"), errors="coerce").fillna(0.0),
        ):
            rows.append({
                "entry_policy": str(pol),
                "strategy_key": str(sk),
                "exit_ts": ts,
                "day_number": int(dn) if dn is not None and pd.notna(dn) else None,
                "pnl_btc": float(pb),
                "pnl_usd": float(pnl_u),
                "portfolio_value_usd": float(v),
                "cumulative_pnl_usd": float(c),
                "cumulative_return_pct": 100.0 * (float(v) / float(starting_capital_usd) - 1.0),
            })
    return pd.DataFrame(rows)


def capital_daily_series(
    legs: pd.DataFrame,
    *,
    starting_capital_usd: float = 1000.0,
    max_day: int | None = None,
) -> pd.DataFrame:
    """Daily account snapshot per (entry_policy, strategy_key).

    Columns: day_number, starting_value, ending_value, daily_PnL, daily_return_pct,
    cumulative_PnL, cumulative_return_pct.
    """
    trade = capital_trade_series(legs, starting_capital_usd=starting_capital_usd)
    out_cols = [
        "entry_policy", "strategy_key", "day_number",
        "starting_value", "ending_value", "daily_PnL", "daily_return_pct",
        "cumulative_PnL", "cumulative_return_pct",
    ]
    if trade.empty or trade["day_number"].isna().all():
        return pd.DataFrame(columns=out_cols)

    rows: list[dict[str, Any]] = []
    for (pol, sk), g in trade.groupby(["entry_policy", "strategy_key"]):
        g = g.dropna(subset=["day_number"]).sort_values(["day_number", "exit_ts"])
        if g.empty:
            continue
        max_d = int(g["day_number"].max()) if max_day is None else int(max_day)
        # Map day -> last portfolio value that day
        by_day = g.groupby("day_number", sort=True)["pnl_usd"].sum()
        equity = float(starting_capital_usd)
        for day in range(1, max_d + 1):
            start_v = equity
            day_pnl = float(by_day.get(day, 0.0))
            equity = start_v + day_pnl
            rows.append({
                "entry_policy": str(pol),
                "strategy_key": str(sk),
                "day_number": day,
                "starting_value": start_v,
                "ending_value": equity,
                "daily_PnL": day_pnl,
                "daily_return_pct": (100.0 * day_pnl / start_v) if start_v else None,
                "cumulative_PnL": equity - float(starting_capital_usd),
                "cumulative_return_pct": 100.0 * (equity / float(starting_capital_usd) - 1.0),
            })
    return pd.DataFrame(rows)
