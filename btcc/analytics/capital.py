"""Independent $1,000 starting-capital equity curves (per policy × strategy).

Trade sizing still uses ``notional_usd`` (default $100) per opportunity.
This module tracks a separate standardized portfolio starting at
``starting_capital_usd`` (default $1000) that compounds from closed-trade PnL.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def _pnl_usd(legs: pd.DataFrame) -> pd.Series:
    """Realized USD P/L — prefer stored pnl_usd_equiv (written at exit)."""
    if "pnl_usd_equiv" in legs.columns:
        u = pd.to_numeric(legs["pnl_usd_equiv"], errors="coerce")
        if u.notna().any():
            return u.fillna(0.0)
    pnl_btc = pd.to_numeric(legs.get("pnl_btc"), errors="coerce").fillna(0.0)
    px = None
    for col in ("entry_btc_usdt", "exit_btc_usdt", "btc_usdt"):
        if col in legs.columns:
            px = pd.to_numeric(legs[col], errors="coerce")
            break
    if px is None:
        return pd.Series(0.0, index=legs.index)
    return pnl_btc * px.fillna(0.0)


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


def capital_drawdown_series(
    daily: pd.DataFrame,
    *,
    starting_capital_usd: float = 1000.0,
) -> pd.DataFrame:
    """Portfolio drawdown % from running peak of ending_value.

    drawdown_pct[t] = 100 * (capital[t] - peak[t]) / peak[t]
    """
    cols = [
        "entry_policy", "strategy_key", "day_number",
        "ending_value", "running_peak", "drawdown_pct",
    ]
    if daily is None or daily.empty:
        return pd.DataFrame(columns=cols)
    rows: list[dict[str, Any]] = []
    for (pol, sk), g in daily.groupby(["entry_policy", "strategy_key"], dropna=False):
        g = g.sort_values("day_number")
        vals = list(g["ending_value"].astype(float))
        days = list(g["day_number"].astype(int))
        if not days or days[0] > 1:
            days = [1] + days
            vals = [float(starting_capital_usd)] + vals
        peak = float(vals[0])
        seen: set[int] = set()
        for d, v in zip(days, vals):
            if d in seen:
                continue
            seen.add(d)
            peak = max(peak, float(v))
            dd = 100.0 * (float(v) - peak) / peak if peak else 0.0
            rows.append({
                "entry_policy": str(pol),
                "strategy_key": str(sk),
                "day_number": int(d),
                "ending_value": float(v),
                "running_peak": float(peak),
                "drawdown_pct": float(dd),
            })
    return pd.DataFrame(rows)


def btc_equity_daily(
    legs: pd.DataFrame,
    *,
    max_day: int | None = None,
    starting_capital_usd: float = 1000.0,
    btc_price_by_day: pd.Series | None = None,
) -> pd.DataFrame:
    """Daily BTC metrics per account.

    Distinguishes:
      - cumulative_btc_pnl: BTC gained/lost from trading (unchanged accounting)
      - total_btc_equiv: USD portfolio value / BTC price (account value in BTC terms)
    Plot-only derived columns (do not affect accounting):
      - cumulative_btc_pnl_pct: trading BTC PnL as % of initial BTC stake
      - total_btc_equiv_return_pct: BTC-equiv cumulative return from each strategy's start (0%)
    """
    cols = [
        "entry_policy", "strategy_key", "day_number",
        "daily_pnl_btc", "cumulative_btc_pnl",
        "ending_value_usd", "btc_price", "total_btc_equiv",
        "cumulative_btc_pnl_pct", "total_btc_equiv_return_pct",
    ]
    if legs is None or legs.empty or "strategy_key" not in legs.columns:
        return pd.DataFrame(columns=cols)
    df = legs.copy()
    if "closed" in df.columns:
        df = df[df["closed"] == True]  # noqa: E712
    if "entry_policy" not in df.columns:
        df["entry_policy"] = "ALL"
    if "day_number" not in df.columns:
        return pd.DataFrame(columns=cols)
    df["pnl_btc"] = pd.to_numeric(df.get("pnl_btc"), errors="coerce").fillna(0.0)
    cap = capital_daily_series(
        legs, starting_capital_usd=starting_capital_usd, max_day=max_day,
    )
    price_map: dict[int, float] = {}
    if btc_price_by_day is not None and len(btc_price_by_day):
        for d, px in btc_price_by_day.items():
            try:
                if pd.notna(px) and float(px) > 0:
                    price_map[int(d)] = float(px)
            except (TypeError, ValueError):
                continue
    rows: list[dict[str, Any]] = []
    for (pol, sk), g in df.groupby(["entry_policy", "strategy_key"], dropna=False):
        g = g.dropna(subset=["day_number"]).sort_values("day_number")
        if g.empty:
            continue
        by_day = g.groupby("day_number")["pnl_btc"].sum()
        max_d = int(g["day_number"].max()) if max_day is None else int(max_day)
        cum = 0.0
        cap_g = (
            cap[(cap["entry_policy"] == pol) & (cap["strategy_key"] == sk)]
            if not cap.empty else pd.DataFrame()
        )
        cap_by_day = (
            cap_g.set_index("day_number")["ending_value"]
            if not cap_g.empty else pd.Series(dtype=float)
        )
        # Initial BTC stake for % normalization (per strategy account)
        first_px = next((price_map[d] for d in range(1, max_d + 1) if d in price_map), None)
        init_btc_stake = (
            float(starting_capital_usd) / float(first_px)
            if first_px and first_px > 0 else None
        )
        init_total_eq: float | None = None
        for day in range(1, max_d + 1):
            day_pnl = float(by_day.get(day, 0.0))
            cum += day_pnl
            usd = float(cap_by_day.get(day, starting_capital_usd)) if len(cap_by_day) else float(starting_capital_usd)
            px = price_map.get(int(day))
            total_eq = (usd / px) if px else None
            if init_total_eq is None and total_eq is not None:
                init_total_eq = float(total_eq)
            pnl_pct = (
                100.0 * cum / init_btc_stake
                if init_btc_stake else None
            )
            ret_pct = (
                100.0 * (float(total_eq) / init_total_eq - 1.0)
                if (total_eq is not None and init_total_eq) else None
            )
            rows.append({
                "entry_policy": str(pol),
                "strategy_key": str(sk),
                "day_number": day,
                "daily_pnl_btc": day_pnl,
                "cumulative_btc_pnl": cum,
                "ending_value_usd": usd,
                "btc_price": px,
                "total_btc_equiv": total_eq,
                "cumulative_btc_pnl_pct": pnl_pct,
                "total_btc_equiv_return_pct": ret_pct,
            })
    return pd.DataFrame(rows)


def daily_btc_price_map(pred: pd.DataFrame) -> pd.Series:
    """Mean BTC/USDT price per day_number from predictions (for BTC-equivalent conversion)."""
    if pred is None or pred.empty:
        return pd.Series(dtype=float)
    if "day_number" not in pred.columns or "btc_price" not in pred.columns:
        return pd.Series(dtype=float)
    g = pred.copy()
    g["btc_price"] = pd.to_numeric(g["btc_price"], errors="coerce")
    g = g.dropna(subset=["day_number", "btc_price"])
    g = g[g["btc_price"] > 0]
    if g.empty:
        return pd.Series(dtype=float)
    return g.groupby("day_number")["btc_price"].mean()
