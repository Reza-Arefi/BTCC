"""Calendar monthly / yearly period analysis for selector experiments (3y-ready).

Primary arms: T1–T10 + A–F (16). T11 and T12 are excluded from all outputs.
Equity is sequential (no per-month capital reset).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PERIOD_FIXED_LABELS = tuple(f"T{i}" for i in range(1, 11))
PERIOD_SELECTOR_LABELS = ("A", "B", "C", "D", "E", "F")
PERIOD_ARM_LABELS = PERIOD_FIXED_LABELS + PERIOD_SELECTOR_LABELS
EXCLUDED_ARMS = ("T11", "T12")
EXCLUDED_KEYS = ("trail_11", "trail_12")


def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def _label_to_key(arm: str) -> str:
    if arm.startswith("T") and arm[1:].isdigit():
        return f"trail_{int(arm[1:])}"
    return f"selector_{arm.lower()}"


def calendar_month_periods(
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
) -> pd.DataFrame:
    """Build calendar-month windows covering [eval_start, eval_end].

    Partial first/last months are kept and flagged.
    """
    start = _utc(eval_start).normalize()
    end = _utc(eval_end)
    end_day = end.normalize()
    # Work in UTC-naive period space then re-attach tz to avoid Period tz warnings.
    start_naive = start.tz_convert("UTC").tz_localize(None)
    end_naive = end_day.tz_convert("UTC").tz_localize(None)
    cursor = start_naive.to_period("M").to_timestamp(how="start")
    rows = []
    idx = 0
    while cursor <= end_naive:
        idx += 1
        month_start = cursor
        next_month = (cursor + pd.offsets.MonthBegin(1))
        month_end = min(end_naive, (next_month - pd.Timedelta(days=1)))
        full_start = month_start
        full_end = next_month - pd.Timedelta(days=1)
        period_start = max(start_naive, month_start)
        period_end = min(end_naive, month_end)
        n_days = int((period_end - period_start).days) + 1
        is_partial = (period_start > full_start) or (period_end < full_end)
        ps = pd.Timestamp(period_start, tz="UTC")
        pe = pd.Timestamp(period_end, tz="UTC")
        rows.append(
            {
                "month_index": idx,
                "month_id": str(ps.strftime("%Y-%m")),
                "month_start": ps.isoformat(),
                "month_end": pe.isoformat(),
                "n_days": n_days,
                "is_partial": bool(is_partial),
                "partial_label": (
                    "partial_first"
                    if period_start > full_start and idx == 1
                    else ("partial_last" if period_end < full_end else ("partial" if is_partial else "full"))
                ),
            }
        )
        cursor = next_month
    df = pd.DataFrame(rows)
    if not df.empty:
        start_y, start_m = start.year, start.month

        def _year_index(mid: str) -> int:
            y, m = map(int, mid.split("-"))
            months_from_start = (y - start_y) * 12 + (m - start_m)
            return months_from_start // 12 + 1

        df["year_index"] = df["month_id"].map(_year_index)
    return df


def _filter_arms(legs: pd.DataFrame) -> pd.DataFrame:
    if legs.empty:
        return legs
    df = legs.copy()
    if "arm_key" not in df.columns:
        df["arm_key"] = df.get("strategy_key")
    df = df[~df["arm_key"].isin(EXCLUDED_ARMS)].copy()
    df = df[df["arm_key"].isin(PERIOD_ARM_LABELS)].copy()
    return df


def _trade_stats(g: pd.DataFrame) -> dict[str, Any]:
    if g is None or g.empty:
        return {
            "n_trades": 0,
            "n_wins": 0,
            "n_losses": 0,
            "win_rate_pct": None,
            "avg_trade_pct": None,
            "profit_factor": None,
            "largest_win_pct": None,
            "largest_loss_pct": None,
            "fees_btc": 0.0,
            "slippage_btc": 0.0,
            "avg_hold_hours": None,
            "median_hold_hours": None,
        }
    pct = pd.to_numeric(g.get("pnl_pct"), errors="coerce")
    usd = pd.to_numeric(g.get("pnl_usd_equiv"), errors="coerce")
    wins = pct > 0
    losses = pct < 0
    gp = float(usd[wins].sum()) if usd is not None else 0.0
    gl = float(abs(usd[losses].sum())) if usd is not None else 0.0
    hold = pd.to_numeric(g.get("holding_hours"), errors="coerce")
    fees = pd.to_numeric(g.get("fees_btc"), errors="coerce").fillna(0.0).sum() if "fees_btc" in g.columns else 0.0
    slip = (
        pd.to_numeric(g.get("slippage_btc_approx"), errors="coerce").fillna(0.0).sum()
        if "slippage_btc_approx" in g.columns
        else 0.0
    )
    return {
        "n_trades": int(len(g)),
        "n_wins": int(wins.sum()),
        "n_losses": int(losses.sum()),
        "win_rate_pct": float(100.0 * wins.mean()) if len(g) else None,
        "avg_trade_pct": float(100.0 * pct.mean()) if pct.notna().any() else None,
        "profit_factor": float(gp / gl) if gl > 1e-12 else (None if gp <= 0 else float("inf")),
        "largest_win_pct": float(100.0 * pct.max()) if pct.notna().any() else None,
        "largest_loss_pct": float(100.0 * pct.min()) if pct.notna().any() else None,
        "fees_btc": float(fees),
        "slippage_btc": float(slip),
        "avg_hold_hours": float(hold.mean()) if hold.notna().any() else None,
        "median_hold_hours": float(hold.median()) if hold.notna().any() else None,
    }


def _max_dd_pct(equity: pd.Series) -> float:
    if equity is None or len(equity) == 0:
        return 0.0
    peak = equity.cummax()
    dd = 100.0 * (equity / peak - 1.0)
    return float(dd.min()) if len(dd) else 0.0


def build_daily_equity(
    legs: pd.DataFrame,
    *,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    starting_capital_usd: float = 1000.0,
) -> pd.DataFrame:
    """Daily ending equity per arm on the calendar day axis."""
    from btcc.analytics.capital import capital_daily_series

    df = _filter_arms(legs)
    if df.empty:
        return pd.DataFrame()
    if "strategy_key" not in df.columns or (df["strategy_key"] != df["arm_key"]).any():
        df = df.copy()
        df["strategy_key"] = df["arm_key"]
    if "entry_policy" not in df.columns:
        df["entry_policy"] = "COMMON"
    if "pnl_btc" not in df.columns:
        df["pnl_btc"] = 0.0
    # day_number from exit relative to eval_start when missing
    if "day_number" not in df.columns or df["day_number"].isna().all():
        df = df.copy()
        exit_ts = pd.to_datetime(df.get("exit_ts"), utc=True, errors="coerce")
        start = _utc(eval_start).normalize()
        df["day_number"] = ((exit_ts.dt.normalize() - start).dt.days + 1).astype("Int64")
    max_day = int((_utc(eval_end).normalize() - _utc(eval_start).normalize()).days) + 1
    daily = capital_daily_series(df, starting_capital_usd=starting_capital_usd, max_day=max_day)
    if daily.empty:
        return daily
    start = _utc(eval_start).normalize()
    daily = daily.copy()
    daily["date"] = daily["day_number"].map(lambda d: (start + pd.Timedelta(days=int(d) - 1)).normalize())
    daily["month_id"] = daily["date"].dt.strftime("%Y-%m")
    return daily


def monthly_results_table(
    legs: pd.DataFrame,
    *,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    starting_capital_usd: float = 1000.0,
    periods: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-arm × calendar-month metrics with sequential equity compounding."""
    periods = periods if periods is not None else calendar_month_periods(eval_start, eval_end)
    daily = build_daily_equity(
        legs, eval_start=eval_start, eval_end=eval_end, starting_capital_usd=starting_capital_usd
    )
    closed = _filter_arms(legs)
    if "closed" in closed.columns:
        closed = closed[closed["closed"] == True].copy()  # noqa: E712
    if not closed.empty:
        closed["_exit"] = pd.to_datetime(closed.get("exit_ts"), utc=True, errors="coerce")

    rows: list[dict[str, Any]] = []
    for arm in PERIOD_ARM_LABELS:
        d_arm = daily[daily["strategy_key"] == arm].sort_values("day_number") if not daily.empty else pd.DataFrame()
        # One equity point per calendar date (duplicates can appear on short/smoke windows).
        equity_by_date = (
            d_arm.drop_duplicates("date", keep="last")
            .set_index("date")["ending_value"]
            .astype(float)
            if not d_arm.empty
            else pd.Series(dtype=float)
        )
        for _, p in periods.iterrows():
            m_start = _utc(p["month_start"]).normalize()
            m_end = _utc(p["month_end"]).normalize()
            # Starting equity = ending equity of day before month, else start capital / last known
            prev = m_start - pd.Timedelta(days=1)
            if equity_by_date.empty:
                start_eq = float(starting_capital_usd)
                end_eq = start_eq
                max_dd = 0.0
            else:
                if prev in equity_by_date.index:
                    start_eq = float(equity_by_date.loc[prev])
                else:
                    prior = equity_by_date[equity_by_date.index < m_start]
                    start_eq = float(prior.iloc[-1]) if len(prior) else float(starting_capital_usd)
                in_month = equity_by_date[(equity_by_date.index >= m_start) & (equity_by_date.index <= m_end)]
                if len(in_month):
                    end_eq = float(in_month.iloc[-1])
                    max_dd = _max_dd_pct(in_month)
                else:
                    end_eq = start_eq
                    max_dd = 0.0
            monthly_ret = 100.0 * (end_eq / start_eq - 1.0) if start_eq else 0.0
            cum_ret = 100.0 * (end_eq / float(starting_capital_usd) - 1.0)

            if not closed.empty:
                g = closed[
                    (closed["arm_key"] == arm)
                    & (closed["_exit"].notna())
                    & (closed["_exit"].dt.normalize() >= m_start)
                    & (closed["_exit"].dt.normalize() <= m_end)
                ]
            else:
                g = pd.DataFrame()
            stats = _trade_stats(g)
            rows.append(
                {
                    "arm": arm,
                    "arm_kind": "selector" if arm in PERIOD_SELECTOR_LABELS else "fixed",
                    "month_index": int(p["month_index"]),
                    "month_id": p["month_id"],
                    "year_index": int(p["year_index"]),
                    "month_start": p["month_start"],
                    "month_end": p["month_end"],
                    "n_days": int(p["n_days"]),
                    "is_partial": bool(p["is_partial"]),
                    "partial_label": p["partial_label"],
                    "starting_equity_usd": start_eq,
                    "ending_equity_usd": end_eq,
                    "monthly_return_pct": monthly_ret,
                    "cumulative_return_pct": cum_ret,
                    "max_drawdown_pct": max_dd,
                    **stats,
                }
            )
    return pd.DataFrame(rows)


def yearly_results_table(
    monthly: pd.DataFrame,
    *,
    starting_capital_usd: float = 1000.0,
) -> pd.DataFrame:
    """Aggregate sequential months into Year 1/2/3 and Full."""
    if monthly.empty:
        return pd.DataFrame()
    rows = []
    for arm in PERIOD_ARM_LABELS:
        g = monthly[monthly["arm"] == arm].sort_values("month_index")
        if g.empty:
            continue
        # Full
        rows.append(_year_block(g, arm, "FULL", starting_capital_usd=starting_capital_usd))
        for y in sorted(g["year_index"].dropna().unique()):
            gy = g[g["year_index"] == int(y)]
            rows.append(_year_block(gy, arm, f"Y{int(y)}", starting_capital_usd=starting_capital_usd))
    return pd.DataFrame(rows)


def _year_block(g: pd.DataFrame, arm: str, label: str, *, starting_capital_usd: float) -> dict[str, Any]:
    start_eq = float(g.iloc[0]["starting_equity_usd"])
    end_eq = float(g.iloc[-1]["ending_equity_usd"])
    rets = g["monthly_return_pct"].astype(float)
    n_prof = int((rets > 0).sum())
    n_loss = int((rets < 0).sum())
    return {
        "arm": arm,
        "period": label,
        "year_index": None if label == "FULL" else int(label[1:]),
        "n_months": int(len(g)),
        "month_start": g.iloc[0]["month_start"],
        "month_end": g.iloc[-1]["month_end"],
        "starting_equity_usd": start_eq,
        "ending_equity_usd": end_eq,
        "period_return_pct": 100.0 * (end_eq / start_eq - 1.0) if start_eq else 0.0,
        "cumulative_return_pct": 100.0 * (end_eq / float(starting_capital_usd) - 1.0),
        "n_trades": int(g["n_trades"].sum()),
        "n_wins": int(g["n_wins"].sum()),
        "n_losses": int(g["n_losses"].sum()),
        "win_rate_pct": (
            100.0 * float(g["n_wins"].sum()) / float(g["n_trades"].sum()) if g["n_trades"].sum() else None
        ),
        "profit_factor": _combined_pf(g),
        "max_drawdown_pct": float(g["max_drawdown_pct"].min()) if len(g) else 0.0,
        "profitable_months": n_prof,
        "losing_months": n_loss,
        "pct_profitable_months": 100.0 * n_prof / len(g) if len(g) else None,
        "mean_monthly_return_pct": float(rets.mean()),
        "median_monthly_return_pct": float(rets.median()),
        "std_monthly_return_pct": float(rets.std(ddof=0)) if len(g) > 1 else 0.0,
        "best_month_id": str(g.loc[rets.idxmax(), "month_id"]) if len(g) else None,
        "best_month_return_pct": float(rets.max()) if len(g) else None,
        "worst_month_id": str(g.loc[rets.idxmin(), "month_id"]) if len(g) else None,
        "worst_month_return_pct": float(rets.min()) if len(g) else None,
    }


def _combined_pf(g: pd.DataFrame) -> float | None:
    # Reconstruct approx PF from monthly is imperfect; leave None if no trades
    n = int(g["n_trades"].sum())
    if n <= 0:
        return None
    # Weighted by available monthly PFs when present is messy; use win/loss $ if we had them.
    # Fall back: mean of finite monthly PFs.
    pfs = pd.to_numeric(g["profit_factor"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(pfs.mean()) if len(pfs) else None


def monthly_rankings_table(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame()
    rows = []
    for mid, g in monthly.groupby("month_id", sort=True):
        g = g.copy()
        g["rank_return"] = g["monthly_return_pct"].rank(ascending=False, method="min")
        pf = pd.to_numeric(g["profit_factor"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        g["rank_pf"] = pf.rank(ascending=False, method="min", na_option="bottom")
        g["rank_dd"] = g["max_drawdown_pct"].rank(ascending=False, method="min")
        by_ret = g.sort_values("monthly_return_pct", ascending=False)
        worst = by_ret.iloc[-1]
        e_row = g[g["arm"] == "E"]
        t1_row = g[g["arm"] == "T1"]
        best_pf = g.sort_values("rank_pf", na_position="last").iloc[0]["arm"]
        best_dd = g.sort_values("rank_dd", ascending=False).iloc[0]["arm"]
        rows.append(
            {
                "month_id": mid,
                "month_index": int(g["month_index"].iloc[0]),
                "rank1_arm": by_ret.iloc[0]["arm"],
                "rank1_return_pct": float(by_ret.iloc[0]["monthly_return_pct"]),
                "rank2_arm": by_ret.iloc[1]["arm"] if len(by_ret) > 1 else None,
                "rank2_return_pct": float(by_ret.iloc[1]["monthly_return_pct"]) if len(by_ret) > 1 else None,
                "rank3_arm": by_ret.iloc[2]["arm"] if len(by_ret) > 2 else None,
                "rank3_return_pct": float(by_ret.iloc[2]["monthly_return_pct"]) if len(by_ret) > 2 else None,
                "E_rank": int(e_row["rank_return"].iloc[0]) if len(e_row) else None,
                "E_return_pct": float(e_row["monthly_return_pct"].iloc[0]) if len(e_row) else None,
                "T1_rank": int(t1_row["rank_return"].iloc[0]) if len(t1_row) else None,
                "T1_return_pct": float(t1_row["monthly_return_pct"].iloc[0]) if len(t1_row) else None,
                "worst_arm": worst["arm"],
                "worst_return_pct": float(worst["monthly_return_pct"]),
                "best_pf_arm": best_pf,
                "best_dd_arm": best_dd,
            }
        )
    return pd.DataFrame(rows)


def monthly_rankings_long(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame()
    parts = []
    for mid, g in monthly.groupby("month_id", sort=True):
        gg = g.copy()
        gg["rank_return"] = gg["monthly_return_pct"].rank(ascending=False, method="min").astype(int)
        pf = pd.to_numeric(gg["profit_factor"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        gg["rank_pf"] = pf.rank(ascending=False, method="min", na_option="bottom")
        gg["rank_pf"] = gg["rank_pf"].fillna(float(len(gg))).astype(int)
        gg["rank_dd"] = gg["max_drawdown_pct"].rank(ascending=False, method="min").astype(int)
        parts.append(
            gg[
                [
                    "month_id",
                    "month_index",
                    "arm",
                    "monthly_return_pct",
                    "profit_factor",
                    "max_drawdown_pct",
                    "rank_return",
                    "rank_pf",
                    "rank_dd",
                ]
            ]
        )
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def monthly_selector_distribution(selection: pd.DataFrame, periods: pd.DataFrame) -> pd.DataFrame:
    if selection is None or selection.empty or periods.empty:
        return pd.DataFrame()
    sel = selection.copy()
    sel = sel[~sel.get("selected_arm_label", pd.Series(dtype=str)).isin(EXCLUDED_ARMS)]
    if "selected_arm_label" in sel.columns:
        sel = sel[sel["selected_arm_label"].isin(PERIOD_FIXED_LABELS)]
    sel["_ts"] = pd.to_datetime(sel.get("entry_ts"), utc=True, errors="coerce")
    rows = []
    for _, p in periods.iterrows():
        m_start = _utc(p["month_start"]).normalize()
        m_end = _utc(p["month_end"]).normalize() + pd.Timedelta(hours=23, minutes=59)
        block = sel[(sel["_ts"] >= m_start) & (sel["_ts"] <= m_end)]
        for arm in PERIOD_SELECTOR_LABELS:
            ga = block[block["arm_label"] == arm] if "arm_label" in block.columns else pd.DataFrame()
            n = len(ga)
            freq = ga["selected_arm_label"].value_counts(normalize=True) * 100 if n else pd.Series(dtype=float)
            row = {
                "month_id": p["month_id"],
                "month_index": int(p["month_index"]),
                "selector": arm,
                "n_selections": n,
            }
            for t in PERIOD_FIXED_LABELS:
                row[f"pct_{t}"] = float(freq.get(t, 0.0))
            rows.append(row)
    return pd.DataFrame(rows)


def monthly_selector_regret(monthly: pd.DataFrame, selection: pd.DataFrame, periods: pd.DataFrame) -> pd.DataFrame:
    """Per month: best fixed vs each selector return + regret."""
    if monthly.empty:
        return pd.DataFrame()
    fixed = monthly[monthly["arm"].isin(PERIOD_FIXED_LABELS)]
    rows = []
    for mid, g in monthly.groupby("month_id", sort=True):
        gf = fixed[fixed["month_id"] == mid]
        if gf.empty:
            continue
        best = gf.loc[gf["monthly_return_pct"].idxmax()]
        for sel in PERIOD_SELECTOR_LABELS:
            gs = g[g["arm"] == sel]
            if gs.empty:
                continue
            sel_ret = float(gs.iloc[0]["monthly_return_pct"])
            best_ret = float(best["monthly_return_pct"])
            rows.append(
                {
                    "month_id": mid,
                    "month_index": int(gs.iloc[0]["month_index"]),
                    "selector": sel,
                    "selector_return_pct": sel_ret,
                    "best_fixed_arm": best["arm"],
                    "best_fixed_return_pct": best_ret,
                    "regret_pp": best_ret - sel_ret,
                    "beats_all_fixed": bool(sel_ret > best_ret + 1e-12),
                    "below_best_fixed": bool(sel_ret < best_ret - 1e-12),
                }
            )
    return pd.DataFrame(rows)


def consistency_summary(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame()
    rows = []
    for arm in PERIOD_ARM_LABELS:
        g = monthly[monthly["arm"] == arm].sort_values("month_index")
        if g.empty:
            continue
        rets = g["monthly_return_pct"].astype(float).tolist()
        signs = [1 if r > 0 else (-1 if r < 0 else 0) for r in rets]

        def _streak(target: int) -> int:
            best = cur = 0
            for s in signs:
                if s == target:
                    cur += 1
                    best = max(best, cur)
                else:
                    cur = 0
            return best

        n_prof = sum(1 for r in rets if r > 0)
        n_loss = sum(1 for r in rets if r < 0)
        rows.append(
            {
                "arm": arm,
                "n_months": len(rets),
                "profitable_months": n_prof,
                "losing_months": n_loss,
                "pct_profitable_months": 100.0 * n_prof / len(rets) if rets else None,
                "best_month_id": g.iloc[int(np.argmax(rets))]["month_id"],
                "best_month_return_pct": float(max(rets)),
                "worst_month_id": g.iloc[int(np.argmin(rets))]["month_id"],
                "worst_month_return_pct": float(min(rets)),
                "median_monthly_return_pct": float(np.median(rets)),
                "mean_monthly_return_pct": float(np.mean(rets)),
                "std_monthly_return_pct": float(np.std(rets)),
                "longest_win_streak_months": _streak(1),
                "longest_loss_streak_months": _streak(-1),
                "final_cumulative_return_pct": float(g.iloc[-1]["cumulative_return_pct"]),
                "final_equity_usd": float(g.iloc[-1]["ending_equity_usd"]),
            }
        )
    return pd.DataFrame(rows)


def robustness_summary(
    monthly: pd.DataFrame,
    yearly: pd.DataFrame,
    *,
    starting_capital_usd: float = 1000.0,
) -> pd.DataFrame:
    cons = consistency_summary(monthly)
    if cons.empty:
        return cons
    rows = []
    for _, c in cons.iterrows():
        arm = c["arm"]
        ymap = {}
        if not yearly.empty:
            for _, y in yearly[yearly["arm"] == arm].iterrows():
                ymap[y["period"]] = y
        full = ymap.get("FULL")
        rows.append(
            {
                "arm": arm,
                "return_3y_pct": float(full["period_return_pct"]) if full is not None else c["final_cumulative_return_pct"],
                "Y1_return_pct": float(ymap["Y1"]["period_return_pct"]) if "Y1" in ymap else None,
                "Y2_return_pct": float(ymap["Y2"]["period_return_pct"]) if "Y2" in ymap else None,
                "Y3_return_pct": float(ymap["Y3"]["period_return_pct"]) if "Y3" in ymap else None,
                "profitable_months": int(c["profitable_months"]),
                "losing_months": int(c["losing_months"]),
                "pct_profitable_months": c["pct_profitable_months"],
                "max_dd_pct": float(monthly[monthly["arm"] == arm]["max_drawdown_pct"].min()) if len(monthly) else None,
                "avg_monthly_return_pct": c["mean_monthly_return_pct"],
                "median_monthly_return_pct": c["median_monthly_return_pct"],
                "std_monthly_return_pct": c["std_monthly_return_pct"],
                "longest_win_streak": int(c["longest_win_streak_months"]),
                "longest_loss_streak": int(c["longest_loss_streak_months"]),
                "best_month_return_pct": c["best_month_return_pct"],
                "worst_month_return_pct": c["worst_month_return_pct"],
            }
        )
    out = pd.DataFrame(rows)
    out = out.sort_values("return_3y_pct", ascending=False)
    return out


def regime_month_summary(legs: pd.DataFrame, periods: pd.DataFrame) -> pd.DataFrame:
    """Avg trade % by fixed arm × regime within each month (analysis only)."""
    cf = _filter_arms(legs)
    if cf.empty or "regime" not in cf.columns:
        return pd.DataFrame()
    if "is_counterfactual" in cf.columns:
        cf = cf[cf["is_counterfactual"] == True].copy()  # noqa: E712
    cf = cf[cf["arm_key"].isin(PERIOD_FIXED_LABELS)].copy()
    if cf.empty:
        return pd.DataFrame()
    cf["_exit"] = pd.to_datetime(cf.get("exit_ts"), utc=True, errors="coerce")
    rows = []
    for _, p in periods.iterrows():
        m_start = _utc(p["month_start"]).normalize()
        m_end = _utc(p["month_end"]).normalize()
        block = cf[(cf["_exit"].dt.normalize() >= m_start) & (cf["_exit"].dt.normalize() <= m_end)]
        for regime, gr in block.groupby("regime"):
            for arm in PERIOD_FIXED_LABELS:
                ga = gr[gr["arm_key"] == arm]
                if ga.empty:
                    continue
                pct = pd.to_numeric(ga["pnl_pct"], errors="coerce")
                rows.append(
                    {
                        "month_id": p["month_id"],
                        "month_index": int(p["month_index"]),
                        "regime": str(regime),
                        "arm": arm,
                        "n_trades": int(len(ga)),
                        "avg_trade_pct": float(100.0 * pct.mean()) if pct.notna().any() else None,
                    }
                )
    return pd.DataFrame(rows)


def write_period_analysis(
    out_dir: Path,
    *,
    legs: pd.DataFrame,
    selection: pd.DataFrame | None,
    eval_start: str | pd.Timestamp,
    eval_end: str | pd.Timestamp,
    starting_capital_usd: float = 1000.0,
    analytics_root: Path | None = None,
) -> dict[str, Path]:
    """Compute and persist all monthly/yearly artifacts. Does not run a backtest."""
    out_dir = Path(out_dir)
    analytics_root = Path(analytics_root or out_dir / "analytics")
    period_dir = analytics_root / "periods"
    period_dir.mkdir(parents=True, exist_ok=True)

    eval_start_ts = _utc(eval_start)
    eval_end_ts = _utc(eval_end)
    periods = calendar_month_periods(eval_start_ts, eval_end_ts)
    periods.to_csv(period_dir / "month_calendar.csv", index=False)

    monthly = monthly_results_table(
        legs,
        eval_start=eval_start_ts,
        eval_end=eval_end_ts,
        starting_capital_usd=starting_capital_usd,
        periods=periods,
    )
    yearly = yearly_results_table(monthly, starting_capital_usd=starting_capital_usd)
    rankings = monthly_rankings_table(monthly)
    rankings_long = monthly_rankings_long(monthly)
    dist = monthly_selector_distribution(selection if selection is not None else pd.DataFrame(), periods)
    regret = monthly_selector_regret(monthly, selection if selection is not None else pd.DataFrame(), periods)
    cons = consistency_summary(monthly)
    robust = robustness_summary(monthly, yearly, starting_capital_usd=starting_capital_usd)
    regime = regime_month_summary(legs, periods)

    paths: dict[str, Path] = {}
    for name, df in (
        ("monthly_results.csv", monthly),
        ("yearly_results.csv", yearly),
        ("monthly_rankings.csv", rankings),
        ("monthly_rankings_long.csv", rankings_long),
        ("monthly_selector_distribution.csv", dist),
        ("monthly_regret.csv", regret),
        ("monthly_consistency.csv", cons),
        ("robustness_summary.csv", robust),
        ("monthly_regime_summary.csv", regime),
    ):
        p = period_dir / name
        df.to_csv(p, index=False)
        paths[name] = p

    manifest = {
        "analysis": "selector_period_analysis_v1",
        "arms": list(PERIOD_ARM_LABELS),
        "excluded_arms": list(EXCLUDED_ARMS),
        "eval_start": str(eval_start_ts),
        "eval_end": str(eval_end_ts),
        "n_months": int(len(periods)),
        "n_years": int(periods["year_index"].max()) if len(periods) else 0,
        "starting_capital_usd": starting_capital_usd,
        "compounding": "sequential_calendar_months",
        "files": {k: str(v) for k, v in paths.items()},
    }
    mp = period_dir / "period_analysis_manifest.json"
    mp.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    paths["period_analysis_manifest.json"] = mp
    return paths
