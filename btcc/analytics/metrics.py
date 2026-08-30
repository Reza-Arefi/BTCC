"""Shared metric definitions for backtest and live (no lookahead).

Temporal rules:
- 4h outcomes only after maturity (pred_ts + 4h <= asof)
- weight history uses effective/update timestamps as recorded
- BTC.D / market features already as-of in source tables
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from btcc.sim.maturity import filter_matured_for_learning
from btcc.sim.score import FACTOR_KEYS

DEFAULT_SCORE_BINS = [0.0, 0.20, 0.40, 0.60, 0.70, 0.80, 1.01]
STRATEGY_KEYS = ("strategy_1", "strategy_2", "strategy_3")


def _utc_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")


def load_arm_tables(arm_dir) -> dict[str, pd.DataFrame]:
    from pathlib import Path

    arm_dir = Path(arm_dir)
    out: dict[str, pd.DataFrame] = {}
    for name, file in (
        ("predictions", "predictions.csv"),
        ("opportunities", "opportunities.csv"),
        ("legs", "strategy_legs.csv"),
        ("weight_history", "weight_history.csv"),
    ):
        p = arm_dir / file
        if p.exists() and p.stat().st_size > 0:
            try:
                out[name] = pd.read_csv(p, low_memory=False)
            except pd.errors.EmptyDataError:
                out[name] = pd.DataFrame()
        else:
            out[name] = pd.DataFrame()
    return out


def matured_predictions(
    pred: pd.DataFrame,
    *,
    asof_ts=None,
    horizon_hours: int = 4,
) -> pd.DataFrame:
    if pred.empty:
        return pred
    if asof_ts is None:
        # End-of-sample: keep rows that already have a realized label
        if "future_return_4h" not in pred.columns:
            return pred.iloc[0:0].copy()
        return pred[pred["future_return_4h"].notna()].copy()
    return filter_matured_for_learning(pred, asof_ts=asof_ts, horizon_hours=horizon_hours)


def cumulative_btc(legs: pd.DataFrame) -> pd.DataFrame:
    """Per-strategy cumulative BTC PnL through time (exit order)."""
    rows = []
    if legs.empty or "strategy_key" not in legs.columns:
        return pd.DataFrame(columns=["strategy_key", "exit_ts", "pnl_btc", "cum_btc"])
    df = legs.copy()
    df["exit_ts"] = _utc_series(df.get("exit_ts", pd.Series(dtype=str)))
    df = df.dropna(subset=["exit_ts"])
    for sk, g in df.groupby("strategy_key"):
        g = g.sort_values("exit_ts")
        pnl = pd.to_numeric(g.get("pnl_btc"), errors="coerce").fillna(0.0)
        cum = pnl.cumsum()
        for ts, p, c in zip(g["exit_ts"], pnl, cum):
            rows.append({"strategy_key": sk, "exit_ts": ts, "pnl_btc": float(p), "cum_btc": float(c)})
    return pd.DataFrame(rows)


def rolling_win_rate(legs: pd.DataFrame, window_days: int = 30) -> pd.DataFrame:
    rows = []
    if legs.empty or "strategy_key" not in legs.columns:
        return pd.DataFrame(columns=["strategy_key", "exit_ts", "rolling_win_rate", "cum_win_rate"])
    df = legs.copy()
    df["exit_ts"] = _utc_series(df.get("exit_ts"))
    df["win"] = (pd.to_numeric(df.get("pnl_btc"), errors="coerce") > 0).astype(float)
    df = df.dropna(subset=["exit_ts"])
    for sk, g in df.groupby("strategy_key"):
        g = g.sort_values("exit_ts").set_index("exit_ts")
        roll = g["win"].rolling(f"{int(window_days)}D").mean()
        cum = g["win"].expanding().mean()
        for ts, rw, cw in zip(roll.index, roll.values, cum.values):
            rows.append({
                "strategy_key": sk,
                "exit_ts": ts,
                "rolling_win_rate": float(rw) if pd.notna(rw) else None,
                "cum_win_rate": float(cw) if pd.notna(cw) else None,
            })
    return pd.DataFrame(rows)


def weight_timeseries(wh: pd.DataFrame) -> pd.DataFrame:
    if wh.empty:
        return pd.DataFrame(columns=["update_timestamp", "indicator", "new_weight", "update_id", "phase"])
    df = wh.copy()
    if "update_timestamp" in df.columns:
        df["update_timestamp"] = _utc_series(df["update_timestamp"])
    keep = [c for c in ("update_timestamp", "indicator", "new_weight", "old_weight", "update_id", "phase", "update_status") if c in df.columns]
    return df[keep].sort_values("update_timestamp") if "update_timestamp" in df.columns else df


def indicator_predictive_correlation(
    pred: pd.DataFrame,
    *,
    asof_ts=None,
    horizon_hours: int = 4,
    min_obs: int = 30,
) -> pd.DataFrame:
    """Rolling / full-sample corr(signed_i or factor_i, future_return_4h) on matured rows only."""
    m = matured_predictions(pred, asof_ts=asof_ts, horizon_hours=horizon_hours)
    rows = []
    if m.empty or "future_return_4h" not in m.columns:
        return pd.DataFrame(columns=["indicator", "n", "correlation", "asof"])
    y = pd.to_numeric(m["future_return_4h"], errors="coerce")
    asof = str(asof_ts) if asof_ts is not None else "end_of_sample"
    for k in FACTOR_KEYS:
        col = f"signed_{k}" if f"signed_{k}" in m.columns else f"factor_{k}"
        if col not in m.columns:
            continue
        x = pd.to_numeric(m[col], errors="coerce")
        mask = x.notna() & y.notna()
        n = int(mask.sum())
        if n < min_obs:
            corr = None
        else:
            corr = float(x[mask].corr(y[mask]))
        rows.append({"indicator": k, "n": n, "correlation": corr, "asof": asof, "signal_col": col})
    return pd.DataFrame(rows)


def rolling_indicator_correlation(
    pred: pd.DataFrame,
    *,
    horizon_hours: int = 4,
    window_days: int = 30,
    step_days: int = 7,
    min_obs: int = 30,
) -> pd.DataFrame:
    """Through-time correlation using only matured outcomes as of each grid date."""
    if pred.empty or "timestamp" not in pred.columns:
        return pd.DataFrame()
    ts = _utc_series(pred["timestamp"])
    t0, t1 = ts.min(), ts.max()
    if pd.isna(t0) or pd.isna(t1):
        return pd.DataFrame()
    rows = []
    grid = pd.date_range(t0 + pd.Timedelta(days=window_days), t1, freq=f"{step_days}D", tz="UTC")
    for asof in grid:
        matured = matured_predictions(pred, asof_ts=asof, horizon_hours=horizon_hours)
        if matured.empty:
            continue
        matured = matured.copy()
        matured["_ts"] = _utc_series(matured["timestamp"])
        window = matured[matured["_ts"] >= asof - pd.Timedelta(days=window_days)]
        part = indicator_predictive_correlation(window, asof_ts=asof, horizon_hours=horizon_hours, min_obs=min_obs)
        if not part.empty:
            part["window_days"] = window_days
            rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def weight_vs_usefulness(wh: pd.DataFrame, corr_ts: pd.DataFrame) -> pd.DataFrame:
    """Align latest weight per update with nearest matured correlation snapshot."""
    if wh.empty or corr_ts.empty:
        return pd.DataFrame()
    w = weight_timeseries(wh)
    c = corr_ts.copy()
    c["asof"] = _utc_series(c["asof"])
    rows = []
    for _, wr in w.iterrows():
        ind = wr.get("indicator")
        wt = wr.get("update_timestamp")
        if pd.isna(wt):
            continue
        sub = c[(c["indicator"] == ind) & (c["asof"] <= wt)]
        if sub.empty:
            continue
        nearest = sub.iloc[-1]
        rows.append({
            "timestamp": wt,
            "indicator": ind,
            "weight": float(wr.get("new_weight")) if pd.notna(wr.get("new_weight")) else None,
            "correlation": nearest.get("correlation"),
            "corr_asof": nearest.get("asof"),
            "update_id": wr.get("update_id"),
        })
    return pd.DataFrame(rows)


def score_vs_return(pred: pd.DataFrame, *, asof_ts=None) -> pd.DataFrame:
    m = matured_predictions(pred, asof_ts=asof_ts)
    if m.empty:
        return pd.DataFrame(columns=["S", "future_return_4h", "timestamp", "symbol"])
    out = m[["timestamp", "symbol"]].copy() if "symbol" in m.columns else m[["timestamp"]].copy()
    out["S"] = pd.to_numeric(m.get("S"), errors="coerce")
    out["future_return_4h"] = pd.to_numeric(m.get("future_return_4h"), errors="coerce")
    return out.dropna(subset=["S", "future_return_4h"])


def accuracy_by_score_bucket(
    pred: pd.DataFrame,
    bins: list[float] | None = None,
    *,
    asof_ts=None,
) -> pd.DataFrame:
    bins = bins or DEFAULT_SCORE_BINS
    m = score_vs_return(pred, asof_ts=asof_ts)
    if m.empty:
        return pd.DataFrame()
    # Map S in [-1,1] to [0,1] confidence-like for bucketing of |positive| scores:
    # Use raw S mapped via (S+1)/2 for full range; also report sign hit when S>=0 vs return>0
    m = m.copy()
    m["score_01"] = (m["S"] + 1.0) / 2.0
    m["hit"] = ((m["S"] >= 0) == (m["future_return_4h"] > 0)).astype(float)
    m["bucket"] = pd.cut(m["score_01"], bins=bins, right=False, include_lowest=True)
    rows = []
    for b, g in m.groupby("bucket", observed=False):
        rows.append({
            "bucket": str(b),
            "n": int(len(g)),
            "accuracy": float(g["hit"].mean()) if len(g) else None,
            "mean_S": float(g["S"].mean()) if len(g) else None,
            "mean_future_return_4h": float(g["future_return_4h"].mean()) if len(g) else None,
        })
    return pd.DataFrame(rows)


def win_loss_summary(legs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if legs.empty or "strategy_key" not in legs.columns:
        return pd.DataFrame()
    for sk, g in legs.groupby("strategy_key"):
        pnl = pd.to_numeric(g.get("pnl_btc"), errors="coerce").dropna()
        wins = pnl[pnl > 0]
        losses = pnl[pnl <= 0]
        gp, gl = float(wins.sum()) if len(wins) else 0.0, float((-losses).sum()) if len(losses) else 0.0
        rows.append({
            "strategy_key": sk,
            "winning_trades": int(len(wins)),
            "losing_trades": int(len(losses)),
            "avg_winner_btc": float(wins.mean()) if len(wins) else None,
            "avg_loser_btc": float(losses.mean()) if len(losses) else None,
            "total_winning_pnl_btc": gp,
            "total_losing_pnl_btc": -gl,
            "profit_factor": (gp / gl) if gl > 0 else None,
            "net_btc_pnl": float(pnl.sum()) if len(pnl) else None,
        })
    return pd.DataFrame(rows)


def drawdown_series(legs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if legs.empty or "strategy_key" not in legs.columns:
        return pd.DataFrame()
    df = legs.copy()
    df["exit_ts"] = _utc_series(df.get("exit_ts"))
    df = df.dropna(subset=["exit_ts"])
    for sk, g in df.groupby("strategy_key"):
        g = g.sort_values("exit_ts")
        pnl = pd.to_numeric(g.get("pnl_btc"), errors="coerce").fillna(0.0)
        equity = pnl.cumsum()
        peak = equity.cummax()
        dd = equity - peak
        for ts, eq, pk, d in zip(g["exit_ts"], equity, peak, dd):
            rows.append({
                "strategy_key": sk,
                "exit_ts": ts,
                "equity_btc": float(eq),
                "peak_btc": float(pk),
                "drawdown_btc": float(d),
            })
    return pd.DataFrame(rows)


def drawdown_stats(dd: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if dd.empty:
        return pd.DataFrame()
    for sk, g in dd.groupby("strategy_key"):
        g = g.sort_values("exit_ts")
        max_dd = float(g["drawdown_btc"].min()) if len(g) else None
        current_dd = float(g["drawdown_btc"].iloc[-1]) if len(g) else None
        # crude recovery: bars from max_dd trough until dd returns to 0
        recovery = None
        if max_dd is not None and max_dd < 0:
            idx = g["drawdown_btc"].idxmin()
            after = g.loc[idx:]
            recovered = after[after["drawdown_btc"] >= -1e-15]
            if len(recovered) > 1:
                t0 = g.loc[idx, "exit_ts"]
                t1 = recovered.iloc[1]["exit_ts"] if len(recovered) > 1 else recovered.iloc[0]["exit_ts"]
                recovery = (pd.Timestamp(t1) - pd.Timestamp(t0)).total_seconds() / 3600.0
        rows.append({
            "strategy_key": sk,
            "max_drawdown_btc": max_dd,
            "current_drawdown_btc": current_dd,
            "recovery_hours": recovery,
        })
    return pd.DataFrame(rows)


def simultaneous_open_series(
    opp: pd.DataFrame,
    legs: pd.DataFrame,
    *,
    max_open: int = 10,
) -> pd.DataFrame:
    """Rebuild open-count timeline from opportunity open + leg close times."""
    if opp.empty:
        return pd.DataFrame(columns=["timestamp", "n_open", "at_capacity"])
    opens = _utc_series(opp.get("opened_ts") if "opened_ts" in opp.columns else opp.get("signal_timestamp"))
    # close time = max exit among legs for that opportunity, else opened (still open at end → use last exit overall)
    close_map: dict[str, pd.Timestamp] = {}
    if not legs.empty and "opportunity_id" in legs.columns:
        tmp = legs.copy()
        tmp["exit_ts"] = _utc_series(tmp.get("exit_ts"))
        for oid, g in tmp.groupby("opportunity_id"):
            if g["exit_ts"].notna().any():
                close_map[str(oid)] = g["exit_ts"].max()
    events = []
    for i, row in opp.iterrows():
        oid = str(row.get("opportunity_id"))
        ot = opens.iloc[i] if i < len(opens) else pd.NaT
        if pd.isna(ot):
            continue
        events.append((ot, +1))
        ct = close_map.get(oid)
        if ct is not None and pd.notna(ct):
            events.append((ct, -1))
    if not events:
        return pd.DataFrame()
    events.sort(key=lambda x: x[0])
    n = 0
    rows = []
    for ts, delta in events:
        n = max(0, n + delta)
        rows.append({"timestamp": ts, "n_open": n, "at_capacity": int(n >= max_open), "max_open_limit": max_open})
    return pd.DataFrame(rows)


def simultaneous_stats(sim_df: pd.DataFrame, pred: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {
        "avg_simultaneous": None,
        "max_simultaneous": None,
        "pct_time_at_capacity": None,
        "n_rejected_max_open_trades": 0,
    }
    if not sim_df.empty:
        out["avg_simultaneous"] = float(sim_df["n_open"].mean())
        out["max_simultaneous"] = int(sim_df["n_open"].max())
        out["pct_time_at_capacity"] = float(sim_df["at_capacity"].mean()) if "at_capacity" in sim_df.columns else None
    if not pred.empty and "rejection_reason" in pred.columns:
        out["n_rejected_max_open_trades"] = int((pred["rejection_reason"] == "MAX_OPEN_TRADES").sum())
    return out


def rejection_breakdown(pred: pd.DataFrame) -> pd.DataFrame:
    if pred.empty or "rejection_reason" not in pred.columns:
        return pd.DataFrame(columns=["rejection_reason", "count"])
    vc = pred["rejection_reason"].fillna("NONE").value_counts()
    return vc.rename_axis("rejection_reason").reset_index(name="count")


def learning_progress(
    pred: pd.DataFrame,
    legs: pd.DataFrame,
    updates: list[dict] | None,
) -> pd.DataFrame:
    """Performance metrics vs adaptive update number (post-effective periods)."""
    if not updates:
        return pd.DataFrame()
    rows = []
    pred = pred.copy()
    if "timestamp" in pred.columns:
        pred["_ts"] = _utc_series(pred["timestamp"])
    for u in sorted(updates, key=lambda x: int(x.get("update_number") or 0)):
        n = int(u.get("update_number") or 0)
        eff = u.get("effective_from")
        if not eff:
            continue
        eff_ts = pd.Timestamp(eff)
        if eff_ts.tzinfo is None:
            eff_ts = eff_ts.tz_localize("UTC")
        # Next update effective bound
        nxt = None
        for u2 in updates:
            e2 = u2.get("effective_from")
            if not e2:
                continue
            t2 = pd.Timestamp(e2)
            if t2.tzinfo is None:
                t2 = t2.tz_localize("UTC")
            if t2 > eff_ts and (nxt is None or t2 < nxt):
                nxt = t2
        sub = pred[pred["_ts"] >= eff_ts]
        if nxt is not None:
            sub = sub[sub["_ts"] < nxt]
        matured = matured_predictions(sub)
        acc = None
        avg_ret = None
        if not matured.empty and "future_return_4h" in matured.columns:
            y = pd.to_numeric(matured["future_return_4h"], errors="coerce")
            S = pd.to_numeric(matured.get("S"), errors="coerce")
            mask = y.notna() & S.notna()
            if mask.any():
                acc = float(((S[mask] >= 0) == (y[mask] > 0)).mean())
                avg_ret = float(y[mask].mean())
        # trades with entry after eff
        avg_trade = None
        win_rate = None
        if not legs.empty and "entry_ts" in legs.columns:
            lg = legs.copy()
            lg["_ets"] = _utc_series(lg["entry_ts"])
            lg = lg[lg["_ets"] >= eff_ts]
            if nxt is not None:
                lg = lg[lg["_ets"] < nxt]
            pnl = pd.to_numeric(lg.get("pnl_btc"), errors="coerce").dropna()
            if len(pnl):
                avg_trade = float(pnl.mean())
                win_rate = float((pnl > 0).mean())
        rows.append({
            "update_number": n,
            "effective_from": str(eff_ts),
            "prediction_accuracy": acc,
            "mean_future_return_4h": avg_ret,
            "mean_trade_pnl_btc": avg_trade,
            "win_rate": win_rate,
            "n_samples": u.get("n_samples"),
            "status": u.get("status"),
        })
    return pd.DataFrame(rows)


def monthly_performance(pred: pd.DataFrame, legs: pd.DataFrame, opp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if pred.empty:
        return pd.DataFrame()
    p = pred.copy()
    p["_ts"] = _utc_series(p["timestamp"])
    p["month"] = p["_ts"].dt.to_period("M").astype(str)
    for month, g in p.groupby("month"):
        signals = int(g["signal_generated"].sum()) if "signal_generated" in g.columns else 0
        opens = int(g["trade_opened"].sum()) if "trade_opened" in g.columns else 0
        rejected = int(g["rejection_reason"].notna().sum()) if "rejection_reason" in g.columns else 0
        # legs in month
        wr = None
        pnl = None
        avg_trade = None
        dd = None
        if not legs.empty:
            lg = legs.copy()
            lg["_ts"] = _utc_series(lg.get("exit_ts"))
            lg = lg[lg["_ts"].dt.to_period("M").astype(str) == month]
            pn = pd.to_numeric(lg.get("pnl_btc"), errors="coerce").dropna()
            if len(pn):
                wr = float((pn > 0).mean())
                pnl = float(pn.sum())
                avg_trade = float(pn.mean())
                eq = pn.cumsum()
                dd = float((eq - eq.cummax()).min())
        rows.append({
            "month": month,
            "signals": signals,
            "opportunities": opens,
            "rejected_signals": rejected,
            "win_rate": wr,
            "btc_pnl": pnl,
            "avg_trade_btc": avg_trade,
            "max_drawdown_btc": dd,
        })
    return pd.DataFrame(rows)


def btcd_regime_analysis(pred: pd.DataFrame, *, asof_ts=None) -> pd.DataFrame:
    m = matured_predictions(pred, asof_ts=asof_ts)
    if m.empty or "btc_dominance" not in m.columns:
        return pd.DataFrame()
    m = m.copy()
    m["btc_dominance"] = pd.to_numeric(m["btc_dominance"], errors="coerce")
    m = m.dropna(subset=["btc_dominance"])
    # Tertile regimes of relative proxy
    try:
        m["regime"] = pd.qcut(m["btc_dominance"], 3, labels=["low_btc_d", "mid_btc_d", "high_btc_d"])
    except ValueError:
        m["regime"] = "all"
    rows = []
    for reg, g in m.groupby("regime", observed=False):
        y = pd.to_numeric(g.get("future_return_4h"), errors="coerce")
        S = pd.to_numeric(g.get("S"), errors="coerce")
        mask = y.notna() & S.notna()
        rows.append({
            "regime": str(reg),
            "n": int(len(g)),
            "mean_btc_d": float(g["btc_dominance"].mean()),
            "signal_rate": float(g["signal_generated"].mean()) if "signal_generated" in g.columns else None,
            "prediction_accuracy": float(((S[mask] >= 0) == (y[mask] > 0)).mean()) if mask.any() else None,
            "mean_future_return_4h": float(y[mask].mean()) if mask.any() else None,
        })
    return pd.DataFrame(rows)


def adaptive_advantage_series(
    cum_by_arm: dict[str, pd.DataFrame],
    strategy_key: str,
) -> pd.DataFrame:
    """Adaptive cum BTC − Static / Equal through time (merged on exit timeline)."""
    ada = cum_by_arm.get("adaptive")
    if ada is None or ada.empty:
        return pd.DataFrame()
    a = ada[ada["strategy_key"] == strategy_key].sort_values("exit_ts").copy()
    if a.empty:
        return pd.DataFrame()
    out = a[["exit_ts", "cum_btc"]].rename(columns={"cum_btc": "adaptive_cum_btc"})
    for arm, col in (("static", "vs_static"), ("equal", "vs_equal")):
        other = cum_by_arm.get(arm)
        if other is None or other.empty:
            out[col] = None
            continue
        o = other[other["strategy_key"] == strategy_key][["exit_ts", "cum_btc"]].sort_values("exit_ts")
        merged = pd.merge_asof(out.sort_values("exit_ts"), o.rename(columns={"cum_btc": f"{arm}_cum"}), on="exit_ts")
        out = merged
        out[col] = out["adaptive_cum_btc"] - out[f"{arm}_cum"]
    out["strategy_key"] = strategy_key
    return out
