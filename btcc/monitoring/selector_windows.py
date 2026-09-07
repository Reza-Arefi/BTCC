"""Rolling analysis windows and E concentration diagnostics."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from btcc.sim.selector_live_config import LIVE_ARM_LABELS


def _as_utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def selection_entropy(freq: pd.Series) -> float:
    p = freq[freq > 0].astype(float)
    if p.empty:
        return 0.0
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


def selection_concentration_report(selection: pd.DataFrame, *, selector: str = "E") -> dict[str, Any]:
    g = selection[selection.get("arm_label") == selector] if "arm_label" in selection.columns else selection
    if g.empty:
        return {"selector": selector, "n_selections": 0}
    freq = g["selected_arm_label"].value_counts(normalize=True).sort_index()
    return {
        "selector": selector,
        "n_selections": int(len(g)),
        "n_unique_strategies": int(g["selected_arm_label"].nunique()),
        "frequency_pct": {str(k): round(100 * float(v), 2) for k, v in freq.items()},
        "selection_entropy": round(selection_entropy(freq), 4),
        "top_strategy": str(freq.index[0]) if len(freq) else None,
        "top_strategy_pct": round(100 * float(freq.iloc[0]), 2) if len(freq) else 0.0,
        "diagnosis": (
            f"{selector} ≈ {freq.index[0]} ({100*freq.iloc[0]:.1f}%)"
            if len(freq) and freq.iloc[0] > 0.85
            else f"{selector} diversified across {g['selected_arm_label'].nunique()} strategies"
        ),
    }


def filter_legs_window(
    legs: pd.DataFrame,
    *,
    window_id: str,
    window_cfg: dict[str, Any],
    now: datetime | None = None,
) -> pd.DataFrame:
    if legs.empty:
        return legs
    df = legs.copy()
    df["exit_ts"] = pd.to_datetime(df.get("exit_ts"), utc=True, errors="coerce")
    df = df.dropna(subset=["exit_ts"])
    now = now or datetime.now(timezone.utc)
    kind = window_cfg.get("kind")
    if kind == "calendar_yesterday":
        today = now.date()
        y0 = datetime.combine(today - timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
        y1 = datetime.combine(today, datetime.min.time(), tzinfo=timezone.utc)
        return df[(df["exit_ts"] >= y0) & (df["exit_ts"] < y1)]
    hours = float(window_cfg.get("hours", 24))
    cutoff = now - timedelta(hours=hours)
    return df[df["exit_ts"] >= cutoff]


def compute_arm_metrics(
    legs: pd.DataFrame,
    *,
    arm_key: str,
    starting_capital: float = 1000.0,
) -> dict[str, Any]:
    g = legs[legs.get("arm_key") == arm_key].copy() if "arm_key" in legs.columns else pd.DataFrame()
    if g.empty:
        return {"arm": arm_key, "n_trades": 0}
    pnl_pct = pd.to_numeric(g.get("pnl_pct"), errors="coerce").fillna(0.0)
    pnl_usd = pd.to_numeric(g.get("pnl_usd_equiv"), errors="coerce").fillna(0.0)
    wins = int((pnl_pct > 0).sum())
    n = len(g)
    gp = float(pnl_usd[pnl_usd > 0].sum())
    gl = float(abs(pnl_usd[pnl_usd < 0].sum()))
    eq = starting_capital + pnl_usd.cumsum()
    peak = eq.cummax()
    dd = 100 * (eq / peak - 1.0)
    return {
        "arm": arm_key,
        "n_trades": n,
        "cum_return_pct": round(100 * (starting_capital + pnl_usd.sum()) / starting_capital - 100, 2),
        "avg_trade_pct": round(100 * float(pnl_pct.mean()), 4),
        "median_trade_pct": round(100 * float(pnl_pct.median()), 4),
        "win_rate_pct": round(100 * wins / n, 2),
        "profit_factor": round(gp / gl, 2) if gl > 1e-9 else None,
        "max_dd_pct": round(float(dd.min()), 2),
        "gross_profit_usd": round(gp, 2),
        "gross_loss_usd": round(-gl, 2),
    }


def compute_regret_metrics(
    sel_legs: pd.DataFrame,
    cf_legs: pd.DataFrame,
    selection: pd.DataFrame,
    *,
    selector: str = "E",
) -> dict[str, Any]:
    if sel_legs.empty or cf_legs.empty or selection.empty:
        return {}
    cf = cf_legs.copy()
    cf["pnl_pct"] = pd.to_numeric(cf["pnl_pct"], errors="coerce")
    best = cf.groupby("opportunity_id")["pnl_pct"].max()
    s = selection[selection.get("arm_label") == selector]
    regrets = []
    for _, row in s.iterrows():
        oid = row["opportunity_id"]
        leg = sel_legs[(sel_legs["opportunity_id"] == oid) & (sel_legs.get("arm_key") == selector)]
        if leg.empty:
            continue
        realized = float(pd.to_numeric(leg.iloc[0]["pnl_pct"], errors="coerce") or 0.0)
        best_v = float(best.get(oid, realized))
        regrets.append(best_v - realized)
    if not regrets:
        return {}
    arr = np.array(regrets)
    return {
        "mean_regret_pct": round(100 * float(arr.mean()), 4),
        "median_regret_pct": round(100 * float(np.median(arr)), 4),
        "n": int(len(arr)),
    }


def build_window_dashboard(
    legs: pd.DataFrame,
    selection: pd.DataFrame,
    *,
    windows: list[dict[str, Any]],
    selector: str = "E",
    starting_capital: float = 1000.0,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    closed = legs[legs.get("closed").astype(str).str.lower().isin(["true", "1"])] if "closed" in legs.columns else legs
    if "is_counterfactual" in closed.columns:
        is_cf = closed["is_counterfactual"].astype(str).str.lower().isin(["true", "1"])
        sel_legs_all = closed[~is_cf]
        cf_legs_all = closed[is_cf]
    else:
        sel_legs_all = closed[closed.get("arm_key") == selector]
        cf_legs_all = closed

    for w in windows:
        wid = str(w.get("id", ""))
        label = str(w.get("label", wid))
        win_legs = filter_legs_window(closed, window_id=wid, window_cfg=w, now=now)
        if win_legs.empty:
            rows.append({"window": label, "window_id": wid, "n_trades": 0})
            continue
        is_cf = win_legs["is_counterfactual"].astype(str).str.lower().isin(["true", "1"]) if "is_counterfactual" in win_legs.columns else pd.Series(False, index=win_legs.index)
        e_legs = win_legs[(~is_cf) & (win_legs.get("arm_key") == selector)]
        m = compute_arm_metrics(e_legs, arm_key=selector, starting_capital=starting_capital)
        conc = selection_concentration_report(
            filter_legs_window(
                selection.assign(exit_ts=selection.get("entry_ts")),
                window_id=wid,
                window_cfg={**w, "hours": w.get("hours")} if w.get("kind") != "calendar_yesterday" else w,
                now=now,
            ) if "entry_ts" in selection.columns else selection,
            selector=selector,
        )
        regret = compute_regret_metrics(
            e_legs,
            win_legs[is_cf],
            selection,
            selector=selector,
        )
        top = conc.get("top_strategy") or "—"
        top_pct = conc.get("top_strategy_pct") or 0
        rows.append({
            "window": label,
            "window_id": wid,
            "E_return_pct": m.get("cum_return_pct"),
            "PF": m.get("profit_factor"),
            "max_dd_pct": m.get("max_dd_pct"),
            "win_rate_pct": m.get("win_rate_pct"),
            "n_trades": m.get("n_trades"),
            "avg_trade_pct": m.get("avg_trade_pct"),
            "selected_strategy": f"{top} ({top_pct}%)",
            "regret_pct": regret.get("mean_regret_pct"),
            "selection_entropy": conc.get("selection_entropy"),
            "n_unique_selected": conc.get("n_unique_strategies"),
        })
    return rows


def counterfactual_performance_by_arm(cf_legs: pd.DataFrame) -> dict[str, float]:
    if cf_legs.empty:
        return {}
    g = cf_legs.copy()
    g["pnl_pct"] = pd.to_numeric(g.get("pnl_pct"), errors="coerce")
    out = {}
    for arm in LIVE_ARM_LABELS:
        sub = g[g.get("arm_key") == arm]
        if len(sub):
            out[arm] = round(100 * float(sub["pnl_pct"].mean()), 4)
    return out
