"""Analytics pipeline for selector experiment (18 arms, day-axis % returns)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from btcc.analytics.capital import capital_daily_series, capital_trade_series
from btcc.analytics.selector_plots import generate_selector_plots
from btcc.sim.selector_config import ALL_ARM_LABELS, FIXED_ARM_LABELS, SELECTOR_ARM_LABELS

logger = logging.getLogger(__name__)


def _load_tables(out_dir: Path) -> dict[str, pd.DataFrame]:
    out_dir = Path(out_dir)
    tables: dict[str, pd.DataFrame] = {}
    for name, fn in (
        ("predictions", "predictions.csv"),
        ("opportunities", "opportunities.csv"),
        ("legs", "strategy_legs.csv"),
        ("selection", "selection_audit.csv"),
    ):
        p = out_dir / fn
        if p.exists() and p.stat().st_size > 0:
            try:
                tables[name] = pd.read_csv(p, low_memory=False)
            except pd.errors.EmptyDataError:
                tables[name] = pd.DataFrame()
        else:
            tables[name] = pd.DataFrame()
    return tables


def _legs_for_capital(legs: pd.DataFrame) -> pd.DataFrame:
    """Group independent accounts by arm_key (T1–T12 + A–F)."""
    if legs.empty:
        return legs
    df = legs.copy()
    if "arm_key" not in df.columns:
        df["arm_key"] = df.get("strategy_key")
    df["strategy_key"] = df["arm_key"]
    if "entry_policy" not in df.columns:
        df["entry_policy"] = "COMMON"
    return df


def _compute_metrics(legs: pd.DataFrame, *, starting_capital_usd: float, max_day: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if legs.empty:
        return out
    pnl = pd.to_numeric(legs.get("pnl_usd_equiv"), errors="coerce").fillna(0.0)
    legs = legs.copy()
    legs["_pnl_usd"] = pnl
    for arm in ALL_ARM_LABELS:
        g = legs[legs["arm_key"] == arm] if "arm_key" in legs.columns else pd.DataFrame()
        if g.empty:
            continue
        wins = (pd.to_numeric(g.get("pnl_pct"), errors="coerce") > 0).sum()
        gp = g.loc[g["_pnl_usd"] > 0, "_pnl_usd"].sum()
        gl = abs(g.loc[g["_pnl_usd"] < 0, "_pnl_usd"].sum())
        out[arm] = {
            "n_trades": int(len(g)),
            "win_rate_pct": 100.0 * wins / len(g) if len(g) else 0.0,
            "profit_factor": float(gp / gl) if gl > 1e-9 else None,
            "avg_trade_return_pct": 100.0 * float(pd.to_numeric(g.get("pnl_pct"), errors="coerce").mean()),
            "total_pnl_usd": float(g["_pnl_usd"].sum()),
            "final_equity_usd": starting_capital_usd + float(g["_pnl_usd"].sum()),
            "cumulative_return_pct": 100.0 * float(g["_pnl_usd"].sum()) / starting_capital_usd,
        }
    return out


def _regime_metrics(legs: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if legs.empty or "regime" not in legs.columns:
        return out
    cf = legs[legs.get("is_counterfactual") == True] if "is_counterfactual" in legs.columns else legs  # noqa: E712
    if cf.empty:
        cf = legs
    pnl = 100.0 * pd.to_numeric(cf.get("pnl_pct"), errors="coerce").fillna(0.0)
    for regime, g in cf.groupby("regime"):
        idx = g.index
        by_arm: dict[str, float] = {}
        for arm in FIXED_ARM_LABELS:
            ga = g[g["arm_key"] == arm] if "arm_key" in g.columns else g[g["strategy_key"].str.endswith(arm[-1])]
            if len(ga):
                by_arm[arm] = float(100.0 * pd.to_numeric(ga["pnl_pct"], errors="coerce").mean())
        out[str(regime)] = by_arm
    return out


def _selection_regret(legs: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    if legs.empty or selection.empty:
        return pd.DataFrame()
    cf = legs[legs.get("is_counterfactual") == True].copy()  # noqa: E712
    if cf.empty:
        return pd.DataFrame()
    cf["pnl_pct"] = pd.to_numeric(cf["pnl_pct"], errors="coerce")
    best = cf.groupby("opportunity_id")["pnl_pct"].max().rename("best_cf_pnl_pct")
    rows = []
    sel_legs = legs[legs.get("is_counterfactual") != True].copy()  # noqa: E712
    for _, s in selection.iterrows():
        oid = s["opportunity_id"]
        arm = s["arm_label"]
        sel_leg = sel_legs[(sel_legs["opportunity_id"] == oid) & (sel_legs["arm_key"] == arm)]
        if sel_leg.empty:
            continue
        realized = float(pd.to_numeric(sel_leg.iloc[0]["pnl_pct"], errors="coerce") or 0.0)
        best_v = float(best.get(oid, realized))
        rows.append({
            "opportunity_id": oid,
            "arm_label": arm,
            "realized_pnl_pct": realized,
            "best_counterfactual_pnl_pct": best_v,
            "regret_pct": best_v - realized,
        })
    return pd.DataFrame(rows)


def build_selector_analytics_asof(
    out_dir: Path,
    *,
    day_number: int,
    eval_start: str,
    starting_capital_usd: float = 1000.0,
    analytics_root: Path | None = None,
) -> Path:
    out_dir = Path(out_dir)
    analytics_root = Path(analytics_root or out_dir / "analytics")
    plots_root = analytics_root / "plots" / "selector"
    plots_root.mkdir(parents=True, exist_ok=True)

    tables = _load_tables(out_dir)
    legs = tables["legs"]
    opps = tables["opportunities"]
    selection = tables["selection"]
    if not legs.empty and "day_number" in legs.columns:
        legs = legs[legs["day_number"] <= int(day_number)].copy()
    if not opps.empty and "day_number" in opps.columns:
        opps = opps[opps["day_number"] <= int(day_number)].copy()
    if not selection.empty and "day_number" in selection.columns:
        selection = selection[selection["day_number"] <= int(day_number)].copy()

    closed = legs[legs["closed"] == True].copy() if "closed" in legs.columns else legs.copy()  # noqa: E712
    cap_legs = _legs_for_capital(closed)
    metrics = _compute_metrics(closed, starting_capital_usd=starting_capital_usd, max_day=day_number)
    regime = _regime_metrics(closed)
    regret = _selection_regret(closed, selection)

    manifest = {
        "experiment_kind": "selector_experiment_v1",
        "as_of_day": int(day_number),
        "eval_start": eval_start,
        "starting_capital_usd": starting_capital_usd,
        "arms": list(ALL_ARM_LABELS),
        "metrics": metrics,
        "regime_metrics": regime,
    }
    analytics_root.mkdir(parents=True, exist_ok=True)
    (analytics_root / "selector_metrics.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    if not regret.empty:
        regret.to_csv(analytics_root / "selection_regret.csv", index=False)

    daily_cap = capital_daily_series(cap_legs, starting_capital_usd=starting_capital_usd, max_day=day_number)
    trade_cap = capital_trade_series(cap_legs, starting_capital_usd=starting_capital_usd)
    generate_selector_plots(
        closed,
        daily_cap=daily_cap,
        trade_cap=trade_cap,
        out_dir=plots_root,
        max_day=day_number,
        starting_capital_usd=starting_capital_usd,
        opportunities=opps,
        selection=selection,
        regret=regret,
    )
    logger.info("Selector analytics day=%s → %s", day_number, analytics_root)
    return analytics_root
