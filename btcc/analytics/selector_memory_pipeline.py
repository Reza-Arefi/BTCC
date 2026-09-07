"""Analytics pipeline for E-memory walk-forward experiment."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from btcc.analytics.capital import capital_daily_series, capital_trade_series
from btcc.analytics.selector_memory_plots import generate_memory_plots
from btcc.sim.selector_memory_config import (
    CF_ARM_LABELS,
    FIXED_T1_LATE_FILTER,
    FIXED_T1_NO_LATE,
    MEMORY_ARM_LABELS,
    selector_arm_labels,
)

logger = logging.getLogger(__name__)

REGIMES = ("HIGH_VOLATILITY", "LOW_VOL_TREND", "NORMAL_TREND", "RANGE", "STRONG_TREND")
SBAND_BINS = [(0.60, 0.65), (0.65, 0.70), (0.70, 0.80)]


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
            tables[name] = pd.read_csv(p, low_memory=False)
        else:
            tables[name] = pd.DataFrame()
    return tables


def _eval_legs(legs: pd.DataFrame) -> pd.DataFrame:
    if legs.empty:
        return legs
    df = legs.copy()
    if "eval_phase" in df.columns:
        df = df[df["eval_phase"].astype(str).isin(("True", "true", "1", "1.0", True))].copy()
    if "closed" in df.columns:
        df = df[df["closed"].astype(str).isin(("True", "true", "1", "1.0", True))].copy()
    return df


def _selector_legs(legs: pd.DataFrame) -> pd.DataFrame:
    df = _eval_legs(legs)
    if df.empty:
        return df
    if "is_counterfactual" in df.columns:
        df = df[df["is_counterfactual"].astype(str).isin(("False", "false", "0", "0.0", False))].copy()
    return df


def _arm_metrics(g: pd.DataFrame, *, starting_capital_usd: float) -> dict[str, Any]:
    if g.empty:
        return {}
    pnl_pct = pd.to_numeric(g.get("pnl_pct"), errors="coerce").fillna(0.0)
    pnl_usd = pd.to_numeric(g.get("pnl_usd_equiv"), errors="coerce").fillna(0.0)
    wins = pnl_pct > 0
    losses = pnl_pct < 0
    gp = float(pnl_usd[wins].sum())
    gl = abs(float(pnl_usd[losses].sum()))
    cum = float(pnl_usd.sum())
    eq = starting_capital_usd + cum
    dd = 0.0
    if not pnl_usd.empty:
        curve = starting_capital_usd + pnl_usd.cumsum()
        peak = curve.cummax()
        dd = float((curve - peak).min())
    consec = 0
    max_consec = 0
    for v in pnl_pct:
        if v < 0:
            consec += 1
            max_consec = max(max_consec, consec)
        else:
            consec = 0
    return {
        "n_trades": int(len(g)),
        "cumulative_return_pct": 100.0 * cum / starting_capital_usd,
        "avg_trade_return_pct": 100.0 * float(pnl_pct.mean()),
        "median_trade_return_pct": 100.0 * float(pnl_pct.median()),
        "win_rate_pct": 100.0 * float(wins.sum()) / len(g),
        "profit_factor": float(gp / gl) if gl > 1e-9 else None,
        "max_drawdown_usd": dd,
        "max_drawdown_pct": 100.0 * dd / starting_capital_usd if starting_capital_usd else 0.0,
        "gross_profit_usd": gp,
        "gross_loss_usd": gl,
        "avg_winning_trade_pct": 100.0 * float(pnl_pct[wins].mean()) if wins.any() else None,
        "avg_losing_trade_pct": 100.0 * float(pnl_pct[losses].mean()) if losses.any() else None,
        "largest_win_pct": 100.0 * float(pnl_pct.max()),
        "largest_loss_pct": 100.0 * float(pnl_pct.min()),
        "max_consecutive_losses": max_consec,
        "return_volatility_pct": 100.0 * float(pnl_pct.std()) if len(g) > 1 else 0.0,
    }


def _selection_entropy(freqs: dict[str, float]) -> float:
    vals = [v for v in freqs.values() if v > 0]
    if not vals:
        return 0.0
    return -sum(p * math.log(p) for p in vals)


def _selection_behavior(selection: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if selection.empty:
        return out
    for arm in MEMORY_ARM_LABELS:
        g = selection[selection["arm_label"] == arm]
        if g.empty:
            continue
        counts = g["selected_arm_label"].value_counts(normalize=True)
        freqs = {str(k): float(v) for k, v in counts.items()}
        out[arm] = {
            "selection_frequency": {f"T{i}": freqs.get(f"T{i}", 0.0) for i in range(1, 11)},
            "unique_strategies": int(g["selected_arm_label"].nunique()),
            "dominant_strategy": str(counts.index[0]) if len(counts) else None,
            "dominant_pct": float(counts.iloc[0]) if len(counts) else 0.0,
            "entropy": _selection_entropy(freqs),
        }
    return out


def _fixed_t1_summary(legs: pd.DataFrame) -> dict[str, Any]:
    sel = _selector_legs(legs)
    if sel.empty:
        return {}
    out: dict[str, Any] = {}
    for arm in (FIXED_T1_NO_LATE, FIXED_T1_LATE_FILTER):
        g = sel[sel["arm_key"] == arm]
        if g.empty:
            continue
        out[arm] = {"n_trades": int(len(g))}
    if FIXED_T1_NO_LATE in out and FIXED_T1_LATE_FILTER in out:
        out["late_filter_rejections"] = int(out[FIXED_T1_NO_LATE]["n_trades"] - out[FIXED_T1_LATE_FILTER]["n_trades"])
    return out


def _selection_regret(legs: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    if legs.empty or selection.empty:
        return pd.DataFrame()
    cf = _eval_legs(legs)
    cf = cf[cf["is_counterfactual"].astype(str).isin(("True", "true", "1", "1.0", True))].copy()
    if cf.empty:
        return pd.DataFrame()
    cf["pnl_pct"] = pd.to_numeric(cf["pnl_pct"], errors="coerce")
    best_pnl = cf.groupby("opportunity_id")["pnl_pct"].max()
    # idxmax → row label of best CF arm per opportunity (safe; avoids positional-index mixups)
    best_idx = cf.groupby("opportunity_id")["pnl_pct"].idxmax()
    best_arm_by_oid = cf.loc[best_idx, ["opportunity_id", "arm_key"]].set_index("opportunity_id")["arm_key"]
    rows = []
    sel_legs = _selector_legs(legs)
    for _, s in selection.iterrows():
        oid = s["opportunity_id"]
        arm = s["arm_label"]
        sel_leg = sel_legs[(sel_legs["opportunity_id"] == oid) & (sel_legs["arm_key"] == arm)]
        if sel_leg.empty:
            continue
        realized = float(pd.to_numeric(sel_leg.iloc[0]["pnl_pct"], errors="coerce") or 0.0)
        best_v = float(best_pnl.loc[oid]) if oid in best_pnl.index else realized
        best_arm = str(best_arm_by_oid.loc[oid]) if oid in best_arm_by_oid.index else s.get("selected_arm_label")
        rows.append({
            "opportunity_id": oid,
            "arm_label": arm,
            "selected_strategy": s.get("selected_arm_label"),
            "best_counterfactual_strategy": best_arm,
            "selected_return_pct": 100.0 * realized,
            "best_counterfactual_return_pct": 100.0 * best_v,
            "regret_pct": 100.0 * (best_v - realized),
            "lookback_days": s.get("lookback_days"),
        })
    return pd.DataFrame(rows)


def _subperiod_metrics(legs: pd.DataFrame, *, eval_start: str, eval_days: int, starting_capital_usd: float) -> dict[str, Any]:
    sel = _selector_legs(legs)
    if sel.empty:
        return {}
    eval_start_ts = pd.Timestamp(eval_start)
    if eval_start_ts.tzinfo is None:
        eval_start_ts = eval_start_ts.tz_localize("UTC")
    exit_ts = pd.to_datetime(sel.get("exit_ts"), utc=True, errors="coerce")
    sel = sel.copy()
    sel["_exit_ts"] = exit_ts
    total_end = eval_start_ts + pd.Timedelta(days=eval_days)
    quarters = [
        ("full_year", eval_start_ts, total_end),
        ("Q1", eval_start_ts, eval_start_ts + pd.Timedelta(days=eval_days // 4)),
        ("Q2", eval_start_ts + pd.Timedelta(days=eval_days // 4), eval_start_ts + pd.Timedelta(days=eval_days // 2)),
        ("Q3", eval_start_ts + pd.Timedelta(days=eval_days // 2), eval_start_ts + pd.Timedelta(days=3 * eval_days // 4)),
        ("Q4", eval_start_ts + pd.Timedelta(days=3 * eval_days // 4), total_end),
    ]
    out: dict[str, Any] = {}
    trading_arms = selector_arm_labels()
    for arm in trading_arms:
        ga = sel[sel["arm_key"] == arm]
        out[arm] = {}
        for label, start, end in quarters:
            block = ga[(ga["_exit_ts"] >= start) & (ga["_exit_ts"] < end)]
            m = _arm_metrics(block, starting_capital_usd=starting_capital_usd)
            out[arm][label] = m
    return out


def _regime_table(legs: pd.DataFrame, opps: pd.DataFrame, *, starting_capital_usd: float) -> dict[str, Any]:
    sel = _selector_legs(legs)
    if sel.empty or "regime" not in sel.columns:
        return {}
    out: dict[str, Any] = {}
    for arm in selector_arm_labels():
        ga = sel[sel["arm_key"] == arm]
        out[arm] = {}
        for regime in REGIMES:
            g = ga[ga["regime"] == regime]
            out[arm][regime] = _arm_metrics(g, starting_capital_usd=starting_capital_usd)
    return out


def _sband_table(legs: pd.DataFrame, opps: pd.DataFrame, *, starting_capital_usd: float) -> dict[str, Any]:
    sel = _selector_legs(legs)
    if sel.empty or opps.empty:
        return {}
    opp_s = opps[["opportunity_id", "S"]].drop_duplicates("opportunity_id")
    opp_s["S"] = pd.to_numeric(opp_s["S"], errors="coerce")
    merged = sel.merge(opp_s, on="opportunity_id", how="left")
    out: dict[str, Any] = {}
    for arm in selector_arm_labels():
        ga = merged[merged["arm_key"] == arm]
        out[arm] = {}
        for lo, hi in SBAND_BINS:
            label = f"{lo:.2f}-{hi:.2f}"
            g = ga[(ga["S"] >= lo) & (ga["S"] < hi)]
            out[arm][label] = _arm_metrics(g, starting_capital_usd=starting_capital_usd)
    return out


def _robustness_score(metrics: dict[str, Any], regret_stats: dict[str, Any], behavior: dict[str, Any]) -> list[tuple[str, float]]:
    """Higher is better — composite stability score (not return-only)."""
    scores = []
    for arm in MEMORY_ARM_LABELS:
        m = metrics.get(arm) or {}
        r = regret_stats.get(arm) or {}
        b = behavior.get(arm) or {}
        ret = float(m.get("cumulative_return_pct") or 0.0)
        dd = abs(float(m.get("max_drawdown_pct") or 0.0))
        pf = float(m.get("profit_factor") or 1.0)
        avg_reg = float(r.get("mean_regret_pct") or 0.0)
        entropy = float(b.get("entropy") or 0.0)
        # Penalize high DD and regret; reward PF and moderate entropy
        s = ret * 0.35 + min(pf, 5.0) * 8.0 - dd * 0.5 - avg_reg * 0.3 + entropy * 5.0
        scores.append((arm, s))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def _recommendation(metrics: dict, regret_stats: dict, robustness: list) -> dict[str, Any]:
    if not robustness:
        return {"candidate": None, "rationale": "Insufficient data"}
    top = robustness[0][0]
    m = metrics.get(top) or {}
    r = regret_stats.get(top) or {}
    return {
        "candidate": top,
        "rationale": (
            f"{top} ranks highest on composite robustness (return={m.get('cumulative_return_pct', 0):.1f}%, "
            f"maxDD={m.get('max_drawdown_pct', 0):.1f}%, PF={m.get('profit_factor')}, "
            f"mean regret={r.get('mean_regret_pct', 0):.2f}%). "
            "Recommendation is evidence-based, not hard-coded."
        ),
        "robustness_ranking": [{"arm": a, "score": s} for a, s in robustness],
    }


def build_memory_analytics_asof(
    out_dir: Path,
    *,
    day_number: int,
    eval_start: str,
    starting_capital_usd: float = 1000.0,
    analytics_root: Path | None = None,
) -> Path:
    out_dir = Path(out_dir)
    analytics_root = Path(analytics_root or out_dir / "analytics")
    plots_root = analytics_root / "plots" / "memory"
    plots_root.mkdir(parents=True, exist_ok=True)

    tables = _load_tables(out_dir)
    legs = tables["legs"]
    opps = tables["opportunities"]
    selection = tables["selection"]
    if not legs.empty and "day_number" in legs.columns:
        legs = legs[legs["day_number"] <= int(day_number)].copy()
    if not selection.empty and "day_number" in selection.columns:
        selection = selection[selection["day_number"] <= int(day_number)].copy()

    sel_legs = _selector_legs(legs)
    cap_legs = sel_legs.copy()
    if not cap_legs.empty:
        cap_legs["strategy_key"] = cap_legs["arm_key"]
        cap_legs["entry_policy"] = "COMMON"

    metrics = {arm: _arm_metrics(sel_legs[sel_legs["arm_key"] == arm], starting_capital_usd=starting_capital_usd) for arm in selector_arm_labels()}
    regret_df = _selection_regret(legs, selection)
    regret_stats: dict[str, Any] = {}
    for arm in selector_arm_labels():
        g = regret_df[regret_df["arm_label"] == arm]["regret_pct"] if not regret_df.empty else pd.Series(dtype=float)
        if g.empty:
            continue
        regret_stats[arm] = {
            "mean_regret_pct": float(g.mean()),
            "median_regret_pct": float(g.median()),
            "p90_regret_pct": float(g.quantile(0.90)),
            "p95_regret_pct": float(g.quantile(0.95)),
            "max_regret_pct": float(g.max()),
        }
    behavior = _selection_behavior(selection)
    fixed_t1 = _fixed_t1_summary(legs)

    manifest_path = out_dir / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    eval_days = int(manifest.get("eval_days", 365))

    payload = {
        "day_number": day_number,
        "metrics": metrics,
        "regret_stats": regret_stats,
        "selection_behavior": behavior,
        "fixed_t1_baselines": fixed_t1,
        "subperiods": _subperiod_metrics(legs, eval_start=eval_start, eval_days=eval_days, starting_capital_usd=starting_capital_usd),
        "regime": _regime_table(legs, opps, starting_capital_usd=starting_capital_usd),
        "s_band": _sband_table(legs, opps, starting_capital_usd=starting_capital_usd),
    }
    analytics_root.mkdir(parents=True, exist_ok=True)
    (analytics_root / "memory_metrics.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    if not regret_df.empty:
        regret_df.to_csv(analytics_root / "selection_regret.csv", index=False)

    daily_cap = (
        capital_daily_series(cap_legs, starting_capital_usd=starting_capital_usd, max_day=day_number)
        if not cap_legs.empty
        else pd.DataFrame()
    )
    if not daily_cap.empty and "equity_usd" not in daily_cap.columns and "ending_value" in daily_cap.columns:
        daily_cap = daily_cap.copy()
        daily_cap["equity_usd"] = daily_cap["ending_value"]
    generate_memory_plots(
        analytics_root,
        metrics=metrics,
        regret_df=regret_df,
        regret_stats=regret_stats,
        behavior=behavior,
        daily_cap=daily_cap,
        regime=payload["regime"],
        s_band=payload["s_band"],
        max_day=day_number,
    )
    return analytics_root


def build_memory_analytics_final(out_dir: Path, *, starting_capital_usd: float = 1000.0) -> Path:
    out_dir = Path(out_dir)
    window_path = out_dir / "window.json"
    manifest_path = out_dir / "experiment_manifest.json"
    eval_start = str(json.loads(window_path.read_text())["eval_start"]) if window_path.exists() else str(json.loads(manifest_path.read_text())["eval_start"])
    legs = _load_tables(out_dir)["legs"]
    max_day = int(pd.to_numeric(legs.get("day_number"), errors="coerce").max()) if not legs.empty and "day_number" in legs.columns else 365
    analytics = build_memory_analytics_asof(
        out_dir, day_number=max_day, eval_start=eval_start, starting_capital_usd=starting_capital_usd,
    )

    payload = json.loads((analytics / "memory_metrics.json").read_text(encoding="utf-8"))
    metrics = payload.get("metrics") or {}
    regret_stats = payload.get("regret_stats") or {}
    behavior = payload.get("selection_behavior") or {}
    robustness = _robustness_score(metrics, regret_stats, behavior)
    recommendation = _recommendation(metrics, regret_stats, robustness)
    payload["robustness_ranking"] = recommendation.get("robustness_ranking")
    payload["recommendation"] = recommendation
    (analytics / "memory_metrics.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    _write_report(out_dir, payload, recommendation)
    return analytics


def _write_report(out_dir: Path, payload: dict, recommendation: dict) -> None:
    manifest = json.loads((out_dir / "experiment_manifest.json").read_text(encoding="utf-8"))
    lines = [
        "# E-Memory Walk-Forward Experiment Report",
        "",
        "## 1. Objective",
        "Compare Selector E lookback lengths (3–90 days) under strict walk-forward evaluation.",
        "",
        "## 2. Dataset period",
        f"- CF warmup start: {manifest.get('cf_warmup_start')}",
        f"- Eval start: {manifest.get('eval_start')}",
        f"- Eval end: {manifest.get('eval_end')}",
        f"- Eval days: {manifest.get('eval_days')}",
        "",
        "## 3. Walk-forward methodology",
        "Counterfactual T1–T10 accumulate during warmup; selector legs trade only during eval window.",
        "No future information used at decision time (`exit_ts < asof`, lookback window enforced).",
        "",
        "## 4. E configurations",
    ]
    for arm in MEMORY_ARM_LABELS:
        lb = (manifest.get("lookbacks_days") or {}).get(arm)
        lines.append(f"- {arm}: {lb}d lookback")
    lines += [
        "",
        "## 5. Fixed parameters",
        f"- EWMA half-life: {manifest.get('ewma_half_life_days')}d",
        f"- Min hold: {(manifest.get('switching') or {}).get('minimum_selection_duration_hours')}h",
        f"- BTC.D enabled: {manifest.get('btc_d_enabled', manifest.get('btc_d', {}).get('enabled'))}",
        f"- BTC.D source: {manifest.get('btc_d_source', 'n/a')}",
        f"- BTC.D affects trading: {manifest.get('btc_d_affects_trading', False)}",
        "",
        "## 6–13. Results",
        "See `analytics/memory_metrics.json`, plots under `analytics/plots/memory/`.",
        "",
        "## 14. Limitations",
        "E-90 requires 90-day warmup; all variants share the same 365-day eval window.",
        "",
        "## 15. Recommended candidate",
        f"**{recommendation.get('candidate')}** — {recommendation.get('rationale')}",
    ]
    (out_dir / "analytics" / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
