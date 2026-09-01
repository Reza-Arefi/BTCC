"""Analytics pipeline for trailing-exit experiment (day-axis, % returns)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from btcc.analytics.capital import capital_daily_series, capital_trade_series
from btcc.analytics.trail_plots import generate_trail_plots
from btcc.sim.trail_config import TRAIL_STRATEGY_KEYS

logger = logging.getLogger(__name__)


def _load_tables(out_dir: Path) -> dict[str, pd.DataFrame]:
    out_dir = Path(out_dir)
    tables: dict[str, pd.DataFrame] = {}
    for name, fn in (
        ("predictions", "predictions.csv"),
        ("opportunities", "opportunities.csv"),
        ("legs", "strategy_legs.csv"),
    ):
        p = out_dir / fn
        if p.exists() and p.stat().st_size > 0:
            tables[name] = pd.read_csv(p, low_memory=False)
        else:
            tables[name] = pd.DataFrame()
    return tables


def build_trail_analytics_asof(
    out_dir: Path,
    *,
    day_number: int,
    eval_start: str,
    starting_capital_usd: float = 1000.0,
    analytics_root: Path | None = None,
    benchmark_key: str = "trail_3",
) -> Path:
    """Build as-of analytics and plots for trail experiment through day_number."""
    out_dir = Path(out_dir)
    analytics_root = Path(analytics_root or out_dir / "analytics")
    plots_root = analytics_root / "plots" / "trail"
    plots_root.mkdir(parents=True, exist_ok=True)

    tables = _load_tables(out_dir)
    legs = tables["legs"]
    opps = tables["opportunities"]
    if not legs.empty and "day_number" in legs.columns:
        legs = legs[legs["day_number"] <= int(day_number)].copy()
    if not opps.empty and "day_number" in opps.columns:
        opps = opps[opps["day_number"] <= int(day_number)].copy()

    closed = legs[legs["closed"] == True].copy() if "closed" in legs.columns else legs.copy()  # noqa: E712
    if "entry_policy" not in closed.columns:
        closed["entry_policy"] = "COMMON"

    metrics = _compute_metrics(closed, starting_capital_usd=starting_capital_usd, max_day=day_number)
    regime = _regime_metrics(closed)

    manifest = {
        "experiment_kind": "trail_exit_v1",
        "as_of_day": int(day_number),
        "eval_start": eval_start,
        "starting_capital_usd": starting_capital_usd,
        "benchmark_strategy_key": benchmark_key,
        "strategy_keys": list(TRAIL_STRATEGY_KEYS),
        "metrics": metrics,
        "regime_metrics": regime,
    }
    analytics_root.mkdir(parents=True, exist_ok=True)
    (analytics_root / "trail_metrics.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    daily_cap = capital_daily_series(closed, starting_capital_usd=starting_capital_usd, max_day=day_number)
    trade_cap = capital_trade_series(closed, starting_capital_usd=starting_capital_usd)
    generate_trail_plots(
        closed,
        daily_cap=daily_cap,
        trade_cap=trade_cap,
        out_dir=plots_root,
        max_day=day_number,
        benchmark_key=benchmark_key,
        starting_capital_usd=starting_capital_usd,
    )
    logger.info("Trail analytics day=%s → %s", day_number, analytics_root)
    return analytics_root


def _compute_metrics(legs: pd.DataFrame, *, starting_capital_usd: float, max_day: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if legs.empty:
        return out
    pnl = pd.to_numeric(legs.get("pnl_usd_equiv"), errors="coerce").fillna(0.0)
    legs = legs.copy()
    legs["_pnl_usd"] = pnl
    legs["_win"] = pd.to_numeric(legs.get("pnl_pct"), errors="coerce").fillna(0) > 0
    for sk, g in legs.groupby("strategy_key"):
        wins = g[g["_win"]]
        losses = g[~g["_win"]]
        gp = wins["_pnl_usd"].sum()
        gl = abs(losses["_pnl_usd"].sum())
        pf = (gp / gl) if gl > 1e-9 else None
        out[str(sk)] = {
            "n_trades": int(len(g)),
            "win_rate_pct": round(100 * g["_win"].mean(), 2),
            "avg_pnl_pct": round(100 * pd.to_numeric(g["pnl_pct"], errors="coerce").mean(), 4),
            "cumulative_return_pct": round(100 * g["_pnl_usd"].sum() / starting_capital_usd, 2),
            "profit_factor": round(pf, 3) if pf is not None else None,
            "avg_mfe_pct": round(100 * pd.to_numeric(g.get("mfe_pct"), errors="coerce").mean(), 4) if "mfe_pct" in g else None,
            "avg_mae_pct": round(100 * pd.to_numeric(g.get("mae_pct"), errors="coerce").mean(), 4) if "mae_pct" in g else None,
            "trail_activation_rate_pct": round(100 * g.get("trail_activated", pd.Series(False)).astype(bool).mean(), 2) if "trail_activated" in g else None,
        }
    return out


def _regime_metrics(legs: pd.DataFrame) -> dict[str, Any]:
    if legs.empty or "regime" not in legs.columns:
        return {}
    out: dict[str, Any] = {}
    pnl = pd.to_numeric(legs.get("pnl_usd_equiv"), errors="coerce").fillna(0.0)
    legs = legs.copy()
    legs["_pnl_usd"] = pnl
    legs["_win"] = pd.to_numeric(legs.get("pnl_pct"), errors="coerce").fillna(0) > 0
    for (sk, regime), g in legs.groupby(["strategy_key", "regime"]):
        out.setdefault(str(sk), {})[str(regime)] = {
            "n": int(len(g)),
            "win_rate_pct": round(100 * g["_win"].mean(), 2),
            "sum_pnl_usd": round(float(g["_pnl_usd"].sum()), 2),
            "avg_pnl_pct": round(100 * pd.to_numeric(g["pnl_pct"], errors="coerce").mean(), 4),
        }
    return out
