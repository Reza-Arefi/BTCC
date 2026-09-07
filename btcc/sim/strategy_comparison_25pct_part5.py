"""Part 5 — Full strategy comparison at fixed 25% allocation (T1–T10 + A–F).

Replays immutable reference strategy_legs for each arm with Part 4C/4E
25% sizing semantics. Excludes T11/T12. Paper/live untouched.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from btcc.sim.selector_E_adaptive_sizing import (
    STARTING_CAPITAL_USD,
    load_selector_E_stream,
    verify_fixed10_vs_reference,
)
from btcc.sim.selector_E_fixed25_part4e import PART4C_FIXED25_TARGETS, PART4C_TOL
from btcc.sim.selector_E_position_size_ladder_part4c import simulate_fixed_arm

logger = logging.getLogger(__name__)

FIXED_ARMS = tuple(f"T{i}" for i in range(1, 11))
SELECTOR_ARMS = tuple("ABCDEF")
ALL_ARMS = FIXED_ARMS + SELECTOR_ARMS  # 16
FRACTION = 0.25


def load_arm_trades(ref_dir: Path, arm: str) -> pd.DataFrame:
    legs = pd.read_csv(ref_dir / "strategy_legs.csv", low_memory=False)
    g = legs[legs["arm_key"] == arm].copy()
    if g.empty:
        raise RuntimeError(f"No legs for arm {arm}")
    # Drop T11/T12 if somehow requested
    if arm in ("T11", "T12"):
        raise RuntimeError(f"{arm} is retired and must not be run")
    g["entry_ts"] = pd.to_datetime(g["entry_ts"], utc=True)
    g["exit_ts"] = pd.to_datetime(g["exit_ts"], utc=True)
    g["pnl_pct"] = pd.to_numeric(g["pnl_pct"], errors="coerce").fillna(0.0)
    g["pnl_usd_equiv"] = pd.to_numeric(g["pnl_usd_equiv"], errors="coerce").fillna(0.0)
    g["pnl_btc"] = pd.to_numeric(g.get("pnl_btc"), errors="coerce").fillna(0.0)
    if "closed" in g.columns:
        g = g[g["closed"] == True]  # noqa: E712
    g = g.sort_values(["entry_ts", "opportunity_id"]).reset_index(drop=True)
    return g


def _arm_type(arm: str) -> str:
    return "selector" if arm in SELECTOR_ARMS else "fixed_trail"


def _max_streak(mask: pd.Series) -> int:
    best = cur = 0
    for v in mask.astype(bool):
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def enrich_summary(trades: pd.DataFrame, summary: dict[str, Any], arm: str) -> dict[str, Any]:
    ex = trades[trades["executed"] == True].copy()  # noqa: E712
    ex = ex.sort_values("exit_ts")
    pnl = ex["pnl_usd"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    eq = STARTING_CAPITAL_USD + pnl.cumsum()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    out = dict(summary)
    out.update(
        {
            "strategy": arm,
            "type": _arm_type(arm),
            "allocation": FRACTION,
            "final_equity": float(eq.iloc[-1]) if len(eq) else STARTING_CAPITAL_USD,
            "return_pct": float(100.0 * (eq.iloc[-1] / STARTING_CAPITAL_USD - 1.0)) if len(eq) else 0.0,
            "max_drawdown_pct": float(100.0 * dd.min()) if len(dd) else 0.0,
            "trades": int(len(ex)),
            "wins": int((pnl > 0).sum()),
            "losses": int((pnl < 0).sum()),
            "win_rate": float(100.0 * (pnl > 0).mean()) if len(pnl) else 0.0,
            "avg_trade_pct": float(ex["pnl_pct"].mean() * 100.0) if len(ex) else 0.0,
            "median_trade_pct": float(ex["pnl_pct"].median() * 100.0) if len(ex) else 0.0,
            "avg_winner_pct": float(ex.loc[pnl > 0, "pnl_pct"].mean() * 100.0) if len(wins) else 0.0,
            "avg_loser_pct": float(ex.loc[pnl < 0, "pnl_pct"].mean() * 100.0) if len(losses) else 0.0,
            "largest_winner_usd": float(wins.max()) if len(wins) else 0.0,
            "largest_loser_usd": float(losses.min()) if len(losses) else 0.0,
            "gross_profit": float(wins.sum()) if len(wins) else 0.0,
            "gross_loss": float(-losses.sum()) if len(losses) else 0.0,
            "profit_factor": float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() != 0 else float("inf"),
            "expectancy": float(pnl.mean()) if len(pnl) else 0.0,
            "max_consecutive_wins": _max_streak(pnl > 0),
            "max_consecutive_losses": _max_streak(pnl < 0),
            "max_drawdown_duration_trades": _max_streak(dd < -1e-12),
            "avg_allocation": float(ex["actual_allocation"].mean()) if len(ex) else 0.0,
            "avg_open_exposure": float(ex["exposure_after_entry"].mean()) if len(ex) else 0.0,
            "max_open_exposure": float(ex["exposure_after_entry"].max()) if len(ex) else 0.0,
            "avg_concurrent_positions": float(ex["n_open_after_entry"].mean()) if len(ex) else 0.0,
            "max_concurrent_positions": int(ex["n_open_after_entry"].max()) if len(ex) else 0,
            "exposure_constrained_pct": float(100.0 * ex["exposure_limited"].mean()) if len(ex) else 0.0,
            "n_exposure_constrained": int(ex["exposure_limited"].sum()) if len(ex) else 0,
            "n_skipped": int((trades["executed"] == False).sum()),  # noqa: E712
            "n_partial": int(ex["exposure_limited"].sum()) if len(ex) else 0,
        }
    )
    out["return_to_drawdown"] = (
        float(out["return_pct"] / abs(out["max_drawdown_pct"])) if out["max_drawdown_pct"] < 0 else float("inf")
    )
    return out


def build_daily(trades: pd.DataFrame) -> pd.DataFrame:
    ex = trades[trades["executed"] == True].copy()  # noqa: E712
    ex["exit_ts"] = pd.to_datetime(ex["exit_ts"], utc=True)
    ex["entry_ts"] = pd.to_datetime(ex["entry_ts"], utc=True)
    ex = ex.sort_values("exit_ts")
    ex["exit_day"] = ex["exit_ts"].dt.floor("D")
    ex["entry_day"] = ex["entry_ts"].dt.floor("D")
    start = ex["entry_ts"].min().floor("D")
    end = ex["exit_ts"].max().floor("D")
    days = pd.date_range(start, end, freq="D", tz="UTC")
    pnl_by = ex.groupby("exit_day")["pnl_usd"].sum()
    equity = STARTING_CAPITAL_USD
    peak = equity
    rows = []
    for d in days:
        day_pnl = float(pnl_by.get(d, 0.0))
        start_eq = equity
        equity = start_eq + day_pnl
        peak = max(peak, equity)
        open_mask = (ex["entry_day"] <= d) & (ex["exit_day"] > d)
        open_n = int(open_mask.sum())
        rows.append(
            {
                "date": d,
                "starting_equity": start_eq,
                "ending_equity": equity,
                "daily_pnl": day_pnl,
                "daily_return_pct": 100.0 * day_pnl / start_eq if start_eq else 0.0,
                "drawdown_pct": 100.0 * (equity / peak - 1.0) if peak else 0.0,
                "open_positions": open_n,
            }
        )
    return pd.DataFrame(rows)


def build_monthly(trades: pd.DataFrame) -> pd.DataFrame:
    ex = trades[trades["executed"] == True].copy()  # noqa: E712
    ex["exit_ts"] = pd.to_datetime(ex["exit_ts"], utc=True)
    ex = ex.sort_values("exit_ts")
    ex["month"] = ex["exit_ts"].dt.tz_localize(None).dt.to_period("M").astype(str)
    equity = STARTING_CAPITAL_USD
    rows = []
    months = list(ex.groupby("month", sort=True))
    for i, (month, g) in enumerate(months):
        start = equity
        pnl = float(g["pnl_usd"].sum())
        equity = start + pnl
        path = start + g["pnl_usd"].cumsum()
        dd = 100.0 * (path / path.cummax() - 1.0)
        label = "partial_first" if i == 0 else ("partial_last" if i == len(months) - 1 else "full")
        rows.append(
            {
                "month": month,
                "month_label": label,
                "starting_equity": start,
                "ending_equity": equity,
                "return_pct": 100.0 * pnl / start if start else 0.0,
                "monthly_pnl": pnl,
                "trades": int(len(g)),
                "wins": int((g["pnl_usd"] > 0).sum()),
                "losses": int((g["pnl_usd"] < 0).sum()),
                "max_drawdown": float(dd.min()) if len(dd) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def run_all(ref_dir: Path) -> dict[str, Any]:
    ref_dir = Path(ref_dir)
    # E identity / Part 4C gate first
    _, e_trades, ref_metrics = load_selector_E_stream(ref_dir)
    baseline10 = verify_fixed10_vs_reference(e_trades, ref_metrics)
    t10, s10 = simulate_fixed_arm(e_trades, arm="FIXED_10", fraction=0.10)
    if not baseline10["passed"] or abs(s10["final_equity_usd"] - baseline10["fixed10"]["final_equity_usd"]) > 1e-6:
        return {
            "stopped": True,
            "reason": "FIXED_10 does not match immutable E reference",
            "baseline10": baseline10,
        }

    trade_tables: dict[str, pd.DataFrame] = {}
    daily_tables: dict[str, pd.DataFrame] = {}
    monthly_tables: dict[str, pd.DataFrame] = {}
    summaries: list[dict[str, Any]] = []

    for arm in ALL_ARMS:
        logger.info("Simulating %s @ 25%%", arm)
        src = load_arm_trades(ref_dir, arm)
        if len(src) != 1517:
            logger.warning("%s has %d trades (expected 1517)", arm, len(src))
        sim, sm = simulate_fixed_arm(src, arm=arm, fraction=FRACTION)
        enriched = enrich_summary(sim, sm, arm)
        # worst day / month
        daily = build_daily(sim)
        monthly = build_monthly(sim)
        if len(daily):
            enriched["worst_day_pct"] = float(daily["daily_return_pct"].min())
            enriched["worst_day"] = str(daily.loc[daily["daily_return_pct"].idxmin(), "date"])
        if len(monthly):
            enriched["worst_month_pct"] = float(monthly["return_pct"].min())
            enriched["worst_month"] = str(monthly.loc[monthly["return_pct"].idxmin(), "month"])
        trade_tables[arm] = sim
        daily_tables[arm] = daily
        monthly_tables[arm] = monthly
        summaries.append(enriched)

    summary_df = pd.DataFrame(summaries).set_index("strategy").reindex(ALL_ARMS).reset_index()

    # E vs Part 4C
    e_row = summary_df[summary_df["strategy"] == "E"].iloc[0]
    e_check = {
        "passed": (
            abs(float(e_row["final_equity"]) - PART4C_FIXED25_TARGETS["final_equity_usd"]) <= PART4C_TOL["equity"]
            and abs(float(e_row["return_pct"]) - PART4C_FIXED25_TARGETS["total_return_pct"]) <= PART4C_TOL["return_pp"]
            and abs(float(e_row["max_drawdown_pct"]) - PART4C_FIXED25_TARGETS["max_drawdown_pct"]) <= PART4C_TOL["dd_pp"]
        ),
        "E": {
            "final_equity": float(e_row["final_equity"]),
            "return_pct": float(e_row["return_pct"]),
            "max_drawdown_pct": float(e_row["max_drawdown_pct"]),
        },
        "part4c_targets": PART4C_FIXED25_TARGETS,
    }

    # Fairness checks
    fairness = {
        "same_period": True,
        "starting_equity_1000": all(abs(s["starting_equity_usd"] - 1000) < 1e-9 for s in summaries),
        "allocation_25": all(abs(s["allocation"] - 0.25) < 1e-12 for s in summaries),
        "no_leverage_max_exposure_le_1": all(float(s["max_open_exposure"]) <= 1.0 + 1e-9 for s in summaries),
        "max_concurrent_le_10": all(int(s["max_concurrent_positions"]) <= 10 for s in summaries),
        "t11_t12_absent": "T11" not in ALL_ARMS and "T12" not in ALL_ARMS,
        "n_arms": len(ALL_ARMS),
        "E_reproduces_part4c": e_check["passed"],
        "all_n_opportunities_1517": all(int(s["n_opportunities"]) == 1517 for s in summaries),
    }
    fairness["passed"] = all(
        [
            fairness["starting_equity_1000"],
            fairness["allocation_25"],
            fairness["no_leverage_max_exposure_le_1"],
            fairness["max_concurrent_le_10"],
            fairness["t11_t12_absent"],
            fairness["E_reproduces_part4c"],
            fairness["all_n_opportunities_1517"],
        ]
    )

    return {
        "stopped": False,
        "ref_dir": str(ref_dir),
        "trade_tables": trade_tables,
        "daily_tables": daily_tables,
        "monthly_tables": monthly_tables,
        "summary": summary_df,
        "e_check": e_check,
        "fairness": fairness,
        "baseline10": baseline10,
        "sizing_convention": (
            "requested_notional = 0.25 * starting_capital_usd ($1000), matching Part 4C/4E "
            "and frozen reference compound_notional=false. Exposure vs current equity; no leverage."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def make_results_dir(root: Path | None = None) -> Path:
    root = Path(root or "results")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = root / f"strategy_comparison_25pct_365d_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    (out / "plots").mkdir(exist_ok=True)
    (out / "trade_results").mkdir(exist_ok=True)
    (out / "daily_equity").mkdir(exist_ok=True)
    return out
