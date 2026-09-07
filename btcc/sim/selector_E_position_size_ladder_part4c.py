"""Part 4C — Selector E fixed position-size ladder.

Isolates whether Part 4A 7D_INVERSE gains are mostly larger size vs adaptive timing.

Accounting (matches Parts 4A/4B + frozen reference compound_notional=false):
  requested_notional = fixed_fraction * starting_capital_usd (1000)
  actual_notional = min(requested, available equity under 100% exposure)
  pnl = reference_pnl_usd_equiv * (actual_notional / 100)

FIXED_10 must reproduce $1,702.43. Paper/live bot untouched.
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
    BASE_NOTIONAL_USD,
    MAX_OPEN,
    STARTING_CAPITAL_USD,
    _utc,
    load_selector_E_stream,
    verify_fixed10_vs_reference,
)

logger = logging.getLogger(__name__)

ARM_SPECS: tuple[tuple[str, float], ...] = (
    ("FIXED_5", 0.05),
    ("FIXED_7_5", 0.075),
    ("FIXED_10", 0.10),
    ("FIXED_12_5", 0.125),
    ("FIXED_15", 0.15),
    ("FIXED_17_5", 0.175),
    ("FIXED_20", 0.20),
    ("FIXED_25", 0.25),
)

ARM_ORDER = tuple(k for k, _ in ARM_SPECS)
FRACTION_BY_ARM = dict(ARM_SPECS)


def simulate_fixed_arm(trades: pd.DataFrame, *, arm: str, fraction: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Event-driven portfolio replay with fixed requested allocation fraction."""
    rows: list[dict[str, Any]] = []
    equity = STARTING_CAPITAL_USD
    reserved = 0.0
    open_book: dict[str, dict[str, Any]] = {}

    events: list[tuple[pd.Timestamp, int, int, str]] = []
    for i, r in trades.iterrows():
        events.append((_utc(r["entry_ts"]), 0, int(i), "entry"))
        events.append((_utc(r["exit_ts"]), 1, int(i), "exit"))
    events.sort(key=lambda x: (x[0], x[1], x[2]))

    peak = equity
    max_dd = 0.0
    max_exposure = 0.0
    max_simultaneous = 0
    sum_simultaneous_at_entry = 0
    exposure_limited = 0
    fully_available = 0
    skipped_max_open = 0
    max_act_alloc = 0.0
    daily_pnl: dict[pd.Timestamp, float] = {}

    for ts, _, i, kind in events:
        r = trades.loc[i]
        oid = str(r["opportunity_id"])

        if kind == "exit":
            pos = open_book.pop(oid, None)
            if pos is None:
                continue
            reserved = max(0.0, reserved - float(pos["actual_notional"]))
            equity += float(pos["pnl_usd"])
            peak = max(peak, equity)
            dd = (equity - peak) / peak if peak else 0.0
            max_dd = min(max_dd, dd)
            day = _utc(r["exit_ts"]).floor("D")
            daily_pnl[day] = daily_pnl.get(day, 0.0) + float(pos["pnl_usd"])
            rows.append({**pos, "equity_after": equity, "drawdown_pct": 100.0 * dd})
            continue

        # entry
        if len(open_book) >= MAX_OPEN:
            skipped_max_open += 1
            rows.append(
                {
                    "opportunity_id": oid,
                    "arm": arm,
                    "entry_ts": r["entry_ts"],
                    "exit_ts": r["exit_ts"],
                    "executed": False,
                    "skip_reason": "MAX_OPEN",
                    "requested_allocation": fraction,
                    "actual_allocation": 0.0,
                    "requested_notional": 0.0,
                    "actual_notional": 0.0,
                    "pnl_usd": 0.0,
                    "pnl_pct": float(r["pnl_pct"]),
                    "ref_pnl_usd": float(r["pnl_usd_equiv"]),
                    "exposure_limited": False,
                    "fully_available": False,
                    "equity_before": equity,
                    "n_open_before": len(open_book),
                    "strategy_key": r.get("strategy_key"),
                    "symbol": r.get("symbol"),
                }
            )
            continue

        req_notional = fraction * STARTING_CAPITAL_USD
        available = max(0.0, equity - reserved)
        actual_notional = min(req_notional, available)
        exp_lim = actual_notional + 1e-9 < req_notional
        if exp_lim:
            exposure_limited += 1
        else:
            fully_available += 1

        act_alloc = actual_notional / STARTING_CAPITAL_USD if STARTING_CAPITAL_USD else 0.0
        max_act_alloc = max(max_act_alloc, act_alloc)
        scale = actual_notional / BASE_NOTIONAL_USD if BASE_NOTIONAL_USD else 0.0
        pnl_usd = float(r["pnl_usd_equiv"]) * scale

        reserved += actual_notional
        n_open = len(open_book) + 1
        max_simultaneous = max(max_simultaneous, n_open)
        sum_simultaneous_at_entry += n_open
        exposure_now = reserved / equity if equity else 0.0
        max_exposure = max(max_exposure, exposure_now)

        open_book[oid] = {
            "opportunity_id": oid,
            "arm": arm,
            "entry_ts": r["entry_ts"],
            "exit_ts": r["exit_ts"],
            "day_number": r.get("day_number"),
            "symbol": r.get("symbol"),
            "strategy_key": r.get("strategy_key"),
            "executed": True,
            "skip_reason": "",
            "requested_allocation": fraction,
            "actual_allocation": act_alloc,
            "requested_notional": req_notional,
            "actual_notional": actual_notional,
            "pnl_usd": pnl_usd,
            "pnl_pct": float(r["pnl_pct"]),
            "ref_pnl_usd": float(r["pnl_usd_equiv"]),
            "scale": scale,
            "exposure_limited": exp_lim,
            "fully_available": not exp_lim,
            "equity_before": equity,
            "reserved_before": reserved - actual_notional,
            "available_before": available,
            "exposure_after_entry": exposure_now,
            "n_open_after_entry": n_open,
        }

    if open_book:
        raise RuntimeError(f"{arm}: {len(open_book)} trades still open")

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("exit_ts").reset_index(drop=True)
    executed = out[out["executed"] == True] if not out.empty else out  # noqa: E712

    wins = executed[executed["pnl_usd"] > 0]["pnl_usd"] if len(executed) else pd.Series(dtype=float)
    losses = executed[executed["pnl_usd"] < 0]["pnl_usd"] if len(executed) else pd.Series(dtype=float)
    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0
    pf = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

    worst_day = min(daily_pnl.values()) if daily_pnl else 0.0
    worst_day_date = min(daily_pnl, key=daily_pnl.get) if daily_pnl else None

    # monthly worst
    monthly = {}
    if len(executed):
        tmp = executed.copy()
        tmp["month"] = pd.to_datetime(tmp["exit_ts"], utc=True).dt.tz_localize(None).dt.to_period("M").astype(str)
        monthly = tmp.groupby("month")["pnl_usd"].sum().to_dict()
    worst_month_pnl = min(monthly.values()) if monthly else 0.0
    worst_month = min(monthly, key=monthly.get) if monthly else None

    n_exec = int(len(executed))
    summary = {
        "arm": arm,
        "nominal_allocation": fraction,
        "starting_equity_usd": STARTING_CAPITAL_USD,
        "final_equity_usd": float(equity),
        "total_return_pct": 100.0 * (equity / STARTING_CAPITAL_USD - 1.0),
        "total_pnl_usd": float(equity - STARTING_CAPITAL_USD),
        "n_opportunities": int(len(trades)),
        "n_executed": n_exec,
        "n_skipped_max_open": int(skipped_max_open),
        "n_exposure_limited": int(exposure_limited),
        "n_fully_available": int(fully_available),
        "pct_fully_available": 100.0 * fully_available / n_exec if n_exec else 0.0,
        "pct_exposure_limited": 100.0 * exposure_limited / n_exec if n_exec else 0.0,
        "win_rate_pct": 100.0 * float((executed["pnl_usd"] > 0).mean()) if n_exec else 0.0,
        "profit_factor": float(pf),
        "avg_trade_pnl_usd": float(executed["pnl_usd"].mean()) if n_exec else 0.0,
        "median_trade_pnl_usd": float(executed["pnl_usd"].median()) if n_exec else 0.0,
        "gross_profit_usd": gross_profit,
        "gross_loss_usd": gross_loss,
        "max_drawdown_pct": 100.0 * float(max_dd),
        "worst_trade_usd": float(executed["pnl_usd"].min()) if n_exec else 0.0,
        "worst_day_usd": float(worst_day),
        "worst_day": str(worst_day_date) if worst_day_date is not None else None,
        "worst_month_usd": float(worst_month_pnl),
        "worst_month": worst_month,
        "mean_actual_allocation": float(executed["actual_allocation"].mean()) if n_exec else 0.0,
        "median_actual_allocation": float(executed["actual_allocation"].median()) if n_exec else 0.0,
        "min_actual_allocation": float(executed["actual_allocation"].min()) if n_exec else 0.0,
        "max_actual_allocation": float(max_act_alloc),
        "max_simultaneous_positions": int(max_simultaneous),
        "avg_simultaneous_positions": float(sum_simultaneous_at_entry / n_exec) if n_exec else 0.0,
        "avg_exposure_at_entry": (
            float(executed["exposure_after_entry"].mean()) if n_exec and "exposure_after_entry" in executed else 0.0
        ),
        "max_exposure": float(max_exposure),
        "return_over_abs_dd": (
            float(100.0 * (equity / STARTING_CAPITAL_USD - 1.0) / abs(100.0 * max_dd))
            if max_dd < 0
            else float("inf")
        ),
    }
    return out, summary


def verify_identity_across_arms(trade_tables: dict[str, pd.DataFrame], ref_trades: pd.DataFrame) -> dict[str, Any]:
    """Checks B–E: same opportunities / strategies / timestamps across arms."""
    ref_ids = list(ref_trades["opportunity_id"].astype(str))
    ref_set = set(ref_ids)
    checks = {}
    for arm, df in trade_tables.items():
        ex = df[df["executed"] == True] if "executed" in df.columns else df  # noqa: E712
        ids = list(ex["opportunity_id"].astype(str))
        same_ids = set(ids) == ref_set and len(ids) == len(ref_ids)
        # join strategy / timestamps
        m = ex.merge(
            ref_trades[
                ["opportunity_id", "strategy_key", "entry_ts", "exit_ts", "pnl_pct"]
            ].assign(opportunity_id=lambda x: x["opportunity_id"].astype(str)),
            on="opportunity_id",
            how="left",
            suffixes=("", "_ref"),
        )
        same_strat = bool((m["strategy_key"].astype(str) == m["strategy_key_ref"].astype(str)).all()) if len(m) else False
        same_entry = bool(
            (pd.to_datetime(m["entry_ts"], utc=True) == pd.to_datetime(m["entry_ts_ref"], utc=True)).all()
        )
        same_exit = bool(
            (pd.to_datetime(m["exit_ts"], utc=True) == pd.to_datetime(m["exit_ts_ref"], utc=True)).all()
        )
        same_pnl_pct = bool(np.allclose(m["pnl_pct"].astype(float), m["pnl_pct_ref"].astype(float), equal_nan=True))
        checks[arm] = {
            "same_opportunity_ids": same_ids,
            "same_strategy_keys": same_strat,
            "same_entry_ts": same_entry,
            "same_exit_ts": same_exit,
            "same_pnl_pct": same_pnl_pct,
            "n_executed": int(len(ex)),
        }
    ok = all(all(v.values()) if isinstance(v, dict) else v for arm, c in checks.items() for v in [c] for _ in [0])
    # simpler
    ok = all(
        c["same_opportunity_ids"]
        and c["same_strategy_keys"]
        and c["same_entry_ts"]
        and c["same_exit_ts"]
        and c["same_pnl_pct"]
        for c in checks.values()
    )
    return {"passed": ok, "per_arm": checks}


def run_all_arms(ref_dir: Path) -> dict[str, Any]:
    opp, trades, ref_metrics = load_selector_E_stream(ref_dir)
    baseline = verify_fixed10_vs_reference(trades, ref_metrics)

    trade_tables: dict[str, pd.DataFrame] = {}
    summaries: list[dict[str, Any]] = []

    # FIXED_10 first
    t10, s10 = simulate_fixed_arm(trades, arm="FIXED_10", fraction=0.10)
    trade_tables["FIXED_10"] = t10
    summaries.append(s10)
    fixed_match = abs(s10["final_equity_usd"] - baseline["fixed10"]["final_equity_usd"]) <= 1e-6
    baseline["fixed10_sim_matches_replay"] = fixed_match
    baseline["fixed10_sim_final_equity_usd"] = s10["final_equity_usd"]

    if not baseline["passed"] or not fixed_match:
        return {
            "stopped": True,
            "reason": "FIXED_10 does not reproduce Selector E reference",
            "baseline_check": baseline,
            "trades": trades,
            "ref_metrics": ref_metrics,
            "trade_tables": trade_tables,
            "summaries": summaries,
        }

    for arm, frac in ARM_SPECS:
        if arm == "FIXED_10":
            continue
        tdf, sm = simulate_fixed_arm(trades, arm=arm, fraction=frac)
        trade_tables[arm] = tdf
        summaries.append(sm)

    identity = verify_identity_across_arms(trade_tables, trades)

    return {
        "stopped": False,
        "baseline_check": baseline,
        "identity_check": identity,
        "trades": trades,
        "opportunities": opp,
        "ref_metrics": ref_metrics,
        "trade_tables": trade_tables,
        "summaries": summaries,
        "ref_dir": str(ref_dir),
        "sizing_convention": (
            "requested_notional_usd = fixed_fraction * starting_capital_usd (1000), matching "
            "Parts 4A/4B and the frozen reference (compound_notional=false, $100 baseline notional). "
            "PnL scaled from reference pnl_usd_equiv by actual_notional/100. "
            "Exposure: open notional sum ≤ current equity (no leverage). "
            "Max simultaneous positions = 10. Insufficient remaining exposure → allocate remaining "
            "(existing research-replay semantics), never skip solely for partial exposure."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def make_results_dir(root: Path | None = None) -> Path:
    root = Path(root or "results")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = root / f"selector_E_position_size_ladder_part4c_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    (out / "plots").mkdir(exist_ok=True)
    return out
