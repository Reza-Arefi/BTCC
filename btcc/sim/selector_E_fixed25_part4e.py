"""Part 4E — Dedicated Selector E fixed-25% 365d standalone reference run.

Reuses Part 4C FIXED_25 accounting so the dedicated package reproduces:
  final equity ≈ $2,750, return ≈ +175.0%, max DD ≈ −4.31%.

Paper/live untouched. No adaptive sizing. No strategy changes.
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
from btcc.sim.selector_E_position_size_ladder_part4c import simulate_fixed_arm

logger = logging.getLogger(__name__)

PART4C_FIXED25_TARGETS = {
    "final_equity_usd": 2750.236887,
    "total_return_pct": 175.023689,
    "max_drawdown_pct": -4.312769,
}
PART4C_TOL = {"equity": 1e-3, "return_pp": 1e-3, "dd_pp": 1e-3}


def run_fixed25(ref_dir: Path) -> dict[str, Any]:
    opp, trades, ref_metrics = load_selector_E_stream(ref_dir)

    # Identity vs reference E stream
    baseline10 = verify_fixed10_vs_reference(trades, ref_metrics)
    t10, s10 = simulate_fixed_arm(trades, arm="FIXED_10", fraction=0.10)
    fixed10_ok = (
        baseline10["passed"]
        and abs(s10["final_equity_usd"] - baseline10["fixed10"]["final_equity_usd"]) <= 1e-6
    )

    t25, s25 = simulate_fixed_arm(trades, arm="FIXED_25", fraction=0.25)

    # Identity: same opportunity set / strategies / timestamps / pnl_pct
    ex = t25[t25["executed"] == True].copy()  # noqa: E712
    m = ex.merge(
        trades[
            ["opportunity_id", "strategy_key", "entry_ts", "exit_ts", "pnl_pct", "symbol"]
        ].assign(opportunity_id=lambda x: x["opportunity_id"].astype(str)),
        on="opportunity_id",
        how="left",
        suffixes=("", "_ref"),
    )
    identity = {
        "n_ref_opportunities": int(len(trades)),
        "n_executed": int(len(ex)),
        "n_partial_or_constrained": int(ex["exposure_limited"].sum()) if len(ex) else 0,
        "same_opportunity_ids": set(ex["opportunity_id"].astype(str)) == set(trades["opportunity_id"].astype(str)),
        "same_strategy_keys": bool(
            (m["strategy_key"].astype(str) == m["strategy_key_ref"].astype(str)).all()
        )
        if len(m)
        else False,
        "same_entry_ts": bool(
            (pd.to_datetime(m["entry_ts"], utc=True) == pd.to_datetime(m["entry_ts_ref"], utc=True)).all()
        )
        if len(m)
        else False,
        "same_exit_ts": bool(
            (pd.to_datetime(m["exit_ts"], utc=True) == pd.to_datetime(m["exit_ts_ref"], utc=True)).all()
        )
        if len(m)
        else False,
        "same_pnl_pct": bool(
            np.allclose(m["pnl_pct"].astype(float), m["pnl_pct_ref"].astype(float), equal_nan=True)
        )
        if len(m)
        else False,
    }
    identity["passed"] = all(
        [
            identity["same_opportunity_ids"],
            identity["same_strategy_keys"],
            identity["same_entry_ts"],
            identity["same_exit_ts"],
            identity["same_pnl_pct"],
            identity["n_ref_opportunities"] == 1517,
        ]
    )

    return {
        "ref_dir": str(ref_dir),
        "trades_ref": trades,
        "opportunities": opp,
        "ref_metrics": ref_metrics,
        "fixed10_check": {
            "passed": fixed10_ok,
            "baseline": baseline10,
            "sim_summary": s10,
        },
        "trades_25": t25,
        "summary_25": s25,
        "trades_10": t10,
        "summary_10": s10,
        "identity": identity,
        "sizing_convention": (
            "requested_notional = 0.25 * starting_capital_usd ($1000), matching Part 4C / "
            "frozen reference compound_notional=false semantics. PnL scaled from reference "
            "pnl_usd_equiv by actual_notional/100. Exposure: open notional ≤ current equity; "
            "partial fill when remaining exposure < requested (Part 4C Policy A)."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def make_results_dir(root: Path | None = None) -> Path:
    root = Path(root or "results")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = root / f"selector_E_fixed25_365d_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    (out / "plots").mkdir(exist_ok=True)
    return out
