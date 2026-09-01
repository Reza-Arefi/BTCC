#!/usr/bin/env python3
"""Smoke test for selector experiment (5 days)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["BTCC_TELEGRAM_FORCE_OFF"] = "1"


def main() -> int:
    import pandas as pd

    from btcc.sim.selector_backtest import run_selector_experiment_backtest
    from btcc.sim.selector_config import (
        ALL_ARM_LABELS,
        FIXED_ARM_LABELS,
        FIXED_STRATEGY_KEYS,
        load_selector_experiment_config,
        pre_run_config_summary,
        validate_selector_experiment,
    )

    sim = load_selector_experiment_config()
    errs = validate_selector_experiment(sim)
    assert not errs, errs
    assert float(sim["long_threshold"]) == 0.60
    assert sim.get("upper_threshold") is None
    assert len(sim["strategies"]) == 12
    print(pre_run_config_summary(sim))

    out = run_selector_experiment_backtest(days=5, force_download=False, sim_cfg=sim, resume=False)
    legs = pd.read_csv(out / "strategy_legs.csv")
    opps = pd.read_csv(out / "opportunities.csv")
    sel = pd.read_csv(out / "selection_audit.csv")

    assert not legs.empty, "no legs"
    # Each opportunity: 12 counterfactual + 6 selector = 18 legs
    by_opp = legs.groupby("opportunity_id")["arm_key"].nunique()
    assert (by_opp == 18).all(), f"Expected 18 arms per opp: {by_opp[by_opp != 18].to_dict()}"

    cf = legs[legs["is_counterfactual"] == True]  # noqa: E712
    for oid, g in cf.groupby("opportunity_id"):
        fills = set(g["entry_fill_price"].astype(float).round(12))
        assert len(fills) == 1, f"Divergent CF entry fills {oid}"

    if not opps.empty:
        assert (opps["S"].astype(float) >= 0.60).all(), "S below 0.60"

    assert not sel.empty, "missing selection audit"
    assert set(sel["arm_label"].unique()) <= set("ABCDEF")

    # Analytics artifacts
    assert (out / "analytics" / "selector_metrics.json").exists() or (out / "daily_checkpoints").exists()

    print("SMOKE_OK", out)
    print("n_opportunities", len(opps), "n_legs", len(legs), "n_selections", len(sel))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
