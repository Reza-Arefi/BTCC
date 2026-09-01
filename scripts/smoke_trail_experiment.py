#!/usr/bin/env python3
"""Smoke test for trail experiment (few days)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["BTCC_TELEGRAM_FORCE_OFF"] = "1"


def main() -> int:
    import pandas as pd

    from btcc.sim.trail_backtest import run_trail_experiment_backtest
    from btcc.sim.trail_config import (
        TRAIL_STRATEGY_KEYS,
        load_trail_experiment_config,
        pre_run_config_summary,
        validate_trail_strategies,
    )

    sim = load_trail_experiment_config()
    errs = validate_trail_strategies(sim)
    assert not errs, errs
    assert float(sim["long_threshold"]) == 0.55
    assert float(sim["upper_threshold"]) == 0.85
    assert sim["benchmark_strategy_key"] == "trail_5"
    assert len(sim["strategies"]) == 14
    print(pre_run_config_summary(sim))

    out = run_trail_experiment_backtest(days=5, force_download=False, sim_cfg=sim)
    legs_path = out / "strategy_legs.csv"
    opp_path = out / "opportunities.csv"
    assert legs_path.exists(), "missing strategy_legs.csv"
    legs = pd.read_csv(legs_path)
    opps = pd.read_csv(opp_path) if opp_path.exists() else pd.DataFrame()

    # Identical entries: every opportunity must open all 14 strategies
    if not legs.empty:
        by_opp = legs.groupby("opportunity_id")["strategy_key"].nunique()
        bad = by_opp[by_opp != len(TRAIL_STRATEGY_KEYS)]
        assert bad.empty, f"Non-identical strategy coverage: {bad.to_dict()}"
        # Entry fills identical within each opportunity
        for oid, g in legs.groupby("opportunity_id"):
            fills = set(g["entry_fill_price"].astype(float).round(12))
            assert len(fills) == 1, f"Divergent entry fills for {oid}: {fills}"

    if not opps.empty and "S" in opps.columns:
        s = opps["S"].astype(float)
        assert (s >= 0.55).all() and (s < 0.85).all(), "Opportunity S outside [0.55, 0.85)"

    print("SMOKE_OK", out)
    print("n_opportunities", len(opps), "n_legs", len(legs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
