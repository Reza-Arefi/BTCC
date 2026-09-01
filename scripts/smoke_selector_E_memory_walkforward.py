#!/usr/bin/env python3
"""Smoke test for E-memory walk-forward experiment (E-3..E-90)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["BTCC_TELEGRAM_FORCE_OFF"] = "1"


def main() -> int:
    import pandas as pd

    from btcc.backtest.dominance_history import HistoricalDominanceSeries
    from btcc.factors.combine import compute_all_factors
    from btcc.sim.selector_engine import CounterfactualHistory, _scores_rank_ewma, build_selector_memory_group
    from btcc.sim.selector_memory_backtest import run_selector_memory_backtest
    from btcc.sim.selector_memory_config import (
        CF_ARM_LABELS,
        MEMORY_ARM_LABELS,
        load_selector_memory_config,
        pre_run_memory_summary,
        validate_selector_memory,
    )

    sim = load_selector_memory_config()
    errs = validate_selector_memory(sim)
    assert not errs, errs
    assert sim.get("allow_trading") is False
    assert (sim.get("btc_d") or {}).get("enabled") is False
    print(pre_run_memory_summary(sim))

    lookbacks = {"E-3": 3, "E-7": 7, "E-14": 14, "E-30": 30, "E-60": 60, "E-90": 90}
    sels = build_selector_memory_group(lookbacks)
    assert len(sels) == 6
    assert set(s.lookback_days for s in sels.values()) == {3.0, 7.0, 14.0, 30.0, 60.0, 90.0}
    for s in sels.values():
        assert s.kind == "rank_ewma"
        assert s.cfg["half_life_days"] == 7

    # No lookahead
    hist = CounterfactualHistory()
    t0 = pd.Timestamp("2025-01-01", tz="UTC")
    for i in range(20):
        hist.record(
            opportunity_id="o1", strategy_key="trail_1",
            exit_ts=t0 + pd.Timedelta(days=i), pnl_pct=0.01, regime="RANGE",
        )
    asof = t0 + pd.Timedelta(days=15)
    n0 = len(hist.prior("trail_1", asof, lookback_days=7))
    hist.record(opportunity_id="o2", strategy_key="trail_1", exit_ts=asof + pd.Timedelta(days=1), pnl_pct=0.99, regime="RANGE")
    assert len(hist.prior("trail_1", asof, lookback_days=7)) == n0

    # Distinct lookbacks change rank_ewma scores
    base = pd.Timestamp("2024-01-01", tz="UTC")
    h2 = CounterfactualHistory()
    for d in range(40):
        h2.record(opportunity_id=f"o{d}", strategy_key="trail_1", exit_ts=base + pd.Timedelta(days=d), pnl_pct=0.02, regime="RANGE")
        h2.record(opportunity_id=f"o{d}b", strategy_key="trail_2", exit_ts=base + pd.Timedelta(days=d), pnl_pct=-0.01 if d < 20 else 0.05, regime="RANGE")
    asof2 = base + pd.Timedelta(days=35)
    s7 = _scores_rank_ewma(h2, asof2, 7.0, lookback_days=7, strategy_keys=("trail_1", "trail_2"))
    s30 = _scores_rank_ewma(h2, asof2, 7.0, lookback_days=30, strategy_keys=("trail_1", "trail_2"))
    assert s7 != s30

    # BTC.D does not affect S
    cfg = {"factors": {"weights": {"momentum": 0.2, "trend": 0.2, "btc_regime": 0.5, "volume": 0.15, "volatility": 0.15, "rsi": 0.15, "structure": 0.15}}}
    idx = pd.date_range("2024-01-01", periods=120, freq="15min", tz="UTC")
    rel = pd.DataFrame({"timestamp": idx, "open": 1.0, "high": 1.01, "low": 0.99, "close": 1.0, "volume": 1000.0})
    assert compute_all_factors(rel, rel, rel, None, {}, cfg, "15m")["signal_score"] == compute_all_factors(rel, rel, rel, 40.0, {1: 5, 4: 5, 12: 5, 24: 5}, cfg, "15m")["signal_score"]

    # Short integration: 90d warmup + 5d eval (455d data window vs old 730d)
    sim_smoke = dict(sim)
    me = dict(sim_smoke["selector_memory_experiment"])
    me["eval_days"] = 5
    me["warmup_days"] = 90
    me["lookbacks_days"] = dict(lookbacks)
    sim_smoke["selector_memory_experiment"] = me

    with patch.object(HistoricalDominanceSeries, "fetch_coingecko") as mock_cg:
        out = run_selector_memory_backtest(
            force_download=False,
            sim_cfg=sim_smoke,
            resume=False,
            eval_days=5,
            warmup_days=90,
        )
        mock_cg.assert_not_called()

    manifest = json.loads((out / "experiment_manifest.json").read_text(encoding="utf-8"))
    assert manifest.get("btc_d_enabled") is False
    assert manifest.get("btc_d_source") == "disabled"
    assert set(manifest.get("lookbacks_days", {}).keys()) == set(MEMORY_ARM_LABELS)

    legs = pd.read_csv(out / "strategy_legs.csv")
    opps = pd.read_csv(out / "opportunities.csv")
    sel = pd.read_csv(out / "selection_audit.csv")

    eval_opps = opps[opps["eval_phase"].astype(str).isin(("True", "true", "1"))]
    assert not eval_opps.empty, "no eval opportunities"
    eval_legs = legs[legs["eval_phase"].astype(str).isin(("True", "true", "1"))]
    cf = eval_legs[eval_legs["is_counterfactual"].astype(str).isin(("True", "true", "1"))]
    sel_legs = eval_legs[eval_legs["is_counterfactual"].astype(str).isin(("False", "false", "0"))]

    assert set(sel["lookback_days"].astype(int).unique()) == {3, 7, 14, 30, 60, 90}
    assert (cf.groupby("opportunity_id")["arm_key"].nunique() == len(CF_ARM_LABELS)).all()
    assert (sel_legs.groupby("opportunity_id")["arm_key"].nunique() == len(MEMORY_ARM_LABELS)).all()

    cf_map = cf.set_index(["opportunity_id", "arm_key"])["pnl_pct"].astype(float)
    mismatches = 0
    for _, row in sel_legs.iterrows():
        chosen = sel[(sel["opportunity_id"] == row["opportunity_id"]) & (sel["arm_label"] == row["arm_key"])]
        if chosen.empty:
            continue
        t_arm = str(chosen.iloc[0]["selected_arm_label"])
        cf_pnl = cf_map.get((row["opportunity_id"], t_arm))
        if cf_pnl is not None and abs(float(row["pnl_pct"]) - cf_pnl) > 1e-9:
            mismatches += 1
    assert mismatches == 0, f"selector/counterfactual P/L mismatches: {mismatches}"

    regret = pd.read_csv(out / "analytics" / "selection_regret.csv")
    assert (regret["regret_pct"] >= -1e-6).all()

    print("SMOKE_OK", out)
    print("n_eval_opps", len(eval_opps), "n_sel", len(sel), "n_regret", len(regret))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
