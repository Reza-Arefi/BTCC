#!/usr/bin/env python3
"""Smoke test for E-memory walk-forward experiment (all six E variants)."""

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
    from btcc.sim.regime import classify_regime
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

    lookbacks = {"E-10": 10, "E-30": 30, "E-60": 60, "E-90": 90, "E-180": 180, "E-365": 365}
    sels = build_selector_memory_group(lookbacks)
    assert len(sels) == 6
    assert set(s.lookback_days for s in sels.values()) == {10.0, 30.0, 60.0, 90.0, 180.0, 365.0}
    for s in sels.values():
        assert s.kind == "rank_ewma"
        assert s.cfg["half_life_days"] == 7

    # No lookahead in prior()
    hist = CounterfactualHistory()
    t0 = pd.Timestamp("2025-01-01", tz="UTC")
    for i in range(20):
        hist.record(
            opportunity_id="o1", strategy_key="trail_1",
            exit_ts=t0 + pd.Timedelta(days=i), pnl_pct=0.01, regime="RANGE",
        )
    asof = t0 + pd.Timedelta(days=15)
    prior_10 = hist.prior("trail_1", asof, lookback_days=10)
    assert len(prior_10) == 10 and all(t.exit_ts < asof for t in prior_10)
    hist.record(opportunity_id="o2", strategy_key="trail_1", exit_ts=asof + pd.Timedelta(days=1), pnl_pct=0.99, regime="RANGE")
    assert len(hist.prior("trail_1", asof, lookback_days=10)) == len(prior_10)

    # Distinct lookbacks change rank_ewma inputs
    base = pd.Timestamp("2024-01-01", tz="UTC")
    h2 = CounterfactualHistory()
    for d in range(40):
        h2.record(opportunity_id=f"o{d}", strategy_key="trail_1", exit_ts=base + pd.Timedelta(days=d), pnl_pct=0.02, regime="RANGE")
        h2.record(opportunity_id=f"o{d}b", strategy_key="trail_2", exit_ts=base + pd.Timedelta(days=d), pnl_pct=-0.01 if d < 20 else 0.05, regime="RANGE")
    asof2 = base + pd.Timedelta(days=35)
    s10 = _scores_rank_ewma(h2, asof2, 7.0, lookback_days=10, strategy_keys=("trail_1", "trail_2"))
    s30 = _scores_rank_ewma(h2, asof2, 7.0, lookback_days=30, strategy_keys=("trail_1", "trail_2"))
    assert s10 != s30 or s10["trail_1"] != s30["trail_1"]

    # BTC.D does not affect S; regime ignores BTC.D
    cfg = {"factors": {"weights": {"momentum": 0.2, "trend": 0.2, "btc_regime": 0.5, "volume": 0.15, "volatility": 0.15, "rsi": 0.15, "structure": 0.15}}}
    idx = pd.date_range("2024-01-01", periods=120, freq="15min", tz="UTC")
    rel = pd.DataFrame({"timestamp": idx, "open": 1.0, "high": 1.01, "low": 0.99, "close": 1.0, "volume": 1000.0})
    f0 = compute_all_factors(rel, rel, rel, None, {}, cfg, "15m")
    f1 = compute_all_factors(rel, rel, rel, 40.0, {1: 5.0, 4: 5.0, 12: 5.0, 24: 5.0}, cfg, "15m")
    assert f0["signal_score"] == f1["signal_score"]
    factors = {"trend": {"adx": 22.0}, "volatility": {"natr": 0.02}}
    assert classify_regime(factors)["regime"] == classify_regime(factors)["regime"]

    # Short integration run — all six lookbacks distinct; warmup covers E-365
    sim_smoke = dict(sim)
    me = dict(sim_smoke["selector_memory_experiment"])
    me["eval_days"] = 5
    me["warmup_days"] = 365
    me["lookbacks_days"] = dict(lookbacks)
    sim_smoke["selector_memory_experiment"] = me

    with patch.object(HistoricalDominanceSeries, "fetch_coingecko") as mock_cg:
        out = run_selector_memory_backtest(
            force_download=False,
            sim_cfg=sim_smoke,
            resume=False,
            eval_days=5,
            warmup_days=365,
        )
        mock_cg.assert_not_called()

    manifest = json.loads((out / "experiment_manifest.json").read_text(encoding="utf-8"))
    assert manifest.get("btc_d_enabled") is False
    assert manifest.get("btc_d_source") == "disabled"
    assert manifest.get("btc_d_affects_trading") is False
    assert set(manifest.get("lookbacks_days", {}).keys()) == set(MEMORY_ARM_LABELS)

    legs = pd.read_csv(out / "strategy_legs.csv")
    opps = pd.read_csv(out / "opportunities.csv")
    sel = pd.read_csv(out / "selection_audit.csv")

    eval_opps = opps[opps["eval_phase"].astype(str).isin(("True", "true", "1"))]
    assert not eval_opps.empty, "no eval opportunities"
    eval_legs = legs[legs["eval_phase"].astype(str).isin(("True", "true", "1"))]
    cf = eval_legs[eval_legs["is_counterfactual"].astype(str).isin(("True", "true", "1"))]
    sel_legs = eval_legs[eval_legs["is_counterfactual"].astype(str).isin(("False", "false", "0"))]

    assert set(sel["arm_label"].unique()) <= set(MEMORY_ARM_LABELS)
    assert set(sel["lookback_days"].astype(int).unique()) == {10, 30, 60, 90, 180, 365}

    by_opp_cf = cf.groupby("opportunity_id")["arm_key"].nunique()
    assert (by_opp_cf == len(CF_ARM_LABELS)).all(), f"Expected {len(CF_ARM_LABELS)} CF arms per opp"

    by_opp_sel = sel_legs.groupby("opportunity_id")["arm_key"].nunique()
    assert (by_opp_sel == len(MEMORY_ARM_LABELS)).all(), f"Expected {len(MEMORY_ARM_LABELS)} selector arms per opp"

    # Selector P/L must match counterfactual P/L for the selected strategy
    cf_map = cf.set_index(["opportunity_id", "arm_key"])["pnl_pct"].astype(float)
    mismatches = 0
    for _, row in sel_legs.iterrows():
        oid = row["opportunity_id"]
        chosen = sel[(sel["opportunity_id"] == oid) & (sel["arm_label"] == row["arm_key"])]
        if chosen.empty:
            continue
        t_arm = str(chosen.iloc[0]["selected_arm_label"])
        cf_pnl = cf_map.get((oid, t_arm))
        sel_pnl = float(row["pnl_pct"])
        if cf_pnl is not None and abs(sel_pnl - cf_pnl) > 1e-9:
            mismatches += 1
    assert mismatches == 0, f"selector/counterfactual P/L mismatches: {mismatches}"

    # Regret sanity
    regret_path = out / "analytics" / "selection_regret.csv"
    assert regret_path.exists(), "missing selection_regret.csv"
    regret = pd.read_csv(regret_path)
    assert "regret_pct" in regret.columns
    assert (regret["regret_pct"] >= -1e-6).all(), "regret should be non-negative"
    assert set(regret["arm_label"].unique()) <= set(MEMORY_ARM_LABELS)

    assert (out / "window.json").exists()
    assert (out / "analytics" / "memory_metrics.json").exists()

    print("SMOKE_OK", out)
    print("n_eval_opps", len(eval_opps), "n_sel_rows", len(sel), "n_regret", len(regret))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
