"""Tests for E-memory lookback walk-forward experiment."""

from __future__ import annotations

import pandas as pd

from btcc.sim.selector_engine import CounterfactualHistory, SelectorState, _scores_rank_ewma, build_selector_memory_group
from btcc.sim.selector_memory_config import MEMORY_ARM_LABELS, load_selector_memory_config, validate_selector_memory


def test_memory_config_valid():
    sim = load_selector_memory_config()
    assert not validate_selector_memory(sim)
    lb = sim["selector_memory_experiment"]["lookbacks_days"]
    assert set(lb.keys()) == set(MEMORY_ARM_LABELS)


def test_lookback_no_lookahead():
    hist = CounterfactualHistory()
    base = pd.Timestamp("2024-06-01", tz="UTC")
    for d in range(100):
        for k in ("trail_1", "trail_2"):
            hist.record(opportunity_id="o", strategy_key=k, exit_ts=base + pd.Timedelta(days=d), pnl_pct=0.01 * d, regime="RANGE")
    asof = base + pd.Timedelta(days=50)
    scores_full = _scores_rank_ewma(hist, asof, 7.0, lookback_days=None, strategy_keys=("trail_1", "trail_2"))
    scores_10 = _scores_rank_ewma(hist, asof, 7.0, lookback_days=10, strategy_keys=("trail_1", "trail_2"))
    assert scores_full.keys() == scores_10.keys()
    prior = hist.prior("trail_1", asof, lookback_days=10)
    assert all(t.exit_ts < asof for t in prior)
    assert all(t.exit_ts >= asof - pd.Timedelta(days=10) for t in prior)
    future = base + pd.Timedelta(days=51)
    hist.record(opportunity_id="o2", strategy_key="trail_1", exit_ts=future, pnl_pct=0.99, regime="RANGE")
    prior2 = hist.prior("trail_1", asof, lookback_days=10)
    assert len(prior2) == len(prior)


def test_build_memory_group_distinct_lookbacks():
    lb = {"E-10": 10, "E-30": 30, "E-60": 60, "E-90": 90, "E-180": 180, "E-365": 365}
    group = build_selector_memory_group(lb)
    assert len(group) == 6
    vals = {s.lookback_days for s in group.values()}
    assert vals == {10.0, 30.0, 60.0, 90.0, 180.0, 365.0}
    for s in group.values():
        assert s.kind == "rank_ewma"
        assert s.cfg["half_life_days"] == 7


def test_selector_state_roundtrip_lookback():
    st = SelectorState(
        selector_id="selector_e_10",
        arm_label="E-10",
        kind="rank_ewma",
        cfg={"half_life_days": 7},
        lookback_days=10.0,
    )
    restored = SelectorState.from_dict(st.to_dict())
    assert restored.lookback_days == 10.0
