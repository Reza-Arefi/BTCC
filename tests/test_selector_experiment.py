"""Tests for selector experiment configuration and engine."""

from __future__ import annotations

import pandas as pd

from btcc.sim.selector_config import (
    FIXED_STRATEGY_KEYS,
    load_selector_experiment_config,
    validate_selector_experiment,
)
from btcc.sim.selector_engine import CounterfactualHistory, SelectorState, build_selector_group


def test_config_valid():
    sim = load_selector_experiment_config()
    assert not validate_selector_experiment(sim)
    assert sim.get("upper_threshold") is None
    assert set(sim["strategies"].keys()) == set(FIXED_STRATEGY_KEYS)


def test_selector_no_lookahead():
    sim = load_selector_experiment_config()
    selectors = build_selector_group(sim)
    hist = CounterfactualHistory()
    hist.record(
        opportunity_id="o1",
        strategy_key="trail_1",
        exit_ts="2025-01-02 12:00:00+00:00",
        pnl_pct=0.01,
        regime="RANGE",
    )
    sel = selectors["selector_a"]
    pick_before = sel.select(hist, pd.Timestamp("2025-01-02 11:00:00+00:00"), "RANGE")
    pick_after = sel.select(hist, pd.Timestamp("2025-01-02 13:00:00+00:00"), "RANGE")
    assert pick_before["scores"]["trail_1"] == 0.0
    assert pick_after["scores"]["trail_1"] > 0.0


def test_switching_hysteresis():
    cfg = {"kind": "ewma_7d", "half_life_days": 7}
    sel = SelectorState(
        selector_id="selector_a",
        arm_label="A",
        kind="ewma_7d",
        cfg=cfg,
        min_duration_hours=6.0,
        switch_margin=0.01,
        current_strategy_key="trail_1",
    )
    sel.last_switch_ts = pd.Timestamp("2025-01-01 00:00:00+00:00")
    hist = CounterfactualHistory()
    for i, sk in enumerate(["trail_1", "trail_2"]):
        hist.record(
            opportunity_id=f"o{i}",
            strategy_key=sk,
            exit_ts=f"2025-01-01 {10+i}:00:00+00:00",
            pnl_pct=0.001 if sk == "trail_1" else 0.05,
            regime="RANGE",
        )
    pick = sel.select(hist, pd.Timestamp("2025-01-01 02:00:00+00:00"), "RANGE")
    assert pick["selected_strategy_key"] == "trail_1"
