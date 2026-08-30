"""A/B/C weight-arm unit tests — Static / Equal / Adaptive fairness invariants."""

from __future__ import annotations

import pandas as pd

from btcc.config import load_config
from btcc.sim.maturity import filter_matured_for_learning, outcome_mature_at
from btcc.sim.score import (
    FACTOR_KEYS,
    combined_score,
    equal_factor_weights,
    normalize_weights,
    static_factor_weights,
)
from btcc.sim.weight_schedule import WeightSchedule, WeightVersion


def test_static_weights_from_signal_config():
    cfg = load_config()
    w = static_factor_weights(cfg)
    assert set(w) == set(FACTOR_KEYS)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    raw = cfg["factors"]["weights"]
    expected = normalize_weights(raw)
    for k in FACTOR_KEYS:
        assert abs(w[k] - expected[k]) < 1e-12


def test_equal_weights_one_over_n():
    w = equal_factor_weights()
    assert set(w) == set(FACTOR_KEYS)
    n = len(FACTOR_KEYS)
    for k in FACTOR_KEYS:
        assert abs(w[k] - 1.0 / n) < 1e-12
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_static_and_equal_same_indicator_set():
    cfg = load_config()
    a = static_factor_weights(cfg)
    b = equal_factor_weights()
    assert set(a) == set(b) == set(FACTOR_KEYS)
    assert a != b


def test_combined_score_identical_factors_across_arms():
    factors = {k: 0.7 for k in FACTOR_KEYS}
    factors["momentum"] = 0.9
    factors["rsi"] = 0.3
    cfg = load_config()
    sa = combined_score(factors, static_factor_weights(cfg))
    sb = combined_score(factors, equal_factor_weights())
    assert sa["signed"] == sb["signed"]
    assert sa["weights"] != sb["weights"]


def test_static_schedule_never_changes():
    cfg = load_config()
    w = static_factor_weights(cfg)
    sched = WeightSchedule(w, version_id="static_config_weights")
    assert len(sched.history()) == 1
    t = pd.Timestamp("2026-06-01", tz="UTC")
    assert sched.active_at(t).weights == w


def test_equal_weights_remain_equal_after_normalize():
    w = equal_factor_weights()
    assert normalize_weights(w) == w


def test_adaptive_effective_from_next_bar():
    cfg = load_config()
    seed = static_factor_weights(cfg)
    sched = WeightSchedule(seed, version_id="config_pre_init")
    t0 = pd.Timestamp("2026-04-01", tz="UTC")
    t1 = pd.Timestamp("2026-04-01 00:15", tz="UTC")
    new_w = equal_factor_weights()
    sched.add(WeightVersion(
        version_id="init_x",
        weights=new_w,
        calculated_at=str(t0),
        effective_from=str(t1),
        phase="init",
        update_number=0,
    ))
    assert sched.active_at(t0).weights == seed
    assert sched.active_at(t1).weights == new_w


def test_adaptive_learning_only_matured_outcomes():
    rows = [
        {"timestamp": "2026-01-01T00:00:00+00:00", "future_return_4h": 0.01, "factor_momentum": 0.6},
        {"timestamp": "2026-01-01T04:00:00+00:00", "future_return_4h": -0.01, "factor_momentum": 0.4},
        {"timestamp": "2026-01-01T03:00:00+00:00", "future_return_4h": 0.02, "factor_momentum": 0.5},
    ]
    asof = "2026-01-01T04:00:00+00:00"
    matured = filter_matured_for_learning(rows, asof_ts=asof, horizon_hours=4)
    assert len(matured) == 1
    assert str(matured.iloc[0]["timestamp"]).startswith("2026-01-01")
    assert outcome_mature_at("2026-01-01T00:00:00+00:00", horizon_hours=4, asof_ts=asof)
    assert not outcome_mature_at("2026-01-01T03:00:00+00:00", horizon_hours=4, asof_ts=asof)


def test_threshold_fixed_060_in_sim_config():
    from btcc.sim.config import load_sim_config
    sim = load_sim_config()
    assert float(sim["long_threshold"]) == 0.60


def test_execution_costs_identical_config():
    from btcc.sim.config import load_sim_config
    sim = load_sim_config()
    assert float(sim["fee_rate_per_side"]) == 0.001
    assert float(sim["slippage_rate_per_side"]) == 0.0005
    assert sim.get("same_candle_conflict") == "assume_sl_first"
    assert int(sim["max_open_opportunities"]) == 10


def test_allow_trading_false():
    cfg = load_config()
    assert cfg["safety"]["allow_trading"] is False


def test_walk_forward_not_four_folds():
    from btcc.sim.config import load_sim_config
    wf = (load_sim_config().get("walk_forward") or {})
    assert int(wf["init_days"]) == 90
    assert int(wf["daily_rolling_window_days"]) == 90
    assert "folds" not in wf
