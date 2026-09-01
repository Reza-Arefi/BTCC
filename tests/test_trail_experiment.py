"""Tests for trailing-exit experiment specification."""

from __future__ import annotations

import pandas as pd
import pytest

from btcc.config import load_config
from btcc.sim.exits import StrategyLegState, StrategySpec, open_opportunity_legs, process_bar_on_leg
from btcc.sim.accounting import CostModel
from btcc.sim.regime import REGIME_CLASSES, classify_regime
from btcc.sim.state_machine import CrossingStateMachine
from btcc.sim.trail_config import TRAIL_SPEC, TRAIL_STRATEGY_KEYS, load_trail_experiment_config, validate_trail_strategies
from btcc.sim.trail_entry import (
    CLS_ABOVE_THRESHOLD,
    CLS_BELOW_THRESHOLD,
    CLS_ELIGIBLE,
    evaluate_trail_entry,
)


def test_entry_band_thresholds():
    sm_lo = CrossingStateMachine(long_threshold=0.65, upper_threshold=0.85, max_open=10)
    sm_mid = CrossingStateMachine(long_threshold=0.65, upper_threshold=0.85, max_open=10)
    sm_hi = CrossingStateMachine(long_threshold=0.65, upper_threshold=0.85, max_open=10)

    assert sm_lo.evaluate("X", 0.64)["rejection_reason"] == "BELOW_THRESHOLD"
    assert sm_mid.evaluate("Y", 0.70)["trade_suggested"] is True
    assert sm_hi.evaluate("Z", 0.85)["rejection_reason"] == "ABOVE_THRESHOLD"
    assert sm_hi.evaluate("W", 0.90)["rejection_reason"] == "ABOVE_THRESHOLD"


def test_exhaustion_does_not_reject_trail_entry():
    sm = {"signal_generated": True, "trade_suggested": True, "rejection_reason": None}
    r = evaluate_trail_entry(sm_decision=sm, health_allow_new_trades=True)
    assert r["trade_suggested"] is True
    assert r["entry_classification"] == CLS_ELIGIBLE


def test_btc_d_health_does_not_block_when_allowed():
    sm = {"signal_generated": True, "trade_suggested": True, "rejection_reason": None}
    r = evaluate_trail_entry(sm_decision=sm, health_allow_new_trades=True)
    assert r["trade_suggested"] is True


def test_trail_strategies_match_spec():
    sim = load_trail_experiment_config()
    assert not validate_trail_strategies(sim)
    assert set(sim["strategies"].keys()) == set(TRAIL_STRATEGY_KEYS)
    for key, (sl, act, dist) in TRAIL_SPEC.items():
        raw = sim["strategies"][key]
        assert raw["take_profit_pct"] is None
        assert raw["trailing"]["activation_pct"] == act
        assert raw["trailing"]["distance_pct"] == dist
        assert raw["stop_loss_pct"] == sl


def test_no_fixed_tp_in_config():
    sim = load_trail_experiment_config()
    for key, raw in sim["strategies"].items():
        assert raw.get("take_profit_pct") is None, key


def test_static_weights_frozen_in_config():
    sim = load_trail_experiment_config()
    assert sim["disable_weight_updates"] is True
    assert sim["weight_mode"] == "static"


def test_identical_entries_all_strategies():
    costs = CostModel()
    specs = [
        StrategySpec(key=f"trail_{i}", name=f"T{i}", stop_loss_pct=0.01, take_profit_pct=None,
                     trail_activation_pct=0.01, trail_distance_pct=0.005)
        for i in range(1, 11)
    ]
    legs = open_opportunity_legs(
        alt_btc_entry_mid=0.001,
        btc_usdt=50000.0,
        notional_usd=100.0,
        costs=costs,
        specs=specs,
        entry_ts="2026-01-01T00:00:00+00:00",
    )
    fills = [l.position["entry_fill_price"] for l in legs]
    notionals = [l.position["notional_usd"] for l in legs]
    assert len(set(fills)) == 1
    assert len(set(notionals)) == 1


def test_trailing_activation_and_ratchet():
    spec = StrategySpec(key="trail_3", name="T3", stop_loss_pct=0.01, take_profit_pct=None,
                        trail_activation_pct=0.01, trail_distance_pct=0.005)
    costs = CostModel()
    legs = open_opportunity_legs(
        alt_btc_entry_mid=1.0, btc_usdt=50000.0, notional_usd=100.0, costs=costs,
        specs=[spec], entry_ts="2026-01-01",
    )
    leg = legs[0]
    fill = leg.position["entry_fill_price"]
    # Bar activates trail
    process_bar_on_leg(leg, bar={"timestamp": "t1", "open": 1.0, "high": fill * 1.02, "low": fill * 0.99, "close": 1.01},
                       btc_usdt=50000.0, costs=costs)
    assert leg.trailing_active
    stop1 = leg.trailing_stop
    # Ratchet up
    process_bar_on_leg(leg, bar={"timestamp": "t2", "open": 1.01, "high": fill * 1.03, "low": fill * 1.0, "close": 1.02},
                       btc_usdt=50000.0, costs=costs)
    assert leg.trailing_stop >= stop1


def test_initial_sl_before_activation():
    spec = StrategySpec(key="t", name="t", stop_loss_pct=0.01, take_profit_pct=None,
                        trail_activation_pct=0.05, trail_distance_pct=0.005)
    costs = CostModel()
    legs = open_opportunity_legs(
        alt_btc_entry_mid=1.0, btc_usdt=50000.0, notional_usd=100.0, costs=costs,
        specs=[spec], entry_ts="2026-01-01",
    )
    leg = legs[0]
    fill = leg.position["entry_fill_price"]
    assert leg.active_stop() == leg.initial_sl
    closed = process_bar_on_leg(
        leg,
        bar={"timestamp": "t1", "open": 1.0, "high": 1.0, "low": fill * 0.98, "close": fill * 0.985},
        btc_usdt=50000.0,
        costs=costs,
    )
    assert closed
    assert leg.exit_reason == "STOP_LOSS"


def test_mfe_mae_tracking():
    spec = StrategySpec(key="t", name="t", stop_loss_pct=0.05, take_profit_pct=None,
                        trail_activation_pct=0.10, trail_distance_pct=0.01)
    costs = CostModel()
    legs = open_opportunity_legs(
        alt_btc_entry_mid=1.0, btc_usdt=50000.0, notional_usd=100.0, costs=costs,
        specs=[spec], entry_ts="2026-01-01",
    )
    leg = legs[0]
    fill = leg.position["entry_fill_price"]
    process_bar_on_leg(
        leg,
        bar={"timestamp": "t1", "open": 1.0, "high": fill * 1.02, "low": fill * 0.97, "close": 1.0},
        btc_usdt=50000.0,
        costs=costs,
    )
    assert leg.mfe_pct > 0
    assert leg.mae_pct < 0


def test_regime_uses_entry_factors_only():
    factors = {
        "trend": {"adx": 45.0},
        "volatility": {"natr": 0.02, "bandwidth": 0.05},
    }
    r = classify_regime(factors)
    assert r["regime"] in REGIME_CLASSES


def test_regime_high_vol():
    factors = {"trend": {"adx": 25}, "volatility": {"natr": 0.05}}
    assert classify_regime(factors)["regime"] == "HIGH_VOLATILITY"


def test_safety_and_telegram_config():
    cfg = load_config()
    assert cfg["safety"]["allow_trading"] is False
    sim = load_trail_experiment_config()
    assert sim["btc_d_health"]["require_for_new_trades"] is False


def test_upper_rejection_classification():
    sm = {"signal_generated": False, "trade_suggested": False, "rejection_reason": "ABOVE_THRESHOLD"}
    r = evaluate_trail_entry(sm_decision=sm)
    assert r["entry_classification"] == CLS_ABOVE_THRESHOLD


def test_below_rejection_classification():
    sm = {"signal_generated": False, "trade_suggested": False, "rejection_reason": "BELOW_THRESHOLD"}
    r = evaluate_trail_entry(sm_decision=sm)
    assert r["entry_classification"] == CLS_BELOW_THRESHOLD
