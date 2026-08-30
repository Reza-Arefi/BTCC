"""Unit tests for Adaptive V2 — score, SM, accounting, exits, weights."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from btcc.sim.accounting import CostModel, close_long_alt_btc, open_long_alt_btc
from btcc.sim.exits import StrategySpec, open_opportunity_legs, process_bar_on_leg
from btcc.sim.score import combined_score, normalize_weights, signed_from_score
from btcc.sim.state_machine import CrossingStateMachine, PairSignalState
from btcc.sim.weights import estimate_weights_from_window


def test_signed_mapping():
    assert signed_from_score(0.0) == -1.0
    assert signed_from_score(0.5) == 0.0
    assert signed_from_score(1.0) == 1.0


def test_combined_score_bounds():
    w = normalize_weights({
        "momentum": 0.2, "trend": 0.2, "btc_regime": 0.2,
        "volume": 0.1, "volatility": 0.1, "rsi": 0.1, "structure": 0.1,
    })
    scores = {k: 1.0 for k in w}
    out = combined_score(scores, w)
    assert abs(out["S"] - 1.0) < 1e-9
    scores = {k: 0.0 for k in w}
    out = combined_score(scores, w)
    assert abs(out["S"] + 1.0) < 1e-9
    scores = {k: 0.5 for k in w}
    out = combined_score(scores, w)
    assert abs(out["S"]) < 1e-9


def test_crossing_state_machine_no_duplicates():
    sm = CrossingStateMachine(long_threshold=0.60, max_open=10)
    d1 = sm.evaluate("ETHUSDT", 0.40)
    assert d1["rejection_reason"] == "BELOW_THRESHOLD"
    d2 = sm.evaluate("ETHUSDT", 0.72)
    assert d2["crossed_into"] and d2["trade_suggested"]
    sm.register_open("ETHUSDT", "opp1")
    d3 = sm.evaluate("ETHUSDT", 0.75)
    assert d3["still_in_zone"] and not d3["trade_suggested"]
    assert d3["rejection_reason"] == "SIGNAL_CONTINUATION"
    d4 = sm.evaluate("ETHUSDT", 0.40)
    assert d4["crossed_out"]
    d5 = sm.evaluate("ETHUSDT", 0.70)
    assert d5["crossed_into"] and not d5["trade_suggested"]
    assert d5["rejection_reason"] == "SAME_PAIR_ALREADY_OPEN"
    sm.register_close("opp1", "ETHUSDT")
    d6 = sm.evaluate("ETHUSDT", 0.40)
    d7 = sm.evaluate("ETHUSDT", 0.71)
    assert d7["trade_suggested"]


def test_max_open_trades():
    sm = CrossingStateMachine(long_threshold=0.60, max_open=2)
    for i, sym in enumerate(["A", "B"]):
        d = sm.evaluate(sym, 0.7)
        assert d["trade_suggested"]
        sm.register_open(sym, f"opp{i}")
        sm.evaluate(sym, 0.4)  # leave zone so next pair logic independent
    d = sm.evaluate("C", 0.8)
    assert d["rejection_reason"] == "MAX_OPEN_TRADES"


def test_btc_accounting_long_alt_btc():
    costs = CostModel(0.001, 0.0005)
    pos = open_long_alt_btc(alt_btc_mid=0.05, btc_usdt=100000, notional_usd=100, costs=costs)
    assert pos["entry_btc_spent"] == pytest.approx(0.001)
    assert pos["alt_qty"] > 0
    # ALT/BTC rises 3%
    res = close_long_alt_btc(position=pos, alt_btc_mid=0.05 * 1.03, btc_usdt=100000, costs=costs)
    assert res["pnl_btc"] > 0
    assert res["fees_btc"] > 0


def test_same_candle_sl_first():
    costs = CostModel(0.001, 0.0005)
    specs = [StrategySpec("strategy_1", "S1", stop_loss_pct=0.01, take_profit_pct=0.015)]
    legs = open_opportunity_legs(
        alt_btc_entry_mid=100.0,
        btc_usdt=50000,
        notional_usd=100,
        costs=costs,
        specs=specs,
        entry_ts=datetime.now(timezone.utc),
    )
    leg = legs[0]
    fill = leg.position["entry_fill_price"]
    # Bar that touches both SL and TP
    bar = {
        "timestamp": datetime.now(timezone.utc),
        "open": fill,
        "high": fill * 1.02,
        "low": fill * 0.98,
        "close": fill,
    }
    closed = process_bar_on_leg(leg, bar=bar, btc_usdt=50000, costs=costs)
    assert closed
    assert leg.exit_reason == "STOP_LOSS_SAME_CANDLE_CONFLICT"


def test_trailing_stop_ratchet():
    costs = CostModel(0.0, 0.0)
    specs = [StrategySpec(
        "strategy_3", "S3", stop_loss_pct=0.01,
        trail_activation_pct=0.01, trail_distance_pct=0.005,
    )]
    legs = open_opportunity_legs(
        alt_btc_entry_mid=100.0, btc_usdt=50000, notional_usd=100,
        costs=costs, specs=specs, entry_ts="t0",
    )
    leg = legs[0]
    fill = leg.position["entry_fill_price"]
    # Activate trailing
    process_bar_on_leg(leg, bar={
        "timestamp": "t1", "open": fill, "high": fill * 1.011, "low": fill, "close": fill * 1.01,
    }, btc_usdt=50000, costs=costs)
    assert leg.trailing_active
    stop1 = leg.trailing_stop
    # Higher high → stop rises
    process_bar_on_leg(leg, bar={
        "timestamp": "t2", "open": fill * 1.01, "high": fill * 1.02, "low": fill * 1.01, "close": fill * 1.015,
    }, btc_usdt=50000, costs=costs)
    assert leg.trailing_stop >= stop1
    # Pullback to trailing stop
    process_bar_on_leg(leg, bar={
        "timestamp": "t3", "open": fill * 1.015, "high": fill * 1.015,
        "low": leg.trailing_stop - 0.01, "close": leg.trailing_stop - 0.01,
    }, btc_usdt=50000, costs=costs)
    assert leg.closed
    assert leg.exit_reason == "TRAILING_STOP"


def test_weight_update_safeguards_insufficient():
    prev = normalize_weights({
        "momentum": 0.2, "trend": 0.2, "btc_regime": 0.2,
        "volume": 0.1, "volatility": 0.1, "rsi": 0.1, "structure": 0.1,
    })
    new_w, stats = estimate_weights_from_window(
        pd.DataFrame(),
        prev,
        half_life_days=45,
        weight_min=0.05,
        weight_max=0.40,
        min_obs_total=200,
        min_obs_per_factor=50,
        stability_blend=0.25,
    )
    assert new_w is None
    assert stats["status"] == "FAILED"


def test_weight_update_learns_from_signed_factors():
    prev = normalize_weights({
        "momentum": 0.2, "trend": 0.2, "btc_regime": 0.2,
        "volume": 0.1, "volatility": 0.1, "rsi": 0.1, "structure": 0.1,
    })
    rows = []
    base = pd.Timestamp("2026-01-01", tz="UTC")
    for i in range(300):
        # momentum high → positive future return
        mom = 0.8 if i % 2 == 0 else 0.2
        ret = 0.01 if mom > 0.5 else -0.01
        rows.append({
            "timestamp": base + pd.Timedelta(hours=i),
            "factor_momentum": mom,
            "factor_trend": 0.5,
            "factor_btc_regime": 0.5,
            "factor_volume": 0.5,
            "factor_volatility": 0.5,
            "factor_rsi": 0.5,
            "factor_structure": 0.5,
            "future_return_4h": ret,
        })
    new_w, stats = estimate_weights_from_window(
        pd.DataFrame(rows),
        prev,
        half_life_days=45,
        weight_min=0.05,
        weight_max=0.40,
        min_obs_total=200,
        min_obs_per_factor=50,
        stability_blend=1.0,  # full new estimate for test
    )
    assert new_w is not None
    assert stats["status"] == "OK"
    assert new_w["momentum"] > new_w["trend"]
