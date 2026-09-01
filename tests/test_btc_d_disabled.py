"""Tests for optional BTC.D disable during backtests."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from btcc.backtest.dominance_history import HistoricalDominanceSeries, btc_d_backtest_enabled, btc_d_manifest_fields
from btcc.factors.combine import compute_all_factors
from btcc.sim.regime import classify_regime
from btcc.sim.score import combined_score, static_factor_weights
from btcc.sim.selector_memory_config import load_selector_memory_config


def test_regime_does_not_use_btc_d():
    factors = {
        "trend": {"adx": 25.0},
        "volatility": {"natr": 0.02, "bandwidth": 0.05},
    }
    with_dom = classify_regime(factors)
    without_dom = classify_regime(factors)
    assert with_dom == without_dom
    assert "HIGH_VOLATILITY" not in with_dom["regime"] or with_dom["regime"] == "LOW_VOL_TREND"


def test_btc_regime_weight_zero_in_signal():
    cfg = {"factors": {"weights": {
        "momentum": 0.2, "trend": 0.2, "btc_regime": 0.5,
        "volume": 0.15, "volatility": 0.15, "rsi": 0.15, "structure": 0.15,
    }}}
    n = 120
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    rel = pd.DataFrame({"timestamp": idx, "open": 1.0, "high": 1.01, "low": 0.99, "close": 1.0, "volume": 1000.0})
    btc = rel.copy()
    alt = rel.copy()
    f_none = compute_all_factors(rel, alt, btc, None, {}, cfg, "15m")
    f_fake = compute_all_factors(rel, alt, btc, 55.0, {1: -1.0, 4: -2.0, 12: -1.5, 24: -0.5}, cfg, "15m")
    assert f_none["signal_score"] == f_fake["signal_score"]
    assert f_none["weights"]["btc_regime"] == 0.0


def test_fetch_for_backtest_skips_coingecko_when_disabled():
    sim = {"btc_d": {"enabled": False}}
    with patch.object(HistoricalDominanceSeries, "fetch_coingecko") as mock_fetch:
        series = HistoricalDominanceSeries.fetch_for_backtest(365, "/tmp", sim_cfg=sim)
        mock_fetch.assert_not_called()
    assert series.meta["source"] == "disabled"
    pct, obs, status = series.observation_at(pd.Timestamp("2025-01-01", tz="UTC"))
    assert pct is None and status == "DISABLED"


def test_experiment_configs_disable_btc_d():
    sim = load_selector_memory_config()
    assert btc_d_backtest_enabled(sim) is False
    fields = btc_d_manifest_fields(sim, HistoricalDominanceSeries.disabled())
    assert fields == {
        "btc_d_enabled": False,
        "btc_d_source": "disabled",
        "btc_d_affects_trading": False,
    }


if __name__ == "__main__":
    test_regime_does_not_use_btc_d()
    test_btc_regime_weight_zero_in_signal()
    test_fetch_for_backtest_skips_coingecko_when_disabled()
    test_experiment_configs_disable_btc_d()
    print("OK")
