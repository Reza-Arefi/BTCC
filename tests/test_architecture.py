"""Smoke tests — architecture + safety (no live network required for unit parts)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from btcc.safety.no_trading import TradingForbiddenError, assert_no_trading_config, deny_trading
from btcc.series.relative import build_alt_btc
from btcc.factors.combine import compute_all_factors
from btcc.late_entry.score import late_entry_score
from btcc.probability.score import (
    classify_signal,
    horizons_probabilities,
    score_to_baseline_model_probability,
    score_to_probability,
)
from btcc.ranking.ranker import rank_signals
from btcc.config import load_config
from btcc.universe import build_universe_audit, resolve_btc_market


def _ohlcv(n=300, seed=0, drift=0.0002):
    rng = np.random.default_rng(seed)
    px = 100 * np.cumprod(1 + drift + rng.normal(0, 0.002, n))
    ts = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts,
        "open": px,
        "high": px * 1.002,
        "low": px * 0.998,
        "close": px,
        "volume": rng.uniform(1000, 5000, n),
    })


def test_trading_forbidden():
    with pytest.raises(TradingForbiddenError):
        deny_trading()
    cfg = load_config()
    assert_no_trading_config(cfg)
    with pytest.raises(TradingForbiddenError):
        assert_no_trading_config({"safety": {"allow_trading": True}})


def test_alt_btc_relative_definition():
    alt = _ohlcv(seed=1, drift=0.0001)
    btc = _ohlcv(seed=2, drift=0.0005)  # BTC rises faster → ALT/BTC falls
    rel = build_alt_btc(alt, btc)
    assert rel is not None
    # ALT underperformed BTC in USDT → relative series should trend down
    assert rel["close"].iloc[-1] < rel["close"].iloc[0]


def test_score_is_not_raw_probability():
    cfg = load_config()
    p = score_to_baseline_model_probability(0.78, cfg, 4)
    assert 0.01 <= p <= 0.99
    # Must not literally equal 0.78
    assert abs(p - 0.78) > 1e-6
    # Alias still works
    assert abs(score_to_probability(0.78, cfg, 4) - p) < 1e-12


def test_factors_and_late_entry_and_rank():
    cfg = load_config()
    alt = _ohlcv(seed=3, drift=0.0004)
    btc = _ohlcv(seed=4, drift=0.0001)
    rel = build_alt_btc(alt, btc)
    factors = compute_all_factors(rel, alt, btc, 55.0, {1: -0.1, 4: -0.3, 12: -0.2, 24: -0.5}, cfg)
    assert 0 <= factors["signal_score"] <= 1
    for k in ("momentum", "trend", "btc_regime", "volume", "volatility", "rsi", "structure"):
        assert 0 <= factors[k]["score"] <= 1
    late = late_entry_score(rel, factors, cfg)
    assert 0 <= late["late_entry_score"] <= 1
    assert late["classification"] in {
        "NORMAL", "EXTENDED", "HIGH_LATE_ENTRY_RISK", "VERY_HIGH_LATE_ENTRY_RISK"
    }
    assert "top_reasons" in late and len(late["top_reasons"]) >= 1
    probs = horizons_probabilities(factors["signal_score"], cfg)
    assert "p_4h" in probs
    assert probs["probability_kind"] == "baseline_model_probability"
    assert probs["p_4h_status"] == "baseline_model_probability"
    assert probs["calibration_applied"] is False
    sc = classify_signal(0.82, 0.21, cfg)
    assert sc["signal_class"] == "STRONG_SIGNAL"
    sc_late = classify_signal(0.82, 0.84, cfg)
    assert sc_late["signal_class"] == "STRONG_BUT_LATE"
    sc_weak = classify_signal(0.57, 0.15, cfg)
    assert sc_weak["signal_class"] == "WEAK_SIGNAL"
    rows = [
        {"base": "AAA", "p_4h": 0.6, "p_1h": 0.5},
        {"base": "BBB", "p_4h": 0.8, "p_1h": 0.5},
    ]
    ranked = rank_signals(rows, 4)
    assert ranked[0]["base"] == "BBB"
    assert ranked[0]["rank"] == 1


def test_universe_exact_20_matches_research():
    cfg = load_config()
    assert len(cfg["universe"]["bases"]) == 20
    assert resolve_btc_market("ETH", cfg) == "ETHBTC"
    assert resolve_btc_market("CKBTC", cfg) == "CKBTCBTC"
    assert resolve_btc_market("A", cfg) == "ABTC"
    from btcc.universe import resolve_data_market, uses_native_btc_market
    assert uses_native_btc_market("CKBTC", cfg) is True
    ck = resolve_data_market("CKBTC", cfg)
    assert ck["logical_pair"] == "CKBTC"
    assert ck["resolved_market"] == "CKBTCBTC"
    assert ck["mode"] == "native_btc"
    assert "CKBTCUSDT" not in ck["resolved_market"]
    eth = resolve_data_market("ETH", cfg)
    assert eth["resolved_market"] == "ETHUSDT"
    assert eth["mode"] == "synthetic_usdt"
    audit = build_universe_audit(cfg, {}, set())
    assert len(audit) == 20
    ck_row = next(r for r in audit if r["symbol"] == "CKBTC")
    assert ck_row["resolved_market"] == "CKBTCBTC"
    assert ck_row["btc_relative_construction"].startswith("native ")
    eth_row = next(r for r in audit if r["symbol"] == "ETH")
    assert eth_row["btc_relative_construction"].endswith("/ BTCUSDT")


def test_ckbtc_never_resolves_to_usdt_fetch():
    """Regression: CKBTC must resolve to CKBTCBTC before any API request."""
    from btcc.universe import resolve_data_market, resolve_btc_market
    from unittest.mock import MagicMock, patch

    cfg = load_config()
    assert resolve_btc_market("CKBTC", cfg) == "CKBTCBTC"
    meta = resolve_data_market("CKBTC", cfg)
    assert meta["resolved_market"] == "CKBTCBTC"
    assert meta["logical_pair"] == "CKBTC"

    # download_panels must bootstrap CKBTCBTC only — never CKBTCUSDT
    from btcc.backtest.data_loader import download_panels

    calls: list[str] = []

    def fake_bootstrap(symbol, interval, lookback, candle_dir, force=False):
        calls.append(symbol)
        if symbol == "CKBTCUSDT":
            raise AssertionError("CKBTCUSDT must not be requested")
        import pandas as pd
        n = 150
        ts = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
        px = pd.Series(range(n), dtype=float) + 1.0
        return pd.DataFrame({
            "timestamp": ts,
            "open": px, "high": px * 1.01, "low": px * 0.99, "close": px, "volume": 1.0,
        })

    cfg_bt = dict(cfg)
    cfg_bt["backtest"] = {"interval": "15m", "min_warmup_bars": 200}
    cfg_bt["backtest_data"] = {"candle_dir": "/tmp/btcc_test_candles"}
    cfg_bt["data"] = {**cfg["data"], "mexc_rest": "https://api.mexc.com"}

    with patch("btcc.backtest.data_loader.MexcPublicREST") as MockREST:
        inst = MockREST.return_value
        inst.bootstrap_symbol.side_effect = fake_bootstrap
        panels = download_panels(cfg_bt, days=2, warmup_bars=200, force=False)

    assert "CKBTCUSDT" not in calls
    assert "CKBTCBTC" in calls
    assert "CKBTC" in panels["coins"]
    coin = panels["coins"]["CKBTC"]
    assert coin["logical_pair"] == "CKBTC"
    assert coin["resolved_market"] == "CKBTCBTC"



def test_btc_d_max_age_is_7200():
    from btcc.sim.config import load_sim_config
    sim = load_sim_config()
    assert int((sim.get("btc_d_health") or {}).get("max_age_seconds")) == 7200


def test_btc_d_does_not_require_for_new_trades():
    from btcc.sim.config import load_sim_config
    from btcc.sim.health import evaluate_health
    from datetime import datetime, timezone, timedelta

    sim = load_sim_config()
    assert (sim.get("btc_d_health") or {}).get("require_for_new_trades") is False
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    # Stale BTC.D (age >> 7200) must NOT block trades
    h = evaluate_health(
        sim_cfg=sim,
        dominance_pct=55.0,
        dominance_ts=now - timedelta(hours=6),
        dominance_source="test",
        decision_candle_ts=now,
        now=now,
        n_relative_bars=500,
        indicator_ok=True,
    )
    assert h.btc_d_status == "STALE"
    assert h.btc_d_available is False
    assert h.allow_new_trades is True
    assert any("DIAG_BTC_D_STALE" in r or "BTC_D_STALE" in r for r in h.reasons)


def test_btc_regime_weight_zero_in_score():
    from btcc.sim.score import (
        ACTIVE_SIGNAL_KEYS,
        combined_score,
        equal_factor_weights,
        static_factor_weights,
    )
    from btcc.config import load_config

    cfg = load_config()
    sw = static_factor_weights(cfg)
    ew = equal_factor_weights()
    assert sw["btc_regime"] == 0.0
    assert ew["btc_regime"] == 0.0
    assert abs(sum(sw[k] for k in ACTIVE_SIGNAL_KEYS) - 1.0) < 1e-9
    factors = {k: 0.8 for k in ACTIVE_SIGNAL_KEYS}
    factors["btc_regime"] = 1.0  # would pull S up if weighted
    s0 = combined_score(factors, sw)["S"]
    factors["btc_regime"] = 0.0
    s1 = combined_score(factors, sw)["S"]
    assert abs(s0 - s1) < 1e-12


def test_capital_uses_pnl_usd_equiv():
    import pandas as pd
    from btcc.analytics.capital import capital_daily_series

    legs = pd.DataFrame([
        {
            "entry_policy": "NORMAL_FILTERED",
            "strategy_key": "strategy_1",
            "closed": True,
            "pnl_btc": 0.001,
            "pnl_usd_equiv": 12.5,
            "exit_ts": "2026-01-02",
            "day_number": 1,
        },
    ])
    d = capital_daily_series(legs, starting_capital_usd=1000.0, max_day=1)
    assert abs(float(d.iloc[0]["ending_value"]) - 1012.5) < 1e-6


def test_verify_btc_d_parity_uses_config_max_age():
    """Parity report must not hardcode 1800; uses sim_config 7200."""
    import scripts.verify_btc_d_parity as mod
    from btcc.sim.config import load_sim_config
    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert 'stale_max_age_seconds"] = 1800' not in src
    assert "load_sim_config" in src
    assert int((load_sim_config().get("btc_d_health") or {})["max_age_seconds"]) == 7200


def test_bootstrap_does_not_truncate_disk_cache(tmp_path):
    """Short lookback must not destroy longer on-disk candle history."""
    import pandas as pd
    from btcc.data.candles import load_candles, save_candles, candle_path
    from btcc.data.websocket import MexcPublicREST

    candle_dir = tmp_path / "candles"
    interval = "15m"
    symbol = "BTCUSDT"
    # Seed ~200 bars of history
    idx = pd.date_range("2025-01-01", periods=200, freq="15min", tz="UTC")
    seeded = pd.DataFrame({
        "timestamp": idx,
        "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 10.0,
    })
    save_candles(seeded, candle_path(str(candle_dir), symbol, interval))

    rest = MexcPublicREST("https://api.mexc.com")
    # Stub network: probe ok, no new range fetches
    rest.fetch_klines = lambda *a, **k: seeded.tail(5).copy()  # type: ignore
    rest._fetch_range = lambda *a, **k: []  # type: ignore

    out = rest.bootstrap_symbol(symbol, interval, lookback_bars=50, candle_dir=str(candle_dir), force=True)
    assert out is not None
    disk = load_candles(candle_path(str(candle_dir), symbol, interval))
    assert disk is not None
    assert len(disk) >= 200, f"disk truncated to {len(disk)}"


def test_no_order_symbols_in_package():
    """Ensure package source does not define trading order helpers."""
    root = ROOT / "btcc"
    bad = []
    for p in root.rglob("*.py"):
        text = p.read_text(encoding="utf-8")
        for tok in ("create_order", "place_order", "cancel_order"):
            if tok in text and "deny" not in text.lower() and "forbidden" not in text.lower() and "block" not in text.lower():
                if "no_trading" in str(p):
                    continue
                bad.append((str(p), tok))
    assert bad == [], bad
