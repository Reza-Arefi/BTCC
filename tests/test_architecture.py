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
    audit = build_universe_audit(cfg, {}, set())
    assert len(audit) == 20
    assert all(r["btc_relative_construction"].endswith("/ BTCUSDT") for r in audit)


def test_no_order_symbols_in_package():
    """Ensure package source does not define trading order helpers."""
    root = ROOT / "btcc"
    bad = []
    for p in root.rglob("*.py"):
        text = p.read_text(encoding="utf-8")
        for tok in ("create_order", "place_order", "cancel_order"):
            # allow mentions inside safety deny list / comments about blocking
            if tok in text and "deny" not in text.lower() and "forbidden" not in text.lower() and "block" not in text.lower():
                # safety module lists them — skip
                if "no_trading" in str(p):
                    continue
                bad.append((str(p), tok))
    assert bad == [], bad
