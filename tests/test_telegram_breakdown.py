"""Tests for Telegram indicator breakdown message."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from btcc.config import load_config
from btcc.factors.combine import compute_all_factors
from btcc.late_entry.score import late_entry_score
from btcc.probability.score import classify_signal, horizons_probabilities
from btcc.ranking.ranker import rank_signals
from btcc.safety.no_trading import TradingForbiddenError, deny_trading
from btcc.series.relative import build_alt_btc
from btcc.telegram.notifier import TelegramNotifier, COIN_SEP, SEP


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


def _build_ranked_rows(cfg, n_coins=3):
    rows = []
    for seed in range(n_coins):
        alt = _ohlcv(seed=seed + 10, drift=0.0001 + seed * 0.0002)
        btc = _ohlcv(seed=99, drift=0.00005)
        rel = build_alt_btc(alt, btc)
        factors = compute_all_factors(rel, alt, btc, 55.0, {1: -0.1, 4: -0.3, 12: -0.2, 24: -0.5}, cfg)
        probs = horizons_probabilities(factors["signal_score"], cfg)
        late = late_entry_score(rel, factors, cfg)
        sc = classify_signal(probs["p_4h"], late["late_entry_score"], cfg)
        rows.append({
            "base": f"COIN{seed}",
            "symbol": f"COIN{seed}USDT",
            "p_1h": probs["p_1h"],
            "p_4h": probs["p_4h"],
            "p_8h": probs["p_8h"],
            "p_12h": probs["p_12h"],
            "p_24h": probs["p_24h"],
            "signal_score": factors["signal_score"],
            "factors": factors,
            "late_entry": late,
            "signal_class": sc,
            "probability_kind": probs["probability_kind"],
        })
    return rank_signals(rows, 4)


def test_breakdown_uses_same_top5_as_ranking():
    cfg = load_config()
    ranked = _build_ranked_rows(cfg, 5)
    tg = TelegramNotifier("", "", enabled=False)
    ranking_msg = tg.format_ranking(ranked, {}, top_n=5)
    breakdown_msg = tg.format_indicator_breakdown(ranked, top_n=5)

    ranking_bases = []
    for line in ranking_msg.splitlines():
        for medal in ("🥇", "🥈", "🥉", "4.", "5."):
            if line.startswith(medal):
                ranking_bases.append(line.split()[1].split("/")[0])
    breakdown_bases = []
    for line in breakdown_msg.splitlines():
        for medal in ("🥇", "🥈", "🥉", "4.", "5."):
            if line.startswith(medal):
                breakdown_bases.append(line.split()[1].split("/")[0])

    assert len(ranking_bases) == 5
    assert breakdown_bases == ranking_bases


def test_breakdown_contains_all_indicators_and_groups():
    cfg = load_config()
    ranked = _build_ranked_rows(cfg, 3)
    tg = TelegramNotifier("", "", enabled=False)
    msg = tg.format_indicator_breakdown(ranked, top_n=3)

    assert "📊 INDICATOR BREAKDOWN — TOP 5" in msg
    for header in (
        "MOMENTUM", "TREND", "BTC REGIME", "VOLUME", "VOLATILITY", "RSI", "STRUCTURE",
    ):
        assert header in msg
    for label in (
        "Momentum", "EMA", "MACD", "Ichimoku", "ADX",
        "BTC Regime", "BTC Dominance", "RVOL",
        "Bollinger", "ATR", "NATR", "Price/Market",
    ):
        assert label in msg
    assert SEP in msg
    assert COIN_SEP in msg
    assert "4h Probability ⭐" in msg
    assert "Late Entry" in msg
    assert "SIGNAL ONLY" in msg


def test_breakdown_missing_indicator_displayed_safely():
    tg = TelegramNotifier("", "", enabled=False)
    row = {
        "base": "TEST",
        "p_4h": 0.75,
        "factors": {
            "momentum": {"score": 0.62},
            "trend": {"ema_score": 0.9, "macd_score": 0.82, "ichimoku_score": 0.91, "adx_score": 0.76},
            "btc_regime": {"regime_indicator_score": 0.72, "dominance_indicator_score": None},
            "volume": {"rvol_score": None},
            "volatility": {"bollinger_score": 0.65, "atr_score": None, "natr_score": 0.66},
            "rsi": {"score": 0.75},
            "structure": {"score": 0.90},
        },
        "late_entry": {"late_entry_score": 0.62, "classification": "HIGH_LATE_ENTRY_RISK"},
    }
    msg = tg.format_indicator_breakdown([row], top_n=1)
    assert "BTC Dominance   INSUFFICIENT_DATA" in msg
    assert "RVOL            INSUFFICIENT_DATA" in msg
    assert "ATR             INSUFFICIENT_DATA" in msg
    assert "Momentum        0.62" in msg


def test_probability_unchanged_by_breakdown():
    cfg = load_config()
    alt = _ohlcv(seed=3, drift=0.0004)
    btc = _ohlcv(seed=4, drift=0.0001)
    rel = build_alt_btc(alt, btc)
    factors = compute_all_factors(rel, alt, btc, 55.0, {1: -0.1, 4: -0.3, 12: -0.2, 24: -0.5}, cfg)
    probs_before = horizons_probabilities(factors["signal_score"], cfg)
    # Formatting breakdown does not mutate inputs
    row = {
        "base": "X",
        "p_4h": probs_before["p_4h"],
        "factors": factors,
        "late_entry": {"late_entry_score": 0.3, "classification": "NORMAL"},
    }
    tg = TelegramNotifier("", "", enabled=False)
    tg.format_indicator_breakdown([row])
    probs_after = horizons_probabilities(factors["signal_score"], cfg)
    assert probs_before["p_4h"] == probs_after["p_4h"]
    assert probs_before["signal_score"] if "signal_score" in probs_before else factors["signal_score"] == factors["signal_score"]


def test_no_trading_on_breakdown_send():
    with pytest.raises(TradingForbiddenError):
        deny_trading()
    tg = TelegramNotifier("", "", enabled=False)
    ok1, ok2 = tg.send_ranking_and_breakdown([], {}, top_n=5)
    assert ok1 is False
    assert ok2 is False


def test_factor_subscores_exposed():
    cfg = load_config()
    alt = _ohlcv(seed=1)
    btc = _ohlcv(seed=2)
    rel = build_alt_btc(alt, btc)
    f = compute_all_factors(rel, alt, btc, 59.0, {4: -0.2}, cfg)
    assert "regime_indicator_score" in f["btc_regime"]
    assert "dominance_indicator_score" in f["btc_regime"]
    assert "rvol_score" in f["volume"]
    assert "bollinger_score" in f["volatility"]
    assert "atr_score" in f["volatility"]
    assert "natr_score" in f["volatility"]
    assert "ema_score" in f["trend"]
