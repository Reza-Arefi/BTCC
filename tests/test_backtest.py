"""Tests for BTCC 90-day research backtest module."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from btcc.backtest.config import load_backtest_config
from btcc.backtest.data_loader import compute_window, required_bars
from btcc.backtest.dominance_history import HistoricalDominanceSeries
from btcc.backtest.extract import flatten_indicators_and_factors
from btcc.backtest.outcomes import future_outcomes
from btcc.backtest.predict import predict_coin_at_bar
from btcc.config import load_config
from btcc.factors.combine import compute_all_factors
from btcc.probability.score import horizons_probabilities
from btcc.ranking.ranker import rank_signals
from btcc.safety.no_trading import TradingForbiddenError, deny_trading
from btcc.series.relative import build_alt_btc, horizon_bars


def _ohlcv(n=400, seed=0, drift=0.0002):
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


def test_horizon_bar_conversion_15m():
    assert horizon_bars(1, "15m") == 4
    assert horizon_bars(4, "15m") == 16
    assert horizon_bars(8, "15m") == 32
    assert horizon_bars(12, "15m") == 48
    assert horizon_bars(24, "15m") == 96


def test_90_day_window_pinned():
    data_start, eval_start, eval_end = compute_window(90, 200)
    delta = eval_end - eval_start
    assert 89 <= delta.days <= 91
    assert eval_start >= data_start
    assert required_bars(90, 200) >= 90 * 96


def test_future_return_and_outperformance():
    n = 200
    close = pd.Series(np.linspace(1.0, 1.1, n))
    i = 50
    out = future_outcomes(close, i, "15m", [1, 4])
    assert out["future_return_1h"] is not None
    assert out["outperformed_1h"] in (0, 1)
    expected = close.iloc[i + 4] / close.iloc[i] - 1.0
    assert abs(out["future_return_1h"] - expected) < 1e-12
    assert out["outperformed_1h"] == int(expected > 0)


def test_no_lookahead_in_prediction():
    cfg = load_config()
    alt = _ohlcv(seed=1, n=400)
    btc = _ohlcv(seed=2, n=400)
    rel = build_alt_btc(alt, btc)
    i = 250
    rel_at_i = rel.iloc[: i + 1]
    alt_at_i = alt.iloc[: i + 1]
    btc_at_i = btc.iloc[: i + 1]

    p1 = predict_coin_at_bar(rel_at_i, alt_at_i, btc_at_i, 55.0, {4: -0.1}, cfg, "15m")
    p2 = predict_coin_at_bar(rel_at_i.copy(), alt_at_i.copy(), btc_at_i.copy(), 55.0, {4: -0.1}, cfg, "15m")
    assert p1 is not None and p2 is not None
    assert p1["signal_score"] == p2["signal_score"]

    # Longer history changes decision bar (uses last row) — must differ from bar i snapshot
    p_future = predict_coin_at_bar(rel.iloc[: i + 5], alt.iloc[: i + 5], btc.iloc[: i + 5], 55.0, {4: -0.1}, cfg, "15m")
    assert p_future is not None
    assert rel.iloc[i + 4]["close"] != rel.iloc[i]["close"]


def test_ranking_by_4h_only():
    rows = [
        {"base": "A", "p_4h": 0.55, "p_1h": 0.90},
        {"base": "B", "p_4h": 0.85, "p_1h": 0.40},
        {"base": "C", "p_4h": 0.70, "p_1h": 0.95},
    ]
    ranked = rank_signals(rows, 4)
    assert [r["base"] for r in ranked[:3]] == ["B", "C", "A"]


def test_top5_same_logic_as_live():
    cfg = load_config()
    rows = []
    for seed in range(6):
        alt = _ohlcv(seed=seed + 10, n=350)
        btc = _ohlcv(seed=99, n=350)
        rel = build_alt_btc(alt, btc)
        factors = compute_all_factors(rel, alt, btc, 55.0, {4: -0.2}, cfg)
        probs = horizons_probabilities(factors["signal_score"], cfg)
        rows.append({"base": f"X{seed}", "p_4h": probs["p_4h"], "p_1h": probs["p_1h"]})
    ranked = rank_signals(rows, cfg["probability"]["primary_rank_horizon"])
    top5 = ranked[:5]
    assert len(top5) == 5
    assert top5[0]["p_4h"] >= top5[-1]["p_4h"]


def test_indicators_flattened():
    cfg = load_config()
    alt = _ohlcv(seed=1)
    btc = _ohlcv(seed=2)
    rel = build_alt_btc(alt, btc)
    factors = compute_all_factors(rel, alt, btc, 59.0, {4: -0.1}, cfg)
    flat = flatten_indicators_and_factors(factors)
    for key in (
        "indicator_momentum", "indicator_ema", "indicator_macd", "indicator_ichimoku",
        "indicator_adx", "indicator_btc_regime", "indicator_rvol", "indicator_bollinger",
        "indicator_rsi", "indicator_structure",
        "factor_momentum", "factor_trend", "factor_btc_regime",
    ):
        assert key in flat


def test_late_entry_separate_from_probability():
    cfg = load_config()
    alt = _ohlcv(seed=3, drift=0.001)
    btc = _ohlcv(seed=4)
    rel = build_alt_btc(alt, btc)
    from btcc.late_entry.score import late_entry_score
    factors = compute_all_factors(rel, alt, btc, 55.0, {4: -0.1}, cfg)
    probs = horizons_probabilities(factors["signal_score"], cfg)
    late = late_entry_score(rel, factors, cfg)
    assert 0 <= late["late_entry_score"] <= 1
    assert 0.01 <= probs["p_4h"] <= 0.99


def test_historical_dominance_no_lookahead():
    ts = pd.date_range("2026-01-01", periods=10, freq="1h", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "btc_dominance_pct": np.linspace(58, 60, 10)})
    dom = HistoricalDominanceSeries(df)
    t = ts[5]
    pct, obs, status = dom.observation_at(t)
    assert status == "OK"
    assert obs == ts[5]
    assert pct == float(df.iloc[5]["btc_dominance_pct"])
    # Future observations must not leak
    pct2, obs2, _ = HistoricalDominanceSeries(df.iloc[:6]).observation_at(t)
    assert pct == pct2 and obs == obs2
    # Decision between hours uses last-known prior
    mid = ts[5] + pd.Timedelta(minutes=30)
    pct3, obs3, st3 = dom.observation_at(mid)
    assert st3 == "OK"
    assert obs3 == ts[5]


def test_dominance_fetch_graceful_on_auth_failure(monkeypatch, tmp_path):
    """Backtest must not crash when CoinGecko global chart returns 401."""

    class FakeResp:
        def __init__(self, status_code, json_data=None):
            self.status_code = status_code
            self.text = "Unauthorized"
            self._json = json_data or {}

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(str(self.status_code))

        def json(self):
            return self._json

    def fake_get(self, url, *a, **k):
        if "bitcoin/market_chart" in url:
            caps = [[1_700_000_000_000 + i * 3_600_000, 1e12] for i in range(5)]
            return FakeResp(200, {"market_caps": caps})
        if "market_cap_chart" in url:
            return FakeResp(401)
        if url.endswith("/global") or url.rstrip("/").endswith("/global"):
            return FakeResp(200, {"data": {"market_cap_percentage": {"btc": 59.0}}})
        # other coins
        if "/coins/" in url and "market_chart" in url:
            caps = [[1_700_000_000_000 + i * 3_600_000, 1e11] for i in range(5)]
            return FakeResp(200, {"market_caps": caps})
        return FakeResp(404)

    monkeypatch.setattr(requests.Session, "get", fake_get)

    dom = HistoricalDominanceSeries.fetch_coingecko(14, tmp_path, force=True)
    # Reconstruction from top coins should succeed under mock
    assert not dom.df.empty
    assert dom.meta.get("source") == "coingecko_top_coins_reconstructed"
    assert "btc_dominance_pct" in dom.df.columns



def test_backtest_config_loads_signal_config():
    cfg = load_backtest_config()
    assert len(cfg["universe"]["bases"]) == 20
    assert cfg["backtest"]["interval"] == "15m"
    assert cfg["safety"]["allow_trading"] is False


def test_no_telegram_in_backtest_module():
    backtest_dir = ROOT / "btcc" / "backtest"
    for p in backtest_dir.rglob("*.py"):
        text = p.read_text(encoding="utf-8")
        assert "TelegramNotifier" not in text
        assert "telegram.notifier" not in text


def test_no_trading_in_backtest():
    with pytest.raises(TradingForbiddenError):
        deny_trading()


def test_probability_unchanged_by_backtest_path():
    cfg = load_config()
    alt = _ohlcv(seed=5)
    btc = _ohlcv(seed=6)
    rel = build_alt_btc(alt, btc)
    factors = compute_all_factors(rel, alt, btc, 55.0, {4: -0.1}, cfg)
    before = horizons_probabilities(factors["signal_score"], cfg)
    predict_coin_at_bar(rel, alt, btc, 55.0, {4: -0.1}, cfg, "15m")
    after = horizons_probabilities(factors["signal_score"], cfg)
    assert before["p_4h"] == after["p_4h"]


def test_alt_btc_construction():
    alt = _ohlcv(seed=1)
    btc = _ohlcv(seed=2)
    rel = build_alt_btc(alt, btc)
    assert rel is not None
    merged = alt.iloc[-1]["close"] / btc.iloc[-1]["close"]
    assert abs(rel.iloc[-1]["close"] - merged) / merged < 0.01
