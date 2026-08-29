"""Tests for adaptive Champion/Challenger walk-forward learning — SIGNAL ONLY."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from btcc.adaptive.learner import (
    estimate_factor_weights,
    temporal_splits,
    time_decay_weights,
)
from btcc.adaptive.metrics import brier_score, calibration_error, evaluate_horizon
from btcc.adaptive.model import (
    FACTOR_KEYS,
    SignalModel,
    create_champion_v1_from_config,
    get_champion,
    next_version,
    promote_challenger,
    save_model,
)
from btcc.adaptive.promotion import compare_and_decide
from btcc.adaptive.rescore import rescore_frame, signal_score_from_row
from btcc.adaptive.store import AdaptivePredictionStore
from btcc.config import load_config
from btcc.factors.combine import compute_all_factors
from btcc.safety.no_trading import TradingForbiddenError, deny_trading


def _synth_labeled(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Synthetic labeled predictions spanning time — ALT/BTC outperform labels."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    rows = []
    for i, t in enumerate(ts):
        # Factor scores correlated (weakly) with outcome
        mom = rng.uniform(0.2, 0.9)
        trend = rng.uniform(0.2, 0.9)
        # Higher trend → more likely outperform
        p_latent = 0.3 * mom + 0.5 * trend + 0.2 * rng.uniform()
        y = int(p_latent > 0.55)
        ret = (0.01 if y else -0.01) + rng.normal(0, 0.002)
        score = 0.2 * mom + 0.2 * trend + 0.2 * 0.5 + 0.12 * 0.5 + 0.10 * 0.5 + 0.08 * 0.5 + 0.10 * 0.5
        rows.append({
            "timestamp": t,
            "symbol": f"ALT{i % 5}USDT",
            "rank": (i % 5) + 1,
            "data_source": "HISTORICAL_INITIAL_DATA" if i < n // 2 else "LIVE_OBSERVATION_DATA",
            "factor_momentum": mom,
            "factor_trend": trend,
            "factor_btc_regime": 0.5,
            "factor_volume": 0.5,
            "factor_volatility": 0.5,
            "factor_rsi": 0.5,
            "factor_structure": 0.5,
            "signal_score": score,
            "probability_1h": 0.5,
            "probability_4h": float(np.clip(0.4 + 0.4 * score, 0.01, 0.99)),
            "probability_8h": 0.5,
            "probability_12h": 0.5,
            "probability_24h": 0.5,
            "future_return_4h": ret,
            "outperformed_4h": y,
            "future_return_1h": ret / 2,
            "outperformed_1h": y,
            "future_return_8h": ret,
            "outperformed_8h": y,
            "future_return_12h": ret,
            "outperformed_12h": y,
            "future_return_24h": ret,
            "outperformed_24h": y,
        })
    return pd.DataFrame(rows)


def test_trading_still_forbidden():
    with pytest.raises(TradingForbiddenError):
        deny_trading()


def test_champion_v1_from_config_weights(tmp_path):
    cfg = load_config()
    model = create_champion_v1_from_config(cfg, tmp_path)
    assert model.version == "v1"
    assert model.model_type == "champion"
    w = model.normalized_weights()
    assert abs(sum(w.values()) - 1.0) < 1e-9
    for k in FACTOR_KEYS:
        assert abs(w[k] - float(cfg["factors"]["weights"][k])) < 1e-9
    loaded = get_champion(tmp_path)
    assert loaded is not None
    assert loaded.version == "v1"


def test_combine_accepts_champion_weight_override():
    cfg = load_config()
    n = 250
    rng = np.random.default_rng(1)
    px = 100 * np.cumprod(1 + 0.0002 + rng.normal(0, 0.002, n))
    ts = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    df = pd.DataFrame({
        "timestamp": ts, "open": px, "high": px * 1.001, "low": px * 0.999,
        "close": px, "volume": rng.uniform(100, 500, n),
    })
    default = compute_all_factors(df, df, df, 55.0, {1: 0, 4: 0, 12: 0, 24: 0}, cfg)
    heavy_trend = {k: 0.05 for k in FACTOR_KEYS}
    heavy_trend["trend"] = 0.70
    overridden = compute_all_factors(
        df, df, df, 55.0, {1: 0, 4: 0, 12: 0, 24: 0}, cfg, factor_weights=heavy_trend
    )
    # Same indicators, different mix → score may differ
    assert overridden["weights"]["trend"] == 0.70
    assert abs(default["trend"]["score"] - overridden["trend"]["score"]) < 1e-12


def test_time_decay_and_temporal_splits():
    df = _synth_labeled(100)
    w = time_decay_weights(df["timestamp"], half_life_days=30)
    assert len(w) == 100
    assert w[-1] >= w[0]  # recent ≥ older
    train, val, holdout = temporal_splits(df, 0.7, 0.2)
    assert len(train) + len(val) + len(holdout) == 100
    assert train["timestamp"].iloc[-1] <= val["timestamp"].iloc[0]
    assert val["timestamp"].iloc[-1] <= holdout["timestamp"].iloc[0]


def test_estimate_factor_weights_sums_and_bounds():
    df = _synth_labeled(400)
    champ = {k: 1 / 7 for k in FACTOR_KEYS}
    w = estimate_factor_weights(
        df, champ, half_life_days=45, adaptation_rate=0.25,
        weight_min=0.05, weight_max=0.40, horizon=4,
    )
    assert abs(sum(w.values()) - 1.0) < 1e-6
    for v in w.values():
        assert 0.05 - 1e-6 <= v <= 0.40 + 1e-6


def test_adaptive_store_delayed_labels(tmp_path):
    path = tmp_path / "preds.csv"
    store = AdaptivePredictionStore(path)
    ts0 = pd.Timestamp("2026-06-01 12:00:00", tz="UTC")
    # Panel with closes spanning 24h+
    bars = 100
    times = pd.date_range(ts0, periods=bars, freq="15min", tz="UTC")
    panel = pd.DataFrame({
        "timestamp": times,
        "close": np.linspace(1.0, 1.10, bars),  # rising ALT/BTC
    })
    store.append_rows([{
        "timestamp": ts0,
        "symbol": "AAAUSDT",
        "base": "AAA",
        "rank": 1,
        "model_version": "v1",
        "model_type": "champion",
        "data_source": "LIVE_OBSERVATION_DATA",
        "probability_kind": "baseline_model_probability",
        "signal_score": 0.6,
        "probability_1h": 0.6,
        "probability_4h": 0.7,
        "probability_8h": 0.65,
        "probability_12h": 0.62,
        "probability_24h": 0.6,
        "factor_momentum": 0.6,
        "factor_trend": 0.6,
        "factor_btc_regime": 0.5,
        "factor_volume": 0.5,
        "factor_volatility": 0.5,
        "factor_rsi": 0.5,
        "factor_structure": 0.5,
    }])
    # Before enough bars, 24h not labeled; 1h (4 bars) should label
    n = store.backfill_outcomes({"AAAUSDT": panel}, "15m")
    assert n > 0
    row = store.df.iloc[0]
    assert pd.notna(row["outperformed_1h"])
    assert int(row["outperformed_1h"]) == 1  # ALT/BTC rose
    assert pd.notna(row["future_return_1h"]) and row["future_return_1h"] > 0
    # Original prediction fields unchanged
    assert abs(row["probability_4h"] - 0.7) < 1e-12


def test_metrics_brier_and_calibration():
    y = np.array([1.0, 0.0, 1.0, 0.0])
    p = np.array([0.9, 0.1, 0.8, 0.2])
    assert brier_score(y, p) < 0.1
    assert calibration_error(y, p, n_bins=2) < 0.2


def test_promotion_requires_meaningful_improvement(tmp_path):
    cfg = load_config()
    adaptive = cfg["adaptive"]
    labeled = _synth_labeled(600)
    champ = SignalModel(
        version="v1",
        model_type="champion",
        factor_weights={k: float(cfg["factors"]["weights"][k]) for k in FACTOR_KEYS},
        probability_kind="baseline_model_probability",
        created_utc="2026-01-01T00:00:00+00:00",
    )
    # Identical challenger → should KEEP
    chal = SignalModel(
        version="v2",
        model_type="challenger",
        factor_weights=dict(champ.factor_weights),
        probability_kind="baseline_model_probability",
        created_utc="2026-01-02T00:00:00+00:00",
    )
    train, val, _ = temporal_splits(labeled, 0.7, 0.2)
    decision = compare_and_decide(champ, chal, val, cfg, adaptive)
    assert decision["action"] == "KEEP_CHAMPION"
    assert decision["promote"] is False


def test_promote_keeps_prior_champion_folder(tmp_path):
    cfg = load_config()
    create_champion_v1_from_config(cfg, tmp_path)
    chal = SignalModel(
        version="v2",
        model_type="challenger",
        factor_weights={
            "momentum": 0.15, "trend": 0.25, "btc_regime": 0.15,
            "volume": 0.12, "volatility": 0.10, "rsi": 0.08, "structure": 0.15,
        },
        probability_kind="baseline_model_probability",
        created_utc="2026-01-02T00:00:00+00:00",
    )
    save_model(chal, tmp_path, "challenger_v2")
    promote_challenger(tmp_path, chal, {"action": "PROMOTE_CHALLENGER", "promote": True})
    assert (tmp_path / "champion_v1" / "model.json").exists()
    assert (tmp_path / "champion_v2" / "model.json").exists()
    assert get_champion(tmp_path).version == "v2"
    assert next_version(tmp_path) == "v3"


def test_checkpoint_no_update_when_sample_too_small(tmp_path):
    from btcc.adaptive.checkpoint import run_checkpoint

    cfg = load_config()
    adaptive = dict(cfg["adaptive"])
    adaptive["min_samples_conservative"] = 500
    create_champion_v1_from_config(cfg, tmp_path)
    store = AdaptivePredictionStore(tmp_path / "preds.csv")
    # Only 50 labeled rows
    df = _synth_labeled(50)
    store.append_rows(df.to_dict(orient="records"))
    result = run_checkpoint(store, tmp_path, tmp_path / "reports", cfg, adaptive)
    assert result["action"] == "KEEP_CHAMPION"
    assert "n=" in str(result.get("reason", "")) or result.get("mature_4h_observations", 0) < 500
    assert get_champion(tmp_path).version == "v1"
    reports = list((tmp_path / "reports").glob("adaptive_report_*.md"))
    assert reports


def test_promotion_blocked_without_live_labels(tmp_path):
    """Historical Top-5 alone must not promote even if metrics look better."""
    from btcc.adaptive.checkpoint import run_checkpoint

    cfg = load_config()
    adaptive = dict(cfg["adaptive"])
    adaptive["min_samples_conservative"] = 100
    adaptive["require_live_for_promotion"] = True
    adaptive["min_live_samples_for_promotion"] = 200
    create_champion_v1_from_config(cfg, tmp_path)
    store = AdaptivePredictionStore(tmp_path / "preds.csv")
    df = _synth_labeled(600)
    df["data_source"] = "HISTORICAL_INITIAL_DATA"
    store.append_rows(df.to_dict(orient="records"))
    result = run_checkpoint(store, tmp_path, tmp_path / "reports2", cfg, adaptive)
    assert result["action"] == "KEEP_CHAMPION"
    assert get_champion(tmp_path).version == "v1"
    assert "LIVE" in str(result.get("promotion_blocked", ""))


def test_rescore_uses_model_weights():
    cfg = load_config()
    df = _synth_labeled(20)
    model = SignalModel(
        version="v1",
        model_type="champion",
        factor_weights={k: float(cfg["factors"]["weights"][k]) for k in FACTOR_KEYS},
    )
    scored = rescore_frame(df, model, cfg)
    assert "probability_4h" in scored.columns
    assert scored["probability_kind"].iloc[0] in (
        "baseline_model_probability",
        "mixed_or_calibrated",
    )
    row = df.iloc[0]
    s = signal_score_from_row(row, model.normalized_weights())
    assert 0 <= s <= 1


def test_bootstrap_imports_historical_tag(tmp_path):
    from btcc.adaptive.bootstrap import bootstrap_from_backtest

    cfg = load_config()
    adaptive = dict(cfg["adaptive"])
    # Point at real backtest if present; else skip
    bt = Path(adaptive.get("initial_backtest", ""))
    if not (bt / "top5_predictions.csv").exists():
        pytest.skip("90d backtest CSV not present")
    store = AdaptivePredictionStore(tmp_path / "adaptive.csv")
    report = bootstrap_from_backtest(cfg, adaptive, tmp_path, store, force=True)
    assert "imported" in str(report.get("historical_import", ""))
    assert (store.df["data_source"] == "HISTORICAL_INITIAL_DATA").all()
    assert get_champion(tmp_path) is not None
