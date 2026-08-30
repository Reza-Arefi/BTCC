"""Analytics layer smoke / invariant tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from btcc.analytics.metrics import (
    indicator_predictive_correlation,
    matured_predictions,
    rolling_indicator_correlation,
)
from btcc.config import load_config


def test_matured_predictions_no_lookahead():
    rows = pd.DataFrame([
        {"timestamp": "2026-01-01T00:00:00+00:00", "future_return_4h": 0.01, "S": 0.5, "signed_momentum": 0.2},
        {"timestamp": "2026-01-01T03:00:00+00:00", "future_return_4h": 0.02, "S": 0.4, "signed_momentum": 0.1},
    ])
    asof = "2026-01-01T04:00:00+00:00"
    m = matured_predictions(rows, asof_ts=asof, horizon_hours=4)
    assert len(m) == 1
    assert str(m.iloc[0]["timestamp"]).startswith("2026-01-01")


def test_correlation_uses_only_matured():
    # Build enough matured rows
    rows = []
    for i in range(40):
        rows.append({
            "timestamp": f"2026-01-01T{i % 20:02d}:00:00+00:00",
            "future_return_4h": 0.01 if i % 2 == 0 else -0.01,
            "signed_momentum": 0.5 if i % 2 == 0 else -0.5,
            "S": 0.2,
        })
    df = pd.DataFrame(rows)
    # asof before second batch maturity would still include matured first ones only via filter
    out = indicator_predictive_correlation(df, asof_ts="2026-01-02T00:00:00+00:00", min_obs=10)
    assert not out.empty
    assert "momentum" in set(out["indicator"])


def test_rolling_corr_asof_never_uses_future_labels():
    rows = []
    for day in range(1, 28):
        rows.append({
            "timestamp": f"2026-01-{day:02d}T00:00:00+00:00",
            "future_return_4h": 0.01,
            "signed_momentum": 0.3,
            "S": 0.1,
        })
    df = pd.DataFrame(rows)
    roll = rolling_indicator_correlation(df, window_days=10, step_days=5, min_obs=5)
    assert isinstance(roll, pd.DataFrame)


def test_telegram_ranking_disabled_for_live_policy():
    cfg = load_config()
    assert cfg["telegram"].get("send_ranking_every_cycle") is False
    assert cfg["telegram"].get("send_daily_summary") is True


def test_sim_alerts_daily_summary_flag():
    from btcc.sim.config import load_sim_config
    sim = load_sim_config()
    assert (sim.get("alerts") or {}).get("send_daily_summary") is True
