"""Additional Adaptive V2 tests: weight schedule, maturity, fingerprint fields."""

from __future__ import annotations

import pandas as pd

from btcc.sim.maturity import filter_matured_for_learning, outcome_mature_at
from btcc.sim.weight_schedule import WeightSchedule, WeightVersion


def test_outcome_maturity_gate():
    t0 = pd.Timestamp("2026-01-01 12:00", tz="UTC")
    assert not outcome_mature_at(t0, horizon_hours=4, asof_ts=t0 + pd.Timedelta(hours=3, minutes=59))
    assert outcome_mature_at(t0, horizon_hours=4, asof_ts=t0 + pd.Timedelta(hours=4))


def test_filter_matured_blocks_premature():
    rows = [
        {"timestamp": "2026-01-01T00:00:00Z", "future_return_4h": 0.01},
        {"timestamp": "2026-01-01T10:00:00Z", "future_return_4h": 0.02},
    ]
    asof = pd.Timestamp("2026-01-01 05:00", tz="UTC")
    m = filter_matured_for_learning(rows, asof_ts=asof, horizon_hours=4)
    assert len(m) == 1


def test_weight_schedule_effective_next_bar():
    sch = WeightSchedule({
        "momentum": 0.2, "trend": 0.2, "btc_regime": 0.2,
        "volume": 0.1, "volatility": 0.1, "rsi": 0.1, "structure": 0.1,
    })
    t_calc = "2026-01-10T02:00:00+00:00"
    t_eff = "2026-01-10T02:15:00+00:00"
    new = {k: 0.14 for k in ["momentum", "trend", "btc_regime", "volume", "volatility", "rsi", "structure"]}
    sch.add(WeightVersion(
        version_id="daily_1",
        weights=new,
        calculated_at=t_calc,
        effective_from=t_eff,
        phase="daily",
        update_number=1,
    ))
    a = sch.active_at(t_calc)
    assert a.version_id != "daily_1"
    b = sch.active_at(t_eff)
    assert b.version_id == "daily_1"
