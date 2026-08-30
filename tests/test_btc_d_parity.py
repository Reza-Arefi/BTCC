"""Prove live and backtest BTC.D use the same relative definition."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import pytest

from btcc.data.dominance import DominanceFeed
from btcc.data.relative_btc_d import (
    TOP_COIN_IDS,
    compute_relative_btc_d_pct,
)
from btcc.backtest.dominance_history import HistoricalDominanceSeries, _TOP_COIN_IDS


def test_top_coin_ids_identical_live_and_backtest():
    assert tuple(_TOP_COIN_IDS) == TOP_COIN_IDS
    assert TOP_COIN_IDS[0] == "bitcoin"


def test_relative_formula_shared():
    caps = {cid: 1e10 for cid in TOP_COIN_IDS[:12]}
    caps["bitcoin"] = 5e11
    pct, meta = compute_relative_btc_d_pct(caps)
    assert meta["status"] == "OK"
    assert meta["calibration"] == "none_no_present_day_scaling"
    assert abs(pct - 100.0 * 5e11 / sum(caps.values())) < 1e-9


def test_missing_coin_skipped_same_as_backtest():
    caps = {cid: 1e10 for cid in TOP_COIN_IDS[:10]}
    caps["bitcoin"] = 4e11
    # drop one non-btc
    caps.pop("ethereum", None)
    pct, meta = compute_relative_btc_d_pct(caps)
    assert pct is not None
    assert "ethereum" not in meta["coins_used"]
    assert meta["bitcoin_present"] is True


def test_bitcoin_missing_fails():
    caps = {cid: 1e10 for cid in TOP_COIN_IDS[1:12]}
    pct, meta = compute_relative_btc_d_pct(caps)
    assert pct is None
    assert meta["status"] == "BTC_MCAP_MISSING"


def test_live_observation_at_no_future_leak(tmp_path):
    feed = DominanceFeed(
        history_path=tmp_path / "dom.json",
        poll_seconds=0,
    )
    t0 = datetime(2026, 8, 29, 14, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 8, 29, 15, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 8, 29, 16, 0, tzinfo=timezone.utc)
    from btcc.data.dominance import DominanceSnapshot

    feed.history = [
        DominanceSnapshot(t0, 55.0, "coingecko_top_coins_relative", {}),
        DominanceSnapshot(t1, 56.0, "coingecko_top_coins_relative", {}),
        DominanceSnapshot(t2, 57.0, "coingecko_top_coins_relative", {}),
    ]
    # 14:15 prediction must not see 15:00 or 16:00
    decision = datetime(2026, 8, 29, 14, 15, tzinfo=timezone.utc)
    snap, obs, st = feed.observation_at(decision)
    assert st == "OK"
    assert obs == t0
    assert snap.btc_dominance_pct == 55.0

    # Aligns with HistoricalDominanceSeries
    df = pd.DataFrame({
        "timestamp": [t0, t1, t2],
        "btc_dominance_pct": [55.0, 56.0, 57.0],
    })
    hist = HistoricalDominanceSeries(df)
    pct_h, obs_h, st_h = hist.observation_at(decision)
    assert st_h == "OK"
    assert float(pct_h) == 55.0
    assert pd.Timestamp(obs_h) == pd.Timestamp(t0)


def test_live_ignores_legacy_absolute_history(tmp_path):
    path = tmp_path / "dom.json"
    path.write_text(
        """[
      {"timestamp": "2026-08-29T10:00:00+00:00", "btc_dominance_pct": 59.0,
       "source": "coingecko_global", "raw": {}},
      {"timestamp": "2026-08-29T11:00:00+00:00", "btc_dominance_pct": 54.0,
       "source": "coingecko_top_coins_relative",
       "representation": "relative_btc_share_of_top_n", "raw": {}}
    ]""",
        encoding="utf-8",
    )
    feed = DominanceFeed(history_path=path)
    assert len(feed.history) == 1
    assert feed.history[0].btc_dominance_pct == 54.0


def test_signal_config_points_at_relative():
    from btcc.config import load_config

    cfg = load_config()
    assert cfg["data"]["dominance_source"] == "coingecko_top_coins_relative"
    assert "markets" in cfg["data"]["dominance_url"]
    assert "relative" in cfg["data"]["dominance_history_path"]
