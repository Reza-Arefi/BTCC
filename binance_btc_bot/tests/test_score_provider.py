"""Production score provider — cross detection, guards, backtest identity."""

from __future__ import annotations

import math
import unittest
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from binance_btc_bot.config_loader import load_config
from binance_btc_bot.market_data.klines import drop_incomplete_candle, klines_to_dataframe
from binance_btc_bot.strategy.entries import LiveEntryEngine
from binance_btc_bot.strategy.score_provider import (
    ProductionScoreProvider,
    backtest_signed_S,
    load_signal_config,
    new_cross_into,
    score_from_frames,
)


def _synth_ohlcv(
    n: int,
    *,
    start: datetime | None = None,
    interval_min: int = 15,
    seed: int = 7,
    base: float = 100.0,
    vol: float = 1000.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = start or datetime(2024, 1, 1, tzinfo=timezone.utc)
    rets = rng.normal(0.0, 0.002, size=n)
    close = base * np.cumprod(1.0 + rets)
    open_ = np.concatenate([[base], close[:-1]])
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.0, 0.001, size=n))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.0, 0.001, size=n))
    volume = vol * (1.0 + rng.uniform(-0.2, 0.2, size=n))
    ts = [start + timedelta(minutes=interval_min * i) for i in range(n)]
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(ts, utc=True),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _aligned_pair(n: int = 160) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Synthetic BASEUSDT / BTCUSDT / relative frames."""
    start = datetime(2024, 6, 1, tzinfo=timezone.utc)
    alt = _synth_ohlcv(n, start=start, seed=11, base=3000.0)
    btc = _synth_ohlcv(n, start=start, seed=22, base=60_000.0)
    from btcc.series.relative import build_alt_btc

    rel = build_alt_btc(alt, btc)
    assert rel is not None and len(rel) == n
    return rel, alt, btc


class TestNewCrossRule(unittest.TestCase):
    def test_below_to_at_or_above_is_entry_candidate(self):
        self.assertTrue(new_cross_into(0.64, 0.65))
        self.assertTrue(new_cross_into(0.10, 0.90))
        eng = LiveEntryEngine(strategy_key="T1", long_threshold=0.65)
        eng.evaluate(symbol="ETHBTC", score=0.64, open_symbols=[], open_count=0)
        d = eng.evaluate(symbol="ETHBTC", score=0.65, open_symbols=[], open_count=0)
        self.assertTrue(d.trade_suggested)
        self.assertTrue(d.crossed_into)

    def test_stay_above_is_not_new_entry(self):
        self.assertFalse(new_cross_into(0.65, 0.70))
        self.assertFalse(new_cross_into(0.80, 0.80))
        eng = LiveEntryEngine(strategy_key="T1", long_threshold=0.65)
        eng.evaluate(symbol="ETHBTC", score=0.50, open_symbols=[], open_count=0)
        eng.evaluate(symbol="ETHBTC", score=0.70, open_symbols=[], open_count=0)
        d = eng.evaluate(symbol="ETHBTC", score=0.75, open_symbols=[], open_count=0)
        self.assertFalse(d.trade_suggested)

    def test_stay_below_is_no_entry(self):
        self.assertFalse(new_cross_into(0.40, 0.50))
        self.assertFalse(new_cross_into(0.64, 0.649))
        eng = LiveEntryEngine(strategy_key="T1", long_threshold=0.65)
        d1 = eng.evaluate(symbol="ETHBTC", score=0.40, open_symbols=[], open_count=0)
        d2 = eng.evaluate(symbol="ETHBTC", score=0.50, open_symbols=[], open_count=0)
        self.assertFalse(d1.trade_suggested)
        self.assertFalse(d2.trade_suggested)


class TestScoreGuards(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_signal_config()
        self.now = datetime(2024, 6, 2, 12, 0, tzinfo=timezone.utc)

    def _provider_from_frames(
        self,
        alt: pd.DataFrame,
        btc: pd.DataFrame,
        *,
        max_age: float = 1800.0,
        min_hist: int = 100,
    ) -> ProductionScoreProvider:
        store = {"ETHUSDT": alt, "BTCUSDT": btc}

        def fetch(symbol: str, interval: str, limit: int) -> list:
            df = store[symbol.upper()].iloc[-limit:]
            rows = []
            for _, r in df.iterrows():
                open_ms = int(pd.Timestamp(r["timestamp"]).timestamp() * 1000)
                rows.append(
                    [
                        open_ms,
                        float(r["open"]),
                        float(r["high"]),
                        float(r["low"]),
                        float(r["close"]),
                        float(r["volume"]),
                        open_ms + 899_999,
                        float(r["close"]) * float(r["volume"]),
                        10,
                        0.0,
                        0.0,
                        "0",
                    ]
                )
            return rows

        return ProductionScoreProvider(
            signal_cfg=self.cfg,
            kline_fetcher=fetch,
            interval="15m",
            lookback_bars=1000,
            min_history_bars=min_hist,
            long_threshold=0.65,
            max_closed_candle_age_sec=max_age,
            diagnostics=False,
            now_fn=lambda: self.now,
        )

    def test_missing_market_data_no_entry(self):
        def fetch(symbol: str, interval: str, limit: int) -> list:
            return []

        prov = ProductionScoreProvider(
            signal_cfg=self.cfg,
            kline_fetcher=fetch,
            now_fn=lambda: self.now,
        )
        snap = prov.evaluate("ETHBTC")
        self.assertIsNone(snap.S_current)
        self.assertFalse(snap.cross_detected)
        self.assertEqual(snap.reason, "MISSING_MARKET_DATA")
        self.assertIsNone(prov("ETHBTC", {}))

    def test_stale_market_data_no_entry(self):
        # Last closed candle far in the past relative to now.
        start = self.now - timedelta(days=3)
        alt = _synth_ohlcv(160, start=start, seed=1)
        btc = _synth_ohlcv(160, start=start, seed=2)
        prov = self._provider_from_frames(alt, btc, max_age=1800.0)
        snap = prov.evaluate("ETHBTC")
        self.assertIsNone(snap.S_current)
        self.assertTrue(str(snap.reason or "").startswith("STALE_MARKET_DATA"))
        self.assertIsNone(prov("ETHBTC", {"relative_price": 0.05}))

    def test_insufficient_history_no_entry(self):
        # Fresh tip near now, but only 40 bars.
        start = self.now - timedelta(minutes=15 * 45)
        alt = _synth_ohlcv(40, start=start, seed=3)
        btc = _synth_ohlcv(40, start=start, seed=4)
        # Tip must be closed: last open + 15m <= now
        prov = self._provider_from_frames(alt, btc, min_hist=100, max_age=10_000.0)
        snap = prov.evaluate("ETHBTC")
        self.assertIsNone(snap.S_current)
        self.assertTrue(str(snap.reason or "").startswith("INSUFFICIENT_HISTORY"))
        self.assertIsNone(prov("ETHBTC", {}))

    def test_nan_invalid_score_no_entry(self):
        rel, alt, btc = _aligned_pair(120)
        # Poison closes → NaN factors
        rel = rel.copy()
        rel.loc[rel.index[-5]:, "close"] = float("nan")
        s = score_from_frames(rel, alt, btc, self.cfg)
        self.assertIsNone(s)

    def test_provider_cross_diag_fields(self):
        # Enough history with tip closed just before now.
        start = self.now - timedelta(minutes=15 * 170)
        alt = _synth_ohlcv(170, start=start, seed=5)
        btc = _synth_ohlcv(170, start=start, seed=6)
        prov = self._provider_from_frames(alt, btc, max_age=10_000.0)
        snap = prov.evaluate("ETHBTC")
        self.assertIsNotNone(snap.S_current)
        self.assertTrue(math.isfinite(float(snap.S_current)))
        self.assertIn("timestamp", snap.to_dict())
        self.assertIn("relative_price", snap.to_dict())
        self.assertIn("S_previous", snap.to_dict())
        self.assertIn("S_current", snap.to_dict())
        self.assertIn("cross_detected", snap.to_dict())
        # Bar-level cross must match helper.
        expected = new_cross_into(snap.S_previous, snap.S_current, 0.65)
        self.assertEqual(snap.cross_detected, expected)
        line = snap.log_line()
        self.assertIn("SCORE_DIAG", line)
        self.assertNotIn("api_key", line.lower())
        self.assertNotIn("secret", line.lower())
        self.assertNotIn("private", line.lower())


class TestLiveMatchesBacktest(unittest.TestCase):
    def test_identical_input_identical_S(self):
        cfg = load_signal_config()
        rel, alt, btc = _aligned_pair(180)
        live_s = score_from_frames(rel, alt, btc, cfg, interval="15m")
        bt_s = backtest_signed_S(rel, alt, btc, cfg, interval="15m")
        self.assertIsNotNone(live_s)
        self.assertIsNotNone(bt_s)
        self.assertAlmostEqual(float(live_s), float(bt_s), places=12)

    def test_score_uses_combined_S_not_raw_signal_score(self):
        """Guard: production must use signed combined_score, not factor signal_score."""
        from btcc.factors.combine import compute_all_factors
        from btcc.sim.score import extract_factor_scores, combined_score, static_factor_weights

        cfg = load_signal_config()
        rel, alt, btc = _aligned_pair(150)
        factors = compute_all_factors(rel, alt, btc, None, {}, cfg, "15m")
        raw = float(factors["signal_score"])
        signed = float(
            combined_score(extract_factor_scores(factors), static_factor_weights(cfg))["S"]
        )
        live = score_from_frames(rel, alt, btc, cfg)
        self.assertAlmostEqual(float(live), signed, places=12)
        # Signed S is in [-1,1]; raw signal_score is in [0,1] — they are different maps.
        self.assertGreaterEqual(signed, -1.0)
        self.assertLessEqual(signed, 1.0)
        self.assertGreaterEqual(raw, 0.0)
        self.assertLessEqual(raw, 1.0)
        # Not required to differ for every seed, but mapping identity is what matters.
        self.assertAlmostEqual(signed, 2.0 * raw - 1.0, places=10)

    def test_default_config_still_frozen_dry(self):
        cfg = load_config()
        self.assertFalse(cfg["live"]["enabled"])
        self.assertTrue(cfg["live"]["dry_run"])
        self.assertEqual(cfg["live"]["strategy"], "T1")
        self.assertIn(cfg["live"]["selector"], (None, "null", ""))
        self.assertEqual(cfg["portfolio"]["max_simultaneous_trades"], 8)
        self.assertAlmostEqual(cfg["portfolio"]["allocation_per_trade"], 0.125)
        self.assertEqual(cfg["signal"]["interval"], "15m")
        self.assertAlmostEqual(cfg["entry"]["long_threshold"], 0.65)


class TestKlineCloseTiming(unittest.TestCase):
    def test_drops_forming_candle(self):
        now = datetime(2024, 6, 1, 12, 7, tzinfo=timezone.utc)  # mid-bar
        open0 = datetime(2024, 6, 1, 11, 45, tzinfo=timezone.utc)
        open1 = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)  # forming
        raw = [
            [int(open0.timestamp() * 1000), 1, 2, 0.5, 1.5, 10, 0, 0, 0, 0, 0, "0"],
            [int(open1.timestamp() * 1000), 1.5, 2, 1, 1.8, 11, 0, 0, 0, 0, 0, "0"],
        ]
        df = drop_incomplete_candle(klines_to_dataframe(raw), interval="15m", now=now)
        self.assertEqual(len(df), 1)
        self.assertEqual(pd.Timestamp(df["timestamp"].iloc[0]).to_pydatetime(), open0)


class TestBuildProviderWiring(unittest.TestCase):
    def test_build_uses_exchange_get_klines(self):
        from binance_btc_bot.strategy.score_provider import build_production_score_provider

        cfg = load_config()
        ex = MagicMock()
        ex.get_klines.return_value = []
        prov = build_production_score_provider(cfg, exchange=ex, diagnostics=False)
        self.assertIsNone(prov("ETHBTC", {}))
        self.assertTrue(ex.get_klines.called)
        # Never touch credentials attributes in scoring path
        for call in ex.mock_calls:
            self.assertNotIn("api_key", str(call).lower())


if __name__ == "__main__":
    unittest.main()
