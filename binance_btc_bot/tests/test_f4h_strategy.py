"""F4H (HTF4H) live pieces: 4h bar cross, cooldown, fixed-TP OCO, 1m-built 4h bars, time exit."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from btcc.series.relative import build_alt_btc
from binance_btc_bot.exchange.base import SymbolInfo
from binance_btc_bot.execution.engine import BinanceBotEngine
from binance_btc_bot.execution.protection_requests import build_t1_oco_request_params
from binance_btc_bot.portfolio.manager import PortfolioManager
from binance_btc_bot.risk.safety_manager import PositionView, SafetyManager, SafetyManagerConfig
from binance_btc_bot.strategy.entries import (
    CLS_BAR_ALREADY_DECIDED,
    CLS_NEED_PRIOR_BELOW,
    CLS_SAME_SYMBOL_COOLDOWN,
    CLS_SIGNAL_CONTINUATION,
    LiveEntryEngine,
)
from binance_btc_bot.strategy.htf_bars import BAR_MS, MIN_MS, BarNotReady, Rel4hBarStore, aggregate_4h
from binance_btc_bot.strategy.trails import get_strategy, map_trail_to_binance_oco


def _meta(symbol: str = "ETHBTC") -> SymbolInfo:
    return SymbolInfo(
        symbol=symbol, status="TRADING", base_asset=symbol[:-3], quote_asset="BTC",
        quantity_step=0.0001, min_quantity=0.0001, max_quantity=1000.0, price_tick=0.000001,
        min_notional=0.0001,
        order_types=("LIMIT", "MARKET", "STOP_LOSS", "TAKE_PROFIT", "STOP_LOSS_LIMIT", "TAKE_PROFIT_LIMIT"),
        oco_allowed=True, min_trailing_above_delta=10, max_trailing_above_delta=2000,
        min_trailing_below_delta=10, max_trailing_below_delta=2000,
    )


def _engine(**kw) -> LiveEntryEngine:
    pm = PortfolioManager.from_config(
        {"portfolio": {"max_simultaneous_trades": 6, "allocation_per_trade": 0.1666, "max_total_allocation": 1.0}}
    )
    return LiveEntryEngine(strategy_key="F4H", long_threshold=0.65, portfolio=pm, max_open=6,
                           cross_source="BAR", **kw)


# ---------------------------------------------------------------- bar cross
def test_bar_cross_fires_once_per_bar():
    eng = _engine()
    d = eng.evaluate(symbol="ETHBTC", score=0.70, previous_score=0.60, bar_ts="b1", reserve_slot=True)
    assert d.trade_suggested and d.reservation_id
    again = eng.evaluate(symbol="ETHBTC", score=0.70, previous_score=0.60, bar_ts="b1", reserve_slot=True)
    assert not again.trade_suggested and again.classification == CLS_BAR_ALREADY_DECIDED


def test_bar_cross_needs_previous_bar_below():
    eng = _engine()
    assert eng.evaluate(symbol="A", score=0.70, previous_score=0.66, bar_ts="b1").classification == \
        CLS_SIGNAL_CONTINUATION
    assert eng.evaluate(symbol="B", score=0.70, previous_score=None, bar_ts="b1").classification == \
        CLS_NEED_PRIOR_BELOW
    assert not eng.evaluate(symbol="C", score=0.64, previous_score=0.10, bar_ts="b1").trade_suggested


def test_bar_cross_cold_start_fires_without_memory():
    """Unlike MEMORY mode, a restart does not need to observe S below first — the bar carries it."""
    eng = _engine()
    assert eng.evaluate(symbol="ETHBTC", score=0.80, previous_score=0.40, bar_ts="b9").trade_suggested


def test_bar_cross_skipped_for_full_slots_is_consumed():
    pm = PortfolioManager.from_config(
        {"portfolio": {"max_simultaneous_trades": 1, "allocation_per_trade": 0.5, "max_total_allocation": 1.0}}
    )
    eng = LiveEntryEngine(strategy_key="F4H", portfolio=pm, max_open=1, cross_source="BAR")
    assert eng.evaluate(symbol="AAABTC", score=0.7, previous_score=0.5, bar_ts="b1", reserve_slot=True).trade_suggested
    blocked = eng.evaluate(symbol="BBBBTC", score=0.7, previous_score=0.5, bar_ts="b1", reserve_slot=True)
    assert not blocked.trade_suggested
    pm.release(next(iter(pm._reservations)))
    late = eng.evaluate(symbol="BBBBTC", score=0.7, previous_score=0.5, bar_ts="b1", reserve_slot=True)
    assert late.classification == CLS_BAR_ALREADY_DECIDED


def test_same_symbol_cooldown():
    now = datetime(2026, 10, 14, 12, 5, tzinfo=timezone.utc)
    exits = {"ETHBTC": (now - timedelta(minutes=90)).timestamp(), "SOLBTC": (now - timedelta(minutes=130)).timestamp()}
    eng = _engine(same_symbol_cooldown_sec=7200, last_exit_lookup=exits.get)
    d = eng.evaluate(symbol="ETHBTC", score=0.7, previous_score=0.5, bar_ts="b1", now=now, reserve_slot=True)
    assert not d.trade_suggested and d.classification == CLS_SAME_SYMBOL_COOLDOWN
    assert eng.evaluate(symbol="SOLBTC", score=0.7, previous_score=0.5, bar_ts="b1", now=now).trade_suggested
    assert eng.evaluate(symbol="ADABTC", score=0.7, previous_score=0.5, bar_ts="b1", now=now).trade_suggested


# ---------------------------------------------------------------- fixed-TP OCO
def test_f4h_maps_to_fixed_take_profit_oco():
    f4h = get_strategy("F4H")
    assert f4h.fixed_take_profit and f4h.max_hold_hours == 120
    m = map_trail_to_binance_oco(strategy=f4h, symbol="ETHBTC", entry_price=0.04, quantity=1.0, symbol_info=_meta())
    assert m.allowed, m.constraint_notes
    assert m.mode == "OCO_FIXED_TAKE_PROFIT_PLUS_STOP_LOSS"
    assert m.request.above_type == "TAKE_PROFIT" and m.request.below_type == "STOP_LOSS"
    assert m.request.above_trailing_delta == 0
    assert m.activation_price == pytest.approx(0.052, abs=1e-6)
    assert m.initial_stop_price == pytest.approx(0.030, abs=1e-6)
    params = build_t1_oco_request_params(strategy=f4h, symbol="ETHBTC", entry_price=0.04, quantity=1.0,
                                         symbol_info=_meta(), list_client_order_id="t_x")["params"]
    assert "aboveTrailingDelta" not in params


def test_trailing_strategies_still_send_trailing_delta():
    params = build_t1_oco_request_params(strategy=get_strategy("TRN4"), symbol="ETHBTC", entry_price=0.04,
                                         quantity=1.0, symbol_info=_meta(), list_client_order_id="t_x")["params"]
    assert params["aboveTrailingDelta"] == 50


# ---------------------------------------------------------------- 1m-built 4h bars
T0 = int(pd.Timestamp("2026-09-01", tz="UTC").value // 1_000_000)


def _minutes(n_bars: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_bars * BAR_MS // MIN_MS
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    ts = pd.to_datetime(T0 + np.arange(n) * MIN_MS, unit="ms", utc=True)
    spread = np.abs(rng.normal(0, 0.0005, n))
    return pd.DataFrame({"timestamp": ts, "open": close * (1 - spread / 2), "high": close * (1 + spread),
                         "low": close * (1 - spread), "close": close, "volume": rng.uniform(1, 5, n)})


class FakeFeed:
    def __init__(self, frames: dict[str, pd.DataFrame]):
        self.frames = frames
        self.calls = 0

    def __call__(self, symbol: str, start_ms: int, end_ms: int) -> list:
        self.calls += 1
        d = self.frames[symbol]
        ms = d["timestamp"].astype("int64") // 1_000_000
        sel = d[(ms >= start_ms) & (ms <= end_ms)].head(1000)
        return [[int(t.value // 1_000_000), o, h, lo, c, v] for t, o, h, lo, c, v in
                zip(sel["timestamp"], sel["open"], sel["high"], sel["low"], sel["close"], sel["volume"])]


def test_bar_store_matches_research_construction(tmp_path):
    alt, btc = _minutes(12, 1), _minutes(12, 2)
    feed = FakeFeed({"ETHUSDT": alt, "BTCUSDT": btc})
    store = Rel4hBarStore(feed, bars=10, cache_dir=tmp_path, request_pause_sec=0)
    last = T0 + 11 * BAR_MS
    got = store.bars("ETH", last)
    ref = aggregate_4h(build_alt_btc(alt, btc)).tail(10).reset_index(drop=True)
    assert len(got) == 10
    pd.testing.assert_frame_equal(got.reset_index(drop=True), ref, check_dtype=False)
    assert (tmp_path / "ETH.csv").exists()


def test_bar_store_is_incremental_and_persistent(tmp_path):
    alt, btc = _minutes(12, 3), _minutes(12, 4)
    feed = FakeFeed({"ETHUSDT": alt, "BTCUSDT": btc})
    store = Rel4hBarStore(feed, bars=5, cache_dir=tmp_path, request_pause_sec=0)
    store.bars("ETH", T0 + 10 * BAR_MS)
    before = feed.calls
    store.bars("ETH", T0 + 10 * BAR_MS)
    assert feed.calls == before
    store.bars("ETH", T0 + 11 * BAR_MS)
    assert feed.calls - before == 2  # one 1m request per leg for the new bar
    fresh = Rel4hBarStore(FakeFeed({"ETHUSDT": alt, "BTCUSDT": btc}), bars=5, cache_dir=tmp_path,
                          request_pause_sec=0)
    assert len(fresh.bars("ETH", T0 + 11 * BAR_MS)) == 5 and fresh.fetch_1m.calls == 0


def test_bar_store_waits_for_final_minute(tmp_path):
    alt, btc = _minutes(6, 5), _minutes(6, 6)
    truncated = alt.iloc[:-3]
    store = Rel4hBarStore(FakeFeed({"ETHUSDT": truncated, "BTCUSDT": btc}), bars=3, cache_dir=tmp_path,
                          request_pause_sec=0)
    with pytest.raises(BarNotReady):
        store.bars("ETH", T0 + 5 * BAR_MS)
    store.fetch_1m = FakeFeed({"ETHUSDT": alt, "BTCUSDT": btc})
    assert len(store.bars("ETH", T0 + 5 * BAR_MS)) == 3


# ---------------------------------------------------------------- time exit
class _FakeDb:
    def __init__(self, trades):
        self.trades = trades
        self.events = []

    def open_trades(self):
        return self.trades

    def insert_event(self, event, **kw):
        self.events.append((event, kw))


def _trade(tid: str, hours_ago: float, strategy: str = "F4H", status: str = "PROTECTED", hold: float | None = 120):
    cfg = {} if hold is None else {"max_hold_hours": hold}
    et = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {"trade_id": tid, "symbol": f"{tid}BTC", "strategy": strategy, "status": status, "entry_time": et,
            "strategy_config_json": json.dumps(cfg)}


def test_time_exit_closes_only_expired_f4h_positions():
    closed = []

    class Runtime:
        def exit_position(self, trade, *, reason, current_price=None):
            closed.append((trade["trade_id"], reason))
            return {"trade_id": trade["trade_id"], "result": "CLOSED"}

    trades = [
        _trade("OLD", 121), _trade("NEW", 100), _trade("TRN", 500, strategy="TRN4", hold=None),
        _trade("EMG", 200, status="PROTECTED_EMERGENCY"), _trade("LEGACY", 130, hold=None),
    ]
    fake = SimpleNamespace(db=_FakeDb(trades), safety_runtime=Runtime())
    out = BinanceBotEngine.time_exit_cycle(fake)
    assert sorted(closed) == [("LEGACY", "TIME_EXIT"), ("OLD", "TIME_EXIT")]
    assert {r["symbol"] for r in out} == {"OLDBTC", "LEGACYBTC"}
    assert BinanceBotEngine.time_exit_cycle(fake) == []  # throttled


# ---------------------------------------------------------------- safety manager
def test_defensive_mode_disabled(tmp_path):
    from binance_btc_bot.risk.safety_manager import SafetyStateStore

    cfg = SafetyManagerConfig.from_config({"safety_manager": {"defensive_enabled": False}})
    sm = SafetyManager(cfg, SafetyStateStore(tmp_path / "s.json"))
    views = [PositionView(str(i), f"S{i}BTC", 1.0, 0.9, 0.75) for i in range(6)]
    assert sm.would_trigger_defensive(views) is False
    assert sm.evaluate_portfolio(views) is False


def test_profitable_time_exit_resets_sl_counter(tmp_path):
    from binance_btc_bot.risk.safety_manager import SafetyStateStore

    sm = SafetyManager(SafetyManagerConfig(), SafetyStateStore(tmp_path / "s.json"))
    assert sm.record_trade_closed(trade_id="a", symbol="X", close_reason="HARD_SL") == "SL_COUNTED"
    assert sm.record_trade_closed(trade_id="b", symbol="Y", close_reason="TIME_EXIT", pnl_pct=-0.01) == "NO_CHANGE"
    assert sm.record_trade_closed(trade_id="c", symbol="Z", close_reason="TIME_EXIT", pnl_pct=0.02) == "COUNTER_RESET"
