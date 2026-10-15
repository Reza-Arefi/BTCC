"""Weekly entry blackout: FRI 12:00 → MON 00:00 UTC blocks NEW entries only."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from binance_btc_bot.config_loader import load_config
from binance_btc_bot.execution.live3 import LIVE3_BLACKOUT_LABEL, build_live3_target_config
from binance_btc_bot.strategy.entries import CLS_ENTRY_BLACKOUT, CLS_ELIGIBLE, LiveEntryEngine
from binance_btc_bot.strategy.entry_blackout import EntryBlackout

WKND = EntryBlackout.from_config({"enabled": True, "timezone": "UTC", "start": "FRI 12:00", "end": "MON 00:00"})


def _utc(y, mo, d, h, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=timezone.utc)


# 2026-10-09 is a Friday.
@pytest.mark.parametrize(
    "ts,blocked",
    [
        (_utc(2026, 10, 9, 11, 59), False),
        (_utc(2026, 10, 9, 12, 0), True),
        (_utc(2026, 10, 10, 0, 0), True),
        (_utc(2026, 10, 11, 23, 59), True),
        (_utc(2026, 10, 12, 0, 0), False),
        (_utc(2026, 10, 14, 12, 0), False),
    ],
)
def test_window_boundaries(ts, blocked):
    assert WKND.is_blocked(ts) is blocked


def test_non_utc_input_is_converted():
    brt = timezone(timedelta(hours=-3))
    assert WKND.is_blocked(datetime(2026, 10, 9, 9, 0, tzinfo=brt)) is True  # = FRI 12:00 UTC
    assert WKND.is_blocked(datetime(2026, 10, 9, 8, 59, tzinfo=brt)) is False


def test_disabled_or_missing_config_returns_none():
    assert EntryBlackout.from_config(None) is None
    assert EntryBlackout.from_config({"enabled": False, "start": "FRI 12:00", "end": "MON 00:00"}) is None


@pytest.mark.parametrize("bad", ["FRIDAY 12:00", "FRI 25:00", "12:00"])
def test_bad_time_rejected(bad):
    with pytest.raises(ValueError):
        EntryBlackout.from_config({"enabled": True, "start": bad, "end": "MON 00:00"})


def test_cross_in_blackout_is_blocked_and_consumed():
    eng = LiveEntryEngine(strategy_key="TRN4", long_threshold=0.65, blackout=WKND)
    sat = _utc(2026, 10, 10, 15)
    eng.evaluate(symbol="ETHBTC", score=0.50, open_symbols=[], open_count=0, now=sat)
    d = eng.evaluate(symbol="ETHBTC", score=0.70, open_symbols=[], open_count=0, now=sat)
    assert d.trade_suggested is False
    assert d.classification == CLS_ENTRY_BLACKOUT
    mon = _utc(2026, 10, 12, 0, 15)
    d2 = eng.evaluate(symbol="ETHBTC", score=0.70, open_symbols=[], open_count=0, now=mon)
    assert d2.trade_suggested is False  # still in band — no deferred weekend entry
    eng.evaluate(symbol="ETHBTC", score=0.50, open_symbols=[], open_count=0, now=mon)
    d3 = eng.evaluate(symbol="ETHBTC", score=0.70, open_symbols=[], open_count=0, now=mon)
    assert d3.trade_suggested is True
    assert d3.classification == CLS_ELIGIBLE


def test_blackout_does_not_reserve_slot():
    from binance_btc_bot.portfolio.manager import PortfolioManager

    pm = PortfolioManager.from_config(
        {"portfolio": {"max_simultaneous_trades": 4, "allocation_per_trade": 0.25, "max_total_allocation": 1.0}}
    )
    eng = LiveEntryEngine(strategy_key="TRN4", long_threshold=0.65, portfolio=pm, max_open=4, blackout=WKND)
    sun = _utc(2026, 10, 11, 10)
    eng.evaluate(symbol="ETHBTC", score=0.50, reserve_slot=True, now=sun)
    d = eng.evaluate(symbol="ETHBTC", score=0.70, reserve_slot=True, now=sun)
    assert d.trade_suggested is False
    assert d.reservation_id is None
    assert pm.slots_used() == 0


def test_production_config_and_live3_overlay_have_no_blackout():
    cfg = load_config()
    assert EntryBlackout.from_config(cfg["entry"]["blackout"]) is None
    assert LIVE3_BLACKOUT_LABEL == "OFF"
    tgt = build_live3_target_config(cfg)
    assert tgt["live"]["strategy"] == "F4H"
    assert EntryBlackout.from_config(tgt["entry"]["blackout"]) is None
