"""Production freeze + kill-switch + archive smoke tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from binance_btc_bot.archive.live_archive import LiveDailyArchive, attach_archive_to_database
from binance_btc_bot.config_loader import (
    PRODUCTION_CONFIG_VERSION,
    PRODUCTION_STRATEGY,
    PRODUCTION_ENTRY_PROFILE,
    PRODUCTION_THRESHOLD,
    PRODUCTION_TAU,
    PRODUCTION_MAX_TRADES,
    PRODUCTION_ALLOC,
    env_live_trading_enabled,
    format_check_config_report,
    is_live_trading_enabled,
    load_config,
    validate_production_freeze,
)
from binance_btc_bot.exchange.binance import BinanceExchange
from binance_btc_bot.storage.database import BotDatabase


def test_production_config_loads_and_matches_freeze():
    cfg = load_config()
    assert cfg["live"]["strategy"] == PRODUCTION_STRATEGY
    assert cfg["signal"]["momentum_profile"] == PRODUCTION_ENTRY_PROFILE
    assert cfg["entry"]["mode"] == PRODUCTION_ENTRY_PROFILE
    assert float(cfg["entry"]["long_threshold"]) == pytest.approx(PRODUCTION_THRESHOLD)
    assert float(cfg["entry"]["tau"]) == pytest.approx(PRODUCTION_TAU)
    assert cfg["entry"]["late_entry_enabled"] is False
    assert cfg["live"]["selector"] in (None, "null", "")
    w2 = cfg["strategies"]["W2"]
    assert float(w2["arm_sl_activation_trail"]) == pytest.approx(0.05)
    assert float(w2["activation"]) == pytest.approx(0.04)
    assert float(w2["trail_distance"]) == pytest.approx(0.02)
    assert int(cfg["portfolio"]["max_simultaneous_trades"]) == PRODUCTION_MAX_TRADES
    assert float(cfg["portfolio"]["allocation_per_trade"]) == pytest.approx(PRODUCTION_ALLOC)
    assert (cfg.get("production") or {}).get("config_version") == PRODUCTION_CONFIG_VERSION
    assert validate_production_freeze(cfg) == []
    report = format_check_config_report(cfg)
    assert "STATUS: OK" in report
    assert "LIVE TRADING DISABLED" in report


def test_live_trading_kill_switch_default_false(monkeypatch):
    monkeypatch.delenv("LIVE_TRADING_ENABLED", raising=False)
    assert env_live_trading_enabled() is False
    cfg = load_config()
    cfg["live"]["enabled"] = True
    cfg["live"]["dry_run"] = False
    assert is_live_trading_enabled(cfg) is False


def test_exchange_writes_blocked_when_kill_switch_false(monkeypatch):
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "false")
    ex = BinanceExchange(dry_run=False, live_enabled=True, api_key="x", signer=object())
    ok, reason = ex._writes_allowed()
    assert ok is False
    assert reason == "LIVE_TRADING_ENABLED_FALSE"


def test_live_archive_append_only(tmp_path: Path):
    arch = LiveDailyArchive(
        tmp_path / "archive",
        fingerprint={"config_version": PRODUCTION_CONFIG_VERSION, "strategy": PRODUCTION_STRATEGY},
    )
    arch.record_signal({"symbol": "ETHBTC", "score": 1.0, "payload": {"tau": 0.03, "ext_pct_close": 0.031}})
    arch.record_order({"symbol": "ETHBTC", "side": "BUY", "status": "DRY_RUN", "order_id": "1"})
    arch.record_trade({"trade_id": "t1", "symbol": "ETHBTC", "status": "CLOSED", "realized_pnl_btc": 0.0001})
    arch.write_daily_summary({"number_of_signals": 1, "number_of_trades": 1, "wins": 1, "losses": 0})
    day_dirs = list((tmp_path / "archive").iterdir())
    assert len(day_dirs) == 1
    d = day_dirs[0]
    assert (d / "signals.csv").exists()
    assert (d / "orders.csv").exists()
    assert (d / "trades.csv").exists()
    assert (d / "daily_summary.json").exists()
    assert (d / "metadata.json").exists()
    assert (d / "entry_factors.jsonl").exists()


def test_archive_hooks_database(tmp_path: Path):
    db = BotDatabase(str(tmp_path / "t.sqlite3"))
    arch = LiveDailyArchive(tmp_path / "arch", fingerprint={"config_version": PRODUCTION_CONFIG_VERSION})
    attach_archive_to_database(db, arch)
    db.insert_signal(symbol="ETHBTC", score=1.0, strategy="W2", classification="NEW_CROSS", payload={"S": 1.0})
    days = list((tmp_path / "arch").glob("*"))
    assert len(days) == 1
