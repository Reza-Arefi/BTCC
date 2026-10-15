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
from binance_btc_bot.strategy.trails import get_strategy


def _write_variant(tmp_path: Path, mutate) -> Path:
    import yaml

    from binance_btc_bot.config_loader import DEFAULT_CONFIG_PATH

    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    mutate(raw)
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return p


def test_production_freeze_rejects_weekend_blackout(tmp_path: Path):
    def mutate(raw):
        raw["entry"]["blackout"] = {"enabled": True, "timezone": "UTC", "start": "FRI 12:00", "end": "MON 00:00"}

    with pytest.raises(ValueError, match="entry.blackout"):
        load_config(_write_variant(tmp_path, mutate))


@pytest.mark.parametrize(
    "section,key,value,match",
    [
        ("signal", "interval", "15m", "signal.interval"),
        ("signal", "factor_interval", "4h", "signal.factor_interval"),
        ("signal", "bar_source", "klines", "signal.bar_source"),
        ("entry", "cross_source", "MEMORY", "entry.cross_source"),
        ("entry", "same_symbol_cooldown_minutes", 0, "same_symbol_cooldown_minutes"),
    ],
)
def test_production_freeze_rejects_signal_drift(tmp_path: Path, section, key, value, match):
    def mutate(raw):
        raw[section][key] = value

    with pytest.raises(ValueError, match=match):
        load_config(_write_variant(tmp_path, mutate))


def test_production_freeze_rejects_f4h_hold_change(tmp_path: Path):
    def mutate(raw):
        raw["strategies"]["F4H"]["max_hold_hours"] = 96

    with pytest.raises(ValueError, match="max_hold_hours"):
        load_config(_write_variant(tmp_path, mutate))


def test_production_config_loads_and_matches_freeze():
    cfg = load_config()
    assert cfg["live"]["strategy"] == PRODUCTION_STRATEGY
    assert cfg["signal"]["momentum_profile"] == PRODUCTION_ENTRY_PROFILE
    assert cfg["entry"]["mode"] == PRODUCTION_ENTRY_PROFILE
    assert float(cfg["entry"]["long_threshold"]) == pytest.approx(PRODUCTION_THRESHOLD)
    assert cfg["entry"]["late_entry_enabled"] is False
    assert cfg["live"]["selector"] in (None, "null", "")
    assert PRODUCTION_STRATEGY == "F4H"
    f4h = get_strategy("F4H", cfg["strategies"])
    assert f4h.arm_sl_activation_trail == pytest.approx(0.25)
    assert f4h.activation == pytest.approx(0.30)
    assert f4h.trail_distance_bips == 0 and f4h.fixed_take_profit
    assert f4h.max_hold_hours == pytest.approx(120)
    fp = cfg["_production_fingerprint"]
    assert fp["entry_blackout"] == "OFF"
    assert (fp["signal_interval"], fp["factor_interval"], fp["bar_source"]) == ("4h", "15m", "1m_ratio")
    assert fp["lookback_bars"] == 400
    assert fp["cross_source"] == "BAR" and fp["same_symbol_cooldown_minutes"] == 120
    assert cfg["safety_manager"]["defensive_enabled"] is False
    assert float(cfg["risk"]["max_loss_per_trade"]) == pytest.approx(PRODUCTION_ALLOC * 0.25)
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
