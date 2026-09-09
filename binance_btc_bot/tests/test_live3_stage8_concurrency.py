"""Stage 8 — expand LIVE concurrency hard max from 3 → 8 (no real orders)."""

from __future__ import annotations

import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from binance_btc_bot.control.hourly_report import build_hourly_report_data
from binance_btc_bot.control.runtime import RuntimeController, RuntimeStateStore
from binance_btc_bot.control.telegram_control import format_status
from binance_btc_bot.execution.live3 import (
    LIVE3_ALLOC,
    LIVE3_MAX,
    LIVE3_RISK,
    LIVE3_THRESHOLD,
    LIVE3_TOTAL_CAP,
    build_live3_target_config,
    format_live3_config_updated_message,
    seed_live3_runtime_state,
)
from binance_btc_bot.notifications.telegram_reports import format_hourly_report
from binance_btc_bot.portfolio.manager import PortfolioManager
from binance_btc_bot.strategy.trails import get_strategy


class TestStage8MaxEight(unittest.TestCase):
    def test_01_max_8_accepted(self):
        cfg = build_live3_target_config()
        self.assertEqual(cfg["portfolio"]["max_simultaneous_trades"], 8)
        self.assertEqual(LIVE3_MAX, 8)
        pm = PortfolioManager.from_config(cfg)
        self.assertEqual(pm.max_simultaneous_trades, 8)

    def test_02_zero_of_eight_entry_allowed(self):
        pm = PortfolioManager.from_config(build_live3_target_config())
        self.assertEqual(pm.slots_used(), 0)
        self.assertTrue(pm.try_reserve("ETHBTC").ok)

    def test_03_seven_of_eight_entry_allowed(self):
        pm = PortfolioManager.from_config(build_live3_target_config())
        for i in range(7):
            self.assertTrue(pm.try_reserve(f"S{i}BTC").ok)
        self.assertTrue(pm.try_reserve("EIGHTHBTC").ok)
        self.assertEqual(pm.slots_used(), 8)

    def test_04_eight_of_eight_entry_blocked(self):
        pm = PortfolioManager.from_config(build_live3_target_config())
        for i in range(8):
            self.assertTrue(pm.try_reserve(f"S{i}BTC").ok)
        blocked = pm.try_reserve("NINTHBTC")
        self.assertFalse(blocked.ok)
        self.assertEqual(blocked.reason, "MAX_OPEN_TRADES")

    def test_05_simultaneous_entries_cannot_create_nine(self):
        pm = PortfolioManager.from_config(build_live3_target_config())
        for i in range(7):
            r = pm.try_reserve(f"S{i}BTC")
            pm.mark_protected(r.reservation.reservation_id, f"t{i}")
        results: list = []
        barrier = threading.Barrier(3)

        def _claim(sym: str) -> None:
            barrier.wait()
            results.append(pm.try_reserve(sym))

        with ThreadPoolExecutor(max_workers=3) as pool:
            futs = [pool.submit(_claim, f"RACE{i}BTC") for i in range(3)]
            for f in futs:
                f.result()
        oks = [r for r in results if r.ok]
        fails = [r for r in results if not r.ok]
        self.assertEqual(len(oks), 1)
        self.assertEqual(len(fails), 2)
        self.assertTrue(all(f.reason == "MAX_OPEN_TRADES" for f in fails))
        self.assertEqual(pm.slots_used(), 8)

    def test_06_existing_positions_unchanged_when_max_changes(self):
        cfg = build_live3_target_config()
        cfg["portfolio"]["max_simultaneous_trades"] = 3
        cfg["portfolio"]["max_total_allocation"] = 0.375
        pm = PortfolioManager.from_config(cfg)
        r1 = pm.try_reserve("INJBTC")
        self.assertTrue(r1.ok)
        rid = r1.reservation.reservation_id
        pm.mark_protected(rid, "inj1")
        before_sym = set(pm.open_symbols())
        before_used = pm.slots_used()
        before_tid = pm._reservations[rid].trade_id
        pm.set_max_simultaneous_trades(8)
        self.assertEqual(pm.max_simultaneous_trades, 8)
        self.assertEqual(pm.slots_used(), before_used)
        self.assertEqual(pm.open_symbols(), before_sym)
        self.assertEqual(pm._reservations[rid].trade_id, before_tid)

    def test_07_telegram_status_shows_x_of_8(self):
        with TemporaryDirectory() as td:
            db = Path(td) / "bot.sqlite3"
            db.write_text("", encoding="utf-8")
            path = seed_live3_runtime_state(db)
            ctrl = RuntimeController(RuntimeStateStore(path), authorized_chat_id="1")
            text = format_status({"open_count": 2, "safety_halted": False}, ctrl)
            self.assertIn("2/8", text)
            self.assertEqual(ctrl.state.max_simultaneous_trades, 8)

    def test_08_hourly_report_shows_x_of_8(self):
        with TemporaryDirectory() as td:
            db = Path(td) / "bot.sqlite3"
            db.write_text("", encoding="utf-8")
            path = seed_live3_runtime_state(db)
            ctrl = RuntimeController(RuntimeStateStore(path), authorized_chat_id="1")
            view = {
                "open_count": 3,
                "open_trades": [],
                "safety_halted": False,
                "equity_btc": 0.01,
                "btc_free": 0.01,
                "btc_locked": 0.0,
                "slots_remaining": 5,
            }
            data = build_hourly_report_data(ctrl, view)
            self.assertEqual(data["system"]["max_trades"], 8)
            self.assertEqual(data["system"]["open_count"], 3)
            msg = format_hourly_report(view, data)
            self.assertIn("3/8", msg)

    def test_09_restart_preserves_max_8(self):
        with TemporaryDirectory() as td:
            db = Path(td) / "bot.sqlite3"
            db.write_text("", encoding="utf-8")
            path = seed_live3_runtime_state(db)
            ctrl1 = RuntimeController(RuntimeStateStore(path), authorized_chat_id="1")
            self.assertEqual(ctrl1.state.max_simultaneous_trades, 8)
            ctrl2 = RuntimeController(RuntimeStateStore(path), authorized_chat_id="1")
            self.assertEqual(ctrl2.state.max_simultaneous_trades, 8)
            self.assertEqual(ctrl2.state.mode, "RUNNING")

    def test_10_reconciliation_preserves_max_8(self):
        pm = PortfolioManager.from_config(build_live3_target_config())
        self.assertEqual(pm.max_simultaneous_trades, 8)
        # Rehydrate with fewer opens must keep capacity at 8
        trades = [
            {
                "trade_id": f"t{i}",
                "symbol": f"S{i}BTC",
                "status": "PROTECTED",
                "btc_value": 0.001,
            }
            for i in range(2)
        ]
        pm.rehydrate_from_trades(trades)
        self.assertEqual(pm.slots_used(), 2)
        self.assertEqual(pm.max_simultaneous_trades, 8)
        self.assertTrue(pm.try_reserve("NEARBTC").ok)

    def test_11_recovery_uses_max_8(self):
        pm = PortfolioManager.from_config(build_live3_target_config())
        trades = [
            {
                "trade_id": f"t{i}",
                "symbol": f"S{i}BTC",
                "status": "PROTECTED",
                "btc_value": 0.001,
            }
            for i in range(8)
        ]
        pm.rehydrate_from_trades(trades)
        self.assertEqual(pm.slots_used(), 8)
        blocked = pm.try_reserve("EXTRA")
        self.assertFalse(blocked.ok)
        self.assertEqual(blocked.reason, "MAX_OPEN_TRADES")

    def test_12_no_hardcoded_live3_max_3_constant(self):
        self.assertEqual(LIVE3_MAX, 8)
        self.assertAlmostEqual(LIVE3_TOTAL_CAP, 1.0)
        # Overlay must not still freeze Stage-7 37.5% total
        cfg = build_live3_target_config()
        self.assertNotAlmostEqual(float(cfg["portfolio"]["max_total_allocation"]), 0.375)

    def test_13_live_true_in_overlay(self):
        cfg = build_live3_target_config()
        self.assertTrue(cfg["live"]["enabled"])

    def test_14_dry_run_false_in_overlay(self):
        cfg = build_live3_target_config()
        self.assertFalse(cfg["live"]["dry_run"])

    def test_15_live_t4_and_frozen_t1_geometry(self):
        cfg = build_live3_target_config()
        self.assertEqual(cfg["live"]["strategy"], "T4")
        t4 = get_strategy("T4")
        self.assertAlmostEqual(t4.activation, 0.02)
        self.assertAlmostEqual(t4.trail_distance, 0.005)
        self.assertAlmostEqual(t4.arm_sl_activation_trail, 0.02)
        t1 = get_strategy("T1")
        self.assertAlmostEqual(t1.activation, 0.0075)
        self.assertAlmostEqual(t1.trail_distance, 0.0025)
        self.assertAlmostEqual(t1.arm_sl_activation_trail, 0.0075)

    def test_16_selector_none(self):
        cfg = build_live3_target_config()
        self.assertIsNone(cfg["live"]["selector"])
        with TemporaryDirectory() as td:
            db = Path(td) / "bot.sqlite3"
            db.write_text("", encoding="utf-8")
            ctrl = RuntimeController(
                RuntimeStateStore(seed_live3_runtime_state(db)), authorized_chat_id="1"
            )
            self.assertEqual(ctrl.state.selector, "NONE")

    def test_17_no_real_test_orders_generated(self):
        # Pure unit path: overlay + reserve never touches exchange.
        cfg = build_live3_target_config()
        pm = PortfolioManager.from_config(cfg)
        self.assertTrue(pm.try_reserve("DOTBTC").ok)
        msg = format_live3_config_updated_message(old_max=3, new_max=8)
        self.assertIn("12.5%", msg)
        self.assertIn("Strategy: T4", msg)
        self.assertAlmostEqual(LIVE3_ALLOC, 0.125)
        self.assertAlmostEqual(LIVE3_RISK, 0.005)
        self.assertAlmostEqual(LIVE3_THRESHOLD, 0.65)

    def test_engine_rejects_non_eight_live3(self):
        from binance_btc_bot.execution.engine import BinanceBotEngine

        cfg = build_live3_target_config()
        cfg["portfolio"]["max_simultaneous_trades"] = 4
        with patch.dict(
            "os.environ",
            {"BINANCE_LIVE3_AUTHORIZED": "true", "DRY_RUN": "true"},
            clear=False,
        ):
            with self.assertRaises(RuntimeError) as ctx:
                BinanceBotEngine(cfg, allow_live_writes=True)
        self.assertIn("max_simultaneous_trades=8", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
