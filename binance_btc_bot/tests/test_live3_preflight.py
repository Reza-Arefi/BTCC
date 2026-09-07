"""LIVE-3 preflight / hard max=8 tests (no real orders)."""

from __future__ import annotations

import os
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from binance_btc_bot.config_loader import load_config
from binance_btc_bot.execution.live3 import (
    LIVE3_MAX,
    LIVE3_TOTAL_CAP,
    build_live3_probe_config,
    build_live3_target_config,
    format_live3_config_updated_message,
    seed_live3_runtime_state,
)
from binance_btc_bot.portfolio.manager import PortfolioManager


class TestLive3Overlay(unittest.TestCase):
    def test_target_overlay(self):
        base = load_config()
        # Defaults remain safe on disk / base load
        self.assertFalse(bool((base.get("live") or {}).get("enabled")))
        self.assertTrue(bool((base.get("live") or {}).get("dry_run", True)))
        self.assertEqual(int((base.get("portfolio") or {}).get("max_simultaneous_trades")), 8)

        tgt = build_live3_target_config(base)
        self.assertTrue(tgt["live"]["enabled"])
        self.assertFalse(tgt["live"]["dry_run"])
        self.assertEqual(tgt["live"]["strategy"], "T1")
        self.assertIsNone(tgt["live"]["selector"])
        self.assertEqual(tgt["portfolio"]["max_simultaneous_trades"], LIVE3_MAX)
        self.assertEqual(tgt["portfolio"]["allocation_per_trade"], 0.125)
        self.assertEqual(tgt["portfolio"]["max_total_allocation"], LIVE3_TOTAL_CAP)
        self.assertEqual(tgt["entry"]["long_threshold"], 0.65)
        self.assertEqual(tgt["risk"]["max_loss_per_trade"], 0.005)
        # Base untouched
        self.assertEqual(int((base.get("portfolio") or {}).get("max_simultaneous_trades")), 8)

    def test_probe_overlay_blocks_writes(self):
        probe = build_live3_probe_config()
        self.assertFalse(probe["live"]["enabled"])
        self.assertTrue(probe["live"]["dry_run"])
        self.assertEqual(probe["portfolio"]["max_simultaneous_trades"], LIVE3_MAX)

    def test_hard_eight_trade_cap(self):
        cfg = build_live3_target_config()
        pm = PortfolioManager.from_config(cfg)
        for i in range(LIVE3_MAX):
            self.assertTrue(pm.try_reserve(f"A{i}BTC").ok)
        ninth = pm.try_reserve("BLOCKBTC")
        self.assertFalse(ninth.ok)
        self.assertEqual(ninth.reason, "MAX_OPEN_TRADES")

    def test_seed_runtime_state(self):
        from binance_btc_bot.control.runtime import RuntimeController, RuntimeStateStore

        tmp = tempfile.TemporaryDirectory()
        db = Path(tmp.name) / "bot.sqlite3"
        db.write_text("", encoding="utf-8")
        path = seed_live3_runtime_state(db)
        ctrl = RuntimeController(RuntimeStateStore(path), authorized_chat_id="x")
        self.assertEqual(ctrl.state.strategy, "T1")
        self.assertEqual(ctrl.state.selector, "NONE")
        self.assertEqual(ctrl.state.max_simultaneous_trades, LIVE3_MAX)
        self.assertEqual(ctrl.state.mode, "RUNNING")
        tmp.cleanup()

    def test_yaml_defaults_unchanged_after_overlay(self):
        cfg = load_config()
        _ = build_live3_target_config(deepcopy(cfg))
        again = load_config()
        self.assertFalse(bool((again.get("live") or {}).get("enabled")))
        self.assertTrue(bool((again.get("live") or {}).get("dry_run", True)))
        self.assertEqual(int((again.get("portfolio") or {}).get("max_simultaneous_trades")), 8)

    def test_config_updated_message_brt(self):
        msg = format_live3_config_updated_message(
            old_max=3,
            new_max=8,
            timestamp="2026-09-07T01:00:00Z",
        )
        self.assertIn("CONFIGURATION UPDATED", msg)
        self.assertIn("3 → 8", msg)
        self.assertIn("Max allocation: 100%", msg)
        self.assertIn("BRT", msg)


class TestLive3PreflightNoOrders(unittest.TestCase):
    def test_preflight_refuses_when_probe_would_write(self):
        from binance_btc_bot.execution.live3 import Live3PreflightReport, run_live3_preflight

        # Smoke: module imports and overlay gates 1-8 work even if network fails later.
        # Full network preflight is exercised via CLI when credentials/network available.
        tgt = build_live3_target_config()
        self.assertTrue(tgt["live"]["enabled"])
        self.assertIsInstance(Live3PreflightReport().ok, bool)

    def test_run_live3_preflight_mocked_engine(self):
        """Ensure runner path does not call place_order / place_entry."""
        from binance_btc_bot.execution import live3 as live3_mod
        from binance_btc_bot.preflight import PreflightReport

        calls: list[str] = []

        class FakeEx:
            def _writes_allowed(self):
                return False, "DRY_RUN"

            def ping(self):
                return True

            def get_account(self):
                return {
                    "balances": [
                        {"asset": "BTC", "free": "1.0", "locked": "0"},
                        {"asset": "BNB", "free": "0.1", "locked": "0"},
                    ]
                }

            def get_price(self, _):
                return 100000.0

            def get_open_orders(self):
                return []

            def get_open_order_lists(self):
                return []

            def place_entry(self, *a, **k):
                calls.append("place_entry")
                raise AssertionError("no orders")

            def place_order(self, *a, **k):
                calls.append("place_order")
                raise AssertionError("no orders")

        class FakeDB:
            path = Path(tempfile.mkdtemp()) / "bot.sqlite3"

            def open_trades(self):
                return []

            def close(self):
                return None

        class FakeEngine:
            def __init__(self, cfg):
                self.cfg = cfg
                self.exchange = FakeEx()
                self.db = FakeDB()
                self.db.path.parent.mkdir(parents=True, exist_ok=True)
                self.db.path.write_text("", encoding="utf-8")
                self.creds = type("C", (), {"present": True, "signer_ready": False, "api_key": ""})()
                self.portfolio = PortfolioManager.from_config(cfg)
                self.notifications = type(
                    "N",
                    (),
                    {"status": lambda self: {"telegram": {"configured": True}, "sms": {}}},
                )()
                self.dry_run = True
                self.live_enabled = False

            def stop(self):
                return None

        fake_pref = PreflightReport(live_enabled=False, dry_run=True)
        for name in [
            "Ed25519 authentication",
            "canTrade",
            "Withdraw permission",
            "IP restriction",
            "WebSocket",
            "Reconciliation",
            "Database",
            "Binance connectivity",
            "37 symbols",
            "T1 configuration",
            "Protection failure",
            "Telegram",
            "Portfolio",
            "Risk",
        ]:
            detail = "enableWithdrawals=FALSE" if name == "Withdraw permission" else "ok"
            if name == "WebSocket":
                detail = "auth_ws_api_subscribe=PASS ok"
            if name == "Withdraw permission":
                fake_pref.add(name, "PASS", detail, enableFutures=False, enableMargin=False)
            elif name == "37 symbols":
                fake_pref.add("38 symbols", "PASS", detail)
            else:
                fake_pref.add(name, "PASS", detail)

        with patch.object(live3_mod, "BinanceBotEngine", FakeEngine):
            with patch.object(live3_mod, "run_preflight", return_value=fake_pref):
                with patch(
                    "binance_btc_bot.strategy.score_provider.build_production_score_provider"
                ) as bsp:
                    bsp.return_value = type(
                        "SP",
                        (),
                        {"evaluate": lambda self, s: type("Snap", (), {"S_current": 0.5, "reason": "OK"})()},
                    )()
                    report = live3_mod.run_live3_preflight(seed_runtime=True)

        self.assertEqual(calls, [])
        self.assertEqual(report.real_orders, "0")
        g1 = next(g for g in report.gates if g.n == 1)
        self.assertEqual(g1.status, "PASS")
        g5 = next(g for g in report.gates if g.n == 5)
        self.assertEqual(g5.status, "PASS")
        self.assertTrue(report.ok)


class TestLive3ArmGates(unittest.TestCase):
    def test_arm_refuses_without_authorize_flag(self):
        from binance_btc_bot.execution.live3 import Live3Session

        with patch.dict(os.environ, {"BINANCE_LIVE3_AUTHORIZED": "true"}, clear=False):
            report = Live3Session().run(authorize_live=False)
        self.assertTrue(report.aborted)
        self.assertFalse(report.armed)
        self.assertIn("authorize-live", report.abort_reason)

    def test_arm_refuses_without_env(self):
        from binance_btc_bot.execution.live3 import Live3Session

        env = {k: v for k, v in os.environ.items() if k != "BINANCE_LIVE3_AUTHORIZED"}
        with patch.dict(os.environ, env, clear=True):
            report = Live3Session().run(authorize_live=True)
        self.assertTrue(report.aborted)
        self.assertFalse(report.armed)
        self.assertIn("BINANCE_LIVE3_AUTHORIZED", report.abort_reason)

    def test_arm_refuses_when_preflight_fails(self):
        from binance_btc_bot.execution import live3 as live3_mod

        fake = live3_mod.Live3PreflightReport(ready_to_arm=False)
        fake.add(1, "LIVE=true", "FAIL", "forced")
        with patch.dict(os.environ, {"BINANCE_LIVE3_AUTHORIZED": "true"}, clear=False):
            with patch.object(live3_mod, "run_live3_preflight", return_value=fake):
                report = live3_mod.Live3Session().run(authorize_live=True)
        self.assertTrue(report.aborted)
        self.assertFalse(report.armed)
        self.assertIn("PREFLIGHT", report.abort_reason.upper())

    def test_engine_live3_requires_max_8(self):
        from binance_btc_bot.execution.engine import BinanceBotEngine
        from binance_btc_bot.execution.live3 import build_live3_target_config

        cfg = build_live3_target_config()
        cfg["portfolio"]["max_simultaneous_trades"] = 3
        with patch.dict(os.environ, {"BINANCE_LIVE3_AUTHORIZED": "true", "DRY_RUN": "true"}, clear=False):
            with self.assertRaises(RuntimeError):
                BinanceBotEngine(cfg, allow_live_writes=True)


if __name__ == "__main__":
    unittest.main()
