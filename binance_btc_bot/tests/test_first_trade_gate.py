"""First-trade oneshot gate tests — must refuse live on failed Stage-6 gate."""

from __future__ import annotations

import os
import unittest

from binance_btc_bot.config_loader import load_config
from binance_btc_bot.execution.first_trade import FirstTradeController, build_first_trade_config
from binance_btc_bot.portfolio.manager import validate_portfolio_config


class TestFirstTradeGate(unittest.TestCase):
    def test_default_config_still_dry(self):
        cfg = load_config()
        self.assertFalse(cfg["live"]["enabled"])
        self.assertTrue(cfg["live"]["dry_run"])
        self.assertEqual(cfg["portfolio"]["max_simultaneous_trades"], 8)

    def test_oneshot_overlay_max1(self):
        overlay = build_first_trade_config()
        self.assertTrue(overlay["live"]["enabled"])
        self.assertFalse(overlay["live"]["dry_run"])
        self.assertEqual(overlay["portfolio"]["max_simultaneous_trades"], 1)
        self.assertAlmostEqual(overlay["portfolio"]["allocation_per_trade"], 0.125)
        # Production file unchanged
        cfg = load_config()
        self.assertEqual(cfg["portfolio"]["max_simultaneous_trades"], 8)
        self.assertFalse(cfg["live"]["enabled"])

    def test_max1_portfolio_valid(self):
        cfg = validate_portfolio_config(
            {"max_simultaneous_trades": 1, "allocation_per_trade": 0.125, "max_total_allocation": 1.0}
        )
        self.assertEqual(cfg.max_simultaneous_trades, 1)

    def test_run_refuses_without_authorize(self):
        ctrl = FirstTradeController(load_config())
        r = ctrl.run(authorize_live=False)
        self.assertTrue(r.aborted)
        self.assertFalse(r.armed)
        self.assertIn("authorize-live", r.abort_reason)

    def test_run_refuses_without_env_even_if_authorize_flag(self):
        prev = os.environ.pop("BINANCE_FIRST_TRADE_AUTHORIZED", None)
        try:
            ctrl = FirstTradeController(load_config())
            r = ctrl.run(authorize_live=True)
            self.assertTrue(r.aborted)
            self.assertFalse(r.armed)
            self.assertIn("BINANCE_FIRST_TRADE_AUTHORIZED", r.abort_reason)
        finally:
            if prev is not None:
                os.environ["BINANCE_FIRST_TRADE_AUTHORIZED"] = prev

    def test_run_refuses_when_stage6_gate_fails(self):
        """Even with authorize flags, a failed Stage-6 gate must never arm live writes."""
        from unittest.mock import patch

        os.environ["BINANCE_FIRST_TRADE_AUTHORIZED"] = "true"
        try:
            ctrl = FirstTradeController(load_config())
            fake_gate = {
                "ok": False,
                "failures": ["Ed25519 authentication=FAIL"],
                "results": {},
                "message": "DO NOT ENABLE LIVE: Ed25519 authentication=FAIL",
            }
            with patch.object(
                FirstTradeController,
                "authorize_or_abort",
                return_value=(False, None, fake_gate),
            ):
                r = ctrl.run(authorize_live=True)
            self.assertTrue(r.aborted)
            self.assertFalse(r.armed)
            self.assertIn("DO NOT ENABLE LIVE", r.abort_reason)
        finally:
            os.environ.pop("BINANCE_FIRST_TRADE_AUTHORIZED", None)


if __name__ == "__main__":
    unittest.main()
