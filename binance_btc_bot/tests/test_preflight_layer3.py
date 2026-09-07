"""Layer 3 — production preflight / Stage-6 readiness tests (no real orders)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from binance_btc_bot.accounting.equity import compute_equity_btc, worked_risk_example
from binance_btc_bot.config_loader import load_config
from binance_btc_bot.exchange.base import AccountSnapshot, Balance, OrderResult, SymbolInfo
from binance_btc_bot.execution.dry_broker import DryRunBroker
from binance_btc_bot.execution.fills import aggregate_fills_from_order
from binance_btc_bot.execution.lifecycle import OrderLifecycle
from binance_btc_bot.execution.trailing import TrailingExecutor
from binance_btc_bot.portfolio.manager import PortfolioManager
from binance_btc_bot.preflight import T1_BINANCE_SEMANTICS, run_preflight
from binance_btc_bot.risk.safety import SafetyState, SafetySystem
from binance_btc_bot.storage.database import BotDatabase
from binance_btc_bot.strategy.trails import get_strategy, map_trail_to_binance_oco


def _meta(symbol: str = "ETHBTC") -> SymbolInfo:
    return SymbolInfo(
        symbol=symbol,
        status="TRADING",
        base_asset=symbol.replace("BTC", ""),
        quote_asset="BTC",
        quantity_step=0.0001,
        min_quantity=0.0001,
        max_quantity=1000.0,
        price_tick=0.000001,
        min_notional=0.0001,
        order_types=("LIMIT", "MARKET", "STOP_LOSS", "TAKE_PROFIT"),
        oco_allowed=True,
        min_trailing_above_delta=10,
        max_trailing_above_delta=2000,
        min_trailing_below_delta=10,
        max_trailing_below_delta=2000,
    )


class TestEquityCalculation(unittest.TestCase):
    def test_never_uses_usdt_alone_as_equity(self):
        acct = AccountSnapshot(
            balances={
                "BTC": Balance("BTC", 1.0, 0.25),
                "USDT": Balance("USDT", 10_000.0, 0.0),
                "ETH": Balance("ETH", 10.0, 0.0),
            }
        )
        eq = compute_equity_btc(acct, btc_usdt=100_000.0, alt_prices_btc={"ETH": 0.05})
        # BTC 1.25 + USDT 0.1 + ETH 0.5 = 1.85
        self.assertAlmostEqual(eq.total_equity_btc, 1.85)
        self.assertAlmostEqual(eq.available_btc, 1.0)
        self.assertAlmostEqual(eq.trading_capital_btc, 1.85)
        self.assertNotAlmostEqual(eq.total_equity_btc, eq.usdt_free)
        self.assertNotAlmostEqual(eq.total_equity_btc, eq.usdt_as_btc)

    def test_available_is_btc_free_only(self):
        acct = AccountSnapshot(balances={"BTC": Balance("BTC", 0.4, 0.6)})
        eq = compute_equity_btc(acct, btc_usdt=100_000.0)
        self.assertAlmostEqual(eq.available_btc, 0.4)
        self.assertAlmostEqual(eq.total_equity_btc, 1.0)


class TestRiskWorkedExample(unittest.TestCase):
    def test_half_percent_is_loss_budget_not_stop(self):
        ex = worked_risk_example(equity_btc=1.0)
        self.assertEqual(ex["NOT_meaning"], "stop distance = 0.5%")
        self.assertAlmostEqual(ex["t1_hard_sl_pct"], 0.0075)
        self.assertAlmostEqual(ex["risk_budget_btc"], 0.005)
        self.assertEqual(ex["binding_constraint"], "ALLOCATION")
        self.assertLess(ex["planned_loss_pct_of_equity"], 0.005)
        self.assertAlmostEqual(ex["eight_trades_max_allocation"], 1.0)


class TestT1OcoAgainstDocs(unittest.TestCase):
    def test_construction_matches_spot_api(self):
        strategy = get_strategy("T1")
        mapping = map_trail_to_binance_oco(
            strategy=strategy,
            symbol="ETHBTC",
            entry_price=0.05,
            quantity=2.0,
            symbol_info=_meta(),
        )
        req = mapping.request
        self.assertTrue(mapping.allowed)
        self.assertEqual(req.side, "SELL")
        self.assertEqual(req.above_type, "TAKE_PROFIT")
        self.assertEqual(req.above_trailing_delta, 25)
        self.assertEqual(req.below_type, "STOP_LOSS")
        self.assertEqual(req.quantity, 2.0)
        self.assertAlmostEqual(req.above_stop_price / 0.05, 1.0075, places=4)
        self.assertAlmostEqual(req.below_stop_price / 0.05, 0.9925, places=4)
        self.assertEqual(T1_BINANCE_SEMANTICS["live_order_submit"], False)
        self.assertIn("research_vs_binance", T1_BINANCE_SEMANTICS)


class TestEightSlotRestart(unittest.TestCase):
    def test_rehydrate_eight_blocks_ninth(self):
        cfg = load_config()
        pm = PortfolioManager.from_config(cfg)
        trades = [{"trade_id": f"t{i}", "symbol": f"A{i}BTC", "status": "DRY_RUN_PROTECTED"} for i in range(8)]
        rh = pm.rehydrate_from_trades(trades)
        self.assertEqual(rh["slots_used"], 8)
        self.assertFalse(pm.try_reserve("NINTHBTC").ok)

    def test_rehydrate_seven_allows_one(self):
        cfg = load_config()
        pm = PortfolioManager.from_config(cfg)
        trades = [{"trade_id": f"t{i}", "symbol": f"B{i}BTC", "status": "PROTECTED"} for i in range(7)]
        pm.rehydrate_from_trades(trades)
        self.assertEqual(pm.slots_used(), 7)
        self.assertTrue(pm.try_reserve("NEWBTC").ok)
        self.assertEqual(pm.slots_used(), 8)
        self.assertFalse(pm.try_reserve("TOOMANYBTC").ok)


class TestProtectionFailurePreflight(unittest.TestCase):
    def test_buy_filled_oco_rejected(self):
        cfg = load_config()
        tmp = tempfile.mkdtemp()
        db = BotDatabase(Path(tmp) / "t.sqlite3")
        safety = SafetySystem()
        pm = PortfolioManager.from_config(cfg)
        broker = DryRunBroker()
        broker.oco_behavior = "REJECT"
        broker.allow_emergency_protect = False
        ex = MagicMock()
        ex.get_symbol_info.side_effect = lambda s: _meta(str(s).upper())
        ex.place_trailing_exit.return_value = OrderResult(ok=True, status="DRY_RUN", dry_run=True)
        life = OrderLifecycle(
            ex,
            db,
            safety,
            portfolio=pm,
            trailing=TrailingExecutor(ex, db),
            dry_broker=broker,
            live_enabled=False,
            dry_run=True,
        )
        r = life.run_entry(
            symbol="ETHBTC",
            strategy=get_strategy("T1"),
            price_alt_btc=0.05,
            equity_btc=1.0,
            available_btc=1.0,
        )
        self.assertEqual(r.status, "PROTECTION_FAILED")
        self.assertEqual(safety.state, SafetyState.HALT)
        self.assertEqual(pm.slots_used(), 1)


class TestActualFills(unittest.TestCase):
    def test_weighted_fill_used(self):
        agg = aggregate_fills_from_order(
            symbol="ETHBTC",
            side="BUY",
            order_id="1",
            client_order_id="c",
            status="FILLED",
            executed_qty=None,
            cumulative_quote_qty=None,
            fills_raw=[
                {"price": "0.10", "qty": "1", "commission": "0.001", "commissionAsset": "BTC"},
                {"price": "0.14", "qty": "3", "commission": "0.001", "commissionAsset": "BTC"},
            ],
        )
        self.assertAlmostEqual(agg.avg_price, 0.13)
        self.assertAlmostEqual(agg.executed_qty, 4.0)


class TestEmergencyUncertainty(unittest.TestCase):
    def test_halt_and_idempotent_buy(self):
        safety = SafetySystem()
        safety.halt("API_FAILURE")
        self.assertFalse(safety.allow_new_entries())
        broker = DryRunBroker()
        a = broker.place_market_buy(symbol="ETHBTC", quantity=1, client_order_id="x", ref_price=0.05)
        b = broker.place_market_buy(symbol="ETHBTC", quantity=1, client_order_id="x", ref_price=0.05)
        self.assertEqual(a.order_id, b.order_id)

    def test_notification_failure_isolated(self):
        from binance_btc_bot.notifications.manager import NotificationManager
        from binance_btc_bot.notifications.base import NotificationResult

        class Boom:
            name = "telegram"

            def configured(self):
                return True

            def send(self, event):
                raise RuntimeError("tg down")

            def status(self):
                return {"configured": True}

        class OkSms:
            name = "sms"

            def configured(self):
                return True

            def send(self, event):
                return NotificationResult(ok=True, channel="sms")

            def status(self):
                return {"configured": True}

        n = NotificationManager(telegram=Boom(), sms=OkSms())
        # Must not raise
        results = n.notify_critical("HALT", "test")
        self.assertTrue(any(r.channel == "sms" and r.ok for r in results))


class TestApiKeyWithdrawPermission(unittest.TestCase):
    def test_account_canWithdraw_true_but_key_enableWithdrawals_false_passes(self):
        from binance_btc_bot.preflight import PreflightReport, PreflightRunner

        cfg = load_config()
        report = PreflightReport()
        engine = MagicMock()
        engine.exchange.get_api_key_restrictions.return_value = {
            "enableWithdrawals": False,
            "enableReading": True,
            "enableSpotAndMarginTrading": True,
            "enableFutures": False,
            "enableMargin": False,
            "enableInternalTransfer": False,
            "permitsUniversalTransfer": False,
            "ipRestrict": True,
        }
        PreflightRunner(cfg)._check_api_key_withdraw_permission(
            report, engine, account_can_withdraw=True
        )
        wd = report.by_name("Withdraw permission")
        self.assertEqual(wd.status, "PASS")
        self.assertIn("enableWithdrawals=FALSE", wd.detail)
        self.assertTrue(wd.data.get("account_canWithdraw_informational"))
        self.assertFalse(wd.data.get("enableWithdrawals"))

    def test_enableWithdrawals_true_fails(self):
        from binance_btc_bot.preflight import PreflightReport, PreflightRunner

        cfg = load_config()
        report = PreflightReport()
        engine = MagicMock()
        engine.exchange.get_api_key_restrictions.return_value = {
            "enableWithdrawals": True,
            "enableReading": True,
            "enableSpotAndMarginTrading": True,
        }
        PreflightRunner(cfg)._check_api_key_withdraw_permission(
            report, engine, account_can_withdraw=False
        )
        wd = report.by_name("Withdraw permission")
        self.assertEqual(wd.status, "FAIL")
        self.assertIn("enableWithdrawals=TRUE", wd.detail)

    def test_missing_enableWithdrawals_field_fails_closed(self):
        from binance_btc_bot.preflight import PreflightReport, PreflightRunner

        cfg = load_config()
        report = PreflightReport()
        engine = MagicMock()
        engine.exchange.get_api_key_restrictions.return_value = {"ipRestrict": True}
        PreflightRunner(cfg)._check_api_key_withdraw_permission(
            report, engine, account_can_withdraw=True
        )
        wd = report.by_name("Withdraw permission")
        self.assertEqual(wd.status, "FAIL")
        self.assertIn("fail closed", wd.detail.lower())


class TestLivePostureLocked(unittest.TestCase):
    def test_config_unchanged(self):
        cfg = load_config()
        self.assertFalse(cfg["live"]["enabled"])
        self.assertTrue(cfg["live"]["dry_run"])
        self.assertEqual(cfg["live"]["strategy"], "T1")
        self.assertIsNone(cfg["live"]["selector"])
        self.assertEqual(cfg["portfolio"]["max_simultaneous_trades"], 8)
        self.assertAlmostEqual(cfg["portfolio"]["allocation_per_trade"], 0.125)
        self.assertAlmostEqual(cfg["risk"]["max_loss_per_trade"], 0.005)


class TestPreflightRunnerSmoke(unittest.TestCase):
    def test_run_preflight_no_orders(self):
        cfg = load_config()
        report = run_preflight(cfg)
        self.assertTrue(report.dry_run)
        self.assertFalse(report.live_enabled)
        self.assertEqual(report.real_orders, "DISABLED")
        dry = report.by_name("Dry-run")
        self.assertIsNotNone(dry)
        self.assertEqual(dry.status, "PASS")
        t1 = report.by_name("T1 configuration")
        self.assertEqual(t1.status, "PASS")
        ws = report.by_name("WebSocket")
        self.assertIsNotNone(ws)
        self.assertIn(ws.status, {"PASS", "RESTRICTED"})
        self.assertIn("auth_ws_api_subscribe", ws.detail or "")
        self.assertNotIn("auth_listenKey_live", ws.detail or "")
        sms = report.by_name("SMS")
        self.assertIsNotNone(sms)
        self.assertEqual(sms.status, "OPTIONAL")
        self.assertIn("NOT CONFIGURED", (sms.detail or "").upper())
        fails = {c.name for c in report.checks if c.status == "FAIL"}
        self.assertEqual(fails, set(), f"unexpected preflight FAILs: {fails}")
        wd = report.by_name("Withdraw permission")
        self.assertIsNotNone(wd)
        self.assertEqual(wd.status, "PASS")
        self.assertIn("enableWithdrawals=FALSE", wd.detail or "")
        gate = report.stage6_live_authorize_gate()
        self.assertNotIn("SMS", " ".join(gate.get("failures") or []))
        # STAGE6_READY depends on live trading-safety gates only (SMS optional).
        self.assertTrue(gate.get("ok"), gate.get("message"))


class TestProtectionFailureSilentNotify(unittest.TestCase):
    def test_preflight_protection_failure_does_not_touch_engine_notifications(self):
        """Regression: synthetic ETHBTC drill must not Telegram UNPROTECTED_POSITION."""
        from binance_btc_bot.preflight import PreflightReport, PreflightRunner

        cfg = load_config()
        live_n = MagicMock()
        live_n.notify_critical = MagicMock()
        live_n.notify_warning = MagicMock()
        live_n.notify_error = MagicMock()
        live_n.notify_info = MagicMock()
        engine = MagicMock()
        engine.notifications = live_n

        report = PreflightReport(live_enabled=False, dry_run=True)
        PreflightRunner(cfg)._check_protection_failure(report, engine)

        check = report.by_name("Protection failure")
        self.assertIsNotNone(check)
        self.assertEqual(check.status, "PASS", check.detail)
        live_n.notify_critical.assert_not_called()
        live_n.notify_warning.assert_not_called()
        live_n.notify_error.assert_not_called()
        live_n.notify_info.assert_not_called()


if __name__ == "__main__":
    unittest.main()
