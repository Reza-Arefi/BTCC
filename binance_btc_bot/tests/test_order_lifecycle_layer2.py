"""Layer 2 — Binance Spot order lifecycle (dry-run; no real orders)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from binance_btc_bot.accounting.btc_accounting import TradeAccounting
from binance_btc_bot.exchange.base import OrderResult, SymbolInfo
from binance_btc_bot.execution.dry_broker import DryRunBroker
from binance_btc_bot.execution.fills import FillLeg, aggregate_fills_from_order, weighted_average_fill
from binance_btc_bot.execution.lifecycle import OrderLifecycle
from binance_btc_bot.execution.recovery import RecoveryManager
from binance_btc_bot.execution.trailing import TrailingExecutor
from binance_btc_bot.market_data.user_stream import UserDataStream
from binance_btc_bot.notifications.base import NotificationResult
from binance_btc_bot.notifications.manager import NotificationManager
from binance_btc_bot.portfolio.manager import PortfolioConfig, PortfolioManager
from binance_btc_bot.portfolio.trade_state import TradeStatus
from binance_btc_bot.risk.safety import SafetyState, SafetySystem
from binance_btc_bot.storage.database import BotDatabase
from binance_btc_bot.strategy.trails import get_strategy


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


class RecordingChannel:
    def __init__(self, name: str) -> None:
        self.name = name
        self.sent: list[dict] = []

    def send(self, event) -> NotificationResult:
        self.sent.append(
            {"event": event.event, "severity": event.severity.value, "message": event.message}
        )
        return NotificationResult(ok=True, channel=self.name)

    def status(self) -> dict:
        return {"configured": True, "enabled": True, "name": self.name}


def _notif() -> NotificationManager:
    return NotificationManager(telegram=RecordingChannel("telegram"), sms=RecordingChannel("sms"))


def _exchange() -> MagicMock:
    ex = MagicMock()
    ex.get_symbol_info.side_effect = lambda s: _meta(str(s).upper())
    ex.place_entry.return_value = OrderResult(ok=True, status="DRY_RUN", dry_run=True)
    ex.place_trailing_exit.return_value = OrderResult(
        ok=True, status="DRY_RUN", dry_run=True, order_id=None, client_order_id="t_test"
    )
    ex.get_order.return_value = OrderResult(ok=False, status="UNKNOWN")
    ex.get_open_orders.return_value = []
    ex.get_open_order_lists.return_value = []
    return ex


def _lifecycle(**kw):
    tmp = tempfile.mkdtemp()
    db = BotDatabase(Path(tmp) / "bot.sqlite3")
    safety = SafetySystem()
    portfolio = PortfolioManager(
        PortfolioConfig(max_simultaneous_trades=8, allocation_per_trade=0.125, max_total_allocation=1.0)
    )
    broker = DryRunBroker()
    notifications = kw.pop("notifications", _notif())
    exchange = kw.pop("exchange", _exchange())
    trailing = TrailingExecutor(exchange, db, notifications=notifications)
    life = OrderLifecycle(
        exchange,
        db,
        safety,
        portfolio=portfolio,
        trailing=trailing,
        notifications=notifications,
        dry_broker=broker,
        live_enabled=False,
        dry_run=True,
        **kw,
    )
    return life, db, safety, portfolio, broker, notifications


class TestFillAggregation(unittest.TestCase):
    def test_weighted_average_fill(self):
        legs = (FillLeg(price=10.0, qty=1.0), FillLeg(price=12.0, qty=3.0))
        avg, qty, quote = weighted_average_fill(legs)
        self.assertAlmostEqual(qty, 4.0)
        self.assertAlmostEqual(quote, 46.0)
        self.assertAlmostEqual(avg, 11.5)

    def test_aggregate_from_multiple_fills(self):
        agg = aggregate_fills_from_order(
            symbol="ETHBTC",
            side="BUY",
            order_id="1",
            client_order_id="c1",
            status="FILLED",
            executed_qty=None,
            cumulative_quote_qty=None,
            fills_raw=[
                {"price": "0.05", "qty": "2", "commission": "0.0001", "commissionAsset": "BTC"},
                {"price": "0.07", "qty": "2", "commission": "0.0001", "commissionAsset": "BTC"},
            ],
        )
        self.assertTrue(agg.ok)
        self.assertAlmostEqual(agg.executed_qty, 4.0)
        self.assertAlmostEqual(agg.avg_price, 0.06)
        self.assertAlmostEqual(agg.commission_btc, 0.0002)


class TestOrderLifecycleHappyPath(unittest.TestCase):
    def test_market_buy_actual_fill_oco_protected(self):
        life, db, safety, pm, broker, _ = _lifecycle()
        strategy = get_strategy("T1")
        broker.buy_behavior = "FILL"
        broker.oco_behavior = "ACCEPT"
        r = life.run_entry(
            symbol="ETHBTC",
            strategy=strategy,
            price_alt_btc=0.05,
            equity_btc=1.0,
            available_btc=1.0,
        )
        self.assertTrue(r.ok, r.reason)
        self.assertIn("BUY_FILLED", r.events)
        self.assertIn("OCO_ACCEPTED", r.events)
        self.assertIn("PROTECTED", r.events)
        self.assertEqual(r.status, TradeStatus.PROTECTED.value)
        self.assertIsNotNone(r.fill)
        # Actual fill must differ from request price due to slip + multi-leg avg
        self.assertNotAlmostEqual(r.fill.avg_price, 0.05)
        # T1 geometry on actual fill
        # T1 geometry on actual fill (tick-rounded Binance mapping)
        self.assertEqual(r.oco.trail_bips, 25)
        self.assertGreater(r.oco.activation_price, r.fill.avg_price)
        self.assertLess(r.oco.initial_stop, r.fill.avg_price)
        self.assertAlmostEqual(r.oco.activation_price / r.fill.avg_price, 1.0075, places=4)
        self.assertAlmostEqual(r.oco.initial_stop / r.fill.avg_price, 0.9925, places=4)
        tr = db.get_trade(r.trade_id)
        self.assertEqual(tr["status"], "DRY_RUN_PROTECTED")
        self.assertAlmostEqual(tr["entry_price"], r.fill.avg_price)
        self.assertEqual(pm.slots_used(), 1)
        self.assertNotEqual(safety.state, SafetyState.HALT)


class TestPartialAndRejectedBuys(unittest.TestCase):
    def test_partial_fill_safe_state(self):
        life, db, safety, pm, broker, n = _lifecycle()
        broker.buy_behavior = "PARTIAL"
        r = life.run_entry(
            symbol="ETHBTC",
            strategy=get_strategy("T1"),
            price_alt_btc=0.05,
            equity_btc=1.0,
            available_btc=1.0,
        )
        self.assertFalse(r.ok)
        self.assertEqual(r.reason, "PARTIAL_FILL_UNSAFE")
        self.assertEqual(safety.state, SafetyState.HALT)
        self.assertEqual(pm.slots_used(), 0)
        self.assertTrue(any(x["event"] == "PARTIAL_FILL_UNSAFE" for x in n.telegram.sent))

    def test_rejected_buy_releases_slot(self):
        life, _, _, pm, broker, _ = _lifecycle()
        broker.buy_behavior = "REJECT"
        r = life.run_entry(
            symbol="ETHBTC",
            strategy=get_strategy("T1"),
            price_alt_btc=0.05,
            equity_btc=1.0,
            available_btc=1.0,
        )
        self.assertFalse(r.ok)
        self.assertIn("REJECTED", r.reason)
        self.assertEqual(pm.slots_used(), 0)

    def test_cancelled_buy(self):
        life, _, _, pm, broker, _ = _lifecycle()
        broker.buy_behavior = "CANCEL"
        r = life.run_entry(
            symbol="ETHBTC",
            strategy=get_strategy("T1"),
            price_alt_btc=0.05,
            equity_btc=1.0,
            available_btc=1.0,
        )
        self.assertFalse(r.ok)
        self.assertIn("CANCELED", r.reason)
        self.assertEqual(pm.slots_used(), 0)

    def test_expired_buy(self):
        life, _, _, pm, broker, _ = _lifecycle()
        broker.buy_behavior = "EXPIRE"
        r = life.run_entry(
            symbol="ETHBTC",
            strategy=get_strategy("T1"),
            price_alt_btc=0.05,
            equity_btc=1.0,
            available_btc=1.0,
        )
        self.assertFalse(r.ok)
        self.assertIn("EXPIRED", r.reason)
        self.assertEqual(pm.slots_used(), 0)


class TestProtectionFailure(unittest.TestCase):
    def test_oco_reject_protection_failed_emergency_recover(self):
        life, db, safety, pm, broker, n = _lifecycle()
        broker.oco_behavior = "REJECT"
        broker.allow_emergency_protect = True
        r = life.run_entry(
            symbol="ETHBTC",
            strategy=get_strategy("T1"),
            price_alt_btc=0.05,
            equity_btc=1.0,
            available_btc=1.0,
        )
        self.assertFalse(r.ok)
        self.assertEqual(r.status, "PROTECTED_EMERGENCY")
        self.assertIn("PROTECTION_FAILED", r.events)
        self.assertIn("EMERGENCY_OK", r.events)
        self.assertEqual(safety.state, SafetyState.HALT)
        self.assertFalse(safety.allow_new_entries())
        # Critical Telegram + SMS
        crit_tg = [x for x in n.telegram.sent if x["severity"] == "CRITICAL"]
        crit_sms = [x for x in n.sms.sent if x["severity"] == "CRITICAL"]
        self.assertTrue(any("PROTECTION_FAILED" in x["event"] for x in crit_tg))
        self.assertTrue(crit_sms)
        # Slot still occupied (inventory protected emergently)
        self.assertEqual(pm.slots_used(), 1)
        # HALT blocks new entries but not recovery
        r2 = life.run_entry(
            symbol="SOLBTC",
            strategy=get_strategy("T1"),
            price_alt_btc=0.01,
            equity_btc=1.0,
            available_btc=1.0,
        )
        self.assertFalse(r2.ok)
        self.assertEqual(r2.reason, "SAFETY_HALT")
        recon = life.reconcile_rest(["ETHBTC"])
        self.assertTrue(recon["ok"])

    def test_protection_failure_no_emergency_stays_failed(self):
        life, db, safety, pm, broker, n = _lifecycle()
        broker.oco_behavior = "REJECT"
        broker.allow_emergency_protect = False
        r = life.run_entry(
            symbol="ETHBTC",
            strategy=get_strategy("T1"),
            price_alt_btc=0.05,
            equity_btc=1.0,
            available_btc=1.0,
        )
        self.assertFalse(r.ok)
        self.assertEqual(r.status, TradeStatus.PROTECTION_FAILED.value)
        self.assertIn("EMERGENCY_FAILED", r.events)
        self.assertEqual(safety.state, SafetyState.HALT)
        self.assertEqual(pm.slots_used(), 1)
        self.assertTrue(any(x["event"] == "UNPROTECTED_POSITION" for x in n.telegram.sent))


class TestOcoExitAndAccounting(unittest.TestCase):
    def test_exit_fill_other_leg_cancelled_btc_accounting(self):
        life, db, _, pm, broker, _ = _lifecycle()
        r = life.run_entry(
            symbol="ETHBTC",
            strategy=get_strategy("T1"),
            price_alt_btc=0.05,
            equity_btc=1.0,
            available_btc=1.0,
        )
        self.assertTrue(r.ok)
        oco_id = db.get_trade(r.trade_id)["binance_oco_list_id"]
        exit_px = r.fill.avg_price * 1.01
        fill = broker.trigger_exit(oco_id, exit_price=exit_px, leg="TRAIL")
        self.assertTrue(fill["other_cancelled"])
        other = broker.get_order(order_id=fill["other_cancelled"])
        self.assertEqual(other.status, "CANCELED")
        closed = life.close_from_exit_fill(
            trade_id=r.trade_id,
            reservation_id=r.reservation_id,
            exit_price=exit_px,
            exit_qty=r.fill.executed_qty,
            fees_btc=0.00001,
            other_leg_cancelled=True,
        )
        self.assertTrue(closed.ok)
        self.assertIn("ACCOUNTING_DONE", closed.events)
        self.assertIn("SLOT_RELEASED", closed.events)
        self.assertEqual(pm.slots_used(), 0)
        acc = closed.accounting
        self.assertIsInstance(acc, TradeAccounting)
        self.assertAlmostEqual(acc.exit_btc_value, r.fill.executed_qty * exit_px)
        self.assertIsNotNone(acc.realized_pnl_btc)
        tr = db.get_trade(r.trade_id)
        self.assertEqual(tr["status"], "CLOSED")
        self.assertIsNotNone(tr["realized_pnl_btc"])

    def test_trailing_active_state(self):
        life, db, _, _, broker, _ = _lifecycle()
        r = life.run_entry(
            symbol="ETHBTC",
            strategy=get_strategy("T1"),
            price_alt_btc=0.05,
            equity_btc=1.0,
            available_btc=1.0,
        )
        oco_id = db.get_trade(r.trade_id)["binance_oco_list_id"]
        broker.mark_trailing_active(oco_id)
        oco = broker.get_oco(oco_id)
        self.assertEqual(oco.status, "TRAILING_ACTIVE")


class TestIdempotency(unittest.TestCase):
    def test_duplicate_buy_after_timeout_reuses_client_order(self):
        life, _, _, _, broker, _ = _lifecycle()
        broker.buy_behavior = "FILL"
        strategy = get_strategy("T1")
        r1 = life.run_entry(
            symbol="ETHBTC",
            strategy=strategy,
            price_alt_btc=0.05,
            equity_btc=1.0,
            available_btc=1.0,
        )
        self.assertTrue(r1.ok)
        cid = life._client_buy_id(r1.trade_id)
        existing = life._lookup_existing_buy(cid, "ETHBTC")
        self.assertIsNotNone(existing)
        again = broker.place_market_buy(
            symbol="ETHBTC",
            quantity=r1.fill.executed_qty,
            client_order_id=cid,
            ref_price=0.05,
        )
        self.assertEqual(again.order_id, r1.fill.order_id)
        self.assertEqual(len([o for o in broker._orders.values() if o.side == "BUY"]), 1)


class TestWebSocketRecovery(unittest.TestCase):
    def test_disconnect_rest_reconnect_no_duplicate_notify(self):
        life, db, _, _, broker, n = _lifecycle()
        r = life.run_entry(
            symbol="ETHBTC",
            strategy=get_strategy("T1"),
            price_alt_btc=0.05,
            equity_btc=1.0,
            available_btc=1.0,
        )
        self.assertTrue(r.ok)
        seen_before = len(n.telegram.sent)
        stream = UserDataStream(rest_reconcile=lambda: life.reconcile_rest(["ETHBTC"]))
        stream.connect()
        stream.publish({"event_id": f"protected:{r.trade_id}", "type": "TRADE_PROTECTED"})
        stream.disconnect()
        stream.reconnect()
        stream.publish({"event_id": f"protected:{r.trade_id}", "type": "TRADE_PROTECTED"})
        prot = [x for x in n.telegram.sent if x["event"] == "TRADE_PROTECTED"]
        self.assertEqual(len(prot), 1)
        self.assertGreaterEqual(len(n.telegram.sent), seen_before)


class TestRestartRecovery(unittest.TestCase):
    def test_cases_a_b_c_d_e(self):
        life, db, safety, pm, broker, n = _lifecycle()
        recovery = RecoveryManager(
            life.exchange, db, safety, notifications=n, dry_broker=broker
        )

        # Case B: BUY filled + OCO submitted
        r = life.run_entry(
            symbol="ETHBTC",
            strategy=get_strategy("T1"),
            price_alt_btc=0.05,
            equity_btc=1.0,
            available_btc=1.0,
        )
        self.assertTrue(r.ok)
        rep_b = recovery.recover(["ETHBTC"], dry_run=True)
        self.assertIn("B", rep_b.cases)

        # Case C: trailing active
        oco_id = db.get_trade(r.trade_id)["binance_oco_list_id"]
        broker.mark_trailing_active(oco_id)
        rep_c = recovery.recover(["ETHBTC"], dry_run=True)
        self.assertIn("C", rep_c.cases)

        # Case D: exit while offline
        broker.trigger_exit(oco_id, exit_price=r.fill.avg_price * 1.01, leg="TRAIL")
        rep_d = recovery.recover(["ETHBTC"], dry_run=True)
        self.assertIn("D", rep_d.cases)

        # Case E: orphan exchange OCO
        broker.place_oco(
            symbol="SOLBTC",
            quantity=1.0,
            list_client_order_id="orphan_sol",
            above_stop_price=0.011,
            above_trailing_delta=25,
            below_stop_price=0.009,
            entry_ref_price=0.01,
        )
        rep_e = recovery.recover(["SOLBTC", "ETHBTC"], dry_run=True)
        self.assertIn("E", rep_e.cases)

        # Case A via restart_recover
        db.insert_trade(
            {
                "trade_id": "t_case_a",
                "symbol": "LINKBTC",
                "strategy": "T1",
                "selector": None,
                "entry_time": None,
                "entry_price": None,
                "quantity": None,
                "btc_value": None,
                "usdt_value": None,
                "configured_activation": 0.0075,
                "configured_trailing_distance": 0.0025,
                "strategy_config": {},
                "status": "ENTRY_PENDING",
            }
        )
        rr = life.restart_recover(case_hint="A")
        self.assertIn("A", rr["cases"])


class TestSlotsAndHalt(unittest.TestCase):
    def test_eight_slots_release_on_exit_and_failed_entry(self):
        life, _, _, pm, broker, _ = _lifecycle()
        strategy = get_strategy("T1")
        symbols = [
            "ETHBTC",
            "SOLBTC",
            "BNBBTC",
            "XRPBTC",
            "ADABTC",
            "DOTBTC",
            "LINKBTC",
            "LTCBTC",
        ]
        trades = []
        for i, sym in enumerate(symbols):
            r = life.run_entry(
                symbol=sym,
                strategy=strategy,
                price_alt_btc=0.01 + i * 0.001,
                equity_btc=1.0,
                available_btc=1.0,
                open_exposure_pct=i * 0.125,
            )
            self.assertTrue(r.ok, f"{sym}: {r.reason}")
            trades.append(r)
        self.assertEqual(pm.slots_used(), 8)
        ninth = life.run_entry(
            symbol="AVAXBTC",
            strategy=strategy,
            price_alt_btc=0.02,
            equity_btc=1.0,
            available_btc=1.0,
            open_exposure_pct=1.0,
        )
        self.assertFalse(ninth.ok)
        self.assertEqual(ninth.reason, "MAX_OPEN_TRADES")

        t0 = trades[0]
        oco_id = life.db.get_trade(t0.trade_id)["binance_oco_list_id"]
        broker.trigger_exit(oco_id, exit_price=t0.fill.avg_price * 1.01)
        life.close_from_exit_fill(
            trade_id=t0.trade_id,
            reservation_id=t0.reservation_id,
            exit_price=t0.fill.avg_price * 1.01,
            exit_qty=t0.fill.executed_qty,
        )
        self.assertEqual(pm.slots_used(), 7)
        again = life.run_entry(
            symbol="AVAXBTC",
            strategy=strategy,
            price_alt_btc=0.02,
            equity_btc=1.0,
            available_btc=1.0,
            open_exposure_pct=7 * 0.125,
        )
        self.assertTrue(again.ok, again.reason)
        self.assertEqual(pm.slots_used(), 8)

    def test_failed_buy_does_not_consume_slot(self):
        life, _, _, pm, broker, _ = _lifecycle()
        broker.buy_behavior = "REJECT"
        life.run_entry(
            symbol="ETHBTC",
            strategy=get_strategy("T1"),
            price_alt_btc=0.05,
            equity_btc=1.0,
            available_btc=1.0,
        )
        self.assertEqual(pm.slots_used(), 0)

    def test_halt_blocks_new_not_recovery(self):
        life, _, safety, _, _, _ = _lifecycle()
        safety.halt("TEST_HALT")
        self.assertFalse(safety.allow_new_entries())
        blocked = life.run_entry(
            symbol="ETHBTC",
            strategy=get_strategy("T1"),
            price_alt_btc=0.05,
            equity_btc=1.0,
            available_btc=1.0,
        )
        self.assertEqual(blocked.reason, "SAFETY_HALT")
        recon = life.reconcile_rest(["ETHBTC"])
        self.assertTrue(recon["ok"])


class TestLivePostureUnchanged(unittest.TestCase):
    def test_config_still_dry(self):
        from binance_btc_bot.config_loader import load_config

        cfg = load_config()
        self.assertFalse(cfg["live"]["enabled"])
        self.assertTrue(cfg["live"]["dry_run"])
        self.assertEqual(cfg["live"]["strategy"], "T4")
        self.assertIsNone(cfg["live"]["selector"])
        self.assertEqual(cfg["portfolio"]["max_simultaneous_trades"], 8)
        self.assertAlmostEqual(cfg["portfolio"]["allocation_per_trade"], 0.125)


if __name__ == "__main__":
    unittest.main()
