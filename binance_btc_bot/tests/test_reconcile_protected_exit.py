"""Regression: PROTECTED + flat Binance + ALL_DONE OCO → CLOSED (no false missing-protection warn)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from binance_btc_bot.exchange.base import AccountSnapshot, Balance, OrderResult, SymbolInfo
from binance_btc_bot.execution.lifecycle import OrderLifecycle
from binance_btc_bot.notifications.base import NotificationResult
from binance_btc_bot.notifications.manager import NotificationManager
from binance_btc_bot.portfolio.manager import PortfolioConfig, PortfolioManager
from binance_btc_bot.risk.safety import SafetySystem
from binance_btc_bot.storage.database import BotDatabase
from binance_btc_bot.strategy.trails import get_strategy


def _meta(symbol: str) -> SymbolInfo:
    base = symbol.replace("BTC", "")
    return SymbolInfo(
        symbol=symbol,
        status="TRADING",
        base_asset=base,
        quote_asset="BTC",
        quantity_step=0.01,
        min_quantity=0.01,
        max_quantity=1e6,
        price_tick=1e-7,
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
            {
                "event": event.event,
                "severity": event.severity.value,
                "message": event.message,
                "details": getattr(event, "details", None),
            }
        )
        return NotificationResult(ok=True, channel=self.name)

    def status(self) -> dict:
        return {"configured": True, "enabled": True, "name": self.name}


def _notif() -> NotificationManager:
    return NotificationManager(telegram=RecordingChannel("telegram"), sms=RecordingChannel("sms"))


class TestReconcileProtectedFlatExit(unittest.TestCase):
    def _setup(
        self,
        *,
        symbol: str = "LINKBTC",
        trade_id: str = "6ab324eebd1140dca6ec05e27f8d0a62",
        free: float = 0.0,
        locked: float = 0.0,
        oco_id: str = "1001",
        entry_order_id: str = "9001",
        open_lists: list | None = None,
        open_orders: list | None = None,
    ):
        tmp = tempfile.mkdtemp()
        db = BotDatabase(Path(tmp) / "bot.sqlite3")
        safety = SafetySystem()
        portfolio = PortfolioManager(
            PortfolioConfig(max_simultaneous_trades=8, allocation_per_trade=0.125, max_total_allocation=1.0)
        )
        base = symbol.replace("BTC", "")
        ex = MagicMock()
        ex.get_symbol_info.side_effect = lambda s: _meta(str(s).upper())
        ex.get_account.return_value = AccountSnapshot(
            balances={
                base: Balance(base, free, locked),
                "BTC": Balance("BTC", 0.01, 0.0),
                "BNB": Balance("BNB", 0.02, 0.0),
            }
        )
        ex.get_open_orders.return_value = list(open_orders or [])
        ex.get_open_order_lists.return_value = list(open_lists or [])
        ex.get_price.return_value = 0.00015
        n = _notif()
        life = OrderLifecycle(
            ex,
            db,
            safety,
            portfolio=portfolio,
            notifications=n,
            dry_broker=None,
            live_enabled=True,
            dry_run=False,
        )
        db.insert_trade(
            {
                "trade_id": trade_id,
                "symbol": symbol,
                "strategy": "T1",
                "entry_time": "2026-09-01T12:00:00Z",
                "entry_price": 0.00015,
                "quantity": 10.0,
                "btc_value": 0.0015,
                "fees_btc": 0.000001,
                "configured_activation": 0.01,
                "configured_trailing_distance": 0.005,
                "status": "PROTECTED",
                "binance_entry_order_id": entry_order_id,
                "binance_oco_list_id": oco_id,
                "strategy_config": {"reservation_id": None},
            }
        )
        portfolio.rehydrate_from_trades(db.open_trades())
        return life, db, safety, portfolio, ex, n

    def _all_done_list(
        self,
        *,
        symbol: str,
        oco_id: str,
        child_sell_id: str = "9100",
        order_type: str = "TAKE_PROFIT",
        trailing_delta: int | None = 25,
    ) -> dict:
        report = {
            "orderId": int(child_sell_id),
            "symbol": symbol,
            "side": "SELL",
            "type": order_type,
            "status": "FILLED",
        }
        if trailing_delta is not None:
            report["trailingDelta"] = trailing_delta
        return {
            "orderListId": int(oco_id) if str(oco_id).isdigit() else oco_id,
            "symbol": symbol,
            "contingencyType": "OCO",
            "listStatusType": "ALL_DONE",
            "listOrderStatus": "ALL_DONE",
            "orders": [
                {"symbol": symbol, "orderId": int(child_sell_id), "clientOrderId": "tp1"},
                {"symbol": symbol, "orderId": 9101, "clientOrderId": "sl1"},
            ],
            "orderReports": [report],
        }

    def test_a_protected_flat_all_done_mytrades_closes_no_missing_warn(self):
        life, db, safety, portfolio, ex, n = self._setup(free=0.0, locked=0.0)
        ex.get_order_list.return_value = self._all_done_list(symbol="LINKBTC", oco_id="1001")
        ex.get_my_trades.return_value = [
            {
                "id": 1,
                "orderId": 9001,
                "price": "0.00015",
                "qty": "10.0",
                "isBuyer": True,
                "commission": "0.000001",
                "commissionAsset": "BTC",
                "time": 1_700_000_000_000,
            },
            {
                "id": 2,
                "orderId": 9100,
                "price": "0.000165",
                "qty": "10.0",
                "isBuyer": False,
                "commission": "0.0000015",
                "commissionAsset": "BTC",
                "time": 1_700_000_100_000,
            },
        ]
        out = life.reconcile_rest(universe=["LINKBTC"])
        self.assertIn("6ab324eebd1140dca6ec05e27f8d0a62", out["closed"])
        self.assertEqual(db.get_trade("6ab324eebd1140dca6ec05e27f8d0a62")["status"], "CLOSED")
        self.assertNotIn("RECONCILE_PROTECTION_MISSING", safety.reasons)
        self.assertFalse(out.get("fail_closed"))
        closed_msgs = [x for x in n.telegram.sent if x["event"] == "TRADE_CLOSED"]
        self.assertEqual(len(closed_msgs), 1)

    def test_b_protected_flat_filled_exit_evidence_closes(self):
        life, db, safety, portfolio, ex, n = self._setup(
            symbol="RENDERBTC",
            trade_id="60c4977cabe641a3abb9598831a06de4",
            oco_id="2002",
            entry_order_id="8002",
        )
        ex.get_order_list.return_value = self._all_done_list(
            symbol="RENDERBTC", oco_id="2002", child_sell_id="8200", order_type="STOP_LOSS", trailing_delta=None
        )
        ex.get_my_trades.return_value = [
            {
                "id": 9,
                "orderId": 8200,
                "price": "0.00012",
                "qty": "10.0",
                "isBuyer": False,
                "commission": "0.0",
                "commissionAsset": "BNB",
                "time": 1_700_000_200_000,
            },
        ]
        r = life.reconcile_protected_flat_exits(universe=["RENDERBTC"])
        self.assertEqual(r["closed"], ["60c4977cabe641a3abb9598831a06de4"])
        tr = db.get_trade("60c4977cabe641a3abb9598831a06de4")
        self.assertEqual(tr["status"], "CLOSED")
        self.assertAlmostEqual(float(tr["exit_price"]), 0.00012)
        self.assertEqual(portfolio.slots_used(), 0)

    def test_c_protected_inventory_open_oco_stays_protected(self):
        open_list = {
            "orderListId": 1001,
            "symbol": "LINKBTC",
            "contingencyType": "OCO",
            "listStatusType": "EXEC_STARTED",
            "listOrderStatus": "EXECUTING",
            "orders": [{"symbol": "LINKBTC", "orderId": 9100, "clientOrderId": "tp1"}],
        }
        life, db, safety, portfolio, ex, n = self._setup(
            free=0.0, locked=10.0, open_lists=[open_list], open_orders=[]
        )
        out = life.reconcile_rest(universe=["LINKBTC"])
        self.assertEqual(out.get("closed") or [], [])
        self.assertEqual(db.get_trade("6ab324eebd1140dca6ec05e27f8d0a62")["status"], "PROTECTED")
        self.assertNotIn("RECONCILE_PROTECTION_MISSING", safety.reasons)
        notes = " ".join(out["notes"])
        self.assertIn("PROTECTED_OPEN_LIST", notes)

    def test_d_protected_inventory_no_protection_warns(self):
        life, db, safety, portfolio, ex, n = self._setup(free=10.0, locked=0.0)
        ex.get_order_list.side_effect = Exception("list gone")
        out = life.reconcile_rest(universe=["LINKBTC"])
        self.assertIn("RECONCILE_PROTECTION_MISSING", safety.reasons)
        self.assertTrue(out.get("fail_closed"))
        self.assertEqual(db.get_trade("6ab324eebd1140dca6ec05e27f8d0a62")["status"], "PROTECTED")

    def test_d2_manual_web_take_profit_not_missing_warn(self):
        """Operator replaced bot OCO with a standalone TAKE_PROFIT_LIMIT — do not spam warn."""
        tp = {
            "symbol": "LINKBTC",
            "orderId": 333,
            "orderListId": -1,
            "type": "TAKE_PROFIT_LIMIT",
            "side": "SELL",
            "status": "NEW",
            "clientOrderId": "web_manual_tp",
        }
        life, db, safety, portfolio, ex, n = self._setup(
            free=0.0, locked=10.0, open_lists=[], open_orders=[tp]
        )
        out = life.reconcile_rest(universe=["LINKBTC"])
        notes = " ".join(out["notes"])
        self.assertIn("PROTECTED_EXTERNAL", notes)
        self.assertNotIn("RECONCILE_PROTECTION_MISSING", safety.reasons)
        self.assertFalse(out.get("fail_closed"))

    def test_d3_manual_protection_flag_skips_warn(self):
        life, db, safety, portfolio, ex, n = self._setup(free=10.0, locked=0.0, open_lists=[], open_orders=[])
        db.update_trade(
            "6ab324eebd1140dca6ec05e27f8d0a62",
            strategy_config={"manual_protection": True, "reservation_id": None},
        )
        out = life.reconcile_rest(universe=["LINKBTC"])
        notes = " ".join(out["notes"])
        self.assertIn("PROTECTED_MANUAL", notes)
        self.assertNotIn("RECONCILE_PROTECTION_MISSING", safety.reasons)
        self.assertFalse(out.get("fail_closed"))

    def test_e_second_reconcile_no_duplicate_trade_closed(self):
        life, db, safety, portfolio, ex, n = self._setup()
        ex.get_order_list.return_value = self._all_done_list(symbol="LINKBTC", oco_id="1001")
        ex.get_my_trades.return_value = [
            {
                "id": 2,
                "orderId": 9100,
                "price": "0.00016",
                "qty": "10.0",
                "isBuyer": False,
                "commission": "0.000001",
                "commissionAsset": "BTC",
                "time": 1_700_000_100_000,
            },
        ]
        life.reconcile_rest(universe=["LINKBTC"])
        closed1 = [x for x in n.telegram.sent if x["event"] == "TRADE_CLOSED"]
        self.assertEqual(len(closed1), 1)
        life.reconcile_rest(universe=["LINKBTC"])
        closed2 = [x for x in n.telegram.sent if x["event"] == "TRADE_CLOSED"]
        self.assertEqual(len(closed2), 1)
        r = life.close_from_exit_fill(
            trade_id="6ab324eebd1140dca6ec05e27f8d0a62",
            reservation_id=None,
            exit_price=0.00016,
        )
        self.assertEqual(r.reason, "ALREADY_CLOSED")
        self.assertEqual(len([x for x in n.telegram.sent if x["event"] == "TRADE_CLOSED"]), 1)

    def test_f_render_style_locked_open_oco_protected_open_list(self):
        open_list = {
            "orderListId": 2002,
            "symbol": "RENDERBTC",
            "contingencyType": "OCO",
            "listStatusType": "EXEC_STARTED",
            "listOrderStatus": "EXECUTING",
            "orders": [{"symbol": "RENDERBTC", "orderId": 8200, "clientOrderId": "tp1"}],
        }
        life, db, safety, portfolio, ex, n = self._setup(
            symbol="RENDERBTC",
            trade_id="60c4977cabe641a3abb9598831a06de4",
            free=0.0,
            locked=12.5,
            oco_id="2002",
            open_lists=[open_list],
        )
        out = life.reconcile_rest(universe=["RENDERBTC"])
        notes = " ".join(out["notes"])
        self.assertIn("PROTECTED_OPEN_LIST", notes)
        self.assertEqual(db.get_trade("60c4977cabe641a3abb9598831a06de4")["status"], "PROTECTED")
        self.assertNotIn("RECONCILE_PROTECTION_MISSING", safety.reasons)

    def test_g_dust_after_exit_closes_no_missing_warn(self):
        """ICPBTC-style: OCO ALL_DONE, leftover free below minNotional → close, no spam warn."""
        life, db, safety, portfolio, ex, n = self._setup(
            symbol="ICPBTC",
            trade_id="baee96468f7a4d3491f2424fc37ba05c",
            free=0.01,  # minQty step but notional dust at low ICPBTC price
            locked=0.0,
            oco_id="24561169565",
            entry_order_id="333258649",
        )
        # Override quantity on trade
        db.update_trade("baee96468f7a4d3491f2424fc37ba05c", quantity=24.59, entry_price=0.0000392)
        ex.get_price.return_value = 0.0000392
        ex.get_order_list.return_value = self._all_done_list(
            symbol="ICPBTC", oco_id="24561169565", child_sell_id="333258656", order_type="STOP_LOSS", trailing_delta=None
        )
        ex.get_my_trades.return_value = [
            {
                "id": 1,
                "orderId": 333258649,
                "price": "0.00003920",
                "qty": "24.60",
                "isBuyer": True,
                "time": 1,
            },
            {
                "id": 2,
                "orderId": 333258656,
                "price": "0.00003920",
                "qty": "24.59",
                "isBuyer": False,
                "commission": "0.000001",
                "commissionAsset": "BNB",
                "time": 2,
            },
        ]
        safety.on_notify = lambda *a, **k: None
        notifies: list[str] = []

        def _cap(event, message, **kwargs):
            notifies.append(str(kwargs.get("reason") or event))

        safety.on_notify = _cap
        out = life.reconcile_rest(universe=["ICPBTC"])
        self.assertEqual(db.get_trade("baee96468f7a4d3491f2424fc37ba05c")["status"], "CLOSED")
        self.assertNotIn("RECONCILE_PROTECTION_MISSING", notifies)
        notes = " ".join(out["notes"])
        self.assertTrue(
            "POSITION_CLOSED" in notes or "FLAT_EXIT_CLOSED" in notes or "ECONOMIC_FLAT" in notes
        )

    def test_h_missing_protection_warn_deduped(self):
        life, db, safety, portfolio, ex, n = self._setup(free=10.0, locked=0.0)
        safety.warn_notify_cooldown_sec = 900.0
        sent: list[str] = []

        def _cap(event, message, **kwargs):
            sent.append(str(kwargs.get("reason") or event))

        safety.on_notify = _cap
        ex.get_order_list.side_effect = Exception("list gone")
        life.reconcile_rest(universe=["LINKBTC"])
        life.reconcile_rest(universe=["LINKBTC"])
        life.reconcile_rest(universe=["LINKBTC"])
        self.assertEqual(sent.count("RECONCILE_PROTECTION_MISSING"), 1)
        self.assertGreaterEqual(
            sum(1 for e in safety.events if e.get("event") == "WARNING_NOTIFY_SUPPRESSED"), 1
        )
        # Genuine missing still recorded in reasons each cycle
        self.assertGreaterEqual(safety.reasons.count("RECONCILE_PROTECTION_MISSING"), 2)

    def test_handle_user_data_sell_filled_closes_once(self):
        life, db, safety, portfolio, ex, n = self._setup()
        ev = {
            "e": "executionReport",
            "s": "LINKBTC",
            "S": "SELL",
            "o": "TAKE_PROFIT",
            "X": "FILLED",
            "x": "TRADE",
            "L": "0.00017",
            "l": "10.0",
            "z": "10.0",
            "i": 9100,
            "g": 1001,
            "n": "0.000001",
            "N": "BTC",
            "T": 1_700_000_300_000,
            "d": 25,
        }
        r1 = life.handle_user_data_event(ev)
        self.assertTrue(r1["handled"])
        self.assertEqual(db.get_trade("6ab324eebd1140dca6ec05e27f8d0a62")["status"], "CLOSED")
        self.assertEqual(len([x for x in n.telegram.sent if x["event"] == "TRADE_CLOSED"]), 1)
        r2 = life.handle_user_data_event(ev)
        self.assertEqual(r2.get("reason"), "ALREADY_CLOSED")
        self.assertEqual(len([x for x in n.telegram.sent if x["event"] == "TRADE_CLOSED"]), 1)

    def test_run_entry_open_payload_includes_signal(self):
        from binance_btc_bot.execution.dry_broker import DryRunBroker
        from binance_btc_bot.execution.trailing import TrailingExecutor

        tmp = tempfile.mkdtemp()
        db = BotDatabase(Path(tmp) / "bot.sqlite3")
        safety = SafetySystem()
        portfolio = PortfolioManager(
            PortfolioConfig(max_simultaneous_trades=8, allocation_per_trade=0.125, max_total_allocation=1.0)
        )
        n = _notif()
        ex = MagicMock()
        ex.get_symbol_info.side_effect = lambda s: _meta(str(s).upper())
        ex.place_entry.return_value = OrderResult(ok=True, status="DRY_RUN", dry_run=True)
        ex.place_trailing_exit.return_value = OrderResult(
            ok=True, status="DRY_RUN", dry_run=True, order_id=None, client_order_id="t_test"
        )
        ex.get_order.return_value = OrderResult(ok=False, status="UNKNOWN")
        ex.get_open_orders.return_value = []
        ex.get_open_order_lists.return_value = []
        broker = DryRunBroker()
        life = OrderLifecycle(
            ex,
            db,
            safety,
            portfolio=portfolio,
            notifications=n,
            dry_broker=broker,
            trailing=TrailingExecutor(ex, db, notifications=n),
            live_enabled=False,
            dry_run=True,
        )
        signal = {
            "timestamp": "2026-09-06T12:00:00Z",
            "previous_s": 0.4,
            "current_s": 0.6,
            "threshold": 0.5,
            "genuine_new_cross": True,
        }
        r = life.run_entry(
            symbol="ETHBTC",
            strategy=get_strategy("T1"),
            price_alt_btc=0.05,
            equity_btc=1.0,
            available_btc=1.0,
            signal=signal,
            selector="NONE",
            portfolio_before={"equity_btc": 1.0, "btc_free": 1.0, "open_trades": 0},
        )
        self.assertTrue(r.ok)
        opened = [x for x in n.telegram.sent if x["event"] == "TRADE_OPENED"]
        self.assertEqual(len(opened), 1)
        details = opened[0].get("details") or {}
        payload = details.get("payload") if isinstance(details, dict) else None
        self.assertIsInstance(payload, dict)
        self.assertEqual(payload.get("selector"), "NONE")
        self.assertEqual(payload.get("signal", {}).get("current_s"), 0.6)
        self.assertEqual(payload.get("signal", {}).get("genuine_new_cross"), True)
        self.assertEqual(payload.get("portfolio_before", {}).get("equity_btc"), 1.0)


if __name__ == "__main__":
    unittest.main()
