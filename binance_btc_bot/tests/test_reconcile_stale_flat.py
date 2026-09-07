"""Regression: stale PROTECTION_FAILED local vs flat Binance → RECONCILED_FLAT."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from binance_btc_bot.exchange.base import AccountSnapshot, Balance, SymbolInfo
from binance_btc_bot.execution.lifecycle import OrderLifecycle
from binance_btc_bot.portfolio.manager import PortfolioConfig, PortfolioManager
from binance_btc_bot.portfolio.trade_state import TradeStatus
from binance_btc_bot.risk.safety import SafetySystem
from binance_btc_bot.storage.database import BotDatabase


def _meta(symbol: str = "UNIBTC") -> SymbolInfo:
    return SymbolInfo(
        symbol=symbol,
        status="TRADING",
        base_asset="UNI",
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


class TestReconcileStaleProtectionFailed(unittest.TestCase):
    def _setup(self, *, free_uni: float = 0.0, locked_uni: float = 0.0):
        tmp = tempfile.mkdtemp()
        db = BotDatabase(Path(tmp) / "bot.sqlite3")
        safety = SafetySystem()
        portfolio = PortfolioManager(
            PortfolioConfig(max_simultaneous_trades=8, allocation_per_trade=0.125, max_total_allocation=1.0)
        )
        ex = MagicMock()
        ex.get_symbol_info.side_effect = lambda s: _meta(str(s).upper())
        ex.get_account.return_value = AccountSnapshot(
            balances={
                "UNI": Balance("UNI", free_uni, locked_uni),
                "BTC": Balance("BTC", 0.0078, 0.0),
                "BNB": Balance("BNB", 0.018, 0.0),
            }
        )
        ex.get_open_orders.return_value = []
        ex.get_open_order_lists.return_value = []
        life = OrderLifecycle(
            ex,
            db,
            safety,
            portfolio=portfolio,
            dry_broker=None,
            live_enabled=False,
            dry_run=True,
        )
        db.insert_trade(
            {
                "trade_id": "stale_uni_1",
                "symbol": "UNIBTC",
                "strategy": "T1",
                "entry_price": 9.11e-5,
                "quantity": 10.97,
                "btc_value": 0.001,
                "fees_btc": 0.0,
                "status": "PROTECTION_FAILED",
                "binance_entry_order_id": "938154825",
            }
        )
        # Occupy slot as rehydrate would.
        rh = portfolio.rehydrate_from_trades(db.open_trades())
        self.assertEqual(rh["restored"], 1)
        self.assertEqual(portfolio.slots_used(), 1)
        return life, db, safety, portfolio, ex

    def test_reconcile_flat_closes_without_fabricated_exit(self):
        life, db, safety, portfolio, ex = self._setup(free_uni=0.0)
        r1 = life.reconcile_stale_protection_failed(universe=["UNIBTC"])
        self.assertTrue(r1["ok"])
        self.assertEqual(r1["closed"], ["stale_uni_1"])
        tr = db.get_trade("stale_uni_1")
        self.assertEqual(tr["status"], "CLOSED")
        self.assertIsNone(tr.get("exit_price"))
        self.assertIsNone(tr.get("realized_pnl_btc"))
        self.assertIsNotNone(tr.get("exit_time"))
        # No Binance write adapters used.
        self.assertFalse(ex.place_entry.called)
        self.assertFalse(getattr(ex, "place_protective_sell").called)
        self.assertFalse(ex.place_trailing_exit.called)
        # Slot released
        self.assertEqual(portfolio.slots_used(), 0)
        self.assertEqual(db.open_trades(), [])
        # Historical incident preserved
        cur = db._conn.execute(
            "SELECT event, reason FROM bot_events WHERE trade_id=? ORDER BY id",
            ("stale_uni_1",),
        )
        events = [dict(r) for r in cur.fetchall()]
        names = [e["event"] for e in events]
        self.assertIn("RECONCILED_FLAT", names)
        self.assertIn("INCIDENT_PRESERVED", names)
        self.assertEqual(
            next(e for e in events if e["event"] == "RECONCILED_FLAT")["reason"],
            "EXCHANGE_FLAT_AFTER_PROTECTION_FAILED",
        )

    def test_idempotent_second_reconcile(self):
        life, db, safety, portfolio, ex = self._setup(free_uni=0.0)
        r1 = life.reconcile_stale_protection_failed()
        self.assertEqual(len(r1["closed"]), 1)
        cur = db._conn.execute(
            "SELECT COUNT(*) AS c FROM bot_events WHERE trade_id=? AND event='RECONCILED_FLAT'",
            ("stale_uni_1",),
        )
        count1 = int(cur.fetchone()["c"])
        r2 = life.reconcile_stale_protection_failed()
        self.assertEqual(r2["closed"], [])
        cur = db._conn.execute(
            "SELECT COUNT(*) AS c FROM bot_events WHERE trade_id=? AND event='RECONCILED_FLAT'",
            ("stale_uni_1",),
        )
        count2 = int(cur.fetchone()["c"])
        self.assertEqual(count1, 1)
        self.assertEqual(count2, 1)
        # close_reconciled_flat direct idempotency
        r3 = life.close_reconciled_flat(trade_id="stale_uni_1")
        self.assertTrue(r3["ok"])
        self.assertFalse(r3["changed"])
        self.assertEqual(r3["reason"], "ALREADY_RECONCILED_FLAT")

    def test_does_not_close_when_inventory_remains(self):
        life, db, safety, portfolio, ex = self._setup(free_uni=10.95)
        r = life.reconcile_stale_protection_failed()
        self.assertEqual(r["closed"], [])
        self.assertEqual(db.get_trade("stale_uni_1")["status"], "PROTECTION_FAILED")
        self.assertEqual(portfolio.slots_used(), 1)

    def test_via_reconcile_rest(self):
        life, db, safety, portfolio, ex = self._setup(free_uni=0.0)
        out = life.reconcile_rest(universe=["UNIBTC"])
        self.assertIn("stale_uni_1", out["closed"])
        self.assertEqual(db.get_trade("stale_uni_1")["status"], TradeStatus.CLOSED.value)


if __name__ == "__main__":
    unittest.main()
