"""Unit tests for Binance BTC bot — strategy, risk, native trail mapping, safety."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from binance_btc_bot.config_loader import FROZEN_STRATEGIES, load_config
from binance_btc_bot.exchange.base import OrderResult, SymbolInfo, TrailingOcoRequest
from binance_btc_bot.execution.recovery import RecoveryManager
from binance_btc_bot.execution.trailing import TrailingExecutor
from binance_btc_bot.risk.safety import SafetyState, SafetySystem
from binance_btc_bot.risk.sizing import size_position
from binance_btc_bot.storage.database import BotDatabase
from binance_btc_bot.strategy.entries import LiveEntryEngine
from binance_btc_bot.strategy.relative_price import compute_relative_price
from binance_btc_bot.strategy.selectors import SELECTOR_KINDS, live_selector_disabled
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
        order_types=("LIMIT", "MARKET", "STOP_LOSS", "TAKE_PROFIT", "STOP_LOSS_LIMIT", "TAKE_PROFIT_LIMIT"),
        oco_allowed=True,
        min_trailing_above_delta=10,
        max_trailing_above_delta=2000,
        min_trailing_below_delta=10,
        max_trailing_below_delta=2000,
    )


class TestConfigAndStrategies(unittest.TestCase):
    def test_load_config_frozen_strategies(self):
        cfg = load_config()
        self.assertFalse(cfg["live"]["enabled"])
        self.assertEqual(cfg["live"]["strategy"], "T4")
        self.assertIsNone(cfg["live"]["selector"])
        for k, (sl, act, dist) in FROZEN_STRATEGIES.items():
            s = get_strategy(k, cfg["strategies"])
            self.assertAlmostEqual(s.arm_sl_activation_trail, sl)
            self.assertAlmostEqual(s.activation, act)
            self.assertAlmostEqual(s.trail_distance, dist)

    def test_selectors_research_only(self):
        self.assertEqual(set(SELECTOR_KINDS), {"A", "B", "C", "D", "E", "F"})
        self.assertIsNone(live_selector_disabled())


class TestRelativePrice(unittest.TestCase):
    def test_usdt_legs_not_native(self):
        rel = compute_relative_price(
            btc_pair="ETHBTC",
            base_usdt=2500.0,
            btc_usdt=100_000.0,
            native_btc_price=0.0251,
        )
        self.assertAlmostEqual(rel.relative_price, 0.025)
        self.assertEqual(rel.usdt_pair, "ETHUSDT")


class TestT1BinanceMapping(unittest.TestCase):
    def test_t1_oco_mapping(self):
        t1 = get_strategy("T1")
        mapping = map_trail_to_binance_oco(
            strategy=t1,
            symbol="ETHBTC",
            entry_price=0.04,
            quantity=1.0,
            symbol_info=_meta(),
        )
        self.assertTrue(mapping.allowed)
        self.assertEqual(mapping.trailing_delta_bips, 25)
        self.assertEqual(mapping.above_type, "TAKE_PROFIT")
        self.assertEqual(mapping.below_type, "STOP_LOSS")
        self.assertAlmostEqual(mapping.activation_price, 0.04 * 1.0075, places=8)
        self.assertAlmostEqual(mapping.initial_stop_price, 0.04 * 0.9925, places=8)
        self.assertIsInstance(mapping.request, TrailingOcoRequest)
        self.assertEqual(mapping.request.above_trailing_delta, 25)

    def test_trail_below_min_not_silently_raised(self):
        t1 = get_strategy("T1")
        meta = _meta()
        # Force unsupported min
        meta = SymbolInfo(
            **{**meta.__dict__, "min_trailing_above_delta": 50, "max_trailing_above_delta": 2000}
        )
        mapping = map_trail_to_binance_oco(
            strategy=t1,
            symbol="ETHBTC",
            entry_price=0.04,
            quantity=1.0,
            symbol_info=meta,
        )
        self.assertFalse(mapping.allowed)
        self.assertTrue(any("NOT silently raised" in n for n in mapping.constraint_notes))


class TestRiskSizing(unittest.TestCase):
    def test_max_planned_loss_half_percent(self):
        t1 = get_strategy("T1")
        size = size_position(
            equity_btc=1.0,
            available_btc=1.0,
            price_alt_btc=0.04,
            strategy=t1,
            meta=_meta(),
            max_loss_per_trade=0.005,
            max_allocation_pct=1.0,  # isolate risk math
        )
        self.assertTrue(size.ok)
        self.assertLessEqual(size.planned_loss_btc, 0.005 + 1e-9)
        # risk notional = 0.005/0.0075 = 0.666... BTC
        self.assertAlmostEqual(size.risk_budget_btc, 0.005)
        self.assertLessEqual(size.notional_btc, 0.005 / 0.0075 + 1e-9)

    def test_allocation_cap(self):
        t1 = get_strategy("T1")
        size = size_position(
            equity_btc=1.0,
            available_btc=1.0,
            price_alt_btc=0.04,
            strategy=t1,
            meta=_meta(),
            max_loss_per_trade=0.005,
            max_allocation_pct=0.25,
        )
        self.assertTrue(size.ok)
        self.assertLessEqual(size.notional_btc, 0.25 + 1e-9)


class TestSafetyAndRecovery(unittest.TestCase):
    def test_halt_blocks_entries(self):
        s = SafetySystem()
        self.assertTrue(s.allow_new_entries())
        s.halt("TEST")
        self.assertEqual(s.state, SafetyState.HALT)
        self.assertFalse(s.allow_new_entries())

    def test_duplicate_oco_halts(self):
        with tempfile.TemporaryDirectory() as td:
            db = BotDatabase(Path(td) / "t.sqlite")
            safety = SafetySystem()
            ex = MagicMock()
            ex.get_open_orders.return_value = []
            ex.get_open_order_lists.return_value = [
                {"symbol": "ETHBTC", "orderListId": 1},
                {"symbol": "ETHBTC", "orderListId": 2},
            ]
            mgr = RecoveryManager(ex, db, safety)
            report = mgr.recover(["ETHBTC"], dry_run=False)
            self.assertFalse(report.ok)
            self.assertEqual(safety.state, SafetyState.HALT)
            db.close()

    def test_recovery_matches_local_oco(self):
        with tempfile.TemporaryDirectory() as td:
            db = BotDatabase(Path(td) / "t.sqlite")
            tid = db.new_trade_id()
            db.insert_trade(
                {
                    "trade_id": tid,
                    "symbol": "ETHBTC",
                    "strategy": "T1",
                    "selector": None,
                    "entry_price": 0.04,
                    "quantity": 1.0,
                    "btc_value": 0.04,
                    "configured_activation": 0.0075,
                    "configured_trailing_distance": 0.0025,
                    "strategy_config": {},
                    "binance_oco_list_id": "99",
                    "status": "OPEN",
                }
            )
            safety = SafetySystem()
            ex = MagicMock()
            ex.get_open_orders.return_value = []
            ex.get_open_order_lists.return_value = [{"symbol": "ETHBTC", "orderListId": 99}]
            report = RecoveryManager(ex, db, safety).recover(["ETHBTC"], dry_run=False)
            self.assertTrue(report.ok)
            self.assertIn(tid, report.recovered_trade_ids)
            db.close()


class TestEntryGate(unittest.TestCase):
    def test_provider_driven_strategy_label(self):
        from binance_btc_bot.strategy.provider import FixedStrategyProvider

        provider = FixedStrategyProvider("T1")
        eng = LiveEntryEngine(
            strategy_key=provider.strategy_key(),
            selector_key=provider.selector_key(),
        )
        d2 = eng.evaluate(symbol="ETHBTC", score=0.7, open_symbols=[], open_count=0)
        # First evaluate after default was_below may need below then cross
        eng2 = LiveEntryEngine(strategy_key=provider.strategy_key())
        eng2.evaluate(symbol="ETHBTC", score=0.5, open_symbols=[], open_count=0)
        d2 = eng2.evaluate(symbol="ETHBTC", score=0.7, open_symbols=[], open_count=0)
        self.assertTrue(d2.trade_suggested)
        self.assertEqual(d2.strategy, "T1")
        self.assertIsNone(d2.selector)
        d3 = eng2.evaluate(symbol="ETHBTC", score=0.7, open_symbols=[], open_count=0)
        self.assertFalse(d3.trade_suggested)  # continuation

    def test_cross_into(self):
        eng = LiveEntryEngine(strategy_key="T1", long_threshold=0.6)
        d1 = eng.evaluate(symbol="ETHBTC", score=0.5, open_symbols=[], open_count=0)
        self.assertFalse(d1.trade_suggested)
        d2 = eng.evaluate(symbol="ETHBTC", score=0.7, open_symbols=[], open_count=0)
        self.assertTrue(d2.trade_suggested)
        self.assertEqual(d2.strategy, "T1")
        self.assertIsNone(d2.selector)
        d3 = eng.evaluate(symbol="ETHBTC", score=0.7, open_symbols=[], open_count=0)
        self.assertFalse(d3.trade_suggested)  # continuation


class TestTrailingExecutorDry(unittest.TestCase):
    def test_dry_submit_logs_native_mode(self):
        with tempfile.TemporaryDirectory() as td:
            db = BotDatabase(Path(td) / "t.sqlite")
            ex = MagicMock()
            ex.get_symbol_info.return_value = _meta()
            ex.place_trailing_exit.return_value = OrderResult(
                ok=True,
                status="DRY_RUN",
                dry_run=True,
                order_type="OCO_TRAILING",
                raw={"params": {"aboveTrailingDelta": 25}, "mode": "BINANCE_NATIVE"},
            )
            execu = TrailingExecutor(ex, db)
            res = execu.submit_native_trailing(
                trade_id="abc",
                symbol="ETHBTC",
                strategy=get_strategy("T1"),
                entry_price=0.04,
                quantity=1.0,
            )
            self.assertTrue(res.ok)
            self.assertEqual(res.trail_bips, 25)
            ex.place_trailing_exit.assert_called_once()
            db.close()


if __name__ == "__main__":
    unittest.main()
