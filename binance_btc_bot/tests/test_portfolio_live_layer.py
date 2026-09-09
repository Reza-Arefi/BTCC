"""Portfolio manager, allocation, cross-into, and live-entry layer tests."""

from __future__ import annotations

import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

from binance_btc_bot.config_loader import load_config
from binance_btc_bot.exchange.base import SymbolInfo
from binance_btc_bot.portfolio.manager import PortfolioConfig, PortfolioManager, validate_portfolio_config
from binance_btc_bot.portfolio.trade_state import TradeStatus
from binance_btc_bot.risk.sizing import size_position
from binance_btc_bot.strategy.entries import (
    CLS_ELIGIBLE,
    CLS_MAX_OPEN,
    CLS_NEED_PRIOR_BELOW,
    CLS_SAME_PAIR_OPEN,
    CLS_SIGNAL_CONTINUATION,
    LiveEntryEngine,
)
from binance_btc_bot.strategy.trails import get_strategy


def _meta(
    symbol: str = "ETHBTC",
    *,
    step: float = 0.0001,
    min_qty: float = 0.0001,
    min_notional: float = 0.0001,
) -> SymbolInfo:
    return SymbolInfo(
        symbol=symbol,
        status="TRADING",
        base_asset=symbol.replace("BTC", ""),
        quote_asset="BTC",
        quantity_step=step,
        min_quantity=min_qty,
        max_quantity=1000.0,
        price_tick=0.000001,
        min_notional=min_notional,
        order_types=("LIMIT", "MARKET", "STOP_LOSS", "TAKE_PROFIT"),
        oco_allowed=True,
        min_trailing_above_delta=10,
        max_trailing_above_delta=2000,
        min_trailing_below_delta=10,
        max_trailing_below_delta=2000,
    )


def _pm(max_n: int, alloc: float | None = None) -> PortfolioManager:
    if alloc is None:
        alloc = 1.0 / max_n
    return PortfolioManager(
        PortfolioConfig(
            max_simultaneous_trades=max_n,
            allocation_per_trade=alloc,
            max_total_allocation=1.0,
        )
    )


class TestPortfolioConfig(unittest.TestCase):
    def test_production_config(self):
        cfg = load_config()
        self.assertFalse(cfg["live"]["enabled"])
        self.assertTrue(cfg["live"]["dry_run"])
        self.assertEqual(cfg["live"]["strategy"], "T4")
        self.assertIsNone(cfg["live"]["selector"])
        self.assertEqual(cfg["portfolio"]["max_simultaneous_trades"], 8)
        self.assertAlmostEqual(cfg["portfolio"]["allocation_per_trade"], 0.125)
        self.assertAlmostEqual(cfg["risk"]["max_loss_per_trade"], 0.005)
        self.assertAlmostEqual(cfg["risk"]["max_allocation_pct"], 0.125)
        self.assertAlmostEqual(cfg["entry"]["long_threshold"], 0.65)

    def test_invalid_exceeds_100pct(self):
        with self.assertRaises(ValueError):
            validate_portfolio_config(
                {"max_simultaneous_trades": 8, "allocation_per_trade": 0.20, "max_total_allocation": 1.0}
            )

    def test_configs_3_10_20_30(self):
        for n in (3, 10, 20, 30):
            cfg = validate_portfolio_config(
                {"max_simultaneous_trades": n, "allocation_per_trade": 1.0 / n, "max_total_allocation": 1.0}
            )
            self.assertEqual(cfg.max_simultaneous_trades, n)
            self.assertAlmostEqual(cfg.total_planned_allocation, 1.0)

    def test_out_of_range_max(self):
        with self.assertRaises(ValueError):
            validate_portfolio_config({"max_simultaneous_trades": 0, "allocation_per_trade": 0.5})
        with self.assertRaises(ValueError):
            validate_portfolio_config({"max_simultaneous_trades": 31, "allocation_per_trade": 1 / 31})


class TestPortfolioSlots(unittest.TestCase):
    def test_eight_trades_exactly_ninth_rejected(self):
        pm = _pm(8)
        ids = []
        for i in range(8):
            r = pm.try_reserve(f"S{i}BTC")
            self.assertTrue(r.ok, r.reason)
            ids.append(r.reservation.reservation_id)
            pm.mark_protected(r.reservation.reservation_id, f"t{i}")
        self.assertEqual(pm.slots_used(), 8)
        self.assertAlmostEqual(pm.allocated_pct(), 1.0, places=6)
        ninth = pm.try_reserve("ETHBTC")
        self.assertFalse(ninth.ok)
        self.assertEqual(ninth.reason, "MAX_OPEN_TRADES")

    def test_duplicate_symbol_rejected(self):
        pm = _pm(8)
        r1 = pm.try_reserve("ETHBTC")
        self.assertTrue(r1.ok)
        r2 = pm.try_reserve("ETHBTC")
        self.assertFalse(r2.ok)
        self.assertEqual(r2.reason, "SAME_PAIR_ALREADY_OPEN")

    def test_seven_active_two_signals_one_accepted(self):
        pm = _pm(8)
        for i in range(7):
            r = pm.try_reserve(f"S{i}BTC")
            self.assertTrue(r.ok)
            pm.mark_protected(r.reservation.reservation_id, f"t{i}")
        a = pm.try_reserve("AAVEBTC")
        b = pm.try_reserve("ADABTC")
        self.assertTrue(a.ok)
        self.assertFalse(b.ok)
        self.assertEqual(b.reason, "MAX_OPEN_TRADES")
        self.assertEqual(pm.slots_used(), 8)

    def test_race_final_slot(self):
        pm = _pm(8)
        for i in range(7):
            r = pm.try_reserve(f"S{i}BTC")
            pm.mark_protected(r.reservation.reservation_id, f"t{i}")
        results = []
        barrier = threading.Barrier(2)

        def _claim(sym: str) -> None:
            barrier.wait()
            results.append(pm.try_reserve(sym))

        with ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(_claim, "ETHBTC")
            f2 = pool.submit(_claim, "SOLBTC")
            f1.result()
            f2.result()
        oks = [r for r in results if r.ok]
        fails = [r for r in results if not r.ok]
        self.assertEqual(len(oks), 1)
        self.assertEqual(len(fails), 1)
        self.assertEqual(fails[0].reason, "MAX_OPEN_TRADES")
        self.assertEqual(pm.slots_used(), 8)

    def test_allocation_12p5(self):
        pm = _pm(8, 0.125)
        self.assertAlmostEqual(pm.allocation_per_trade, 0.125)
        r = pm.try_reserve("ETHBTC")
        self.assertAlmostEqual(r.reservation.allocation_pct, 0.125)
        self.assertAlmostEqual(pm.allocated_pct(), 0.125)


class TestSizingAllocationAndRisk(unittest.TestCase):
    def test_requested_vs_actual_allocation(self):
        t1 = get_strategy("T1")
        size = size_position(
            equity_btc=1.0,
            available_btc=1.0,
            price_alt_btc=0.04,
            strategy=t1,
            meta=_meta(),
            max_loss_per_trade=0.005,
            max_allocation_pct=0.125,
            fee_buffer_pct=0.0,
        )
        self.assertTrue(size.ok)
        self.assertAlmostEqual(size.requested_allocation_pct, 0.125)
        self.assertLessEqual(size.actual_allocation_pct, 0.125 + 1e-12)
        self.assertLessEqual(size.notional_btc, 0.125 + 1e-12)
        self.assertLessEqual(size.planned_loss_btc, 0.005 + 1e-12)

    def test_rounding_cannot_exceed_allocation(self):
        t1 = get_strategy("T1")
        # Coarse step that would overshoot without floor-down loop
        size = size_position(
            equity_btc=1.0,
            available_btc=1.0,
            price_alt_btc=0.04,
            strategy=t1,
            meta=_meta(step=0.5, min_qty=0.5, min_notional=0.01),
            max_loss_per_trade=0.005,
            max_allocation_pct=0.125,
            fee_buffer_pct=0.0,
        )
        # 0.125 BTC / 0.04 = 3.125 qty → floor to 3.0 step 0.5? step 0.5 → 3.0 lots = 0.12 notional OK
        # Or may fail min constraints — either ok with actual<=requested or rejected
        if size.ok:
            self.assertLessEqual(size.actual_allocation_pct, 0.125 + 1e-12)
        else:
            self.assertIn(
                size.reason,
                {"ALLOCATION_ROUNDING_EXCEEDED", "BELOW_MIN_QUANTITY", "BELOW_MIN_NOTIONAL", "ZERO_NOTIONAL"},
            )

    def test_insufficient_balance(self):
        t1 = get_strategy("T1")
        size = size_position(
            equity_btc=1.0,
            available_btc=0.0,
            price_alt_btc=0.04,
            strategy=t1,
            meta=_meta(),
            max_allocation_pct=0.125,
        )
        self.assertFalse(size.ok)
        self.assertEqual(size.reason, "INSUFFICIENT_AVAILABLE_BTC")

    def test_min_notional_failure(self):
        t1 = get_strategy("T1")
        size = size_position(
            equity_btc=0.001,
            available_btc=0.001,
            price_alt_btc=0.04,
            strategy=t1,
            meta=_meta(min_notional=1.0),
            max_loss_per_trade=0.005,
            max_allocation_pct=0.125,
            fee_buffer_pct=0.0,
        )
        self.assertFalse(size.ok)
        self.assertIn(size.reason, {"BELOW_MIN_NOTIONAL", "BELOW_MIN_QUANTITY", "ZERO_NOTIONAL"})

    def test_quantity_filter_failure(self):
        t1 = get_strategy("T1")
        size = size_position(
            equity_btc=1.0,
            available_btc=1.0,
            price_alt_btc=0.04,
            strategy=t1,
            meta=_meta(step=10.0, min_qty=10.0),
            max_loss_per_trade=0.005,
            max_allocation_pct=0.125,
            fee_buffer_pct=0.0,
        )
        self.assertFalse(size.ok)
        self.assertIn(size.reason, {"BELOW_MIN_QUANTITY", "ALLOCATION_ROUNDING_EXCEEDED", "BELOW_MIN_NOTIONAL"})

    def test_risk_over_half_percent_rejected(self):
        t1 = get_strategy("T1")
        # Tiny SL distance would make planned loss huge for a large notional —
        # force via absurd allocation with very small SL by mutating strategy copy.
        strat = deepcopy(t1)
        object.__setattr__(strat, "arm_sl_activation_trail", 0.0075)
        size = size_position(
            equity_btc=1.0,
            available_btc=1.0,
            price_alt_btc=0.04,
            strategy=strat,
            meta=_meta(),
            max_loss_per_trade=0.005,
            max_allocation_pct=1.0,
            fee_buffer_pct=0.0,
        )
        # With 100% alloc, risk cap binds: notional <= 0.005/0.0075
        self.assertTrue(size.ok)
        self.assertLessEqual(size.planned_loss_btc, 0.005 + 1e-9)


class TestCrossInto(unittest.TestCase):
    def test_already_above_does_not_repeat(self):
        eng = LiveEntryEngine(strategy_key="T1", long_threshold=0.65)
        d0 = eng.evaluate(symbol="ETHBTC", score=0.70, open_symbols=[], open_count=0)
        self.assertFalse(d0.trade_suggested)
        self.assertEqual(d0.classification, CLS_NEED_PRIOR_BELOW)
        d1 = eng.evaluate(symbol="ETHBTC", score=0.70, open_symbols=[], open_count=0)
        self.assertFalse(d1.trade_suggested)
        self.assertIn(d1.classification, {CLS_NEED_PRIOR_BELOW, CLS_SIGNAL_CONTINUATION})

    def test_genuine_cross_creates_candidate(self):
        eng = LiveEntryEngine(strategy_key="T1", long_threshold=0.65)
        eng.evaluate(symbol="ETHBTC", score=0.50, open_symbols=[], open_count=0)
        d = eng.evaluate(symbol="ETHBTC", score=0.65, open_symbols=[], open_count=0)
        self.assertTrue(d.trade_suggested)
        self.assertEqual(d.classification, CLS_ELIGIBLE)
        d2 = eng.evaluate(symbol="ETHBTC", score=0.70, open_symbols=[], open_count=0)
        self.assertFalse(d2.trade_suggested)
        self.assertEqual(d2.classification, CLS_SIGNAL_CONTINUATION)

    def test_portfolio_reserve_on_cross(self):
        pm = _pm(8)
        eng = LiveEntryEngine(strategy_key="T1", long_threshold=0.65, portfolio=pm, max_open=8)
        eng.evaluate(symbol="ETHBTC", score=0.50, reserve_slot=True)
        d = eng.evaluate(symbol="ETHBTC", score=0.70, reserve_slot=True)
        self.assertTrue(d.trade_suggested)
        self.assertIsNotNone(d.reservation_id)
        self.assertEqual(pm.slots_used(), 1)
        # duplicate while reserved
        eng.evaluate(symbol="SOLBTC", score=0.40, reserve_slot=True)
        d2 = eng.evaluate(symbol="ETHBTC", score=0.40, reserve_slot=True)  # go below
        eng.evaluate(symbol="ETHBTC", score=0.40, reserve_slot=True)
        d3 = eng.evaluate(symbol="ETHBTC", score=0.70, reserve_slot=True)
        self.assertFalse(d3.trade_suggested)
        self.assertEqual(d3.classification, CLS_SAME_PAIR_OPEN)


class TestLiveDisabled(unittest.TestCase):
    def test_live_false_dry_true(self):
        cfg = load_config()
        self.assertFalse(cfg["live"]["enabled"])
        self.assertTrue(cfg["live"]["dry_run"])
        from binance_btc_bot.config_loader import is_live_trading_enabled

        self.assertFalse(is_live_trading_enabled(cfg))

    def test_entry_executor_dry_does_not_write(self):
        from unittest.mock import MagicMock

        from binance_btc_bot.execution.entry import EntryExecutor
        from binance_btc_bot.risk.safety import SafetySystem
        from binance_btc_bot.storage.database import BotDatabase
        import tempfile
        from pathlib import Path
        from binance_btc_bot.exchange.base import OrderResult

        with tempfile.TemporaryDirectory() as td:
            db = BotDatabase(Path(td) / "t.sqlite")
            ex = MagicMock()
            ex.get_symbol_info.return_value = _meta()
            ex.place_entry.return_value = OrderResult(
                ok=True, status="DRY_RUN", dry_run=True, order_type="MARKET", executed_qty=1.0
            )
            pm = _pm(8)
            execu = EntryExecutor(
                ex,
                db,
                SafetySystem(),
                portfolio=pm,
                max_allocation_pct=0.125,
                live_enabled=False,
                dry_run=True,
            )
            t1 = get_strategy("T1")
            att = execu.attempt_entry(
                symbol="ETHBTC",
                strategy=t1,
                price_alt_btc=0.04,
                equity_btc=1.0,
                available_btc=1.0,
                open_exposure_pct=0.0,
            )
            self.assertTrue(att.ok)
            self.assertTrue(att.dry_run)
            # Exchange place_entry was called, but result is dry_run — engine/exchange gates real writes.
            self.assertTrue(ex.place_entry.called)
            self.assertTrue(att.order.dry_run)
            db.close()


class TestEntryEngineWithPortfolioCapacity(unittest.TestCase):
    def test_max_open_via_engine(self):
        pm = _pm(3)
        eng = LiveEntryEngine(strategy_key="T1", long_threshold=0.65, portfolio=pm, max_open=3)
        syms = ["ETHBTC", "SOLBTC", "BNBBTC", "ADABTC"]
        accepted = 0
        for sym in syms:
            eng.evaluate(symbol=sym, score=0.4, reserve_slot=True)
            d = eng.evaluate(symbol=sym, score=0.7, reserve_slot=True)
            if d.trade_suggested:
                accepted += 1
                pm.mark_protected(d.reservation_id, f"t_{sym}")
            else:
                self.assertEqual(d.classification, CLS_MAX_OPEN)
        self.assertEqual(accepted, 3)
        self.assertEqual(pm.slots_used(), 3)


if __name__ == "__main__":
    unittest.main()
