"""Portfolio SafetyManager — circuit breaker, defensive mode, stop protection, /restart."""

from __future__ import annotations

import itertools
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from binance_btc_bot.control.runtime import RuntimeController, RuntimeStateStore
from binance_btc_bot.control.telegram_control import TelegramControlPlane
from binance_btc_bot.exchange.base import Balance, OrderResult, SymbolInfo
from binance_btc_bot.execution.lifecycle import OrderLifecycle
from binance_btc_bot.execution.safety_runtime import SafetyRuntime
from binance_btc_bot.market_data.rest import RestMarketData
from binance_btc_bot.risk.safety import SafetySystem
from binance_btc_bot.risk.safety_manager import (
    ACTION_EXIT,
    ACTION_REPLACE,
    STAGE_BREAKEVEN,
    STAGE_TRAIL,
    PositionView,
    SafetyManager,
    SafetyManagerConfig,
    SafetyStateStore,
)
from binance_btc_bot.storage.database import BotDatabase
from binance_btc_bot.strategy.trails import get_strategy

W2 = {"arm_sl_activation_trail": 0.05, "activation": 0.04, "trail_distance": 0.02}
SYMBOLS = ["AAABTC", "BBBBTC", "CCCBTC", "DDDBTC"]


class FakeExchange:
    """Minimal Binance spot fake: OCO lists, balances, market sells."""

    def __init__(self) -> None:
        self.prices: dict[str, float] = {"BTCUSDT": 60000.0}
        self.lists: dict[str, dict] = {}
        self.balances: dict[str, float] = {}
        self.placed_ocos: list = []
        self.cancelled: list[str] = []
        self.market_orders: list = []
        self.fail_place_oco = False
        self.fail_market = False
        self.fail_ping = False
        self.fail_prices = False
        self._ids = itertools.count(1000)

    def ping(self) -> bool:
        if self.fail_ping:
            raise ConnectionError("ping timeout")
        return True

    def get_account(self):
        return SimpleNamespace(balances={})

    def get_balance(self, asset: str) -> Balance:
        return Balance(asset, self.balances.get(asset.upper(), 0.0), 0.0)

    def get_price(self, symbol: str) -> float:
        return self.prices[symbol.upper()]

    def get_prices(self, symbols):
        if self.fail_prices:
            raise ConnectionError("price endpoint down")
        return {s: self.prices[s] for s in symbols if s in self.prices}

    def get_symbol_info(self, symbol: str) -> SymbolInfo:
        return SymbolInfo(symbol, "TRADING", symbol[:-3], "BTC", 0.01, 0.01, None, 1e-8, 0.0001,
                          ("MARKET", "STOP_LOSS", "TAKE_PROFIT"), True, 10, 2000, 10, 2000)

    def add_list(self, symbol: str) -> str:
        oid = str(next(self._ids))
        self.lists[oid] = {"orderListId": int(oid), "symbol": symbol, "listOrderStatus": "EXECUTING",
                           "listStatusType": "EXEC_STARTED", "listClientOrderId": f"t_{oid}"}
        return oid

    def get_open_order_lists(self, symbol=None):
        rows = list(self.lists.values())
        return [r for r in rows if symbol is None or r["symbol"] == symbol]

    def get_open_orders(self, symbol=None):
        return []

    def cancel_order_list(self, symbol, order_list_id=None, list_client_order_id=None):
        self.lists.pop(str(order_list_id), None)
        self.cancelled.append(str(order_list_id))
        return OrderResult(ok=True, order_id=str(order_list_id), status="ALL_DONE")

    def place_trailing_exit(self, req):
        if self.fail_place_oco:
            return OrderResult(ok=False, reason="REJECTED")
        oid = self.add_list(req.symbol)
        self.placed_ocos.append(req)
        return OrderResult(ok=True, order_id=oid, client_order_id=req.list_client_order_id, status="EXECUTING")

    def place_order(self, req):
        if self.fail_market:
            return OrderResult(ok=False, reason="REJECTED")
        px = self.prices[req.symbol]
        self.market_orders.append(req)
        base = req.symbol[:-3]
        self.balances[base] = max(0.0, self.balances.get(base, 0.0) - float(req.quantity))
        return OrderResult(ok=True, order_id=str(next(self._ids)), status="FILLED",
                           executed_qty=float(req.quantity), cumulative_quote_qty=float(req.quantity) * px)


class Clock:
    def __init__(self) -> None:
        self.t = 1_000_000.0

    def __call__(self) -> float:
        return self.t


def make_manager(state_path: Path, clock: Clock | None = None, **cfg) -> SafetyManager:
    return SafetyManager(SafetyManagerConfig(**cfg), SafetyStateStore(state_path), clock=clock or Clock())


class Harness:
    """Engine-shaped object wiring the real lifecycle/DB/SafetySystem to a fake exchange."""

    def __init__(self, tmp: Path, *, live: bool = True, clock: Clock | None = None) -> None:
        self.clock = clock or Clock()
        self.fx = FakeExchange()
        self.db = BotDatabase(tmp / "bot.sqlite3")
        self.safety = SafetySystem()
        self.lifecycle = OrderLifecycle(self.fx, self.db, self.safety, live_enabled=live, dry_run=not live)
        self.manager = make_manager(tmp / "safety_manager_state.json", self.clock)
        self.safety.add_entry_guard(self.manager.blocks_new_entries)
        self.lifecycle.on_trade_closed = self.manager.record_trade_closed
        self.engine = SimpleNamespace(
            db=self.db, exchange=self.fx, lifecycle=self.lifecycle, safety=self.safety,
            live_enabled=live, dry_run=not live, notifications=None,
            portfolio=SimpleNamespace(max_simultaneous_trades=4),
            market=RestMarketData(self.fx, stale_after_sec=30), universe=[], _runtime_controller=None,
        )
        self.rt = SafetyRuntime(self.engine, self.manager)
        self.lifecycle.reconcile_rest = lambda universe=None: {"ok": True}

    def open_trade(self, symbol: str, entry: float, qty: float = 10.0, price: float | None = None) -> str:
        tid = self.db.new_trade_id()
        oco = self.fx.add_list(symbol)
        self.db.insert_trade({
            "trade_id": tid, "symbol": symbol, "strategy": "W2", "entry_price": entry, "quantity": qty,
            "btc_value": entry * qty, "configured_activation": 0.04, "configured_trailing_distance": 0.02,
            "strategy_config": W2, "binance_oco_list_id": oco, "status": "PROTECTED",
            "entry_time": "2026-09-27T00:00:00Z",
        })
        self.fx.prices[symbol] = price if price is not None else entry
        self.fx.balances[symbol[:-3]] = qty
        return tid

    def exchange_sl_fill(self, tid: str) -> None:
        tr = self.db.get_trade(tid)
        self.lifecycle.close_from_exit_fill(
            trade_id=tid, reservation_id=None, exit_price=float(tr["entry_price"]) * 0.95,
            close_reason="HARD_SL")

    def entry_attempt(self, symbol: str = "EEEBTC"):
        return self.lifecycle.run_entry(symbol=symbol, strategy=get_strategy("W2"), price_alt_btc=0.001,
                                        equity_btc=1.0, available_btc=1.0)


class TestSafetyManager(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    # 1
    def test_01_first_sl_does_not_halt(self):
        h = Harness(self.tmp)
        tid = h.open_trade("AAABTC", 0.001)
        h.exchange_sl_fill(tid)
        self.assertEqual(h.manager.state.consecutive_stop_losses, 1)
        self.assertFalse(h.manager.state.safety_halt)
        self.assertTrue(h.safety.allow_new_entries())

    # 2
    def test_02_second_sl_enters_safety_halt(self):
        h = Harness(self.tmp)
        a = h.open_trade("AAABTC", 0.001)
        b = h.open_trade("BBBBTC", 0.002)
        h.exchange_sl_fill(a)
        h.exchange_sl_fill(b)
        self.assertTrue(h.manager.state.safety_halt)
        self.assertIn("CONSECUTIVE_STOP_LOSSES", h.manager.state.halt_reason)
        self.assertEqual(h.manager.mode(), "SAFETY_HALT")
        self.assertFalse(h.safety.allow_new_entries())

    def test_02b_duplicate_close_event_counts_once_and_win_resets(self):
        m = make_manager(self.tmp / "s.json")
        m.record_trade_closed(trade_id="a", symbol="AAABTC", close_reason="HARD_SL")
        self.assertEqual(m.record_trade_closed(trade_id="a", symbol="AAABTC", close_reason="HARD_SL"),
                         "IGNORED_DUPLICATE")
        self.assertEqual(m.state.consecutive_stop_losses, 1)
        m.record_trade_closed(trade_id="b", symbol="BBBBTC", close_reason="TRAILING_EXIT")
        self.assertEqual(m.state.consecutive_stop_losses, 0)
        m.record_trade_closed(trade_id="c", symbol="CCCBTC", close_reason="MANUAL_CLOSE")
        m.record_trade_closed(trade_id="d", symbol="DDDBTC", close_reason="HARD_SL")
        self.assertEqual(m.state.consecutive_stop_losses, 1)
        self.assertFalse(m.state.safety_halt)

    # 3
    def test_03_new_signal_during_safety_halt_rejected(self):
        h = Harness(self.tmp)
        h.manager.enter_halt("2_CONSECUTIVE_STOP_LOSSES")
        res = h.entry_attempt()
        self.assertFalse(res.ok)
        self.assertEqual(res.reason, "SAFETY_HALT")
        self.assertEqual(h.fx.market_orders, [])

    # 4
    def test_04_process_restart_while_halted_stays_halted(self):
        path = self.tmp / "s.json"
        m = make_manager(path)
        m.record_trade_closed(trade_id="a", symbol="AAABTC", close_reason="HARD_SL")
        m.record_trade_closed(trade_id="b", symbol="BBBBTC", close_reason="HARD_SL")
        m2 = make_manager(path)
        self.assertTrue(m2.state.safety_halt)
        self.assertEqual(m2.state.consecutive_stop_losses, 2)
        self.assertFalse(m2.allow_new_entries())
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(data["safety_halt"])
        self.assertIsNotNone(data["last_sl_at"])

    def test_04b_corrupt_state_file_fails_closed(self):
        path = self.tmp / "s.json"
        path.write_text("{not json", encoding="utf-8")
        m = make_manager(path)
        self.assertTrue(m.state.safety_halt)
        self.assertIn("SAFETY_STATE_CORRUPT", m.state.halt_reason)

    # 5
    def test_05_restart_with_failed_check_stays_halted(self):
        h = Harness(self.tmp)
        h.manager.enter_halt("2_CONSECUTIVE_STOP_LOSSES")
        h.fx.fail_ping = True
        msg = h.rt.restart()
        self.assertIn("RESTART BLOCKED", msg)
        self.assertIn("api_connection", msg)
        self.assertTrue(h.manager.state.safety_halt)
        self.assertFalse(h.safety.allow_new_entries())

    # 6
    def test_06_restart_with_all_checks_passing_resumes(self):
        h = Harness(self.tmp)
        h.open_trade("AAABTC", 0.001)
        h.manager.record_trade_closed(trade_id="x", symbol="XBTC", close_reason="HARD_SL")
        h.manager.record_trade_closed(trade_id="y", symbol="YBTC", close_reason="HARD_SL")
        self.assertTrue(h.manager.state.safety_halt)
        msg = h.rt.restart()
        self.assertIn("BOT RESTARTED", msg, msg)
        self.assertIn("Open positions: 1/4", msg)
        self.assertFalse(h.manager.state.safety_halt)
        self.assertEqual(h.manager.state.consecutive_stop_losses, 0)
        self.assertTrue(h.safety.allow_new_entries())

    # 7
    def test_07_three_of_four_at_minus_two_pct_enters_defensive(self):
        m = make_manager(self.tmp / "s.json")
        views = [PositionView("a", "AAABTC", 100, 97.9), PositionView("b", "BBBBTC", 100, 97.6),
                 PositionView("c", "CCCBTC", 100, 97.3), PositionView("d", "DDDBTC", 100, 100.3)]
        self.assertTrue(m.evaluate_portfolio(views))
        self.assertTrue(m.state.defensive_mode)
        self.assertEqual(sum(p["triggering"] for p in m.state.defensive_positions), 3)
        self.assertFalse(m.allow_new_entries())

    def test_07b_individual_losses_do_not_trigger_defensive(self):
        m = make_manager(self.tmp / "s.json")
        one_loser = [PositionView("a", "AAABTC", 100, 97.9), PositionView("b", "BBBBTC", 100, 100.5),
                     PositionView("c", "CCCBTC", 100, 101.2), PositionView("d", "DDDBTC", 100, 100.2)]
        self.assertFalse(m.evaluate_portfolio(one_loser))
        three_open = [PositionView(x, x, 100, 97.0) for x in ("a", "b", "c")]
        self.assertFalse(m.evaluate_portfolio(three_open))
        self.assertFalse(m.state.defensive_mode)
        self.assertEqual(m.plan_protection(one_loser), [])

    def test_07c_defensive_persists_after_recovery_until_restart(self):
        m = make_manager(self.tmp / "s.json")
        m.evaluate_portfolio([PositionView(x, x, 100, 97.5) for x in "abcd"])
        m.evaluate_portfolio([PositionView(x, x, 100, 101.0) for x in "abcd"])
        self.assertTrue(make_manager(self.tmp / "s.json").state.defensive_mode)

    # 8
    def test_08_new_signal_during_defensive_rejected(self):
        h = Harness(self.tmp)
        for s in SYMBOLS[:3]:
            h.open_trade(s, 0.001, price=0.001 * 0.975)
        h.open_trade("DDDBTC", 0.001, price=0.001 * 1.003)
        h.rt.cycle(force=True)
        self.assertTrue(h.manager.state.defensive_mode)
        res = h.entry_attempt()
        self.assertFalse(res.ok)
        self.assertEqual(res.reason, "SAFETY_HALT")
        self.assertEqual(h.fx.market_orders, [])

    # 9
    def test_09_profit_plus_one_pct_tightens_stop(self):
        h = Harness(self.tmp)
        h.manager.evaluate_portfolio([PositionView(x, x, 100, 97.0) for x in "wxyz"])
        tid = h.open_trade("AAABTC", 0.001, price=0.00101)
        old_oco = h.db.get_trade(tid)["binance_oco_list_id"]
        out = h.rt.cycle(force=True)
        res = [a for a in out["actions"] if a["trade_id"] == tid][0]
        self.assertEqual(res["result"], "TIGHTENED")
        self.assertAlmostEqual(res["stop"], 0.001 * 1.002, places=10)
        self.assertIn(old_oco, h.fx.cancelled)
        new_oco = h.db.get_trade(tid)["binance_oco_list_id"]
        self.assertNotEqual(new_oco, old_oco)
        req = h.fx.placed_ocos[-1]
        self.assertEqual(req.above_trailing_delta, 200)
        self.assertGreater(req.below_stop_price, 0.001 * 0.95)
        self.assertEqual(h.manager.protection_for(tid).stage, STAGE_BREAKEVEN)

    def test_09b_profit_protection_inactive_in_normal_mode(self):
        h = Harness(self.tmp)
        tid = h.open_trade("AAABTC", 0.001, price=0.00103)
        out = h.rt.cycle(force=True)
        self.assertEqual(out["actions"], [])
        self.assertEqual(h.fx.placed_ocos, [])
        self.assertEqual(h.db.get_trade(tid)["status"], "PROTECTED")

    # 10
    def test_10_profit_plus_two_pct_activates_trailing(self):
        m = make_manager(self.tmp / "s.json")
        m.evaluate_portfolio([PositionView(x, x, 100, 97.0) for x in "wxyz"])
        acts = m.plan_protection([PositionView("t", "AAABTC", 100.0, 102.5, 95.0)])
        self.assertEqual(len(acts), 1)
        self.assertEqual(acts[0].action, ACTION_REPLACE)
        self.assertEqual(acts[0].stage, STAGE_TRAIL)
        self.assertAlmostEqual(acts[0].new_stop, 102.5 * 0.99, places=9)
        m.confirm_stop("t", acts[0].new_stop, acts[0].stage)
        # Price pulls back: the trailing stop does not move down.
        m.clock.t += 3600
        self.assertEqual(m.plan_protection([PositionView("t", "AAABTC", 100.0, 102.0, 95.0)]), [])
        self.assertAlmostEqual(m.protection_for("t").stop, 102.5 * 0.99, places=9)

    # 11
    def test_11_losing_stop_tightens_never_widens(self):
        m = make_manager(self.tmp / "s.json")
        m.evaluate_portfolio([PositionView(x, x, 100, 97.0) for x in "wxyz"])
        acts = m.plan_protection([PositionView("t", "AAABTC", 100.0, 97.5, 95.0)])
        self.assertEqual(acts[0].action, ACTION_REPLACE)
        first = acts[0].new_stop
        self.assertGreater(first, 95.0)
        self.assertGreaterEqual(first, 97.0 - 1e-9)
        m.confirm_stop("t", first, acts[0].stage)
        m.clock.t += 3600
        self.assertEqual(m.plan_protection([PositionView("t", "AAABTC", 100.0, 97.2, 95.0)]), [])
        self.assertEqual(m.protection_for("t").stop, first)
        with self.assertRaises(ValueError):
            m.confirm_stop("t", first - 0.5, "LOSS_STRONG")
        hard = m.plan_protection([PositionView("t", "AAABTC", 100.0, 96.9, 95.0)])
        self.assertEqual(hard[0].action, ACTION_EXIT)

    def test_11b_existing_tighter_stop_is_not_interfered_with(self):
        m = make_manager(self.tmp / "s.json")
        m.evaluate_portfolio([PositionView(x, x, 100, 97.0) for x in "wxyz"])
        m.register_position("t", "AAABTC", 100.0, 98.2)
        self.assertEqual(m.plan_protection([PositionView("t", "AAABTC", 100.0, 97.9, 98.2)]), [])
        self.assertEqual(m.protection_for("t").stop, 98.2)

    # 12
    def test_12_positions_still_managed_while_entries_blocked(self):
        h = Harness(self.tmp)
        h.manager.enter_halt("2_CONSECUTIVE_STOP_LOSSES")
        h.manager.evaluate_portfolio([PositionView(x, x, 100, 97.0) for x in "wxyz"])
        managed = h.open_trade("AAABTC", 0.001, price=0.001 * 1.012)
        exits = h.open_trade("BBBBTC", 0.002)
        self.assertFalse(h.safety.allow_new_entries())
        out = h.rt.cycle(force=True)
        self.assertTrue(any(a["trade_id"] == managed and a["result"] == "TIGHTENED" for a in out["actions"]))
        h.lifecycle.close_from_exit_fill(trade_id=exits, reservation_id=None, exit_price=0.00210,
                                         close_reason="TRAILING_EXIT")
        self.assertEqual(h.db.get_trade(exits)["status"], "CLOSED")
        self.assertTrue(h.manager.state.safety_halt)

    # 13
    def test_13_no_duplicate_protective_orders(self):
        h = Harness(self.tmp)
        h.manager.evaluate_portfolio([PositionView(x, x, 100, 97.0) for x in "wxyz"])
        tid = h.open_trade("AAABTC", 0.001, price=0.00101)
        h.rt.cycle(force=True)
        h.rt.cycle(force=True)
        self.assertEqual(len(h.fx.get_open_order_lists("AAABTC")), 1)
        self.assertEqual(len(h.fx.placed_ocos), 1)
        # An unexpected second list on the book → halt, never add a third.
        h.fx.add_list("AAABTC")
        h.fx.prices["AAABTC"] = 0.00103
        h.clock.t += 3600
        out = h.rt.cycle(force=True)
        self.assertEqual(out["actions"][0]["result"], "DUPLICATE_LISTS_HALT")
        self.assertEqual(len(h.fx.placed_ocos), 1)
        self.assertTrue(h.manager.state.safety_halt)
        self.assertEqual(h.db.get_trade(tid)["status"], "PROTECTED")

    def test_13b_replace_failure_sells_at_market(self):
        h = Harness(self.tmp)
        h.manager.evaluate_portfolio([PositionView(x, x, 100, 97.0) for x in "wxyz"])
        tid = h.open_trade("AAABTC", 0.001, price=0.00101)
        h.fx.fail_place_oco = True
        out = h.rt.cycle(force=True)
        self.assertEqual(out["actions"][0]["result"], "CLOSED")
        self.assertEqual(len(h.fx.market_orders), 1)
        self.assertEqual(h.db.get_trade(tid)["status"], "CLOSED")
        self.assertEqual(h.manager.state.consecutive_stop_losses, 0)

    def test_13c_replace_and_market_failure_reprotects_or_halts(self):
        h = Harness(self.tmp)
        h.manager.evaluate_portfolio([PositionView(x, x, 100, 97.0) for x in "wxyz"])
        tid = h.open_trade("AAABTC", 0.001, price=0.00101)
        h.fx.fail_place_oco = True
        h.fx.fail_market = True
        out = h.rt.cycle(force=True)
        self.assertEqual(out["actions"][0]["result"], "UNPROTECTED_HALTED")
        self.assertEqual(h.db.get_trade(tid)["status"], "PROTECTION_FAILED")
        self.assertTrue(h.manager.state.safety_halt)
        self.assertFalse(h.safety.allow_new_entries())

    # 14
    def test_14_reconciliation_or_price_failure_blocks_entries(self):
        h = Harness(self.tmp)
        h.open_trade("AAABTC", 0.001)
        h.fx.fail_prices = True
        h.rt.cycle(force=True)
        self.assertFalse(h.manager.monitor_ok)
        self.assertFalse(h.safety.allow_new_entries())
        self.assertEqual(h.entry_attempt().reason, "SAFETY_HALT")
        h.fx.fail_prices = False
        h.rt.cycle(force=True)
        self.assertTrue(h.safety.allow_new_entries())
        # Reconciliation failure also blocks /restart.
        h.manager.enter_halt("TEST")
        h.lifecycle.reconcile_rest = lambda universe=None: {"ok": False, "error": "mismatch"}
        msg = h.rt.restart()
        self.assertIn("RESTART BLOCKED", msg)
        self.assertIn("positions_reconciled", msg)
        self.assertFalse(h.safety.allow_new_entries())

    def test_14b_stale_monitor_blocks_entries(self):
        clock = Clock()
        h = Harness(self.tmp, clock=clock)
        h.rt.cycle(force=True)
        self.assertTrue(h.safety.allow_new_entries())
        clock.t += 10_000
        self.assertFalse(h.safety.allow_new_entries())
        self.assertIn("STALE", h.manager.blocking_reason())


class TestSafetyTelegram(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.tmp = Path(self._tmp.name)
        self.h = Harness(self.tmp)
        self.ctrl = RuntimeController(RuntimeStateStore(self.tmp / "rt.json"), authorized_chat_id="42")
        self.ctrl.apply_to_engine(self.h.engine)
        self.h.engine._runtime_controller = self.ctrl
        self.plane = TelegramControlPlane(self.ctrl, bot_token="", chat_id="42",
                                          engine_view=lambda: {"positions": []},
                                          reconcile_fn=lambda: {"ok": True},
                                          safety_runtime=self.h.rt)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def send(self, text: str) -> str:
        return self.plane.dispatch(chat_id="42", text=text)

    def test_resume_does_not_bypass_halt(self):
        self.h.manager.enter_halt("2_CONSECUTIVE_STOP_LOSSES")
        reply = self.send("/resume")
        self.assertIn("still BLOCKED", reply)
        self.assertTrue(self.h.manager.state.safety_halt)
        self.assertFalse(self.h.safety.allow_new_entries())

    def test_pause_then_restart_flow(self):
        self.send("/pause")
        self.assertFalse(self.h.safety.allow_new_entries())
        self.h.manager.enter_halt("2_CONSECUTIVE_STOP_LOSSES")
        self.assertIn("BOT RESTARTED", self.send("/restart"))
        self.assertTrue(self.h.safety.allow_new_entries())

    def test_status_and_risk(self):
        self.h.manager.record_trade_closed(trade_id="a", symbol="AAABTC", close_reason="HARD_SL")
        status = self.send("/status")
        self.assertIn("Consecutive SLs: 1/2", status)
        self.assertIn("New entries:", status)
        self.assertIn("RISK / SAFETY MANAGER", self.send("/risk"))

    def test_close_requires_confirm(self):
        tid = self.h.open_trade("AAABTC", 0.001)
        reply = self.send("/close AAABTC")
        self.assertIn("/confirm", reply)
        self.assertEqual(self.h.db.get_trade(tid)["status"], "PROTECTED")
        reply = self.send("/confirm")
        self.assertIn("CLOSED", reply)
        self.assertEqual(self.h.db.get_trade(tid)["status"], "CLOSED")
        self.assertEqual(len(self.h.fx.market_orders), 1)
        self.assertEqual(self.h.fx.get_open_order_lists("AAABTC"), [])
        self.assertEqual(self.h.manager.state.consecutive_stop_losses, 0)


if __name__ == "__main__":
    unittest.main()
