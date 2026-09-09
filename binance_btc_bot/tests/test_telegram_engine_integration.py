"""Stage 7.1 — Telegram control plane → Trading Engine dry-run integration.

Proves: Telegram command → TelegramControlPlane → RuntimeController →
persisted runtime state → RuntimeStrategyProvider / portfolio max / entries
blocker → BinanceBotEngine behavior.

No live Telegram control. No real Binance orders. No YAML/strategy/risk edits.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from binance_btc_bot.config_loader import load_config
from binance_btc_bot.control.hourly_report import build_hourly_report
from binance_btc_bot.control.runtime import (
    OperatorMode,
    RuntimeController,
    RuntimeStateStore,
    default_runtime_state_path,
)
from binance_btc_bot.control.telegram_control import TelegramControlPlane
from binance_btc_bot.execution.engine import BinanceBotEngine
from binance_btc_bot.strategy.trails import get_strategy


CHAT = "integration-chat-7101"


class TestTelegramEngineIntegration(unittest.TestCase):
    """End-to-end dry-run path into the real BinanceBotEngine."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        # Engine resolves relative sqlite as Path(_package_root).parent / sqlite_path
        pkg = root / "pkg"
        pkg.mkdir()
        self.cfg = load_config()
        self.cfg = deepcopy(self.cfg)
        self.cfg.setdefault("live", {})
        self.cfg["live"]["enabled"] = False
        self.cfg["live"]["dry_run"] = True
        self.cfg["live"]["strategy"] = "T4"
        self.cfg["live"]["selector"] = None
        self.cfg.setdefault("storage", {})
        self.cfg["storage"]["sqlite_path"] = "bot.sqlite3"
        self.cfg["_package_root"] = str(pkg)
        self.db_path = root / "bot.sqlite3"

        self.engine = BinanceBotEngine(self.cfg)
        self.assertFalse(self.engine.live_enabled)
        self.assertTrue(self.engine.dry_run)

        self.state_path = default_runtime_state_path(self.engine.db.path)
        self.store = RuntimeStateStore(self.state_path)
        self.reconcile_calls: list[dict] = []
        self.emergency_calls: list[dict] = []
        self.market_sell_attempts = 0

        self.ctrl = RuntimeController(
            self.store,
            strategies_cfg=self.cfg.get("strategies"),
            authorized_chat_id=CHAT,
            on_apply=lambda c: c.apply_to_engine(self.engine),
        )
        self.ctrl.apply_to_engine(self.engine)

        def reconcile_fn() -> dict:
            out = {"ok": True, "dry_run": True, "source": "integration"}
            self.reconcile_calls.append(out)
            self.engine.last_reconciliation_at = "2026-09-06T00:00:00Z"
            return out

        def emergency_fn() -> dict:
            # Mirror attach_runtime_control emergency semantics without live writes.
            self.engine.safety.halt("OPERATOR_EMERGENCY")
            # Explicitly do NOT market-sell; count any accidental sell path if wired later.
            cancelled: list[str] = []
            recon = reconcile_fn()
            out = {
                "halted": True,
                "cancelled": cancelled,
                "reconcile": recon,
                "protection_preserved": True,
                "market_sell": False,
            }
            self.emergency_calls.append(out)
            return out

        self.plane = TelegramControlPlane(
            self.ctrl,
            bot_token="integration-dry-token",
            chat_id=CHAT,
            engine_view=self.engine.control_view,
            reconcile_fn=reconcile_fn,
            emergency_fn=emergency_fn,
        )
        self._uid = 10_000

        # Baseline: T4 / NONE / 8
        self.assertEqual(self.ctrl.state.strategy, "T4")
        self.assertEqual(self.ctrl.state.selector, "NONE")
        self.assertEqual(self.ctrl.state.max_simultaneous_trades, 8)
        self._assert_engine_runtime("T4", None, 8)

    def tearDown(self) -> None:
        try:
            self.engine.stop()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.engine.db.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.tmp.cleanup()
        except Exception:  # noqa: BLE001
            # Windows may keep the sqlite handle briefly; ignore cleanup races.
            pass

    def _cmd(self, text: str) -> str:
        self._uid += 1
        return self.plane.dispatch(chat_id=CHAT, text=text, update_id=self._uid)

    def _assert_engine_runtime(self, strategy: str, selector: str | None, max_n: int) -> None:
        sp = self.engine.strategy_provider
        self.assertEqual(sp.strategy_key(), strategy)
        self.assertEqual(sp.selector_key(), selector)
        self.assertEqual(self.engine.portfolio.max_simultaneous_trades, max_n)
        self.assertEqual(self.engine.entry_engine.strategy_key, strategy)
        self.assertEqual(self.engine.entry_engine.selector_key, selector)
        self.assertEqual(self.engine.entry_engine.max_open, max_n)
        self.assertEqual(self.engine.current_strategy().key, strategy)

    def _open_simulated_position(self, *, strategy: str, selector: str, symbol: str = "ETHBTC") -> str:
        strat = get_strategy(strategy)
        trade_id = self.engine.db.new_trade_id()
        cfg_snap = {
            "arm_sl_activation_trail": strat.arm_sl_activation_trail,
            "activation": strat.activation,
            "trail_distance": strat.trail_distance,
            "selector_at_entry": selector,
            "strategy_at_entry": strategy,
        }
        self.engine.db.insert_trade(
            {
                "trade_id": trade_id,
                "symbol": symbol,
                "strategy": strategy,
                "selector": selector,
                "entry_time": "2026-09-06T12:00:00Z",
                "entry_price": 0.05,
                "quantity": 1.0,
                "btc_value": 0.05,
                "configured_activation": strat.activation,
                "configured_trailing_distance": strat.trail_distance,
                "strategy_config": cfg_snap,
                "binance_oco_list_id": "sim-oco-7101",
                "status": "DRY_RUN_PROTECTED",
            }
        )
        self.engine.db.freeze_strategy_config(strategy, cfg_snap)
        res = self.engine.portfolio.try_reserve(symbol)
        self.assertTrue(res.ok, res.reason)
        assert res.reservation is not None
        self.engine.portfolio.mark_protected(res.reservation.reservation_id, trade_id)
        return trade_id

    def test_telegram_engine_full_integration(self) -> None:
        # --- Apply T3 / B / 3 via real Telegram dispatch path ---
        out = self._cmd("/strategy T3")
        self.assertIn("Confirm", out)
        self.assertIn("Applied", self._cmd("/confirm"))
        out = self._cmd("/selector B")
        self.assertIn("Confirm", out)
        self.assertIn("Applied", self._cmd("/confirm"))
        out = self._cmd("/max 3")
        self.assertIn("Confirm", out)
        self.assertIn("Applied", self._cmd("/confirm"))

        # Persisted runtime state
        raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.assertEqual(raw["strategy"], "T3")
        self.assertEqual(raw["selector"], "B")
        self.assertEqual(raw["max_simultaneous_trades"], 3)

        # Actual engine sees runtime settings
        self._assert_engine_runtime("T3", "B", 3)

        # Existing simulated open position under T3/B protection snapshot
        trade_id = self._open_simulated_position(strategy="T3", selector="B")
        before = next(t for t in self.engine.db.open_trades() if t["trade_id"] == trade_id)
        before_act = before["configured_activation"]
        before_trail = before["configured_trailing_distance"]
        before_oco = before["binance_oco_list_id"]
        before_cfg = json.loads(before["strategy_config_json"])
        self.assertEqual(before["strategy"], "T3")
        self.assertEqual(before["selector"], "B")
        self.assertEqual(before["status"], "DRY_RUN_PROTECTED")

        # --- Change runtime to T5 / E / 5 ---
        self._cmd("/strategy T5")
        self._cmd("/confirm")
        self._cmd("/selector E")
        self._cmd("/confirm")
        self._cmd("/max 5")
        self._cmd("/confirm")
        self._assert_engine_runtime("T5", "E", 5)

        # Existing position unchanged (strategy/selector/protection)
        after = next(t for t in self.engine.db.open_trades() if t["trade_id"] == trade_id)
        self.assertEqual(after["strategy"], "T3")
        self.assertEqual(after["selector"], "B")
        self.assertEqual(after["configured_activation"], before_act)
        self.assertEqual(after["configured_trailing_distance"], before_trail)
        self.assertEqual(after["binance_oco_list_id"], before_oco)
        self.assertEqual(json.loads(after["strategy_config_json"]), before_cfg)
        self.assertEqual(after["status"], "DRY_RUN_PROTECTED")
        self.assertIn("ETHBTC", self.engine.portfolio.open_symbols())

        # Future entries only: provider / entry_engine / portfolio capacity
        self.assertEqual(self.engine.strategy_provider.strategy_key(), "T5")
        self.assertEqual(self.engine.strategy_provider.selector_key(), "E")
        self.assertEqual(self.engine.portfolio.max_simultaneous_trades, 5)
        self.assertEqual(self.engine.portfolio.slots_used(), 1)
        self.assertEqual(self.engine.portfolio.slots_remaining(), 4)

        # --- pause / resume / stop / start against real safety gate ---
        self.assertTrue(self.engine.safety.allow_new_entries())
        self._cmd("/pause")
        self.assertEqual(self.ctrl.state.mode, OperatorMode.PAUSED.value)
        self.assertFalse(self.engine.safety.allow_new_entries())
        # Existing protection still present
        self.assertEqual(
            next(t for t in self.engine.db.open_trades() if t["trade_id"] == trade_id)["status"],
            "DRY_RUN_PROTECTED",
        )

        n_recon = len(self.reconcile_calls)
        self._cmd("/resume")
        self.assertEqual(self.ctrl.state.mode, OperatorMode.RUNNING.value)
        self.assertGreater(len(self.reconcile_calls), n_recon)
        self.assertTrue(self.engine.safety.allow_new_entries())

        self._cmd("/stop")
        self.assertEqual(self.ctrl.state.mode, OperatorMode.STOPPED.value)
        self.assertFalse(self.engine.safety.allow_new_entries())

        n_recon = len(self.reconcile_calls)
        self._cmd("/start")
        self.assertEqual(self.ctrl.state.mode, OperatorMode.RUNNING.value)
        self.assertGreater(len(self.reconcile_calls), n_recon)
        self.assertTrue(self.engine.safety.allow_new_entries())

        # --- emergency ---
        # Real exchange write gate stays dry-run (no live sells possible).
        can_write, reason = self.engine.exchange._writes_allowed()
        self.assertFalse(can_write)
        self.assertIn(reason, {"DRY_RUN", "LIVE_DISABLED"})

        self._cmd("/emergency")
        out = self._cmd("/confirm_emergency")
        self.assertIn("HALTED", out)
        self.assertEqual(self.ctrl.state.mode, OperatorMode.HALTED.value)
        self.assertFalse(self.engine.safety.allow_new_entries())
        self.assertEqual(len(self.emergency_calls), 1)
        self.assertFalse(self.emergency_calls[0].get("market_sell"))
        self.assertTrue(self.emergency_calls[0].get("protection_preserved"))
        self.assertTrue(self.emergency_calls[0].get("reconcile", {}).get("ok"))
        # Simulated protection preserved
        still = next(t for t in self.engine.db.open_trades() if t["trade_id"] == trade_id)
        self.assertEqual(still["status"], "DRY_RUN_PROTECTED")
        self.assertEqual(still["binance_oco_list_id"], "sim-oco-7101")
        self.assertEqual(still["strategy"], "T3")
        self.assertEqual(self.market_sell_attempts, 0)

        # --- restart: runtime state survives ---
        ctrl2 = RuntimeController(
            RuntimeStateStore(self.state_path),
            strategies_cfg=self.cfg.get("strategies"),
            authorized_chat_id=CHAT,
            on_apply=lambda c: c.apply_to_engine(self.engine),
        )
        ctrl2.apply_to_engine(self.engine)
        self.assertEqual(ctrl2.state.strategy, "T5")
        self.assertEqual(ctrl2.state.selector, "E")
        self.assertEqual(ctrl2.state.max_simultaneous_trades, 5)
        self.assertEqual(ctrl2.state.mode, OperatorMode.HALTED.value)
        self._assert_engine_runtime("T5", "E", 5)
        self.assertTrue(ctrl2.blocks_new_entries())
        self.assertFalse(self.engine.safety.allow_new_entries())

        # --- corrupt runtime → fail closed on engine ---
        self.state_path.write_text("{not-json", encoding="utf-8")
        ctrl3 = RuntimeController(
            RuntimeStateStore(self.state_path),
            authorized_chat_id=CHAT,
            on_apply=lambda c: c.apply_to_engine(self.engine),
        )
        ctrl3.apply_to_engine(self.engine)
        self.assertTrue(ctrl3.state.corrupt)
        self.assertTrue(ctrl3.blocks_new_entries())
        self.assertFalse(self.engine.safety.allow_new_entries())

        # --- hourly report uses SAME engine control_view (not a separate config) ---
        # Restore a clean controller for report content checks (corrupt store blocks).
        clean_path = Path(self.tmp.name) / "runtime_clean.json"
        clean_store = RuntimeStateStore(clean_path)
        ctrl_h = RuntimeController(
            clean_store,
            authorized_chat_id=CHAT,
            on_apply=lambda c: c.apply_to_engine(self.engine),
        )
        # Force known runtime for report alignment
        with ctrl_h._lock:
            ctrl_h.state.strategy = "T5"
            ctrl_h.state.selector = "E"
            ctrl_h.state.max_simultaneous_trades = 5
            ctrl_h.state.mode = OperatorMode.HALTED.value
            ctrl_h._persist()
        ctrl_h.apply_to_engine(self.engine)
        view = self.engine.control_view()
        report = build_hourly_report(ctrl_h, view)
        self.assertIn("Strategy:  T5", report)
        self.assertIn("Selector:  E", report)
        self.assertIn("Open/Max:", report)
        self.assertIn("/5", report)
        # View must come from engine (positions / dry_run flags), not a parallel source
        self.assertTrue(view.get("dry_run") is True)
        self.assertFalse(view.get("live_enabled"))
        self.assertEqual(view.get("execution_state"), "DRY_RUN")
        pos_syms = {p.get("symbol") for p in (view.get("positions") or [])}
        self.assertIn("ETHBTC", pos_syms)
        eth = next(p for p in view["positions"] if p.get("symbol") == "ETHBTC")
        self.assertEqual(eth.get("strategy"), "T3")
        self.assertEqual(eth.get("selector"), "B")
        self.assertIn("ETHBTC", report)

        # Posture unchanged
        self.assertFalse(self.engine.live_enabled)
        self.assertTrue(self.engine.dry_run)


if __name__ == "__main__":
    unittest.main()
