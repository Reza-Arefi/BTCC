"""Telegram runtime control + hourly report tests (A–Z)."""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from binance_btc_bot.control.hourly_report import HourlyReportScheduler, build_hourly_report
from binance_btc_bot.control.runtime import (
    ALLOWED_MAX_TRADES,
    ALLOWED_SELECTORS,
    ALLOWED_STRATEGIES,
    OperatorMode,
    RuntimeController,
    RuntimeStateStore,
    RuntimeStrategyProvider,
)
from binance_btc_bot.control.telegram_control import TelegramControlPlane, format_config, format_status
from binance_btc_bot.portfolio.manager import PortfolioManager
from binance_btc_bot.risk.safety import SafetySystem
from binance_btc_bot.strategy.trails import get_strategy


class TestTelegramRuntimeControl(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state_path = Path(self.tmp.name) / "runtime_control_state.json"
        self.store = RuntimeStateStore(self.state_path)
        self.chat = "999001"
        self.ctrl = RuntimeController(
            self.store,
            strategies_cfg=None,
            authorized_chat_id=self.chat,
        )
        self.uid = 1

    def tearDown(self):
        self.tmp.cleanup()

    def _cmd(self, text: str, *, chat: str | None = None, uid: int | None = None) -> object:
        if uid is None:
            uid = self.uid
            self.uid += 1
        return self.ctrl.handle_text(chat_id=chat or self.chat, text=text, update_id=uid)

    # A
    def test_a_authorized_user(self):
        r = self._cmd("/status")
        self.assertTrue(r.ok)
        self.assertEqual(r.data.get("dispatch"), "status")

    # B
    def test_b_unauthorized_user(self):
        r = self._cmd("/pause", chat="evil")
        self.assertFalse(r.ok)
        self.assertIn("Unauthorized", r.message)
        self.assertEqual(self.ctrl.state.mode, OperatorMode.RUNNING.value)

    # C
    def test_c_strategy_t1_t10(self):
        for key in sorted(ALLOWED_STRATEGIES):
            r = self._cmd(f"/strategy {key}")
            if r.need_confirm:
                c = self._cmd("/confirm")
                self.assertTrue(c.ok, key)
            else:
                self.assertTrue(r.ok, key)
            self.assertEqual(self.ctrl.state.strategy, key)
            self.assertEqual(self.ctrl.provider.strategy_key(), key)

    # D
    def test_d_invalid_strategy(self):
        r = self._cmd("/strategy T99")
        self.assertFalse(r.ok)
        self.assertEqual(self.ctrl.state.strategy, "T30")

    # E
    def test_e_selector_none_a_f(self):
        for key in sorted(ALLOWED_SELECTORS):
            r = self._cmd(f"/selector {key}")
            self.assertTrue(r.need_confirm, key)
            c = self._cmd("/confirm")
            self.assertTrue(c.ok, key)
            self.assertEqual(self.ctrl.state.selector, key)

    # F
    def test_f_invalid_selector(self):
        r = self._cmd("/selector Z")
        self.assertFalse(r.ok)

    # G
    def test_g_max_whitelist(self):
        for n in sorted(ALLOWED_MAX_TRADES):
            r = self._cmd(f"/max {n}")
            self.assertTrue(r.need_confirm, n)
            c = self._cmd("/confirm")
            self.assertTrue(c.ok, n)
            self.assertEqual(self.ctrl.state.max_simultaneous_trades, n)

    # H
    def test_h_invalid_max(self):
        r = self._cmd("/max 7")
        self.assertFalse(r.ok)
        r2 = self._cmd("/max 2")
        self.assertFalse(r2.ok)

    # I
    def test_i_pause(self):
        r = self._cmd("/pause")
        self.assertTrue(r.ok)
        self.assertEqual(self.ctrl.state.mode, OperatorMode.PAUSED.value)
        self.assertTrue(self.ctrl.blocks_new_entries())

    # J
    def test_j_resume(self):
        self._cmd("/pause")
        r = self._cmd("/resume")
        self.assertTrue(r.ok)
        self.assertEqual(self.ctrl.state.mode, OperatorMode.RUNNING.value)
        self.assertFalse(self.ctrl.blocks_new_entries())

    # K
    def test_k_stop(self):
        r = self._cmd("/stop")
        self.assertTrue(r.ok)
        self.assertEqual(self.ctrl.state.mode, OperatorMode.STOPPED.value)
        self.assertTrue(self.ctrl.blocks_new_entries())

    # L
    def test_l_start(self):
        self._cmd("/stop")
        r = self._cmd("/start")
        self.assertTrue(r.ok)
        self.assertEqual(self.ctrl.state.mode, OperatorMode.RUNNING.value)

    # M
    def test_m_emergency_confirmation(self):
        r = self._cmd("/emergency")
        self.assertTrue(r.need_confirm)
        c = self._cmd("/confirm_emergency")
        self.assertTrue(c.ok)
        self.assertEqual(self.ctrl.state.mode, OperatorMode.HALTED.value)
        self.assertTrue(c.data.get("emergency"))

    # N
    def test_n_emergency_without_confirmation(self):
        self._cmd("/emergency")
        # /confirm is wrong for emergency
        bad = self._cmd("/confirm")
        self.assertFalse(bad.ok)
        self.assertNotEqual(self.ctrl.state.mode, OperatorMode.HALTED.value)

    # O–T dispatch markers
    def test_o_status(self):
        self.assertEqual(self._cmd("/status").data.get("dispatch"), "status")

    def test_p_config(self):
        self.assertEqual(self._cmd("/config").data.get("dispatch"), "config")

    def test_q_positions(self):
        self.assertEqual(self._cmd("/positions").data.get("dispatch"), "positions")

    def test_r_balance(self):
        self.assertEqual(self._cmd("/balance").data.get("dispatch"), "balance")

    def test_s_health(self):
        self.assertEqual(self._cmd("/health").data.get("dispatch"), "health")

    def test_t_reconcile(self):
        self.assertEqual(self._cmd("/reconcile").data.get("dispatch"), "reconcile")

    # U
    def test_u_duplicate_telegram_update(self):
        r1 = self.ctrl.handle_text(chat_id=self.chat, text="/pause", update_id=42)
        self.assertTrue(r1.ok)
        r2 = self.ctrl.handle_text(chat_id=self.chat, text="/pause", update_id=42)
        self.assertTrue(r2.data.get("duplicate"))

    # V
    def test_v_malformed_command(self):
        r = self._cmd("pause")
        self.assertFalse(r.ok)
        r2 = self._cmd("/strategy")
        self.assertFalse(r2.ok)
        r3 = self._cmd("/nope")
        self.assertFalse(r3.ok)

    # W
    def test_w_state_persistence(self):
        self._cmd("/strategy T3")
        self._cmd("/confirm")
        self._cmd("/selector F")
        self._cmd("/confirm")
        self._cmd("/max 4")
        self._cmd("/confirm")
        self._cmd("/pause")
        ctrl2 = RuntimeController(self.store, authorized_chat_id=self.chat)
        self.assertEqual(ctrl2.state.strategy, "T3")
        self.assertEqual(ctrl2.state.selector, "F")
        self.assertEqual(ctrl2.state.max_simultaneous_trades, 4)
        self.assertEqual(ctrl2.state.mode, OperatorMode.PAUSED.value)

    # X
    def test_x_restart_recovery_fail_closed_on_corrupt(self):
        self.state_path.write_text("{not-json", encoding="utf-8")
        ctrl = RuntimeController(RuntimeStateStore(self.state_path), authorized_chat_id=self.chat)
        self.assertTrue(ctrl.state.corrupt)
        self.assertTrue(ctrl.blocks_new_entries())
        self.assertEqual(ctrl.state.mode, OperatorMode.HALTED.value)

    # Y
    def test_y_hourly_report(self):
        sent: list[str] = []
        sched = HourlyReportScheduler(
            self.ctrl,
            view_fn=lambda: {
                "equity_btc": 1.0,
                "btc_free": 0.9,
                "btc_locked": 0.1,
                "stables_alts_as_btc": 0.0,
                "open_count": 0,
                "binance_rest": "OK",
                "user_data_ws": "OK",
                "market_data": "OK",
                "reconciliation": "OK",
                "last_heartbeat": "now",
                "realized_pnl_btc": 0,
                "unrealized_pnl_btc": 0,
                "safety_halted": False,
            },
            send_fn=sent.append,
            interval_sec=3600,
        )
        text = sched.maybe_send(force=True)
        self.assertIsNotNone(text)
        self.assertIn("HOURLY REPORT", text or "")
        self.assertIn("Net realized", text or "")
        self.assertIn("Unrealized", text or "")
        self.assertIn("No trading activity during the last hour.", text or "")
        self.assertEqual(len(sent), 1)

    # Z
    def test_z_secret_redaction(self):
        secret = "SECRET_TOKEN_ABCDEF"
        self.ctrl.handle_text(chat_id=self.chat, text=f"/nope {secret}", update_id=9001)
        blob = self.state_path.read_text(encoding="utf-8")
        self.assertNotIn("TELEGRAM_BOT_TOKEN=", blob)
        self.assertNotIn("BEGIN PRIVATE KEY", blob)
        # Command text is scrubbed in audit via scrub_text (pass-through unless secrets known);
        # ensure status()/config formatters never embed token fields.
        cfg = format_config({"live_enabled": False, "dry_run": True}, self.ctrl)
        self.assertNotIn("bot_token", cfg.lower())
        st = format_status({"open_count": 0, "safety_halted": False}, self.ctrl)
        self.assertNotIn(secret, st)

    def test_strategy_change_affects_provider_only_future(self):
        # Open-trade freeze is external; provider returns new key after confirm.
        before = get_strategy("T1")
        self._cmd("/strategy T5")
        self._cmd("/confirm")
        after = self.ctrl.provider.get_strategy()
        self.assertEqual(after.key, "T5")
        self.assertEqual(before.key, "T1")

    def test_max_change_does_not_close_positions(self):
        pm = PortfolioManager.from_config(
            {"portfolio": {"max_simultaneous_trades": 8, "allocation_per_trade": 0.125, "max_total_allocation": 1.0}}
        )
        r = pm.try_reserve("ETHBTC")
        self.assertTrue(r.ok)
        pm.set_max_simultaneous_trades(3)
        self.assertEqual(pm.slots_used(), 1)
        self.assertIn("ETHBTC", pm.open_symbols())

    def test_pause_stop_never_close_via_safety_gate(self):
        safety = SafetySystem()
        safety.set_entries_blocker(self.ctrl.blocks_new_entries)
        self.assertTrue(safety.allow_new_entries())
        self._cmd("/pause")
        self.assertFalse(safety.allow_new_entries())
        self._cmd("/resume")
        self._cmd("/stop")
        self.assertFalse(safety.allow_new_entries())

    def test_frozen_config_cannot_change_trail_via_telegram(self):
        t1 = get_strategy("T1")
        self._cmd("/strategy T2")
        self._cmd("/confirm")
        # T1 definition unchanged
        t1b = get_strategy("T1")
        self.assertEqual(t1.activation, t1b.activation)
        self.assertEqual(t1.trail_distance, t1b.trail_distance)

    def test_plane_unauthorized_and_status_format(self):
        plane = TelegramControlPlane(
            self.ctrl,
            bot_token="x",
            chat_id=self.chat,
            engine_view=lambda: {
                "open_count": 0,
                "slots_remaining": 8,
                "equity_btc": 1.0,
                "btc_free": 1.0,
                "safety_halted": False,
                "live_enabled": False,
                "dry_run": True,
            },
            reconcile_fn=lambda: {"ok": True},
        )
        out = plane.dispatch(chat_id="bad", text="/pause", update_id=7)
        self.assertEqual(out, "Unauthorized.")
        ok = plane.dispatch(chat_id=self.chat, text="/status", update_id=8)
        self.assertIn("STATUS", ok)
        cfg = format_config({"live_enabled": False, "dry_run": True}, self.ctrl)
        self.assertIn("RUNTIME CHANGEABLE", cfg)
        self.assertIn("FROZEN", cfg)
        self.assertIn("S threshold 0.65", cfg)
        self.assertIn("new-cross-only", cfg)

    def test_startup_reconcile_fail_closed(self):
        plane = TelegramControlPlane(
            self.ctrl,
            bot_token="x",
            chat_id=self.chat,
            engine_view=lambda: {"safety_halted": False, "dry_run": True, "live_enabled": False},
            reconcile_fn=lambda: {"ok": False, "error": "boom"},
        )
        self._cmd("/pause")
        out = plane.dispatch(chat_id=self.chat, text="/resume", update_id=501)
        self.assertIn("FAIL CLOSED", out)
        self.assertEqual(self.ctrl.state.mode, OperatorMode.HALTED.value)
        self.assertTrue(self.ctrl.blocks_new_entries())

    def test_stale_confirm_rejected(self):
        self._cmd("/max 3")
        with self.ctrl._lock:
            assert self.ctrl.state.pending is not None
            self.ctrl.state.pending["created_at"] = time.time() - 10_000
            self.ctrl._persist()
        bad = self._cmd("/confirm")
        self.assertFalse(bad.ok)
        self.assertEqual(self.ctrl.state.max_simultaneous_trades, 8)

    def test_selector_f_included(self):
        self.assertIn("F", ALLOWED_SELECTORS)
        self._cmd("/selector F")
        self._cmd("/confirm")
        self.assertEqual(self.ctrl.provider.selector_key(), "F")

    def test_runtime_strategy_provider_with_candidate_scores(self):
        p = RuntimeStrategyProvider("T1", selector_key="A")
        strat = p.get_strategy({"candidate_scores": {"T1": 0.1, "T3": 0.9, "T2": 0.2}})
        self.assertEqual(strat.key, "T3")


class TestHourlyReportBuild(unittest.TestCase):
    def test_distinguishes_realized_unrealized(self):
        store = RuntimeStateStore(Path(tempfile.mkdtemp()) / "s.json")
        ctrl = RuntimeController(store, authorized_chat_id="1")
        text = build_hourly_report(
            ctrl,
            {
                "equity_btc": 1.0,
                "open_count": 1,
                "realized_pnl_btc": 0.01,
                "unrealized_pnl_btc": -0.002,
                "hour_net_realized_pnl_btc": 0.01,
                "safety_halted": False,
                "performance": {"realized_pnl_btc": 0.01, "unrealized_pnl_btc": -0.002},
            },
        )
        self.assertIn("Net realized", text)
        self.assertIn("Unrealized", text)
        self.assertIn("0.01000000", text)
        self.assertIn("-0.00200000", text)


if __name__ == "__main__":
    unittest.main()
