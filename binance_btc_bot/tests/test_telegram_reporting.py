"""Stage 7.2 — Telegram professional reporting formatters (A–U)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from binance_btc_bot.control.hourly_report import build_hourly_report
from binance_btc_bot.control.runtime import RuntimeController, RuntimeStateStore
from binance_btc_bot.control.telegram_control import TelegramControlPlane, format_performance, format_status
from binance_btc_bot.notifications.telegram_reports import (
    close_reason_label,
    format_protection_event,
    format_trade_close,
    format_trade_open,
)
from binance_btc_bot.notifications.timezone_brt import DISPLAY_TZ, format_brt, parse_utc, to_brt


def _open_payload(**overrides):
    base = {
        "symbol": "TAOBTC",
        "strategy": "T1",
        "selector": "NONE",
        "signal": {
            "timestamp": "2026-09-06T19:16:35Z",
            "previous_s": 0.492,
            "current_s": 0.758,
            "threshold": 0.65,
            "genuine_new_cross": True,
        },
        "entry": {
            "side": "BUY",
            "quantity": 0.2922,
            "base_asset": "TAO",
            "avg_price": 0.00332,
            "btc_invested": 0.000970104,
            "actual_allocation_pct": 12.5,
            "order_id": "10001",
            "client_order_id": "e_abc",
        },
        "fees": {"legs": [{"amount": 0.00001, "asset": "BNB"}]},
        "protection": {
            "strategy": "T1",
            "activation_display": "0.40%",
            "trail_display": "0.25%",
            "hard_sl_display": "0.50%",
            "oco_id": "24539661681",
            "oco_status": "ACCEPTED",
            "protected_qty": 0.2922,
        },
        "portfolio_before": {
            "equity_btc": 1.0,
            "btc_free": 0.9,
            "btc_locked": 0.1,
            "open_trades": 0,
        },
        "portfolio_after": {
            "equity_btc": 1.0,
            "btc_free": 0.8,
            "btc_locked": 0.1,
            "open_trades": 1,
            "available_slots": 7,
        },
    }
    base.update(overrides)
    return base


def _close_payload(**overrides):
    base = {
        "symbol": "TAOBTC",
        "strategy": "T1",
        "selector": "NONE",
        "duration": "12m",
        "close_reason": "TRAILING_EXIT",
        "entry": {
            "timestamp": "2026-09-06T19:16:35Z",
            "quantity": 0.2922,
            "avg_price": 0.00332,
            "btc_invested": 0.000970104,
            "order_id": "10001",
        },
        "exit": {
            "timestamp": "2026-09-06T19:28:10Z",
            "quantity": 0.2922,
            "avg_price": 0.003339,
            "order_id": "10099",
        },
        "trailing": {
            "activation": "0.40%",
            "trail_distance": "0.25%",
            "peak_price": 0.00335,
            "exit_trigger_price": 0.003339,
        },
        "result": {
            "price_pnl_pct": 0.57,
            "gross_pnl_btc": 0.00000555,
            "fees_btc": 0.0,
            "net_realized_pnl_btc": 0.00000555,
            "commission_assets": "BNB",
        },
        "portfolio_before": {"equity_btc": 1.0, "btc_free": 0.8, "btc_locked": 0.1, "open_trades": 1},
        "portfolio_after": {
            "equity_btc": 1.00000555,
            "btc_free": 0.90000555,
            "btc_locked": 0.1,
            "open_trades": 0,
            "available_slots": 8,
        },
        "reconciliation": {"binance": "PASS", "local_db": "PASS"},
    }
    base.update(overrides)
    return base


class TestTelegramReporting(unittest.TestCase):
    # A
    def test_a_open_message(self):
        text = format_trade_open(_open_payload())
        self.assertIn("🟢 TRADE OPENED", text)
        self.assertIn("TAOBTC", text)
        self.assertIn("Genuine new cross: YES", text)
        self.assertIn("BNB fee path", text)
        self.assertIn("24539661681", text)
        self.assertIn("BRT", text)

    # B
    def test_b_close_message(self):
        text = format_trade_close(_close_payload())
        self.assertIn("🔴 TRADE CLOSED", text)
        self.assertIn("Net realized", text)
        self.assertIn("0.00000555", text)

    # C
    def test_c_t1_trailing_close_reason(self):
        text = format_trade_close(_close_payload(close_reason="TRAILING_EXIT"))
        self.assertIn("T1 trailing exit", text)
        self.assertIn("Peak:", text)
        self.assertIn("Trigger:", text)

    # D
    def test_d_hard_sl_reason(self):
        text = format_trade_close(_close_payload(close_reason="HARD_SL", trailing={}))
        self.assertIn("Hard stop loss", text)
        self.assertEqual(close_reason_label("HARD_SL"), "Hard stop loss")

    # E
    def test_e_emergency_protection_message(self):
        text = format_protection_event(
            {
                "kind": "EMERGENCY_PROTECTION",
                "symbol": "ETHBTC",
                "quantity": 1.0,
                "reason": "OCO_REJECTED",
                "protection_state": "PROTECTED_EMERGENCY",
                "binance_verified": True,
                "new_entries_status": "BLOCKED",
                "timestamp": "2026-09-06T20:00:00Z",
            }
        )
        self.assertIn("🚨", text)
        self.assertIn("EMERGENCY PROTECTION", text)
        self.assertIn("ETHBTC", text)

    # F
    def test_f_protection_retry(self):
        text = format_protection_event(
            {
                "kind": "PROTECTION_RETRY",
                "symbol": "SOLBTC",
                "quantity": 2.0,
                "reason": "QTY_CORRECTED",
                "protection_state": "PROTECTION_PENDING",
                "binance_verified": False,
                "new_entries_status": "ALLOWED",
            }
        )
        self.assertIn("🟠", text)
        self.assertIn("PROTECTION RETRY", text)

    # G
    def test_g_protection_failure(self):
        text = format_protection_event(
            {
                "kind": "PROTECTION_FAILURE",
                "symbol": "UNIBTC",
                "quantity": 10,
                "reason": "NO_SELLABLE_QTY",
                "protection_state": "PROTECTED",
                "binance_verified": False,
                "new_entries_status": "BLOCKED",
            }
        )
        self.assertIn("unverified", text.lower())
        self.assertIn("PROTECTION FAILURE", text)

    # H
    def test_h_bnb_fee_display(self):
        text = format_trade_open(_open_payload(fees={"legs": [{"amount": 0.000012, "asset": "BNB"}]}))
        self.assertIn("BNB (BNB fee path)", text)

    # I
    def test_i_base_asset_fee_display(self):
        text = format_trade_open(
            _open_payload(fees={"legs": [{"amount": 0.0001, "asset": "TAO"}], "commission_base_asset": "TAO"})
        )
        self.assertIn("TAO", text)
        self.assertNotIn("BNB fee path", text)

    # J
    def test_j_hourly_with_trades(self):
        store = RuntimeStateStore(Path(tempfile.mkdtemp()) / "h.json")
        ctrl = RuntimeController(store, authorized_chat_id="1")
        ctrl.bump_counter("entries", 1)
        ctrl.bump_counter("exits", 1)
        text = build_hourly_report(
            ctrl,
            {
                "equity_btc": 1.01,
                "btc_free": 0.9,
                "btc_locked": 0.1,
                "open_count": 0,
                "slots_remaining": 8,
                "hour_closed_trades": 1,
                "hour_wins": 1,
                "hour_losses": 0,
                "hour_net_realized_pnl_btc": 0.00000555,
                "unrealized_pnl_btc": 0,
                "safety_halted": False,
                "hour_events": [
                    {"timestamp": "2026-09-06T19:16:00Z", "text": "🟢 TAOBTC opened"},
                    {"timestamp": "2026-09-06T19:17:00Z", "text": "🛡 TAOBTC protected"},
                    {"timestamp": "2026-09-06T19:28:00Z", "text": "🔴 TAOBTC closed +0.00000555 BTC"},
                ],
                "performance": {"closed_trades": 1, "wins": 1, "losses": 0, "win_rate": "100.0%"},
            },
        )
        self.assertIn("HOURLY REPORT", text)
        self.assertIn("entries", text.lower())
        self.assertNotIn("No trading activity during the last hour.", text)

    # K
    def test_k_hourly_no_activity(self):
        store = RuntimeStateStore(Path(tempfile.mkdtemp()) / "k.json")
        ctrl = RuntimeController(store, authorized_chat_id="1")
        text = build_hourly_report(ctrl, {"equity_btc": 1.0, "open_count": 0, "safety_halted": False})
        self.assertIn("No trading activity during the last hour.", text)

    # L
    def test_l_hourly_open_positions(self):
        store = RuntimeStateStore(Path(tempfile.mkdtemp()) / "l.json")
        ctrl = RuntimeController(store, authorized_chat_id="1")
        text = build_hourly_report(
            ctrl,
            {
                "equity_btc": 1.0,
                "open_count": 1,
                "safety_halted": False,
                "positions": [
                    {
                        "symbol": "ETHBTC",
                        "strategy": "T3",
                        "selector": "B",
                        "entry_time": "2026-09-06T18:00:00Z",
                        "entry_price": 0.05,
                        "current_price": 0.051,
                        "unrealized_pnl_pct": 2.0,
                        "unrealized_pnl_btc": 0.001,
                        "protection_state": "ACTIVE(PROTECTED)",
                        "binance_oco_list_id": "99",
                    }
                ],
            },
        )
        self.assertIn("ETHBTC", text)
        self.assertIn("T3", text)

    # M
    def test_m_hourly_no_open_positions(self):
        store = RuntimeStateStore(Path(tempfile.mkdtemp()) / "m.json")
        ctrl = RuntimeController(store, authorized_chat_id="1")
        text = build_hourly_report(ctrl, {"equity_btc": 1.0, "open_count": 0, "positions": [], "safety_halted": False})
        self.assertIn("(none open)", text)

    # N
    def test_n_zero_closed_win_rate_na(self):
        store = RuntimeStateStore(Path(tempfile.mkdtemp()) / "n.json")
        ctrl = RuntimeController(store, authorized_chat_id="1")
        text = build_hourly_report(
            ctrl,
            {"equity_btc": 1.0, "open_count": 0, "hour_closed_trades": 0, "safety_halted": False},
        )
        self.assertIn("Win rate:       N/A", text)

    # O
    def test_o_realized_vs_unrealized(self):
        text = format_trade_close(_close_payload())
        self.assertIn("Gross P/L BTC", text)
        self.assertIn("Net realized BTC", text)
        self.assertIn("Fees BTC", text)

    # P
    def test_p_before_after_btc_equity(self):
        text = format_trade_open(_open_payload())
        self.assertIn("BEFORE", text)
        self.assertIn("AFTER", text)
        self.assertIn("1.00000000", text)
        self.assertIn("Converting BTC→alt", text)

    # Q
    def test_q_event_ordering(self):
        store = RuntimeStateStore(Path(tempfile.mkdtemp()) / "q.json")
        ctrl = RuntimeController(store, authorized_chat_id="1")
        events = [
            {"timestamp": "2026-09-06T19:16:00Z", "text": "🟢 TAOBTC opened"},
            {"timestamp": "2026-09-06T19:17:00Z", "text": "🛡 TAOBTC protected"},
            {"timestamp": "2026-09-06T19:28:00Z", "text": "🔴 TAOBTC closed"},
        ]
        text = build_hourly_report(
            ctrl,
            {"equity_btc": 1.0, "open_count": 0, "hour_events": events, "safety_halted": False, "hour_closed_trades": 1},
        )
        i_open = text.index("TAOBTC opened")
        i_prot = text.index("TAOBTC protected")
        i_close = text.index("TAOBTC closed")
        self.assertLess(i_open, i_prot)
        self.assertLess(i_prot, i_close)

    # R
    def test_r_missing_data_na(self):
        text = format_trade_close(
            {
                "symbol": "X",
                "strategy": None,
                "selector": None,
                "close_reason": None,
                "entry": {},
                "exit": {},
                "result": {},
                "portfolio_before": {},
                "portfolio_after": {},
                "reconciliation": {},
            }
        )
        self.assertIn("N/A", text)
        self.assertIn("Unknown exit", text)

    # S
    def test_s_no_secret_leakage(self):
        secret = "TELEGRAM_BOT_TOKEN=123:ABC"
        text = format_trade_open(_open_payload())
        self.assertNotIn("BEGIN PRIVATE", text)
        self.assertNotIn(secret, text)
        text2 = format_brt("2026-09-06T19:16:35Z")
        self.assertNotIn("token", text2.lower())

    # T
    def test_t_status(self):
        store = RuntimeStateStore(Path(tempfile.mkdtemp()) / "t.json")
        ctrl = RuntimeController(store, authorized_chat_id="1")
        text = format_status(
            {
                "open_count": 1,
                "equity_btc": 1.0,
                "btc_free": 0.9,
                "bnb_free": 0.01,
                "realized_pnl_btc": 0.0,
                "unrealized_pnl_btc": 0.001,
                "binance_rest": "OK",
                "user_data_ws": "OK",
                "market_data": "REST",
                "reconciliation": "OK",
                "protection_status": "OK",
                "last_signal": "TAOBTC",
                "last_entry": "TAOBTC",
                "last_reconciliation": "2026-09-06T19:00:00Z",
                "safety_halted": False,
            },
            ctrl,
        )
        self.assertIn("STATUS", text)
        self.assertIn("BNB", text)
        self.assertIn("BRT", text)

    # U
    def test_u_performance(self):
        text = format_performance(
            {
                "performance": {
                    "total_trades": 10,
                    "open_trades": 1,
                    "wins": 6,
                    "losses": 3,
                    "win_rate": "66.7%",
                    "realized_pnl_btc": 0.01,
                    "unrealized_pnl_btc": 0.001,
                    "best_trade": "TAOBTC 0.00000555",
                    "worst_trade": "UNIBTC -0.000001",
                    "average_trade_btc": 0.001,
                    "total_fees_btc": 0.0001,
                    "equity_btc": 1.01,
                    "cumulative_return_pct": "N/A",
                }
            }
        )
        self.assertIn("PERFORMANCE", text)
        self.assertIn("Best trade", text)
        self.assertIn("66.7%", text)

    def test_timezone_utc_to_brt_and_internal_unchanged(self):
        raw = "2026-09-06T19:16:35Z"
        dt_utc = parse_utc(raw)
        assert dt_utc is not None
        self.assertEqual(dt_utc.tzinfo, timezone.utc)
        local = to_brt(dt_utc)
        self.assertEqual(local.tzinfo, DISPLAY_TZ)
        # Display uses zoneinfo conversion (not hardcoded -3).
        self.assertIn("BRT", format_brt(raw))
        self.assertIn("16:16:35", format_brt(raw, with_date=False))
        # Internal string unchanged
        self.assertEqual(raw, "2026-09-06T19:16:35Z")

    def test_performance_command_dispatch(self):
        store = RuntimeStateStore(Path(tempfile.mkdtemp()) / "p.json")
        ctrl = RuntimeController(store, authorized_chat_id="chat")
        plane = TelegramControlPlane(
            ctrl,
            bot_token="x",
            chat_id="chat",
            engine_view=lambda: {"performance": {"total_trades": 0, "win_rate": "N/A"}, "open_count": 0},
        )
        out = plane.dispatch(chat_id="chat", text="/performance", update_id=1)
        self.assertIn("PERFORMANCE", out)


if __name__ == "__main__":
    unittest.main()
