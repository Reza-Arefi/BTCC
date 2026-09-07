"""Dry-run exercise of every Telegram control command (no network, no orders).

Does not enable live Telegram control. Does not place real orders.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from binance_btc_bot.control.hourly_report import build_hourly_report
from binance_btc_bot.control.runtime import OperatorMode, RuntimeController, RuntimeStateStore
from binance_btc_bot.control.telegram_control import TelegramControlPlane


def main() -> int:
    tmp = tempfile.TemporaryDirectory()
    path = Path(tmp.name) / "runtime_control_state.json"
    chat = "dryrun-chat"
    store = RuntimeStateStore(path)
    ctrl = RuntimeController(store, authorized_chat_id=chat)
    view = {
        "live_enabled": False,
        "dry_run": True,
        "safety_halted": False,
        "safety_state": "OK",
        "open_count": 0,
        "slots_remaining": 8,
        "equity_btc": 1.0,
        "btc_free": 1.0,
        "btc_locked": 0.0,
        "bnb_free": 0.01,
        "stables_as_btc": 0.0,
        "alts_as_btc": 0.0,
        "last_signal": "none",
        "last_entry": "none",
        "last_exit": "none",
        "last_reconciliation": "n/a",
        "critical_errors": [],
        "positions": [],
        "binance_rest": "OK",
        "user_data_ws": "LOCAL_BUS",
        "market_data": "REST",
        "reconciliation": "OK",
        "telegram": "DRY_RUN",
        "execution_state": "DRY_RUN",
        "last_heartbeat": "now",
        "errors_warnings": "none",
        "protection_failures": 0,
        "unknown_order_states": 0,
        "reconciliation_failures": 0,
        "pending_orders": 0,
        "realized_pnl_btc": 0,
        "unrealized_pnl_btc": 0,
        "fees_btc": 0,
    }
    plane = TelegramControlPlane(
        ctrl,
        bot_token="dry-run-token",
        chat_id=chat,
        engine_view=lambda: view,
        reconcile_fn=lambda: {"ok": True, "dry_run": True},
        emergency_fn=lambda: {"halted": True, "cancelled": [], "protection_preserved": True},
    )

    uid = 1

    def run(text: str) -> str:
        nonlocal uid
        out = plane.dispatch(chat_id=chat, text=text, update_id=uid)
        uid += 1
        return out

    results: list[tuple[str, bool, str]] = []

    def check(name: str, text: str, *, expect_ok_substr: str | None = None) -> None:
        out = run(text)
        ok = True
        if expect_ok_substr and expect_ok_substr not in out:
            ok = False
        results.append((name, ok, out[:200].replace("\n", " | ")))

    # Read-only / reports
    check("help", "/help", expect_ok_substr="Runtime")
    check("status", "/status", expect_ok_substr="STATUS")
    check("config", "/config", expect_ok_substr="FROZEN")
    check("positions", "/positions", expect_ok_substr="POSITIONS")
    check("balance", "/balance", expect_ok_substr="BALANCE")
    check("health", "/health", expect_ok_substr="HEALTH")
    check("reconcile", "/reconcile", expect_ok_substr="RECONCILE")

    # Strategy / selector / max with confirm
    check("strategy_prompt", "/strategy T2", expect_ok_substr="Confirm")
    check("strategy_confirm", "/confirm", expect_ok_substr="Applied")
    check("selector_prompt", "/selector A", expect_ok_substr="Confirm")
    check("selector_confirm", "/confirm", expect_ok_substr="Applied")
    check("max_prompt", "/max 4", expect_ok_substr="Confirm")
    check("max_confirm", "/confirm", expect_ok_substr="Applied")

    # Bot state
    check("pause", "/pause", expect_ok_substr="PAUSED")
    check("resume", "/resume", expect_ok_substr="RUNNING")
    check("stop", "/stop", expect_ok_substr="STOPPED")
    check("start", "/start", expect_ok_substr="RUNNING")

    # Emergency
    check("emergency", "/emergency", expect_ok_substr="EMERGENCY")
    check("confirm_emergency", "/confirm_emergency", expect_ok_substr="HALTED")

    # Unauthorized
    bad = plane.dispatch(chat_id="evil", text="/pause", update_id=uid)
    uid += 1
    results.append(("unauthorized", bad == "Unauthorized.", bad))

    # Hourly
    report = build_hourly_report(ctrl, view)
    results.append(
        (
            "hourly",
            "No trading activity during the last hour." in report and "Net realized" in report,
            report[:120].replace("\n", " | "),
        )
    )

    # Restore defaults for cleanliness in persisted file (not production yaml)
    run("/strategy T1")
    run("/confirm")
    run("/selector NONE")
    run("/confirm")
    run("/max 8")
    run("/confirm")

    failed = [r for r in results if not r[1]]
    print("DRY_RUN_TELEGRAM_COMMAND_EXERCISE")
    print(f"LIVE=false DRY_RUN=true REAL_ORDERS=0")
    print(f"mode={ctrl.state.mode} strategy={ctrl.state.strategy} selector={ctrl.state.selector} max={ctrl.state.max_simultaneous_trades}")
    for name, ok, snippet in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {snippet}")
    print(f"PASS={len(results) - len(failed)} FAIL={len(failed)} TOTAL={len(results)}")
    tmp.cleanup()
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
