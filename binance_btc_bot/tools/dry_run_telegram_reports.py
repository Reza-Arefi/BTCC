"""Generate dry-run sample Telegram trade/hourly messages (no network, no orders)."""

from __future__ import annotations

from binance_btc_bot.control.hourly_report import build_hourly_report
from binance_btc_bot.control.runtime import RuntimeController, RuntimeStateStore
from binance_btc_bot.notifications.telegram_reports import (
    format_protection_event,
    format_trade_close,
    format_trade_open,
)
from pathlib import Path
import tempfile


def main() -> int:
    open_msg = format_trade_open(
        {
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
                "order_id": "dry-entry-1",
                "client_order_id": "e_dry",
            },
            "fees": {"legs": [{"amount": 0.00001, "asset": "BNB"}]},
            "protection": {
                "strategy": "T1",
                "activation_display": "0.40%",
                "trail_display": "0.25%",
                "hard_sl_display": "0.50%",
                "oco_id": "dry-oco-1",
                "oco_status": "ACCEPTED",
                "protected_qty": 0.2922,
            },
            "portfolio_before": {"equity_btc": 1.0, "btc_free": 0.9, "btc_locked": 0.0, "open_trades": 0},
            "portfolio_after": {
                "equity_btc": 1.0,
                "btc_free": 0.8,
                "btc_locked": 0.0,
                "open_trades": 1,
                "available_slots": 7,
            },
        }
    )
    prot = format_protection_event(
        {
            "kind": "OCO_ACCEPTED",
            "symbol": "TAOBTC",
            "quantity": 0.2922,
            "reason": "LIST_ACCEPTED",
            "protection_state": "PROTECTED",
            "binance_verified": True,
            "new_entries_status": "ALLOWED",
            "oco_id": "dry-oco-1",
            "timestamp": "2026-09-06T19:17:00Z",
        }
    )
    close_msg = format_trade_close(
        {
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
                "order_id": "dry-entry-1",
            },
            "exit": {
                "timestamp": "2026-09-06T19:28:10Z",
                "quantity": 0.2922,
                "avg_price": 0.003339,
                "order_id": "dry-exit-1",
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
            "portfolio_before": {"equity_btc": 1.0, "btc_free": 0.8, "btc_locked": 0.0, "open_trades": 1},
            "portfolio_after": {
                "equity_btc": 1.00000555,
                "btc_free": 0.90000555,
                "btc_locked": 0.0,
                "open_trades": 0,
                "available_slots": 8,
            },
            "reconciliation": {"binance": "PASS", "local_db": "PASS"},
        }
    )
    tmp = tempfile.TemporaryDirectory()
    ctrl = RuntimeController(RuntimeStateStore(Path(tmp.name) / "r.json"), authorized_chat_id="dry")
    hourly = build_hourly_report(
        ctrl,
        {
            "equity_btc": 1.00000555,
            "btc_free": 0.9,
            "btc_locked": 0.0,
            "bnb_free": 0.05,
            "open_count": 0,
            "slots_remaining": 8,
            "safety_halted": False,
            "hour_closed_trades": 1,
            "hour_wins": 1,
            "hour_losses": 0,
            "hour_net_realized_pnl_btc": 0.00000555,
            "unrealized_pnl_btc": 0,
            "hour_events": [
                {"timestamp": "2026-09-06T19:16:00Z", "text": "🟢 TAOBTC opened"},
                {"timestamp": "2026-09-06T19:17:00Z", "text": "🛡 TAOBTC protected"},
                {"timestamp": "2026-09-06T19:28:00Z", "text": "🔴 TAOBTC closed +0.00000555 BTC"},
            ],
            "performance": {
                "closed_trades": 1,
                "wins": 1,
                "losses": 0,
                "win_rate": "100.0%",
                "realized_pnl_btc": 0.00000555,
            },
            "binance_rest": "OK",
            "user_data_ws": "LOCAL_BUS",
            "market_data": "REST",
            "reconciliation": "OK",
            "execution_state": "DRY_RUN",
            "telegram": "DRY_RUN",
            "last_heartbeat": "2026-09-06T19:30:00Z",
        },
    )
    print("=== TRADE OPEN ===")
    print(open_msg)
    print("\n=== PROTECTION ===")
    print(prot)
    print("\n=== TRADE CLOSE ===")
    print(close_msg)
    print("\n=== HOURLY ===")
    print(hourly)
    print("\nLIVE=false DRY_RUN=true REAL_ORDERS=0")
    tmp.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
