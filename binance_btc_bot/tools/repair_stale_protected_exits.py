"""One-shot: close local PROTECTED trades that are already flat + ALL_DONE on Binance.

Read-only vs Binance writes: exchange is constructed with dry_run=True (blocks order placement).
Lifecycle uses live_enabled=True / dry_run=False so the live reconcile path runs (DB updates + REST reads only).

Does NOT place orders.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from binance_btc_bot.config_loader import load_config
from binance_btc_bot.credentials import load_binance_credentials_from_env
from binance_btc_bot.envfile import load_dotenv
from binance_btc_bot.exchange.binance import BinanceExchange
from binance_btc_bot.execution.lifecycle import OrderLifecycle
from binance_btc_bot.portfolio.manager import PortfolioConfig, PortfolioManager
from binance_btc_bot.risk.safety import SafetySystem
from binance_btc_bot.storage.database import BotDatabase


def main() -> int:
    load_dotenv()
    cfg = load_config()
    storage = cfg.get("storage") or {}
    db_path = Path(storage.get("sqlite_path") or "data/binance_btc_bot/bot.sqlite3")
    risk = cfg.get("risk") or {}
    portfolio_cfg = cfg.get("portfolio") or {}
    ex_cfg = cfg.get("exchange") or {}

    creds = load_binance_credentials_from_env()
    signer = creds.get_signer() if creds.private_key_path else None
    # dry_run=True on exchange → place_* refuse writes
    exchange = BinanceExchange(
        api_key=creds.api_key,
        signer=signer,
        public_rest_base=str(ex_cfg.get("public_rest_base") or "https://data-api.binance.vision"),
        private_rest_base=str(ex_cfg.get("private_rest_base") or "https://api.binance.com"),
        recv_window_ms=int(ex_cfg.get("recv_window_ms") or 5000),
        dry_run=True,
        live_enabled=False,
    )

    db = BotDatabase(db_path)
    safety = SafetySystem()
    portfolio = PortfolioManager(
        PortfolioConfig(
            max_simultaneous_trades=int(portfolio_cfg.get("max_simultaneous_trades") or 8),
            allocation_per_trade=float(portfolio_cfg.get("allocation_per_trade") or 0.125),
            max_total_allocation=float(portfolio_cfg.get("max_total_allocation") or 1.0),
        )
    )
    open_before = db.open_trades()
    rh = portfolio.rehydrate_from_trades(open_before)
    print(
        "BEFORE",
        json.dumps(
            {
                "db": str(db_path),
                "open_trades": [
                    {
                        "trade_id": t.get("trade_id"),
                        "symbol": t.get("symbol"),
                        "status": t.get("status"),
                        "oco": t.get("binance_oco_list_id"),
                    }
                    for t in open_before
                ],
                "rehydrate": rh,
                "slots": portfolio.slots_used(),
            },
            indent=2,
            default=str,
        ),
    )

    life = OrderLifecycle(
        exchange,
        db,
        safety,
        portfolio=portfolio,
        dry_broker=None,
        live_enabled=True,
        dry_run=False,
        max_loss_per_trade=float(risk.get("max_loss_per_trade") or 0.005),
        max_allocation_pct=float(portfolio_cfg.get("allocation_per_trade") or 0.125),
        max_aggregate_exposure=float(portfolio_cfg.get("max_total_allocation") or 1.0),
    )

    result = life.reconcile_protected_flat_exits()
    print("RECONCILE_PROTECTED_FLAT_EXITS", json.dumps(result, indent=2, default=str))

    open_after = db.open_trades()
    closed_rows = []
    for tid in result.get("closed") or []:
        tr = db.get_trade(tid)
        if not tr:
            continue
        closed_rows.append(
            {
                "trade_id": tid,
                "symbol": tr.get("symbol"),
                "status": tr.get("status"),
                "exit_price": tr.get("exit_price"),
                "exit_time": tr.get("exit_time"),
                "realized_pnl_btc": tr.get("realized_pnl_btc"),
            }
        )
    print(
        "AFTER",
        json.dumps(
            {
                "open_trades": len(open_after),
                "slots": portfolio.slots_used(),
                "closed": closed_rows,
                "safety_reasons": list(safety.reasons),
                "orders_submitted": False,
                "exchange_dry_run": True,
            },
            indent=2,
            default=str,
        ),
    )

    out_path = Path("data/binance_btc_bot/repair_stale_protected_exits.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"result": result, "closed": closed_rows}, indent=2, default=str),
        encoding="utf-8",
    )
    print("WROTE", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
