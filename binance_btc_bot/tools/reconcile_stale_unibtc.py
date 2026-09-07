"""One-shot: verify Binance UNI flat, reconcile stale local PROTECTION_FAILED.

No orders. LIVE writes remain blocked.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from binance_btc_bot.config_loader import load_config
from binance_btc_bot.execution.engine import BinanceBotEngine
from binance_btc_bot.envfile import load_dotenv


def main() -> int:
    load_dotenv()
    cfg = load_config()
    live = cfg.get("live") or {}
    assert live.get("enabled") is False
    assert bool(live.get("dry_run", True)) is True

    engine = BinanceBotEngine(cfg)
    assert engine.dry_run is True
    assert engine.live_enabled is False

    ex = engine.exchange
    meta = ex.get_symbol_info("UNIBTC")
    acct = ex.get_account()
    uni = acct.balances.get("UNI")
    free = float(uni.free) if uni else 0.0
    locked = float(uni.locked) if uni else 0.0
    opens = ex.get_open_orders("UNIBTC")
    lists = ex.get_open_order_lists("UNIBTC")
    # Broader: any open list mentioning UNIBTC
    all_lists = ex.get_open_order_lists()

    verify = {
        "base_asset": meta.base_asset,
        "uni_free": free,
        "uni_locked": locked,
        "open_orders": len(opens),
        "open_lists_filtered": len(lists),
        "open_lists_all": len(all_lists),
        "btc_free": acct.free("BTC"),
        "bnb_free": acct.free("BNB"),
    }
    print("VERIFY", json.dumps(verify, indent=2))
    assert free == 0.0 and locked == 0.0, "UNIBTC balance not flat"
    assert len(opens) == 0 and len(lists) == 0, "UNIBTC orders/OCO not flat"

    before = [t for t in engine.db.open_trades() if str(t.get("symbol")).upper() == "UNIBTC"]
    print("LOCAL_BEFORE", [{k: t.get(k) for k in ("trade_id", "status", "quantity", "binance_oco_list_id")} for t in before])

    r1 = engine.lifecycle.reconcile_stale_protection_failed(universe=["UNIBTC"])
    print("RECONCILE_1", json.dumps(r1, indent=2, default=str))
    r2 = engine.lifecycle.reconcile_stale_protection_failed(universe=["UNIBTC"])
    print("RECONCILE_2", json.dumps(r2, indent=2, default=str))

    after = engine.db.open_trades()
    print("LOCAL_OPEN_AFTER", len(after), after)
    for tid in r1.get("closed") or []:
        tr = engine.db.get_trade(tid)
        print(
            "CLOSED_TRADE",
            {
                "trade_id": tid,
                "status": tr.get("status"),
                "exit_price": tr.get("exit_price"),
                "realized_pnl_btc": tr.get("realized_pnl_btc"),
                "exit_time": tr.get("exit_time"),
                "quantity": tr.get("quantity"),
            },
        )
        cur = engine.db._conn.execute(
            "SELECT event, reason FROM bot_events WHERE trade_id=? AND event IN ('RECONCILED_FLAT','INCIDENT_PRESERVED')",
            (tid,),
        )
        print("EVENTS", [dict(x) for x in cur.fetchall()])

    assert r2.get("closed") == [], "second reconcile must not close again"
    assert not any(str(t.get("symbol")).upper() == "UNIBTC" for t in after)

    # Portfolio slot must not be occupied by UNIBTC after rehydrate
    rh = engine.portfolio.rehydrate_from_trades(engine.db.open_trades())
    print("REHYDRATE", rh)
    assert "UNIBTC" not in engine.portfolio.open_symbols()

    out = {
        "verify": verify,
        "reconcile_1": r1,
        "reconcile_2": r2,
        "rehydrate": rh,
        "live_enabled": False,
        "dry_run": True,
        "orders_submitted": False,
    }
    path = Path("data/binance_btc_bot/stale_unibtc_reconcile.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("WROTE", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
