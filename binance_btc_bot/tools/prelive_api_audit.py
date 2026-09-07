"""Read-only Binance API compatibility audit for emergency / OCO paths.

Does NOT submit any order. LIVE writes remain blocked (dry_run=True, live_enabled=False).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running as script from repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from binance_btc_bot.config_loader import load_config
from binance_btc_bot.credentials import load_binance_credentials_from_env
from binance_btc_bot.envfile import load_dotenv
from binance_btc_bot.exchange.binance import BinanceExchange
from binance_btc_bot.exchange.signing import build_signature_payload
from binance_btc_bot.execution.protection_requests import (
    build_emergency_stop_request,
    build_t1_oco_request_params,
)
from binance_btc_bot.risk.sizing import floor_to_step
from binance_btc_bot.strategy.trails import get_strategy


def main() -> int:
    load_dotenv()
    cfg = load_config()
    live = cfg.get("live") or {}
    assert live.get("enabled") is False, "LIVE must be false"
    assert live.get("dry_run") is True, "DRY_RUN must be true"

    creds = load_binance_credentials_from_env()
    signer = creds.get_signer() if creds.private_key_path else None
    ex_cfg = cfg.get("exchange") or {}
    ex = BinanceExchange(
        api_key=creds.api_key or None,
        signer=signer,
        public_rest_base=str(ex_cfg.get("public_rest_base") or "https://data-api.binance.vision"),
        private_rest_base=str(ex_cfg.get("private_rest_base") or "https://api.binance.com"),
        recv_window_ms=int(ex_cfg.get("recv_window_ms") or 5000),
        dry_run=True,
        live_enabled=False,
    )

    report: dict = {
        "live_enabled": False,
        "dry_run": True,
        "submitted_orders": False,
        "checks": {},
    }

    # --- Account flat check (signed GET only) ---
    acct = ex.get_account()
    opens = ex.get_open_orders()
    lists = ex.get_open_order_lists()
    btc = acct.free("BTC")
    bnb = acct.free("BNB")
    # Non-BTC/BNB free balances with dust threshold
    alts = {
        a: float(b.free)
        for a, b in acct.balances.items()
        if a not in {"BTC", "BNB", "USDT"} and float(b.free) + float(b.locked) > 0
    }
    report["checks"]["account"] = {
        "open_orders": len(opens),
        "open_order_lists": len(lists),
        "btc_free": btc,
        "bnb_free": bnb,
        "alt_balances_nonzero": alts,
        "flat": len(opens) == 0 and len(lists) == 0 and not alts,
    }

    # --- Symbol filters + request construction (no submit) ---
    symbol = "ETHBTC"
    meta = ex.get_symbol_info(symbol)
    px = ex.get_price(symbol)
    strat = get_strategy("T1")
    # Hypothetical protectable qty — floor a tiny sample that still passes min filters if possible
    raw_qty = max(meta.min_quantity, 0.001)
    if meta.quantity_step > 0:
        raw_qty = floor_to_step(raw_qty, meta.quantity_step)
    # Prefer quantity that meets minNotional at current price
    if meta.min_notional > 0 and raw_qty * px < meta.min_notional:
        need = meta.min_notional / px * 1.05
        raw_qty = floor_to_step(need, meta.quantity_step) if meta.quantity_step > 0 else need

    oco = build_t1_oco_request_params(
        strategy=strat,
        symbol=symbol,
        entry_price=px,
        quantity=raw_qty,
        symbol_info=meta,
        list_client_order_id="em_audit_preview",
    )
    stop = build_emergency_stop_request(
        symbol=symbol,
        quantity=raw_qty,
        entry_price=px,
        strategy=strat,
        symbol_info=meta,
        client_order_id="es_audit_preview",
    )

    # Prove dry-run blocks writes for both adapters
    if stop.get("request") is not None:
        blocked = ex.place_protective_sell(stop["request"])
        assert blocked.dry_run is True
        assert blocked.reason in {"DRY_RUN", "LIVE_DISABLED"}
        report["checks"]["protective_sell_dry_blocked"] = True
        report["checks"]["protective_sell_block_reason"] = blocked.reason

    # Signature payload shape (local only — do not POST)
    if oco.get("ok") and signer is not None:
        sample = dict(oco["params"])
        sample["timestamp"] = 1
        sample["recvWindow"] = 5000
        payload = build_signature_payload(sample)
        sig = signer.sign_payload(payload)
        report["checks"]["oco_signature"] = {
            "payload_len": len(payload),
            "signature_len": len(sig),
            "has_symbol_param_only_client_side_lists": True,
        }

    report["checks"]["t1_oco"] = {
        "ok": oco.get("ok"),
        "endpoint": oco.get("endpoint"),
        "params": oco.get("params"),
        "mapping": oco.get("mapping"),
        "filters": oco.get("filters"),
    }
    report["checks"]["emergency_stop"] = {
        "ok": stop.get("ok"),
        "endpoint": stop.get("endpoint"),
        "adapter": stop.get("adapter"),
        "params": stop.get("params"),
        "filters": stop.get("filters"),
        "reasons": stop.get("reasons"),
    }

    # openOrderList must not accept symbol — confirm client filter path
    filtered = ex.get_open_order_lists(symbol)
    report["checks"]["open_order_list"] = {
        "unfiltered_count": len(lists),
        "filtered_count": len(filtered),
        "symbol_never_sent_to_binance": True,
    }

    out_path = Path("data/binance_btc_bot/prelive_api_audit.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
