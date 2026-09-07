#!/usr/bin/env python3
"""Binance BTC compounding bot entrypoint.

Default: dry-run only. live.enabled remains false.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from binance_btc_bot.config_loader import load_config
from binance_btc_bot.envfile import load_dotenv
from binance_btc_bot.execution.engine import BinanceBotEngine
from binance_btc_bot.logging_setup import setup_logging


def main(argv: list[str] | None = None) -> int:
    # Load repo .env before any credential reads (does not print secrets).
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Binance BTC compounding bot (FixedStrategyProvider→T1, dry-run default)"
    )
    parser.add_argument("--config", default=None, help="Path to binance_bot.yaml")
    parser.add_argument(
        "--stage",
        choices=["1", "2", "3", "4", "5", "recovery", "dry", "all"],
        default=None,
        help="Validation stage (default: dry when not using --status)",
    )
    parser.add_argument("--status", action="store_true", help="Print operational status and exit")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Layer 3 production preflight / Stage-6 readiness (no real orders)",
    )
    parser.add_argument("--symbol", default="ETHBTC")
    parser.add_argument(
        "--demo-score",
        type=float,
        default=None,
        help="DRY-ONLY inject score for --symbol. Forbidden for --first-trade / live path.",
    )
    parser.add_argument(
        "--signal-diag",
        action="store_true",
        help="Compute/log production S diagnostics (timestamp,symbol,relative_price,S_prev,S_curr,cross). No orders.",
    )
    parser.add_argument(
        "--preflight-json",
        action="store_true",
        help="With --preflight, also emit machine-readable JSON after the text report",
    )
    parser.add_argument(
        "--first-trade",
        action="store_true",
        help="Controlled first-real-trade oneshot (max=1). Runs Stage-6 gate; refuses if not PASS.",
    )
    parser.add_argument(
        "--authorize-live",
        action="store_true",
        help="Required with --first-trade to request live arming (still blocked unless Stage-6 gate PASS).",
    )
    parser.add_argument(
        "--telegram-control",
        action="store_true",
        help="Run Telegram runtime control plane (LIVE=false/DRY_RUN=true). No real orders.",
    )
    parser.add_argument(
        "--telegram-control-seconds",
        type=float,
        default=0.0,
        help="With --telegram-control, exit after N seconds (0 = until Ctrl+C).",
    )
    parser.add_argument(
        "--live3-preflight",
        action="store_true",
        help="LIVE-3 deployment preflight (max=8). No real orders; does not auto-arm live trading.",
    )
    parser.add_argument(
        "--live3-preflight-json",
        action="store_true",
        help="With --live3-preflight, also emit machine-readable JSON after the text report",
    )
    parser.add_argument(
        "--live3-arm",
        action="store_true",
        help="Arm LIVE-3 after preflight PASS. Requires --authorize-live + BINANCE_LIVE3_AUTHORIZED=true.",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    log_cfg = cfg.get("logging") or {}
    setup_logging(str(log_cfg.get("level") or "INFO"), log_cfg.get("dir"))

    if args.telegram_control:
        import time as _time

        # Fail closed: never arm live writes from this path.
        cfg.setdefault("live", {})["enabled"] = False
        cfg.setdefault("live", {})["dry_run"] = True
        cfg["_runtime_control"] = True
        engine = BinanceBotEngine(cfg)
        ctrl = engine.attach_runtime_control(start_polling=True, hourly=True)
        print("TELEGRAM_CONTROL started (dry-run, no real orders)")
        print(f"mode={ctrl.state.mode} strategy={ctrl.state.strategy} selector={ctrl.state.selector} max={ctrl.state.max_simultaneous_trades}")
        print(f"corrupt={ctrl.state.corrupt} blocks_new_entries={ctrl.blocks_new_entries()}")
        try:
            if args.telegram_control_seconds and args.telegram_control_seconds > 0:
                _time.sleep(float(args.telegram_control_seconds))
            else:
                while True:
                    _time.sleep(1.0)
        except KeyboardInterrupt:
            print("Stopping telegram control...")
        finally:
            engine.stop()
        return 0

    if args.first_trade:
        if args.demo_score is not None:
            print("REFUSED: --demo-score cannot be used with --first-trade (no forced scores).")
            return 2
        from binance_btc_bot.execution.first_trade import FirstTradeController
        from binance_btc_bot.strategy.score_provider import build_production_score_provider

        # Build a dry engine solely to obtain the public exchange client for klines.
        probe = BinanceBotEngine(cfg)
        score_provider = build_production_score_provider(
            cfg, exchange=probe.exchange, diagnostics=True
        )
        ctrl = FirstTradeController(cfg, score_provider=score_provider)
        report = ctrl.run(authorize_live=bool(args.authorize_live))
        print(report.text())
        try:
            probe.stop()
        except Exception:  # noqa: BLE001
            pass
        # Always non-zero when aborted / not armed for live.
        if report.aborted or not report.armed:
            return 2
        return 0

    if args.live3_arm:
        if args.demo_score is not None:
            print("REFUSED: --demo-score cannot be used with --live3-arm (no forced scores).")
            return 2
        from binance_btc_bot.execution.live3 import Live3Session

        session = Live3Session(cfg)
        report = session.run(authorize_live=bool(args.authorize_live))
        print(report.text())
        if report.aborted or not report.armed:
            return 2
        return 0

    from binance_btc_bot.strategy.score_provider import build_production_score_provider

    engine = BinanceBotEngine(cfg)
    score_provider = build_production_score_provider(
        cfg, exchange=engine.exchange, diagnostics=bool(args.signal_diag)
    )
    engine.score_provider = score_provider

    if args.signal_diag:
        symbols = [args.symbol.upper()] if args.symbol else list(engine.universe)
        if args.symbol.upper() == "ALL":
            symbols = list(engine.universe)
        out = []
        for sym in symbols:
            snap = score_provider.evaluate(sym)
            out.append(snap.to_dict())
            print(snap.log_line())
        print("---JSON---")
        print(json.dumps(out, indent=2, default=str))
        engine.stop()
        return 0

    if args.live3_preflight:
        from binance_btc_bot.execution.live3 import run_live3_preflight

        # Do not use the default engine for LIVE-3 — runner builds a write-blocked probe.
        try:
            engine.stop()
        except Exception:  # noqa: BLE001
            pass
        report = run_live3_preflight(cfg, seed_runtime=True)
        print(report.text())
        if args.live3_preflight_json or args.preflight_json:
            print("---JSON---")
            print(json.dumps(report.to_dict(), indent=2, default=str))
        return 0 if report.ok else 1

    if args.preflight:
        from binance_btc_bot.preflight import run_preflight

        report = run_preflight(cfg, engine=engine)
        print(report.text())
        gate = report.stage6_live_authorize_gate()
        print("")
        print("STAGE6_LIVE_AUTHORIZE_GATE:", "PASS" if gate["ok"] else "FAIL")
        print(gate["message"])
        if args.preflight_json:
            print("---JSON---")
            payload = report.to_dict()
            payload["stage6_live_authorize_gate"] = gate
            print(json.dumps(payload, indent=2, default=str))
        engine.stop()
        return 0 if report.ok else 1

    if args.status:
        # Optional light reconciliation note for status display
        try:
            engine.stage_recovery()
        except Exception:  # noqa: BLE001
            pass
        print(engine.status_text())
        engine.stop()
        return 0

    stage = args.stage or "dry"
    reports = []

    if stage in {"1", "all"}:
        reports.append(engine.stage1_connectivity())
    if stage in {"2", "all"}:
        reports.append(engine.stage2_market_data())
    if stage in {"3", "all"}:
        reports.append(engine.stage3_relative_signal_sample(args.symbol))
    if stage in {"4", "all"}:
        reports.append(engine.stage4_risk_sizing(args.symbol))
    if stage in {"5", "all"}:
        reports.append(engine.stage5_order_validation_dry(args.symbol))
    if stage in {"recovery", "all"}:
        rec = engine.stage_recovery()
        reports.append(type("R", (), {"stage": "recovery", "ok": rec.ok, "details": rec.__dict__})())
    if stage in {"dry", "all"}:
        demo = {}
        if args.demo_score is not None:
            # Explicit dry-only injection for pipeline tests — never used by --first-trade.
            demo[args.symbol.upper()] = float(args.demo_score)
        reports.append(engine.run_dry_pipeline(demo_scores=demo or None))

    out = [{"stage": r.stage, "ok": r.ok, "details": _jsonable(r.details)} for r in reports]
    print(json.dumps(out, indent=2, default=str))
    engine.stop()
    return 0 if all(r.ok for r in reports) else 1


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    return obj


if __name__ == "__main__":
    raise SystemExit(main())
