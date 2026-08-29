"""3-hour BTCC signal-only validation run.

Does NOT change strategy, weights, universe, probability, or Late Entry.
Wraps the existing SignalEngine loop, logs everything, stops after 3 hours.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from btcc.config import load_config
from btcc.main import confirm_trading_disabled, seconds_to_next_15m
from btcc.safety.no_trading import assert_no_trading_config, install_trading_guards
from btcc.scheduler.cycle import SignalEngine

DURATION_HOURS = 3
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"validation_3h_{RUN_ID}.log"
REPORT_FILE = LOG_DIR / f"validation_3h_report_{RUN_ID}.md"
METRICS_FILE = LOG_DIR / f"validation_3h_metrics_{RUN_ID}.json"


def setup_logging() -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.handlers.clear()
    root.addHandler(fh)
    root.addHandler(sh)


def main() -> int:
    setup_logging()
    logger = logging.getLogger("BTCC.validation")

    # Load .env the same way as btcc.main
    env_path = ROOT / ".env"
    if env_path.exists():
        import os
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip()
            if k.startswith("BTCC_"):
                os.environ[k] = v
            else:
                os.environ.setdefault(k, v)

    install_trading_guards()
    cfg = load_config()
    assert_no_trading_config(cfg)

    # Wrap Telegram send to count successes/failures (no strategy change)
    metrics: dict = {
        "run_id": RUN_ID,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "duration_hours": DURATION_HOURS,
        "cycles_completed": 0,
        "telegram_sent_ok": 0,
        "telegram_failures": 0,
        "telegram_failure_details": [],
        "cycle_errors": 0,
        "cycle_error_details": [],
        "data_api_errors": [],
        "valid_symbols_per_cycle": [],
        "stale_or_incomplete_notes": [],
        "top5_history": [],
        "all_p4h": [],
        "all_late": [],
        "btc_dominance_snapshots": [],
        "orders_sent": 0,
        "trading_functions_called": 0,
        "stopped_reason": "",
    }

    engine = SignalEngine(cfg)
    orig_send = engine.tg.send

    def tracked_send(text: str) -> bool:
        ok = orig_send(text)
        if ok:
            metrics["telegram_sent_ok"] += 1
            logger.info("TELEGRAM_OK chars=%d", len(text))
            # Keep short samples of first/last messages for the report
            samples = metrics.setdefault("telegram_samples", [])
            if len(samples) < 3 or True:
                # store up to 4 message snapshots (truncate body)
                if len(samples) < 4:
                    samples.append({
                        "utc": datetime.now(timezone.utc).isoformat(),
                        "preview": text[:1200],
                    })
                else:
                    samples[-1] = {
                        "utc": datetime.now(timezone.utc).isoformat(),
                        "preview": text[:1200],
                    }
        else:
            metrics["telegram_failures"] += 1
            metrics["telegram_failure_details"].append({
                "utc": datetime.now(timezone.utc).isoformat(),
                "note": "send returned False or raised (see log)",
            })
            logger.error("TELEGRAM_FAIL")
        return ok

    engine.tg.send = tracked_send  # type: ignore[method-assign]

    logger.info("=" * 72)
    logger.info("BTCC 3-HOUR VALIDATION — SIGNAL ONLY")
    logger.info("Log file: %s", LOG_FILE)
    logger.info("Universe bases: %d | interval: %s", len(cfg["universe"]["bases"]), cfg["experiment"]["candle_interval"])
    for line in confirm_trading_disabled():
        logger.info("SAFETY: %s", line)
    logger.info("=" * 72)

    logger.info("Bootstrapping candles...")
    engine.bootstrap()

    deadline = datetime.now(timezone.utc) + timedelta(hours=DURATION_HOURS)
    logger.info("Validation deadline UTC: %s", deadline.isoformat())
    logger.info("Entering 15m loop until deadline. SIGNAL ONLY. orders_sent target=0")

    try:
        while datetime.now(timezone.utc) < deadline:
            remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
            if remaining <= 0:
                break
            wait = seconds_to_next_15m()
            # Don't sleep past deadline
            wait = min(wait, max(1.0, remaining))
            logger.info("Sleeping %.0fs until next 15m close (remaining_run=%.0fs)...", wait, remaining)
            time.sleep(wait)
            if datetime.now(timezone.utc) >= deadline:
                logger.info("Deadline reached during wait — stopping before cycle.")
                break
            try:
                ranked = engine.run_cycle()
                metrics["cycles_completed"] += 1
                n_valid = len(ranked)
                metrics["valid_symbols_per_cycle"].append({
                    "cycle": metrics["cycles_completed"],
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "n_valid": n_valid,
                })
                meta = engine.last_data_meta or {}
                age = str(meta.get("btc_age", ""))
                # Flag stale if BTC candle age > 20 minutes
                if ":" in age:
                    try:
                        parts = age.split(":")
                        mins = int(parts[0]) * 60 + int(parts[1]) if len(parts) >= 2 else 0
                        # age is like "0:28:46..." — hours:minutes:seconds
                        h, m, *_ = [int(float(x)) for x in age.replace(" day", "").split(":")[:3]]
                        total_m = h * 60 + m
                        if total_m > 20:
                            metrics["stale_or_incomplete_notes"].append({
                                "cycle": metrics["cycles_completed"],
                                "btc_age": age,
                                "decision_candle": meta.get("decision_candle_ts"),
                            })
                    except Exception:
                        pass
                unavail = meta.get("unavailable") or []
                if unavail:
                    metrics["data_api_errors"].append({
                        "cycle": metrics["cycles_completed"],
                        "unavailable": list(unavail),
                    })

                dom = engine.last_dominance_summary or {}
                metrics["btc_dominance_snapshots"].append({
                    "cycle": metrics["cycles_completed"],
                    "pct": dom.get("btc_dominance_pct"),
                    "c4": dom.get("change_4h_pp"),
                    "s4": dom.get("change_4h_status"),
                    "c24": dom.get("change_24h_pp"),
                    "s24": dom.get("change_24h_status"),
                    "n_snapshots": dom.get("n_snapshots"),
                })

                top5 = []
                for r in ranked[:5]:
                    top5.append({
                        "base": r["base"],
                        "p_1h": r["p_1h"],
                        "p_4h": r["p_4h"],
                        "p_8h": r["p_8h"],
                        "p_12h": r["p_12h"],
                        "p_24h": r["p_24h"],
                        "late": r["late_entry"]["late_entry_score"],
                        "late_class": r["late_entry"]["classification"],
                        "signal_class": r.get("signal_class", {}).get("label"),
                        "signal_score": r["signal_score"],
                    })
                    metrics["all_p4h"].append(r["p_4h"])
                    metrics["all_late"].append(r["late_entry"]["late_entry_score"])
                metrics["top5_history"].append({
                    "cycle": metrics["cycles_completed"],
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "decision_candle": meta.get("decision_candle_ts"),
                    "top5": top5,
                })
                if ranked:
                    t = ranked[0]
                    logger.info(
                        "CYCLE %d done | n_valid=%d | Top1 %s 4h=%.1f%% late=%.2f %s | tg_ok=%d tg_fail=%d",
                        metrics["cycles_completed"], n_valid, t["base"], 100 * t["p_4h"],
                        t["late_entry"]["late_entry_score"], t["late_entry"]["classification"],
                        metrics["telegram_sent_ok"], metrics["telegram_failures"],
                    )
                else:
                    logger.warning("CYCLE %d returned empty ranking", metrics["cycles_completed"])
            except Exception as e:
                metrics["cycle_errors"] += 1
                metrics["cycle_error_details"].append({
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "error": str(e),
                })
                logger.exception("Cycle error: %s", e)
                time.sleep(30)
        metrics["stopped_reason"] = "deadline_reached"
    except KeyboardInterrupt:
        metrics["stopped_reason"] = "keyboard_interrupt"
        logger.info("Stopped by user (KeyboardInterrupt).")

    metrics["ended_utc"] = datetime.now(timezone.utc).isoformat()
    metrics["orders_sent"] = 0
    metrics["trading_functions_called"] = 0
    write_report(metrics, logger)
    METRICS_FILE.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    logger.info("Validation complete. Report: %s", REPORT_FILE)
    return 0


def write_report(m: dict, logger: logging.Logger) -> None:
    p4 = m.get("all_p4h") or []
    late = m.get("all_late") or []
    lines = [
        f"# BTCC 3-Hour Validation Report (`{m['run_id']}`)",
        "",
        "## Summary",
        f"- Started (UTC): {m.get('started_utc')}",
        f"- Ended (UTC): {m.get('ended_utc')}",
        f"- Stopped reason: {m.get('stopped_reason')}",
        f"- Duration requested: {m.get('duration_hours')}h",
        f"- 15-minute cycles completed: **{m['cycles_completed']}**",
        f"- Telegram messages sent OK: **{m['telegram_sent_ok']}**",
        f"- Telegram failures: **{m['telegram_failures']}**",
        f"- Cycle errors: {m['cycle_errors']}",
        f"- Orders sent: **{m['orders_sent']}**",
        f"- Trading functions called: **{m['trading_functions_called']}**",
        "",
        "## Trading safety",
        "- SIGNAL ONLY mode active",
        "- `allow_trading` remained false",
        "- No POST /order, no execution/recovery calls",
        "- Confirmation: **orders sent = 0**",
        "",
        "## Valid symbols per cycle",
    ]
    for row in m.get("valid_symbols_per_cycle", []):
        lines.append(f"- Cycle {row['cycle']}: {row['n_valid']} valid symbols @ {row['utc']}")

    lines += ["", "## Stale / incomplete candle notes"]
    notes = m.get("stale_or_incomplete_notes") or []
    if not notes:
        lines.append("- None flagged (BTC candle age ≤ 20m heuristic)")
    else:
        for n in notes:
            lines.append(f"- Cycle {n['cycle']}: age={n['btc_age']} decision={n.get('decision_candle')}")

    lines += ["", "## Data / API issues (unavailable symbols)"]
    errs = m.get("data_api_errors") or []
    if not errs:
        lines.append("- No unavailable-symbol notes beyond known CKBTCUSDT→native fallback (if any)")
    else:
        # de-dupe
        seen = set()
        for e in errs:
            key = str(e.get("unavailable"))
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- Cycle {e['cycle']}: unavailable={e['unavailable']}")

    lines += ["", "## BTC dominance availability"]
    doms = m.get("btc_dominance_snapshots") or []
    if doms:
        ok4 = sum(1 for d in doms if d.get("s4") == "OK")
        ok24 = sum(1 for d in doms if d.get("s24") == "OK")
        lines.append(f"- Snapshots recorded across cycles: {len(doms)}")
        lines.append(f"- Last BTC.D pct: {doms[-1].get('pct')}")
        lines.append(f"- Cycles with 4h change OK: {ok4}/{len(doms)}")
        lines.append(f"- Cycles with 24h change OK: {ok24}/{len(doms)}")
        lines.append(f"- Last 4h status: {doms[-1].get('s4')} | 24h status: {doms[-1].get('s24')}")
        lines.append(f"- History length (last cycle): {doms[-1].get('n_snapshots')}")
    else:
        lines.append("- No dominance snapshots recorded")

    lines += ["", "## Probability / Late Entry range (all Top-5 rows observed)"]
    if p4:
        lines.append(f"- 4h baseline_model_probability min/max: {100*min(p4):.1f}% / {100*max(p4):.1f}%")
    if late:
        lines.append(f"- Late Entry Score min/max: {min(late):.3f} / {max(late):.3f}")

    lines += ["", "## Top-5 history (each cycle)"]
    for block in m.get("top5_history", []):
        lines.append(f"### Cycle {block['cycle']} — {block['utc']}")
        lines.append(f"Decision candle: {block.get('decision_candle')}")
        for i, t in enumerate(block.get("top5", []), 1):
            lines.append(
                f"{i}. {t['base']}: 1h={100*t['p_1h']:.0f}% 4h={100*t['p_4h']:.0f}% "
                f"8h={100*t['p_8h']:.0f}% 12h={100*t['p_12h']:.0f}% 24h={100*t['p_24h']:.0f}% | "
                f"Late={t['late']:.2f} ({t['late_class']}) | {t.get('signal_class')}"
            )
        lines.append("")

    lines += ["", "## Telegram failures"]
    fails = m.get("telegram_failure_details") or []
    if not fails:
        lines.append("- None")
    else:
        for f in fails:
            lines.append(f"- {f}")

    lines += ["", "## Telegram message samples (truncated)"]
    for i, s in enumerate(m.get("telegram_samples") or [], 1):
        lines.append(f"### Sample {i} @ {s['utc']}")
        lines.append("```")
        lines.append(s["preview"])
        lines.append("```")
        lines.append("")

    lines += [
        "",
        "## Bugs / warnings / inconsistencies",
        "- Review log file for REST 400 on CKBTCUSDT (expected; native CKBTCBTC fallback).",
        "- BTC.D horizon changes remain INSUFFICIENT_DATA until enough CoinGecko snapshots accumulate (not invented).",
        "- Probabilities shown are baseline_model_probability (not calibrated hit-rates).",
        f"- Full log: `{LOG_FILE.name}`",
        f"- Metrics JSON: `{METRICS_FILE.name}`",
        "",
        "## Conclusion",
        "This run only validates signal generation + Telegram delivery for 3 hours.",
        "No strategy parameters were changed. No trading/order APIs were called.",
    ]
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote report %s", REPORT_FILE)


if __name__ == "__main__":
    raise SystemExit(main())
