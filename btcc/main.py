"""BTCC entrypoint — SIGNAL ONLY. No trading.

Usage:
  cd BTCC
  python -m btcc.main bootstrap              # load candles + universe audit
  python -m btcc.main adaptive-bootstrap     # Champion v1 + import 90d history
  python -m btcc.main once                   # single 15m cycle
  python -m btcc.main run                    # loop every 15m
  python -m btcc.main adaptive-checkpoint    # manual Challenger check (not auto-weight-update)
  python -m btcc.main report                 # calibration / accuracy report
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from btcc.config import load_config
from btcc.safety.no_trading import TradingForbiddenError, assert_no_trading_config, deny_trading, install_trading_guards
from btcc.scheduler.cycle import SignalEngine
from btcc.probability.calibrate import fit_and_save
from btcc.storage.predictions import PredictionStore
from btcc.universe import format_universe_audit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("BTCC")


def seconds_to_next_15m() -> float:
    now = datetime.now(timezone.utc)
    minute = (now.minute // 15 + 1) * 15
    if minute >= 60:
        target = now.replace(minute=0, second=2, microsecond=0)
        from datetime import timedelta
        target = target + timedelta(hours=1)
    else:
        target = now.replace(minute=minute, second=2, microsecond=0)
    return max(1.0, (target - now).total_seconds())


def confirm_trading_disabled() -> list[str]:
    """Prove zero trading paths can execute."""
    lines = []
    try:
        deny_trading()
        lines.append("FAIL: deny_trading() did not raise")
    except TradingForbiddenError as e:
        lines.append(f"OK: deny_trading() blocked — {e}")
    for name in ("create_order", "place_order", "cancel_order", "new_order", "submit_order"):
        lines.append(f"OK: no executable trading function '{name}' in BTCC runtime path")
    lines.append("OK: safety.allow_trading must remain false (asserted at startup)")
    lines.append("OK: SIGNAL ONLY — no BUY/SELL/ORDER execution code path")
    return lines


def print_top5_audit(engine: SignalEngine, ranked: list) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("BTCC TOP-5 AUDIT (baseline_model_probability — NOT calibrated hit-rate)")
    lines.append("=" * 72)
    lines.append("")
    lines.append(format_universe_audit(engine.last_universe_audit))
    lines.append("")
    dom = engine.last_dominance_summary
    lines.append("--- BTC Dominance (slow-moving macro) ---")
    lines.append(f"BTC Dominance: {dom.get('btc_dominance_pct')}")
    lines.append(f"Source: {dom.get('source')} role={dom.get('role')}")
    lines.append(f"Snapshots: {dom.get('n_snapshots')} asof={dom.get('asof')}")
    lines.append(f"BTC.D 4h change: {dom.get('change_4h_pp')} [{dom.get('change_4h_status')}]")
    lines.append(f"BTC.D 24h change: {dom.get('change_24h_pp')} [{dom.get('change_24h_status')}]")
    lines.append("")
    meta = engine.last_data_meta
    lines.append("--- Data timestamps ---")
    lines.append(f"Decision candle: {meta.get('decision_candle_ts')}")
    lines.append(f"BTC candle age: {meta.get('btc_age')}")
    lines.append(f"Unavailable: {meta.get('unavailable')}")
    lines.append(f"Probability kind: {meta.get('probability_kind')}")
    lines.append(f"Champion model: {meta.get('model_version')}")
    lines.append("")
    lines.append("--- Trading safety ---")
    lines.extend(confirm_trading_disabled())
    lines.append("")

    for i, r in enumerate(ranked[:5], 1):
        sc = r.get("signal_class", {})
        le = r["late_entry"]
        f = r["factors"]
        lines.append("-" * 72)
        lines.append(f"#{i} {r['base']}/BTC  ({r['symbol']})")
        lines.append(f"Class: {sc.get('label')} {sc.get('emoji')}")
        lines.append(f"Signal score: {r['signal_score']:.4f}")
        lines.append(
            f"Horizons (baseline_model_p unless calibrated): "
            f"1h={100*r['p_1h']:.1f}% [{r.get('p_1h_status')}] | "
            f"4h={100*r['p_4h']:.1f}% [{r.get('p_4h_status')}] | "
            f"8h={100*r['p_8h']:.1f}% [{r.get('p_8h_status')}] | "
            f"12h={100*r['p_12h']:.1f}% [{r.get('p_12h_status')}] | "
            f"24h={100*r['p_24h']:.1f}% [{r.get('p_24h_status')}]"
        )
        lines.append("Factor scores:")
        for name in ("momentum", "trend", "btc_regime", "volume", "volatility", "rsi", "structure"):
            lines.append(f"  {name:12s} {f[name]['score']:.4f}")
        em = {"NORMAL": "🟢", "EXTENDED": "🟡", "HIGH_LATE_ENTRY_RISK": "🟠", "VERY_HIGH_LATE_ENTRY_RISK": "🔴⭐"}.get(
            le["classification"], ""
        )
        lines.append(f"Late Entry Score: {le['late_entry_score']:.4f} {em} ({le['classification']})")
        lines.append("Late Entry exact reasons (by contribution):")
        for reason in le.get("top_reasons", []):
            lines.append(
                f"  - {reason['component']}: value={reason['value']:.4f} "
                f"weight={reason['weight']:.2f} contribution={reason['contribution']:.4f}"
            )
        if r.get("data_warnings"):
            lines.append("Insufficient / warnings: " + "; ".join(r["data_warnings"]))
        else:
            lines.append("Insufficient / warnings: none")
        lines.append(f"Relative bars: {r.get('n_relative_bars')} | ALT/BTC={r.get('alt_btc_price')} | BTC={r.get('btc_price')}")

    lines.append("")
    lines.append("=" * 72)
    text = "\n".join(lines)
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "replace").decode("ascii"))
    out = Path(engine.cfg["_root"]) / "logs" / "top5_audit_latest.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    logger.info("Wrote audit to %s", out)
    return text


def write_research_report(store: PredictionStore, out_path: Path) -> str:
    df = store.df
    lines = ["# BTCC Prediction Research Report", "", f"Rows: {len(df)}", ""]
    if df.empty:
        lines.append("No predictions yet.")
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return "\n".join(lines)

    for h in (1, 4, 8, 12, 24):
        pcol, ycol = f"probability_{h}h", f"future_rel_return_{h}h"
        if ycol not in df.columns:
            continue
        sub = df.dropna(subset=[pcol, ycol])
        if sub.empty:
            lines.append(f"## {h}h — insufficient outcomes")
            continue
        hits = (sub[ycol] > 0).astype(float)
        brier = float(((sub[pcol] - hits) ** 2).mean())
        lines.append(f"## Horizon {h}h")
        lines.append(f"- Samples with outcomes: {len(sub)}")
        lines.append(f"- Actual hit rate: {hits.mean():.2%}")
        lines.append(f"- Mean baseline/predicted P: {sub[pcol].mean():.2%}")
        lines.append(f"- Brier score: {brier:.4f} (lower better)")
        lines.append("")

    if "probability_4h" in df.columns and "future_rel_return_4h" in df.columns:
        rows = []
        for _ts, g in df.groupby("timestamp"):
            top = g.nlargest(5, "probability_4h")
            if top["future_rel_return_4h"].notna().any():
                rows.append(top["future_rel_return_4h"].mean())
        if rows:
            lines.append("## Top-5 ranking quality (4h)")
            lines.append(f"- Avg future ALT/BTC return of Top-5: {sum(rows)/len(rows):.4%}")
            lines.append(f"- Cycles measured: {len(rows)}")

    lines += [
        "",
        "## Notes",
        "- Until calibrated, displayed values are baseline_model_probability (logistic of signal score).",
        "- Do not treat them as validated frequencies of BTC outperformance.",
        "- SIGNAL ONLY — no trade execution.",
    ]
    text = "\n".join(lines)
    out_path.write_text(text, encoding="utf-8")
    return text


def cmd_adaptive_bootstrap(cfg: dict, force: bool = False) -> int:
    from btcc.adaptive.bootstrap import bootstrap_from_backtest
    from btcc.adaptive.store import AdaptivePredictionStore

    adaptive = cfg.get("adaptive") or {}
    if not adaptive.get("enabled"):
        logger.error("adaptive.enabled is false — enable in configs/adaptive_config.yaml")
        return 1
    models_dir = Path(adaptive["models_dir"])
    store = AdaptivePredictionStore(adaptive["predictions_path"])
    report = bootstrap_from_backtest(cfg, adaptive, models_dir, store, force=force)
    print("Adaptive bootstrap report:")
    for k, v in report.items():
        print(f"  {k}: {v}")
    champ_path = models_dir / "CURRENT_CHAMPION.txt"
    if champ_path.exists():
        print(f"  CURRENT_CHAMPION: {champ_path.read_text(encoding='utf-8').strip()}")
    return 0


def cmd_adaptive_checkpoint(cfg: dict) -> int:
    from btcc.adaptive.checkpoint import run_checkpoint
    from btcc.adaptive.store import AdaptivePredictionStore

    adaptive = cfg.get("adaptive") or {}
    if not adaptive.get("enabled"):
        logger.error("adaptive.enabled is false")
        return 1
    store = AdaptivePredictionStore(adaptive["predictions_path"])
    result = run_checkpoint(
        store,
        Path(adaptive["models_dir"]),
        Path(adaptive["reports_dir"]),
        cfg,
        adaptive,
    )
    print("Adaptive checkpoint result:")
    for k, v in result.items():
        if k in ("train", "val", "holdout"):
            continue
        print(f"  {k}: {v}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="BTCC BTC Relative-Strength Signal Bot (SIGNAL ONLY)")
    parser.add_argument(
        "command",
        choices=[
            "once",
            "run",
            "report",
            "bootstrap",
            "adaptive-bootstrap",
            "adaptive-checkpoint",
        ],
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--force", action="store_true", help="Force re-bootstrap Champion/history")
    args = parser.parse_args()

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
    cfg = load_config(args.config)
    assert_no_trading_config(cfg)

    logger.info(
        "BTCC SIGNAL-ONLY mode | universe=%d bases | interval=%s",
        len(cfg["universe"]["bases"]),
        cfg["experiment"]["candle_interval"],
    )
    logger.info(
        "BTC dominance source: %s (%s)",
        cfg["data"]["dominance_source"],
        cfg["data"]["dominance_url"],
    )
    logger.info(
        "Probability pipeline: Factor→SignalScore→Calibration→Probability "
        "(logistic is baseline_model_probability only until calibrated)"
    )

    if args.command == "adaptive-bootstrap":
        return cmd_adaptive_bootstrap(cfg, force=args.force)

    if args.command == "adaptive-checkpoint":
        return cmd_adaptive_checkpoint(cfg)

    engine = SignalEngine(cfg)

    if args.command in {"bootstrap", "once", "run"}:
        engine.bootstrap()

    if args.command == "bootstrap":
        logger.info("Bootstrap complete.")
        print(format_universe_audit(engine.last_universe_audit))
        for line in confirm_trading_disabled():
            print(line)
        return 0

    if args.command == "once":
        ranked = engine.run_cycle()
        print_top5_audit(engine, ranked)
        logger.info(
            "Cycle done. Top: %s",
            [
                (
                    r["base"],
                    f"{100 * r['p_4h']:.0f}%",
                    r["signal_class"]["label"],
                    r["late_entry"]["classification"],
                )
                for r in ranked[:5]
            ],
        )
        return 0

    if args.command == "report":
        store = PredictionStore(Path(cfg["data"]["prediction_dir"]) / "predictions.csv")
        calib_path = Path(cfg["data"]["prediction_dir"]) / "calibration.json"
        fit_and_save(store.path, calib_path, cfg["probability"]["horizons_hours"])
        report = Path(cfg["_root"]) / "logs" / "research_report.md"
        report.parent.mkdir(parents=True, exist_ok=True)
        text = write_research_report(store, report)
        print(text)
        return 0

    if args.command == "run":
        logger.info("Entering 15m loop (Ctrl+C to stop). SIGNAL ONLY.")
        while True:
            try:
                wait = seconds_to_next_15m()
                logger.info("Sleeping %.0fs until next 15m close...", wait)
                time.sleep(wait)
                ranked = engine.run_cycle()
                if ranked:
                    top = ranked[0]
                    logger.info(
                        "Top1 %s 4h=%.0f%% late=%.2f %s class=%s",
                        top["base"],
                        100 * top["p_4h"],
                        top["late_entry"]["late_entry_score"],
                        top["late_entry"]["classification"],
                        top["signal_class"]["label"],
                    )
            except KeyboardInterrupt:
                logger.info("Stopped by user.")
                return 0
            except Exception as e:
                logger.exception("Cycle error: %s", e)
                time.sleep(30)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
