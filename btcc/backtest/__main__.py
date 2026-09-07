"""BTCC 90-day historical prediction research backtest entrypoint.

Usage:
  python -m btcc.backtest --days 90
  python -m btcc.backtest --days 90 --force-download
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from btcc.backtest.analyze import run_all_analyses
from btcc.backtest.config import load_backtest_config
from btcc.backtest.engine import run_backtest
from btcc.backtest.report import generate_plots, generate_report
from btcc.safety.no_trading import assert_no_trading_config, install_trading_guards

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("BTCC.backtest")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="BTCC 90-day historical prediction research (NO trading, NO Telegram)"
    )
    parser.add_argument("--days", type=int, default=90, help="Evaluation window in days")
    parser.add_argument("--config", default=None, help="Backtest config YAML path")
    parser.add_argument("--force-download", action="store_true", help="Re-download candles and dominance")
    parser.add_argument(
        "--compare-to",
        default=None,
        help="Prior results dir name or path for BTC.D before/after comparison",
    )
    args = parser.parse_args(argv)

    # Load .env (BTCC_* and CoinGecko keys)
    env_path = ROOT / ".env"
    if env_path.exists():
        import os
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip()
            if k.startswith("BTCC_") or k in ("COINGECKO_API_KEY",):
                os.environ[k] = v
            else:
                os.environ.setdefault(k, v)

    install_trading_guards()
    cfg = load_backtest_config(args.config)
    assert_no_trading_config(cfg)

    logger.info("=" * 72)
    logger.info("BTCC RESEARCH BACKTEST — SIGNAL ONLY — NO TELEGRAM — NO TRADING")
    logger.info("Universe: %d coins | interval: %s | days: %d",
                len(cfg["universe"]["bases"]), cfg["backtest"]["interval"], args.days)
    logger.info("=" * 72)

    out_dir = run_backtest(cfg, days=args.days, force_download=args.force_download)

    import json
    import pandas as pd

    meta_path = out_dir / "run_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    pred_path = out_dir / "top5_predictions.csv"
    if not pred_path.exists():
        logger.error("No predictions produced.")
        return 1

    df = pd.read_csv(pred_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    analyses = run_all_analyses(df, out_dir)
    report_text = generate_report(out_dir, df, analyses, meta)
    generate_plots(out_dir, df, analyses)

    # Optional before/after comparison
    compare_arg = args.compare_to or "backtest_90d_20260828_005424"
    baseline = Path(compare_arg)
    if not baseline.is_absolute():
        baseline = Path(cfg["backtest_output"]["results_root"]) / compare_arg
    if baseline.exists() and (baseline / "run_meta.json").exists():
        from btcc.backtest.compare import compare_runs
        cmp_text = compare_runs(baseline, out_dir)
        logger.info("Wrote comparison vs %s", baseline.name)
        print("\n" + cmp_text)

    print("\n" + "=" * 72)
    print("BACKTEST COMPLETE")
    print("=" * 72)
    print(f"Period:        {meta['eval_start']} → {meta['eval_end']}")
    print(f"Decision bars: {meta['decision_bars']}")
    print(f"Top-5 rows:    {meta['top5_predictions']}")
    print(f"Valid symbols: {meta['valid_symbols']} / {meta['universe_size']}")
    print(f"BTC.D status:  {meta.get('dominance_status')} ({meta.get('dominance', {}).get('source')})")
    print(f"BTC.D points:  {meta.get('dominance', {}).get('n_points', 0)}")
    print(f"Orders sent:   {meta.get('orders_sent', 0)}")
    print(f"Output:        {out_dir}")
    print("=" * 72)
    print(report_text[:2000] + ("..." if len(report_text) > 2000 else ""))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
