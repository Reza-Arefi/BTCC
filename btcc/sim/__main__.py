"""python -m btcc.sim — Adaptive V2 backtest / threshold sweep entrypoints."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from btcc.config import load_config
from btcc.safety.no_trading import assert_no_trading_config, install_trading_guards
from btcc.sim.backtest import run_adaptive_sim_backtest, run_threshold_sweep

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="BTCC Adaptive V2 research backtest (PAPER ONLY)")
    p.add_argument("command", choices=["backtest", "threshold-sweep"])
    p.add_argument("--days", type=int, default=None)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--force-download", action="store_true")
    args = p.parse_args(argv)

    # Load .env
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

    install_trading_guards()
    cfg = load_config()
    assert_no_trading_config(cfg)

    if args.command == "backtest":
        days = int(args.days or 365)
        out = run_adaptive_sim_backtest(
            cfg,
            days=days,
            force_download=args.force_download,
            long_threshold=args.threshold,
        )
        print(out)
        return 0

    days = int(args.days or 90)
    out = run_threshold_sweep(days=days, force_download=args.force_download)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
