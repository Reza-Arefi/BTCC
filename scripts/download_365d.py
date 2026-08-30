#!/usr/bin/env python3
"""Force-download >=365 days + warmup of 15m candles for all universe pairs."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from btcc.config import load_config
from btcc.backtest.config import load_backtest_config
from btcc.backtest.data_loader import download_panels, required_bars


def main() -> int:
    cfg = load_config()
    bt = load_backtest_config()
    for k in ("backtest", "backtest_data", "backtest_output", "_root"):
        if k in bt:
            cfg[k] = bt[k]
    days = 365
    warmup = int(cfg["backtest"]["min_warmup_bars"])
    lookback = required_bars(days, warmup)
    print(f"Starting 365d force download lookback_bars={lookback}")
    panels = download_panels(cfg, days=days, warmup_bars=warmup, force=True)
    print("DONE unavailable:", panels.get("unavailable"))
    print("window:", panels.get("window"))
    for base, coin in sorted(panels["coins"].items()):
        rel = coin["rel"]
        print(base, len(rel), rel["timestamp"].min(), rel["timestamp"].max())
    return 0 if not panels.get("unavailable") else 1


if __name__ == "__main__":
    raise SystemExit(main())
