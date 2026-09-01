#!/usr/bin/env python3
"""Smoke test for trail experiment (few days)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["BTCC_TELEGRAM_FORCE_OFF"] = "1"


def main() -> int:
    from btcc.sim.trail_backtest import run_trail_experiment_backtest
    from btcc.sim.trail_config import load_trail_experiment_config, validate_trail_strategies

    sim = load_trail_experiment_config()
    assert not validate_trail_strategies(sim)
    out = run_trail_experiment_backtest(days=5, force_download=False, sim_cfg=sim)
    legs = out / "strategy_legs.csv"
    assert legs.exists(), "missing strategy_legs.csv"
    print("SMOKE_OK", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
