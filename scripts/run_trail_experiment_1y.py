#!/usr/bin/env python3
"""Launch 365-day trailing-exit experiment (historical only, no live bot).

Run detached via systemd-run as btcc-trail-experiment-1y.
DO NOT start until readiness audit passes.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("trail_experiment_1y")


def main() -> int:
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ["BTCC_TELEGRAM_FORCE_OFF"] = "1"

    from btcc.config import load_config
    from btcc.sim.trail_backtest import run_trail_experiment_backtest
    from btcc.sim.trail_config import load_trail_experiment_config, validate_trail_strategies

    cfg = load_config()
    if bool((cfg.get("safety") or {}).get("allow_trading", False)):
        logger.error("REFUSING: safety.allow_trading is true")
        return 2

    sim = load_trail_experiment_config()
    sim["daily_checkpoint_refresh_analytics"] = True
    errs = validate_trail_strategies(sim)
    if errs:
        logger.error("Strategy validation failed: %s", errs)
        return 2

    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        git_sha = "UNKNOWN"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    meta = {
        "experiment": "trail_exit_geometry_1y",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "days": 365,
        "entry_band": [sim["long_threshold"], sim["upper_threshold"]],
        "strategies": list((sim.get("strategies") or {}).keys()),
        "benchmark": sim.get("benchmark_strategy_key", "trail_3"),
        "telegram": "OFF",
        "allow_trading": False,
        "auto_live_handoff": False,
        "unit": "btcc-trail-experiment-1y",
    }
    meta_path = ROOT / "results" / f"TRAIL_EXPERIMENT_LAUNCH_{stamp}.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Launch meta → %s sha=%s", meta_path, git_sha)

    out_dir = run_trail_experiment_backtest(days=365, force_download=False, sim_cfg=sim)
    meta["output_directory"] = str(out_dir)
    meta["status"] = "COMPLETED"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Trail experiment complete → %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
