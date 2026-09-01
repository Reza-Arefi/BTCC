#!/usr/bin/env python3
"""Launch 365-day selector experiment (T1–T12 + A–F). Historical only."""

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
logger = logging.getLogger("selector_experiment_1y")


def main() -> int:
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ["BTCC_TELEGRAM_FORCE_OFF"] = "1"

    from btcc.config import load_config
    from btcc.sim.selector_backtest import run_selector_experiment_backtest
    from btcc.sim.selector_config import (
        load_selector_experiment_config,
        pre_run_config_summary,
        validate_selector_experiment,
    )

    cfg = load_config()
    if bool((cfg.get("safety") or {}).get("allow_trading", False)):
        logger.error("REFUSING: safety.allow_trading is true")
        return 2

    sim = load_selector_experiment_config()
    sim["daily_checkpoint_refresh_analytics"] = True
    errs = validate_selector_experiment(sim)
    if errs:
        logger.error("Validation failed: %s", errs)
        return 2

    summary_text = pre_run_config_summary(sim)
    logger.info("PRE-RUN CONFIG:\n%s", summary_text)

    try:
        git_sha = subprocess.check_output(
            ["git", "-c", f"safe.directory={ROOT}", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
    except Exception:
        git_sha = "UNKNOWN"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    meta = {
        "experiment": "selector_experiment_v1_1y",
        "experiment_kind": "selector_experiment_v1",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "days": 365,
        "entry": "S >= 0.60 (no upper cap)",
        "arms": {"fixed": list(range(1, 13)), "selectors": list("ABCDEF")},
        "exhaustion_rejection": False,
        "weight_mode": "static",
        "telegram": "OFF",
        "allow_trading": False,
        "auto_live_handoff": False,
        "unit": "btcc-selector-experiment-1y",
        "pre_run_config": summary_text,
    }
    meta_path = ROOT / "results" / f"SELECTOR_EXPERIMENT_LAUNCH_{stamp}.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Launch meta → %s sha=%s", meta_path, git_sha)

    out_dir = run_selector_experiment_backtest(days=365, force_download=False, sim_cfg=sim)
    meta["output_directory"] = str(out_dir)
    meta["status"] = "COMPLETED"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Selector experiment complete → %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
