#!/usr/bin/env python3
"""Launch E-memory walk-forward experiment (E-10..E-365). Historical only."""

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
logger = logging.getLogger("selector_E_memory_walkforward")


def main() -> int:
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ["BTCC_TELEGRAM_FORCE_OFF"] = "1"

    from btcc.config import load_config
    from btcc.sim.selector_memory_backtest import run_selector_memory_backtest
    from btcc.sim.selector_memory_config import (
        load_selector_memory_config,
        pre_run_memory_summary,
        validate_selector_memory,
    )

    cfg = load_config()
    if bool((cfg.get("safety") or {}).get("allow_trading", False)):
        logger.error("REFUSING: safety.allow_trading is true")
        return 2

    sim = load_selector_memory_config()
    sim["daily_checkpoint_refresh_analytics"] = True
    errs = validate_selector_memory(sim)
    if errs:
        logger.error("Validation failed: %s", errs)
        return 2

    me = sim.get("selector_memory_experiment") or {}
    eval_days = int(me.get("eval_days", 365))
    warmup_days = int(me.get("warmup_days", 365))
    summary_text = pre_run_memory_summary(sim)
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
        "experiment": "selector_E_memory_walkforward_v1",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "eval_days": eval_days,
        "warmup_days": warmup_days,
        "lookbacks_days": me.get("lookbacks_days"),
        "ewma_half_life_days": me.get("ewma_half_life_days", 7),
        "telegram": "OFF",
        "allow_trading": False,
        "unit": "btcc-selector-E-memory-walkforward",
        "pre_run_config": summary_text,
    }
    meta_path = ROOT / "results" / f"SELECTOR_E_MEMORY_LAUNCH_{stamp}.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Launch meta → %s sha=%s", meta_path, git_sha)

    out_dir_env = os.environ.get("BTCC_MEMORY_OUT_DIR")
    resume = os.environ.get("BTCC_MEMORY_RESUME", "1") != "0"
    out = run_selector_memory_backtest(
        force_download=os.environ.get("BTCC_FORCE_DOWNLOAD", "0") == "1",
        out_dir=Path(out_dir_env) if out_dir_env else None,
        sim_cfg=sim,
        resume=resume,
    )
    meta["out_dir"] = str(out)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Complete → %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
