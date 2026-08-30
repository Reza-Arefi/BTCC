#!/usr/bin/env python3
"""Detached 1-year ABC walk-forward (Telegram OFF, no real trading).

Launched via systemd-run as unit btcc-walk-forward-1y.
Does NOT start live simulation — that waits for FINAL_CHECKPOINT verification
after this process exits successfully.
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("walk_forward_1y")


def main() -> int:
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    # Hard safety: never enable Telegram / trading in this process
    os.environ["BTCC_TELEGRAM_FORCE_OFF"] = "1"

    from btcc.config import load_config
    from btcc.sim.abc_compare import run_abc_comparison
    from btcc.sim.config import load_sim_config

    cfg = load_config()
    sim = load_sim_config()
    allow = bool((cfg.get("safety") or {}).get("allow_trading", False))
    if allow:
        logger.error("REFUSING to run: safety.allow_trading is true")
        return 2

    capital = float(sim.get("starting_capital_usd", 1000.0))
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        git_sha = "UNKNOWN"

    meta = {
        "experiment": "abc_walk_forward_1y",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "days_requested": 365,
        "starting_capital_usd": capital,
        "notional_usd_per_opportunity": float(sim.get("notional_usd", 100.0)),
        "long_threshold": float(sim.get("long_threshold", 0.60)),
        "max_open_opportunities": int(sim.get("max_open_opportunities", 10)),
        "telegram": "OFF",
        "allow_trading": False,
        "pid": os.getpid(),
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    meta_path = ROOT / "results" / f"WALK_FORWARD_1Y_LAUNCH_{stamp}.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("LAUNCH meta → %s", meta_path)
    logger.info(
        "Starting ABC walk-forward days=365 capital=$%.0f telegram=OFF trading=OFF sha=%s",
        capital, git_sha,
    )

    cmp_dir = run_abc_comparison(
        days=365,
        force_download=False,
        long_threshold=float(sim.get("long_threshold", 0.60)),
    )
    logger.info("ABC complete → %s", cmp_dir)

    # Verify FINAL_CHECKPOINTs
    from btcc.sim.checkpoint import verify_final_checkpoint
    from btcc.sim.handoff import load_final_checkpoint

    results = {"cmp_dir": str(cmp_dir), "arms": {}}
    ok_all = True
    for arm_dir in sorted(cmp_dir.glob("adaptive_v2_*")):
        final = arm_dir / "FINAL_CHECKPOINT"
        v = verify_final_checkpoint(final) if final.exists() else {"ok": False, "errors": ["missing"]}
        loaded = None
        try:
            loaded = load_final_checkpoint(final) if final.exists() else None
        except Exception as e:
            v = {"ok": False, "errors": [str(e)]}
        arm_ok = bool(v.get("ok")) and bool(loaded)
        ok_all = ok_all and arm_ok
        results["arms"][arm_dir.name] = {
            "final_checkpoint": str(final),
            "verify_ok": bool(v.get("ok")),
            "reload_ok": bool(loaded),
            "errors": v.get("errors"),
        }
        logger.info("FINAL %s ok=%s", arm_dir.name, arm_ok)

    results["all_final_ok"] = ok_all
    results["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (cmp_dir / "WALK_FORWARD_1Y_COMPLETE.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    (ROOT / "results" / "WALK_FORWARD_1Y_LATEST.json").write_text(
        json.dumps({**meta, **results}, indent=2, default=str), encoding="utf-8"
    )
    if not ok_all:
        logger.error("FINAL_CHECKPOINT verification failed — not starting live")
        return 1
    logger.info("Historical walk-forward SUCCESS. Live handoff is a separate step.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
