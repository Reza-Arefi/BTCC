#!/usr/bin/env python3
"""Detached 1-year ABC walk-forward (Telegram OFF, no real trading).

Launched via systemd-run as unit btcc-walk-forward-1y.

On SUCCESS:
  FINAL_CHECKPOINT verify → apply Adaptive arm handoff → start live `btcc`
  (Telegram ON, real trading OFF). Historical state is not reset.
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


def _start_live_from_adaptive_checkpoint(cmp_dir: Path) -> dict:
    """Verify Adaptive FINAL_CHECKPOINT, seed live store, start btcc.service."""
    from btcc.config import load_config
    from btcc.sim.checkpoint import verify_final_checkpoint
    from btcc.sim.config import load_sim_config
    from btcc.sim.handoff import apply_checkpoint_to_live_state, load_final_checkpoint
    from btcc.sim.store import SimStore

    adaptive_dirs = sorted(cmp_dir.glob("adaptive_v2_adaptive_*"))
    if not adaptive_dirs:
        raise RuntimeError(f"No adaptive arm dir under {cmp_dir}")
    arm_dir = adaptive_dirs[-1]
    final = arm_dir / "FINAL_CHECKPOINT"
    verify = verify_final_checkpoint(final)
    if not verify.get("ok"):
        raise RuntimeError(f"Adaptive FINAL_CHECKPOINT invalid: {verify.get('errors')}")

    checkpoint = load_final_checkpoint(final)
    cfg = load_config()
    sim = load_sim_config()
    sim["_fallback_factor_weights"] = dict((cfg.get("factors") or {}).get("weights") or {})
    store = SimStore(sim)
    apply_checkpoint_to_live_state(store, checkpoint, fallback_weights=sim["_fallback_factor_weights"])

    # Live: Telegram ON (config), trading OFF (safety). Clear backtest force-off.
    os.environ.pop("BTCC_TELEGRAM_FORCE_OFF", None)

    # Ensure live service is installed/enabled, then start (survives disconnect).
    subprocess.run(
        ["sudo", "systemctl", "start", "btcc.service"],
        check=True,
        cwd=str(ROOT),
    )
    active = subprocess.check_output(
        ["systemctl", "is-active", "btcc.service"], text=True
    ).strip()
    return {
        "adaptive_arm_dir": str(arm_dir),
        "final_checkpoint": str(final),
        "verify_ok": True,
        "live_service": "btcc.service",
        "live_active": active,
        "telegram_live": "ON",
        "allow_trading": False,
    }


def main() -> int:
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    # Hard safety during historical: never enable Telegram / trading in this process
    os.environ["BTCC_TELEGRAM_FORCE_OFF"] = "1"

    from btcc.config import load_config
    from btcc.sim.abc_compare import run_abc_comparison
    from btcc.sim.config import load_sim_config

    cfg = load_config()
    sim = load_sim_config()
    # Full historical: daily checkpoints must refresh analytics/plots
    sim["daily_checkpoint_refresh_analytics"] = True

    allow = bool((cfg.get("safety") or {}).get("allow_trading", False))
    if allow:
        logger.error("REFUSING to run: safety.allow_trading is true")
        return 2

    strategies = sorted((sim.get("strategies") or {}).keys())
    bd = sim.get("btc_d_health") or {}
    capital = float(sim.get("starting_capital_usd", 1000.0))
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        git_sha = "UNKNOWN"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    meta = {
        "experiment": "abc_walk_forward_1y",
        "experiment_kind": "NEW_DAY1_S1_S5_BTCD_CONTEXTUAL",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "days_requested": 365,
        "starting_capital_usd": capital,
        "notional_usd_per_opportunity": float(sim.get("notional_usd", 100.0)),
        "long_threshold": float(sim.get("long_threshold", 0.60)),
        "max_open_opportunities": int(sim.get("max_open_opportunities", 10)),
        "strategies": strategies,
        "btc_d_mode": "CONTEXT_ONLY",
        "require_for_new_trades": bool(bd.get("require_for_new_trades", False)),
        "telegram_backtest": "OFF",
        "allow_trading": False,
        "pid": os.getpid(),
        "unit": "btcc-walk-forward-1y",
    }
    meta_path = ROOT / "results" / f"WALK_FORWARD_1Y_LAUNCH_{stamp}.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("LAUNCH meta → %s", meta_path)
    logger.info(
        "Starting NEW Day-1 ABC walk-forward days=365 capital=$%.0f strategies=%s "
        "btc_d=CONTEXT_ONLY telegram=OFF trading=OFF sha=%s",
        capital, strategies, git_sha,
    )

    cmp_dir = run_abc_comparison(
        days=365,
        force_download=False,
        long_threshold=float(sim.get("long_threshold", 0.60)),
        sim_cfg=sim,
    )
    logger.info("ABC complete → %s", cmp_dir)
    meta["output_directory"] = str(cmp_dir)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

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

    live_info: dict | None = None
    if ok_all:
        try:
            live_info = _start_live_from_adaptive_checkpoint(cmp_dir)
            results["live_handoff"] = live_info
            logger.info("Live simulation started → %s", live_info)
        except Exception as e:
            logger.exception("Live handoff/start failed after successful historical")
            results["live_handoff_error"] = str(e)
            ok_all = False

    (cmp_dir / "WALK_FORWARD_1Y_COMPLETE.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    (ROOT / "results" / "WALK_FORWARD_1Y_LATEST.json").write_text(
        json.dumps({**meta, **results}, indent=2, default=str), encoding="utf-8"
    )
    if not ok_all:
        logger.error("FINAL_CHECKPOINT verification or live handoff failed")
        return 1
    logger.info("Historical walk-forward SUCCESS + live simulation started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
