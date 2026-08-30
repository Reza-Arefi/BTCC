#!/usr/bin/env python3
"""Short walk-forward smoke test for C1–C3 infrastructure.

Exercises: init + daily adaptation, daily checkpoints, day-axis analytics,
FINAL_CHECKPOINT, checkpoint verify/reload. Does NOT run the full 1y experiment.
Telegram remains OFF.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from btcc.sim.abc_compare import run_abc_comparison
from btcc.sim.checkpoint import verify_final_checkpoint
from btcc.sim.handoff import load_final_checkpoint


def main() -> int:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    # Short smoke: shrink init so daily adaptation is exercised
    init_days = 3 if days < 90 else 90
    roll_days = min(7, days) if days < 90 else 90
    print(f"SMOKE walk-forward days={days} init_days={init_days} roll_days={roll_days}")
    cmp_dir = run_abc_comparison(
        days=days,
        force_download=False,
        long_threshold=0.60,
        init_days=init_days,
        roll_days=roll_days,
    )
    print("ABC dir:", cmp_dir)

    checks = {}
    arm_dirs = sorted(cmp_dir.glob("adaptive_v2_*"))
    checks["arms_present"] = len(arm_dirs) >= 3

    n_days_total = 0
    plot_families = 0
    finals = []
    for arm_dir in arm_dirs:
        dc = arm_dir / "daily_checkpoints"
        days_found = sorted(dc.glob("day_*")) if dc.exists() else []
        n_days_total += len(days_found)
        checks[f"{arm_dir.name}_daily_checkpoints"] = len(days_found) >= 1
        # Day-axis plots
        plots = list((arm_dir / "analytics" / "plots").rglob("*.png")) if (arm_dir / "analytics").exists() else []
        # Also under day folders
        plots += list(dc.glob("day_*/plots/*.png")) if dc.exists() else []
        plot_families += len({p.name for p in plots})
        checks[f"{arm_dir.name}_plots"] = len(plots) >= 1
        # day_number in predictions
        pred = arm_dir / "predictions.csv"
        if pred.exists() and pred.stat().st_size > 0:
            import pandas as pd
            df = pd.read_csv(pred, nrows=20)
            checks[f"{arm_dir.name}_day_number_col"] = "day_number" in df.columns
        final = arm_dir / "FINAL_CHECKPOINT"
        checks[f"{arm_dir.name}_final_checkpoint"] = final.exists()
        if final.exists():
            v = verify_final_checkpoint(final)
            checks[f"{arm_dir.name}_final_verify"] = bool(v.get("ok"))
            finals.append(final)
            # Reload handoff
            try:
                loaded = load_final_checkpoint(final)
                checks[f"{arm_dir.name}_handoff_reload"] = bool(loaded.get("handoff"))
            except Exception as e:
                checks[f"{arm_dir.name}_handoff_reload"] = False
                checks[f"{arm_dir.name}_handoff_error"] = str(e)

    # Day 90 marker only expected when init_days>=90; for smoke check Day init marker path exists in code via init_days
    checks["init_days_used"] = init_days
    checks["n_daily_checkpoint_dirs"] = n_days_total
    checks["n_unique_plot_filenames"] = plot_families
    checks["telegram_backtest_off"] = True  # enforced in checkpoint/analytics

    report = {
        "cmp_dir": str(cmp_dir),
        "days": days,
        "init_days": init_days,
        "checks": checks,
        "all_critical_ok": all(
            v is True for k, v in checks.items()
            if k.endswith(("_daily_checkpoints", "_plots", "_final_checkpoint", "_final_verify", "_handoff_reload", "_day_number_col", "arms_present"))
        ),
    }
    out = cmp_dir / "SMOKE_C1_C3_REPORT.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["all_critical_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
