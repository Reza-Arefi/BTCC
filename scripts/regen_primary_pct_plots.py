#!/usr/bin/env python3
"""Regenerate primary percentage plots from existing arm CSVs (plot-only).

Does NOT modify trading state or restart btcc-walk-forward-1y.
Snapshots CSVs before reading; writes/replaces PNG plots only.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _arm_name(arm_dir: Path) -> str:
    name = arm_dir.name
    if "_static_" in name:
        return "static"
    if "_equal_" in name:
        return "equal"
    return "adaptive"


def _snapshot_arm(arm_dir: Path, tmp: Path) -> Path:
    snap = tmp / "arm_snap"
    snap.mkdir(parents=True, exist_ok=True)
    for name in (
        "predictions.csv",
        "opportunities.csv",
        "strategy_legs.csv",
        "weight_history.csv",
        "weight_updates.json",
        "weight_schedule.json",
        "summary.json",
        "fingerprint.json",
    ):
        src = arm_dir / name
        if src.exists() and src.is_file():
            shutil.copy2(src, snap / name)
    return snap


def _sync_pngs(src_plots: Path, dst_plots: Path) -> list[str]:
    copied: list[str] = []
    if not src_plots.exists():
        return copied
    dst_plots.mkdir(parents=True, exist_ok=True)
    for png in src_plots.rglob("*.png"):
        rel = png.relative_to(src_plots)
        dst = dst_plots / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        shutil.copy2(png, tmp)
        tmp.replace(dst)
        copied.append(str(dst))
    return copied


def regen_arm(arm_dir: Path, *, init_days: int = 90, starting_capital_usd: float = 1000.0) -> dict:
    import pandas as pd

    from btcc.analytics.pipeline import build_arm_analytics_asof

    arm_dir = Path(arm_dir)
    arm = _arm_name(arm_dir)
    with tempfile.TemporaryDirectory(prefix="btcc_pct_plots_") as td:
        td_path = Path(td)
        snap = _snapshot_arm(arm_dir, td_path)
        # Determine as-of day from snapshot legs
        legs_p = snap / "strategy_legs.csv"
        max_day = 1
        if legs_p.exists():
            legs = pd.read_csv(legs_p, usecols=lambda c: c in {"day_number", "strategy_key"})
            if "day_number" in legs.columns and legs["day_number"].notna().any():
                max_day = int(pd.to_numeric(legs["day_number"], errors="coerce").max())
            strategies = (
                sorted(legs["strategy_key"].astype(str).unique())
                if "strategy_key" in legs.columns else []
            )
        else:
            strategies = []

        stage_analytics = td_path / "analytics"
        build_arm_analytics_asof(
            snap,
            arm=arm,
            day_number=max_day,
            init_days=init_days,
            analytics_root=stage_analytics,
            telegram_enabled=False,
            starting_capital_usd=starting_capital_usd,
        )
        # emit writes organized subdirs under analytics/plots/{primary,...}
        src = stage_analytics / "plots"
        live = arm_dir / "analytics" / "plots"
        copied = _sync_pngs(src, live)

        # Refresh latest day plot snapshot PNGs only (if day folder has plots/)
        dc = arm_dir / "daily_checkpoints"
        if dc.exists():
            days = sorted(dc.glob("day_*"))
            if days:
                day_plots = days[-1] / "plots"
                if day_plots.exists() or True:
                    _sync_pngs(src, day_plots)

        return {
            "arm_dir": str(arm_dir),
            "arm": arm,
            "ok": bool(copied),
            "max_day": max_day,
            "n_png": len(copied),
            "strategies": strategies,
            "sample": copied[:8],
        }


def watch(cmp_dir: Path, *, poll_sec: float = 45.0) -> None:
    seen: dict[str, int] = {}
    print(f"PLOT_REGEN_WATCH cmp={cmp_dir} poll={poll_sec}s", flush=True)
    while True:
        try:
            for arm_dir in sorted(Path(cmp_dir).glob("adaptive_v2_*")):
                dc = arm_dir / "daily_checkpoints"
                n = len(list(dc.glob("day_*"))) if dc.exists() else 0
                key = str(arm_dir)
                if n > 0 and seen.get(key) != n:
                    time.sleep(4.0)  # let checkpoint writer finish
                    info = regen_arm(arm_dir)
                    seen[key] = n
                    print(f"REGEN {json.dumps(info, default=str)}", flush=True)
        except Exception as e:
            print(f"REGEN_ERROR {e}", flush=True)
        time.sleep(poll_sec)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmp-dir", type=Path, required=True)
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--poll-sec", type=float, default=45.0)
    args = ap.parse_args()
    if args.watch:
        watch(args.cmp_dir, poll_sec=args.poll_sec)
        return 0
    results = [regen_arm(d) for d in sorted(args.cmp_dir.glob("adaptive_v2_*"))]
    print(json.dumps(results, indent=2, default=str))
    return 0 if results and all(r.get("ok") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
