"""5-minute S-threshold sweep: 8 strict-`>` thresholds × T1–T10 + A–F.

Does not implement 5m→15m confirmation. Parent output dir aggregates per-threshold runs.
"""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from btcc.analytics.selector_5m_threshold_analytics import write_threshold_sweep_analytics
from btcc.sim.selector_backtest import run_selector_experiment_backtest
from btcc.sim.selector_config import (
    DEFAULT_5M_THRESHOLDS,
    SELECTOR_ARM_LABELS,
    _thr_tag,
    active_fixed_arm_labels,
    apply_threshold_override,
    load_selector_experiment_5m_thr_sweep_config,
    pre_run_config_summary,
    validate_selector_experiment,
)

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={ROOT}", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
    except Exception:
        return "UNKNOWN"


def run_5m_threshold_sweep(
    *,
    days: int | None = None,
    thresholds: list[float] | None = None,
    force_download: bool = False,
    out_root: Path | None = None,
    parent_dir: Path | None = None,
    resume: bool = True,
    enable_period_analysis: bool | None = None,
) -> Path:
    """Run one selector experiment per threshold under a parent sweep directory."""
    base = load_selector_experiment_5m_thr_sweep_config()
    se = base.get("selector_experiment") or {}
    thr_list = [float(x) for x in (thresholds or base.get("thresholds") or DEFAULT_5M_THRESHOLDS)]
    run_days = int(days if days is not None else se.get("total_days") or base.get("total_days") or 365)
    if enable_period_analysis is not None:
        base["enable_period_analysis"] = bool(enable_period_analysis)

    out_root = Path(out_root) if out_root else ROOT / "results"
    if parent_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        parent_dir = out_root / f"selector_5m_thr_sweep_{stamp}"
    parent_dir = Path(parent_dir)
    parent_dir.mkdir(parents=True, exist_ok=True)

    launch_meta = {
        "experiment": "selector_5m_threshold_sweep",
        "experiment_kind": base.get("experiment_kind"),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "days": run_days,
        "candle_interval": "5m",
        "threshold_operator": ">",
        "threshold_strict": True,
        "thresholds": thr_list,
        "arms": {
            "fixed": list(active_fixed_arm_labels(base)),
            "selectors": list(SELECTOR_ARM_LABELS),
            "excluded": ["T11", "T12"],
        },
        "no_15m_confirmation": True,
        "btc_d": "OFF",
        "note": (
            "Baseline 15m experiments use S >= thr. "
            "This 5m sweep uses strict S > thr as requested."
        ),
    }
    (parent_dir / "SWEEP_LAUNCH.json").write_text(json.dumps(launch_meta, indent=2), encoding="utf-8")
    (parent_dir / "PRE_RUN_CONFIG.txt").write_text(pre_run_config_summary(base) + "\n", encoding="utf-8")

    run_dirs: dict[str, str] = {}
    for thr in thr_list:
        sim = apply_threshold_override(base, thr)
        errs = validate_selector_experiment(sim)
        if errs:
            raise ValueError(f"Validation failed for thr={thr}: {errs}")
        tag = _thr_tag(thr)
        child = parent_dir / tag
        logger.info("=== 5m threshold sweep: S > %.2f → %s ===", thr, child)
        out = run_selector_experiment_backtest(
            days=run_days,
            force_download=force_download,
            out_dir=child,
            sim_cfg=sim,
            resume=resume,
        )
        run_dirs[tag] = str(out)

    (parent_dir / "run_dirs.json").write_text(json.dumps(run_dirs, indent=2), encoding="utf-8")
    write_threshold_sweep_analytics(parent_dir)
    launch_meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
    launch_meta["run_dirs"] = run_dirs
    (parent_dir / "SWEEP_LAUNCH.json").write_text(json.dumps(launch_meta, indent=2), encoding="utf-8")
    logger.info("5m threshold sweep complete → %s", parent_dir)
    return parent_dir
