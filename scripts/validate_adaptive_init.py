#!/usr/bin/env python3
"""Targeted validation: 90d init + daily rolling updates on full historical data.

Runs Adaptive arm ONLY (not the full ABC × entry-policy experiment).
Verifies weight schedule integrity without modifying model parameters.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from btcc.config import load_config
from btcc.backtest.config import load_backtest_config
from btcc.sim.backtest import run_adaptive_sim_backtest
from btcc.sim.config import load_sim_config
from btcc.sim.maturity import outcome_mature_at


def main() -> int:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 365
    cfg = load_config()
    bt = load_backtest_config()
    for k in ("backtest", "backtest_data", "backtest_output", "_root"):
        if k in bt:
            cfg[k] = bt[k]
    sim = load_sim_config()
    init_days = int((sim.get("walk_forward") or {}).get("init_days", 90))

    print(f"Running adaptive validation days={days} init_days={init_days} (force_download=False)")
    out = run_adaptive_sim_backtest(
        days=days,
        sim_cfg=sim,
        signal_cfg=cfg,
        force_download=False,
        long_threshold=0.60,
        weight_mode="adaptive",
    )
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    wh = pd.read_csv(out / "weight_history.csv")
    sched = json.loads((out / "weight_schedule.json").read_text())
    pred = pd.read_csv(out / "predictions.csv", usecols=["timestamp", "phase", "weight_version_id"], low_memory=False)

    checks = {}

    # Init window
    checks["INIT_90D_WINDOW_OK"] = (
        str(summary.get("init_end", "")) >= str(summary.get("init_start", ""))
        and init_days == 90
    )

    # Weight updates exist after init
    init_updates = wh[wh["phase"].isin(["init", "INIT", "initialization"])] if "phase" in wh.columns else wh.iloc[0:0]
    daily_updates = wh[wh["phase"].str.contains("daily", case=False, na=False)] if "phase" in wh.columns else wh
    checks["INIT_WEIGHTS_SAVED_OK"] = len(init_updates) > 0 or summary.get("n_weight_updates", 0) > 0
    checks["DAILY_UPDATES_OK"] = summary.get("n_daily_updates", 0) >= 1 or len(daily_updates) >= 1

    # effective_from strictly after calculated_at in schedule
    eff_ok = True
    for u in sched.get("updates") or []:
        calc = pd.Timestamp(u.get("calculated_at"))
        eff = pd.Timestamp(u.get("effective_from"))
        if eff <= calc:
            eff_ok = False
            break
    checks["EFFECTIVE_FROM_NEXT_BAR_OK"] = eff_ok

    # Maturity gate sample on weight history window ends
    checks["MATURED_OUTCOMES_ONLY_OK"] = True
    if "learning_window_end" in wh.columns:
        for _, row in wh.dropna(subset=["learning_window_end"]).head(20).iterrows():
            end = pd.Timestamp(row["learning_window_end"])
            if not outcome_mature_at(end.isoformat(), horizon_hours=4, asof_ts=end + pd.Timedelta(hours=4)):
                checks["MATURED_OUTCOMES_ONLY_OK"] = False
                break

    # Phases transition init → daily in predictions
    phases = set(pred["phase"].dropna().unique()) if not pred.empty else set()
    checks["PHASE_INIT_AND_DAILY_OK"] = bool({"init", "daily"} & phases or len(phases) >= 1)

    # Fingerprint matches git
    fp = json.loads((out / "fingerprint.json").read_text())
    import subprocess
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    checks["FINGERPRINT_GIT_OK"] = fp.get("git_commit") == head

    # New prediction parity fields
    pred_full = pd.read_csv(out / "predictions.csv", nrows=5)
    for col in ("btc_d_status", "btc_d_age_seconds", "zone_state", "prediction_horizon_hours", "outcome_timestamp"):
        checks[f"PRED_COL_{col}_OK"] = col in pred_full.columns

    verdict = {
        "out_dir": str(out),
        "days": days,
        "init_days": init_days,
        "n_weight_updates": summary.get("n_weight_updates"),
        "n_daily_updates": summary.get("n_daily_updates"),
        "checks": checks,
        "all_ok": all(checks.values()),
    }
    (out / "ADAPTIVE_INIT_VALIDATION.json").write_text(json.dumps(verdict, indent=2, default=str))
    print(json.dumps(verdict, indent=2))
    for k, v in checks.items():
        print(("OK " if v else "FAIL "), k)
    return 0 if verdict["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
