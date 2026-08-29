"""Walk-forward probability calibration utilities (run offline / periodically)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_calibration_curve(df: pd.DataFrame, horizon: int, n_bins: int = 10) -> list[list[float]]:
    """Use only rows with realized outcomes; no future leakage if called on past-only frame."""
    pcol = f"probability_{horizon}h"
    ycol = f"future_rel_return_{horizon}h"
    if pcol not in df.columns or ycol not in df.columns:
        return []
    sub = df.dropna(subset=[pcol, ycol]).copy()
    if len(sub) < 50:
        return []
    sub["hit"] = (sub[ycol] > 0).astype(float)
    sub["bin"] = pd.qcut(sub[pcol], q=min(n_bins, len(sub) // 10 or 1), duplicates="drop")
    curve = []
    for _, g in sub.groupby("bin", observed=True):
        curve.append([float(g[pcol].mean()), float(g["hit"].mean())])
    curve.sort(key=lambda x: x[0])
    return curve


def fit_and_save(predictions_csv: Path, out_json: Path, horizons: list[int]) -> dict:
    df = pd.read_csv(predictions_csv)
    calib = {}
    for h in horizons:
        calib[str(h)] = build_calibration_curve(df, h)
    out_json.write_text(json.dumps(calib, indent=2), encoding="utf-8")
    logger.info("Calibration saved → %s", out_json)
    return calib
