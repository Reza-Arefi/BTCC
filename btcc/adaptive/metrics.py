"""Evaluation metrics for Champion vs Challenger (primary: 4h)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def brier_score(y_true: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y_true) ** 2))


def calibration_error(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE)."""
    if len(p) < n_bins:
        n_bins = max(2, len(p) // 5 or 2)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(p)
    for i in range(n_bins):
        mask = (p >= bins[i]) & (p < bins[i + 1] if i < n_bins - 1 else p <= bins[i + 1])
        if not np.any(mask):
            continue
        ece += (mask.sum() / n) * abs(y_true[mask].mean() - p[mask].mean())
    return float(ece)


def ranking_quality(df: pd.DataFrame, horizon: int = 4) -> float:
    """Average future ALT/BTC return of Top-5 by predicted probability (higher better)."""
    pcol = f"probability_{horizon}h"
    rcol = f"future_return_{horizon}h"
    if pcol not in df.columns or rcol not in df.columns or "timestamp" not in df.columns:
        return float("nan")
    rows = []
    for _, g in df.groupby("timestamp"):
        top = g.nlargest(min(5, len(g)), pcol)
        if top[rcol].notna().any():
            rows.append(float(top[rcol].mean()))
    return float(np.mean(rows)) if rows else float("nan")


def success_rate(y_true: np.ndarray) -> float:
    return float(np.mean(y_true)) if len(y_true) else float("nan")


def evaluate_horizon(df: pd.DataFrame, horizon: int = 4) -> dict[str, Any]:
    pcol = f"probability_{horizon}h"
    ycol = f"outperformed_{horizon}h"
    rcol = f"future_return_{horizon}h"
    sub = df.dropna(subset=[pcol, ycol]).copy()
    if sub.empty:
        return {"horizon": horizon, "n": 0}
    y = sub[ycol].astype(float).to_numpy()
    p = sub[pcol].astype(float).to_numpy()
    out = {
        "horizon": horizon,
        "n": int(len(sub)),
        "brier": brier_score(y, p),
        "calibration_error": calibration_error(y, p),
        "success_rate": success_rate(y),
        "mean_predicted_p": float(p.mean()),
        "mean_future_return": float(sub[rcol].dropna().mean()) if rcol in sub.columns else None,
        "ranking_quality_top5": ranking_quality(sub, horizon),
    }
    return out


def evaluate_model_on_frame(df: pd.DataFrame, horizons: list[int] | None = None) -> dict[str, Any]:
    horizons = horizons or [1, 4, 8, 12, 24]
    out = {"by_horizon": {}}
    for h in horizons:
        out["by_horizon"][str(h)] = evaluate_horizon(df, h)
    # Primary summary = 4h
    out["primary"] = out["by_horizon"].get("4", {})
    return out
