"""Re-score stored factor rows under a candidate model's weights (no look-ahead)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from btcc.adaptive.learner import FACTOR_COL
from btcc.adaptive.model import FACTOR_KEYS, SignalModel
from btcc.probability.score import horizons_probabilities


def signal_score_from_row(row: pd.Series, weights: dict[str, float]) -> float:
    s = 0.0
    for k in FACTOR_KEYS:
        col = FACTOR_COL[k]
        v = row.get(col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            v = 0.5
        s += float(weights[k]) * float(v)
    return float(np.clip(s, 0.0, 1.0))


def signal_scores_vectorized(df: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    s = np.zeros(len(df), dtype=float)
    for k in FACTOR_KEYS:
        col = FACTOR_COL[k]
        if col in df.columns:
            x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            x = np.where(np.isfinite(x), x, 0.5)
        else:
            x = np.full(len(df), 0.5)
        s += float(weights[k]) * x
    return np.clip(s, 0.0, 1.0)


def rescore_frame(
    df: pd.DataFrame,
    model: SignalModel,
    signal_cfg: dict[str, Any],
) -> pd.DataFrame:
    """Return copy with probabilities recomputed from factor scores × model weights."""
    out = df.copy()
    w = model.normalized_weights()
    scores = signal_scores_vectorized(out, w)
    out["signal_score"] = scores
    calib = model.calibration
    # Vectorized-friendly: still call horizons_probabilities per unique score bucket
    # to preserve exact logistic/calibration semantics.
    p_rows = [horizons_probabilities(float(s), signal_cfg, calib) for s in scores]
    for h in signal_cfg["probability"]["horizons_hours"]:
        out[f"probability_{h}h"] = [r[f"p_{h}h"] for r in p_rows]
    out["probability_kind"] = [
        r.get("probability_kind", model.probability_kind) for r in p_rows
    ]
    return out
