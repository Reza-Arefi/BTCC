"""Rolling 90-day weight estimation with safeguards (always-update V2).

Does NOT use Champion/Challenger promotion gates.

Safeguards:
- require min observations total and per factor
- reject NaN/invalid estimates
- clip to [weight_min, weight_max] and renormalize
- blend toward previous known-good weights (stability_blend)
- never overwrite last known-good on failure
- append full weight history on every attempt
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from btcc.sim.score import FACTOR_KEYS, normalize_weights

logger = logging.getLogger(__name__)

FACTOR_COL = {
    "momentum": "factor_momentum",
    "trend": "factor_trend",
    "btc_regime": "factor_btc_regime",
    "volume": "factor_volume",
    "volatility": "factor_volatility",
    "rsi": "factor_rsi",
    "structure": "factor_structure",
}


def time_decay_weights(timestamps: pd.Series, half_life_days: float) -> np.ndarray:
    ts = pd.to_datetime(timestamps, utc=True)
    latest = ts.max()
    age_days = (latest - ts).dt.total_seconds() / 86400.0
    hl = max(float(half_life_days), 1e-6)
    return np.power(0.5, age_days.to_numpy(dtype=float) / hl)


def filter_rolling_window(df: pd.DataFrame, end_ts: pd.Timestamp, days: int) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    end_ts = pd.Timestamp(end_ts)
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    start = end_ts - pd.Timedelta(days=int(days))
    return d[(d["timestamp"] >= start) & (d["timestamp"] <= end_ts)].copy()


def estimate_weights_from_window(
    window: pd.DataFrame,
    prev_weights: dict[str, float],
    *,
    half_life_days: float,
    weight_min: float,
    weight_max: float,
    min_obs_total: int,
    min_obs_per_factor: int,
    stability_blend: float,
    horizon: int = 4,
) -> tuple[dict[str, float] | None, dict[str, Any]]:
    """Estimate new weights from rolling window. Returns (weights|None, stats)."""
    ycol = f"future_return_{horizon}h"
    # Prefer continuous ALT/BTC return; fall back to binary outperform if needed
    if window.empty:
        return None, {"status": "FAILED", "reason": "empty_window"}

    if ycol not in window.columns or window[ycol].notna().sum() == 0:
        alt = f"outperformed_{horizon}h"
        if alt in window.columns and window[alt].notna().sum() > 0:
            ycol = alt
        else:
            return None, {"status": "FAILED", "reason": f"no_outcome_column_{horizon}h"}

    labeled = window.dropna(subset=[ycol]).copy()
    n = len(labeled)
    if n < int(min_obs_total):
        return None, {
            "status": "FAILED",
            "reason": f"n={n}<min_observations_total={min_obs_total}",
            "n_samples": n,
        }

    w_sample = time_decay_weights(labeled["timestamp"], half_life_days)
    y = labeled[ycol].astype(float).to_numpy()
    ics: dict[str, float] = {}
    ns: dict[str, int] = {}
    raw_scores: dict[str, float] = {}

    for key in FACTOR_KEYS:
        col = FACTOR_COL[key]
        if col not in labeled.columns:
            ics[key] = float("nan")
            ns[key] = 0
            raw_scores[key] = float("nan")
            continue
        # Use signed factor: 2*score-1 correlated with future ALT/BTC return
        x = (2.0 * labeled[col].astype(float).to_numpy() - 1.0)
        mask = np.isfinite(x) & np.isfinite(y)
        ns[key] = int(mask.sum())
        if ns[key] < int(min_obs_per_factor):
            ics[key] = float("nan")
            raw_scores[key] = float("nan")
            continue
        ww = w_sample[mask]
        xx = x[mask]
        yy = y[mask]
        wsum = ww.sum()
        mx = (ww * xx).sum() / wsum
        my = (ww * yy).sum() / wsum
        cov = (ww * (xx - mx) * (yy - my)).sum() / wsum
        vx = (ww * (xx - mx) ** 2).sum() / wsum
        vy = (ww * (yy - my) ** 2).sum() / wsum
        if vx <= 1e-12 or vy <= 1e-12:
            corr = 0.0
        else:
            corr = float(cov / np.sqrt(vx * vy))
        ics[key] = corr
        # Map IC to positive weight mass (negative IC → tiny floor)
        raw_scores[key] = max(corr, 0.0) + 1e-3

    if any(not np.isfinite(v) for v in raw_scores.values()):
        # Replace invalid with previous weight mass so update can still proceed partially
        prev = normalize_weights(prev_weights)
        for k, v in list(raw_scores.items()):
            if not np.isfinite(v):
                raw_scores[k] = prev.get(k, 1e-3)

    if sum(raw_scores.values()) <= 0 or not np.isfinite(sum(raw_scores.values())):
        return None, {
            "status": "FAILED",
            "reason": "invalid_raw_scores",
            "n_samples": n,
            "ics": ics,
            "ns": ns,
        }

    raw_sum = sum(raw_scores.values())
    raw = {k: float(v / raw_sum) for k, v in raw_scores.items()}
    prev = normalize_weights(prev_weights)
    b = float(np.clip(stability_blend, 0.0, 1.0))
    mixed = {k: (1.0 - b) * prev[k] + b * raw[k] for k in FACTOR_KEYS}
    mixed = {k: float(np.clip(v, weight_min, weight_max)) for k, v in mixed.items()}
    if any(not np.isfinite(v) for v in mixed.values()) or sum(mixed.values()) <= 0:
        return None, {
            "status": "FAILED",
            "reason": "nan_after_clip",
            "n_samples": n,
            "ics": ics,
            "ns": ns,
        }
    new_w = normalize_weights(mixed)
    stats = {
        "status": "OK",
        "n_samples": n,
        "ics": ics,
        "ns": ns,
        "raw_weights": raw,
        "ycol": ycol,
        "learning_window_start": str(labeled["timestamp"].min()),
        "learning_window_end": str(labeled["timestamp"].max()),
    }
    return new_w, stats
