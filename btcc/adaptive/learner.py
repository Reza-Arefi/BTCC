"""Challenger trainer — factor-level weights with time decay + Champion shrinkage."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from btcc.adaptive.model import FACTOR_KEYS, SignalModel

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
    """Exponential decay: weight = 0.5 ** (age_days / half_life)."""
    ts = pd.to_datetime(timestamps, utc=True)
    latest = ts.max()
    age_days = (latest - ts).dt.total_seconds() / 86400.0
    hl = max(float(half_life_days), 1e-6)
    w = np.power(0.5, age_days.to_numpy(dtype=float) / hl)
    return w


def temporal_splits(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Strict temporal split — never shuffle. Train → Validation → Holdout."""
    d = df.sort_values("timestamp").reset_index(drop=True)
    n = len(d)
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))
    return d.iloc[:i_train], d.iloc[i_train:i_val], d.iloc[i_val:]


def estimate_factor_weights(
    train: pd.DataFrame,
    champion_weights: dict[str, float],
    half_life_days: float,
    adaptation_rate: float,
    weight_min: float,
    weight_max: float,
    horizon: int = 4,
) -> dict[str, float]:
    """Estimate factor weights from labeled train set.

    Method: weighted mean factor score among hits vs misses → discriminative score,
    then normalize and shrink toward Champion. Does not optimize all 13 indicators.
    """
    ycol = f"outperformed_{horizon}h"
    if train.empty or ycol not in train.columns:
        return dict(champion_weights)

    w_sample = time_decay_weights(train["timestamp"], half_life_days)
    y = train[ycol].astype(float).to_numpy()
    scores = {}
    for key in FACTOR_KEYS:
        col = FACTOR_COL[key]
        if col not in train.columns:
            scores[key] = 0.0
            continue
        x = train[col].astype(float).to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 20:
            scores[key] = float(champion_weights.get(key, 0.1))
            continue
        ww = w_sample[mask]
        xx = x[mask]
        yy = y[mask]
        # Weighted correlation with outcome (discriminative)
        wsum = ww.sum()
        mx = (ww * xx).sum() / wsum
        my = (ww * yy).sum() / wsum
        cov = (ww * (xx - mx) * (yy - my)).sum() / wsum
        vx = (ww * (xx - mx) ** 2).sum() / wsum
        vy = (ww * (yy - my) ** 2).sum() / wsum
        if vx <= 1e-12 or vy <= 1e-12:
            corr = 0.0
        else:
            corr = cov / np.sqrt(vx * vy)
        # Map correlation to positive weight contribution (floor at small epsilon)
        scores[key] = max(float(corr), 0.0) + 1e-3

    raw_sum = sum(scores.values())
    raw = {k: v / raw_sum for k, v in scores.items()}

    # Shrinkage toward Champion
    a = float(np.clip(adaptation_rate, 0.0, 1.0))
    champ = {k: float(champion_weights.get(k, 0.0)) for k in FACTOR_KEYS}
    champ_sum = sum(champ.values()) or 1.0
    champ = {k: v / champ_sum for k, v in champ.items()}

    mixed = {k: (1 - a) * champ[k] + a * raw[k] for k in FACTOR_KEYS}
    # Clip
    mixed = {k: float(np.clip(v, weight_min, weight_max)) for k, v in mixed.items()}
    s = sum(mixed.values())
    return {k: v / s for k, v in mixed.items()}


def build_challenger(
    labeled: pd.DataFrame,
    champion: SignalModel,
    cfg_adaptive: dict[str, Any],
) -> tuple[SignalModel | None, dict[str, Any]]:
    """Train Challenger on temporal train split; return model + split meta. No promotion here."""
    horizon = int(cfg_adaptive.get("primary_horizon", 4))
    n = len(labeled)
    min_cons = int(cfg_adaptive.get("min_samples_conservative", 500))
    if n < min_cons:
        return None, {"decision": "NO_UPDATE", "reason": f"n={n} < min_samples_conservative={min_cons}"}

    train, val, holdout = temporal_splits(
        labeled,
        float(cfg_adaptive.get("train_fraction", 0.70)),
        float(cfg_adaptive.get("validation_fraction", 0.20)),
    )
    if len(train) < 100 or len(val) < 50:
        return None, {"decision": "NO_UPDATE", "reason": "insufficient train/val after split"}

    adapt_rate = float(cfg_adaptive.get("adaptation_rate", 0.25))
    # Conservative mode if below normal threshold
    if n < int(cfg_adaptive.get("min_samples_normal", 2000)):
        adapt_rate *= 0.5

    weights = estimate_factor_weights(
        train,
        champion.normalized_weights(),
        half_life_days=float(cfg_adaptive.get("time_decay_half_life_days", 45)),
        adaptation_rate=adapt_rate,
        weight_min=float(cfg_adaptive.get("weight_min", 0.05)),
        weight_max=float(cfg_adaptive.get("weight_max", 0.40)),
        horizon=horizon,
    )

    from datetime import datetime, timezone
    ver = champion.version  # overwritten by caller with next version
    challenger = SignalModel(
        version=ver,
        model_type="challenger",
        factor_weights=weights,
        probability_kind=champion.probability_kind,
        calibration=champion.calibration,
        training_period={
            "start": str(train["timestamp"].iloc[0]),
            "end": str(train["timestamp"].iloc[-1]),
            "n": len(train),
        },
        validation_period={
            "start": str(val["timestamp"].iloc[0]),
            "end": str(val["timestamp"].iloc[-1]),
            "n": len(val),
        },
        sample_count=n,
        metrics={
            "holdout_n": len(holdout),
            "adaptation_rate_used": adapt_rate,
        },
        created_utc=datetime.now(timezone.utc).isoformat(),
        notes="Factor-level Challenger (not sub-indicator). Regularized toward Champion.",
        source_tags=list(set(labeled.get("data_source", pd.Series(dtype=str)).dropna().unique().tolist())),
    )
    meta = {
        "train_n": len(train),
        "val_n": len(val),
        "holdout_n": len(holdout),
        "train": train,
        "val": val,
        "holdout": holdout,
    }
    return challenger, meta
