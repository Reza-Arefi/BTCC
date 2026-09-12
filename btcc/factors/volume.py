"""Volume factor from ALTUSDT volume (RVOL + confirmation/divergence)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from btcc.factors.helpers import clamp01, last, sigmoid

# Live / research default internal mix (sums to 1.0).
DEFAULT_RVOL_WEIGHT = 0.65
DEFAULT_CONFIRMATION_WEIGHT = 0.35


def _normalize_volume_weights(
    rvol_weight: float | None = None,
    confirmation_weight: float | None = None,
    *,
    weights: dict[str, float] | None = None,
) -> tuple[float, float]:
    """Return (rvol_w, conf_w) renormalized to sum 1.0."""
    if weights:
        rw = float(weights.get("rvol", weights.get("RVOL", DEFAULT_RVOL_WEIGHT)))
        cw = float(
            weights.get(
                "confirmation",
                weights.get("Confirmation", weights.get("conf", DEFAULT_CONFIRMATION_WEIGHT)),
            )
        )
    else:
        rw = DEFAULT_RVOL_WEIGHT if rvol_weight is None else float(rvol_weight)
        cw = DEFAULT_CONFIRMATION_WEIGHT if confirmation_weight is None else float(confirmation_weight)
    rw = max(0.0, rw)
    cw = max(0.0, cw)
    s = rw + cw
    if s <= 0:
        return DEFAULT_RVOL_WEIGHT, DEFAULT_CONFIRMATION_WEIGHT
    return rw / s, cw / s


def volume_factor(
    alt_usdt: pd.DataFrame,
    alt_btc: pd.DataFrame,
    sma: int = 20,
    *,
    rvol_weight: float | None = None,
    confirmation_weight: float | None = None,
    internal_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """RVOL + confirmation/divergence mix.

    Optional ``rvol_weight`` / ``confirmation_weight`` (or ``internal_weights``)
    override the default 65/35 mix without changing overall S factor weights.
    """
    vol = alt_usdt["volume"]
    v_sma = vol.rolling(sma).mean()
    rvol = (last(vol) / last(v_sma)) if last(v_sma) else None

    # price change vs volume change (last bar)
    px = alt_btc["close"].pct_change()
    vv = vol.pct_change()
    px_up = (last(px) or 0) > 0
    vol_up = (last(vv) or 0) > 0
    confirmation = px_up and vol_up
    divergence = px_up and not vol_up and (last(vv) or 0) < 0

    rvol_score = sigmoid((rvol or 1.0) - 1.0, 0.0, 1.5) if rvol is not None else 0.5
    conf = 0.7 if confirmation else (0.35 if divergence else 0.5)
    rw, cw = _normalize_volume_weights(
        rvol_weight, confirmation_weight, weights=internal_weights
    )
    score = clamp01(rw * rvol_score + cw * conf)

    return {
        "score": score,
        "rvol_score": rvol_score if rvol is not None else None,
        "rvol": rvol,
        "confirmation_score": conf,
        "volume_confirmation": confirmation,
        "volume_divergence": divergence,
        "rvol_weight": rw,
        "confirmation_weight": cw,
    }
