"""Signed directional scores and combined S ∈ [-1, +1].

Existing factor scores are continuous in [0, 1] (0.5 ≈ neutral).
We map without hard-thresholding:

    s_i = 2 * score_i - 1     # 0→-1, 0.5→0, 1→+1
    S   = Σ w_i * s_i         # weights normalized to sum 1 → S ∈ [-1,+1]

Late Entry is intentionally excluded from S.
"""

from __future__ import annotations

from typing import Any

import numpy as np

FACTOR_KEYS = (
    "momentum",
    "trend",
    "btc_regime",
    "volume",
    "volatility",
    "rsi",
    "structure",
)


def signed_from_score(score: float | None) -> float | None:
    if score is None or not np.isfinite(score):
        return None
    return float(np.clip(2.0 * float(score) - 1.0, -1.0, 1.0))


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    w = {k: max(0.0, float(weights.get(k, 0.0))) for k in FACTOR_KEYS}
    s = sum(w.values())
    if s <= 0 or not np.isfinite(s):
        n = len(FACTOR_KEYS)
        return {k: 1.0 / n for k in FACTOR_KEYS}
    return {k: v / s for k, v in w.items()}


def equal_factor_weights() -> dict[str, float]:
    """Arm B: w_i = 1/N for all N=len(FACTOR_KEYS) eligible indicator groups.

    Missing/invalid factor scores still receive weight 1/N but contribute
    signed 0 in combined_score (neutral). N is the model factor count, not
    a per-bar subset — keeps A/B/C indicator sets identical.
    """
    n = len(FACTOR_KEYS)
    return {k: 1.0 / n for k in FACTOR_KEYS}


def static_factor_weights(signal_cfg: dict[str, Any]) -> dict[str, float]:
    """Arm A: fixed weights from signal_config factors.weights (pre-Adaptive V2)."""
    raw = dict((signal_cfg.get("factors") or {}).get("weights") or {})
    return normalize_weights(raw)


def combined_score(
    factor_scores: dict[str, float],
    weights: dict[str, float],
) -> dict[str, Any]:
    """Build signed indicator predictions and weighted S.

    ``factor_scores`` keys are FACTOR_KEYS with values in [0,1].
    Missing/invalid factors contribute 0 (neutral) and are flagged.
    """
    w = normalize_weights(weights)
    signed: dict[str, float] = {}
    contributions: dict[str, float] = {}
    warnings: list[str] = []
    total = 0.0
    for k in FACTOR_KEYS:
        s = signed_from_score(factor_scores.get(k))
        if s is None:
            s = 0.0
            warnings.append(f"{k}_SIGNED_MISSING")
        signed[k] = s
        contrib = w[k] * s
        contributions[k] = contrib
        total += contrib
    S = float(np.clip(total, -1.0, 1.0))
    return {
        "S": S,
        "signed": signed,
        "weights": w,
        "contributions": contributions,
        "warnings": warnings,
    }


def extract_factor_scores(factors: dict[str, Any]) -> dict[str, float]:
    """Pull group scores from compute_all_factors() output."""
    out: dict[str, float] = {}
    for k in FACTOR_KEYS:
        block = factors.get(k) or {}
        sc = block.get("score") if isinstance(block, dict) else block
        try:
            out[k] = float(sc)
        except (TypeError, ValueError):
            out[k] = float("nan")
    return out
