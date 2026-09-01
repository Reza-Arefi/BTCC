"""Signed directional scores and combined S ∈ [-1, +1].

Existing factor scores are continuous in [0, 1] (0.5 ≈ neutral).
We map without hard-thresholding:

    s_i = 2 * score_i - 1     # 0→-1, 0.5→0, 1→+1
    S   = Σ w_i * s_i         # weights normalized to sum 1 → S ∈ [-1,+1]

Architecture (isolation experiment):
  ACTIVE_SIGNAL_KEYS drive S and Adaptive weight updates.
  CONTEXT_FACTOR_KEYS (btc_regime / BTC.D-linked group) are still computed and
  logged with weight forced to 0 — they do NOT contribute to S.

Late Entry is intentionally excluded from S.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# All factor groups still computed / stored for diagnostics.
ALL_FACTOR_KEYS = (
    "momentum",
    "trend",
    "btc_regime",
    "volume",
    "volatility",
    "rsi",
    "structure",
)

# Trading score + Adaptive optimization set (BTC.D / btc_regime excluded).
ACTIVE_SIGNAL_KEYS = (
    "momentum",
    "trend",
    "volume",
    "volatility",
    "rsi",
    "structure",
)

CONTEXT_FACTOR_KEYS = ("btc_regime",)

# Back-compat alias — most call sites mean "all logged factors"
FACTOR_KEYS = ALL_FACTOR_KEYS


def signed_from_score(score: float | None) -> float | None:
    if score is None or not np.isfinite(score):
        return None
    return float(np.clip(2.0 * float(score) - 1.0, -1.0, 1.0))


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """Normalize over ACTIVE_SIGNAL_KEYS only; context factors forced to 0."""
    w = {k: 0.0 for k in ALL_FACTOR_KEYS}
    for k in ACTIVE_SIGNAL_KEYS:
        w[k] = max(0.0, float(weights.get(k, 0.0)))
    s = sum(w[k] for k in ACTIVE_SIGNAL_KEYS)
    if s <= 0 or not np.isfinite(s):
        n = len(ACTIVE_SIGNAL_KEYS)
        for k in ACTIVE_SIGNAL_KEYS:
            w[k] = 1.0 / n
        return w
    for k in ACTIVE_SIGNAL_KEYS:
        w[k] = w[k] / s
    # Explicit isolation: never let context factors carry score mass
    for k in CONTEXT_FACTOR_KEYS:
        w[k] = 0.0
    return w


def equal_factor_weights() -> dict[str, float]:
    """Arm B: w_i = 1/N over ACTIVE_SIGNAL_KEYS only; btc_regime = 0."""
    n = len(ACTIVE_SIGNAL_KEYS)
    w = {k: 0.0 for k in ALL_FACTOR_KEYS}
    for k in ACTIVE_SIGNAL_KEYS:
        w[k] = 1.0 / n
    return w


def static_factor_weights(signal_cfg: dict[str, Any]) -> dict[str, float]:
    """Arm A: fixed weights from signal_config; btc_regime forced to 0 then renormalized."""
    raw = dict((signal_cfg.get("factors") or {}).get("weights") or {})
    raw["btc_regime"] = 0.0
    return normalize_weights(raw)


def combined_score(
    factor_scores: dict[str, float],
    weights: dict[str, float],
) -> dict[str, Any]:
    """Build signed indicator predictions and weighted S.

    ``factor_scores`` may include context factors (logged); only ACTIVE_SIGNAL_KEYS
    contribute to S. Missing/invalid active factors contribute 0 (neutral).
    """
    w = normalize_weights(weights)
    signed: dict[str, float] = {}
    contributions: dict[str, float] = {}
    warnings: list[str] = []
    total = 0.0
    for k in ALL_FACTOR_KEYS:
        s = signed_from_score(factor_scores.get(k))
        if s is None:
            s = 0.0
            warnings.append(f"{k}_SIGNED_MISSING")
        signed[k] = s
        contrib = w[k] * s
        contributions[k] = contrib
        if k in ACTIVE_SIGNAL_KEYS:
            total += contrib
    S = float(np.clip(total, -1.0, 1.0))
    return {
        "S": S,
        "signed": signed,
        "weights": w,
        "contributions": contributions,
        "warnings": warnings,
        "active_signal_keys": list(ACTIVE_SIGNAL_KEYS),
        "context_factor_keys": list(CONTEXT_FACTOR_KEYS),
    }


def extract_factor_scores(factors: dict[str, Any]) -> dict[str, float]:
    """Pull group scores from compute_all_factors() output."""
    out: dict[str, float] = {}
    for k in ALL_FACTOR_KEYS:
        block = factors.get(k) or {}
        sc = block.get("score") if isinstance(block, dict) else block
        try:
            out[k] = float(sc)
        except (TypeError, ValueError):
            out[k] = float("nan")
    return out
