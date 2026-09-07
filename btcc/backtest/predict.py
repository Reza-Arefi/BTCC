"""Single-coin prediction at historical bar — reuses live factor/probability logic."""

from __future__ import annotations

from typing import Any

import pandas as pd

from btcc.backtest.extract import flatten_indicators_and_factors
from btcc.factors.combine import compute_all_factors
from btcc.late_entry.score import late_entry_score
from btcc.probability.score import horizons_probabilities


def predict_coin_at_bar(
    rel_hist: pd.DataFrame,
    alt_vol_hist: pd.DataFrame,
    btc_hist: pd.DataFrame,
    dominance_pct: float | None,
    dom_changes: dict[int, float | None],
    cfg: dict[str, Any],
    interval: str,
    calib: dict | None = None,
) -> dict[str, Any] | None:
    """Compute signal for one coin using ONLY data in *_hist frames (no lookahead)."""
    if rel_hist is None or len(rel_hist) < 100:
        return None

    factors = compute_all_factors(
        rel_hist, alt_vol_hist, btc_hist, dominance_pct, dom_changes, cfg, interval
    )
    probs = horizons_probabilities(factors["signal_score"], cfg, calib)
    late = late_entry_score(rel_hist, factors, cfg, interval)
    flat = flatten_indicators_and_factors(factors)

    return {
        "signal_score": factors["signal_score"],
        "probability_kind": probs["probability_kind"],
        "probability_1h": probs["p_1h"],
        "probability_4h": probs["p_4h"],
        "probability_8h": probs["p_8h"],
        "probability_12h": probs["p_12h"],
        "probability_24h": probs["p_24h"],
        "p_1h_status": probs["p_1h_status"],
        "p_4h_status": probs["p_4h_status"],
        "p_8h_status": probs["p_8h_status"],
        "p_12h_status": probs["p_12h_status"],
        "p_24h_status": probs["p_24h_status"],
        "baseline_model_p_4h": probs.get("baseline_model_p_4h", probs["p_4h"]),
        "late_entry_score": late["late_entry_score"],
        "late_entry_class": late["classification"],
        "alt_btc_price": float(rel_hist["close"].iloc[-1]),
        "btc_price": float(btc_hist["close"].iloc[-1]),
        "btc_dominance": dominance_pct,
        "factors": factors,
        "late_entry": late,
        **flat,
    }
