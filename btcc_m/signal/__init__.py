"""BTCC_M signal package."""

from btcc_m.signal.factors import BASELINE_WEIGHTS, compute_btc_factors
from btcc_m.signal.score import generate_entries, frozen_baseline_weights

__all__ = [
    "BASELINE_WEIGHTS",
    "compute_btc_factors",
    "generate_entries",
    "frozen_baseline_weights",
]
