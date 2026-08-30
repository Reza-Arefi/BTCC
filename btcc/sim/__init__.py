"""BTCC Adaptive V2 — virtual LONG ALT/BTC simulation + rolling weight adaptation.

Economic objective: predict ALT/BTC appreciation so that BTC→ALT→BTC
increases BTC quantity. SIGNAL / PAPER ONLY — no exchange orders.
"""

from __future__ import annotations

from btcc.sim.config import load_sim_config
from btcc.sim.score import FACTOR_KEYS, combined_score, signed_from_score

__all__ = [
    "load_sim_config",
    "FACTOR_KEYS",
    "signed_from_score",
    "combined_score",
]
