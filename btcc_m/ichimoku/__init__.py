"""BTCC_M Ichimoku research package (primary signal engine)."""

from btcc_m.ichimoku.indicators import attach_forward_returns, compute_ichimoku
from btcc_m.ichimoku.signals import PHASE1_SIGNALS

__all__ = ["compute_ichimoku", "attach_forward_returns", "PHASE1_SIGNALS"]

# cloud_extension is imported by experiments directly (Phase A+)
