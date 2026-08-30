"""Shared analytics / visualization for Adaptive V2 backtest + live.

Metric definitions are identical in both modes. Plots are always backed by
saved CSV tables under metrics/ (never transient-only).
"""

from btcc.analytics.pipeline import (
    build_abc_analytics,
    build_arm_analytics,
    update_live_analytics,
)

__all__ = [
    "build_abc_analytics",
    "build_arm_analytics",
    "update_live_analytics",
]
