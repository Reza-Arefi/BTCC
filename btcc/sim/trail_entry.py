"""Frozen entry gate for trailing-exit experiment.

Eligible when: long_threshold <= S < upper_threshold
No exhaustion / late-entry rejection. BTC.D does not block.
"""

from __future__ import annotations

from typing import Any

CLS_ELIGIBLE = "ELIGIBLE"
CLS_BELOW_THRESHOLD = "BELOW_THRESHOLD"
CLS_ABOVE_THRESHOLD = "ABOVE_THRESHOLD"
CLS_SIGNAL_CONTINUATION = "SIGNAL_CONTINUATION"
CLS_SAME_PAIR_OPEN = "SAME_PAIR_ALREADY_OPEN"
CLS_MAX_OPEN = "MAX_OPEN_TRADES"
CLS_NO_NEXT_BAR = "NO_NEXT_BAR"
CLS_DATA_INVALID = "DATA_INVALID"
CLS_HEALTH = "DATA_HEALTH_BLOCK"


def evaluate_trail_entry(
    *,
    sm_decision: dict[str, Any],
    health_allow_new_trades: bool = True,
) -> dict[str, Any]:
    """Return trade decision for trail experiment (no late-entry gate)."""
    if not health_allow_new_trades:
        return {
            "trade_suggested": False,
            "entry_classification": CLS_HEALTH,
            "rejection_reason": CLS_HEALTH,
        }

    sm_rej = sm_decision.get("rejection_reason")
    if sm_rej == "BELOW_THRESHOLD":
        return {
            "trade_suggested": False,
            "entry_classification": CLS_BELOW_THRESHOLD,
            "rejection_reason": CLS_BELOW_THRESHOLD,
        }
    if sm_rej == "ABOVE_THRESHOLD":
        return {
            "trade_suggested": False,
            "entry_classification": CLS_ABOVE_THRESHOLD,
            "rejection_reason": CLS_ABOVE_THRESHOLD,
        }
    if sm_rej == "SIGNAL_CONTINUATION":
        return {
            "trade_suggested": False,
            "entry_classification": CLS_SIGNAL_CONTINUATION,
            "rejection_reason": CLS_SIGNAL_CONTINUATION,
        }
    if sm_rej == "SAME_PAIR_ALREADY_OPEN":
        return {
            "trade_suggested": False,
            "entry_classification": CLS_SAME_PAIR_OPEN,
            "rejection_reason": CLS_SAME_PAIR_OPEN,
        }
    if sm_rej == "MAX_OPEN_TRADES":
        return {
            "trade_suggested": False,
            "entry_classification": CLS_MAX_OPEN,
            "rejection_reason": CLS_MAX_OPEN,
        }
    if bool(sm_decision.get("trade_suggested")):
        return {
            "trade_suggested": True,
            "entry_classification": CLS_ELIGIBLE,
            "rejection_reason": None,
        }
    return {
        "trade_suggested": False,
        "entry_classification": sm_rej or "NOT_OPENED",
        "rejection_reason": sm_rej or "NOT_OPENED",
    }
