"""Entry policies for Adaptive V2 — orthogonal to exit strategies S1/S2/S3.

Policies:
  NORMAL_FILTERED   — apply Late Entry / extended-filter rejection
  LATE_ENTRY_ALLOWED — same fundamentals, but do NOT reject solely for late/extended

Neither policy bypasses: S threshold, data/BTC.D health, max-10, one-per-pair,
closed-bar timing, fees/slippage, long-only, or other hard safety gates.
"""

from __future__ import annotations

from typing import Any

POLICY_NORMAL = "NORMAL_FILTERED"
POLICY_LATE_ALLOWED = "LATE_ENTRY_ALLOWED"
ENTRY_POLICIES = (POLICY_NORMAL, POLICY_LATE_ALLOWED)

# Classifications recorded on every candidate signal
CLS_NORMAL_ENTRY = "NORMAL_ENTRY"
CLS_LATE_ENTRY_ACCEPTED = "LATE_ENTRY_ACCEPTED"
CLS_LATE_ENTRY_REJECTED = "LATE_ENTRY_REJECTED"
CLS_BELOW_THRESHOLD = "BELOW_THRESHOLD"
CLS_SAME_PAIR_OPEN = "SAME_PAIR_ALREADY_OPEN"
CLS_MAX_OPEN = "MAX_OPEN_TRADES"
CLS_BTC_D = "BTC_D_UNAVAILABLE"
CLS_DATA_INVALID = "DATA_INVALID"
CLS_SIGNAL_CONTINUATION = "SIGNAL_CONTINUATION"
CLS_NO_NEXT_BAR = "NO_NEXT_BAR"
CLS_HEALTH = "DATA_HEALTH_BLOCK"


def is_late_extended(
    *,
    late_entry_score: float | None,
    late_entry_class: str | None,
    alert_threshold: float = 0.75,
) -> bool:
    """True if Late Entry / extended filter would fire."""
    if late_entry_class in ("HIGH_LATE_ENTRY_RISK", "VERY_HIGH_LATE_ENTRY_RISK"):
        return True
    if late_entry_score is not None and float(late_entry_score) >= float(alert_threshold):
        return True
    return False


def evaluate_entry_policy(
    *,
    policy: str,
    sm_decision: dict[str, Any],
    late_entry_score: float | None,
    late_entry_class: str | None,
    alert_threshold: float = 0.75,
    health_allow_new_trades: bool = True,
    health_btc_d_available: bool = True,
) -> dict[str, Any]:
    """Decide whether to open under a given entry policy.

    Returns keys:
      trade_suggested, entry_classification, late_extended,
      normal_would_reject_for_late, recovered_by_late_allowed,
      rejection_reason
    """
    if policy not in ENTRY_POLICIES:
        raise ValueError(f"Unknown entry policy: {policy}")

    late = is_late_extended(
        late_entry_score=late_entry_score,
        late_entry_class=late_entry_class,
        alert_threshold=alert_threshold,
    )

    # Map SM rejection first (threshold / continuation / capacity / pair)
    sm_rej = sm_decision.get("rejection_reason")
    signal_ok = bool(sm_decision.get("signal_generated"))
    trade_sm = bool(sm_decision.get("trade_suggested"))

    if not health_allow_new_trades:
        reason = CLS_BTC_D if not health_btc_d_available else CLS_HEALTH
        return {
            "trade_suggested": False,
            "entry_classification": reason,
            "late_extended": late,
            "normal_would_reject_for_late": late and trade_sm,
            "recovered_by_late_allowed": False,
            "rejection_reason": reason,
        }

    if sm_rej == "BELOW_THRESHOLD" or (not signal_ok and sm_rej == "BELOW_THRESHOLD"):
        return {
            "trade_suggested": False,
            "entry_classification": CLS_BELOW_THRESHOLD,
            "late_extended": late,
            "normal_would_reject_for_late": False,
            "recovered_by_late_allowed": False,
            "rejection_reason": CLS_BELOW_THRESHOLD,
        }

    if sm_rej == "SIGNAL_CONTINUATION":
        return {
            "trade_suggested": False,
            "entry_classification": CLS_SIGNAL_CONTINUATION,
            "late_extended": late,
            "normal_would_reject_for_late": False,
            "recovered_by_late_allowed": False,
            "rejection_reason": CLS_SIGNAL_CONTINUATION,
        }

    if sm_rej == "SAME_PAIR_ALREADY_OPEN":
        return {
            "trade_suggested": False,
            "entry_classification": CLS_SAME_PAIR_OPEN,
            "late_extended": late,
            "normal_would_reject_for_late": False,
            "recovered_by_late_allowed": False,
            "rejection_reason": CLS_SAME_PAIR_OPEN,
        }

    if sm_rej == "MAX_OPEN_TRADES":
        return {
            "trade_suggested": False,
            "entry_classification": CLS_MAX_OPEN,
            "late_extended": late,
            "normal_would_reject_for_late": False,
            "recovered_by_late_allowed": False,
            "rejection_reason": CLS_MAX_OPEN,
        }

    if not trade_sm:
        return {
            "trade_suggested": False,
            "entry_classification": sm_rej or "NOT_OPENED",
            "late_extended": late,
            "normal_would_reject_for_late": False,
            "recovered_by_late_allowed": False,
            "rejection_reason": sm_rej or "NOT_OPENED",
        }

    # SM wants to open — apply Late Entry gate only for NORMAL_FILTERED
    normal_rejects_late = late
    if policy == POLICY_NORMAL and normal_rejects_late:
        return {
            "trade_suggested": False,
            "entry_classification": CLS_LATE_ENTRY_REJECTED,
            "late_extended": True,
            "normal_would_reject_for_late": True,
            "recovered_by_late_allowed": False,
            "rejection_reason": CLS_LATE_ENTRY_REJECTED,
        }

    if policy == POLICY_LATE_ALLOWED and normal_rejects_late:
        return {
            "trade_suggested": True,
            "entry_classification": CLS_LATE_ENTRY_ACCEPTED,
            "late_extended": True,
            "normal_would_reject_for_late": True,
            "recovered_by_late_allowed": True,
            "rejection_reason": None,
        }

    return {
        "trade_suggested": True,
        "entry_classification": CLS_NORMAL_ENTRY,
        "late_extended": late,
        "normal_would_reject_for_late": False,
        "recovered_by_late_allowed": False,
        "rejection_reason": None,
    }
