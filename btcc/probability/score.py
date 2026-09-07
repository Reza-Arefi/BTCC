"""Signal score → baseline model probability (NOT a calibrated empirical probability).

Pipeline (required):
  Factor scores → Signal Score → Historical calibration → Probability

Until enough labeled outcomes exist, the logistic map produces only a
**baseline_model_probability**. It must NEVER be presented as a validated
frequency estimate of outperformance.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

PROBABILITY_STATUS_BASELINE = "baseline_model_probability"
PROBABILITY_STATUS_CALIBRATED = "calibrated_probability"


def score_to_baseline_model_probability(signal_score: float, cfg: dict[str, Any], horizon_h: int) -> float:
    """Deterministic logistic map of signal_score → baseline_model_probability.

    This is NOT empirical P(outperform). It is a monotonic transform used until
    walk-forward calibration has enough outcomes.
    """
    center = float(cfg["probability"]["logistic_center"])
    scale = float(cfg["probability"]["logistic_scale"])
    damp = {1: 1.0, 4: 0.95, 8: 0.90, 12: 0.85, 24: 0.80}.get(horizon_h, 0.9)
    x = (signal_score - center) * scale * damp
    x = max(-60.0, min(60.0, x))
    p = 1.0 / (1.0 + np.exp(-x))
    return float(max(0.01, min(0.99, p)))


# Back-compat alias
score_to_probability = score_to_baseline_model_probability


def apply_calibration(p: float, horizon_h: int, calib: dict | None) -> float | None:
    """Return calibrated probability if curve exists; else None (do not invent)."""
    if not calib or str(horizon_h) not in calib:
        return None
    curve = calib[str(horizon_h)]
    if not curve or len(curve) < 3:
        return None
    xs = [c[0] for c in curve]
    ys = [c[1] for c in curve]
    return float(np.interp(p, xs, ys))


def load_calibration(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if data else None
    except Exception:
        return None


def horizons_probabilities(
    signal_score: float,
    cfg: dict[str, Any],
    calib: dict | None = None,
) -> dict[str, Any]:
    """Build per-horizon outputs with explicit status labels."""
    out: dict[str, Any] = {
        "probability_kind": PROBABILITY_STATUS_BASELINE,
        "calibration_applied": False,
        "disclaimer": (
            "Displayed values are baseline_model_probability from logistic(signal_score) "
            "unless calibration_applied=true. They are NOT validated hit-rates."
        ),
    }
    any_calibrated = False
    for h in cfg["probability"]["horizons_hours"]:
        baseline = score_to_baseline_model_probability(signal_score, cfg, int(h))
        calibrated = apply_calibration(baseline, int(h), calib)
        out[f"baseline_model_p_{h}h"] = baseline
        if calibrated is not None:
            out[f"p_{h}h"] = calibrated
            out[f"p_{h}h_status"] = PROBABILITY_STATUS_CALIBRATED
            any_calibrated = True
        else:
            out[f"p_{h}h"] = baseline
            out[f"p_{h}h_status"] = PROBABILITY_STATUS_BASELINE
    if any_calibrated:
        out["probability_kind"] = "mixed_or_calibrated"
        out["calibration_applied"] = True
    return out


def classify_signal(p_4h: float, late_entry_score: float, cfg: dict[str, Any] | None = None) -> dict[str, str]:
    """Distinguish Strong / Strong but late / Weak (and weak+late)."""
    strong_thr = 0.70
    weak_thr = 0.60
    late_thr = 0.75
    if cfg:
        strong_thr = float(cfg.get("signal_classes", {}).get("strong_p4h", strong_thr))
        weak_thr = float(cfg.get("signal_classes", {}).get("weak_p4h_below", weak_thr))
        late_thr = float(cfg.get("late_entry", {}).get("alert_threshold", late_thr))

    strong = p_4h >= strong_thr
    weak = p_4h < weak_thr
    late = late_entry_score >= late_thr

    if strong and late:
        return {
            "signal_class": "STRONG_BUT_LATE",
            "label": "Strong but late",
            "emoji": "🔴⭐",
        }
    if strong and not late:
        return {
            "signal_class": "STRONG_SIGNAL",
            "label": "Strong signal",
            "emoji": "🟢",
        }
    if weak:
        return {
            "signal_class": "WEAK_SIGNAL",
            "label": "Weak signal",
            "emoji": "⚪",
        }
    if late:
        return {
            "signal_class": "MODERATE_BUT_LATE",
            "label": "Moderate but late",
            "emoji": "🟠",
        }
    return {
        "signal_class": "MODERATE_SIGNAL",
        "label": "Moderate signal",
        "emoji": "🟡",
    }
