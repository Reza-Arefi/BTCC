"""Champion vs Challenger promotion rules."""

from __future__ import annotations

from typing import Any

from btcc.adaptive.metrics import evaluate_model_on_frame
from btcc.adaptive.model import SignalModel
from btcc.adaptive.rescore import rescore_frame


def compare_and_decide(
    champion: SignalModel,
    challenger: SignalModel,
    val_df,
    signal_cfg: dict[str, Any],
    adaptive_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Out-of-sample comparison on validation frame. Never uses training rows."""
    champ_scored = rescore_frame(val_df, champion, signal_cfg)
    chal_scored = rescore_frame(val_df, challenger, signal_cfg)

    m_champ = evaluate_model_on_frame(champ_scored)
    m_chal = evaluate_model_on_frame(chal_scored)
    p_champ = m_champ["primary"]
    p_chal = m_chal["primary"]

    brier_eps = float(adaptive_cfg.get("promote_brier_improvement", 0.005))
    cal_eps = float(adaptive_cfg.get("promote_calibration_improvement", 0.005))
    rank_eps = float(adaptive_cfg.get("promote_ranking_improvement", 0.01))

    brier_ok = (
        p_chal.get("n", 0) >= 50
        and p_champ.get("brier") is not None
        and p_chal.get("brier") is not None
        and (p_champ["brier"] - p_chal["brier"]) >= brier_eps
    )
    cal_ok = (
        p_champ.get("calibration_error") is not None
        and p_chal.get("calibration_error") is not None
        and (p_champ["calibration_error"] - p_chal["calibration_error"]) >= cal_eps
    )
    # ranking: higher is better
    rq_c = p_champ.get("ranking_quality_top5")
    rq_n = p_chal.get("ranking_quality_top5")
    rank_ok = (
        rq_c == rq_c and rq_n == rq_n  # not NaN
        and (rq_n - rq_c) >= rank_eps
    )

    # Require Brier improvement AND (calibration OR ranking improvement)
    promote = bool(brier_ok and (cal_ok or rank_ok))

    decision = {
        "action": "PROMOTE_CHALLENGER" if promote else "KEEP_CHAMPION",
        "promote": promote,
        "champion_metrics_4h": p_champ,
        "challenger_metrics_4h": p_chal,
        "checks": {
            "brier_improved": brier_ok,
            "calibration_improved": cal_ok,
            "ranking_improved": rank_ok,
            "brier_delta": (p_champ.get("brier") or 0) - (p_chal.get("brier") or 0),
            "calibration_delta": (p_champ.get("calibration_error") or 0) - (p_chal.get("calibration_error") or 0),
            "ranking_delta": (rq_n - rq_c) if (rq_c == rq_c and rq_n == rq_n) else None,
        },
        "thresholds": {
            "brier_eps": brier_eps,
            "cal_eps": cal_eps,
            "rank_eps": rank_eps,
        },
        "rule": "Promote only if 4h Brier improves by ≥eps AND (calibration OR ranking improves).",
    }
    challenger.metrics["validation"] = m_chal
    challenger.metrics["champion_validation"] = m_champ
    return decision
