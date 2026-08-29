"""Late Entry / Exhaustion score — separate from probability."""

from __future__ import annotations

from typing import Any

import pandas as pd

from btcc.factors.helpers import bollinger, clamp01, ema, last, percentile_rank, rsi
from btcc.series.relative import horizon_bars, relative_return


def _classify(score: float) -> str:
    if score < 0.30:
        return "NORMAL"
    if score < 0.55:
        return "EXTENDED"
    if score < 0.75:
        return "HIGH_LATE_ENTRY_RISK"
    return "VERY_HIGH_LATE_ENTRY_RISK"


def _emoji(score: float) -> str:
    if score < 0.30:
        return "GREEN"
    if score < 0.55:
        return "YELLOW"
    if score < 0.75:
        return "ORANGE"
    return "RED_STAR"


def late_entry_score(
    alt_btc: pd.DataFrame,
    factors: dict[str, Any],
    cfg: dict[str, Any],
    interval: str = "15m",
) -> dict[str, Any]:
    w = cfg["late_entry"]["weights"]
    c = alt_btc["close"]

    # Extension — percentile of multi-horizon returns
    ext_parts = []
    for h in (4, 8, 24):
        bars = horizon_bars(h, interval)
        r = relative_return(c, bars)
        if r is None or len(c) < bars + 50:
            ext_parts.append(0.5)
            continue
        hist = c.pct_change(bars).dropna()
        ext_parts.append(percentile_rank(hist, r))
    extension = sum(ext_parts) / len(ext_parts)

    # EMA distance
    e20, e50 = ema(c, 20), ema(c, 50)
    px = last(c) or 1.0
    d20 = ((last(c) or 0) - (last(e20) or 0)) / px
    d50 = ((last(c) or 0) - (last(e50) or 0)) / px
    ema_dist = clamp01(0.5 * clamp01(d20 / 0.05) + 0.5 * clamp01(d50 / 0.08))

    # Bollinger
    _, _, _, pct_b, bw = bollinger(c)
    pb = last(pct_b) or 0.5
    bb_ext = clamp01((pb - 0.5) / 0.7) if pb > 0.5 else 0.2

    # RSI exhaustion
    rsi_f = factors.get("rsi", {})
    rv = rsi_f.get("rsi14")
    rsi_ex = 0.3
    if rv is not None:
        rsi_ex = clamp01((rv - 55) / 40)
        if rsi_f.get("bearish_divergence"):
            rsi_ex = clamp01(rsi_ex + 0.15)

    # Momentum decay
    mom = factors.get("momentum", {})
    decay = 0.7 if mom.get("momentum_decay") else 0.25
    accel = mom.get("accel_4h")
    if accel is not None and accel < 0 and (mom.get("return_4h") or 0) > 0:
        decay = clamp01(decay + 0.15)

    # Volume divergence
    vol_div = 0.75 if factors.get("volume", {}).get("volume_divergence") else 0.25

    # Resistance proximity
    st = factors.get("structure", {})
    dist = st.get("dist_to_resistance")
    if dist is None:
        resist = 0.5
    elif dist <= 0:
        resist = 0.85  # at/through resistance
    else:
        resist = clamp01(1.0 - dist / 0.05)

    # Reversal evidence (small weight)
    rev = 0.2
    if rsi_f.get("bearish_divergence"):
        rev += 0.25
    if mom.get("momentum_decay"):
        rev += 0.2
    if factors.get("trend", {}).get("macd_score", 0.5) < 0.4:
        rev += 0.15
    rev = clamp01(rev)

    components = {
        "extension": extension,
        "ema_distance": ema_dist,
        "bollinger_extension": bb_ext,
        "rsi_exhaustion": rsi_ex,
        "momentum_decay": decay,
        "volume_divergence": vol_div,
        "resistance_proximity": resist,
        "reversal_evidence": rev,
    }
    score = sum(w[k] * components[k] for k in w)
    score = clamp01(score)

    # Ranked reasons for audit / Telegram
    reasons = sorted(
        ((k, float(v), float(w[k])) for k, v in components.items()),
        key=lambda x: x[1] * x[2],
        reverse=True,
    )
    top_reasons = [
        {"component": k, "value": v, "weight": wt, "contribution": v * wt}
        for k, v, wt in reasons[:5]
    ]

    return {
        "late_entry_score": score,
        "classification": _classify(score),
        "emoji_key": _emoji(score),
        "components": components,
        "weights": w,
        "top_reasons": top_reasons,
    }
