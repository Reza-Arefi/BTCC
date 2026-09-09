"""Research-only early-breakout entry scoring variants (E1-E5).

Does NOT modify frozen live factor modules (momentum / rsi / volatility / structure).
BASE scoring continues to use btcc.factors.* unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from btcc.factors.helpers import atr, bollinger, clamp01, last, rsi, sigmoid
from btcc.factors.structure import structure_factor
from btcc.factors.trend import trend_factor
from btcc.factors.volume import volume_factor
from btcc.series.relative import horizon_bars, relative_return
from btcc.sim.score import ACTIVE_SIGNAL_KEYS, combined_score, normalize_weights

# E3 anti-chase defaults (configurable for sensitivity tests)
E3_MAX_15M_RET = 0.015
E3_MAX_PCT_B = 1.20
E3_STRONG_BOOST = 0.12
E3_MODERATE_BOOST = 0.07
E3_RVOL_MIN = 1.5


def momentum_factor_e2(alt_btc: pd.DataFrame, interval: str = "15m") -> dict[str, Any]:
    """Faster momentum profile — delegates to shared momentum_factor(profile='e2')."""
    from btcc.factors.momentum import momentum_factor

    return momentum_factor(alt_btc, interval, profile="e2")



def rsi_factor_e1(alt_btc: pd.DataFrame) -> dict[str, Any]:
    """Breakout-aware RSI: elevate fresh 70 cross + rising slope; keep bearish-div."""
    r = rsi(alt_btc["close"], 14)
    rv = last(r)
    slope = last(r.diff())
    prev = float(r.iloc[-2]) if len(r) >= 2 and pd.notna(r.iloc[-2]) else None
    rsi_cross_70 = bool(prev is not None and rv is not None and prev < 70.0 and rv >= 70.0)

    score = 0.5
    if rv is not None:
        if 50 <= rv <= 70:
            base = 0.65 + 0.2 * (rv - 50) / 20
            if slope is not None and slope > 0:
                base += 0.08
        elif rv < 50:
            base = 0.35 + 0.3 * (rv / 50)
            if slope is not None and slope > 0:
                base += 0.08
        elif rv <= 75:
            if rsi_cross_70 and slope is not None and slope > 0:
                base = 0.80
            else:
                base = 0.55
                if slope is not None and slope > 0:
                    base += 0.08
        else:
            if slope is not None and slope > 0:
                base = 0.65
            else:
                base = 0.35
        if slope is not None and slope < 0 and rv > 70:
            base -= 0.10
        score = clamp01(base)

    bear_div = False
    if len(alt_btc) >= 20 and rv is not None:
        px = alt_btc["close"].iloc[-20:]
        rs = r.iloc[-20:]
        if px.iloc[-1] >= px.max() * 0.995 and rs.iloc[-1] < rs.max() * 0.97:
            bear_div = True
            score = clamp01(score - 0.08)

    return {
        "score": score,
        "rsi14": rv,
        "rsi_slope": slope,
        "rsi_prev": prev,
        "rsi_cross_70": rsi_cross_70,
        "bearish_divergence": bear_div,
        "overheated": bool(rv is not None and rv > 75),
        "variant": "E1",
    }


def volatility_factor_e1(alt_btc: pd.DataFrame) -> dict[str, Any]:
    """Breakout-aware Bollinger: reward %B cross above 1 while bandwidth expands."""
    c = alt_btc["close"]
    mid, upper, lower, pct_b, bw = bollinger(c)
    a = atr(alt_btc, 14)
    natr = (last(a) / last(c)) if last(a) and last(c) else None
    pb = last(pct_b)
    pb_prev = float(pct_b.iloc[-2]) if len(pct_b) >= 2 and pd.notna(pct_b.iloc[-2]) else None
    bwv = last(bw)
    bw_chg = last(bw.diff()) if bw is not None else None
    bb_breakout = bool(pb_prev is not None and pb is not None and pb_prev < 1.0 and pb >= 1.0)
    bw_expanding = bool(bw_chg is not None and bw_chg > 0)

    score = 0.5
    bollinger_score = None
    atr_score = None
    natr_score = None
    loc_score_raw = None
    if pb is not None and bwv is not None:
        expand = clamp01(((bw_chg or 0) * 50) + 0.5)
        loc = clamp01(pb)
        loc_score = 1.0 - abs(loc - 0.7) / 0.7
        loc_score_raw = loc_score
        if bb_breakout and bw_expanding:
            loc_score = 0.90
        natr_score = clamp01(1.0 - min(abs((natr or 0.015) - 0.015) / 0.05, 1.0))
        bollinger_score = clamp01(0.55 * expand + 0.45 * clamp01(loc_score))
        atr_score = clamp01(min((natr or 0) / 0.05, 1.0)) if natr is not None else None
        score = clamp01(0.45 * expand + 0.35 * clamp01(loc_score) + 0.20 * (natr_score or 0.5))

    return {
        "score": score,
        "bollinger_score": bollinger_score,
        "atr_score": atr_score,
        "natr_score": natr_score,
        "percent_b": pb,
        "percent_b_prev": pb_prev,
        "bandwidth": bwv,
        "bandwidth_change": bw_chg,
        "natr": natr,
        "bb_breakout": bb_breakout,
        "bw_expanding": bw_expanding,
        "loc_score": loc_score_raw,
        "above_upper_band": bool(pb is not None and pb > 1.0),
        "variant": "E1",
    }


def _breakout_diagnostics(
    alt_btc: pd.DataFrame,
    volu: dict[str, Any],
    volat: dict[str, Any],
    rsi_f: dict[str, Any],
    struct: dict[str, Any],
    interval: str,
) -> dict[str, Any]:
    c = alt_btc["close"]
    ret_15m = float(c.iloc[-1] / c.iloc[-2] - 1.0) if len(c) >= 2 else None
    b4 = horizon_bars(4, interval)
    cur4 = relative_return(c, b4)
    prev4 = float(c.iloc[-1 - b4] / c.iloc[-1 - 2 * b4] - 1.0) if len(c) > 2 * b4 else None
    accel_4h = (float(cur4) - float(prev4)) if cur4 is not None and prev4 is not None else None

    pb = volat.get("percent_b")
    pb_prev = volat.get("percent_b_prev")
    if pb is None or pb_prev is None:
        _, _, _, pct_b, _ = bollinger(c)
        if pb is None:
            pb = last(pct_b)
        if pb_prev is None:
            pb_prev = float(pct_b.iloc[-2]) if len(pct_b) >= 2 and pd.notna(pct_b.iloc[-2]) else None
    bb_breakout = bool(pb_prev is not None and pb is not None and pb_prev < 1.0 and pb >= 1.0)

    bw_chg = volat.get("bandwidth_change")
    if bw_chg is None:
        _, _, _, _, bw = bollinger(c)
        bw_chg = last(bw.diff()) if bw is not None else None
    bw_expanding = bool(bw_chg is not None and bw_chg > 0)

    rsi14 = rsi_f.get("rsi14")
    rsi_prev = rsi_f.get("rsi_prev")
    rsi_slope = rsi_f.get("rsi_slope")
    if rsi14 is None or rsi_prev is None or rsi_slope is None:
        rser = rsi(c, 14)
        if rsi14 is None:
            rsi14 = last(rser)
        if rsi_prev is None:
            rsi_prev = float(rser.iloc[-2]) if len(rser) >= 2 and pd.notna(rser.iloc[-2]) else None
        if rsi_slope is None:
            rsi_slope = last(rser.diff())
    rsi_cross_70 = bool(
        rsi_prev is not None
        and rsi14 is not None
        and rsi_prev < 70.0
        and rsi14 >= 70.0
        and rsi_slope is not None
        and rsi_slope > 0
    )

    structure_breakout = bool(struct.get("breakout"))
    close_px = float(c.iloc[-1]) if len(c) else None
    resistance = struct.get("resistance")
    if resistance is not None and close_px is not None:
        structure_breakout = bool(close_px >= float(resistance) * 0.999)

    return {
        "ret_15m": ret_15m,
        "accel_4h": accel_4h,
        "percent_b": pb,
        "percent_b_prev": pb_prev,
        "bb_breakout": bb_breakout,
        "bw_expanding": bw_expanding,
        "bandwidth_change": bw_chg,
        "rsi14": rsi14,
        "rsi_prev": rsi_prev,
        "rsi_slope": rsi_slope,
        "rsi_cross_70": rsi_cross_70,
        "rvol": volu.get("rvol"),
        "structure_breakout": structure_breakout,
        "resistance": resistance,
        "support": struct.get("support"),
        "close": close_px,
    }


def score_variant(
    *,
    alt_btc: pd.DataFrame,
    alt_usdt: pd.DataFrame,
    btc_usdt: pd.DataFrame,
    weights: dict[str, float],
    interval: str = "15m",
    use_e1: bool = False,
    use_e2: bool = False,
    shared: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Signed S for BASE / E1 / E2 / E4 factor mixes."""
    from btcc.factors.btc_regime import btc_regime_factor
    from btcc.factors.momentum import momentum_factor
    from btcc.factors.rsi_factor import rsi_factor
    from btcc.factors.volatility import volatility_factor

    if shared is None:
        shared = {
            "trend": trend_factor(alt_btc),
            "volu": volume_factor(alt_usdt, alt_btc),
            "struct": structure_factor(alt_btc),
            "regime": btc_regime_factor(btc_usdt, None, {}, interval),
        }

    mom = momentum_factor_e2(alt_btc, interval) if use_e2 else momentum_factor(alt_btc, interval)
    volat = volatility_factor_e1(alt_btc) if use_e1 else volatility_factor(alt_btc)
    rsi_f = rsi_factor_e1(alt_btc) if use_e1 else rsi_factor(alt_btc)
    trend = shared["trend"]
    volu = shared["volu"]
    struct = shared["struct"]
    regime = shared["regime"]

    factor_scores = {
        "momentum": float(mom["score"]),
        "trend": float(trend["score"]),
        "volume": float(volu["score"]),
        "volatility": float(volat["score"]),
        "rsi": float(rsi_f["score"]),
        "structure": float(struct["score"]),
        "btc_regime": float(regime["score"]),
    }
    S = float(combined_score(factor_scores, normalize_weights(weights))["S"])
    diag = _breakout_diagnostics(alt_btc, volu, volat, rsi_f, struct, interval)
    return {
        "S": S,
        "factor_scores": factor_scores,
        "momentum": mom,
        "trend": trend,
        "volume": volu,
        "volatility": volat,
        "rsi": rsi_f,
        "structure": struct,
        "diagnostics": diag,
        "weights": normalize_weights(weights),
        "active_signal_keys": list(ACTIVE_SIGNAL_KEYS),
    }


def e3_breakout_assessment(
    diag: dict[str, Any],
    *,
    max_15m_ret: float = E3_MAX_15M_RET,
    max_pct_b: float = E3_MAX_PCT_B,
    rvol_min: float = E3_RVOL_MIN,
    strong_boost: float = E3_STRONG_BOOST,
    moderate_boost: float = E3_MODERATE_BOOST,
) -> dict[str, Any]:
    """Causal breakout boost assessed on the current closed 15m candle only."""
    conds = {
        "bb_breakout": bool(diag.get("bb_breakout")),
        "rsi_breakout": bool(diag.get("rsi_cross_70")),
        "volume_confirmation": bool(diag.get("rvol") is not None and float(diag["rvol"]) >= rvol_min),
        "short_term_accel": bool(
            diag.get("ret_15m") is not None
            and float(diag["ret_15m"]) > 0
            and diag.get("accel_4h") is not None
            and float(diag["accel_4h"]) > 0
        ),
        "structure_breakout": bool(diag.get("structure_breakout")),
    }
    n = int(sum(1 for v in conds.values() if v))
    strong = n >= 4
    moderate = n >= 3
    ret15 = diag.get("ret_15m")
    pb = diag.get("percent_b")
    chase = bool(
        (ret15 is not None and float(ret15) > max_15m_ret)
        or (pb is not None and float(pb) > max_pct_b)
    )

    boost = 0.0
    rejected_chase = False
    if strong:
        boost = float(strong_boost)
    elif moderate:
        if chase:
            boost = 0.0
            rejected_chase = True
        else:
            boost = float(moderate_boost)

    return {
        "conditions": conds,
        "n_conditions": n,
        "strong_breakout": strong,
        "moderate_breakout": moderate and not strong,
        "boost": float(boost),
        "rejected_chase": rejected_chase,
        "chase_flag": chase,
        "max_15m_ret": max_15m_ret,
        "max_pct_b": max_pct_b,
        "rvol_min": rvol_min,
        "strong_boost": strong_boost,
        "moderate_boost": moderate_boost,
    }


def apply_e3(S: float, diag: dict[str, Any], **kwargs: Any) -> tuple[float, dict[str, Any]]:
    assess = e3_breakout_assessment(diag, **kwargs)
    s_eff = float(np.clip(float(S) + float(assess["boost"]), -1.0, 1.0))
    return s_eff, assess


def score_all_variants(
    *,
    alt_btc: pd.DataFrame,
    alt_usdt: pd.DataFrame,
    btc_usdt: pd.DataFrame,
    weights: dict[str, float],
    interval: str = "15m",
    e3_kwargs: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute BASE/E1/E2/E3/E4/E5 effective S on the same closed bar."""
    from btcc.factors.btc_regime import btc_regime_factor

    shared = {
        "trend": trend_factor(alt_btc),
        "volu": volume_factor(alt_usdt, alt_btc),
        "struct": structure_factor(alt_btc),
        "regime": btc_regime_factor(btc_usdt, None, {}, interval),
    }
    e3_kwargs = dict(e3_kwargs or {})

    base = score_variant(
        alt_btc=alt_btc,
        alt_usdt=alt_usdt,
        btc_usdt=btc_usdt,
        weights=weights,
        interval=interval,
        use_e1=False,
        use_e2=False,
        shared=shared,
    )
    e1 = score_variant(
        alt_btc=alt_btc,
        alt_usdt=alt_usdt,
        btc_usdt=btc_usdt,
        weights=weights,
        interval=interval,
        use_e1=True,
        use_e2=False,
        shared=shared,
    )
    e2 = score_variant(
        alt_btc=alt_btc,
        alt_usdt=alt_usdt,
        btc_usdt=btc_usdt,
        weights=weights,
        interval=interval,
        use_e1=False,
        use_e2=True,
        shared=shared,
    )
    e4 = score_variant(
        alt_btc=alt_btc,
        alt_usdt=alt_usdt,
        btc_usdt=btc_usdt,
        weights=weights,
        interval=interval,
        use_e1=True,
        use_e2=True,
        shared=shared,
    )

    s_e3, a_e3 = apply_e3(base["S"], base["diagnostics"], **e3_kwargs)
    s_e5, a_e5 = apply_e3(e4["S"], e4["diagnostics"], **e3_kwargs)

    return {
        "BASE": {**base, "S_effective": base["S"], "e3": None},
        "E1": {**e1, "S_effective": e1["S"], "e3": None},
        "E2": {**e2, "S_effective": e2["S"], "e3": None},
        "E3": {**base, "S_effective": s_e3, "e3": a_e3},
        "E4": {**e4, "S_effective": e4["S"], "e3": None},
        "E5": {**e4, "S_effective": s_e5, "e3": a_e5},
    }
