"""Phase VII — B0 pullback/retest continuation signals (Ichimoku-only).

Frozen discovery parameters (NOT optimized / not grid-searched):

  Established trend (bull): close > KumoTop AND Kumo bullish (SenkouA > SenkouB)
  Established trend (bear): close < KumoBot AND Kumo bearish

  Meaningful move:
      M = |C_t - C_{t-k}| / ATR14_t ,  k = MOVE_K = 5
      require M >= MOVE_MIN_ATR = 1.0
      and extension from structure >= EXT_MIN_ATR = 0.5

  Retest window: RETEST_LOOKBACK = 12 bars
  Near-touch band: NEAR_ATR = 0.35
  Min bars established in lookback: MIN_EST_BARS = 3

Entry = close of the reclaim / recross confirmation bar (immediate).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

Direction = Literal["bull", "bear"]
Variant = Literal["B0A", "B0B", "B0C"]

# Frozen discovery constants — do not tune in Phase VII.
MOVE_K = 5
MOVE_MIN_ATR = 1.0
EXT_MIN_ATR = 0.5
RETEST_LOOKBACK = 12
NEAR_ATR = 0.35
MIN_EST_BARS = 3
COOLDOWN_BARS = 3  # suppress duplicate fires from same pullback cluster


@dataclass
class B0Event:
    variant: str
    direction: str
    tf: str
    idx: int
    timestamp: str
    entry_px: float
    move_m_atr: float
    ext_atr: float
    deepest_pull_atr: float
    penetration_atr: float
    touch_class: str  # near_miss | exact_touch | shallow_pen | deep_pen
    bars_near_structure: int
    dist_to_structure_atr: float
    kumo_top: float
    kumo_bot: float
    kijun: float
    tenkan: float
    atr14: float


def _valid_row(df: pd.DataFrame, i: int) -> bool:
    row = df.iloc[i]
    return bool(
        pd.notna(row.get("tenkan"))
        and pd.notna(row.get("kijun"))
        and pd.notna(row.get("kumo_top"))
        and pd.notna(row.get("kumo_bot"))
        and pd.notna(row.get("atr14"))
        and float(row["atr14"]) > 0
    )


def established_bull(df: pd.DataFrame, i: int) -> bool:
    row = df.iloc[i]
    return bool(float(row["close"]) > float(row["kumo_top"]) and bool(row.get("kumo_bullish")))


def established_bear(df: pd.DataFrame, i: int) -> bool:
    row = df.iloc[i]
    return bool(float(row["close"]) < float(row["kumo_bot"]) and not bool(row.get("kumo_bullish")))


def _move_m(df: pd.DataFrame, t: int) -> float:
    if t < MOVE_K:
        return float("nan")
    atrv = float(df.iloc[t]["atr14"])
    if not np.isfinite(atrv) or atrv <= 0:
        return float("nan")
    return abs(float(df.iloc[t]["close"]) - float(df.iloc[t - MOVE_K]["close"])) / atrv


def _touch_class(pen_atr: float, near: bool, touched: bool) -> str:
    if touched and pen_atr > 0.5:
        return "deep_pen"
    if touched and pen_atr > 0:
        return "shallow_pen"
    if touched or (near and pen_atr >= 0):
        return "exact_touch" if touched else "near_miss"
    if near:
        return "near_miss"
    return "none"


def _prior_impulse(
    df: pd.DataFrame,
    j: int,
    *,
    bull: bool,
) -> tuple[bool, float, float, int]:
    """Return (ok, best_M, best_ext, n_established) in lookback before j."""
    start = max(MOVE_K, j - RETEST_LOOKBACK)
    best_m = float("nan")
    best_ext = float("nan")
    n_est = 0
    ok = False
    for t in range(start, j):
        if not _valid_row(df, t):
            continue
        est = established_bull(df, t) if bull else established_bear(df, t)
        if not est:
            continue
        n_est += 1
        m = _move_m(df, t)
        atrv = float(df.iloc[t]["atr14"])
        c = float(df.iloc[t]["close"])
        if bull:
            ext = (c - float(df.iloc[t]["kumo_top"])) / atrv
        else:
            ext = (float(df.iloc[t]["kumo_bot"]) - c) / atrv
        if np.isfinite(m) and (not np.isfinite(best_m) or m > best_m):
            best_m = m
        if np.isfinite(ext) and (not np.isfinite(best_ext) or ext > best_ext):
            best_ext = ext
        if np.isfinite(m) and m >= MOVE_MIN_ATR and np.isfinite(ext) and ext >= EXT_MIN_ATR:
            ok = True
    return ok and n_est >= MIN_EST_BARS, best_m, best_ext, n_est


def _bars_near(df: pd.DataFrame, j: int, *, bull: bool, level: str) -> int:
    """Count bars in lookback whose low/high interacted with structure band."""
    start = max(0, j - RETEST_LOOKBACK)
    cnt = 0
    for t in range(start, j + 1):
        if not _valid_row(df, t):
            continue
        atrv = float(df.iloc[t]["atr14"])
        if level == "kumo":
            lvl = float(df.iloc[t]["kumo_top"] if bull else df.iloc[t]["kumo_bot"])
        else:
            lvl = float(df.iloc[t]["kijun"])
        if bull:
            near = float(df.iloc[t]["low"]) <= lvl + NEAR_ATR * atrv
        else:
            near = float(df.iloc[t]["high"]) >= lvl - NEAR_ATR * atrv
        if near:
            cnt += 1
    return cnt


def detect_b0a(df: pd.DataFrame, *, tf: str) -> list[B0Event]:
    """B0-A Kumo retest / bounce (bull + bear)."""
    events: list[B0Event] = []
    last_fire = {"bull": -10**9, "bear": -10**9}
    n = len(df)
    for j in range(MOVE_K + 1, n):
        if not _valid_row(df, j):
            continue
        row = df.iloc[j]
        atrv = float(row["atr14"])
        top = float(row["kumo_top"])
        bot = float(row["kumo_bot"])
        c = float(row["close"])
        lo = float(row["low"])
        hi = float(row["high"])

        # --- bull ---
        if j - last_fire["bull"] > COOLDOWN_BARS:
            reclaim = c > top
            touched = lo <= top
            near = lo <= top + NEAR_ATR * atrv
            if reclaim and (touched or near):
                ok, best_m, best_ext, _ = _prior_impulse(df, j, bull=True)
                if ok:
                    pen = (top - lo) / atrv
                    tc = _touch_class(pen, near, touched)
                    events.append(
                        B0Event(
                            variant="B0A",
                            direction="bull",
                            tf=tf,
                            idx=j,
                            timestamp=str(row["timestamp"]),
                            entry_px=c,
                            move_m_atr=float(best_m),
                            ext_atr=float(best_ext),
                            deepest_pull_atr=float(pen),
                            penetration_atr=float(max(0.0, pen)),
                            touch_class=tc,
                            bars_near_structure=_bars_near(df, j, bull=True, level="kumo"),
                            dist_to_structure_atr=float((c - top) / atrv),
                            kumo_top=top,
                            kumo_bot=bot,
                            kijun=float(row["kijun"]),
                            tenkan=float(row["tenkan"]),
                            atr14=atrv,
                        )
                    )
                    last_fire["bull"] = j

        # --- bear ---
        if j - last_fire["bear"] > COOLDOWN_BARS:
            reclaim = c < bot
            touched = hi >= bot
            near = hi >= bot - NEAR_ATR * atrv
            if reclaim and (touched or near):
                ok, best_m, best_ext, _ = _prior_impulse(df, j, bull=False)
                if ok:
                    pen = (hi - bot) / atrv
                    tc = _touch_class(pen, near, touched)
                    events.append(
                        B0Event(
                            variant="B0A",
                            direction="bear",
                            tf=tf,
                            idx=j,
                            timestamp=str(row["timestamp"]),
                            entry_px=c,
                            move_m_atr=float(best_m),
                            ext_atr=float(best_ext),
                            deepest_pull_atr=float(pen),
                            penetration_atr=float(max(0.0, pen)),
                            touch_class=tc,
                            bars_near_structure=_bars_near(df, j, bull=False, level="kumo"),
                            dist_to_structure_atr=float((bot - c) / atrv),
                            kumo_top=top,
                            kumo_bot=bot,
                            kijun=float(row["kijun"]),
                            tenkan=float(row["tenkan"]),
                            atr14=atrv,
                        )
                    )
                    last_fire["bear"] = j
    return events


def detect_b0b(df: pd.DataFrame, *, tf: str) -> list[B0Event]:
    """B0-B Kijun retest / reclaim (bull + bear), still requiring Kumo regime."""
    events: list[B0Event] = []
    last_fire = {"bull": -10**9, "bear": -10**9}
    n = len(df)
    for j in range(MOVE_K + 1, n):
        if not _valid_row(df, j):
            continue
        row = df.iloc[j]
        atrv = float(row["atr14"])
        kijun = float(row["kijun"])
        top = float(row["kumo_top"])
        bot = float(row["kumo_bot"])
        c = float(row["close"])
        lo = float(row["low"])
        hi = float(row["high"])

        if j - last_fire["bull"] > COOLDOWN_BARS:
            # Must finish above Kumo (trend intact) and reclaim Kijun
            reclaim = c > kijun and c > top
            touched = lo <= kijun
            near = lo <= kijun + NEAR_ATR * atrv
            was_below = float(df.iloc[j - 1]["close"]) <= float(df.iloc[j - 1]["kijun"])
            if reclaim and (touched or near) and was_below:
                ok, best_m, best_ext, _ = _prior_impulse(df, j, bull=True)
                if ok:
                    pen = (kijun - lo) / atrv
                    tc = _touch_class(pen, near, touched)
                    events.append(
                        B0Event(
                            variant="B0B",
                            direction="bull",
                            tf=tf,
                            idx=j,
                            timestamp=str(row["timestamp"]),
                            entry_px=c,
                            move_m_atr=float(best_m),
                            ext_atr=float(best_ext),
                            deepest_pull_atr=float(pen),
                            penetration_atr=float(max(0.0, pen)),
                            touch_class=tc,
                            bars_near_structure=_bars_near(df, j, bull=True, level="kijun"),
                            dist_to_structure_atr=float((c - kijun) / atrv),
                            kumo_top=top,
                            kumo_bot=bot,
                            kijun=kijun,
                            tenkan=float(row["tenkan"]),
                            atr14=atrv,
                        )
                    )
                    last_fire["bull"] = j

        if j - last_fire["bear"] > COOLDOWN_BARS:
            reclaim = c < kijun and c < bot
            touched = hi >= kijun
            near = hi >= kijun - NEAR_ATR * atrv
            was_above = float(df.iloc[j - 1]["close"]) >= float(df.iloc[j - 1]["kijun"])
            if reclaim and (touched or near) and was_above:
                ok, best_m, best_ext, _ = _prior_impulse(df, j, bull=False)
                if ok:
                    pen = (hi - kijun) / atrv
                    tc = _touch_class(pen, near, touched)
                    events.append(
                        B0Event(
                            variant="B0B",
                            direction="bear",
                            tf=tf,
                            idx=j,
                            timestamp=str(row["timestamp"]),
                            entry_px=c,
                            move_m_atr=float(best_m),
                            ext_atr=float(best_ext),
                            deepest_pull_atr=float(pen),
                            penetration_atr=float(max(0.0, pen)),
                            touch_class=tc,
                            bars_near_structure=_bars_near(df, j, bull=False, level="kijun"),
                            dist_to_structure_atr=float((kijun - c) / atrv),
                            kumo_top=top,
                            kumo_bot=bot,
                            kijun=kijun,
                            tenkan=float(row["tenkan"]),
                            atr14=atrv,
                        )
                    )
                    last_fire["bear"] = j
    return events


def detect_b0c(df: pd.DataFrame, *, tf: str) -> list[B0Event]:
    """B0-C Tenkan/Kijun recross after correction inside established trend."""
    events: list[B0Event] = []
    last_fire = {"bull": -10**9, "bear": -10**9}
    n = len(df)
    tenkan = df["tenkan"].astype(float).values
    kijun = df["kijun"].astype(float).values

    for j in range(MOVE_K + 2, n):
        if not _valid_row(df, j) or not _valid_row(df, j - 1):
            continue
        row = df.iloc[j]
        atrv = float(row["atr14"])
        top = float(row["kumo_top"])
        bot = float(row["kumo_bot"])
        c = float(row["close"])

        # Bull: TK cross up while above Kumo, after a prior TK loss in lookback
        if j - last_fire["bull"] > COOLDOWN_BARS:
            cross_up = tenkan[j - 1] <= kijun[j - 1] and tenkan[j] > kijun[j]
            above = c > top and bool(row.get("kumo_bullish"))
            if cross_up and above:
                # Require a prior bullish TK state then loss (correction), plus impulse
                had_tk_bull = False
                had_loss = False
                start = max(MOVE_K, j - RETEST_LOOKBACK)
                for t in range(start, j):
                    if tenkan[t] > kijun[t] and established_bull(df, t):
                        had_tk_bull = True
                    if had_tk_bull and tenkan[t] <= kijun[t]:
                        had_loss = True
                        break
                ok, best_m, best_ext, _ = _prior_impulse(df, j, bull=True)
                if had_loss and ok:
                    events.append(
                        B0Event(
                            variant="B0C",
                            direction="bull",
                            tf=tf,
                            idx=j,
                            timestamp=str(row["timestamp"]),
                            entry_px=c,
                            move_m_atr=float(best_m),
                            ext_atr=float(best_ext),
                            deepest_pull_atr=float("nan"),
                            penetration_atr=float("nan"),
                            touch_class="tk_recross",
                            bars_near_structure=0,
                            dist_to_structure_atr=float((tenkan[j] - kijun[j]) / atrv),
                            kumo_top=top,
                            kumo_bot=bot,
                            kijun=float(kijun[j]),
                            tenkan=float(tenkan[j]),
                            atr14=atrv,
                        )
                    )
                    last_fire["bull"] = j

        if j - last_fire["bear"] > COOLDOWN_BARS:
            cross_dn = tenkan[j - 1] >= kijun[j - 1] and tenkan[j] < kijun[j]
            below = c < bot and not bool(row.get("kumo_bullish"))
            if cross_dn and below:
                had_tk_bear = False
                had_loss = False
                start = max(MOVE_K, j - RETEST_LOOKBACK)
                for t in range(start, j):
                    if tenkan[t] < kijun[t] and established_bear(df, t):
                        had_tk_bear = True
                    if had_tk_bear and tenkan[t] >= kijun[t]:
                        had_loss = True
                        break
                ok, best_m, best_ext, _ = _prior_impulse(df, j, bull=False)
                if had_loss and ok:
                    events.append(
                        B0Event(
                            variant="B0C",
                            direction="bear",
                            tf=tf,
                            idx=j,
                            timestamp=str(row["timestamp"]),
                            entry_px=c,
                            move_m_atr=float(best_m),
                            ext_atr=float(best_ext),
                            deepest_pull_atr=float("nan"),
                            penetration_atr=float("nan"),
                            touch_class="tk_recross",
                            bars_near_structure=0,
                            dist_to_structure_atr=float((kijun[j] - tenkan[j]) / atrv),
                            kumo_top=top,
                            kumo_bot=bot,
                            kijun=float(kijun[j]),
                            tenkan=float(tenkan[j]),
                            atr14=atrv,
                        )
                    )
                    last_fire["bear"] = j
    return events


DETECTORS = {
    "B0A": detect_b0a,
    "B0B": detect_b0b,
    "B0C": detect_b0c,
}


def events_to_frame(events: list[B0Event]) -> pd.DataFrame:
    if not events:
        return pd.DataFrame()
    return pd.DataFrame([asdict(e) for e in events])


def attach_event_forwards(
    panel: pd.DataFrame,
    events: list[B0Event],
    horizons: tuple[int, ...] = (1, 3, 6, 12, 24),
    mfe_horizon: int = 24,
) -> pd.DataFrame:
    """Attach signed forward returns (bear flips sign) + MFE/MAE path stats."""
    rows: list[dict[str, Any]] = []
    c = panel["close"].astype(float).values
    h = panel["high"].astype(float).values
    l = panel["low"].astype(float).values
    n = len(panel)
    for ev in events:
        i = ev.idx
        px = float(ev.entry_px)
        sign = 1.0 if ev.direction == "bull" else -1.0
        row: dict[str, Any] = {**asdict(ev)}
        for hz in horizons:
            if i + hz < n and px > 0:
                raw = float(c[i + hz]) / px - 1.0
                row[f"fwd_{hz}"] = sign * raw
            else:
                row[f"fwd_{hz}"] = float("nan")
        # MFE / MAE over next mfe_horizon bars (signed)
        end = min(n, i + 1 + mfe_horizon)
        if end > i + 1 and px > 0:
            if ev.direction == "bull":
                mfe = float(np.nanmax(h[i + 1 : end]) / px - 1.0)
                mae = float(np.nanmin(l[i + 1 : end]) / px - 1.0)
            else:
                # Bear: favorable = price fall; adverse = price rise
                mfe = float(1.0 - np.nanmin(l[i + 1 : end]) / px)
                mae = float(1.0 - np.nanmax(h[i + 1 : end]) / px)
            row["mfe_pct"] = mfe
            row["mae_pct"] = mae
        else:
            row["mfe_pct"] = float("nan")
            row["mae_pct"] = float("nan")
        # unsigned continuation = signed fwd_6 > 0
        row["continuation_h6"] = bool(row.get("fwd_6") is not None and np.isfinite(row.get("fwd_6")) and row["fwd_6"] > 0)
        rows.append(row)
    return pd.DataFrame(rows)


def detect_impulses_without_retest(
    df: pd.DataFrame,
    *,
    bull: bool,
    events: list[B0Event],
) -> pd.DataFrame:
    """Impulse bars (established + meaningful M) that never receive a matching B0 retest.

    Opportunity-cost diagnostic for waiting on retest.
    """
    fired_dirs = {(e.idx, e.direction) for e in events}
    # Also mark any B0 of same direction in [impulse, impulse+RETEST_LOOKBACK]
    b0_by_dir = [e for e in events if (e.direction == "bull") == bull]
    rows = []
    n = len(df)
    cooldown = -10**9
    for t in range(MOVE_K, n):
        if t - cooldown <= COOLDOWN_BARS:
            continue
        if not _valid_row(df, t):
            continue
        est = established_bull(df, t) if bull else established_bear(df, t)
        if not est:
            continue
        m = _move_m(df, t)
        atrv = float(df.iloc[t]["atr14"])
        c = float(df.iloc[t]["close"])
        if bull:
            ext = (c - float(df.iloc[t]["kumo_top"])) / atrv
        else:
            ext = (float(df.iloc[t]["kumo_bot"]) - c) / atrv
        if not (np.isfinite(m) and m >= MOVE_MIN_ATR and np.isfinite(ext) and ext >= EXT_MIN_ATR):
            continue
        # Did a B0 fire in (t, t+RETEST_LOOKBACK]?
        got = False
        for e in b0_by_dir:
            if t < e.idx <= t + RETEST_LOOKBACK:
                got = True
                break
        if got:
            cooldown = t
            continue
        # Missed retest opportunity
        sign = 1.0 if bull else -1.0
        fwd6 = float("nan")
        if t + 6 < n and c > 0:
            fwd6 = sign * (float(df.iloc[t + 6]["close"]) / c - 1.0)
        rows.append(
            {
                "direction": "bull" if bull else "bear",
                "idx": t,
                "timestamp": str(df.iloc[t]["timestamp"]),
                "move_m_atr": m,
                "ext_atr": ext,
                "fwd_6_signed": fwd6,
                "missed_retest": True,
            }
        )
        cooldown = t
    return pd.DataFrame(rows)


def definitions_dict() -> dict[str, Any]:
    return {
        "established_bull": "close > kumo_top AND kumo_bullish (senkou_a > senkou_b)",
        "established_bear": "close < kumo_bot AND NOT kumo_bullish",
        "meaningful_move": f"M=|C_t-C_{{t-{MOVE_K}}}|/ATR14_t >= {MOVE_MIN_ATR} AND extension_from_kumo >= {EXT_MIN_ATR} ATR",
        "retest_lookback": RETEST_LOOKBACK,
        "near_atr": NEAR_ATR,
        "min_established_bars_in_lookback": MIN_EST_BARS,
        "cooldown_bars": COOLDOWN_BARS,
        "entry": "immediate at confirmation close",
        "variants": {
            "B0A": "Kumo boundary touch/near + close reclaim",
            "B0B": "Kijun touch/near + close reclaim, Kumo regime intact",
            "B0C": "TK recross after correction inside established Kumo trend",
        },
    }
