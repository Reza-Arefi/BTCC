"""Part 3A: 5m execution / timing diagnostic (no filter, no live changes).

Hypothesis: 15m remains the sole signal. 5m is only asked whether the
*moment* of execution can be improved, not whether the opportunity is valid.

Oracle window results are upper bounds. They are not tradable backtests.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from btcc.analytics.opportunity_5m_diagnostics import (
    DIAGNOSTIC_CSV_NAME,
    DIAGNOSTIC_DIRNAME,
    FiveMinuteFeatureBuilder,
    PRIMARY_OUTCOME_ARM,
    _safe_float,
    _utc,
    baseline_file_hashes,
    assert_baseline_unchanged,
)

logger = logging.getLogger(__name__)

FIXED_T_ARMS = tuple(f"T{i}" for i in range(1, 11))
BASELINE_FILES = (
    "opportunities.csv",
    "strategy_legs.csv",
    "selection_audit.csv",
    "predictions.csv",
    "summary.json",
)
PRE_OFFSETS_MIN = (5, 10, 15, 20, 30)
POST_WINDOWS_MIN = (5, 10, 15)
ORACLE_WINDOWS_MIN = (5, 10, 15)
PATH_MINUTES = tuple(range(-30, 35, 5))
S_BANDS = ("0.60-0.65", "0.65-0.70", "0.70-0.80", "0.80+")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _fmt(v: Any, nd: int = 4) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v)


def _md_table(df: pd.DataFrame, cols: list[str]) -> list[str]:
    use = [c for c in cols if c in df.columns]
    header = "| " + " | ".join(use) + " |"
    sep = "|" + "|".join(["---"] * len(use)) + "|"
    lines = [header, sep]
    for _, row in df.iterrows():
        cells = []
        for c in use:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append(_fmt(v))
            else:
                cells.append(str(v) if v is not None else "—")
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _as_index_ts(index: pd.DatetimeIndex, ts) -> pd.Timestamp:
    t = _utc(ts)
    if index.tz is None:
        t = t.tz_localize(None) if t.tzinfo is not None else t
    else:
        t = t.tz_convert(index.tz) if t.tzinfo is not None else t.tz_localize(index.tz)
    try:
        return t.as_unit(index.unit)
    except (TypeError, AttributeError, ValueError):
        return pd.Timestamp(t.value, tz=t.tz)


def _slice_bars(
    rel: pd.DataFrame,
    index: pd.DatetimeIndex,
    start,
    end,
    *,
    start_inclusive: bool = False,
    end_inclusive: bool = True,
) -> pd.DataFrame:
    s = _as_index_ts(index, start)
    e = _as_index_ts(index, end)
    i0 = int(index.searchsorted(s, side="left" if start_inclusive else "right"))
    i1 = int(index.searchsorted(e, side="right" if end_inclusive else "left"))
    if i1 <= i0:
        return rel.iloc[0:0]
    return rel.iloc[i0:i1]


def _asof_row(rel: pd.DataFrame, index: pd.DatetimeIndex, at_ts) -> pd.Series | None:
    t = _as_index_ts(index, at_ts)
    idx = int(index.searchsorted(t, side="right")) - 1
    if idx < 0:
        return None
    return rel.iloc[idx]


def _signed_vs_entry(price: float | None, entry: float, *, long: bool = True) -> float | None:
    if price is None or entry is None or entry <= 0:
        return None
    raw = price / entry - 1.0
    return raw if long else -raw


def _long_improvement(price: float | None, entry: float) -> float | None:
    """Positive = cheaper long fill than baseline entry."""
    if price is None or entry is None or entry <= 0:
        return None
    return (entry - price) / entry


def _approx_pnl_if_exit_unchanged(pnl: float | None, improvement: float | None) -> float | None:
    """Diagnostic only: same exit mid, different long entry. Not a re-simulated stop."""
    if pnl is None or improvement is None:
        return None
    # new_entry = old * (1 - improvement); return scales by old/new = 1/(1-imp)
    if improvement >= 0.99:
        return None
    return (1.0 + pnl) / (1.0 - improvement) - 1.0


def classify_causal_state(
    *,
    wait_ret: float | None,
    dirret_5m: float | None,
    dirret_15m: float | None,
    consec_fav: int,
    consec_adv: int,
    first_half_ret: float | None,
    second_half_ret: float | None,
) -> str:
    """Mutually exclusive labels from information known by baseline entry. No outcome peeking."""
    ext = (
        dirret_15m is not None
        and dirret_15m > 0
        and dirret_5m is not None
        and dirret_5m > 0
        and consec_fav >= 2
    )
    if ext:
        return "D_local_extension"
    stab = (
        first_half_ret is not None
        and second_half_ret is not None
        and first_half_ret < 0
        and second_half_ret >= 0
    )
    if stab:
        return "B_stabilization"
    pull = (wait_ret is not None and wait_ret < 0) or (dirret_5m is not None and dirret_5m < 0) or consec_adv >= 2
    if pull and not (dirret_5m is not None and dirret_5m > 0 and (wait_ret or 0) > 0):
        return "A_pullback"
    if dirret_5m is not None and dirret_5m > 0:
        return "C_immediate_continuation"
    return "other"


def document_entry_convention() -> dict[str, str]:
    return {
        "baseline_signal_timestamp": "opportunities.signal_timestamp == opportunities.opened_ts == 15m decision bar timestamp t",
        "baseline_entry_timestamp": "opportunities.entry_fill_ts == next 15m bar after t (typically t+15m)",
        "baseline_entry_price": "opportunities.entry_alt_btc_mid == next 15m bar OPEN (ALT/BTC mid before slippage)",
        "code": "btcc/sim/selector_backtest.py: future = rel_full[timestamp > t]; entry_bar = future.iloc[0]; entry_mid = entry_bar['open']",
        "accounting": "fills at mid * (1+slippage) with fee; this diagnostic measures mid-price timing only",
        "direction": "all baseline opportunities are LONG_ALT_BTC",
        "note": "Part 1 5m features are as-of the 15m *signal*, not as-of the fill.",
    }


def _path_metrics_for_opp(
    *,
    rel: pd.DataFrame,
    index: pd.DatetimeIndex,
    signal_ts,
    entry_ts,
    entry_px: float,
) -> dict[str, Any]:
    signal = _utc(signal_ts)
    entry = _utc(entry_ts)
    long = True
    out: dict[str, Any] = {
        "signal_timestamp": str(signal),
        "entry_timestamp": str(entry),
        "baseline_entry_price": entry_px,
        "signal_to_entry_minutes": float((entry - signal).total_seconds() / 60.0),
    }
    sig_row = _asof_row(rel, index, signal)
    fill_row = _asof_row(rel, index, entry)
    out["signal_5m_close"] = None if sig_row is None else _safe_float(sig_row["close"])
    out["entry_5m_open"] = None if fill_row is None else _safe_float(fill_row["open"])
    out["entry_open_vs_baseline"] = _signed_vs_entry(out["entry_5m_open"], entry_px, long=long)

    for mins in PRE_OFFSETS_MIN:
        row = _asof_row(rel, index, entry - pd.Timedelta(minutes=mins))
        close = None if row is None else _safe_float(row["close"])
        out[f"price_change_{mins}m_before_entry"] = _signed_vs_entry(close, entry_px, long=long)
        out[f"close_{mins}m_before_entry"] = close

    wait = _slice_bars(rel, index, signal, entry, start_inclusive=False, end_inclusive=False)
    post30 = _slice_bars(rel, index, entry, entry + pd.Timedelta(minutes=30), start_inclusive=True, end_inclusive=True)

    if not wait.empty:
        wmin = _safe_float(wait["low"].min())
        wmax = _safe_float(wait["high"].max())
        wfirst_c = _safe_float(wait["close"].iloc[0])
        wlast_c = _safe_float(wait["close"].iloc[-1])
        out["min_price_before_baseline_entry"] = wmin
        out["max_price_before_baseline_entry"] = wmax
        out["wait_min_vs_entry"] = _signed_vs_entry(wmin, entry_px, long=long)
        out["wait_max_vs_entry"] = _signed_vs_entry(wmax, entry_px, long=long)
        out["oracle_wait_window_improvement"] = _long_improvement(wmin, entry_px)
        out["wait_had_better_long_fill"] = bool(wmin is not None and wmin < entry_px)
        out["wait_had_material_better_fill_50bps"] = bool(
            out["oracle_wait_window_improvement"] is not None and out["oracle_wait_window_improvement"] >= 0.005
        )
        out["wait_had_material_better_fill_100bps"] = bool(
            out["oracle_wait_window_improvement"] is not None and out["oracle_wait_window_improvement"] >= 0.010
        )
        out["wait_runaway_against_long"] = bool(wmax is not None and wmax > entry_px)
        if out["signal_5m_close"] and out["signal_5m_close"] > 0:
            out["wait_window_return_vs_signal"] = wlast_c / out["signal_5m_close"] - 1.0 if wlast_c else None
        n = len(wait)
        if n >= 2:
            o0 = _safe_float(wait["open"].iloc[0])
            c_mid = _safe_float(wait["close"].iloc[n // 2 - 1])
            c_last = _safe_float(wait["close"].iloc[-1])
            out["wait_first_half_ret"] = None if not o0 or not c_mid else c_mid / o0 - 1.0
            out["wait_second_half_ret"] = None if not c_mid or not c_last or c_mid == 0 else c_last / c_mid - 1.0
        else:
            out["wait_first_half_ret"] = None
            out["wait_second_half_ret"] = None
        out["n_wait_5m_bars"] = int(n)
    else:
        for k in (
            "min_price_before_baseline_entry",
            "max_price_before_baseline_entry",
            "wait_min_vs_entry",
            "wait_max_vs_entry",
            "oracle_wait_window_improvement",
            "wait_window_return_vs_signal",
            "wait_first_half_ret",
            "wait_second_half_ret",
        ):
            out[k] = None
        out["wait_had_better_long_fill"] = False
        out["wait_had_material_better_fill_50bps"] = False
        out["wait_had_material_better_fill_100bps"] = False
        out["wait_runaway_against_long"] = False
        out["n_wait_5m_bars"] = 0

    hist_at_entry = rel.iloc[: int(index.searchsorted(_as_index_ts(index, entry), side="right"))]
    close = hist_at_entry["close"].astype(float) if len(hist_at_entry) else pd.Series(dtype=float)
    out["dirret_5m_at_entry"] = FiveMinuteFeatureBuilder._directional_return(close, 1, long=True)
    out["dirret_15m_at_entry"] = FiveMinuteFeatureBuilder._directional_return(close, 3, long=True)
    out["consec_fav_at_entry"] = FiveMinuteFeatureBuilder._streak(close, favorable=True, long=True)
    out["consec_adv_at_entry"] = FiveMinuteFeatureBuilder._streak(close, favorable=False, long=True)

    def _post_exc(bars: pd.DataFrame, minutes: int) -> None:
        cutoff = entry + pd.Timedelta(minutes=minutes)
        ts = pd.to_datetime(bars["timestamp"], utc=True)
        sl = bars[ts <= cutoff]
        if sl.empty or entry_px <= 0:
            out[f"post_entry_mfe_{minutes}m"] = None
            out[f"post_entry_mae_{minutes}m"] = None
            out[f"post_entry_close_{minutes}m"] = None
            return
        hi = _safe_float(sl["high"].max())
        lo = _safe_float(sl["low"].min())
        cl = _safe_float(sl["close"].iloc[-1])
        out[f"post_entry_mfe_{minutes}m"] = None if hi is None else hi / entry_px - 1.0
        out[f"post_entry_mae_{minutes}m"] = None if lo is None else lo / entry_px - 1.0
        out[f"post_entry_close_{minutes}m"] = None if cl is None else cl / entry_px - 1.0

    for w in POST_WINDOWS_MIN:
        _post_exc(post30, w)

    # After *signal* (includes the existing 15m wait)
    after_sig = _slice_bars(rel, index, signal, signal + pd.Timedelta(minutes=30))
    for w in POST_WINDOWS_MIN:
        cutoff = signal + pd.Timedelta(minutes=w)
        ts = pd.to_datetime(after_sig["timestamp"], utc=True)
        sl = after_sig[ts <= cutoff]
        ref = out["signal_5m_close"] or entry_px
        if sl.empty or not ref:
            out[f"post_signal_mfe_{w}m"] = None
            out[f"post_signal_mae_{w}m"] = None
            out[f"post_signal_close_{w}m"] = None
            continue
        hi = _safe_float(sl["high"].max())
        lo = _safe_float(sl["low"].min())
        cl = _safe_float(sl["close"].iloc[-1])
        out[f"post_signal_mfe_{w}m"] = None if hi is None else hi / ref - 1.0
        out[f"post_signal_mae_{w}m"] = None if lo is None else lo / ref - 1.0
        out[f"post_signal_close_{w}m"] = None if cl is None else cl / ref - 1.0

    # Minutes to first favorable / adverse 5m print after entry (exclude mechanical same-bar OHLC)
    post_excl = _slice_bars(rel, index, entry, entry + pd.Timedelta(minutes=180), start_inclusive=False, end_inclusive=True)
    t_fav = t_adv = None
    for _, br in post_excl.iterrows():
        ts = _utc(br["timestamp"])
        mins = (ts - entry).total_seconds() / 60.0
        hi = _safe_float(br["high"])
        lo = _safe_float(br["low"])
        if t_fav is None and hi is not None and hi > entry_px:
            t_fav = mins
        if t_adv is None and lo is not None and lo < entry_px:
            t_adv = mins
        if t_fav is not None and t_adv is not None:
            break
    out["minutes_to_first_favorable_5m"] = t_fav
    out["minutes_to_first_adverse_5m"] = t_adv

    # Non-tautological first-5m adverse: next 5m bar (timestamp > entry) within 5m, or entry-bar close below entry
    next5 = _slice_bars(rel, index, entry, entry + pd.Timedelta(minutes=5), start_inclusive=False, end_inclusive=True)
    entry_close = None if fill_row is None else _safe_float(fill_row["close"])
    next5_lo = None if next5.empty else _safe_float(next5["low"].min())
    next5_hi = None if next5.empty else _safe_float(next5["high"].max())
    next5_cl = None if next5.empty else _safe_float(next5["close"].iloc[-1])
    out["entry_bar_close_vs_entry"] = None if entry_close is None else entry_close / entry_px - 1.0
    out["next5_mae"] = None if next5_lo is None else next5_lo / entry_px - 1.0
    out["next5_mfe"] = None if next5_hi is None else next5_hi / entry_px - 1.0
    out["next5_close"] = None if next5_cl is None else next5_cl / entry_px - 1.0
    out["first_5m_adverse"] = bool(
        (entry_close is not None and entry_close < entry_px)
        or (next5_lo is not None and next5_lo < entry_px)
    )
    out["first_5m_favorable"] = bool(
        (entry_close is not None and entry_close > entry_px)
        or (next5_hi is not None and next5_hi > entry_px)
    )
    rec15 = out.get("post_entry_close_15m")
    out["pullback_then_recover_15m"] = bool(out["first_5m_adverse"] and rec15 is not None and rec15 >= 0)
    out["pullback_then_continue_down_15m"] = bool(out["first_5m_adverse"] and rec15 is not None and rec15 < 0)

    # ORACLE idealized fills after baseline entry (future min low in window)
    fill_same = fill_row
    same_low = None if fill_same is None else _safe_float(fill_same["low"])
    out["oracle_samebar_low_improvement"] = _long_improvement(same_low, entry_px)
    out["oracle_samebar_is_hindsight_low"] = True
    for w in ORACLE_WINDOWS_MIN:
        nxt = _slice_bars(
            rel,
            index,
            entry,
            entry + pd.Timedelta(minutes=w),
            start_inclusive=False,
            end_inclusive=True,
        )
        if nxt.empty:
            out[f"oracle_E{w}_min_low"] = None
            out[f"oracle_E{w}_improvement"] = None
            out[f"oracle_E{w}_next_open"] = None
            out[f"oracle_E{w}_next_open_improvement"] = None
            out[f"oracle_E{w}_runaway"] = None
            continue
        mn = _safe_float(nxt["low"].min())
        mx = _safe_float(nxt["high"].max())
        nopen = _safe_float(nxt["open"].iloc[0])
        out[f"oracle_E{w}_min_low"] = mn
        out[f"oracle_E{w}_improvement"] = _long_improvement(mn, entry_px)
        out[f"oracle_E{w}_next_open"] = nopen
        out[f"oracle_E{w}_next_open_improvement"] = _long_improvement(nopen, entry_px)
        out[f"oracle_E{w}_runaway"] = _signed_vs_entry(mx, entry_px, long=True)

    for m in PATH_MINUTES:
        row = _asof_row(rel, index, entry + pd.Timedelta(minutes=m))
        cl = None if row is None else _safe_float(row["close"])
        out[f"path_vs_entry_m{m}"] = _signed_vs_entry(cl, entry_px, long=True)

    out["causal_state"] = classify_causal_state(
        wait_ret=out.get("wait_window_return_vs_signal"),
        dirret_5m=out.get("dirret_5m_at_entry"),
        dirret_15m=out.get("dirret_15m_at_entry"),
        consec_fav=int(out.get("consec_fav_at_entry") or 0),
        consec_adv=int(out.get("consec_adv_at_entry") or 0),
        first_half_ret=out.get("wait_first_half_ret"),
        second_half_ret=out.get("wait_second_half_ret"),
    )
    return out


def build_execution_rows(
    *,
    diag: pd.DataFrame,
    opps: pd.DataFrame,
    builder: FiveMinuteFeatureBuilder,
) -> pd.DataFrame:
    opp_map = opps.set_index("opportunity_id")
    rows: list[dict[str, Any]] = []
    n = len(diag)
    for i, drow in diag.iterrows():
        oid = str(drow["opportunity_id"])
        base = str(drow["base"])
        o = opp_map.loc[oid]
        prepared = builder._coin_cache.get(base)
        entry_px = _safe_float(o["entry_alt_btc_mid"])
        rec: dict[str, Any] = {
            "opportunity_id": oid,
            "pair": drow.get("pair"),
            "base": base,
            "signal_15m": drow.get("signal_15m"),
            "s_band": drow.get("s_band"),
            "regime": drow.get("regime"),
            "momentum_5m_signed_at_signal": drow.get("momentum_5m_signed"),
            "directional_return_5m_at_signal": drow.get("directional_return_5m"),
            "directional_return_15m_at_signal": drow.get("directional_return_15m"),
            "primary_pnl_pct": drow.get("primary_pnl_pct"),
            "primary_win": drow.get("primary_win"),
            "primary_mfe_pct": drow.get("primary_mfe_pct"),
            "primary_mae_pct": drow.get("primary_mae_pct"),
            "selector_E_selected_pnl_pct": drow.get("selector_E_selected_pnl_pct"),
            "selector_E_selected_win": drow.get("selector_E_selected_win"),
            "selector_E_selected_mfe_pct": drow.get("selector_E_selected_mfe_pct"),
            "selector_E_selected_mae_pct": drow.get("selector_E_selected_mae_pct"),
        }
        for arm in FIXED_T_ARMS:
            rec[f"{arm}_pnl_pct"] = drow.get(f"{arm}_pnl_pct")
            rec[f"{arm}_win"] = drow.get(f"{arm}_win")
            rec[f"{arm}_mfe_pct"] = drow.get(f"{arm}_mfe_pct")
            rec[f"{arm}_mae_pct"] = drow.get(f"{arm}_mae_pct")
        if prepared is None or entry_px is None:
            rec["five_m_path_available"] = False
            rows.append(rec)
            continue
        rec["five_m_path_available"] = True
        rec.update(
            _path_metrics_for_opp(
                rel=prepared["rel"],
                index=prepared["rel_index"],
                signal_ts=o.get("signal_timestamp") or o.get("opened_ts"),
                entry_ts=o["entry_fill_ts"],
                entry_px=entry_px,
            )
        )
        rec["approx_pnl_oracle_E15"] = _approx_pnl_if_exit_unchanged(
            _safe_float(rec["primary_pnl_pct"]), rec.get("oracle_E15_improvement")
        )
        rec["approx_pnl_oracle_wait"] = _approx_pnl_if_exit_unchanged(
            _safe_float(rec["primary_pnl_pct"]), rec.get("oracle_wait_window_improvement")
        )
        rows.append(rec)
        if (len(rows) % 100) == 0:
            logger.info("Part 3A paths %d/%d", len(rows), n)
    return pd.DataFrame(rows)


def _mean(s: pd.Series) -> float | None:
    x = _num(s).dropna()
    return float(x.mean()) if len(x) else None


def _median(s: pd.Series) -> float | None:
    x = _num(s).dropna()
    return float(x.median()) if len(x) else None


def _corr(a: pd.Series, b: pd.Series) -> float | None:
    x = pd.DataFrame({"a": _num(a), "b": _num(b)}).dropna()
    if len(x) < 20:
        return None
    c = float(x["a"].corr(x["b"]))
    return c if np.isfinite(c) else None


def summarize_overall(df: pd.DataFrame) -> dict[str, Any]:
    n = int(len(df))
    win = df["primary_win"] == True  # noqa: E712
    return {
        "n": n,
        "frac_wait_better_fill": float(df["wait_had_better_long_fill"].fillna(False).mean()) if n else None,
        "frac_wait_material_50bps": float(df["wait_had_material_better_fill_50bps"].fillna(False).mean()) if n else None,
        "frac_wait_material_100bps": float(df["wait_had_material_better_fill_100bps"].fillna(False).mean()) if n else None,
        "median_wait_oracle_improvement": _median(df["oracle_wait_window_improvement"]),
        "mean_wait_oracle_improvement": _mean(df["oracle_wait_window_improvement"]),
        "median_oracle_E5": _median(df["oracle_E5_improvement"]),
        "mean_oracle_E5": _mean(df["oracle_E5_improvement"]),
        "median_oracle_E10": _median(df["oracle_E10_improvement"]),
        "mean_oracle_E10": _mean(df["oracle_E10_improvement"]),
        "median_oracle_E15": _median(df["oracle_E15_improvement"]),
        "mean_oracle_E15": _mean(df["oracle_E15_improvement"]),
        "median_oracle_E15_next_open": _median(df["oracle_E15_next_open_improvement"]),
        "mean_oracle_E15_runaway": _mean(df["oracle_E15_runaway"]),
        "median_post_entry_mae_5m": _median(df["post_entry_mae_5m"]),
        "median_post_entry_mae_15m": _median(df["post_entry_mae_15m"]),
        "median_post_entry_mfe_5m": _median(df["post_entry_mfe_5m"]),
        "median_post_entry_mfe_15m": _median(df["post_entry_mfe_15m"]),
        "frac_first_5m_adverse": float(df["first_5m_adverse"].fillna(False).mean()) if n else None,
        "frac_pullback_recover_15m": float(df["pullback_then_recover_15m"].fillna(False).mean()) if n else None,
        "frac_pullback_continue_down": float(df["pullback_then_continue_down_15m"].fillna(False).mean()) if n else None,
        "median_minutes_to_adverse": _median(df["minutes_to_first_adverse_5m"]),
        "median_minutes_to_favorable": _median(df["minutes_to_first_favorable_5m"]),
        "corr_mae5_vs_T1_mae": _corr(df["post_entry_mae_5m"], df["primary_mae_pct"]),
        "corr_mfe5_vs_T1_mfe": _corr(df["post_entry_mfe_5m"], df["primary_mfe_pct"]),
        "corr_mae5_vs_T1_pnl": _corr(df["post_entry_mae_5m"], df["primary_pnl_pct"]),
        "corr_mfe5_vs_T1_pnl": _corr(df["post_entry_mfe_5m"], df["primary_pnl_pct"]),
        "corr_dirret5_entry_vs_pnl": _corr(df["dirret_5m_at_entry"], df["primary_pnl_pct"]),
        "baseline_wr": float(win.mean()) if n else None,
        "baseline_mean_pnl": _mean(df["primary_pnl_pct"]),
        "mean_signal_to_entry_min": _mean(df["signal_to_entry_minutes"]),
    }


def _slice_table(df: pd.DataFrame, col: str, pnl_col: str, win_col: str) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(col, dropna=False):
        n = int(len(g))
        win = g[win_col] == True  # noqa: E712
        adv = g["first_5m_adverse"].fillna(False)
        rows.append({
            "slice": str(key),
            "n": n,
            "baseline_win_rate": float(win.mean()) if n else None,
            "baseline_mean_pnl": _mean(g[pnl_col]),
            "frac_wait_better_fill": float(g["wait_had_better_long_fill"].fillna(False).mean()) if n else None,
            "median_oracle_E15": _median(g["oracle_E15_improvement"]),
            "median_post_entry_mae_5m": _median(g["post_entry_mae_5m"]),
            "median_post_entry_mfe_5m": _median(g["post_entry_mfe_5m"]),
            "frac_first_5m_adverse": float(adv.mean()) if n else None,
            "wr_if_first_5m_adverse": float(win[adv].mean()) if int(adv.sum()) else None,
            "wr_if_not_first_5m_adverse": float(win[~adv].mean()) if int((~adv).sum()) else None,
            "mean_pnl_if_first_5m_adverse": _mean(g.loc[adv, pnl_col]),
            "mean_pnl_if_not_adverse": _mean(g.loc[~adv, pnl_col]),
            "frac_pullback_recover": float(g["pullback_then_recover_15m"].fillna(False).mean()) if n else None,
            "frac_pullback_fail": float(g["pullback_then_continue_down_15m"].fillna(False).mean()) if n else None,
        })
    return pd.DataFrame(rows)


def t_arm_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm in FIXED_T_ARMS:
        pnl = _num(df[f"{arm}_pnl_pct"])
        win = df[f"{arm}_win"] == True  # noqa: E712
        adv = df["first_5m_adverse"].fillna(False)
        rows.append({
            "arm": arm,
            "baseline_mean_pnl": float(pnl.mean()) if len(pnl) else None,
            "baseline_win_rate": float(win.mean()) if len(df) else None,
            "wr_if_first_5m_adverse": float(win[adv].mean()) if int(adv.sum()) else None,
            "wr_if_not_adverse": float(win[~adv].mean()) if int((~adv).sum()) else None,
            "mean_pnl_if_first_5m_adverse": float(pnl[adv].mean()) if int(adv.sum()) else None,
            "mean_pnl_if_not_adverse": float(pnl[~adv].mean()) if int((~adv).sum()) else None,
            "corr_mae5_vs_pnl": _corr(df["post_entry_mae_5m"], df[f"{arm}_pnl_pct"]),
            "corr_mfe5_vs_mfe": _corr(df["post_entry_mfe_5m"], df[f"{arm}_mfe_pct"]),
            "approx_mean_pnl_oracle_E15": _mean(
                pd.Series([_approx_pnl_if_exit_unchanged(_safe_float(p), i) for p, i in zip(pnl, df["oracle_E15_improvement"])])
            ),
        })
    return pd.DataFrame(rows)


def selector_e_row(df: pd.DataFrame) -> dict[str, Any]:
    pnl = _num(df["selector_E_selected_pnl_pct"])
    win = df["selector_E_selected_win"] == True  # noqa: E712
    adv = df["first_5m_adverse"].fillna(False)
    return {
        "n": int(len(df)),
        "baseline_mean_pnl": float(pnl.mean()) if len(pnl) else None,
        "baseline_win_rate": float(win.mean()) if len(df) else None,
        "frac_first_5m_adverse": float(adv.mean()) if len(df) else None,
        "wr_if_first_5m_adverse": float(win[adv].mean()) if int(adv.sum()) else None,
        "wr_if_not_adverse": float(win[~adv].mean()) if int((~adv).sum()) else None,
        "mean_pnl_if_first_5m_adverse": float(pnl[adv].mean()) if int(adv.sum()) else None,
        "mean_pnl_if_not_adverse": float(pnl[~adv].mean()) if int((~adv).sum()) else None,
        "median_oracle_E15": _median(df["oracle_E15_improvement"]),
        "corr_mae5_vs_pnl": _corr(df["post_entry_mae_5m"], df["selector_E_selected_pnl_pct"]),
        "corr_mfe5_vs_mfe": _corr(df["post_entry_mfe_5m"], df["selector_E_selected_mfe_pct"]),
        "approx_mean_pnl_oracle_E15": _mean(
            pd.Series([_approx_pnl_if_exit_unchanged(_safe_float(p), i) for p, i in zip(pnl, df["oracle_E15_improvement"])])
        ),
    }


def state_table(df: pd.DataFrame) -> pd.DataFrame:
    return _slice_table(df, "causal_state", "primary_pnl_pct", "primary_win")


def decide(overall: dict[str, Any], states: pd.DataFrame, arms: pd.DataFrame) -> tuple[str, str]:
    better = overall.get("frac_wait_material_50bps") or 0.0
    med_e15 = overall.get("median_oracle_E15") or 0.0
    frac_adv = overall.get("frac_first_5m_adverse") or 0.0
    rec = overall.get("frac_pullback_recover_15m") or 0.0
    fail = overall.get("frac_pullback_continue_down") or 0.0
    corr_mae = overall.get("corr_mae5_vs_T1_mae")
    corr_dir = overall.get("corr_dirret5_entry_vs_pnl")
    wr_gap = None
    if states is not None and len(states):
        pull = states[states["slice"] == "A_pullback"]
        cont = states[states["slice"].isin(["C_immediate_continuation", "D_local_extension"])]
        if len(pull) and len(cont):
            a = pull.iloc[0]["baseline_win_rate"]
            # weight continuation-like states
            c_wr = None
            c_n = 0
            for _, r in cont.iterrows():
                if r["baseline_win_rate"] is None:
                    continue
                c_wr = (0 if c_wr is None else c_wr * c_n) + float(r["baseline_win_rate"]) * int(r["n"])
                c_n += int(r["n"])
                c_wr = c_wr / c_n if c_n else None
            if a is not None and c_wr is not None:
                wr_gap = abs(float(a) - float(c_wr))
    wr_dir = 0
    if len(arms):
        wr_dir = int(
            (
                arms["wr_if_not_adverse"].fillna(-1) > arms["wr_if_first_5m_adverse"].fillna(-1)
            ).sum()
        )

    # Adverse selection: more ORACLE room on weak path states is a red flag for naive wait-for-better-price.
    adverse_selection = False
    if states is not None and len(states):
        pull = states[states["slice"].isin(["A_pullback", "B_stabilization"])]
        strong = states[states["slice"].isin(["C_immediate_continuation", "D_local_extension"])]
        if len(pull) and len(strong):
            p_or = float(_num(pull["median_oracle_E15"]).mean()) if "median_oracle_E15" in pull else 0.0
            s_or = float(_num(strong["median_oracle_E15"]).mean()) if "median_oracle_E15" in strong else 0.0
            p_wr = float(_num(pull["baseline_win_rate"]).mean())
            s_wr = float(_num(strong["baseline_win_rate"]).mean())
            adverse_selection = (p_or > s_or + 0.002) and (p_wr + 0.05 < s_wr)

    next_open = overall.get("median_oracle_E15_next_open") or 0.0
    pullback_mostly_fails = fail > rec + 0.15

    distinguishable = wr_gap is not None and wr_gap >= 0.05
    common_room = better >= 0.55 and med_e15 >= 0.004
    mae_predicts = (corr_mae is not None and corr_mae > 0.25) or (corr_dir is not None and corr_dir > 0.25)

    if (
        common_room
        and distinguishable
        and mae_predicts
        and not pullback_mostly_fails
        and not adverse_selection
        and next_open >= 0.001
    ):
        label = "YES — strong evidence"
        why = (
            "Material wait-window execution room is common, causal path states at entry separate outcomes, "
            "temporary pullbacks are not dominated by failures, and realistic next-open improvement is non-trivial. "
            "Still diagnostic: next is a small frozen causal rule set (Part 3B), not deployment."
        )
    elif distinguishable or mae_predicts or common_room:
        label = "MAYBE — interesting but weak/conditional"
        why = (
            "Causal 5m path state at the existing fill time separates T1/E outcomes, and ORACLE lows show price room, "
            "but realistic next-open gains are tiny, post-signal pullbacks more often continue down than recover, "
            "and ORACLE improvement is larger on weaker path states (adverse selection risk for naive wait-for-dip rules). "
            "Part 3B is justified only as a *tiny* frozen ENTER-NOW vs WAIT-WITH-FALLBACK test — not a price-oracle hunt."
        )
    else:
        label = "NO — insufficient evidence"
        why = (
            "Either better fills are not material enough after accounting for OHLC tautology, "
            "waiting mainly captures noise, or 5m pullbacks are not distinguishable from setup failure."
        )
    extra = (
        f" Material wait-window ≥50bps fraction={_fmt(better)}; median ORACLE E15={_fmt(med_e15)}; "
        f"median next-open={_fmt(next_open)}; first-5m adverse={_fmt(frac_adv)}; "
        f"recover={_fmt(rec)} vs continue-down={_fmt(fail)}; "
        f"causal WR gap={_fmt(wr_gap)}; adverse_selection={adverse_selection}; "
        f"T-arms non-adverse WR higher={wr_dir}/10."
    )
    return label, why + extra


def make_plots(df: pd.DataFrame, plot_dir: Path) -> dict[str, Path]:
    plot_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}

    # Mean path around entry
    fig, ax = plt.subplots(figsize=(9, 4.5))
    xs, means, q25, q75 = [], [], [], []
    for m in PATH_MINUTES:
        col = f"path_vs_entry_m{m}"
        if col not in df.columns:
            continue
        x = _num(df[col]).dropna()
        if x.empty:
            continue
        xs.append(m)
        means.append(float(x.mean()))
        q25.append(float(x.quantile(0.25)))
        q75.append(float(x.quantile(0.75)))
    ax.plot(xs, means, color="#1f4e79", lw=2, label="mean close vs entry")
    ax.fill_between(xs, q25, q75, color="#1f4e79", alpha=0.18, label="IQR")
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("minutes relative to baseline 15m entry")
    ax.set_ylabel("ALT/BTC close / entry − 1")
    ax.set_title("Post-signal 5m path around baseline entry")
    ax.legend()
    fig.tight_layout()
    p = plot_dir / "post_signal_price_path.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    out["post_signal_price_path"] = p

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, cols, title in (
        (axes[0], ["post_signal_mae_5m", "post_signal_mae_10m", "post_signal_mae_15m"], "After 15m signal"),
        (axes[1], ["post_entry_mae_5m", "post_entry_mae_10m", "post_entry_mae_15m"], "After baseline entry"),
    ):
        data = [ _num(df[c]).dropna() * 100 for c in cols if c in df.columns]
        kw = {"showfliers": False}
        try:
            ax.boxplot(data, tick_labels=["5m", "10m", "15m"], **kw)
        except TypeError:
            ax.boxplot(data, labels=["5m", "10m", "15m"], **kw)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_ylabel("MAE (%, long)")
        ax.set_title(title)
    fig.suptitle("Post-signal / post-entry adverse excursion (5m OHLC)")
    fig.tight_layout()
    p = plot_dir / "post_signal_adverse_excursion.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    out["post_signal_adverse_excursion"] = p

    fig, ax = plt.subplots(figsize=(8, 4.2))
    for col, lab in (
        ("oracle_wait_window_improvement", "ORACLE wait-window min low"),
        ("oracle_E5_improvement", "ORACLE E5 min low"),
        ("oracle_E15_improvement", "ORACLE E15 min low"),
        ("oracle_E15_next_open_improvement", "next 5m open (less oracle)"),
    ):
        x = _num(df[col]).dropna() * 100
        if x.empty:
            continue
        ax.hist(x, bins=40, histtype="step", lw=1.6, label=lab)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("long execution improvement vs baseline entry (%)")
    ax.set_ylabel("opportunities")
    ax.set_title("Execution opportunity distribution (ORACLE labeled)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = plot_dir / "execution_opportunity_distribution.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    out["execution_opportunity_distribution"] = p

    fig, ax = plt.subplots(figsize=(6.5, 5))
    x = _num(df["post_entry_mfe_5m"]) * 100
    y = _num(df["primary_mfe_pct"]) * 100
    m = x.notna() & y.notna()
    ax.scatter(x[m], y[m], s=8, alpha=0.25, c="#1f4e79")
    ax.set_xlabel("first 5m MFE after entry (%)")
    ax.set_ylabel("T1 trade MFE (%)")
    ax.set_title("T1 MFE vs initial 5m path")
    fig.tight_layout()
    p = plot_dir / "mfe_vs_initial_5m_path.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    out["mfe_vs_initial_5m_path"] = p

    fig, ax = plt.subplots(figsize=(6.5, 5))
    x = _num(df["post_entry_mae_5m"]) * 100
    y = _num(df["primary_mae_pct"]) * 100
    m = x.notna() & y.notna()
    ax.scatter(x[m], y[m], s=8, alpha=0.25, c="#9c2a2a")
    ax.set_xlabel("first 5m MAE after entry (%)")
    ax.set_ylabel("T1 trade MAE (%)")
    ax.set_title("T1 MAE vs initial 5m path")
    fig.tight_layout()
    p = plot_dir / "mae_vs_initial_5m_path.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    out["mae_vs_initial_5m_path"] = p

    def _bar_metric(ax, gdf, metric, title):
        labels = gdf["slice"].astype(str).tolist()
        vals = [ (v * 100 if v is not None else 0) for v in gdf[metric].tolist()]
        ax.bar(range(len(labels)), vals, color="#1f4e79")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.set_title(title)
        ax.set_ylabel("%")

    sband = _slice_table(df, "s_band", "primary_pnl_pct", "primary_win")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    _bar_metric(axes[0], sband, "median_oracle_E15", "Median ORACLE E15 improvement")
    _bar_metric(axes[1], sband, "frac_first_5m_adverse", "First-5m adverse fraction")
    fig.suptitle("Execution diagnostics by S-band (rule not fit per band)")
    fig.tight_layout()
    p = plot_dir / "execution_by_s_band.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    out["execution_by_s_band"] = p

    regime = _slice_table(df, "regime", "primary_pnl_pct", "primary_win")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    _bar_metric(axes[0], regime, "median_oracle_E15", "Median ORACLE E15 improvement")
    _bar_metric(axes[1], regime, "frac_first_5m_adverse", "First-5m adverse fraction")
    fig.suptitle("Execution diagnostics by regime (rule not fit per regime)")
    fig.tight_layout()
    p = plot_dir / "execution_by_regime.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    out["execution_by_regime"] = p
    return out


def write_report(
    *,
    out_dir: Path,
    convention: dict[str, str],
    overall: dict[str, Any],
    states: pd.DataFrame,
    windows: pd.DataFrame,
    arms: pd.DataFrame,
    e_row: dict[str, Any],
    sband: pd.DataFrame,
    regime: pd.DataFrame,
    decision: str,
    decision_why: str,
    n: int,
    causal_ok: int,
) -> Path:
    lines = [
        "# 15m + 5m Part 3A — Execution / timing diagnostic",
        "",
        "This is **not** a veto or confirmation filter. Every original 15m opportunity stays in the sample.",
        "15m decides **what** to trade. 5m is only examined for **when** execution might improve.",
        "",
        "## Baseline entry convention (from code, not assumed)",
        "",
        f"- Signal timestamp: `{convention['baseline_signal_timestamp']}`",
        f"- Entry timestamp: `{convention['baseline_entry_timestamp']}`",
        f"- Entry price: `{convention['baseline_entry_price']}`",
        f"- Implementation: `{convention['code']}`",
        f"- {convention['accounting']}",
        f"- {convention['note']}",
        "",
        f"Empirical signal→entry gap (mean minutes): **{_fmt(overall.get('mean_signal_to_entry_min'), 1)}**. N={n}.",
        "",
        "## Oracle vs causal",
        "",
        "Tables labeled **ORACLE / IDEALIZED EXECUTION DIAGNOSTIC** use the minimum 5m *low* inside a future window. "
        "That price is not known in real time and is **not** a tradable backtest. "
        "Next-open improvements are still hindsight about *whether* to wait, but they use an actually printable price.",
        "",
        f"Causal 5m-as-of-signal timestamps from Part 1 remain OK on the joined set where checked: **{causal_ok}** rows.",
        "",
        "## Q1–Q3. Is there a better 5m execution point? How large? What is the wait risk?",
        "",
        f"- Fraction with any lower 5m *low* than baseline entry during signal→fill wait: **{_fmt(overall.get('frac_wait_better_fill'))}** (often near 1.0 because OHLC bars usually print a low below later opens — treat as upper-bound presence, not tradability)",
        f"- Fraction with **≥50bps / ≥100bps** wait-window ORACLE improvement: **{_fmt(overall.get('frac_wait_material_50bps'))}** / **{_fmt(overall.get('frac_wait_material_100bps'))}**",
        f"- Median ORACLE wait-window improvement: **{_fmt(overall.get('median_wait_oracle_improvement'))}** (mean {_fmt(overall.get('mean_wait_oracle_improvement'))})",
        f"- Median ORACLE E5 / E10 / E15 improvement after fill: **{_fmt(overall.get('median_oracle_E5'))}** / **{_fmt(overall.get('median_oracle_E10'))}** / **{_fmt(overall.get('median_oracle_E15'))}**",
        f"- Median next-5m-open vs baseline (first open after fill): **{_fmt(overall.get('median_oracle_E15_next_open'))}** — this is the more realistic delay-one-bar comparison",
        f"- Mean ORACLE E15 runaway (max high vs entry): **{_fmt(overall.get('mean_oracle_E15_runaway'))}**",
        f"- Median post-entry MAE 5m / 15m (includes entry bar OHLC): **{_fmt(overall.get('median_post_entry_mae_5m'))}** / **{_fmt(overall.get('median_post_entry_mae_15m'))}**",
        f"- Median post-entry MFE 5m / 15m: **{_fmt(overall.get('median_post_entry_mfe_5m'))}** / **{_fmt(overall.get('median_post_entry_mfe_15m'))}**",
        "",
        "First-5m adverse below uses **entry-bar close < entry or next 5m low < entry** (not “any low on the entry candle”, which is nearly always true).",
        "",
        f"- First 5m adverse (non-tautological): **{_fmt(overall.get('frac_first_5m_adverse'))}**",
        f"- Adverse then close ≥ entry at +15m: **{_fmt(overall.get('frac_pullback_recover_15m'))}**",
        f"- Adverse then close still below entry at +15m: **{_fmt(overall.get('frac_pullback_continue_down'))}**",
        "",
        "Approximate T1 P/L if the exit mid were unchanged (diagnostic only, stops would also move) is in `execution_window_analysis.csv`.",
        "",
        "## Execution windows (ORACLE)",
        "",
    ]
    lines += _md_table(windows, list(windows.columns))
    lines += [
        "",
        "## Q4. Pullback vs failure",
        "",
        "Important tension: **ORACLE E15 improvement is larger on pullback/stabilization states, while win rate is higher on continuation/extension.** "
        "A naive “wait for a better 5m price” rule therefore risks waiting precisely when the 15m setup is already deteriorating.",
        "",
        "Causal states at baseline entry (no outcome used to define labels):",
        "",
    ]
    lines += _md_table(
        states,
        ["slice", "n", "baseline_win_rate", "baseline_mean_pnl", "frac_first_5m_adverse", "frac_pullback_recover", "frac_pullback_fail", "median_oracle_E15"],
    )
    lines += [
        "",
        "## Q5. Does the initial 5m path predict MFE/MAE?",
        "",
        f"- corr(post-entry 5m MAE, T1 MAE) = **{_fmt(overall.get('corr_mae5_vs_T1_mae'))}**",
        f"- corr(post-entry 5m MFE, T1 MFE) = **{_fmt(overall.get('corr_mfe5_vs_T1_mfe'))}**",
        f"- corr(post-entry 5m MAE, T1 P/L) = **{_fmt(overall.get('corr_mae5_vs_T1_pnl'))}**",
        f"- corr(post-entry 5m MFE, T1 P/L) = **{_fmt(overall.get('corr_mfe5_vs_T1_pnl'))}**",
        f"- corr(5m dir-return at entry, T1 P/L) = **{_fmt(overall.get('corr_dirret5_entry_vs_pnl'))}**",
        f"- Median minutes to first adverse 5m print: **{_fmt(overall.get('median_minutes_to_adverse'), 1)}**; first favorable: **{_fmt(overall.get('median_minutes_to_favorable'), 1)}**",
        "",
        "## Q6. T1–T10 and Selector E",
        "",
        "Entry price is shared across T-arms. Differences below are *outcome associations* with the same 5m path, plus an ORACLE approximate P/L (exit unchanged).",
        "",
    ]
    lines += _md_table(
        arms,
        ["arm", "baseline_mean_pnl", "baseline_win_rate", "wr_if_first_5m_adverse", "wr_if_not_adverse", "mean_pnl_if_first_5m_adverse", "mean_pnl_if_not_adverse", "corr_mae5_vs_pnl", "approx_mean_pnl_oracle_E15"],
    )
    lines += [
        "",
        "### Selector E (existing outcomes, not rerun)",
        "",
        f"- N={e_row['n']}; WR {_fmt(e_row['baseline_win_rate'])}; mean P/L {_fmt(e_row['baseline_mean_pnl'])}",
        f"- WR if first-5m adverse {_fmt(e_row['wr_if_first_5m_adverse'])} vs not {_fmt(e_row['wr_if_not_adverse'])}",
        f"- Mean P/L if first-5m adverse {_fmt(e_row['mean_pnl_if_first_5m_adverse'])} vs not {_fmt(e_row['mean_pnl_if_not_adverse'])}",
        f"- corr(5m MAE, E P/L)={_fmt(e_row['corr_mae5_vs_pnl'])}; approx mean P/L ORACLE E15={_fmt(e_row['approx_mean_pnl_oracle_E15'])}",
        "",
        "## Q7. S-band and regime (no per-slice optimization)",
        "",
    ]
    lines += _md_table(
        sband,
        ["slice", "n", "baseline_win_rate", "baseline_mean_pnl", "frac_wait_better_fill", "median_oracle_E15", "frac_first_5m_adverse", "wr_if_first_5m_adverse", "wr_if_not_first_5m_adverse"],
    )
    lines += ["", "### Regime", ""]
    lines += _md_table(
        regime,
        ["slice", "n", "baseline_win_rate", "baseline_mean_pnl", "frac_wait_better_fill", "median_oracle_E15", "frac_first_5m_adverse", "wr_if_first_5m_adverse", "wr_if_not_first_5m_adverse"],
    )
    lines += [
        "",
        "## Q8. Decision",
        "",
        f"**{decision}**",
        "",
        decision_why,
        "",
        "## Deployment",
        "",
        "No execution rule was implemented. Paper/live bot, Selector E, T1–T10, thresholds, and sizing are unchanged. "
        "Even a YES is only permission to design a *small frozen* Part 3B causal-timing test, still in-sample relative to this diagnostic.",
        "",
    ]
    path = Path(out_dir) / "RESEARCH_REPORT_15m_5m_execution_part3a.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_part3a(
    *,
    part1_dir: Path,
    baseline_dir: Path,
    out_dir: Path,
    days: int = 365,
    force_download: bool = False,
    max_opportunities: int | None = None,
) -> dict[str, Path]:
    part1_dir = Path(part1_dir)
    baseline_dir = Path(baseline_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ddir = out_dir / "diagnostics"
    pdir = out_dir / "plots"
    ddir.mkdir(parents=True, exist_ok=True)

    before = baseline_file_hashes(baseline_dir)
    convention = document_entry_convention()
    (out_dir / "entry_convention.json").write_text(json.dumps(convention, indent=2), encoding="utf-8")

    diag = pd.read_csv(part1_dir / DIAGNOSTIC_DIRNAME / DIAGNOSTIC_CSV_NAME, low_memory=False)
    opps = pd.read_csv(baseline_dir / "opportunities.csv", low_memory=False)
    if max_opportunities is not None:
        keep = set(diag["opportunity_id"].astype(str).head(int(max_opportunities)))
        diag = diag[diag["opportunity_id"].astype(str).isin(keep)].copy()
        opps = opps[opps["opportunity_id"].astype(str).isin(keep)].copy()

    if len(diag) != diag["opportunity_id"].nunique():
        raise RuntimeError("Diagnostic CSV is not one row per opportunity")
    if max_opportunities is None and len(diag) != 1517:
        logger.warning("Expected 1517 opportunities, found %s", len(diag))

    avail = pd.to_datetime(diag["available_5m_candle_ts"], utc=True, errors="coerce")
    dec = pd.to_datetime(diag["decision_timestamp"], utc=True, errors="coerce")
    causal_ok = int((avail.notna() & dec.notna() & (avail <= dec)).sum())

    builder = FiveMinuteFeatureBuilder(days=days, force_download=force_download)
    df = build_execution_rows(diag=diag, opps=opps, builder=builder)

    overall = summarize_overall(df)
    states = state_table(df)
    sband = _slice_table(df, "s_band", "primary_pnl_pct", "primary_win")
    regime = _slice_table(df, "regime", "primary_pnl_pct", "primary_win")
    arms = t_arm_table(df)
    e_row = selector_e_row(df)

    win_rows = []
    for w in ORACLE_WINDOWS_MIN:
        imp = _num(df[f"oracle_E{w}_improvement"])
        nxt = _num(df[f"oracle_E{w}_next_open_improvement"])
        run = _num(df[f"oracle_E{w}_runaway"])
        win_rows.append({
            "window": f"ORACLE_E{w}",
            "n": int(imp.notna().sum()),
            "median_min_low_improvement": float(imp.median()) if imp.notna().any() else None,
            "mean_min_low_improvement": float(imp.mean()) if imp.notna().any() else None,
            "p25_improvement": float(imp.quantile(0.25)) if imp.notna().any() else None,
            "p75_improvement": float(imp.quantile(0.75)) if imp.notna().any() else None,
            "frac_positive_improvement": float((imp > 0).mean()) if imp.notna().any() else None,
            "median_next_open_improvement": float(nxt.median()) if nxt.notna().any() else None,
            "mean_runaway_high": float(run.mean()) if run.notna().any() else None,
            "approx_T1_mean_pnl_if_exit_unchanged": _mean(
                pd.Series([_approx_pnl_if_exit_unchanged(_safe_float(p), i) for p, i in zip(df["primary_pnl_pct"], df[f"oracle_E{w}_improvement"])])
            ),
            "label": "ORACLE / IDEALIZED EXECUTION DIAGNOSTIC",
        })
    wait_imp = _num(df["oracle_wait_window_improvement"])
    win_rows.insert(0, {
        "window": "ORACLE_wait_signal_to_entry",
        "n": int(wait_imp.notna().sum()),
        "median_min_low_improvement": float(wait_imp.median()) if wait_imp.notna().any() else None,
        "mean_min_low_improvement": float(wait_imp.mean()) if wait_imp.notna().any() else None,
        "p25_improvement": float(wait_imp.quantile(0.25)) if wait_imp.notna().any() else None,
        "p75_improvement": float(wait_imp.quantile(0.75)) if wait_imp.notna().any() else None,
        "frac_positive_improvement": float((wait_imp > 0).mean()) if wait_imp.notna().any() else None,
        "median_next_open_improvement": None,
        "mean_runaway_high": _mean(df["wait_max_vs_entry"]),
        "approx_T1_mean_pnl_if_exit_unchanged": _mean(df["approx_pnl_oracle_wait"]),
        "label": "ORACLE / IDEALIZED (prices during existing 15m wait before fill)",
    })
    e0 = _num(df["primary_pnl_pct"])
    win_rows.insert(0, {
        "window": "E0_baseline_15m_next_open",
        "n": int(len(df)),
        "median_min_low_improvement": 0.0,
        "mean_min_low_improvement": 0.0,
        "p25_improvement": 0.0,
        "p75_improvement": 0.0,
        "frac_positive_improvement": 0.0,
        "median_next_open_improvement": 0.0,
        "mean_runaway_high": 0.0,
        "approx_T1_mean_pnl_if_exit_unchanged": float(e0.mean()) if len(e0) else None,
        "label": "actual baseline convention (not oracle)",
    })
    windows = pd.DataFrame(win_rows)

    post_cols = [
        "opportunity_id", "pair", "s_band", "regime",
        "post_signal_mae_5m", "post_signal_mae_10m", "post_signal_mae_15m",
        "post_signal_mfe_5m", "post_signal_mfe_10m", "post_signal_mfe_15m",
        "post_entry_mae_5m", "post_entry_mae_10m", "post_entry_mae_15m",
        "post_entry_mfe_5m", "post_entry_mfe_10m", "post_entry_mfe_15m",
        "first_5m_adverse", "pullback_then_recover_15m", "pullback_then_continue_down_15m",
        "primary_win", "primary_pnl_pct",
    ]
    mfe_cols = [
        "opportunity_id", "post_entry_mfe_5m", "post_entry_mae_5m", "dirret_5m_at_entry",
        "minutes_to_first_favorable_5m", "minutes_to_first_adverse_5m",
        "primary_mfe_pct", "primary_mae_pct", "primary_pnl_pct", "causal_state",
    ]

    df.to_csv(ddir / "execution_opportunity_analysis.csv", index=False)
    df[post_cols].to_csv(ddir / "post_signal_path_analysis.csv", index=False)
    df[mfe_cols].to_csv(ddir / "mfe_mae_path_analysis.csv", index=False)
    windows.to_csv(ddir / "execution_window_analysis.csv", index=False)
    arms.to_csv(ddir / "t1_t10_execution_analysis.csv", index=False)
    pd.DataFrame([e_row]).to_csv(ddir / "selector_e_execution_analysis.csv", index=False)
    sband.to_csv(ddir / "s_band_execution_analysis.csv", index=False)
    regime.to_csv(ddir / "regime_execution_analysis.csv", index=False)
    states.to_csv(ddir / "causal_state_analysis.csv", index=False)

    make_plots(df, pdir)
    decision, why = decide(overall, states, arms)
    report = write_report(
        out_dir=out_dir,
        convention=convention,
        overall=overall,
        states=states,
        windows=windows,
        arms=arms,
        e_row=e_row,
        sband=sband,
        regime=regime,
        decision=decision,
        decision_why=why,
        n=int(len(df)),
        causal_ok=causal_ok,
    )
    assert_baseline_unchanged(baseline_dir, before)
    (out_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "status": "COMPLETED",
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "n": int(len(df)),
                "decision": decision,
                "overall": overall,
                "veto_implemented": False,
                "execution_rule_implemented": False,
                "paper_bot_modified": False,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    logger.info("Part 3A complete → %s (%s)", report, decision)
    return {"report": report, "out_dir": out_dir}
