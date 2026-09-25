"""Cloud-extension research primitives (I2 episodes → τ crossings → path stats).

Uses frozen causal Ichimoku (9/26/52/disp26). No parameter retune.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from btcc_m.ichimoku.signals import kumo_breakout
from btcc_m.signal.factors import BASELINE_WEIGHTS
from btcc_m.signal.score import frozen_baseline_weights, score_at_bar

ANCHOR_TAUS = (0.005, 0.01, 0.02, 0.03, 0.05)
QUANTILE_TAUS = (0.20, 0.40, 0.60, 0.80)
WALL_HORIZONS_H = (1, 2, 4, 8, 12, 24)
MFE_REACH = (0.03, 0.05, 0.10, 0.20)


def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def attach_extension_features(ichi: pd.DataFrame) -> pd.DataFrame:
    """Add % / ATR cloud extension + candle quality + acceleration columns."""
    out = ichi.copy()
    c = out["close"].astype(float)
    h = out["high"].astype(float)
    lo = out["low"].astype(float)
    o = out["open"].astype(float)
    top = out["kumo_top"].astype(float)
    atr = out["atr14"].astype(float).replace(0, np.nan)

    out["ext_pct_close"] = (c - top) / top
    out["ext_pct_high"] = (h - top) / top
    out["ext_atr"] = (c - top) / atr

    rng = (h - lo).replace(0, np.nan)
    body = (c - o).abs()
    upper = h - pd.concat([o, c], axis=1).max(axis=1)
    lower = pd.concat([o, c], axis=1).min(axis=1) - lo
    out["candle_body_pct"] = body / rng
    out["candle_upper_wick_pct"] = upper / rng
    out["candle_lower_wick_pct"] = lower / rng
    out["candle_close_loc"] = (c - lo) / rng
    out["candle_return"] = c / o - 1.0
    if "volume" in out.columns:
        vol = out["volume"].astype(float)
        out["rvol_20"] = vol / vol.rolling(20, min_periods=5).mean()
    else:
        out["rvol_20"] = np.nan

    out["ext_delta"] = out["ext_pct_close"].diff()
    out["tk_bull"] = out["tenkan"] > out["kijun"]
    return out


def compute_s_series(
    signal_df: pd.DataFrame,
    *,
    interval: str,
    warmup_bars: int = 200,
    max_lookback: int = 1500,
    momentum_profile: str = "e2",
) -> pd.Series:
    """Walk panel once; return S aligned to signal_df index (NaN until warmup)."""
    w = frozen_baseline_weights(BASELINE_WEIGHTS)
    s_vals = np.full(len(signal_df), np.nan, dtype=float)
    n = len(signal_df)
    for i in range(n):
        if i + 1 < warmup_bars:
            continue
        start_i = max(0, i + 1 - max_lookback)
        hist = signal_df.iloc[start_i : i + 1]
        try:
            scored = score_at_bar(hist, interval=interval, weights=w, momentum_profile=momentum_profile)
            s_vals[i] = float(scored["S"])
        except Exception:
            s_vals[i] = np.nan
    return pd.Series(s_vals, index=signal_df.index, name="S")


@dataclass
class Episode:
    episode_id: str
    interval: str
    i2_idx: int
    i2_ts: str
    start_idx: int
    end_idx: int
    n_bars: int
    max_ext_pct: float
    max_ext_atr: float
    end_reason: str


def detect_i2_episodes(
    ichi: pd.DataFrame,
    *,
    interval: str,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
) -> tuple[list[Episode], pd.DataFrame]:
    """I2 starts an above-cloud episode; ends when close <= kumo_top."""
    eval_start, eval_end = _utc(eval_start), _utc(eval_end)
    fire = kumo_breakout(ichi).fillna(False)
    ts = ichi["timestamp"].map(_utc)
    in_eval = (ts >= eval_start) & (ts <= eval_end)

    episodes: list[Episode] = []
    bar_rows: list[dict[str, Any]] = []
    n = len(ichi)
    ep_counter = 0
    i = 0
    while i < n - 1:
        if not (bool(fire.iloc[i]) and bool(in_eval.iloc[i])):
            i += 1
            continue
        ep_counter += 1
        eid = f"I2_{interval}_{ep_counter:04d}"
        start = i
        j = i
        max_ext = float(ichi.iloc[i]["ext_pct_close"]) if pd.notna(ichi.iloc[i].get("ext_pct_close")) else 0.0
        max_atr = float(ichi.iloc[i]["ext_atr"]) if pd.notna(ichi.iloc[i].get("ext_atr")) else float("nan")
        end_reason = "data_end"
        while j < n:
            row = ichi.iloc[j]
            if float(row["close"]) <= float(row["kumo_top"]):
                end_reason = "lost_cloud"
                break
            if _utc(row["timestamp"]) > eval_end:
                end_reason = "eval_end"
                break
            ext = float(row["ext_pct_close"]) if pd.notna(row.get("ext_pct_close")) else np.nan
            if np.isfinite(ext):
                max_ext = max(max_ext, ext)
            atr_e = float(row["ext_atr"]) if pd.notna(row.get("ext_atr")) else np.nan
            if np.isfinite(atr_e) and (not np.isfinite(max_atr) or atr_e > max_atr):
                max_atr = atr_e
            bar_rows.append(_bar_row(ichi, j, eid, interval, bars_since_i2=j - start))
            j += 1
        end_idx = max(start, j - 1) if j > start else start
        if end_reason == "lost_cloud":
            end_idx = max(start, j - 1)
        episodes.append(
            Episode(
                episode_id=eid,
                interval=interval,
                i2_idx=start,
                i2_ts=str(_utc(ichi.iloc[start]["timestamp"])),
                start_idx=start,
                end_idx=end_idx,
                n_bars=end_idx - start + 1,
                max_ext_pct=float(max_ext),
                max_ext_atr=float(max_atr) if np.isfinite(max_atr) else float("nan"),
                end_reason=end_reason,
            )
        )
        i = max(j, start + 1)
    return episodes, pd.DataFrame(bar_rows)


def _bar_row(ichi: pd.DataFrame, idx: int, eid: str, interval: str, *, bars_since_i2: int) -> dict[str, Any]:
    r = ichi.iloc[idx]
    return {
        "episode_id": eid,
        "interval": interval,
        "bar_idx": idx,
        "bars_since_i2": bars_since_i2,
        "timestamp": str(_utc(r["timestamp"])),
        "open": float(r["open"]),
        "high": float(r["high"]),
        "low": float(r["low"]),
        "close": float(r["close"]),
        "kumo_top": float(r["kumo_top"]),
        "kumo_bot": float(r["kumo_bot"]),
        "ext_pct_close": float(r["ext_pct_close"]) if pd.notna(r.get("ext_pct_close")) else np.nan,
        "ext_pct_high": float(r["ext_pct_high"]) if pd.notna(r.get("ext_pct_high")) else np.nan,
        "ext_atr": float(r["ext_atr"]) if pd.notna(r.get("ext_atr")) else np.nan,
        "ext_delta": float(r["ext_delta"]) if pd.notna(r.get("ext_delta")) else np.nan,
        "S": float(r["S"]) if "S" in ichi.columns and pd.notna(r.get("S")) else np.nan,
        "kumo_bullish": bool(r["kumo_bullish"]) if pd.notna(r.get("kumo_bullish")) else None,
        "tk_bull": bool(r["tk_bull"]) if pd.notna(r.get("tk_bull")) else None,
        "kijun_slope": float(r["kijun_slope"]) if pd.notna(r.get("kijun_slope")) else np.nan,
        "chikou_above": bool(r["chikou_above"]) if pd.notna(r.get("chikou_above")) else None,
        "candle_body_pct": float(r["candle_body_pct"]) if pd.notna(r.get("candle_body_pct")) else np.nan,
        "candle_upper_wick_pct": float(r["candle_upper_wick_pct"]) if pd.notna(r.get("candle_upper_wick_pct")) else np.nan,
        "candle_lower_wick_pct": float(r["candle_lower_wick_pct"]) if pd.notna(r.get("candle_lower_wick_pct")) else np.nan,
        "candle_close_loc": float(r["candle_close_loc"]) if pd.notna(r.get("candle_close_loc")) else np.nan,
        "candle_return": float(r["candle_return"]) if pd.notna(r.get("candle_return")) else np.nan,
        "rvol_20": float(r["rvol_20"]) if pd.notna(r.get("rvol_20")) else np.nan,
        "atr14": float(r["atr14"]) if pd.notna(r.get("atr14")) else np.nan,
    }


def build_tau_grid(episodes: list[Episode]) -> list[float]:
    """Anchors + empirical quantiles of max episode extension (positive only)."""
    maxes = np.array([e.max_ext_pct for e in episodes if np.isfinite(e.max_ext_pct) and e.max_ext_pct > 0])
    taus = set(ANCHOR_TAUS)
    if len(maxes) >= 5:
        for q in QUANTILE_TAUS:
            v = float(np.nanquantile(maxes, q))
            if np.isfinite(v) and v > 1e-6:
                taus.add(round(v, 6))
    return sorted(taus)


def threshold_crossing_events(
    ichi: pd.DataFrame,
    episodes: list[Episode],
    taus: list[float],
    *,
    interval: str,
) -> pd.DataFrame:
    """First close-extension crossing of τ within each episode (one event per episode×τ)."""
    rows: list[dict[str, Any]] = []
    for ep in episodes:
        for tau in taus:
            for idx in range(ep.start_idx, ep.end_idx + 1):
                if idx <= 0:
                    continue
                prev_v = ichi.iloc[idx - 1].get("ext_pct_close")
                curr_v = ichi.iloc[idx].get("ext_pct_close")
                prev = float(prev_v) if pd.notna(prev_v) else -np.inf
                curr = float(curr_v) if pd.notna(curr_v) else -np.inf
                if float(ichi.iloc[idx]["close"]) <= float(ichi.iloc[idx]["kumo_top"]):
                    break
                if prev < tau <= curr:
                    r = ichi.iloc[idx]
                    if idx + 1 >= len(ichi):
                        break
                    nxt = ichi.iloc[idx + 1]
                    rows.append(
                        {
                            "event_id": f"{ep.episode_id}_tau{tau:.4f}",
                            "episode_id": ep.episode_id,
                            "interval": interval,
                            "tau": float(tau),
                            "signal_idx": idx,
                            "signal_ts": str(_utc(r["timestamp"])),
                            "entry_ts": str(_utc(nxt["timestamp"])),
                            "entry_mid": float(nxt["open"]),
                            "ext_pct_close": curr,
                            "ext_pct_high": float(r["ext_pct_high"]) if pd.notna(r.get("ext_pct_high")) else np.nan,
                            "ext_atr": float(r["ext_atr"]) if pd.notna(r.get("ext_atr")) else np.nan,
                            "ext_delta": float(r["ext_delta"]) if pd.notna(r.get("ext_delta")) else np.nan,
                            "S": float(r["S"]) if "S" in ichi.columns and pd.notna(r.get("S")) else np.nan,
                            "kumo_bullish": bool(r["kumo_bullish"]) if pd.notna(r.get("kumo_bullish")) else None,
                            "tk_bull": bool(r["tk_bull"]) if pd.notna(r.get("tk_bull")) else None,
                            "kijun_slope": float(r["kijun_slope"]) if pd.notna(r.get("kijun_slope")) else np.nan,
                            "chikou_above": bool(r["chikou_above"]) if pd.notna(r.get("chikou_above")) else None,
                            "candle_body_pct": float(r["candle_body_pct"]) if pd.notna(r.get("candle_body_pct")) else np.nan,
                            "candle_upper_wick_pct": float(r["candle_upper_wick_pct"]) if pd.notna(r.get("candle_upper_wick_pct")) else np.nan,
                            "candle_close_loc": float(r["candle_close_loc"]) if pd.notna(r.get("candle_close_loc")) else np.nan,
                            "candle_return": float(r["candle_return"]) if pd.notna(r.get("candle_return")) else np.nan,
                            "rvol_20": float(r["rvol_20"]) if pd.notna(r.get("rvol_20")) else np.nan,
                            "bars_since_i2": idx - ep.start_idx,
                            "i2_ts": ep.i2_ts,
                            "i2_idx": ep.i2_idx,
                            "kumo_top_at_signal": float(r["kumo_top"]),
                            "kumo_bot_at_signal": float(r["kumo_bot"]),
                            "atr14": float(r["atr14"]) if pd.notna(r.get("atr14")) else np.nan,
                        }
                    )
                    break
    return pd.DataFrame(rows)


def attach_1m_path_stats(
    events: pd.DataFrame,
    df_1m: pd.DataFrame,
    *,
    wall_horizons_h: tuple[int, ...] = WALL_HORIZONS_H,
    mfe_horizon_h: int = 24,
) -> pd.DataFrame:
    """Next-open fill path on 1m: fwd returns at wall-clock horizons + MFE/MAE."""
    if events is None or events.empty:
        return events
    ts_1m = pd.to_datetime(df_1m["timestamp"], utc=True)
    c = df_1m["close"].astype(float).values
    h = df_1m["high"].astype(float).values
    lo = df_1m["low"].astype(float).values
    n = len(df_1m)
    ts_ns = ts_1m.values.astype("datetime64[ns]").astype(np.int64)

    out_rows: list[dict[str, Any]] = []
    for _, ev in events.iterrows():
        row = dict(ev)
        entry_ts = _utc(ev["entry_ts"])
        entry_ns = np.int64(pd.Timestamp(entry_ts).to_datetime64().astype("datetime64[ns]").astype(np.int64))
        # robust: use .value
        entry_ns = np.int64(pd.Timestamp(entry_ts).value)
        i0 = int(np.searchsorted(ts_ns, entry_ns, side="left"))
        if i0 >= n:
            out_rows.append(row)
            continue
        px = float(ev["entry_mid"])
        if px <= 0:
            out_rows.append(row)
            continue

        end_ns = entry_ns + np.int64(mfe_horizon_h * 3600 * 1_000_000_000)
        i_end = int(np.searchsorted(ts_ns, end_ns, side="right"))
        i_end = min(i_end, n)
        if i_end > i0:
            path_h = h[i0:i_end]
            path_l = lo[i0:i_end]
            mfe = float(np.nanmax(path_h) / px - 1.0)
            mae = float(np.nanmin(path_l) / px - 1.0)
            j_mfe = int(np.nanargmax(path_h))
            t_mfe = pd.Timestamp(ts_1m.iloc[i0 + j_mfe])
            row["mfe_pct"] = mfe
            row["mae_pct"] = mae
            row["time_to_mfe_h"] = float((t_mfe - entry_ts).total_seconds() / 3600.0)
            for thr in MFE_REACH:
                row[f"reach_{int(thr * 100)}pct"] = bool(mfe >= thr)
        else:
            row["mfe_pct"] = np.nan
            row["mae_pct"] = np.nan
            row["time_to_mfe_h"] = np.nan
            for thr in MFE_REACH:
                row[f"reach_{int(thr * 100)}pct"] = False

        for hh in wall_horizons_h:
            target_ns = entry_ns + np.int64(hh * 3600 * 1_000_000_000)
            j = int(np.searchsorted(ts_ns, target_ns, side="right")) - 1
            if j >= i0 and j < n:
                row[f"fwd_{hh}h"] = float(c[j] / px - 1.0)
            else:
                row[f"fwd_{hh}h"] = np.nan

        if pd.notna(ev.get("i2_ts")):
            i2_ts = _utc(ev["i2_ts"])
            i2_ns = np.int64(pd.Timestamp(i2_ts).value)
            k0 = int(np.searchsorted(ts_ns, i2_ns, side="right"))
            if k0 < n and i_end > k0:
                px_i2 = float(df_1m.iloc[k0]["open"])
                if px_i2 > 0:
                    mfe_from_i2 = float(np.nanmax(h[k0:i_end]) / px_i2 - 1.0)
                    row["mfe_from_i2_pct"] = mfe_from_i2
                    row["mfe_spent_before_tau"] = mfe_from_i2 - float(row.get("mfe_pct") or 0.0)
                else:
                    row["mfe_from_i2_pct"] = np.nan
                    row["mfe_spent_before_tau"] = np.nan
            else:
                row["mfe_from_i2_pct"] = np.nan
                row["mfe_spent_before_tau"] = np.nan

        out_rows.append(row)
    return pd.DataFrame(out_rows)


def continuum_summary(events: pd.DataFrame) -> pd.DataFrame:
    if events is None or events.empty:
        return pd.DataFrame()
    rows = []
    for (interval, tau), g in events.groupby(["interval", "tau"]):
        mfe = pd.to_numeric(g["mfe_pct"], errors="coerce")
        mae = pd.to_numeric(g["mae_pct"], errors="coerce")
        s = pd.to_numeric(g["S"], errors="coerce")
        fwd4 = pd.to_numeric(g["fwd_4h"], errors="coerce") if "fwd_4h" in g.columns else pd.Series(dtype=float)
        spent = pd.to_numeric(g["mfe_spent_before_tau"], errors="coerce") if "mfe_spent_before_tau" in g.columns else pd.Series(dtype=float)
        rows.append(
            {
                "interval": interval,
                "tau": float(tau),
                "tau_pct": float(tau) * 100.0,
                "n": int(len(g)),
                "mean_S": float(s.mean()) if s.notna().any() else np.nan,
                "median_S": float(s.median()) if s.notna().any() else np.nan,
                "mean_mfe": float(mfe.mean()) if mfe.notna().any() else np.nan,
                "median_mfe": float(mfe.median()) if mfe.notna().any() else np.nan,
                "p75_mfe": float(mfe.quantile(0.75)) if mfe.notna().any() else np.nan,
                "mean_mae": float(mae.mean()) if mae.notna().any() else np.nan,
                "median_mae": float(mae.median()) if mae.notna().any() else np.nan,
                "cont_fwd4h": float((fwd4 > 0).mean()) if fwd4.notna().any() else np.nan,
                "reach_3pct": float(g["reach_3pct"].astype(bool).mean()) if "reach_3pct" in g else np.nan,
                "reach_5pct": float(g["reach_5pct"].astype(bool).mean()) if "reach_5pct" in g else np.nan,
                "reach_10pct": float(g["reach_10pct"].astype(bool).mean()) if "reach_10pct" in g else np.nan,
                "mean_spent": float(spent.mean()) if spent.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["interval", "tau"]).reset_index(drop=True)


def episodes_to_frame(episodes: list[Episode]) -> pd.DataFrame:
    if not episodes:
        return pd.DataFrame()
    return pd.DataFrame([asdict(e) for e in episodes])
