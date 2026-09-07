"""Causal 5m diagnostic dataset for fixed 15m selector opportunities.

Research role: the 15m strategy is an immutable baseline. This module only asks
whether 5m state available at each 15m decision timestamp contains information
about subsequent trade quality. No veto / no trading-logic changes.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from btcc.backtest.config import load_backtest_config
from btcc.backtest.data_loader import download_panels
from btcc.backtest.dominance_history import HistoricalDominanceSeries
from btcc.backtest.predict import predict_coin_at_bar
from btcc.sim.score import combined_score, extract_factor_scores, static_factor_weights
from btcc.sim.selector_config import load_selector_experiment_config

logger = logging.getLogger(__name__)

DIAGNOSTIC_DIRNAME = "diagnostics"
DIAGNOSTIC_CSV_NAME = "15m_opportunities_with_5m_features.csv"
FEATURE_SUMMARY_CSV = "feature_summary.csv"
FEATURE_OUTCOME_CSV = "feature_outcome_analysis.csv"
REPORT_NAME = "RESEARCH_REPORT_15m_5m_diagnostics.md"

SELECTOR_ARMS = tuple("ABCDEF")
FIXED_T_ARMS = tuple(f"T{i}" for i in range(1, 11))  # T11/T12 retired
PRIMARY_OUTCOME_ARM = "T1"

# Offsets used for feature construction / evolution (minutes before decision).
S5_LAGS_MIN = (0, 5, 15, 30, 60)
EVOLUTION_OFFSETS_MIN = (60, 30, 15, 10, 5, 0)
PERSIST_WINDOWS_MIN = (15, 30, 60)
RECENT_CANDLE_WINDOW = 12  # 12 × 5m = 60m


def _utc(ts: Any) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def baseline_file_hashes(baseline_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in ("opportunities.csv", "strategy_legs.csv", "selection_audit.csv", "predictions.csv", "summary.json"):
        p = Path(baseline_dir) / name
        if p.exists():
            out[name] = _sha256(p)
    return out


def assert_baseline_unchanged(baseline_dir: Path, before: dict[str, str]) -> None:
    after = baseline_file_hashes(baseline_dir)
    for name, digest in before.items():
        if after.get(name) != digest:
            raise RuntimeError(f"Immutable baseline modified: {name}")


@dataclass
class _Snap:
    signal_score: float | None
    natr: float | None
    recent_vol: float | None
    trend_up: bool | None
    mom_score: float | None
    available_ts: str | None
    close: float | None


class FiveMinuteFeatureBuilder:
    """Compute causal 5m features using only candles with timestamp <= decision t."""

    def __init__(self, *, days: int, sim_cfg: dict[str, Any] | None = None, force_download: bool = False) -> None:
        self.days = int(days)
        self.sim_cfg = dict(sim_cfg or load_selector_experiment_config())
        self.force_download = bool(force_download)
        self.bt_cfg = load_backtest_config()
        self.weights = static_factor_weights({"sim": self.sim_cfg, **self.bt_cfg})
        self.interval = "5m"
        self.min_warmup_bars = 600
        self.cfg_5m = {
            **self.bt_cfg,
            "backtest": {
                **(self.bt_cfg.get("backtest") or {}),
                "interval": self.interval,
                "min_warmup_bars": self.min_warmup_bars,
            },
            "sim": self.sim_cfg,
        }
        logger.info("Loading 5m panels for diagnostics (days=%s, warmup=%s)", self.days, self.min_warmup_bars)
        self.panels = download_panels(self.cfg_5m, self.days, self.min_warmup_bars, force=self.force_download)
        self.btc_df = self.panels["btc"].copy()
        self.btc_df["timestamp"] = pd.to_datetime(self.btc_df["timestamp"], utc=True)
        self.btc_df = self.btc_df.sort_values("timestamp").reset_index(drop=True)
        self.dom_series = HistoricalDominanceSeries.fetch_for_backtest(
            days=self.days,
            cache_dir=self.bt_cfg["backtest_data"]["dominance_cache"],
            sim_cfg=self.sim_cfg,
            force=self.force_download,
        )
        self._coin_cache: dict[str, dict[str, Any]] = {}
        self._snap_cache: dict[tuple[str, str], _Snap | None] = {}
        self._prepare_coins()

    def _prepare_coins(self) -> None:
        for base, coin in self.panels["coins"].items():
            rel = coin["rel"].copy()
            rel["timestamp"] = pd.to_datetime(rel["timestamp"], utc=True)
            rel = rel.sort_values("timestamp").reset_index(drop=True)
            alt = coin["alt_for_volume"].copy()
            alt["timestamp"] = pd.to_datetime(alt["timestamp"], utc=True)
            alt = alt.sort_values("timestamp").reset_index(drop=True)
            # DatetimeIndex enables fast, tz-safe searchsorted as-of cuts.
            self._coin_cache[base] = {
                "rel": rel,
                "alt": alt,
                "rel_index": pd.DatetimeIndex(rel["timestamp"]),
                "alt_index": pd.DatetimeIndex(alt["timestamp"]),
            }
        self._btc_index = pd.DatetimeIndex(self.btc_df["timestamp"])

    def _hist_slice(self, df: pd.DataFrame, index: pd.DatetimeIndex, at_ts: pd.Timestamp) -> pd.DataFrame:
        # Candle open timestamps; include only information available at decision time.
        t = _utc(at_ts)
        idx = int(index.searchsorted(t, side="right"))
        if idx <= 0:
            return df.iloc[0:0]
        return df.iloc[:idx]

    def _snapshot(self, *, base: str, at_ts: pd.Timestamp) -> _Snap | None:
        key = (base, str(at_ts))
        if key in self._snap_cache:
            return self._snap_cache[key]
        prepared = self._coin_cache.get(base)
        if prepared is None:
            self._snap_cache[key] = None
            return None
        rel_hist = self._hist_slice(prepared["rel"], prepared["rel_index"], at_ts)
        if len(rel_hist) < 100:
            self._snap_cache[key] = None
            return None
        alt_hist = self._hist_slice(prepared["alt"], prepared["alt_index"], at_ts)
        btc_idx = int(self._btc_index.searchsorted(_utc(at_ts), side="right"))
        if btc_idx <= 0:
            self._snap_cache[key] = None
            return None
        btc_hist = self.btc_df.iloc[:btc_idx]
        dom_pct, _, _ = self.dom_series.observation_at(at_ts)
        dom_changes = self.dom_series.dom_changes_at(at_ts)
        row = predict_coin_at_bar(rel_hist, alt_hist, btc_hist, dom_pct, dom_changes, self.cfg_5m, self.interval)
        if row is None:
            self._snap_cache[key] = None
            return None
        factors = row.get("factors") or {}
        scored = combined_score(extract_factor_scores(factors), self.weights)
        close = rel_hist["close"].astype(float)
        rets = close.pct_change().dropna()
        recent_vol = float(rets.iloc[-12:].std()) if len(rets) >= 12 else None
        trend = factors.get("trend") or {}
        volat = factors.get("volatility") or {}
        mom = factors.get("momentum") or {}
        avail = str(rel_hist["timestamp"].iloc[-1])
        snap = _Snap(
            signal_score=_safe_float(scored.get("S")),
            natr=_safe_float(volat.get("natr")),
            recent_vol=_safe_float(recent_vol),
            trend_up=(bool(trend.get("ema20_gt_ema50")) if trend.get("ema20_gt_ema50") is not None else None),
            mom_score=_safe_float(mom.get("score")),
            available_ts=avail,
            close=_safe_float(close.iloc[-1]),
        )
        self._snap_cache[key] = snap
        return snap

    @staticmethod
    def _directional_return(close: pd.Series, bars: int, *, long: bool = True) -> float | None:
        if bars <= 0 or len(close) <= bars:
            return None
        now = _safe_float(close.iloc[-1])
        prev = _safe_float(close.iloc[-1 - bars])
        if now is None or prev is None or prev == 0:
            return None
        raw = now / prev - 1.0
        return raw if long else -raw

    @staticmethod
    def _streak(close: pd.Series, *, favorable: bool, long: bool = True) -> int:
        if len(close) < 2:
            return 0
        diffs = close.diff().dropna()
        count = 0
        for d in reversed(diffs.tolist()):
            # LONG: up=favorable; SHORT would invert (not used in this baseline).
            up = d > 0
            is_fav = up if long else (not up)
            if favorable and is_fav:
                count += 1
            elif (not favorable) and (not is_fav):
                count += 1
            else:
                break
        return count

    @staticmethod
    def _frac_favorable(close: pd.Series, window: int, *, long: bool = True) -> float | None:
        if len(close) < 2:
            return None
        diffs = close.diff().dropna().iloc[-window:]
        if diffs.empty:
            return None
        fav = (diffs > 0) if long else (diffs < 0)
        return float(fav.mean())

    def _s5_at(self, base: str, t: pd.Timestamp, minutes_ago: int) -> float | None:
        snap = self._snapshot(base=base, at_ts=t - pd.Timedelta(minutes=minutes_ago))
        return None if snap is None else snap.signal_score

    def build_features_for_opportunity(self, *, base: str, signal_ts: Any, signal_15m: float | None) -> dict[str, Any]:
        t = _utc(signal_ts)
        long = True  # baseline opportunities are LONG_ALT_BTC
        prepared = self._coin_cache.get(base)
        out: dict[str, Any] = {
            "trade_direction": "LONG",
            "decision_timestamp": str(t),
            "feature_asof_rule": "5m_candle_timestamp_le_decision_ts",
        }
        if prepared is None:
            out["five_m_data_available"] = False
            return out

        rel_hist = self._hist_slice(prepared["rel"], prepared["rel_index"], t)
        if len(rel_hist) < 100:
            out["five_m_data_available"] = False
            return out
        out["five_m_data_available"] = True
        close = rel_hist["close"].astype(float)
        out["available_5m_candle_ts"] = str(rel_hist["timestamp"].iloc[-1])
        out["n_5m_candles_asof"] = int(len(rel_hist))

        # --- S5 levels / changes ---
        s_by_lag: dict[int, float | None] = {}
        for lag in S5_LAGS_MIN:
            s_by_lag[lag] = self._s5_at(base, t, lag)
            out[f"S5_{lag}m_ago" if lag else "S5_current"] = s_by_lag[lag]
        # alias requested names
        out["S5_5m_ago"] = s_by_lag.get(5)
        out["S5_15m_ago"] = s_by_lag.get(15)
        out["S5_30m_ago"] = s_by_lag.get(30)
        out["S5_60m_ago"] = s_by_lag.get(60)
        cur = s_by_lag.get(0)
        for lag in (15, 30, 60):
            prev = s_by_lag.get(lag)
            out[f"S5_change_{lag}m"] = None if cur is None or prev is None else cur - prev

        # --- directional returns ---
        for label, bars in (("5m", 1), ("15m", 3), ("30m", 6), ("60m", 12)):
            out[f"directional_return_{label}"] = self._directional_return(close, bars, long=long)

        # --- momentum / trend ---
        cur_snap = self._snapshot(base=base, at_ts=t)
        mom = None if cur_snap is None else cur_snap.mom_score
        out["momentum_5m_score"] = mom
        # signed relative to trade: mom score in [0,1], 0.5 neutral
        if mom is None:
            out["momentum_5m_signed"] = None
            out["momentum_5m_state"] = None
        else:
            signed = (mom - 0.5) * (1.0 if long else -1.0)
            out["momentum_5m_signed"] = signed
            if signed > 0.05:
                out["momentum_5m_state"] = "favorable"
            elif signed < -0.05:
                out["momentum_5m_state"] = "adverse"
            else:
                out["momentum_5m_state"] = "neutral"

        trend_up = None if cur_snap is None else cur_snap.trend_up
        if trend_up is None:
            out["trend_5m_alignment"] = None
            out["trend_5m_state"] = None
        else:
            aligned = bool(trend_up) if long else (not bool(trend_up))
            out["trend_5m_alignment"] = 1 if aligned else -1
            out["trend_5m_state"] = "favorable" if aligned else "adverse"

        # --- volatility ---
        out["natr_5m_current"] = None if cur_snap is None else cur_snap.natr
        out["volatility_5m_recent"] = None if cur_snap is None else cur_snap.recent_vol
        prev_vol_snap = self._snapshot(base=base, at_ts=t - pd.Timedelta(minutes=30))
        prev_natr = None if prev_vol_snap is None else prev_vol_snap.natr
        out["natr_5m_30m_ago"] = prev_natr
        if out["natr_5m_current"] is None or prev_natr is None or prev_natr == 0:
            out["natr_5m_change_ratio_30m"] = None
        else:
            out["natr_5m_change_ratio_30m"] = out["natr_5m_current"] / prev_natr

        # --- candle behavior ---
        out["consecutive_favorable_5m"] = self._streak(close, favorable=True, long=long)
        out["consecutive_adverse_5m"] = self._streak(close, favorable=False, long=long)
        out["frac_favorable_5m_60m"] = self._frac_favorable(close, RECENT_CANDLE_WINDOW, long=long)
        out["frac_adverse_5m_60m"] = (
            None if out["frac_favorable_5m_60m"] is None else 1.0 - float(out["frac_favorable_5m_60m"])
        )

        # --- persistence of S5 >= 0.60 ---
        for win in PERSIST_WINDOWS_MIN:
            vals = []
            n_bars = win // 5
            for i in range(n_bars):
                vals.append(self._s5_at(base, t, i * 5))
            finite = [v for v in vals if v is not None]
            out[f"S5_persist_frac_ge060_{win}m"] = None if not finite else float(np.mean([v >= 0.60 for v in finite]))
            out[f"S5_recent_seq_{win}m"] = "|".join("" if v is None else f"{v:.4f}" for v in vals)

        # --- 5m vs 15m disagreement ---
        s15 = _safe_float(signal_15m)
        out["S15"] = s15
        out["S5_minus_S15"] = None if cur is None or s15 is None else cur - s15
        # directional disagreement: adverse momentum / adverse trend while 15m is a long setup
        out["disagreement_momentum"] = None if out.get("momentum_5m_signed") is None else float(out["momentum_5m_signed"] < 0)
        out["disagreement_trend"] = None if out.get("trend_5m_alignment") is None else float(out["trend_5m_alignment"] < 0)
        out["disagreement_s5_falling_15m"] = None if out.get("S5_change_15m") is None else float(out["S5_change_15m"] < 0)

        # --- evolution series (for Part 3 timing analysis) ---
        for off in EVOLUTION_OFFSETS_MIN:
            snap = self._snapshot(base=base, at_ts=t - pd.Timedelta(minutes=off))
            out[f"evol_S5_m{off}"] = None if snap is None else snap.signal_score
            # directional return over prior 15m ending at this offset
            hist = self._hist_slice(prepared["rel"], prepared["rel_index"], t - pd.Timedelta(minutes=off))
            if len(hist) >= 4:
                out[f"evol_dirret15_m{off}"] = self._directional_return(hist["close"].astype(float), 3, long=long)
            else:
                out[f"evol_dirret15_m{off}"] = None
            if snap is None or snap.trend_up is None:
                out[f"evol_trend_align_m{off}"] = None
            else:
                out[f"evol_trend_align_m{off}"] = 1 if (snap.trend_up if long else (not snap.trend_up)) else -1

        return out


def _selector_pick_map(selection: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for _, row in selection.iterrows():
        out[(str(row["opportunity_id"]), str(row["arm_label"]))] = row.to_dict()
    return out


def _leg_maps(legs: pd.DataFrame) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    fixed: dict[tuple[str, str], dict[str, Any]] = {}
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for _, row in legs.iterrows():
        oid = str(row["opportunity_id"])
        arm = str(row.get("arm_key") or "")
        if arm in FIXED_T_ARMS:
            fixed[(oid, arm)] = row.to_dict()
        elif arm in SELECTOR_ARMS:
            selected[(oid, arm)] = row.to_dict()
    return fixed, selected


def _s_band(s: float | None) -> str | None:
    if s is None:
        return None
    if 0.60 <= s < 0.65:
        return "0.60-0.65"
    if 0.65 <= s < 0.70:
        return "0.65-0.70"
    if 0.70 <= s < 0.80:
        return "0.70-0.80"
    if s >= 0.80:
        return "0.80+"
    return "below_0.60"


def build_diagnostic_dataset_from_baseline(
    *,
    baseline_dir: Path,
    out_dir: Path,
    days: int = 365,
    force_download: bool = False,
    max_opportunities: int | None = None,
    sim_cfg: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Post-process frozen 15m baseline → isolated diagnostic outputs."""
    baseline_dir = Path(baseline_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    diag_dir = out_dir / DIAGNOSTIC_DIRNAME
    diag_dir.mkdir(parents=True, exist_ok=True)

    before = baseline_file_hashes(baseline_dir)
    (out_dir / "baseline_source.json").write_text(
        json.dumps(
            {
                "baseline_dir": str(baseline_dir),
                "baseline_hashes_before": before,
                "note": "Immutable 15m reference; T11/T12 excluded from diagnostic labels.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    opps = pd.read_csv(baseline_dir / "opportunities.csv", low_memory=False)
    legs = pd.read_csv(baseline_dir / "strategy_legs.csv", low_memory=False)
    selection = pd.read_csv(baseline_dir / "selection_audit.csv", low_memory=False)
    if max_opportunities is not None:
        opps = opps.head(int(max_opportunities)).copy()

    builder = FiveMinuteFeatureBuilder(days=days, sim_cfg=sim_cfg, force_download=force_download)
    fixed_legs, selected_legs = _leg_maps(legs)
    pick_map = _selector_pick_map(selection)

    rows: list[dict[str, Any]] = []
    for i, opp in opps.iterrows():
        oid = str(opp["opportunity_id"])
        base = str(opp["base"])
        signal_ts = opp.get("signal_timestamp") or opp.get("opened_ts")
        s15 = _safe_float(opp.get("S"))
        feat = builder.build_features_for_opportunity(base=base, signal_ts=signal_ts, signal_15m=s15)
        row: dict[str, Any] = {
            "opportunity_id": oid,
            "timestamp": signal_ts,
            "pair": opp.get("symbol"),
            "base": base,
            "direction": feat.get("trade_direction", "LONG"),
            "signal_15m": s15,
            "s_band": _s_band(s15),
            "regime": opp.get("regime"),
            "day_number": opp.get("day_number"),
            **feat,
        }
        for arm in FIXED_T_ARMS:
            leg = fixed_legs.get((oid, arm), {})
            pnl = _safe_float(leg.get("pnl_pct"))
            row[f"{arm}_pnl_pct"] = pnl
            row[f"{arm}_mfe_pct"] = _safe_float(leg.get("mfe_pct"))
            row[f"{arm}_mae_pct"] = _safe_float(leg.get("mae_pct"))
            row[f"{arm}_win"] = None if pnl is None else bool(pnl > 0)
        for sel in SELECTOR_ARMS:
            pick = pick_map.get((oid, sel), {})
            leg = selected_legs.get((oid, sel), {})
            pnl = _safe_float(leg.get("pnl_pct"))
            row[f"selector_{sel}_selected_strategy"] = pick.get("selected_arm_label")
            row[f"selector_{sel}_selected_strategy_key"] = pick.get("selected_strategy_key")
            row[f"selector_{sel}_selected_pnl_pct"] = pnl
            row[f"selector_{sel}_selected_mfe_pct"] = _safe_float(leg.get("mfe_pct"))
            row[f"selector_{sel}_selected_mae_pct"] = _safe_float(leg.get("mae_pct"))
            row[f"selector_{sel}_selected_win"] = None if pnl is None else bool(pnl > 0)
        # primary labels for analysis
        p = row.get(f"{PRIMARY_OUTCOME_ARM}_pnl_pct")
        row["primary_arm"] = PRIMARY_OUTCOME_ARM
        row["primary_pnl_pct"] = p
        row["primary_mfe_pct"] = row.get(f"{PRIMARY_OUTCOME_ARM}_mfe_pct")
        row["primary_mae_pct"] = row.get(f"{PRIMARY_OUTCOME_ARM}_mae_pct")
        row["primary_win"] = None if p is None else bool(p > 0)
        rows.append(row)
        if (len(rows) % 50) == 0:
            logger.info("Diagnostics features %d/%d", len(rows), len(opps))

    diag = pd.DataFrame(rows).sort_values(["timestamp", "pair"], kind="stable").reset_index(drop=True)

    # large-loss labels from primary arm
    pnl = pd.to_numeric(diag["primary_pnl_pct"], errors="coerce")
    losses = pnl[pnl <= 0]
    if len(losses):
        thr20 = float(losses.quantile(0.20))  # more negative than 80% of losses
        thr10 = float(losses.quantile(0.10))
    else:
        thr20 = thr10 = None
    diag["primary_large_loss_p20"] = (pnl <= thr20) if thr20 is not None else False
    diag["primary_large_loss_p10"] = (pnl <= thr10) if thr10 is not None else False

    csv_path = diag_dir / DIAGNOSTIC_CSV_NAME
    diag.to_csv(csv_path, index=False)

    assert_baseline_unchanged(baseline_dir, before)
    (out_dir / "baseline_hashes_after.json").write_text(
        json.dumps({"baseline_hashes_after": baseline_file_hashes(baseline_dir)}, indent=2),
        encoding="utf-8",
    )

    # lightweight summary
    summary = {
        "n_opportunities": int(len(diag)),
        "n_unique_ids": int(diag["opportunity_id"].nunique()),
        "n_five_m_available": int(diag.get("five_m_data_available", pd.Series(dtype=bool)).fillna(False).sum())
        if "five_m_data_available" in diag
        else None,
        "primary_arm": PRIMARY_OUTCOME_ARM,
        "winners": int((pnl > 0).sum()),
        "losers": int((pnl <= 0).sum()),
        "avg_pnl_pct": _safe_float(pnl.mean()),
        "avg_mfe_pct": _safe_float(pd.to_numeric(diag["primary_mfe_pct"], errors="coerce").mean()),
        "avg_mae_pct": _safe_float(pd.to_numeric(diag["primary_mae_pct"], errors="coerce").mean()),
        "large_loss_p20_threshold": thr20,
        "large_loss_p10_threshold": thr10,
        "baseline_dir": str(baseline_dir),
        "anti_lookahead": "features use only 5m candles with timestamp <= 15m decision timestamp",
        "excluded_arms": ["T11", "T12"],
    }
    (diag_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {"diagnostic_csv": csv_path, "diagnostics_dir": diag_dir, "out_dir": out_dir}
