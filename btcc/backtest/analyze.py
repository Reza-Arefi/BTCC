"""Research analysis for 90-day BTCC backtest results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HORIZONS = [1, 4, 8, 12, 24]
PROB_BUCKETS = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)]
LATE_BUCKETS = [
    ("0.00-0.30 NORMAL", 0.0, 0.30),
    ("0.30-0.55 EXTENDED", 0.30, 0.55),
    ("0.55-0.75 HIGH", 0.55, 0.75),
    ("0.75-1.00 VERY_HIGH", 0.75, 1.01),
]
FACTOR_COLS = [
    ("factor_momentum", "MOM"),
    ("factor_trend", "TREND"),
    ("factor_btc_regime", "BTC_REGIME"),
    ("factor_volume", "VOL"),
    ("factor_volatility", "VOLAT"),
    ("factor_rsi", "RSI"),
    ("factor_structure", "STRUCT"),
]
INDICATOR_COLS = [
    ("indicator_momentum", "Momentum"),
    ("indicator_ema", "EMA"),
    ("indicator_macd", "MACD"),
    ("indicator_ichimoku", "Ichimoku"),
    ("indicator_adx", "ADX"),
    ("indicator_btc_regime", "BTC Regime"),
    ("indicator_btc_dominance", "BTC Dominance"),
    ("indicator_rvol", "RVOL"),
    ("indicator_bollinger", "Bollinger"),
    ("indicator_atr", "ATR"),
    ("indicator_natr", "NATR"),
    ("indicator_rsi", "RSI"),
    ("indicator_structure", "Structure"),
]


def _safe_stats(series: pd.Series) -> dict[str, Any]:
    s = series.dropna()
    if s.empty:
        return {"n": 0, "success_rate": None, "mean_return": None, "median_return": None, "std": None}
    return {
        "n": int(len(s)),
        "success_rate": float(s.mean()),
        "mean_return": float(series.dropna().astype(float).mean()) if series.name and "outperformed" not in series.name else None,
    }


def horizon_performance(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        oc = f"outperformed_{h}h"
        rc = f"future_return_{h}h"
        if oc not in df.columns:
            continue
        sub = df.dropna(subset=[oc])
        rets = df[rc].dropna()
        rows.append({
            "horizon_h": h,
            "n_predictions": len(sub),
            "n_outperformed": int(sub[oc].sum()),
            "success_rate": float(sub[oc].mean()) if len(sub) else None,
            "mean_future_return": float(rets.mean()) if len(rets) else None,
            "median_future_return": float(rets.median()) if len(rets) else None,
            "std_future_return": float(rets.std()) if len(rets) else None,
            "pct_positive_returns": float((rets > 0).mean()) if len(rets) else None,
        })
    return pd.DataFrame(rows)


def probability_calibration(df: pd.DataFrame, prob_col: str = "probability_4h") -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        oc = f"outperformed_{h}h"
        pc = prob_col if h == 4 else f"probability_{h}h"
        if oc not in df.columns or pc not in df.columns:
            continue
        sub = df.dropna(subset=[pc, oc])
        for lo, hi in PROB_BUCKETS:
            bucket = sub[(sub[pc] >= lo) & (sub[pc] < hi)]
            if bucket.empty:
                rows.append({
                    "horizon_h": h, "bucket": f"{lo:.0%}-{hi:.0%}",
                    "n": 0, "predicted_prob_mean": None, "actual_success_rate": None, "brier": None,
                })
                continue
            hits = bucket[oc].astype(float)
            pred = bucket[pc].astype(float)
            brier = float(((pred - hits) ** 2).mean())
            rows.append({
                "horizon_h": h,
                "bucket": f"{int(lo*100)}-{int(min(hi,1)*100)}%",
                "n": len(bucket),
                "predicted_prob_mean": float(pred.mean()),
                "actual_success_rate": float(hits.mean()),
                "calibration_error": float(abs(pred.mean() - hits.mean())),
                "brier": brier,
            })
    return pd.DataFrame(rows)


def top_n_analysis(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n in (1, 3, 5):
        sub = df[df["rank"] <= n]
        for h in HORIZONS:
            oc = f"outperformed_{h}h"
            rc = f"future_return_{h}h"
            if oc not in sub.columns:
                continue
            valid = sub.dropna(subset=[oc])
            rets = valid[rc].dropna()
            rows.append({
                "top_n": n,
                "horizon_h": h,
                "n": len(valid),
                "success_rate": float(valid[oc].mean()) if len(valid) else None,
                "mean_return": float(rets.mean()) if len(rets) else None,
                "median_return": float(rets.median()) if len(rets) else None,
            })
    return pd.DataFrame(rows)


def factor_analysis(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, name in FACTOR_COLS:
        if col not in df.columns:
            continue
        sub = df.dropna(subset=[col])
        if sub.empty:
            continue
        try:
            sub = sub.copy()
            sub["quartile"] = pd.qcut(sub[col], 4, duplicates="drop")
        except ValueError:
            sub["quartile"] = "all"
        for h in HORIZONS:
            oc = f"outperformed_{h}h"
            rc = f"future_return_{h}h"
            if oc not in sub.columns:
                continue
            for q, g in sub.groupby("quartile", observed=True):
                valid = g.dropna(subset=[oc])
                rows.append({
                    "factor": name,
                    "quartile": str(q),
                    "horizon_h": h,
                    "n": len(valid),
                    "avg_factor_score": float(g[col].mean()),
                    "success_rate": float(valid[oc].mean()) if len(valid) else None,
                    "mean_future_return": float(g[rc].dropna().mean()) if rc in g.columns else None,
                })
    return pd.DataFrame(rows)


def indicator_analysis(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, name in INDICATOR_COLS:
        if col not in df.columns:
            continue
        sub = df.dropna(subset=[col])
        if len(sub) < 20:
            continue
        try:
            sub = sub.copy()
            sub["quartile"] = pd.qcut(sub[col], 4, duplicates="drop")
        except ValueError:
            continue
        for h in HORIZONS:
            oc = f"outperformed_{h}h"
            if oc not in sub.columns:
                continue
            for q, g in sub.groupby("quartile", observed=True):
                valid = g.dropna(subset=[oc])
                rc = f"future_return_{h}h"
                rows.append({
                    "indicator": name,
                    "quartile": str(q),
                    "horizon_h": h,
                    "n": len(valid),
                    "avg_indicator_score": float(g[col].mean()),
                    "success_rate": float(valid[oc].mean()) if len(valid) else None,
                    "mean_future_return": float(g[rc].dropna().mean()) if rc in g.columns else None,
                })
    return pd.DataFrame(rows)


def late_entry_analysis(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if "late_entry_score" not in df.columns:
        return pd.DataFrame(rows)
    for label, lo, hi in LATE_BUCKETS:
        sub = df[(df["late_entry_score"] >= lo) & (df["late_entry_score"] < hi)]
        for h in HORIZONS:
            oc = f"outperformed_{h}h"
            rc = f"future_return_{h}h"
            if oc not in sub.columns:
                continue
            valid = sub.dropna(subset=[oc])
            rets = valid[rc].dropna()
            rows.append({
                "late_entry_bucket": label,
                "horizon_h": h,
                "n": len(valid),
                "success_rate": float(valid[oc].mean()) if len(valid) else None,
                "mean_return": float(rets.mean()) if len(rets) else None,
                "median_return": float(rets.median()) if len(rets) else None,
            })
    # High prob + high late entry
    if "probability_4h" in df.columns:
        hi_p = df[df["probability_4h"] >= 0.70]
        for late_label, lo, hi in [("HIGH_LATE", 0.55, 0.75), ("VERY_HIGH_LATE", 0.75, 1.01)]:
            sub = hi_p[(hi_p["late_entry_score"] >= lo) & (hi_p["late_entry_score"] < hi)]
            for h in HORIZONS:
                oc = f"outperformed_{h}h"
                if oc not in sub.columns:
                    continue
                valid = sub.dropna(subset=[oc])
                rows.append({
                    "late_entry_bucket": f"P>=70% + {late_label}",
                    "horizon_h": h,
                    "n": len(valid),
                    "success_rate": float(valid[oc].mean()) if len(valid) else None,
                    "mean_return": float(sub[f"future_return_{h}h"].dropna().mean()) if f"future_return_{h}h" in sub.columns else None,
                    "median_return": float(sub[f"future_return_{h}h"].dropna().median()) if f"future_return_{h}h" in sub.columns else None,
                })
    return pd.DataFrame(rows)


def btc_regime_analysis(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if "indicator_btc_regime" not in df.columns:
        return pd.DataFrame(rows)
    sub = df.dropna(subset=["indicator_btc_regime"])
    sub = sub.copy()
    sub["regime"] = pd.cut(
        sub["indicator_btc_regime"],
        bins=[0, 0.4, 0.6, 1.01],
        labels=["BTC_weak", "BTC_neutral", "BTC_strong"],
    )
    for regime, g in sub.groupby("regime", observed=True):
        for h in HORIZONS:
            oc = f"outperformed_{h}h"
            if oc not in g.columns:
                continue
            valid = g.dropna(subset=[oc])
            rows.append({
                "btc_regime": str(regime),
                "horizon_h": h,
                "n": len(valid),
                "success_rate": float(valid[oc].mean()) if len(valid) else None,
                "mean_return": float(g[f"future_return_{h}h"].dropna().mean()) if f"future_return_{h}h" in g.columns else None,
            })
    return pd.DataFrame(rows)


def coin_analysis(df: pd.DataFrame) -> pd.DataFrame:
    if "base" not in df.columns:
        return pd.DataFrame()
    rows = []
    for base, g in df.groupby("base"):
        row = {
            "base": base,
            "top5_appearances": len(g),
            "avg_probability_4h": float(g["probability_4h"].mean()) if "probability_4h" in g.columns else None,
            "avg_late_entry": float(g["late_entry_score"].mean()) if "late_entry_score" in g.columns else None,
        }
        for h in HORIZONS:
            oc = f"outperformed_{h}h"
            rc = f"future_return_{h}h"
            if oc in g.columns:
                valid = g.dropna(subset=[oc])
                row[f"success_rate_{h}h"] = float(valid[oc].mean()) if len(valid) else None
            if rc in g.columns:
                row[f"mean_return_{h}h"] = float(g[rc].dropna().mean()) if g[rc].notna().any() else None
        rows.append(row)
    return pd.DataFrame(rows).sort_values("top5_appearances", ascending=False)


def run_all_analyses(df: pd.DataFrame, out_dir: Path) -> dict[str, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "horizon_performance": horizon_performance(df),
        "probability_calibration": probability_calibration(df),
        "top_n_analysis": top_n_analysis(df),
        "factor_analysis": factor_analysis(df),
        "indicator_analysis": indicator_analysis(df),
        "late_entry_analysis": late_entry_analysis(df),
        "btc_regime_analysis": btc_regime_analysis(df),
        "coin_analysis": coin_analysis(df),
    }
    for name, frame in results.items():
        if not frame.empty:
            frame.to_csv(out_dir / f"{name}.csv", index=False)
    return results
