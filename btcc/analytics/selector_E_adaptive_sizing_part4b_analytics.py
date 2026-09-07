"""Analytics, plots, and research report for Part 4B causal adaptive sizing."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from btcc.sim.selector_E_adaptive_sizing import BASE_ALLOCATION, STARTING_CAPITAL_USD
from btcc.sim.selector_E_adaptive_sizing_part4b import (
    ARM_ORDER,
    PART4A_ARM_MAP,
    SCARCITY_BUCKETS,
)

logger = logging.getLogger(__name__)


def _md_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if df is None or df.empty:
        return "_(empty)_"
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(str(c) for c in cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append("—" if not np.isfinite(v) else format(float(v), floatfmt))
            elif v is None:
                cells.append("—")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _max_dd(eq: pd.Series) -> float:
    if eq.empty:
        return 0.0
    return float(100.0 * (eq / eq.cummax() - 1.0).min())


def _seq_pnl(pnl: pd.Series, n: int) -> float:
    if len(pnl) < n:
        return float(pnl.sum()) if len(pnl) else 0.0
    return float(pnl.rolling(n).sum().min())


def enrich_summaries(trade_tables: dict[str, pd.DataFrame], summaries: list[dict]) -> pd.DataFrame:
    rows = []
    for sm in summaries:
        arm = sm["arm"]
        df = trade_tables[arm]
        ex = df[df["executed"] == True].copy() if not df.empty else df  # noqa: E712
        sm = dict(sm)
        if not ex.empty:
            ex = ex.sort_values("exit_ts")
            eq = STARTING_CAPITAL_USD + ex["pnl_usd"].cumsum()
            sm["max_drawdown_pct"] = _max_dd(eq)
            sm["avg_drawdown_pct"] = float(100.0 * ((eq / eq.cummax() - 1.0).mean()))
            sm["largest_single_trade_impact_usd"] = float(ex["pnl_usd"].min())
            sm["largest_single_trade_impact_pct_equity"] = float(
                (ex["pnl_usd"] / ex["equity_before"]).min() * 100.0
            )
            sm["worst_5_trade_seq_usd"] = _seq_pnl(ex["pnl_usd"], 5)
            sm["worst_10_trade_seq_usd"] = _seq_pnl(ex["pnl_usd"], 10)
            sm["final_equity_usd"] = float(eq.iloc[-1])
            sm["total_return_pct"] = 100.0 * (float(eq.iloc[-1]) / STARTING_CAPITAL_USD - 1.0)
        rows.append(sm)
    return pd.DataFrame(rows).set_index("arm").reindex(ARM_ORDER).reset_index()


def build_daily(trade_tables: dict[str, pd.DataFrame], daily_counts: pd.Series) -> pd.DataFrame:
    rows = []
    for arm, df in trade_tables.items():
        ex = df[df["executed"] == True].copy() if not df.empty else df  # noqa: E712
        if ex.empty:
            continue
        ex = ex.sort_values("exit_ts")
        equity = STARTING_CAPITAL_USD
        ex["exit_day"] = pd.to_datetime(ex["exit_ts"], utc=True).dt.floor("D")
        for day, g in ex.groupby("exit_day", sort=True):
            day_pnl = float(g["pnl_usd"].sum())
            start_eq = equity
            equity = start_eq + day_pnl
            rows.append(
                {
                    "arm": arm,
                    "day": day,
                    "starting_equity": start_eq,
                    "ending_equity": equity,
                    "daily_pnl": day_pnl,
                    "daily_return_pct": 100.0 * day_pnl / start_eq if start_eq else 0.0,
                    "n_trades": int(len(g)),
                    "avg_allocation": float(g["requested_allocation"].mean()),
                    "avg_nstar": float(pd.to_numeric(g["nstar"], errors="coerce").mean()),
                    "avg_n7": float(pd.to_numeric(g["n7"], errors="coerce").mean()),
                    "avg_scarcity_ratio": float(
                        pd.to_numeric(g["scarcity_ratio"], errors="coerce").replace(
                            [np.inf, -np.inf], np.nan
                        ).mean()
                    ),
                    "opp_count_day": float(daily_counts.get(day, 0.0)),
                }
            )
    return pd.DataFrame(rows)


def build_allocation_distribution(trade_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for arm, df in trade_tables.items():
        ex = df[df["executed"] == True] if not df.empty else df  # noqa: E712
        if ex.empty:
            continue
        a = ex["requested_allocation"]
        desc = a.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95])
        rows.append(
            {
                "arm": arm,
                "mean": float(desc["mean"]),
                "median": float(desc["50%"]),
                "p25": float(desc["25%"]),
                "p75": float(desc["75%"]),
                "p90": float(desc["90%"]),
                "p95": float(desc["95%"]),
                "max": float(desc["max"]),
                "pct_exactly_10": 100.0 * float((a <= BASE_ALLOCATION + 1e-12).mean()),
                "pct_above_10": 100.0 * float((a > BASE_ALLOCATION + 1e-12).mean()),
                "pct_at_25_cap": 100.0 * float((a >= 0.25 - 1e-12).mean()),
            }
        )
    return pd.DataFrame(rows)


def build_scarcity_ratio_analysis(
    trade_tables: dict[str, pd.DataFrame],
    hist: pd.DataFrame,
    fixed_trades: pd.DataFrame,
) -> pd.DataFrame:
    """Bucket analysis on R_D = N*/N7 (day-level), plus trade outcomes under FIXED_10."""
    h = hist.copy()
    h["date"] = pd.to_datetime(h["date"], utc=True)
    h["r"] = pd.to_numeric(h["scarcity_ratio_N7"], errors="coerce").replace([np.inf, -np.inf], np.nan)

    fx = fixed_trades[fixed_trades["executed"] == True].copy()  # noqa: E712
    fx["exit_day"] = pd.to_datetime(fx["exit_ts"], utc=True).dt.floor("D")
    fx["entry_day"] = pd.to_datetime(fx["entry_ts"], utc=True).dt.floor("D")

    rows = []
    for lab, lo, hi in SCARCITY_BUCKETS:
        if hi == float("inf"):
            mask = h["r"] > lo
        elif lo == 0.0 and lab.startswith("R<"):
            mask = h["r"] < hi
        else:
            mask = (h["r"] >= lo) & (h["r"] < hi)
        days = h.loc[mask, "date"]
        day_set = set(days)
        # FIXED_10 trades whose *entry* day falls in bucket (scarcity at decision time)
        g = fx[fx["entry_day"].isin(day_set)] if len(day_set) else fx.iloc[0:0]
        pnl = g["pnl_usd"] if len(g) else pd.Series(dtype=float)
        win = (pnl > 0).mean() * 100.0 if len(pnl) else float("nan")
        # path DD on bucket trades only
        if len(pnl):
            path = STARTING_CAPITAL_USD + pnl.sort_index().cumsum()
            dd = _max_dd(path)
        else:
            dd = float("nan")
        # allocations for adaptive arms on those entry days
        alloc_cols = {}
        for arm, df in trade_tables.items():
            ex = df[df["executed"] == True].copy()  # noqa: E712
            if ex.empty:
                continue
            ex["entry_day"] = pd.to_datetime(ex["entry_ts"], utc=True).dt.floor("D")
            sub = ex[ex["entry_day"].isin(day_set)]
            alloc_cols[f"avg_alloc_{arm}"] = float(sub["requested_allocation"].mean()) if len(sub) else np.nan

        row = {
            "bucket": lab,
            "n_days": int(mask.sum()),
            "avg_opp_count": float(h.loc[mask, "opp_count_day"].mean()) if mask.any() else np.nan,
            "avg_Nstar": float(h.loc[mask, "historical_normal_Nstar"].mean()) if mask.any() else np.nan,
            "avg_N7": float(h.loc[mask, "recent_N7"].mean()) if mask.any() else np.nan,
            "avg_R": float(h.loc[mask, "r"].mean()) if mask.any() else np.nan,
            "n_fixed10_trades": int(len(g)),
            "fixed10_win_rate_pct": float(win) if np.isfinite(win) else np.nan,
            "fixed10_avg_pnl_usd": float(pnl.mean()) if len(pnl) else np.nan,
            "fixed10_avg_pnl_pct": float(g["pnl_pct"].mean() * 100.0) if len(g) else np.nan,
            "fixed10_bucket_path_dd_pct": dd,
            "fixed10_total_pnl_usd": float(pnl.sum()) if len(pnl) else 0.0,
            **alloc_cols,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def scarcity_predicts_performance(fixed_trades: pd.DataFrame) -> dict[str, Any]:
    """Diagnostic only: association of entry-time scarcity with subsequent E PnL."""
    ex = fixed_trades[fixed_trades["executed"] == True].copy()  # noqa: E712
    if ex.empty:
        return {}
    ex["r"] = pd.to_numeric(ex["scarcity_ratio"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    # For FIXED_10 scarcity_ratio is nan — use r_n7 from features stored on trades
    if ex["r"].isna().all() and "r_n7" in ex.columns:
        ex["r"] = pd.to_numeric(ex["r_n7"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = ex.dropna(subset=["r"])
    if len(valid) < 10:
        return {"n": int(len(valid)), "note": "insufficient finite scarcity ratios"}
    scarce = valid[valid["r"] > 1.0]
    abundant = valid[valid["r"] <= 1.0]
    return {
        "n_trades_with_finite_R": int(len(valid)),
        "n_scarce_R_gt_1": int(len(scarce)),
        "n_abundant_R_le_1": int(len(abundant)),
        "scarce_win_rate_pct": float((scarce["pnl_usd"] > 0).mean() * 100.0) if len(scarce) else np.nan,
        "abundant_win_rate_pct": float((abundant["pnl_usd"] > 0).mean() * 100.0) if len(abundant) else np.nan,
        "scarce_avg_pnl_pct": float(scarce["pnl_pct"].mean() * 100.0) if len(scarce) else np.nan,
        "abundant_avg_pnl_pct": float(abundant["pnl_pct"].mean() * 100.0) if len(abundant) else np.nan,
        "corr_R_vs_pnl_pct": float(valid["r"].corr(valid["pnl_pct"])) if len(valid) > 2 else np.nan,
        "scarce_vs_abundant_avg_pnl_pct_diff": (
            float(scarce["pnl_pct"].mean() - abundant["pnl_pct"].mean()) * 100.0
            if len(scarce) and len(abundant)
            else np.nan
        ),
    }


def build_drawdown_analysis(trade_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for arm, df in trade_tables.items():
        ex = df[df["executed"] == True].copy() if not df.empty else df  # noqa: E712
        if ex.empty:
            continue
        ex = ex.sort_values("exit_ts").reset_index(drop=True)
        eq = STARTING_CAPITAL_USD + ex["pnl_usd"].cumsum()
        dd = 100.0 * (eq / eq.cummax() - 1.0)
        trough_i = int(dd.idxmin()) if len(dd) else 0
        window = ex.iloc[max(0, trough_i - 10) : trough_i + 1]
        rows.append(
            {
                "arm": arm,
                "max_drawdown_pct": float(dd.min()),
                "avg_drawdown_pct": float(dd.mean()),
                "largest_single_trade_impact_usd": float(ex["pnl_usd"].min()),
                "largest_single_trade_impact_pct": float(
                    (ex["pnl_usd"] / ex["equity_before"]).min() * 100.0
                ),
                "largest_requested_allocation": float(ex["requested_allocation"].max()),
                "largest_actual_allocation": float(ex["actual_allocation"].max()),
                "max_simultaneous_exposure": float(
                    ((ex["reserved_before"] + ex["actual_notional"]) / ex["equity_before"]).max()
                ),
                "worst_5_trade_seq_usd": _seq_pnl(ex["pnl_usd"], 5),
                "worst_10_trade_seq_usd": _seq_pnl(ex["pnl_usd"], 10),
                "alloc_before_max_dd_trough_mean": float(window["requested_allocation"].mean())
                if len(window)
                else np.nan,
                "scarcity_before_max_dd_trough_mean": float(
                    pd.to_numeric(window.get("scarcity_ratio"), errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                    .mean()
                )
                if len(window)
                else np.nan,
                "corr_alloc_vs_pnl": float(ex["requested_allocation"].corr(ex["pnl_usd"]))
                if len(ex) > 2
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_time_stability(trade_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for arm, df in trade_tables.items():
        ex = df[df["executed"] == True].copy() if not df.empty else df  # noqa: E712
        if ex.empty:
            continue
        ex = ex.sort_values("exit_ts")
        ex["month"] = pd.to_datetime(ex["exit_ts"], utc=True).dt.tz_localize(None).dt.to_period("M").astype(str)
        equity = STARTING_CAPITAL_USD
        for month, g in ex.groupby("month", sort=True):
            start_eq = equity
            pnl = float(g["pnl_usd"].sum())
            equity = start_eq + pnl
            path = start_eq + g["pnl_usd"].cumsum()
            rows.append(
                {
                    "arm": arm,
                    "month": month,
                    "n_trades": int(len(g)),
                    "monthly_return_pct": 100.0 * pnl / start_eq if start_eq else 0.0,
                    "ending_equity": equity,
                    "cumulative_return_pct": 100.0 * (equity / STARTING_CAPITAL_USD - 1.0),
                    "max_drawdown_pct": _max_dd(path),
                    "avg_allocation": float(g["requested_allocation"].mean()),
                    "avg_scarcity_ratio": float(
                        pd.to_numeric(g.get("scarcity_ratio"), errors="coerce")
                        .replace([np.inf, -np.inf], np.nan)
                        .mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_exposure_analysis(trade_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for arm, df in trade_tables.items():
        ex = df[df["executed"] == True].copy() if not df.empty else df  # noqa: E712
        if ex.empty:
            continue
        exposure = (ex["reserved_before"] + ex["actual_notional"]) / ex["equity_before"]
        r = pd.to_numeric(ex.get("scarcity_ratio"), errors="coerce").replace([np.inf, -np.inf], np.nan)
        scarce = ex[r > 1.0] if r.notna().any() else ex.iloc[0:0]
        abundant = ex[r <= 1.0] if r.notna().any() else ex.iloc[0:0]
        rows.append(
            {
                "arm": arm,
                "avg_exposure": float(exposure.mean()),
                "max_exposure": float(exposure.max()),
                "n_exposure_limited": int(ex["exposure_limited"].sum()),
                "n_cap_25": int(ex["cap_25_hit"].sum()),
                "scarce_R_gt1_n_trades": int(len(scarce)),
                "scarce_avg_allocation": float(scarce["requested_allocation"].mean()) if len(scarce) else np.nan,
                "scarce_total_pnl_usd": float(scarce["pnl_usd"].sum()) if len(scarce) else 0.0,
                "abundant_R_le1_n_trades": int(len(abundant)),
                "abundant_avg_allocation": float(abundant["requested_allocation"].mean())
                if len(abundant)
                else np.nan,
                "abundant_total_pnl_usd": float(abundant["pnl_usd"].sum()) if len(abundant) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_comparison(summary: pd.DataFrame, ref_metrics: dict[str, Any]) -> pd.DataFrame:
    fixed = summary[summary["arm"] == "FIXED_10"].iloc[0]
    rows = []
    for _, r in summary.iterrows():
        rows.append(
            {
                "arm": r["arm"],
                "final_equity_usd": r["final_equity_usd"],
                "total_return_pct": r["total_return_pct"],
                "vs_fixed10_equity_usd": float(r["final_equity_usd"] - fixed["final_equity_usd"]),
                "vs_fixed10_return_pp": float(r["total_return_pct"] - fixed["total_return_pct"]),
                "max_drawdown_pct": r["max_drawdown_pct"],
                "vs_fixed10_dd_pp": float(r["max_drawdown_pct"] - fixed["max_drawdown_pct"]),
                "profit_factor": r["profit_factor"],
                "win_rate_pct": r["win_rate_pct"],
                "avg_allocation": r["avg_allocation"],
                "pct_above_10": r.get("pct_above_10"),
                "pct_cap_25_binding": r.get("pct_cap_25_binding"),
                "max_exposure": r["max_exposure"],
                "n_exposure_limited": r["n_exposure_limited"],
                "reference_E_final_equity_usd": ref_metrics.get("final_equity_usd"),
            }
        )
    return pd.DataFrame(rows)


def load_part4a_summary(part4a_dir: Path | None) -> pd.DataFrame | None:
    if part4a_dir is None:
        return None
    p = Path(part4a_dir) / "adaptive_sizing_summary.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def build_part4a_vs_part4b(summary: pd.DataFrame, part4a: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    if part4a is None:
        return pd.DataFrame(rows)
    p4a = part4a.set_index("arm")
    for arm_4b, arm_4a in PART4A_ARM_MAP.items():
        if arm_4b not in set(summary["arm"]) or arm_4a not in p4a.index:
            continue
        b = summary[summary["arm"] == arm_4b].iloc[0]
        a = p4a.loc[arm_4a]
        rows.append(
            {
                "part4b_arm": arm_4b,
                "part4a_arm": arm_4a,
                "part4a_return_pct": float(a["total_return_pct"]),
                "part4b_return_pct": float(b["total_return_pct"]),
                "return_delta_pp": float(b["total_return_pct"] - a["total_return_pct"]),
                "part4a_equity": float(a["final_equity_usd"]),
                "part4b_equity": float(b["final_equity_usd"]),
                "part4a_max_dd_pct": float(a["max_drawdown_pct"]),
                "part4b_max_dd_pct": float(b["max_drawdown_pct"]),
                "part4a_avg_alloc": float(a["avg_allocation"]),
                "part4b_avg_alloc": float(b["avg_allocation"]),
                "part4a_pct_cap_25": float(a.get("pct_cap_25_binding", np.nan)),
                "part4b_pct_cap_25": float(b.get("pct_cap_25_binding", np.nan)),
            }
        )
    # Also FIXED_10 sanity
    if "FIXED_10" in p4a.index:
        b = summary[summary["arm"] == "FIXED_10"].iloc[0]
        a = p4a.loc["FIXED_10"]
        rows.insert(
            0,
            {
                "part4b_arm": "FIXED_10",
                "part4a_arm": "FIXED_10",
                "part4a_return_pct": float(a["total_return_pct"]),
                "part4b_return_pct": float(b["total_return_pct"]),
                "return_delta_pp": float(b["total_return_pct"] - a["total_return_pct"]),
                "part4a_equity": float(a["final_equity_usd"]),
                "part4b_equity": float(b["final_equity_usd"]),
                "part4a_max_dd_pct": float(a["max_drawdown_pct"]),
                "part4b_max_dd_pct": float(b["max_drawdown_pct"]),
                "part4a_avg_alloc": float(a["avg_allocation"]),
                "part4b_avg_alloc": float(b["avg_allocation"]),
                "part4a_pct_cap_25": 0.0,
                "part4b_pct_cap_25": 0.0,
            },
        )
    return pd.DataFrame(rows)


def classify_arms(summary: pd.DataFrame, time_stab: pd.DataFrame, alloc_dist: pd.DataFrame) -> dict[str, dict]:
    fixed = summary[summary["arm"] == "FIXED_10"].iloc[0]
    ad = alloc_dist.set_index("arm") if not alloc_dist.empty else None
    out = {}
    for _, r in summary.iterrows():
        arm = r["arm"]
        if arm == "FIXED_10":
            out[arm] = {"class": "CONTROL", "note": "Baseline matching Selector E reference."}
            continue
        d_ret = float(r["total_return_pct"] - fixed["total_return_pct"])
        d_dd = float(r["max_drawdown_pct"] - fixed["max_drawdown_pct"])
        dd_worsen = -d_dd
        cap_pct = float(r.get("pct_cap_25_binding", 0.0) or 0.0)
        pct_above = float(r.get("pct_above_10", 0.0) or 0.0)
        if ad is not None and arm in ad.index:
            pct_above = float(ad.loc[arm, "pct_above_10"])
        mt = time_stab[time_stab["arm"] == arm]
        ft = time_stab[time_stab["arm"] == "FIXED_10"]
        months_beat = n_m = 0
        if not mt.empty and not ft.empty:
            m = mt.merge(ft[["month", "monthly_return_pct"]], on="month", suffixes=("", "_fixed"))
            n_m = len(m)
            months_beat = int((m["monthly_return_pct"] > m["monthly_return_pct_fixed"]).sum())

        permanent_upsize = pct_above >= 70.0
        heavy_cap = cap_pct >= 20.0
        unstable = n_m > 0 and months_beat < n_m * 0.4
        reject = d_ret < -1.0 or (dd_worsen >= 4.0 and d_ret < 5.0)
        promising = (
            d_ret >= 3.0
            and dd_worsen <= 1.5
            and not heavy_cap
            and not permanent_upsize
            and not unstable
        )
        conditional = d_ret >= 1.0 and (heavy_cap or permanent_upsize or dd_worsen > 1.0 or unstable)

        if reject:
            cls, note = "D", "Reject: worse equity and/or excessive DD."
        elif promising:
            cls, note = "A", "Promising: equity gain with controlled DD and non-permanent oversizing."
        elif conditional:
            cls, note = "B", "Interesting but conditional/risky."
        elif d_ret >= 0.5:
            cls, note = "C", "Little advantage after correcting the normal-frequency assumption."
        else:
            cls, note = "C", "No meaningful improvement over FIXED_10."
        out[arm] = {
            "class": cls,
            "note": note,
            "delta_return_pp": f"{d_ret:.3f}",
            "delta_dd_pp": f"{d_dd:.3f}",
            "pct_above_10": f"{pct_above:.1f}",
            "months_beat_fixed": f"{months_beat}/{n_m}" if n_m else "n/a",
        }
    return out


def plot_all(
    out_dir: Path,
    trade_tables: dict[str, pd.DataFrame],
    summary: pd.DataFrame,
    time_stab: pd.DataFrame,
    hist: pd.DataFrame,
    scarcity: pd.DataFrame,
) -> dict[str, str]:
    plots = out_dir / "plots"
    paths = {}

    fig, ax = plt.subplots(figsize=(12, 6))
    for arm in ARM_ORDER:
        ex = trade_tables[arm]
        ex = ex[ex["executed"] == True].sort_values("exit_ts")  # noqa: E712
        eq = STARTING_CAPITAL_USD + ex["pnl_usd"].cumsum()
        ax.plot(pd.to_datetime(ex["exit_ts"], utc=True), eq, label=arm, linewidth=1.6 if arm == "FIXED_10" else 1.0)
    ax.axhline(STARTING_CAPITAL_USD, color="grey", ls="--", lw=0.8)
    ax.set_title("Part 4B — Causal adaptive sizing equity curves")
    ax.set_ylabel("Equity (USD)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = plots / "equity_curves.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["equity_curves.png"] = str(p)

    # allocation vs scarcity
    fig, ax = plt.subplots(figsize=(10, 6))
    for arm, color in (
        ("7D_INVERSE_CAUSAL", "C1"),
        ("7D_SQRT_CAUSAL", "C2"),
        ("7D_MEDIAN_SQRT_CAUSAL", "C3"),
        ("7_30_SQRT_CAUSAL", "C4"),
        ("DEADBAND_CAUSAL", "C5"),
    ):
        ex = trade_tables[arm]
        ex = ex[ex["executed"] == True].copy()  # noqa: E712
        r = pd.to_numeric(ex["scarcity_ratio"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        ax.scatter(r, ex["requested_allocation"] * 100, s=8, alpha=0.35, label=arm, c=color)
    ax.axhline(10, color="black", ls="--", lw=0.8)
    ax.axhline(25, color="red", ls=":", lw=0.8)
    ax.axvline(1.0, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("Scarcity ratio R = N*/N_recent")
    ax.set_ylabel("Requested allocation %")
    ax.set_title("Allocation vs causal scarcity ratio")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = plots / "allocation_vs_scarcity.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["allocation_vs_scarcity.png"] = str(p)

    # historical normal vs recent
    fig, ax = plt.subplots(figsize=(12, 5))
    h = hist.copy()
    h["date"] = pd.to_datetime(h["date"], utc=True)
    ax.plot(h["date"], h["historical_normal_Nstar"], label="N* (expanding)", lw=1.5)
    ax.plot(h["date"], h["recent_N7"], label="N7", lw=1.0, alpha=0.8)
    ax.plot(h["date"], h["recent_N30"], label="N30", lw=1.0, alpha=0.8)
    ax.axhline(10, color="red", ls=":", lw=1.0, label="Part 4A fixed normal=10")
    ax.set_ylabel("Opportunities / day")
    ax.set_title("Causal historical normal N* vs recent frequency")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = plots / "historical_normal_vs_recent_frequency.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["historical_normal_vs_recent_frequency.png"] = str(p)

    # allocation distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    data, labels = [], []
    for arm in ARM_ORDER:
        ex = trade_tables[arm]
        ex = ex[ex["executed"] == True]  # noqa: E712
        data.append(ex["requested_allocation"] * 100)
        labels.append(arm)
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_ylabel("Requested allocation %")
    ax.set_title("Allocation distribution (Part 4B)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p = plots / "allocation_distribution.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["allocation_distribution.png"] = str(p)

    # drawdown
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(summary["arm"], summary["max_drawdown_pct"], color=["#333"] + ["steelblue"] * (len(summary) - 1))
    ax.set_ylabel("Max drawdown %")
    ax.set_title("Maximum drawdown by arm (Part 4B)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p = plots / "drawdown_comparison.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["drawdown_comparison.png"] = str(p)

    # monthly
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot = time_stab.pivot(index="month", columns="arm", values="monthly_return_pct")
    focus = [c for c in ARM_ORDER if c in pivot.columns]
    pivot[focus].plot(ax=ax, marker="o", lw=1.1)
    ax.axhline(0, color="grey", ls="--", lw=0.8)
    ax.set_ylabel("Monthly return %")
    ax.set_title("Monthly returns — Part 4B")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = plots / "monthly_comparison.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["monthly_comparison.png"] = str(p)

    # scarcity performance
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    if not scarcity.empty:
        axes[0].bar(scarcity["bucket"], scarcity["fixed10_avg_pnl_pct"], color="steelblue")
        axes[0].axhline(0, color="grey", ls="--", lw=0.8)
        axes[0].set_ylabel("Avg trade return % (FIXED_10)")
        axes[0].set_title("Scarcity bucket vs subsequent E PnL")
        axes[0].tick_params(axis="x", rotation=30)
        axes[1].bar(scarcity["bucket"], scarcity["fixed10_win_rate_pct"], color="darkorange")
        axes[1].axhline(50, color="grey", ls="--", lw=0.8)
        axes[1].set_ylabel("Win rate % (FIXED_10)")
        axes[1].set_title("Scarcity bucket vs win rate")
        axes[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    p = plots / "scarcity_performance.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["scarcity_performance.png"] = str(p)

    return paths


def write_report(
    out_dir: Path,
    *,
    result: dict[str, Any],
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    alloc_dist: pd.DataFrame,
    scarcity: pd.DataFrame,
    dd: pd.DataFrame,
    time_stab: pd.DataFrame,
    exposure: pd.DataFrame,
    part4a_vs: pd.DataFrame,
    scarcity_diag: dict[str, Any],
    classifications: dict[str, dict],
    plot_paths: dict[str, str],
) -> Path:
    baseline = result["baseline_check"]
    lines = []
    lines.append("# RESEARCH REPORT — Selector E Causal Adaptive Sizing (Part 4B)")
    lines.append("")
    lines.append(f"**Generated:** {result.get('generated_at')}")
    lines.append(f"**Reference:** `{result.get('ref_dir')}`")
    lines.append(f"**Results:** `{out_dir}`")
    lines.append("")
    lines.append("## Correction from Part 4A")
    lines.append("")
    lines.append(
        "Part 4A assumed normal frequency = 10 opps/day. This stream averages ~4.2/day, "
        "so Part 4A arms were nearly always oversized. Part 4B replaces that with a "
        "**causal expanding historical normal** N*_D."
    )
    lines.append("")
    lines.append(f"**N* definition:** {result.get('nstar_definition')}")
    lines.append("")
    lines.append(f"**Warmup:** {result.get('warmup_policy')}")
    lines.append("")
    lines.append(f"**Zero-frequency:** {result.get('zero_frequency_policy')}")
    lines.append("")
    lines.append(f"**Sizing:** {result.get('sizing_convention')}")
    lines.append("")
    lines.append("## Baseline check")
    lines.append("")
    lines.append(f"- Passed: **{baseline.get('passed')}**")
    lines.append(f"- FIXED_10 sim match: **{baseline.get('fixed10_sim_matches_replay')}**")
    f10, ref = baseline["fixed10"], baseline["reference"]
    lines.append("")
    lines.append("| Metric | FIXED_10 | Reference E |")
    lines.append("| --- | ---: | ---: |")
    for k in (
        "n_trades",
        "final_equity_usd",
        "cumulative_return_pct",
        "win_rate_pct",
        "profit_factor",
    ):
        lines.append(f"| {k} | {f10.get(k)} | {ref.get(k)} |")
    lines.append("")
    cc = result["causality_check"]
    lines.append(
        f"### Causality audit: N* excludes day D — ok={cc.get('ok')}, "
        f"violations={cc.get('violations')}, zeros_included={cc.get('zeros_included_in_nstar')}"
    )
    lines.append("")
    lines.append("## Primary comparison")
    lines.append("")
    lines.append(_md_table(summary))
    lines.append("")
    lines.append("### vs FIXED_10")
    lines.append("")
    lines.append(_md_table(comparison))
    lines.append("")
    lines.append("## Part 4A vs Part 4B")
    lines.append("")
    lines.append(_md_table(part4a_vs) if not part4a_vs.empty else "_(Part 4A summary not found)_")
    lines.append("")
    lines.append("## Allocation distribution")
    lines.append("")
    lines.append(_md_table(alloc_dist))
    lines.append("")
    lines.append("## Scarcity ratio buckets (R = N*/N7)")
    lines.append("")
    lines.append(_md_table(scarcity))
    lines.append("")
    lines.append("## Does scarcity predict E performance? (diagnostic)")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(scarcity_diag, indent=2, default=str))
    lines.append("```")
    lines.append("")
    lines.append("## Exposure")
    lines.append("")
    lines.append(_md_table(exposure))
    lines.append("")
    lines.append("## Drawdown / concentration")
    lines.append("")
    lines.append(_md_table(dd))
    lines.append("")
    lines.append("## Time stability")
    lines.append("")
    lines.append(_md_table(time_stab))
    lines.append("")
    lines.append("## Classifications")
    lines.append("")
    lines.append("| Arm | Class | Note | Δ ret pp | Δ DD pp | % alloc>10% | Months beat |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | --- |")
    for arm in ARM_ORDER:
        c = classifications[arm]
        lines.append(
            f"| {arm} | {c.get('class')} | {c.get('note')} | {c.get('delta_return_pp','')} | "
            f"{c.get('delta_dd_pp','')} | {c.get('pct_above_10','')} | {c.get('months_beat_fixed','')} |"
        )
    lines.append("")

    fixed_row = summary[summary["arm"] == "FIXED_10"].iloc[0]
    ranked = summary[summary["arm"] != "FIXED_10"].sort_values("final_equity_usd", ascending=False)
    best = ranked.iloc[0]

    def ans(q, a):
        lines.append(f"### {q}")
        lines.append("")
        lines.append(a)
        lines.append("")

    # N* stats
    hist = result["causal_frequency_history"]
    nstar_vals = pd.to_numeric(hist["historical_normal_Nstar"], errors="coerce").dropna()
    ans(
        "1. How does the causal N*_D differ from the fixed 10-opportunity assumption?",
        f"Expanding N* mean≈{float(nstar_vals.mean()):.2f}, median≈{float(nstar_vals.median()):.2f}, "
        f"final≈{float(nstar_vals.iloc[-1]):.2f} vs Part 4A fixed normal=10. "
        f"N* is ~{10/float(nstar_vals.mean()):.1f}× smaller than the Part 4A assumption on average.",
    )

    inv = summary[summary["arm"] == "7D_INVERSE_CAUSAL"].iloc[0]
    p4_inv = part4a_vs[part4a_vs["part4b_arm"] == "7D_INVERSE_CAUSAL"]
    p4_note = (
        f" Part 4A was {float(p4_inv.iloc[0]['part4a_return_pct']):.1f}% → Part 4B "
        f"{float(inv['total_return_pct']):.1f}% (Δ {float(p4_inv.iloc[0]['return_delta_pp']):.1f} pp)."
        if len(p4_inv)
        else ""
    )
    ans(
        "2. Does corrected 7D_INVERSE still produce a large improvement?",
        f"7D_INVERSE_CAUSAL return {inv['total_return_pct']:.2f}% vs FIXED_10 {fixed_row['total_return_pct']:.2f}% "
        f"({inv['total_return_pct']-fixed_row['total_return_pct']:+.2f} pp).{p4_note}",
    )

    ans(
        "3. Does corrected inverse sizing remain too aggressive?",
        f"Cap binding {inv['pct_cap_25_binding']:.1f}%; avg allocation {inv['avg_allocation']*100:.1f}%; "
        f"% above 10% = {inv.get('pct_above_10', float('nan')):.1f}%; DD {inv['max_drawdown_pct']:.3f}% "
        f"vs FIXED_10 {fixed_row['max_drawdown_pct']:.3f}%.",
    )

    for q, arm in (
        ("4. Does corrected 7D_SQRT remain promising?", "7D_SQRT_CAUSAL"),
        ("5. Does corrected 7D_MEDIAN_SQRT remain promising?", "7D_MEDIAN_SQRT_CAUSAL"),
    ):
        r = summary[summary["arm"] == arm].iloc[0]
        ans(
            q,
            f"{arm}: return {r['total_return_pct']:.2f}% (Δ {r['total_return_pct']-fixed_row['total_return_pct']:+.2f} pp), "
            f"DD {r['max_drawdown_pct']:.3f}%, avg alloc {r['avg_allocation']*100:.1f}%, "
            f"class {classifications[arm]['class']}.",
        )

    blend = summary[summary["arm"] == "7_30_SQRT_CAUSAL"].iloc[0]
    s7 = summary[summary["arm"] == "7D_SQRT_CAUSAL"].iloc[0]
    ans(
        "6. Does 7/30 blending improve stability?",
        f"7_30_SQRT_CAUSAL {blend['total_return_pct']:.2f}% / DD {blend['max_drawdown_pct']:.3f}% vs "
        f"7D_SQRT_CAUSAL {s7['total_return_pct']:.2f}% / DD {s7['max_drawdown_pct']:.3f}%.",
    )

    dead = summary[summary["arm"] == "DEADBAND_CAUSAL"].iloc[0]
    ans(
        "7. Does the dead-band improve behavior?",
        f"DEADBAND_CAUSAL {dead['total_return_pct']:.2f}% / DD {dead['max_drawdown_pct']:.3f}% / "
        f"% at 10% floor≈{dead.get('pct_at_10', float('nan')):.1f}% vs blend "
        f"{blend['total_return_pct']:.2f}% / DD {blend['max_drawdown_pct']:.3f}%.",
    )

    ans(
        "8. Is low opportunity frequency actually associated with better or worse subsequent E performance?",
        f"Diagnostic (FIXED_10 trades by entry-time R): {json.dumps(scarcity_diag, default=str)}. "
        "See scarcity_performance.png / scarcity_ratio_analysis.csv.",
    )

    any_better = bool((ranked["final_equity_usd"] > fixed_row["final_equity_usd"]).any())
    ans(
        "9. Does adaptive sizing improve final BTC equity?",
        f"{'Yes' if any_better else 'No'} — best {best['arm']} at {best['final_equity_usd']:.2f} "
        f"vs FIXED_10 {fixed_row['final_equity_usd']:.2f} "
        f"({best['total_return_pct']-fixed_row['total_return_pct']:+.2f} pp).",
    )

    ans(
        "10. How much does maximum drawdown increase?",
        f"Worst adaptive DD {summary['max_drawdown_pct'].min():.3f}% vs FIXED_10 "
        f"{fixed_row['max_drawdown_pct']:.3f}% "
        f"(Δ {summary['max_drawdown_pct'].min()-fixed_row['max_drawdown_pct']:.3f} pp).",
    )

    ans(
        "11. How often does the 25% cap bind?",
        "; ".join(
            f"{r.arm}={r.pct_cap_25_binding:.1f}%" for _, r in summary.iterrows() if r.arm != "FIXED_10"
        ),
    )

    ans(
        "12. Is the allocation genuinely adaptive or effectively permanently above 10%?",
        "; ".join(
            f"{r.arm}: {r.get('pct_above_10', float('nan')):.1f}% of trades >10%, "
            f"mean alloc {r.avg_allocation*100:.1f}%"
            for _, r in summary.iterrows()
            if r.arm != "FIXED_10"
        ),
    )

    a_arms = [a for a, c in classifications.items() if c.get("class") == "A"]
    b_arms = [a for a, c in classifications.items() if c.get("class") == "B"]
    a_sorted = sorted(
        a_arms, key=lambda a: float(summary[summary["arm"] == a].iloc[0]["final_equity_usd"]), reverse=True
    )
    b_sorted = sorted(
        b_arms, key=lambda a: float(summary[summary["arm"] == a].iloc[0]["final_equity_usd"]), reverse=True
    )
    cands = (a_sorted + b_sorted)[:2]

    ans(
        "13. Are results stable across the year?",
        "See monthly table / monthly_comparison.png. "
        + ", ".join(
            f"{a}: months beat FIXED_10 = {classifications[a].get('months_beat_fixed')}"
            for a in ARM_ORDER
            if a != "FIXED_10"
        ),
    )

    ans(
        "14. Which ONE or TWO strategies deserve a final robustness / walk-forward validation?",
        (", ".join(cands) if cands else "None")
        + " | "
        + ", ".join(f"{a}={classifications[a]['class']}" for a in ARM_ORDER if a != "FIXED_10"),
    )

    lines.append("## Plots")
    lines.append("")
    for name, path in plot_paths.items():
        lines.append(f"- `{name}` → `{path}`")
    lines.append("")
    lines.append("## Final recommendation")
    lines.append("")
    lines.append(
        f"After correcting the normal-frequency assumption, best-by-equity is **{best['arm']}** "
        f"({best['total_return_pct']:.2f}% vs FIXED_10 {fixed_row['total_return_pct']:.2f}%, "
        f"DD {best['max_drawdown_pct']:.3f}% vs {fixed_row['max_drawdown_pct']:.3f}%). "
        f"Recommended robustness candidates: **{', '.join(cands) if cands else 'none'}**. "
        "Do not deploy to paper/live from this experiment alone."
    )
    lines.append("")

    path = out_dir / "RESEARCH_REPORT_selector_E_adaptive_sizing_part4b.md"
    path.write_text("\n".join(lines))
    return path


def emit_all(out_dir: Path, result: dict[str, Any], *, part4a_dir: Path | None) -> dict[str, Any]:
    out_dir = Path(out_dir)
    trade_tables = result["trade_tables"]
    daily_counts = result["daily_counts"]
    hist = result["causal_frequency_history"]
    ref_metrics = result["ref_metrics"]

    summary = enrich_summaries(trade_tables, result["summaries"])
    daily = build_daily(trade_tables, daily_counts)
    alloc_dist = build_allocation_distribution(trade_tables)
    scarcity = build_scarcity_ratio_analysis(trade_tables, hist, trade_tables["FIXED_10"])
    # Attach r_n7 onto FIXED_10 for diagnostic (scarcity_ratio is nan for fixed)
    fx = trade_tables["FIXED_10"].copy()
    scarcity_diag = scarcity_predicts_performance(fx)
    dd = build_drawdown_analysis(trade_tables)
    time_stab = build_time_stability(trade_tables)
    exposure = build_exposure_analysis(trade_tables)
    comparison = build_comparison(summary, ref_metrics)
    part4a = load_part4a_summary(part4a_dir)
    part4a_vs = build_part4a_vs_part4b(summary, part4a)
    classifications = classify_arms(summary, time_stab, alloc_dist)

    summary.to_csv(out_dir / "adaptive_sizing_summary.csv", index=False)
    daily.to_csv(out_dir / "adaptive_sizing_daily.csv", index=False)
    hist.to_csv(out_dir / "causal_frequency_history.csv", index=False)
    scarcity.to_csv(out_dir / "scarcity_ratio_analysis.csv", index=False)
    alloc_dist.to_csv(out_dir / "allocation_distribution.csv", index=False)
    comparison.to_csv(out_dir / "selector_E_comparison.csv", index=False)
    dd.to_csv(out_dir / "drawdown_analysis.csv", index=False)
    time_stab.to_csv(out_dir / "time_stability.csv", index=False)
    exposure.to_csv(out_dir / "exposure_analysis.csv", index=False)
    part4a_vs.to_csv(out_dir / "part4a_vs_part4b.csv", index=False)
    (out_dir / "scarcity_performance_diagnostic.json").write_text(
        json.dumps(scarcity_diag, indent=2, default=str)
    )

    for arm, df in trade_tables.items():
        df.to_csv(out_dir / f"trades_{arm}.csv", index=False)

    plot_paths = plot_all(out_dir, trade_tables, summary, time_stab, hist, scarcity)
    report = write_report(
        out_dir,
        result=result,
        summary=summary,
        comparison=comparison,
        alloc_dist=alloc_dist,
        scarcity=scarcity,
        dd=dd,
        time_stab=time_stab,
        exposure=exposure,
        part4a_vs=part4a_vs,
        scarcity_diag=scarcity_diag,
        classifications=classifications,
        plot_paths=plot_paths,
    )

    manifest = {
        "experiment": "selector_E_adaptive_sizing_part4b",
        "ref_dir": result.get("ref_dir"),
        "out_dir": str(out_dir),
        "part4a_dir": str(part4a_dir) if part4a_dir else None,
        "baseline_passed": result["baseline_check"]["passed"],
        "causality_ok": result["causality_check"]["ok"],
        "classifications": classifications,
        "summary": summary.to_dict(orient="records"),
        "scarcity_diagnostic": scarcity_diag,
        "report": str(report),
        "plots": plot_paths,
        "generated_at": result.get("generated_at"),
    }
    (out_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (out_dir / "baseline_check.json").write_text(json.dumps(result["baseline_check"], indent=2))
    (out_dir / "causality_check.json").write_text(
        json.dumps(result["causality_check"], indent=2, default=str)
    )
    return manifest
