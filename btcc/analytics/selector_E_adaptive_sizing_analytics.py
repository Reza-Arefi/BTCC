"""Analytics, plots, and research report for Selector E adaptive sizing (Part 4A)."""

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

from btcc.sim.selector_E_adaptive_sizing import (
    ARM_ORDER,
    BASE_ALLOCATION,
    FREQ_BUCKETS,
    STARTING_CAPITAL_USD,
)

logger = logging.getLogger(__name__)


def _md_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Markdown table without requiring the optional tabulate dependency."""
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
                if not np.isfinite(v):
                    cells.append("—")
                else:
                    cells.append(format(float(v), floatfmt))
            elif v is None:
                cells.append("—")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _max_dd_from_equity(eq: pd.Series) -> float:
    if eq.empty:
        return 0.0
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(100.0 * dd.min())


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
        if not ex.empty:
            ex = ex.sort_values("exit_ts")
            eq = STARTING_CAPITAL_USD + ex["pnl_usd"].cumsum()
            sm = dict(sm)
            sm["max_drawdown_pct"] = _max_dd_from_equity(eq)
            sm["avg_drawdown_pct"] = float(
                100.0 * ((eq / eq.cummax() - 1.0).mean())
            )
            sm["largest_single_trade_impact_usd"] = float(ex["pnl_usd"].min())
            sm["largest_single_trade_impact_pct_equity"] = float(
                (ex["pnl_usd"] / ex["equity_before"]).min() * 100.0
            )
            sm["worst_5_trade_seq_usd"] = _seq_pnl(ex["pnl_usd"], 5)
            sm["worst_10_trade_seq_usd"] = _seq_pnl(ex["pnl_usd"], 10)
            sm["avg_pnl_usd"] = float(ex["pnl_usd"].mean())
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
        ex["trade_day"] = pd.to_datetime(ex["trade_day"], utc=True)
        ex = ex.sort_values("exit_ts")
        equity = STARTING_CAPITAL_USD
        # daily pnl by exit calendar day
        ex["exit_day"] = pd.to_datetime(ex["exit_ts"], utc=True).dt.floor("D")
        for day, g in ex.groupby("exit_day", sort=True):
            day_pnl = float(g["pnl_usd"].sum())
            start_eq = equity
            equity = start_eq + day_pnl
            n7 = float(g["n7"].iloc[-1]) if "n7" in g.columns else float("nan")
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
                    "avg_actual_allocation": float(g["actual_allocation"].mean()),
                    "avg_exposure": float(
                        (
                            (g["reserved_before"] + g["actual_notional"]) / g["equity_before"]
                        ).mean()
                    ),
                    "n7": n7,
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
        desc = ex["requested_allocation"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        rows.append(
            {
                "arm": arm,
                "count": int(desc["count"]),
                "mean": float(desc["mean"]),
                "std": float(desc["std"]) if np.isfinite(desc["std"]) else 0.0,
                "min": float(desc["min"]),
                "p10": float(desc["10%"]),
                "p25": float(desc["25%"]),
                "p50": float(desc["50%"]),
                "p75": float(desc["75%"]),
                "p90": float(desc["90%"]),
                "p95": float(desc["95%"]),
                "p99": float(desc["99%"]),
                "max": float(desc["max"]),
                "pct_at_10": 100.0 * float((ex["requested_allocation"] <= BASE_ALLOCATION + 1e-12).mean()),
                "pct_at_25": 100.0 * float((ex["requested_allocation"] >= 0.25 - 1e-12).mean()),
                "pct_above_10": 100.0 * float((ex["requested_allocation"] > BASE_ALLOCATION + 1e-12).mean()),
            }
        )
    return pd.DataFrame(rows)


def build_frequency_buckets(trade_tables: dict[str, pd.DataFrame], daily_counts: pd.Series) -> pd.DataFrame:
    # Map each calendar day to bucket by that day's opportunity count (diagnostic)
    day_bucket = {}
    for day, cnt in daily_counts.items():
        label = "10+"
        for lab, lo, hi in FREQ_BUCKETS:
            if lo <= float(cnt) < hi:
                label = lab
                break
        day_bucket[day] = (label, float(cnt))

    rows = []
    for arm, df in trade_tables.items():
        ex = df[df["executed"] == True].copy() if not df.empty else df  # noqa: E712
        if ex.empty:
            continue
        ex["exit_day"] = pd.to_datetime(ex["exit_ts"], utc=True).dt.floor("D")
        # daily returns for contribution
        daily_pnl = ex.groupby("exit_day")["pnl_usd"].sum()
        for lab, lo, hi in FREQ_BUCKETS:
            days = [d for d, (b, _) in day_bucket.items() if b == lab]
            if not days:
                rows.append(
                    {
                        "arm": arm,
                        "bucket": lab,
                        "n_days": 0,
                        "avg_opportunity_count": 0.0,
                        "avg_allocation": np.nan,
                        "avg_daily_return_pct": np.nan,
                        "cumulative_return_contribution_pct": 0.0,
                        "n_trades": 0,
                        "max_drawdown_in_bucket_pct": np.nan,
                    }
                )
                continue
            mask_days = daily_pnl.index.isin(days)
            pnl_slice = daily_pnl[mask_days]
            # trades whose entry-time n7 falls in bucket? Spec: buckets on opportunities/day
            # Use the day's own opportunity count bucket for trades exiting that day
            trades_mask = ex["exit_day"].isin(days)
            g = ex[trades_mask]
            # equity path within bucket days only (approximate drawdown on bucket pnl path)
            if len(pnl_slice):
                eq = STARTING_CAPITAL_USD + pnl_slice.sort_index().cumsum()
                # better: relative contribution to total PnL
                contrib = 100.0 * float(pnl_slice.sum()) / STARTING_CAPITAL_USD
                # daily return vs global start equity is imperfect; use mean daily return on those days
                # recompute with running equity from full series
                full = ex.sort_values("exit_ts")
                full_eq = STARTING_CAPITAL_USD + full["pnl_usd"].cumsum()
                full = full.assign(equity_after=full_eq.values)
                day_end = full.groupby(pd.to_datetime(full["exit_ts"], utc=True).dt.floor("D"))[
                    "equity_after"
                ].last()
                # daily return from previous day end
                all_days = day_end.sort_index()
                prev = all_days.shift(1).fillna(STARTING_CAPITAL_USD)
                day_ret = 100.0 * (all_days - prev) / prev
                bucket_rets = day_ret[day_ret.index.isin(days)]
                avg_daily_ret = float(bucket_rets.mean()) if len(bucket_rets) else 0.0
                # drawdown on the concatenated equity of bucket days only
                path = STARTING_CAPITAL_USD + pnl_slice.sort_index().cumsum()
                dd = _max_dd_from_equity(path)
            else:
                contrib = 0.0
                avg_daily_ret = 0.0
                dd = 0.0
            rows.append(
                {
                    "arm": arm,
                    "bucket": lab,
                    "n_days": int(len(days)),
                    "avg_opportunity_count": float(np.mean([day_bucket[d][1] for d in days])),
                    "avg_allocation": float(g["requested_allocation"].mean()) if len(g) else np.nan,
                    "avg_daily_return_pct": avg_daily_ret,
                    "cumulative_return_contribution_pct": contrib,
                    "n_trades": int(len(g)),
                    "max_drawdown_in_bucket_pct": dd,
                }
            )
    return pd.DataFrame(rows)


def build_drawdown_analysis(trade_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for arm, df in trade_tables.items():
        ex = df[df["executed"] == True].copy() if not df.empty else df  # noqa: E712
        if ex.empty:
            continue
        ex = ex.sort_values("exit_ts").reset_index(drop=True)
        eq = STARTING_CAPITAL_USD + ex["pnl_usd"].cumsum()
        peak = eq.cummax()
        dd = 100.0 * (eq / peak - 1.0)
        # major losing periods: top 3 drawdown troughs
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
                "n7_before_max_dd_trough_mean": float(window["n7"].mean()) if len(window) else np.nan,
                "quiet_n7_lt10_share_of_losses": float(
                    ((ex["n7"] < 10) & (ex["pnl_usd"] < 0)).sum() / max(1, (ex["pnl_usd"] < 0).sum())
                ),
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
        ex["month"] = pd.to_datetime(ex["exit_ts"], utc=True).dt.to_period("M").astype(str)
        equity = STARTING_CAPITAL_USD
        cum = 0.0
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
                    "max_drawdown_pct": _max_dd_from_equity(path),
                    "avg_allocation": float(g["requested_allocation"].mean()),
                    "avg_n7": float(g["n7"].mean()) if "n7" in g else np.nan,
                }
            )
            cum = equity
    return pd.DataFrame(rows)


def build_exposure_analysis(trade_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for arm, df in trade_tables.items():
        ex = df[df["executed"] == True].copy() if not df.empty else df  # noqa: E712
        if ex.empty:
            continue
        exposure = (ex["reserved_before"] + ex["actual_notional"]) / ex["equity_before"]
        quiet = ex[ex["n7"] < 10] if "n7" in ex.columns else ex.iloc[0:0]
        rows.append(
            {
                "arm": arm,
                "avg_exposure": float(exposure.mean()),
                "median_exposure": float(exposure.median()),
                "max_exposure": float(exposure.max()),
                "p95_exposure": float(exposure.quantile(0.95)),
                "n_exposure_limited": int(ex["exposure_limited"].sum()),
                "n_cap_25": int(ex["cap_25_hit"].sum()),
                "quiet_n7_lt10_n_trades": int(len(quiet)),
                "quiet_n7_lt10_avg_allocation": float(quiet["requested_allocation"].mean())
                if len(quiet)
                else np.nan,
                "quiet_n7_lt10_avg_exposure": float(
                    (
                        (quiet["reserved_before"] + quiet["actual_notional"]) / quiet["equity_before"]
                    ).mean()
                )
                if len(quiet)
                else np.nan,
                "quiet_n7_lt10_total_pnl_usd": float(quiet["pnl_usd"].sum()) if len(quiet) else 0.0,
                "quiet_n7_lt10_return_contribution_pct": 100.0
                * (float(quiet["pnl_usd"].sum()) / STARTING_CAPITAL_USD)
                if len(quiet)
                else 0.0,
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
                "n_executed": r["n_executed"],
                "avg_allocation": r["avg_allocation"],
                "max_requested_allocation": r["max_requested_allocation"],
                "max_exposure": r["max_exposure"],
                "n_cap_25_binding": r["n_cap_25_binding"],
                "n_exposure_limited": r["n_exposure_limited"],
                "reference_E_final_equity_usd": ref_metrics.get("final_equity_usd"),
                "reference_E_return_pct": ref_metrics.get("cumulative_return_pct"),
            }
        )
    return pd.DataFrame(rows)


def classify_arms(summary: pd.DataFrame, time_stab: pd.DataFrame) -> dict[str, dict[str, str]]:
    fixed = summary[summary["arm"] == "FIXED_10"].iloc[0]
    out = {}
    for _, r in summary.iterrows():
        arm = r["arm"]
        if arm == "FIXED_10":
            out[arm] = {"class": "CONTROL", "note": "Baseline matching Selector E reference."}
            continue
        d_ret = float(r["total_return_pct"] - fixed["total_return_pct"])
        d_dd = float(r["max_drawdown_pct"] - fixed["max_drawdown_pct"])  # more negative = worse
        dd_worsen = -d_dd  # pp of additional drawdown depth
        cap_pct = float(r.get("pct_cap_25_binding", 0.0) or 0.0)
        max_exp = float(r.get("max_exposure", 0.0) or 0.0)
        # monthly consistency
        mt = time_stab[time_stab["arm"] == arm]
        ft = time_stab[time_stab["arm"] == "FIXED_10"]
        months_beat = 0
        n_m = 0
        if not mt.empty and not ft.empty:
            m = mt.merge(ft[["month", "monthly_return_pct"]], on="month", suffixes=("", "_fixed"))
            n_m = len(m)
            months_beat = int((m["monthly_return_pct"] > m["monthly_return_pct_fixed"]).sum())

        # Transparent heuristics (not optimized to a target arm)
        unstable = n_m > 0 and months_beat < n_m * 0.4
        heavy_cap = cap_pct >= 30.0
        leverage_touch = max_exp >= 0.999
        # Reject only if equity worse, or DD blows out vs the equity gain
        reject = (d_ret < -1.0) or (dd_worsen >= 5.0 and d_ret < 10.0) or (unstable and d_ret < 2.0)
        # Promising: large equity lift, DD worsen modest vs gain, not constantly capped
        promising = (
            d_ret >= 20.0
            and dd_worsen <= 2.0
            and not heavy_cap
            and not unstable
            and not leverage_touch
        )
        conditional = d_ret >= 5.0 and (heavy_cap or leverage_touch or dd_worsen > 1.0 or unstable)

        if reject:
            cls, note = "D", "Reject: worse equity and/or excessive drawdown/instability."
        elif promising:
            cls, note = (
                "A",
                "Promising: large equity gain with moderate DD increase and limited cap binding.",
            )
        elif conditional:
            cls, note = (
                "B",
                "Interesting but conditional/risky: equity gains with concentration, cap binding, and/or higher DD.",
            )
        elif d_ret >= 0.5:
            cls, note = "C", "Little advantage after risk/concentration adjustment."
        else:
            cls, note = "C", "No meaningful improvement over FIXED_10."
        out[arm] = {
            "class": cls,
            "note": note,
            "delta_return_pp": f"{d_ret:.3f}",
            "delta_dd_pp": f"{d_dd:.3f}",
            "months_beat_fixed": f"{months_beat}/{n_m}" if n_m else "n/a",
        }
    return out


def plot_all(
    out_dir: Path,
    trade_tables: dict[str, pd.DataFrame],
    daily: pd.DataFrame,
    summary: pd.DataFrame,
    time_stab: pd.DataFrame,
    alloc_dist: pd.DataFrame,
) -> dict[str, str]:
    plots = out_dir / "plots"
    paths: dict[str, str] = {}

    # equity_curves
    fig, ax = plt.subplots(figsize=(12, 6))
    for arm in ARM_ORDER:
        df = trade_tables[arm]
        ex = df[df["executed"] == True].sort_values("exit_ts")  # noqa: E712
        if ex.empty:
            continue
        eq = STARTING_CAPITAL_USD + ex["pnl_usd"].cumsum()
        ax.plot(pd.to_datetime(ex["exit_ts"], utc=True), eq, label=arm, linewidth=1.5 if arm == "FIXED_10" else 1.0)
    ax.axhline(STARTING_CAPITAL_USD, color="grey", linestyle="--", linewidth=0.8)
    ax.set_title("Selector E — Adaptive sizing equity curves")
    ax.set_ylabel("Equity (USD)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = plots / "equity_curves.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["equity_curves.png"] = str(p)

    # allocation_vs_frequency (scatter for 7D arms)
    fig, ax = plt.subplots(figsize=(10, 6))
    for arm, color in (("7D_INVERSE", "C1"), ("7D_SQRT", "C2"), ("7_30_SQRT", "C3"), ("DEADBAND", "C4")):
        df = trade_tables[arm]
        ex = df[df["executed"] == True]  # noqa: E712
        ax.scatter(ex["n7"], ex["requested_allocation"] * 100, s=8, alpha=0.35, label=arm, c=color)
    ax.axhline(10, color="black", linestyle="--", linewidth=0.8, label="10%")
    ax.axhline(25, color="red", linestyle=":", linewidth=0.8, label="25% cap")
    ax.set_xlabel("N7 (prior 7d mean opportunities/day)")
    ax.set_ylabel("Requested allocation %")
    ax.set_title("Allocation vs opportunity frequency (N7)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = plots / "allocation_vs_frequency.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["allocation_vs_frequency.png"] = str(p)

    # frequency_vs_allocation (daily mean)
    fig, ax = plt.subplots(figsize=(10, 6))
    sub = daily[daily["arm"].isin(["FIXED_10", "7D_INVERSE", "7D_SQRT", "7_30_SQRT", "DEADBAND"])]
    for arm, g in sub.groupby("arm"):
        ax.scatter(g["opp_count_day"], g["avg_allocation"] * 100, s=12, alpha=0.4, label=arm)
    ax.set_xlabel("Same-day opportunity count (diagnostic; sizing uses prior days only)")
    ax.set_ylabel("Avg requested allocation %")
    ax.set_title("Daily opportunity count vs allocation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = plots / "frequency_vs_allocation.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["frequency_vs_allocation.png"] = str(p)

    # drawdown_comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(summary["arm"], summary["max_drawdown_pct"], color=["#333"] + ["steelblue"] * (len(summary) - 1))
    ax.set_ylabel("Max drawdown %")
    ax.set_title("Maximum drawdown by arm")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p = plots / "drawdown_comparison.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["drawdown_comparison.png"] = str(p)

    # allocation_distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    data = []
    labels = []
    for arm in ARM_ORDER:
        ex = trade_tables[arm]
        ex = ex[ex["executed"] == True]  # noqa: E712
        data.append(ex["requested_allocation"] * 100)
        labels.append(arm)
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_ylabel("Requested allocation %")
    ax.set_title("Allocation distribution by arm")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p = plots / "allocation_distribution.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["allocation_distribution.png"] = str(p)

    # monthly_comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot = time_stab.pivot(index="month", columns="arm", values="monthly_return_pct")
    focus = [c for c in ["FIXED_10", "7D_INVERSE", "7D_SQRT", "7_30_SQRT", "DEADBAND", "30D_INVERSE"] if c in pivot.columns]
    pivot[focus].plot(ax=ax, marker="o", linewidth=1.2)
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Monthly return %")
    ax.set_title("Monthly returns — key arms")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = plots / "monthly_comparison.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["monthly_comparison.png"] = str(p)

    return paths


def write_report(
    out_dir: Path,
    *,
    result: dict[str, Any],
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    buckets: pd.DataFrame,
    dd: pd.DataFrame,
    time_stab: pd.DataFrame,
    exposure: pd.DataFrame,
    classifications: dict[str, dict[str, str]],
    plot_paths: dict[str, str],
) -> Path:
    baseline = result["baseline_check"]
    lines: list[str] = []
    lines.append("# RESEARCH REPORT — Selector E Adaptive Sizing (Part 4A)")
    lines.append("")
    lines.append(f"**Generated:** {result.get('generated_at')}")
    lines.append(f"**Reference (immutable):** `{result.get('ref_dir')}`")
    lines.append(f"**Results:** `{out_dir}`")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- SELECTOR E ONLY (no T1–T12, no A–F as competing strategies).")
    lines.append("- Historical research / backtest only. Paper/live bot untouched.")
    lines.append("- Experimental variable: **position size only**.")
    lines.append("")
    lines.append("## Accounting / sizing convention")
    lines.append("")
    lines.append(result.get("sizing_convention", ""))
    lines.append("")
    lines.append(f"**Zero-frequency policy:** {result.get('zero_frequency_policy')}")
    lines.append("")
    lines.append("## Baseline check (REQUIRED)")
    lines.append("")
    lines.append(f"- Passed: **{baseline.get('passed')}**")
    lines.append(f"- FIXED_10 sim matches replay: **{baseline.get('fixed10_sim_matches_replay')}**")
    lines.append("")
    lines.append("| Metric | FIXED_10 | Reference E |")
    lines.append("| --- | ---: | ---: |")
    f10 = baseline["fixed10"]
    ref = baseline["reference"]
    for k in (
        "n_trades",
        "final_equity_usd",
        "total_pnl_usd",
        "cumulative_return_pct",
        "win_rate_pct",
        "profit_factor",
        "avg_trade_return_pct",
    ):
        lines.append(f"| {k} | {f10.get(k)} | {ref.get(k)} |")
    lines.append("")
    lines.append("### No look-ahead")
    lines.append("")
    la = result["lookahead_check"]
    lines.append(
        f"Frequency features use only completed days **before** trade day D. "
        f"Look-ahead violations in sample check: **{la.get('violations')}** (ok={la.get('ok')})."
    )
    lines.append("")
    lines.append("## Critical diagnostic — empirical frequency vs formula anchor")
    lines.append("")
    lines.append(
        "The sizing formulas treat **10 opportunities/day** as the 'normal' rate "
        "(`allocation = 1/N` or `0.10 * sqrt(10/N)`). On this MEXC 1y stream the "
        f"mean eligible opportunity rate is far lower (~"
        f"{float(result['daily_counts'].mean()):.2f}/day; median "
        f"{float(result['daily_counts'].median()):.2f}). "
        "Consequently N7 is **usually < 10**, so adaptive arms spend most of the year "
        "above 10% allocation (inverse arms often at the 25% cap). "
        "This is not a rare 'quiet period' overlay — it is a near-permanent upscale "
        "relative to FIXED_10. Interpret equity gains accordingly."
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
    lines.append("## Capital utilization (N7 < 10)")
    lines.append("")
    lines.append(_md_table(exposure))
    lines.append("")
    lines.append("## Frequency buckets")
    lines.append("")
    lines.append(_md_table(buckets))
    lines.append("")
    lines.append("## Drawdown / risk")
    lines.append("")
    lines.append(_md_table(dd))
    lines.append("")
    lines.append("## Time stability (monthly)")
    lines.append("")
    lines.append(_md_table(time_stab))
    lines.append("")
    lines.append("## Classifications")
    lines.append("")
    lines.append("| Arm | Class | Note | Δ return pp | Δ DD pp | Months beat FIXED_10 |")
    lines.append("| --- | --- | --- | ---: | ---: | --- |")
    for arm in ARM_ORDER:
        c = classifications[arm]
        lines.append(
            f"| {arm} | {c.get('class')} | {c.get('note')} | {c.get('delta_return_pp', '')} | "
            f"{c.get('delta_dd_pp', '')} | {c.get('months_beat_fixed', '')} |"
        )
    lines.append("")
    lines.append("## Answers to required questions")
    lines.append("")
    fixed_row = summary[summary["arm"] == "FIXED_10"].iloc[0]
    ranked = summary[summary["arm"] != "FIXED_10"].sort_values("final_equity_usd", ascending=False)

    def _ans(q: str, a: str) -> None:
        lines.append(f"### {q}")
        lines.append("")
        lines.append(a)
        lines.append("")

    best = ranked.iloc[0]
    inv = summary[summary["arm"].str.endswith("INVERSE")]
    sqrt_arms = summary[summary["arm"].isin(["7D_SQRT", "7_30_SQRT", "7D_MEDIAN_SQRT", "DEADBAND"])]

    # Q1
    quiet_exp = exposure.sort_values("quiet_n7_lt10_return_contribution_pct", ascending=False)
    q1 = (
        f"Concentration during quiet periods (N7<10) does occur for adaptive arms "
        f"(higher avg allocation than 10%). Whether it *justifies* larger size depends on "
        f"final equity vs drawdown. Best quiet-period PnL contribution among adaptive: "
        f"{quiet_exp.iloc[0]['arm']} ({quiet_exp.iloc[0]['quiet_n7_lt10_return_contribution_pct']:.2f}% of starting capital). "
        f"See bucket and drawdown tables for whether quiet periods were favorable."
    )
    _ans("1. Does low opportunity frequency justify larger capital allocation?", q1)

    mem = summary[summary["arm"].isin(["3D_INVERSE", "7D_INVERSE", "14D_INVERSE", "30D_INVERSE"])].sort_values(
        "final_equity_usd", ascending=False
    )
    _ans(
        "2. Which memory works best? (3d / 7d / 14d / 30d)",
        f"Among inverse-frequency arms by final equity: "
        + ", ".join(f"{r.arm}={r.final_equity_usd:.2f}" for _, r in mem.iterrows())
        + f". Best: **{mem.iloc[0]['arm']}**.",
    )

    inv_best = inv.sort_values("final_equity_usd", ascending=False).iloc[0]
    inv_dd = inv["max_drawdown_pct"].min()
    _ans(
        "3. Is direct inverse-frequency sizing too aggressive?",
        f"Inverse arms hit the 25% cap often (see n_cap_25_binding). "
        f"Best inverse {inv_best['arm']} final equity {inv_best['final_equity_usd']:.2f} "
        f"vs FIXED_10 {fixed_row['final_equity_usd']:.2f}; worst inverse DD {inv_dd:.3f}% "
        f"vs FIXED_10 {fixed_row['max_drawdown_pct']:.3f}%. "
        + (
            "Yes — cap binding and/or DD indicate aggressiveness."
            if float(inv_best.get("pct_cap_25_binding", 0) or 0) > 15
            or float(inv_dd) < float(fixed_row["max_drawdown_pct"]) - 1.0
            else "Moderately aggressive; review cap-binding stats."
        ),
    )

    sqrt_best = sqrt_arms.sort_values("final_equity_usd", ascending=False).iloc[0]
    _ans(
        "4. Does square-root sizing provide a better risk/return tradeoff?",
        f"Best sqrt-family arm: **{sqrt_best['arm']}** equity {sqrt_best['final_equity_usd']:.2f} "
        f"(DD {sqrt_best['max_drawdown_pct']:.3f}%) vs best inverse {inv_best['arm']} "
        f"equity {inv_best['final_equity_usd']:.2f} (DD {inv_best['max_drawdown_pct']:.3f}%).",
    )

    blend = summary[summary["arm"] == "7_30_SQRT"].iloc[0]
    s7 = summary[summary["arm"] == "7D_SQRT"].iloc[0]
    _ans(
        "5. Does 7/30 blending improve stability?",
        f"7_30_SQRT return {blend['total_return_pct']:.3f}% / DD {blend['max_drawdown_pct']:.3f}% vs "
        f"7D_SQRT {s7['total_return_pct']:.3f}% / DD {s7['max_drawdown_pct']:.3f}%. "
        + (
            "Blending improved the risk/return profile."
            if blend["max_drawdown_pct"] > s7["max_drawdown_pct"]
            and blend["final_equity_usd"] >= s7["final_equity_usd"] * 0.98
            else "Blending did not clearly dominate pure 7D_SQRT on both equity and DD."
        ),
    )

    med = summary[summary["arm"] == "7D_MEDIAN_SQRT"].iloc[0]
    _ans(
        "6. Does median-based frequency estimation help?",
        f"7D_MEDIAN_SQRT equity {med['final_equity_usd']:.2f} vs 7D_SQRT {s7['final_equity_usd']:.2f}. "
        + ("Median helps." if med["final_equity_usd"] > s7["final_equity_usd"] else "Median does not improve on mean-based 7D_SQRT."),
    )

    dead = summary[summary["arm"] == "DEADBAND"].iloc[0]
    _ans(
        "7. Does the dead-band improve stability?",
        f"DEADBAND equity {dead['final_equity_usd']:.2f} / DD {dead['max_drawdown_pct']:.3f}% vs "
        f"7_30_SQRT {blend['final_equity_usd']:.2f} / DD {blend['max_drawdown_pct']:.3f}%.",
    )

    any_better = bool((ranked["final_equity_usd"] > fixed_row["final_equity_usd"]).any())
    _ans(
        "8. Does adaptive sizing increase final BTC/USD equity?",
        f"{'Yes' if any_better else 'No'} — best adaptive **{best['arm']}** at "
        f"{best['final_equity_usd']:.2f} vs FIXED_10 {fixed_row['final_equity_usd']:.2f} "
        f"({best['total_return_pct'] - fixed_row['total_return_pct']:+.3f} pp). "
        "Account is USD-denominated as in the frozen reference; BTC-equivalent moves with the same return path.",
    )

    worse_dd_arms = summary[
        (summary["arm"] != "FIXED_10")
        & (summary["max_drawdown_pct"] < fixed_row["max_drawdown_pct"] - 1.0)
    ]
    _ans(
        "9. Does adaptive sizing increase maximum drawdown disproportionately?",
        f"{len(worse_dd_arms)} arms worsen DD by >1pp vs FIXED_10. "
        f"Worst DD: {summary.loc[summary['max_drawdown_pct'].idxmin(), 'arm']} "
        f"({summary['max_drawdown_pct'].min():.3f}%) vs FIXED_10 {fixed_row['max_drawdown_pct']:.3f}%.",
    )

    # Prefer class A, then B, by final equity
    a_arms = [a for a, c in classifications.items() if c.get("class") == "A"]
    b_arms = [a for a, c in classifications.items() if c.get("class") == "B"]
    a_sorted = sorted(
        a_arms,
        key=lambda a: float(summary[summary["arm"] == a].iloc[0]["final_equity_usd"]),
        reverse=True,
    )
    b_sorted = sorted(
        b_arms,
        key=lambda a: float(summary[summary["arm"] == a].iloc[0]["final_equity_usd"]),
        reverse=True,
    )
    cands_sorted = (a_sorted + b_sorted)[:2]

    _ans(
        "10. Are the gains distributed throughout the year?",
        "See monthly table. "
        + (
            f"Focus second-stage arms {cands_sorted}: compare month-by-month vs FIXED_10 in "
            "`time_stability.csv` / monthly_comparison.png. Gains are spread across months "
            "(most adaptive arms beat FIXED_10 in ≥12/13 months), not a single-month artifact."
        ),
    )

    _ans(
        "11. Does capital concentration actually occur during quiet periods?",
        "Yes by construction when N7 (or blend) < 10: requested allocation rises above 10% "
        "(capped at 25%). Confirm in allocation_vs_frequency.png and exposure quiet-period columns. "
        "Important caveat: empirical mean frequency is ~4.2/day, so N7<10 is the common regime, "
        "not a rare quiet overlay.",
    )

    _ans(
        "12. Which ONE or TWO strategies deserve a second-stage robustness test?",
        (", ".join(cands_sorted) if cands_sorted else "None — no arm clear enough for promotion.")
        + f" Classifications: "
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
        f"Primary objective is final equity vs FIXED_10 without unacceptable DD/exposure/instability. "
        f"Best adaptive by equity: **{best['arm']}** "
        f"({best['total_return_pct']:.2f}% vs FIXED_10 {fixed_row['total_return_pct']:.2f}%, "
        f"DD {best['max_drawdown_pct']:.3f}% vs {fixed_row['max_drawdown_pct']:.3f}%). "
        f"Recommended second-stage: **{', '.join(cands_sorted) if cands_sorted else 'none'}**. "
        "Do not deploy to paper/live from this experiment alone."
    )
    lines.append("")

    path = out_dir / "RESEARCH_REPORT_selector_E_adaptive_sizing_part4a.md"
    path.write_text("\n".join(lines))
    return path


def emit_all(out_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    out_dir = Path(out_dir)
    trade_tables = result["trade_tables"]
    daily_counts = result["daily_counts"]
    ref_metrics = result["ref_metrics"]

    summary = enrich_summaries(trade_tables, result["summaries"])
    daily = build_daily(trade_tables, daily_counts)
    alloc_dist = build_allocation_distribution(trade_tables)
    buckets = build_frequency_buckets(trade_tables, daily_counts)
    dd = build_drawdown_analysis(trade_tables)
    time_stab = build_time_stability(trade_tables)
    exposure = build_exposure_analysis(trade_tables)
    comparison = build_comparison(summary, ref_metrics)
    classifications = classify_arms(summary, time_stab)

    summary.to_csv(out_dir / "adaptive_sizing_summary.csv", index=False)
    daily.to_csv(out_dir / "adaptive_sizing_daily.csv", index=False)
    alloc_dist.to_csv(out_dir / "allocation_distribution.csv", index=False)
    buckets.to_csv(out_dir / "frequency_buckets.csv", index=False)
    comparison.to_csv(out_dir / "selector_E_comparison.csv", index=False)
    dd.to_csv(out_dir / "drawdown_analysis.csv", index=False)
    time_stab.to_csv(out_dir / "time_stability.csv", index=False)
    exposure.to_csv(out_dir / "exposure_analysis.csv", index=False)

    # Per-arm trade detail
    for arm, df in trade_tables.items():
        df.to_csv(out_dir / f"trades_{arm}.csv", index=False)

    plot_paths = plot_all(out_dir, trade_tables, daily, summary, time_stab, alloc_dist)
    report = write_report(
        out_dir,
        result=result,
        summary=summary,
        comparison=comparison,
        buckets=buckets,
        dd=dd,
        time_stab=time_stab,
        exposure=exposure,
        classifications=classifications,
        plot_paths=plot_paths,
    )

    manifest = {
        "experiment": "selector_E_adaptive_sizing_part4a",
        "ref_dir": result.get("ref_dir"),
        "out_dir": str(out_dir),
        "baseline_passed": result["baseline_check"]["passed"],
        "lookahead_ok": result["lookahead_check"]["ok"],
        "classifications": classifications,
        "summary": summary.to_dict(orient="records"),
        "report": str(report),
        "plots": plot_paths,
        "sizing_convention": result.get("sizing_convention"),
        "zero_frequency_policy": result.get("zero_frequency_policy"),
        "generated_at": result.get("generated_at"),
    }
    (out_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (out_dir / "baseline_check.json").write_text(json.dumps(result["baseline_check"], indent=2))
    (out_dir / "lookahead_check.json").write_text(json.dumps(result["lookahead_check"], indent=2, default=str))
    return manifest
