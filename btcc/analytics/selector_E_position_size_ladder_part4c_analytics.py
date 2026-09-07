"""Analytics and research report for Part 4C position-size ladder."""

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

from btcc.sim.selector_E_adaptive_sizing import STARTING_CAPITAL_USD
from btcc.sim.selector_E_position_size_ladder_part4c import ARM_ORDER, FRACTION_BY_ARM

logger = logging.getLogger(__name__)

PART4A_CONTEXT = {
    "FIXED_10": 70.2,
    "7D_INVERSE": 156.2,
    "7D_MEDIAN_SQRT": 120.2,
    "7D_SQRT": 110.6,
}
PART4B_CONTEXT = {
    "FIXED_10": 70.2,
    "7D_MEDIAN_SQRT_CAUSAL": 81.0,
    "7D_INVERSE_CAUSAL": 80.3,
    "7D_SQRT_CAUSAL": 74.8,
    "7_30_INVERSE_CAUSAL": 76.5,
    "7_30_SQRT/DEADBAND": 73.2,
}


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
            sm["final_equity_usd"] = float(eq.iloc[-1])
            sm["total_return_pct"] = 100.0 * (float(eq.iloc[-1]) / STARTING_CAPITAL_USD - 1.0)
            sm["return_over_abs_dd"] = (
                float(sm["total_return_pct"] / abs(sm["max_drawdown_pct"]))
                if sm["max_drawdown_pct"] < 0
                else float("inf")
            )
        rows.append(sm)
    return pd.DataFrame(rows).set_index("arm").reindex(ARM_ORDER).reset_index()


def build_daily_equity(trade_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for arm, df in trade_tables.items():
        ex = df[df["executed"] == True].copy() if not df.empty else df  # noqa: E712
        if ex.empty:
            continue
        ex = ex.sort_values("exit_ts")
        equity = STARTING_CAPITAL_USD
        ex["exit_day"] = pd.to_datetime(ex["exit_ts"], utc=True).dt.floor("D")
        for day, g in ex.groupby("exit_day", sort=True):
            start = equity
            pnl = float(g["pnl_usd"].sum())
            equity = start + pnl
            rows.append(
                {
                    "arm": arm,
                    "day": day,
                    "starting_equity": start,
                    "ending_equity": equity,
                    "daily_pnl": pnl,
                    "daily_return_pct": 100.0 * pnl / start if start else 0.0,
                    "n_trades": int(len(g)),
                    "avg_actual_allocation": float(g["actual_allocation"].mean()),
                    "pct_exposure_limited": 100.0 * float(g["exposure_limited"].mean()),
                }
            )
    return pd.DataFrame(rows)


def build_monthly(trade_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for arm, df in trade_tables.items():
        ex = df[df["executed"] == True].copy() if not df.empty else df  # noqa: E712
        if ex.empty:
            continue
        ex = ex.sort_values("exit_ts")
        ex["month"] = pd.to_datetime(ex["exit_ts"], utc=True).dt.tz_localize(None).dt.to_period("M").astype(str)
        equity = STARTING_CAPITAL_USD
        for month, g in ex.groupby("month", sort=True):
            start = equity
            pnl = float(g["pnl_usd"].sum())
            equity = start + pnl
            path = start + g["pnl_usd"].cumsum()
            rows.append(
                {
                    "arm": arm,
                    "month": month,
                    "n_trades": int(len(g)),
                    "monthly_return_pct": 100.0 * pnl / start if start else 0.0,
                    "ending_equity": equity,
                    "cumulative_return_pct": 100.0 * (equity / STARTING_CAPITAL_USD - 1.0),
                    "max_drawdown_pct": _max_dd(path),
                }
            )
    return pd.DataFrame(rows)


def build_exposure_diagnostics(trade_tables: dict[str, pd.DataFrame], summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, sm in summary.iterrows():
        arm = sm["arm"]
        ex = trade_tables[arm]
        ex = ex[ex["executed"] == True]  # noqa: E712
        rows.append(
            {
                "arm": arm,
                "nominal_allocation": float(sm["nominal_allocation"]),
                "mean_actual_allocation": float(sm["mean_actual_allocation"]),
                "median_actual_allocation": float(sm["median_actual_allocation"]),
                "min_actual_allocation": float(sm["min_actual_allocation"]),
                "max_actual_allocation": float(sm["max_actual_allocation"]),
                "pct_fully_available": float(sm["pct_fully_available"]),
                "pct_exposure_limited": float(sm["pct_exposure_limited"]),
                "n_exposure_limited": int(sm["n_exposure_limited"]),
                "max_simultaneous_positions": int(sm["max_simultaneous_positions"]),
                "avg_simultaneous_positions": float(sm["avg_simultaneous_positions"]),
                "avg_exposure_at_entry": float(sm["avg_exposure_at_entry"]),
                "max_exposure": float(sm["max_exposure"]),
                "actual_over_nominal": (
                    float(sm["mean_actual_allocation"] / sm["nominal_allocation"])
                    if sm["nominal_allocation"]
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def classify_ladder(summary: pd.DataFrame, exposure: pd.DataFrame) -> dict[str, Any]:
    fixed10 = summary[summary["arm"] == "FIXED_10"].iloc[0]
    exp = exposure.set_index("arm")
    notes = []
    classes = {}
    for _, r in summary.iterrows():
        arm = r["arm"]
        if arm == "FIXED_10":
            classes[arm] = "CONTROL"
            continue
        d_ret = float(r["total_return_pct"] - fixed10["total_return_pct"])
        d_dd = float(r["max_drawdown_pct"] - fixed10["max_drawdown_pct"])
        dd_worsen = -d_dd
        pct_lim = float(exp.loc[arm, "pct_exposure_limited"]) if arm in exp.index else 0.0
        frac = float(r["nominal_allocation"])
        if pct_lim >= 5.0:
            cls = "D"
            note = "Exposure-constrained: nominal size often unavailable."
        elif d_ret >= 20 and dd_worsen <= 2.5 and 0.125 <= frac <= 0.20:
            cls = "A"
            note = "Strong intermediate size: material return lift with proportionate DD."
        elif d_ret >= 30 and frac >= 0.20:
            cls = "B"
            note = "Aggressive: large return with substantially higher DD."
        elif d_ret < 5:
            cls = "C"
            note = "No clear benefit vs 10%."
        else:
            cls = "B" if frac >= 0.175 else "A"
            note = "Improved return; review DD proportionality."
        classes[arm] = cls
        notes.append(f"{arm}={cls}: {note} (Δret={d_ret:.1f}pp, ΔDD={d_dd:.2f}pp, constrained={pct_lim:.1f}%)")
    return {"per_arm": classes, "notes": notes}


def plot_all(out_dir: Path, trade_tables: dict[str, pd.DataFrame], summary: pd.DataFrame) -> dict[str, str]:
    plots = out_dir / "plots"
    paths = {}
    fracs = [FRACTION_BY_ARM[a] * 100 for a in ARM_ORDER]

    # 1 equity curves
    fig, ax = plt.subplots(figsize=(12, 6))
    for arm in ARM_ORDER:
        ex = trade_tables[arm]
        ex = ex[ex["executed"] == True].sort_values("exit_ts")  # noqa: E712
        eq = STARTING_CAPITAL_USD + ex["pnl_usd"].cumsum()
        lw = 2.0 if arm == "FIXED_10" else 1.0
        ax.plot(pd.to_datetime(ex["exit_ts"], utc=True), eq, label=f"{arm} ({FRACTION_BY_ARM[arm]*100:g}%)", lw=lw)
    ax.axhline(STARTING_CAPITAL_USD, color="grey", ls="--", lw=0.8)
    ax.set_title("Part 4C — Fixed position-size ladder equity curves")
    ax.set_ylabel("Equity (USD)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = plots / "equity_curves.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["equity_curves.png"] = str(p)

    # 2 return vs DD
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(summary["max_drawdown_pct"].abs(), summary["total_return_pct"], s=80)
    for _, r in summary.iterrows():
        ax.annotate(f"{r['nominal_allocation']*100:g}%", (abs(r["max_drawdown_pct"]), r["total_return_pct"]), fontsize=8)
    ax.set_xlabel("|Max drawdown| %")
    ax.set_ylabel("Total return %")
    ax.set_title("Return vs maximum drawdown")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = plots / "return_vs_drawdown.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["return_vs_drawdown.png"] = str(p)

    # 3 final equity vs size
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fracs, summary["final_equity_usd"], marker="o")
    ax.axhline(1702.43, color="grey", ls="--", lw=0.8, label="Reference FIXED_10")
    ax.set_xlabel("Nominal position size %")
    ax.set_ylabel("Final equity (USD)")
    ax.set_title("Final equity vs nominal position size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = plots / "final_equity_vs_size.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["final_equity_vs_size.png"] = str(p)

    # 4 DD vs size
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fracs, summary["max_drawdown_pct"], marker="o", color="crimson")
    ax.set_xlabel("Nominal position size %")
    ax.set_ylabel("Max drawdown %")
    ax.set_title("Maximum drawdown vs nominal position size")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = plots / "drawdown_vs_size.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["drawdown_vs_size.png"] = str(p)

    # 5 actual vs requested allocation
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fracs, [f for f in fracs], ls="--", color="grey", label="Requested (nominal)")
    ax.plot(fracs, summary["mean_actual_allocation"] * 100, marker="o", label="Mean actual")
    ax.plot(fracs, summary["max_actual_allocation"] * 100, marker="s", label="Max actual")
    ax.set_xlabel("Nominal position size %")
    ax.set_ylabel("Allocation %")
    ax.set_title("Requested vs actual allocation (exposure constraint)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = plots / "actual_vs_requested_allocation.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["actual_vs_requested_allocation.png"] = str(p)

    return paths


def write_report(
    out_dir: Path,
    *,
    result: dict[str, Any],
    summary: pd.DataFrame,
    exposure: pd.DataFrame,
    monthly: pd.DataFrame,
    classification: dict[str, Any],
    plot_paths: dict[str, str],
) -> Path:
    baseline = result["baseline_check"]
    identity = result["identity_check"]
    fixed10 = summary[summary["arm"] == "FIXED_10"].iloc[0]
    f20 = summary[summary["arm"] == "FIXED_20"].iloc[0]
    f25 = summary[summary["arm"] == "FIXED_25"].iloc[0]
    f15 = summary[summary["arm"] == "FIXED_15"].iloc[0]
    f125 = summary[summary["arm"] == "FIXED_12_5"].iloc[0]

    # nearest fixed size to Part 4A mean alloc ~21.3%
    nearest_4a = summary.iloc[(summary["nominal_allocation"] - 0.213).abs().argsort()[:1]].iloc[0]

    lines = []
    lines.append("# Selector E Position-Size Ladder — Part 4C")
    lines.append("")
    lines.append("## 1. Objective")
    lines.append("")
    lines.append(
        "Determine whether Part 4A `7D_INVERSE` (+156.2%) is primarily larger fixed exposure "
        "rather than valuable adaptive timing, by running a pure fixed position-size ladder."
    )
    lines.append("")
    lines.append("## 2. Frozen Reference")
    lines.append("")
    lines.append(f"- Reference dir: `{result.get('ref_dir')}`")
    lines.append("- Selector E frozen; only position size varies.")
    lines.append(f"- Results: `{out_dir}`")
    lines.append("")
    lines.append("## 3. Experimental Arms")
    lines.append("")
    lines.append(", ".join(ARM_ORDER))
    lines.append("")
    lines.append("## 4. Methodology")
    lines.append("")
    lines.append(result.get("sizing_convention", ""))
    lines.append("")
    lines.append(
        "**Note on 'current equity' wording:** The frozen 1y reference and Parts 4A/4B use "
        "`compound_notional=false` with notional = fraction × $1,000 starting capital so FIXED_10 "
        "matches $1,702.43. Part 4C preserves that semantics for an apples-to-apples comparison "
        "with Part 4A `7D_INVERSE`. Exposure capacity still uses current equity (no leverage)."
    )
    lines.append("")
    lines.append("## 5. Reproducibility Checks")
    lines.append("")
    lines.append(f"- Check A FIXED_10 vs reference: **{baseline.get('passed')}** "
                 f"(sim match={baseline.get('fixed10_sim_matches_replay')})")
    f10, ref = baseline["fixed10"], baseline["reference"]
    lines.append("")
    lines.append("| Metric | FIXED_10 | Reference |")
    lines.append("| --- | ---: | ---: |")
    for k in ("n_trades", "final_equity_usd", "cumulative_return_pct", "win_rate_pct", "profit_factor"):
        lines.append(f"| {k} | {f10.get(k)} | {ref.get(k)} |")
    lines.append("")
    lines.append(f"- Checks B–E (same opportunities/strategies/timestamps/pnl_pct): **{identity.get('passed')}**")
    lines.append("")
    lines.append("## 6. Performance Results")
    lines.append("")
    perf_cols = [
        "arm",
        "nominal_allocation",
        "final_equity_usd",
        "total_return_pct",
        "n_executed",
        "n_exposure_limited",
        "win_rate_pct",
        "profit_factor",
        "avg_trade_pnl_usd",
        "median_trade_pnl_usd",
        "gross_profit_usd",
        "gross_loss_usd",
    ]
    lines.append(_md_table(summary[perf_cols]))
    lines.append("")
    lines.append("## 7. Risk Results")
    lines.append("")
    risk_cols = [
        "arm",
        "nominal_allocation",
        "max_drawdown_pct",
        "worst_trade_usd",
        "worst_day_usd",
        "worst_month",
        "worst_month_usd",
        "return_over_abs_dd",
    ]
    lines.append(_md_table(summary[risk_cols]))
    lines.append("")
    lines.append("## 8. Exposure and Constraint Analysis")
    lines.append("")
    lines.append(_md_table(exposure))
    lines.append("")
    lines.append("## 9. Return vs Drawdown")
    lines.append("")
    lines.append(_md_table(summary[["arm", "nominal_allocation", "total_return_pct", "max_drawdown_pct", "return_over_abs_dd"]]))
    lines.append("")
    lines.append("## 10. Comparison with Part 4A")
    lines.append("")
    lines.append("Contextual reported Part 4A returns (not recomputed):")
    for k, v in PART4A_CONTEXT.items():
        lines.append(f"- {k}: +{v}%")
    lines.append("")
    lines.append(
        f"Part 4A `7D_INVERSE` mean allocation was ~21.3% (often near 25%). "
        f"Nearest ladder arm by nominal size: **{nearest_4a['arm']}** "
        f"({nearest_4a['nominal_allocation']*100:g}% → +{nearest_4a['total_return_pct']:.1f}%). "
        f"FIXED_20 = +{f20['total_return_pct']:.1f}%, FIXED_25 = +{f25['total_return_pct']:.1f}%, "
        f"vs Part 4A 7D_INVERSE +156.2%."
    )
    lines.append("")
    lines.append("## 11. Comparison with Part 4B")
    lines.append("")
    lines.append("Contextual reported Part 4B returns:")
    for k, v in PART4B_CONTEXT.items():
        lines.append(f"- {k}: +{v}%")
    lines.append("")
    lines.append(
        f"Part 4B `7D_INVERSE_CAUSAL` (+80.3%) vs FIXED_12.5 (+{f125['total_return_pct']:.1f}%) "
        f"and FIXED_15 (+{f15['total_return_pct']:.1f}%)."
    )
    lines.append("")
    lines.append("## 12. Interpretation")
    lines.append("")
    lines.append("Per-arm classes: " + ", ".join(f"{k}={v}" for k, v in classification["per_arm"].items()))
    lines.append("")
    for n in classification["notes"]:
        lines.append(f"- {n}")
    lines.append("")
    lines.append("## 13. Limitations")
    lines.append("")
    lines.append(
        "- Single 1y MEXC window; no walk-forward here.\n"
        "- Same positive-edge E stream: larger size mechanically scales PnL if unconstrained.\n"
        "- Accounting matches frozen reference / Parts 4A–4B (fraction × $1000), not paper-bot "
        "true mark-to-market % of growing equity."
    )
    lines.append("")
    lines.append("## 14. Decision")
    lines.append("")

    # Q1
    gap_20 = 156.2 - float(f20["total_return_pct"])
    gap_25 = 156.2 - float(f25["total_return_pct"])
    lines.append("### Question 1 — Is Part 4A 7D_INVERSE (+156.2%) mostly larger position sizing?")
    lines.append("")
    lines.append(
        f"FIXED_20 = +{f20['total_return_pct']:.1f}% (Δ vs 4A inverse {gap_20:+.1f} pp), "
        f"FIXED_25 = +{f25['total_return_pct']:.1f}% (Δ {gap_25:+.1f} pp). "
        + (
            "Yes — Part 4A inverse is in the same ballpark as high fixed sizes, supporting the "
            "larger-exposure explanation."
            if abs(gap_25) < 25 or abs(gap_20) < 30
            else "Partially — fixed high sizes explain a large share but not necessarily all of the gap; "
            "residual may reflect path/timing of when size was large."
        )
    )
    lines.append("")

    # best balance by return/|dd|
    s = summary.copy()
    s["score"] = s["return_over_abs_dd"]
    best_bal = s.sort_values("score", ascending=False).iloc[0]
    lines.append("### Question 2 — Which fixed size best balances return and drawdown?")
    lines.append("")
    lines.append(
        f"**{best_bal['arm']}** ({best_bal['nominal_allocation']*100:g}%): "
        f"return {best_bal['total_return_pct']:.1f}%, DD {best_bal['max_drawdown_pct']:.2f}%, "
        f"return/|DD| = {best_bal['return_over_abs_dd']:.2f}."
    )
    lines.append("")

    lines.append("### Question 3 — Does Part 4B adaptive still beat comparable fixed size?")
    lines.append("")
    lines.append(
        f"Part 4B 7D_INVERSE_CAUSAL +80.3% vs FIXED_12.5 +{f125['total_return_pct']:.1f}% "
        f"and FIXED_15 +{f15['total_return_pct']:.1f}%. "
        + (
            "Adaptive looks similar to ~12.5–15% fixed; complexity may not be justified without "
            "robustness proof."
            if abs(80.3 - float(f15["total_return_pct"])) < 15
            or abs(80.3 - float(f125["total_return_pct"])) < 10
            else "Compare carefully against the nearest fixed arm."
        )
    )
    lines.append("")

    lines.append("### Question 4 — Increase paper allocation above 10%?")
    lines.append("")
    lines.append(
        "**RESEARCH ONLY — NO DEPLOYMENT.** Even if 15–25% looks better historically, do not change "
        "the paper/live 10% allocation without a separate robustness / walk-forward experiment."
    )
    lines.append("")
    lines.append("## Plots")
    lines.append("")
    for name, path in plot_paths.items():
        lines.append(f"- `{name}` → `{path}`")
    lines.append("")

    path = out_dir / "RESEARCH_REPORT_selector_E_position_size_ladder_part4c.md"
    path.write_text("\n".join(lines))
    return path


def emit_all(out_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    out_dir = Path(out_dir)
    trade_tables = result["trade_tables"]
    summary = enrich_summaries(trade_tables, result["summaries"])
    daily = build_daily_equity(trade_tables)
    monthly = build_monthly(trade_tables)
    exposure = build_exposure_diagnostics(trade_tables, summary)
    classification = classify_ladder(summary, exposure)

    summary.to_csv(out_dir / "summary.csv", index=False)
    daily.to_csv(out_dir / "daily_equity.csv", index=False)
    monthly.to_csv(out_dir / "monthly_results.csv", index=False)
    exposure.to_csv(out_dir / "exposure_diagnostics.csv", index=False)

    # combined trade-level
    all_trades = pd.concat(
        [trade_tables[a] for a in ARM_ORDER],
        ignore_index=True,
    )
    all_trades.to_csv(out_dir / "trade_level_results.csv", index=False)
    for arm, df in trade_tables.items():
        df.to_csv(out_dir / f"trades_{arm}.csv", index=False)

    plot_paths = plot_all(out_dir, trade_tables, summary)
    report = write_report(
        out_dir,
        result=result,
        summary=summary,
        exposure=exposure,
        monthly=monthly,
        classification=classification,
        plot_paths=plot_paths,
    )

    manifest = {
        "experiment": "selector_E_position_size_ladder_part4c",
        "ref_dir": result.get("ref_dir"),
        "out_dir": str(out_dir),
        "baseline_passed": result["baseline_check"]["passed"],
        "identity_passed": result["identity_check"]["passed"],
        "classification": classification,
        "summary": summary.to_dict(orient="records"),
        "report": str(report),
        "plots": plot_paths,
        "generated_at": result.get("generated_at"),
    }
    (out_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (out_dir / "baseline_check.json").write_text(json.dumps(result["baseline_check"], indent=2))
    (out_dir / "identity_check.json").write_text(json.dumps(result["identity_check"], indent=2))
    return manifest
