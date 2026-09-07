"""Analytics, plots, and research report for Part 5 strategy comparison @ 25%."""

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
from btcc.sim.strategy_comparison_25pct_part5 import ALL_ARMS, FIXED_ARMS, SELECTOR_ARMS

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


def plot_all(
    out_dir: Path,
    summary: pd.DataFrame,
    trade_tables: dict[str, pd.DataFrame],
    daily_tables: dict[str, pd.DataFrame],
    monthly_long: pd.DataFrame,
) -> dict[str, str]:
    plots = out_dir / "plots"
    paths = {}

    def _eq_series(arm: str) -> tuple[pd.Series, pd.Series]:
        ex = trade_tables[arm]
        ex = ex[ex["executed"] == True].sort_values("exit_ts")  # noqa: E712
        ts = pd.to_datetime(ex["exit_ts"], utc=True)
        eq = STARTING_CAPITAL_USD + ex["pnl_usd"].cumsum()
        return ts, eq

    # all equity
    fig, ax = plt.subplots(figsize=(13, 7))
    for arm in ALL_ARMS:
        ts, eq = _eq_series(arm)
        ax.plot(ts, eq, lw=1.2 if arm == "E" else 0.9, label=arm, alpha=0.9 if arm in SELECTOR_ARMS else 0.7)
    ax.axhline(STARTING_CAPITAL_USD, color="grey", ls="--", lw=0.8)
    ax.set_title("All strategies — equity ($) @ 25%")
    ax.set_ylabel("Equity USD")
    ax.legend(fontsize=7, ncol=4)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = plots / "all_equity.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["all_equity.png"] = str(p)

    # all returns
    fig, ax = plt.subplots(figsize=(13, 7))
    for arm in ALL_ARMS:
        ts, eq = _eq_series(arm)
        ax.plot(ts, 100.0 * (eq / STARTING_CAPITAL_USD - 1.0), lw=1.0, label=arm)
    ax.axhline(0, color="grey", ls="--", lw=0.8)
    ax.set_title("All strategies — cumulative return (%) @ 25%")
    ax.set_ylabel("Return %")
    ax.legend(fontsize=7, ncol=4)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = plots / "all_returns.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["all_returns.png"] = str(p)

    # all drawdown
    fig, ax = plt.subplots(figsize=(13, 6))
    for arm in ALL_ARMS:
        ts, eq = _eq_series(arm)
        dd = 100.0 * (eq / eq.cummax() - 1.0)
        ax.plot(ts, dd, lw=0.9, label=arm, alpha=0.85)
    ax.set_title("All strategies — drawdown (%) @ 25%")
    ax.set_ylabel("Drawdown %")
    ax.legend(fontsize=7, ncol=4)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = plots / "all_drawdown.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["all_drawdown.png"] = str(p)

    # T1-T10 equity
    fig, ax = plt.subplots(figsize=(12, 6))
    for arm in FIXED_ARMS:
        ts, eq = _eq_series(arm)
        ax.plot(ts, eq, lw=1.1, label=arm)
    ax.set_title("T1–T10 equity @ 25%")
    ax.legend(fontsize=8, ncol=5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = plots / "T1_T10_equity.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["T1_T10_equity.png"] = str(p)

    # A-F equity
    fig, ax = plt.subplots(figsize=(12, 6))
    for arm in SELECTOR_ARMS:
        ts, eq = _eq_series(arm)
        ax.plot(ts, eq, lw=1.4, label=arm)
    ax.set_title("Selectors A–F equity @ 25%")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = plots / "A_F_equity.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["A_F_equity.png"] = str(p)

    # rankings
    srt = summary.sort_values("return_pct", ascending=False)
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#1f77b4" if t == "selector" else "#7f7f7f" for t in srt["type"]]
    ax.bar(srt["strategy"], srt["return_pct"], color=colors)
    ax.set_title("Return ranking @ 25%")
    ax.set_ylabel("Return %")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p = plots / "return_ranking.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["return_ranking.png"] = str(p)

    srt_dd = summary.sort_values("max_drawdown_pct")  # most negative first
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(srt_dd["strategy"], srt_dd["max_drawdown_pct"], color="#d62728")
    ax.set_title("Max drawdown ranking @ 25%")
    ax.set_ylabel("Max DD %")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p = plots / "drawdown_ranking.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["drawdown_ranking.png"] = str(p)

    srt_ra = summary.sort_values("return_to_drawdown", ascending=False)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(srt_ra["strategy"], srt_ra["return_to_drawdown"], color="#2ca02c")
    ax.set_title("Risk-adjusted ranking (return / |max DD|) @ 25%")
    ax.set_ylabel("Return / |DD|")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p = plots / "risk_adjusted_ranking.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["risk_adjusted_ranking.png"] = str(p)

    # trade count
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(summary["strategy"], summary["trades"], color="steelblue")
    ax.set_title("Trade count")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p = plots / "trade_count.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["trade_count.png"] = str(p)

    # winrate vs PF
    fig, ax = plt.subplots(figsize=(8, 6))
    for typ, color in (("fixed_trail", "#7f7f7f"), ("selector", "#1f77b4")):
        g = summary[summary["type"] == typ]
        ax.scatter(g["win_rate"], g["profit_factor"], s=70, c=color, label=typ)
        for _, r in g.iterrows():
            ax.annotate(r["strategy"], (r["win_rate"], r["profit_factor"]), fontsize=7)
    ax.set_xlabel("Win rate %")
    ax.set_ylabel("Profit factor")
    ax.set_title("Win rate vs profit factor @ 25%")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = plots / "winrate_vs_profitfactor.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["winrate_vs_profitfactor.png"] = str(p)

    # monthly heatmap-ish: line plot of monthly returns for top strategies + E
    fig, ax = plt.subplots(figsize=(12, 6))
    focus = list(srt.head(5)["strategy"]) + (["E"] if "E" not in list(srt.head(5)["strategy"]) else [])
    for arm in focus:
        g = monthly_long[monthly_long["strategy"] == arm]
        ax.plot(g["month"], g["return_pct"], marker="o", lw=1.2, label=arm)
    ax.axhline(0, color="grey", ls="--", lw=0.8)
    ax.set_title("Monthly returns — top strategies + E")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = plots / "monthly_returns.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["monthly_returns.png"] = str(p)

    # return vs DD scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    for typ, color in (("fixed_trail", "#7f7f7f"), ("selector", "#1f77b4")):
        g = summary[summary["type"] == typ]
        ax.scatter(g["max_drawdown_pct"].abs(), g["return_pct"], s=70, c=color, label=typ)
        for _, r in g.iterrows():
            ax.annotate(r["strategy"], (abs(r["max_drawdown_pct"]), r["return_pct"]), fontsize=7)
    ax.set_xlabel("|Max DD| %")
    ax.set_ylabel("Return %")
    ax.set_title("Return vs drawdown @ 25%")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = plots / "return_vs_drawdown.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["return_vs_drawdown.png"] = str(p)

    return paths


def write_report(
    out_dir: Path,
    *,
    result: dict[str, Any],
    summary: pd.DataFrame,
    monthly_long: pd.DataFrame,
    plot_paths: dict[str, str],
) -> Path:
    by_eq = summary.sort_values("final_equity", ascending=False)
    by_ra = summary.sort_values("return_to_drawdown", ascending=False)
    # monthly stability: std of monthly returns (lower better among positive mean)
    stab_rows = []
    for arm in ALL_ARMS:
        g = monthly_long[monthly_long["strategy"] == arm]
        stab_rows.append(
            {
                "strategy": arm,
                "mean_monthly_return": float(g["return_pct"].mean()),
                "std_monthly_return": float(g["return_pct"].std(ddof=1)) if len(g) > 1 else 0.0,
                "pct_positive_months": float(100.0 * (g["return_pct"] > 0).mean()),
                "worst_month": float(g["return_pct"].min()),
            }
        )
    stab = pd.DataFrame(stab_rows).sort_values(["pct_positive_months", "mean_monthly_return"], ascending=[False, False])

    by_pf = summary.sort_values("profit_factor", ascending=False)

    lines = []
    lines.append("# Full Strategy Comparison — 25% Allocation — 365 Days")
    lines.append("")
    lines.append(f"**Generated:** {result.get('generated_at')}")
    lines.append(f"**Reference:** `{result.get('ref_dir')}`")
    lines.append(f"**Results:** `{out_dir}`")
    lines.append("")
    lines.append("## 1. Objective")
    lines.append("")
    lines.append(
        "Compare T1–T10 and selectors A–F under **identical fixed 25% allocation**, "
        "so differences reflect strategy logic rather than sizing rules. T11/T12 excluded."
    )
    lines.append("")
    lines.append("## 2. Experimental Configuration")
    lines.append("")
    lines.append("- Arms: T1–T10 + A–F (**16** strategies)")
    lines.append("- Allocation: 25% fixed")
    lines.append("- Start: $1,000; max exposure 100%; max open 10; no leverage")
    lines.append(f"- {result.get('sizing_convention')}")
    lines.append("")
    lines.append("## 3. Frozen Strategy Definitions")
    lines.append("")
    lines.append("T1–T10 geometries and A–F selector definitions unchanged from the immutable 365d reference.")
    lines.append("")
    lines.append("## 4. Portfolio and Compounding Rules")
    lines.append("")
    lines.append("Same Part 4C/4E research-replay accounting for fair comparison and E reproducibility.")
    lines.append("")
    lines.append("## 5. Reproducibility Checks")
    lines.append("")
    lines.append(f"- Fairness checks passed: **{result['fairness'].get('passed')}**")
    lines.append(f"- E reproduces Part 4C FIXED_25: **{result['e_check'].get('passed')}**")
    lines.append("")
    lines.append(_md_table(pd.DataFrame([
        {"check": k, "value": v} for k, v in result["fairness"].items() if k != "passed"
    ])))
    lines.append("")
    lines.append("E vs Part 4C targets:")
    lines.append("")
    lines.append(_md_table(pd.DataFrame([{
        "metric": "final_equity",
        "E": result["e_check"]["E"]["final_equity"],
        "part4c": result["e_check"]["part4c_targets"]["final_equity_usd"],
    }, {
        "metric": "return_pct",
        "E": result["e_check"]["E"]["return_pct"],
        "part4c": result["e_check"]["part4c_targets"]["total_return_pct"],
    }, {
        "metric": "max_drawdown_pct",
        "E": result["e_check"]["E"]["max_drawdown_pct"],
        "part4c": result["e_check"]["part4c_targets"]["max_drawdown_pct"],
    }])))
    lines.append("")
    lines.append("## 6. Overall Results")
    lines.append("")
    show_cols = [
        "strategy", "type", "final_equity", "return_pct", "max_drawdown_pct", "return_to_drawdown",
        "trades", "win_rate", "profit_factor", "expectancy", "avg_trade_pct",
        "exposure_constrained_pct", "max_concurrent_positions",
    ]
    lines.append(_md_table(by_eq[show_cols]))
    lines.append("")
    lines.append("## 7. T1–T10 Results")
    lines.append("")
    lines.append(_md_table(by_eq[by_eq["type"] == "fixed_trail"][show_cols]))
    lines.append("")
    lines.append("## 8. A–F Results")
    lines.append("")
    lines.append(_md_table(by_eq[by_eq["type"] == "selector"][show_cols]))
    lines.append("")
    lines.append("## 9. Return Comparison")
    lines.append("")
    lines.append(f"Highest absolute return: **{by_eq.iloc[0]['strategy']}** "
                 f"(+{by_eq.iloc[0]['return_pct']:.2f}%, ${by_eq.iloc[0]['final_equity']:.2f}).")
    lines.append("")
    lines.append("## 10. Drawdown Comparison")
    lines.append("")
    best_dd = summary.sort_values("max_drawdown_pct", ascending=False).iloc[0]  # least negative
    lines.append(f"Mildest max DD: **{best_dd['strategy']}** ({best_dd['max_drawdown_pct']:.3f}%).")
    lines.append("")
    lines.append("## 11. Risk-Adjusted Comparison")
    lines.append("")
    lines.append(_md_table(by_ra[["strategy", "return_pct", "max_drawdown_pct", "return_to_drawdown"]].head(10)))
    lines.append("")
    lines.append(f"Best return/|DD|: **{by_ra.iloc[0]['strategy']}** ({by_ra.iloc[0]['return_to_drawdown']:.2f}).")
    lines.append("")
    lines.append("## 12. Monthly Stability")
    lines.append("")
    lines.append(_md_table(stab))
    lines.append("")
    lines.append(f"Most consistent positive-month profile: **{stab.iloc[0]['strategy']}** "
                 f"({stab.iloc[0]['pct_positive_months']:.1f}% positive months).")
    lines.append("")
    lines.append("## 13. Trade-Level Statistics")
    lines.append("")
    lines.append(_md_table(by_pf[["strategy", "profit_factor", "win_rate", "expectancy", "avg_trade_pct",
                                  "avg_winner_pct", "avg_loser_pct"]].head(10)))
    lines.append("")
    lines.append(f"Best PF: **{by_pf.iloc[0]['strategy']}** (PF={by_pf.iloc[0]['profit_factor']:.3f}).")
    lines.append("")
    lines.append("## 14. Exposure and Capacity")
    lines.append("")
    lines.append(_md_table(summary[["strategy", "avg_allocation", "avg_open_exposure", "max_open_exposure",
                                    "avg_concurrent_positions", "max_concurrent_positions",
                                    "exposure_constrained_pct", "n_exposure_constrained"]]))
    lines.append("")
    lines.append("## 15. Equity Curves")
    lines.append("")
    lines.append("See plots: `all_equity.png`, `T1_T10_equity.png`, `A_F_equity.png`.")
    lines.append("")
    lines.append("## 16. Drawdown Curves")
    lines.append("")
    lines.append("See plot: `all_drawdown.png`.")
    lines.append("")
    lines.append("## 17. Important Findings")
    lines.append("")
    # reasoned overall
    top_ret = by_eq.iloc[0]["strategy"]
    top_ra = by_ra.iloc[0]["strategy"]
    top_stab = stab.iloc[0]["strategy"]
    top_pf = by_pf.iloc[0]["strategy"]
    lines.append(f"- Highest equity: **{top_ret}**")
    lines.append(f"- Best risk-adjusted: **{top_ra}**")
    lines.append(f"- Best monthly positivity: **{top_stab}**")
    lines.append(f"- Best profit factor: **{top_pf}**")
    e = summary[summary["strategy"] == "E"].iloc[0]
    lines.append(
        f"- Selector E @ 25%: +{e['return_pct']:.2f}% / DD {e['max_drawdown_pct']:.3f}% "
        f"(rank by equity: {int(by_eq.reset_index().index[by_eq.reset_index()['strategy']=='E'][0])+1}/16)."
    )
    lines.append("")
    lines.append("## 18. Limitations")
    lines.append("")
    lines.append(
        "- Single 365-day MEXC window.\n"
        "- 25% allocation amplifies both gains and losses (Part 4C).\n"
        "- Historical result ≠ future expected annual return.\n"
        "- Research-replay accounting matches Part 4C/4E (fraction × $1000)."
    )
    lines.append("")
    lines.append("## 19. Research Conclusion")
    lines.append("")
    lines.append("### Highest absolute return")
    lines.append(f"**{top_ret}**")
    lines.append("")
    lines.append("### Best risk-adjusted result")
    lines.append(f"**{top_ra}**")
    lines.append("")
    lines.append("### Best stability")
    lines.append(f"**{top_stab}** (by % positive months / mean monthly return).")
    lines.append("")
    lines.append("### Best trade quality")
    lines.append(f"**{top_pf}** (profit factor).")
    lines.append("")
    lines.append("### Best overall candidate")
    # Prefer strategies that appear in top tiers of equity AND risk-adjusted
    top5_eq = set(by_eq.head(5)["strategy"])
    top5_ra = set(by_ra.head(5)["strategy"])
    overlap = [s for s in by_eq["strategy"] if s in top5_eq and s in top5_ra]
    overall = overlap[0] if overlap else top_ra
    lines.append(
        f"**{overall}** — selected as a reasoned candidate because it ranks strongly on "
        f"both absolute equity and return/|DD| without relying on a single cherry-picked metric. "
        "Further robustness / walk-forward testing is required before any paper/live change."
    )
    lines.append("")
    lines.append(
        "These are 365-day historical backtest results using a fixed 25% allocation. "
        "They do **not** establish future annual returns. Strategy quality is distinct from "
        "the position-sizing effect already measured in Part 4C."
    )
    lines.append("")
    lines.append("**RESEARCH ONLY — NO DEPLOYMENT.**")
    lines.append("")
    lines.append("## Plots")
    lines.append("")
    for name, path in plot_paths.items():
        lines.append(f"- `{name}` → `{path}`")
    lines.append("")

    path = out_dir / "RESEARCH_REPORT_strategy_comparison_25pct_365d.md"
    path.write_text("\n".join(lines))
    return path


def emit_all(out_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    out_dir = Path(out_dir)
    summary = result["summary"].copy()

    # master comparison CSV with requested column names
    master = summary.rename(
        columns={
            "final_equity": "final_equity",
            "return_pct": "return_pct",
            "max_drawdown_pct": "max_drawdown_pct",
            "return_to_drawdown": "return_to_drawdown",
            "trades": "trades",
            "wins": "wins",
            "losses": "losses",
            "win_rate": "win_rate",
            "profit_factor": "profit_factor",
            "expectancy": "expectancy",
            "avg_trade_pct": "avg_trade_pct",
            "avg_winner_pct": "avg_winner_pct",
            "avg_loser_pct": "avg_loser_pct",
            "max_concurrent_positions": "max_concurrent_positions",
            "avg_concurrent_positions": "avg_concurrent_positions",
            "exposure_constrained_pct": "exposure_constrained_pct",
        }
    )
    master_cols = [
        "strategy", "type", "allocation", "final_equity", "return_pct", "max_drawdown_pct",
        "return_to_drawdown", "trades", "wins", "losses", "win_rate", "profit_factor",
        "expectancy", "avg_trade_pct", "avg_winner_pct", "avg_loser_pct",
        "max_concurrent_positions", "avg_concurrent_positions", "exposure_constrained_pct",
        "n_exposure_constrained", "n_partial", "n_skipped", "worst_day_pct", "worst_month",
        "worst_month_pct", "gross_profit", "gross_loss",
    ]
    master = master[[c for c in master_cols if c in master.columns]].sort_values(
        "final_equity", ascending=False
    )
    master.to_csv(out_dir / "strategy_comparison_25pct.csv", index=False)
    master.sort_values("return_to_drawdown", ascending=False).to_csv(
        out_dir / "strategy_comparison_25pct_by_risk_adjusted.csv", index=False
    )

    # monthly long
    monthly_parts = []
    for arm, df in result["monthly_tables"].items():
        d = df.copy()
        d.insert(0, "strategy", arm)
        monthly_parts.append(d)
    monthly_long = pd.concat(monthly_parts, ignore_index=True)
    monthly_long.to_csv(out_dir / "monthly_results_25pct.csv", index=False)

    # per-arm files
    for arm in ALL_ARMS:
        result["trade_tables"][arm].to_csv(out_dir / "trade_results" / f"{arm}.csv", index=False)
        result["daily_tables"][arm].to_csv(out_dir / "daily_equity" / f"{arm}.csv", index=False)

    plot_paths = plot_all(
        out_dir,
        summary=summary,
        trade_tables=result["trade_tables"],
        daily_tables=result["daily_tables"],
        monthly_long=monthly_long,
    )
    report = write_report(
        out_dir,
        result=result,
        summary=summary,
        monthly_long=monthly_long,
        plot_paths=plot_paths,
    )

    manifest = {
        "experiment": "strategy_comparison_25pct_365d_part5",
        "out_dir": str(out_dir),
        "arms": list(ALL_ARMS),
        "excluded": ["T11", "T12"],
        "fairness": result["fairness"],
        "e_check": result["e_check"],
        "top_by_equity": master.iloc[0].to_dict(),
        "report": str(report),
        "plots": plot_paths,
        "generated_at": result.get("generated_at"),
    }
    (out_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (out_dir / "fairness_checks.json").write_text(json.dumps(result["fairness"], indent=2))
    (out_dir / "e_part4c_check.json").write_text(json.dumps(result["e_check"], indent=2))
    return manifest
