"""Generate markdown report and optional plots from backtest analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "_No data_\n"
    sub = df.head(max_rows)
    try:
        return sub.to_markdown(index=False) + "\n"
    except Exception:
        return "```\n" + sub.to_string(index=False) + "\n```\n"


def generate_report(
    out_dir: Path,
    df: pd.DataFrame,
    analyses: dict[str, pd.DataFrame],
    meta: dict[str, Any],
) -> str:
    lines = [
        "# BTCC 90-Day Historical Prediction Research Report",
        "",
        "**RESEARCH ONLY — NO TRADING — NO TELEGRAM**",
        "",
        "## Run metadata",
        f"- Run ID: `{meta.get('run_id')}`",
        f"- Evaluation start (UTC): **{meta.get('eval_start')}**",
        f"- Evaluation end (UTC): **{meta.get('eval_end')}**",
        f"- Data start (UTC): {meta.get('data_start')}",
        f"- Days: {meta.get('days')}",
        f"- Decision interval: {meta.get('interval', '15m')}",
        f"- Decision bars: **{meta.get('decision_bars')}**",
        f"- Top-5 predictions stored: **{meta.get('top5_predictions')}**",
        f"- Valid symbols: {meta.get('valid_symbols')} / {meta.get('universe_size')}",
        f"- Unavailable: {meta.get('unavailable_symbols')}",
        f"- Orders sent: **{meta.get('orders_sent', 0)}**",
        "",
        "## Probability note",
        meta.get("probability_note", "baseline_model_probability"),
        "",
        "## Look-ahead policy",
        meta.get("lookahead_policy", ""),
        "",
        "## BTC dominance availability",
        f"- Source: {meta.get('dominance', {}).get('source', 'n/a')}",
        f"- Status: {meta.get('dominance_status')}",
        f"- Historical points: {meta.get('dominance', {}).get('n_points', 0)}",
        f"- Median spacing (hours): {meta.get('dominance', {}).get('median_spacing_hours')}",
        f"- Calibration: {meta.get('dominance', {}).get('calibration')}",
        f"- Decisions missing dominance: {meta.get('dominance_missing_at_decisions', 0)}",
        f"- Alignment: last-known observation ≤ decision time (btc_dominance_obs_ts recorded)",
        "",
        "---",
        "",
        "## A. Overall prediction performance",
        "",
        _table(analyses.get("horizon_performance", pd.DataFrame())),
        "",
        "## B. Probability calibration (predicted vs actual success rate)",
        "",
        _table(analyses.get("probability_calibration", pd.DataFrame())),
        "",
        "### Key question: does higher 4h baseline_model_probability → higher actual 4h success?",
        "",
    ]

    cal = analyses.get("probability_calibration", pd.DataFrame())
    cal4 = cal[cal["horizon_h"] == 4] if not cal.empty and "horizon_h" in cal.columns else cal
    if not cal4.empty:
        for _, row in cal4.iterrows():
            if row.get("n", 0) > 0:
                lines.append(
                    f"- Bucket {row['bucket']}: n={row['n']}, "
                    f"predicted≈{row.get('predicted_prob_mean', 0):.1%}, "
                    f"actual={row.get('actual_success_rate', 0):.1%}"
                )
    else:
        lines.append("_Insufficient data for calibration buckets._")

    lines += [
        "",
        "## C. Top-1 / Top-3 / Top-5 ranking value",
        "",
        _table(analyses.get("top_n_analysis", pd.DataFrame())),
        "",
        "## D. Factor analysis (weights unchanged — observation only)",
        "",
        _table(analyses.get("factor_analysis", pd.DataFrame()), 40),
        "",
        "## E. Individual indicator analysis",
        "",
        _table(analyses.get("indicator_analysis", pd.DataFrame()), 40),
        "",
        "## F. Late Entry analysis (separate from probability)",
        "",
        _table(analyses.get("late_entry_analysis", pd.DataFrame()), 40),
        "",
        "## G. BTC regime analysis",
        "",
        _table(analyses.get("btc_regime_analysis", pd.DataFrame())),
        "",
        "## H. Coin-by-coin analysis",
        "",
        _table(analyses.get("coin_analysis", pd.DataFrame())),
        "",
        "## Summary statistics",
        "",
    ]

    if not df.empty and "probability_4h" in df.columns:
        lines.append(f"- 4h probability min/max: {df['probability_4h'].min():.1%} / {df['probability_4h'].max():.1%}")
    if not df.empty and "late_entry_score" in df.columns:
        lines.append(f"- Late Entry min/max: {df['late_entry_score'].min():.3f} / {df['late_entry_score'].max():.3f}")

    lines += [
        "",
        "## Conclusion framework",
        "",
        "This report answers whether the **current** BTCC signal model, unchanged,",
        "predicted ALT/BTC outperformance over the evaluation window.",
        "",
        "Do NOT optimize weights or indicators based on this same 90-day window.",
        "",
        f"Output directory: `{out_dir}`",
    ]

    text = "\n".join(lines)
    (out_dir / "report.md").write_text(text, encoding="utf-8")
    return text


def generate_plots(out_dir: Path, df: pd.DataFrame, analyses: dict[str, pd.DataFrame]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if df.empty:
        return

    # Calibration curve 4h
    cal = analyses.get("probability_calibration", pd.DataFrame())
    cal4 = cal[cal["horizon_h"] == 4].dropna(subset=["predicted_prob_mean", "actual_success_rate"]) if not cal.empty else pd.DataFrame()
    if not cal4.empty and cal4["n"].sum() > 0:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
        ax.scatter(cal4["predicted_prob_mean"], cal4["actual_success_rate"], s=cal4["n"] / 5, alpha=0.7)
        for _, r in cal4.iterrows():
            ax.annotate(r["bucket"], (r["predicted_prob_mean"], r["actual_success_rate"]), fontsize=7)
        ax.set_xlabel("Predicted 4h baseline_model_probability (bucket mean)")
        ax.set_ylabel("Actual 4h success rate")
        ax.set_title("4h Probability Calibration")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "calibration_curve.png", dpi=120)
        plt.close(fig)

    # Success by horizon
    hp = analyses.get("horizon_performance", pd.DataFrame())
    if not hp.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(hp["horizon_h"].astype(str), hp["success_rate"])
        ax.set_xlabel("Horizon (hours)")
        ax.set_ylabel("Success rate")
        ax.set_title("Outperformance rate by horizon (Top-5 predictions)")
        fig.tight_layout()
        fig.savefig(out_dir / "success_by_horizon.png", dpi=120)
        plt.close(fig)

    # Late entry vs 4h return
    if "late_entry_score" in df.columns and "future_return_4h" in df.columns:
        sub = df.dropna(subset=["late_entry_score", "future_return_4h"])
        if len(sub) > 50:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(sub["late_entry_score"], sub["future_return_4h"], alpha=0.15, s=8)
            ax.axhline(0, color="gray", ls="--")
            ax.set_xlabel("Late Entry Score")
            ax.set_ylabel("Future 4h ALT/BTC return")
            ax.set_title("Late Entry vs future 4h return")
            fig.tight_layout()
            fig.savefig(out_dir / "late_entry_vs_return.png", dpi=120)
            plt.close(fig)
