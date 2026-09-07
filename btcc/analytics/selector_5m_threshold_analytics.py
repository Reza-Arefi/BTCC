"""Aggregate analytics for the 5m S-threshold sweep (16 arms × N thresholds)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from btcc.analytics.selector_period_analysis import PERIOD_ARM_LABELS, write_period_analysis
from btcc.analytics.selector_period_plots import generate_period_plots

ARM_ORDER = list(PERIOD_ARM_LABELS)  # T1–T10 + A–F


def _load_metrics(run_dir: Path) -> dict[str, Any] | None:
    p = run_dir / "analytics" / "selector_metrics.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    # Fallback: build light metrics from legs if analytics missing
    return None


def _load_summary(run_dir: Path) -> dict[str, Any]:
    p = run_dir / "summary.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _thr_from_dirname(name: str) -> float | None:
    # thr0p60 -> 0.60
    if not name.startswith("thr"):
        return None
    try:
        return float(name.replace("thr", "").replace("p", "."))
    except ValueError:
        return None


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def collect_threshold_arm_table(parent_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for child in sorted(Path(parent_dir).iterdir()):
        if not child.is_dir():
            continue
        thr = _thr_from_dirname(child.name)
        if thr is None:
            continue
        summary = _load_summary(child)
        metrics = _load_metrics(child) or {}
        by_arm = (metrics.get("metrics") or {}) if isinstance(metrics, dict) else {}
        n_opps = summary.get("n_opportunities")
        for arm in ARM_ORDER:
            m = by_arm.get(arm) or {}
            rows.append(
                {
                    "threshold": thr,
                    "threshold_label": f"> {thr:.2f}",
                    "arm": arm,
                    "cumulative_return_pct": m.get("cumulative_return_pct"),
                    "final_equity_usd": m.get("final_equity_usd"),
                    "total_pnl_usd": m.get("total_pnl_usd"),
                    "n_trades": m.get("n_trades"),
                    "win_rate_pct": m.get("win_rate_pct"),
                    "profit_factor": m.get("profit_factor"),
                    "avg_trade_return_pct": m.get("avg_trade_return_pct"),
                    "n_opportunities": n_opps,
                    "max_drawdown_pct": m.get("max_drawdown_pct"),
                    "run_dir": str(child),
                    "candle_interval": summary.get("candle_interval"),
                    "threshold_operator": summary.get("threshold_operator"),
                    "coverage_days": summary.get("actual_normal_pair_coverage_days"),
                }
            )
        legs = _safe_read_csv(child / "strategy_legs.csv")
        if legs.empty:
            continue
        closed = legs[legs["closed"] == True] if "closed" in legs.columns else legs  # noqa: E712
        for i, row in enumerate(list(rows)):
            if row["run_dir"] != str(child):
                continue
            if "arm_key" not in closed.columns:
                continue
            g = closed[closed["arm_key"] == row["arm"]]
            if g.empty:
                continue
            if row.get("n_trades") is None:
                rows[i]["n_trades"] = int(len(g))
            if "pnl_pct" in g.columns:
                wins = g[g["pnl_pct"].astype(float) > 0]
                losses = g[g["pnl_pct"].astype(float) <= 0]
                rows[i]["n_wins"] = int(len(wins))
                rows[i]["n_losses"] = int(len(losses))
                rows[i]["median_trade_return_pct"] = float(g["pnl_pct"].astype(float).median() * 100.0)
            if "holding_hours" in g.columns:
                rows[i]["avg_holding_hours"] = float(g["holding_hours"].astype(float).mean())
            fee_cols = [c for c in ("fees_btc", "fee_btc") if c in g.columns]
            slip_cols = [c for c in ("slippage_btc_approx", "slippage_btc") if c in g.columns]
            if fee_cols:
                rows[i]["fees_btc_sum"] = float(g[fee_cols[0]].astype(float).sum())
            if slip_cols:
                rows[i]["slippage_btc_sum"] = float(g[slip_cols[0]].astype(float).sum())
    return pd.DataFrame(rows)


def threshold_summary_table(arm_table: pd.DataFrame) -> pd.DataFrame:
    """One row per threshold: best arm, E/T1/F, trade counts."""
    rows = []
    for thr, g in arm_table.groupby("threshold"):
        g2 = g.dropna(subset=["cumulative_return_pct"]).sort_values("cumulative_return_pct", ascending=False)
        if g2.empty:
            continue
        best = g2.iloc[0]

        def _arm(a: str) -> pd.Series:
            hit = g2[g2["arm"] == a]
            return hit.iloc[0] if len(hit) else pd.Series(dtype=object)

        e, t1, f = _arm("E"), _arm("T1"), _arm("F")
        rows.append(
            {
                "threshold": thr,
                "threshold_label": f"> {float(thr):.2f}",
                "best_arm": best["arm"],
                "best_return_pct": best["cumulative_return_pct"],
                "best_dd_pct": best.get("max_drawdown_pct"),
                "n_opportunities": best.get("n_opportunities"),
                "n_trades_best": best.get("n_trades"),
                "E_return_pct": e.get("cumulative_return_pct"),
                "E_trades": e.get("n_trades"),
                "T1_return_pct": t1.get("cumulative_return_pct"),
                "T1_trades": t1.get("n_trades"),
                "F_return_pct": f.get("cumulative_return_pct"),
                "F_trades": f.get("n_trades"),
            }
        )
    return pd.DataFrame(rows).sort_values("threshold")


def _heatmap(df: pd.DataFrame, value_col: str, out_path: Path, title: str, cmap: str = "RdYlGn") -> None:
    if df.empty or value_col not in df.columns:
        return
    pivot = df.pivot_table(index="arm", columns="threshold", values=value_col, aggfunc="first")
    pivot = pivot.reindex(ARM_ORDER)
    pivot = pivot.reindex(columns=sorted(pivot.columns))
    fig_w = max(10, 1.2 * max(len(pivot.columns), 1) + 3)
    fig, ax = plt.subplots(figsize=(fig_w, 8))
    data = pivot.values.astype(float)
    finite = data[np.isfinite(data)]
    if value_col.endswith("drawdown") or "dd" in value_col.lower():
        vmax = np.nanmax(np.abs(finite)) if finite.size else 1.0
        im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r", vmin=-vmax, vmax=0 if (finite < 0).any() else vmax)
    else:
        vmax = np.nanmax(np.abs(finite)) if finite.size else 1.0
        vmax = max(float(vmax), 1.0)
        im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(list(pivot.index))
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"> {c:.2f}" for c in pivot.columns], rotation=45, ha="right")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def ensure_period_analysis_for_runs(parent_dir: Path) -> None:
    for child in sorted(Path(parent_dir).iterdir()):
        if not child.is_dir() or _thr_from_dirname(child.name) is None:
            continue
        period_dir = child / "analytics" / "periods"
        if (period_dir / "monthly_results.csv").exists():
            continue
        legs_p = child / "strategy_legs.csv"
        man_p = child / "experiment_manifest.json"
        if not legs_p.exists() or not man_p.exists():
            continue
        legs = _safe_read_csv(legs_p)
        if legs.empty:
            continue
        sel_p = child / "selection_audit.csv"
        selection = _safe_read_csv(sel_p)
        man = json.loads(man_p.read_text(encoding="utf-8"))
        paths = write_period_analysis(
            child,
            legs=legs,
            selection=selection,
            eval_start=man["eval_start"],
            eval_end=man["eval_end"],
            starting_capital_usd=1000.0,
        )
        monthly = pd.read_csv(paths["monthly_results.csv"])
        yearly = pd.read_csv(paths["yearly_results.csv"])
        rankings = pd.read_csv(paths["monthly_rankings.csv"])
        generate_period_plots(
            monthly=monthly,
            yearly=yearly,
            rankings=rankings,
            out_dir=child / "analytics" / "plots" / "periods",
        )


def write_threshold_sweep_analytics(parent_dir: Path) -> dict[str, Path]:
    parent_dir = Path(parent_dir)
    # Prefer using existing selector_metrics from each run; rebuild periods if enabled/missing
    ensure_period_analysis_for_runs(parent_dir)

    arm_table = collect_threshold_arm_table(parent_dir)
    out_csv = parent_dir / "analytics"
    out_csv.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    arm_path = out_csv / "threshold_arm_metrics.csv"
    arm_table.to_csv(arm_path, index=False)
    paths["threshold_arm_metrics.csv"] = arm_path

    summary = threshold_summary_table(arm_table)
    sum_path = out_csv / "threshold_summary.csv"
    summary.to_csv(sum_path, index=False)
    paths["threshold_summary.csv"] = sum_path

    plots = parent_dir / "analytics" / "plots"
    _heatmap(
        arm_table,
        "cumulative_return_pct",
        plots / "heatmap_cum_return.png",
        "5m threshold sweep — cumulative return %",
    )
    if arm_table["max_drawdown_pct"].notna().any():
        _heatmap(
            arm_table,
            "max_drawdown_pct",
            plots / "heatmap_max_dd.png",
            "5m threshold sweep — max drawdown %",
            cmap="RdYlGn_r",
        )
    if arm_table["profit_factor"].notna().any():
        _heatmap(
            arm_table,
            "profit_factor",
            plots / "heatmap_profit_factor.png",
            "5m threshold sweep — profit factor",
        )

    # Focus lines E / T1 / F vs threshold
    if not arm_table.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        for arm, color in [("E", "#1f77b4"), ("T1", "#2ca02c"), ("F", "#ff7f0e")]:
            g = arm_table[arm_table["arm"] == arm].sort_values("threshold")
            if g.empty:
                continue
            ax.plot(g["threshold"], g["cumulative_return_pct"], marker="o", label=arm, color=color)
        ax.set_xlabel("S threshold (strict >)")
        ax.set_ylabel("Cumulative return %")
        ax.set_title("E / T1 / F vs 5m threshold")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        focus = plots / "E_T1_F_vs_threshold.png"
        fig.savefig(focus, dpi=120, bbox_inches="tight")
        plt.close(fig)
        paths["E_T1_F_vs_threshold.png"] = focus

        fig, ax = plt.subplots(figsize=(10, 5))
        g = arm_table[arm_table["arm"] == "T1"].sort_values("threshold")
        if not g.empty and g["n_opportunities"].notna().any():
            ax.bar(g["threshold"].astype(str), g["n_opportunities"], color="#888")
            ax.set_xlabel("S threshold")
            ax.set_ylabel("Opportunities")
            ax.set_title("Opportunity count vs threshold (shared book)")
            fig.tight_layout()
            op_path = plots / "opportunities_vs_threshold.png"
            fig.savefig(op_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            paths["opportunities_vs_threshold.png"] = op_path

    meta = {
        "analysis": "selector_5m_threshold_sweep_v1",
        "parent_dir": str(parent_dir),
        "n_threshold_runs": int(arm_table["threshold"].nunique()) if not arm_table.empty else 0,
        "arms": ARM_ORDER,
        "files": {k: str(v) for k, v in paths.items()},
    }
    meta_path = out_csv / "sweep_analytics_manifest.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    paths["sweep_analytics_manifest.json"] = meta_path
    return paths
