"""Plot generation for E-memory walk-forward experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from btcc.sim.selector_memory_config import CF_ARM_LABELS, MEMORY_ARM_LABELS, selector_arm_labels

_ARM_COLORS = {
    **{a: c for a, c in zip(MEMORY_ARM_LABELS, plt.cm.viridis(np.linspace(0.1, 0.9, len(MEMORY_ARM_LABELS))))},
    "T1_NO_LATE": "#d62728",
    "T1_LATE_FILTER": "#ff7f0e",
}


def _arm_color(arm: str) -> Any:
    return _ARM_COLORS.get(arm, "#888888")


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def generate_memory_plots(
    analytics_root: Path,
    *,
    metrics: dict[str, Any],
    regret_df: pd.DataFrame,
    regret_stats: dict[str, Any],
    behavior: dict[str, Any],
    daily_cap: pd.DataFrame,
    regime: dict[str, Any],
    s_band: dict[str, Any],
    max_day: int,
) -> None:
    plots = Path(analytics_root) / "plots" / "memory"
    plots.mkdir(parents=True, exist_ok=True)

    trading_arms = selector_arm_labels()

    # 1. Cumulative return
    if not daily_cap.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        for arm in trading_arms:
            g = daily_cap[daily_cap["strategy_key"] == arm].sort_values("day_number")
            if g.empty:
                continue
            ret = 100.0 * (g["equity_usd"] / g["equity_usd"].iloc[0] - 1.0)
            ax.plot(g["day_number"], ret, label=arm, color=_arm_color(arm), linewidth=2)
        ax.set_title("Cumulative return — E lookbacks + T1 baselines")
        ax.set_xlabel("Day")
        ax.set_ylabel("Return %")
        ax.legend()
        ax.grid(True, alpha=0.3)
        _save(fig, plots / "cumulative_return_all.png")

        # 2. Drawdown
        fig, ax = plt.subplots(figsize=(10, 5))
        for arm in trading_arms:
            g = daily_cap[daily_cap["strategy_key"] == arm].sort_values("day_number")
            if g.empty:
                continue
            curve = g["equity_usd"]
            dd = 100.0 * (curve / curve.cummax() - 1.0)
            ax.plot(g["day_number"], dd, label=arm, color=_arm_color(arm), linewidth=2)
        ax.set_title("Drawdown — E lookbacks + T1 baselines")
        ax.set_xlabel("Day")
        ax.set_ylabel("Drawdown %")
        ax.legend()
        ax.grid(True, alpha=0.3)
        _save(fig, plots / "drawdown_all.png")

        # 10. Rolling 30d return
        fig, ax = plt.subplots(figsize=(10, 5))
        for arm in trading_arms:
            g = daily_cap[daily_cap["strategy_key"] == arm].sort_values("day_number")
            if len(g) < 2:
                continue
            ret = g["equity_usd"].pct_change().rolling(30, min_periods=5).sum() * 100.0
            ax.plot(g["day_number"], ret, label=arm, color=_arm_color(arm), alpha=0.85)
        ax.set_title("30-day rolling return")
        ax.set_xlabel("Day")
        ax.set_ylabel("Rolling return %")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        _save(fig, plots / "rolling_30d_return.png")

    # 3. Avg P/L per trade
    fig, ax = plt.subplots(figsize=(10, 4))
    vals = [float((metrics.get(a) or {}).get("avg_trade_return_pct") or 0.0) for a in trading_arms]
    ax.bar(trading_arms, vals, color=[_arm_color(a) for a in trading_arms])
    ax.set_title("Average P/L per trade")
    ax.set_ylabel("Return %")
    _save(fig, plots / "avg_pnl_per_trade_all.png")

    # 4. Profit factor
    fig, ax = plt.subplots(figsize=(10, 4))
    vals = [float((metrics.get(a) or {}).get("profit_factor") or 0.0) for a in trading_arms]
    ax.bar(trading_arms, vals, color=[_arm_color(a) for a in trading_arms])
    ax.set_title("Profit factor")
    ax.set_ylabel("PF")
    _save(fig, plots / "profit_factor_all.png")

    # 6. Selection entropy
    fig, ax = plt.subplots(figsize=(8, 4))
    vals = [float((behavior.get(a) or {}).get("entropy") or 0.0) for a in MEMORY_ARM_LABELS]
    ax.bar(MEMORY_ARM_LABELS, vals, color=[_arm_color(a) for a in MEMORY_ARM_LABELS])
    ax.set_title("Selection entropy")
    ax.set_ylabel("Entropy")
    _save(fig, plots / "selection_entropy_all.png")

    # 5. Selection frequency heatmap
    rows = []
    for arm in MEMORY_ARM_LABELS:
        freq = (behavior.get(arm) or {}).get("selection_frequency") or {}
        row = {f"T{i}": float(freq.get(f"T{i}", 0.0)) for i in range(1, 11)}
        row["arm"] = arm
        rows.append(row)
    if rows:
        df = pd.DataFrame(rows).set_index("arm")
        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(df.values * 100.0, aspect="auto", cmap="Blues")
        ax.set_xticks(range(len(df.columns)))
        ax.set_xticklabels(df.columns)
        ax.set_yticks(range(len(df.index)))
        ax.set_yticklabels(df.index)
        ax.set_title("Selection frequency (%)")
        fig.colorbar(im, ax=ax, label="%")
        _save(fig, plots / "selection_frequency_heatmap.png")

    # 7. Regret distributions
    if not regret_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        data = [regret_df[regret_df["arm_label"] == a]["regret_pct"].dropna().values for a in trading_arms]
        ax.boxplot(data, tick_labels=trading_arms)
        ax.set_title("Selection regret distribution")
        ax.set_ylabel("Regret (pp)")
        ax.grid(True, axis="y", alpha=0.3)
        _save(fig, plots / "regret_distribution_all.png")

    # 8. Regime heatmap (return %)
    if regime:
        arms = trading_arms
        reg_keys = sorted({r for a in arms for r in (regime.get(a) or {})})
        mat = []
        for arm in arms:
            mat.append([float(((regime.get(arm) or {}).get(r) or {}).get("cumulative_return_pct") or 0.0) for r in reg_keys])
        if mat and reg_keys:
            fig, ax = plt.subplots(figsize=(10, 5))
            im = ax.imshow(mat, aspect="auto", cmap="RdYlGn")
            ax.set_xticks(range(len(reg_keys)))
            ax.set_xticklabels(reg_keys, rotation=45, ha="right")
            ax.set_yticks(range(len(arms)))
            ax.set_yticklabels(arms)
            ax.set_title("Return % by regime × arm")
            fig.colorbar(im, ax=ax)
            _save(fig, plots / "regime_heatmap.png")

    # 9. S-band heatmap
    if s_band:
        arms = trading_arms
        sb_keys = sorted({b for a in arms for b in (s_band.get(a) or {})})
        mat = []
        for arm in arms:
            mat.append([float(((s_band.get(arm) or {}).get(b) or {}).get("avg_trade_return_pct") or 0.0) for b in sb_keys])
        if mat and sb_keys:
            fig, ax = plt.subplots(figsize=(8, 5))
            im = ax.imshow(mat, aspect="auto", cmap="RdYlGn")
            ax.set_xticks(range(len(sb_keys)))
            ax.set_xticklabels(sb_keys, rotation=30, ha="right")
            ax.set_yticks(range(len(arms)))
            ax.set_yticklabels(arms)
            ax.set_title("Avg trade return % by S-band × arm")
            fig.colorbar(im, ax=ax)
            _save(fig, plots / "sband_heatmap.png")
