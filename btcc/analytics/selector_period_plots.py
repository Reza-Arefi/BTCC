"""Plots for selector monthly / yearly period analysis (T1–T10 + A–F)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from btcc.analytics.selector_period_analysis import (
    PERIOD_ARM_LABELS,
    PERIOD_FIXED_LABELS,
    PERIOD_SELECTOR_LABELS,
)


def generate_period_plots(
    *,
    monthly: pd.DataFrame,
    yearly: pd.DataFrame | None = None,
    rankings: pd.DataFrame | None = None,
    out_dir: Path,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if monthly is None or monthly.empty:
        return
    _plot_monthly_heatmap(monthly, out_dir)
    _plot_cumulative_monthly(monthly, out_dir, arms=PERIOD_ARM_LABELS, title="Cumulative return % — all 16 arms", fname="cumulative_return_monthly_all.png")
    _plot_cumulative_monthly(monthly, out_dir, arms=PERIOD_FIXED_LABELS, title="Cumulative return % — fixed T1–T10", fname="cumulative_return_monthly_fixed.png")
    _plot_cumulative_monthly(monthly, out_dir, arms=PERIOD_SELECTOR_LABELS, title="Cumulative return % — selectors A–F", fname="cumulative_return_monthly_selectors.png")
    _plot_profitable_months(monthly, out_dir)
    if yearly is not None and not yearly.empty:
        _plot_yearly_bars(yearly, out_dir)


def _plot_monthly_heatmap(monthly: pd.DataFrame, out_dir: Path) -> None:
    pivot = monthly.pivot_table(
        index="arm", columns="month_index", values="monthly_return_pct", aggfunc="first"
    )
    pivot = pivot.reindex(PERIOD_ARM_LABELS)
    # Prefer month_id labels on x if available
    id_map = (
        monthly.drop_duplicates("month_index")
        .set_index("month_index")["month_id"]
        .to_dict()
    )
    fig_w = max(14, 0.35 * max(len(pivot.columns), 1) + 4)
    fig, ax = plt.subplots(figsize=(fig_w, 8))
    data = pivot.values.astype(float)
    vmax = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1.0
    vmax = max(float(vmax), 1.0)
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(list(pivot.index))
    ax.set_xticks(range(len(pivot.columns)))
    labels = [id_map.get(int(c), str(c)) for c in pivot.columns]
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_xlabel("Month")
    ax.set_ylabel("Arm")
    ax.set_title("Strategy × month — monthly return %")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Monthly return %")
    fig.tight_layout()
    fig.savefig(out_dir / "monthly_return_heatmap.png", dpi=140)
    plt.close(fig)


def _plot_cumulative_monthly(
    monthly: pd.DataFrame,
    out_dir: Path,
    *,
    arms: tuple[str, ...],
    title: str,
    fname: str,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    for arm in arms:
        g = monthly[monthly["arm"] == arm].sort_values("month_index")
        if g.empty:
            continue
        ax.plot(g["month_index"], g["cumulative_return_pct"], label=arm, linewidth=1.8 if arm in PERIOD_SELECTOR_LABELS else 1.3)
    ax.set_xlabel("Month index")
    ax.set_ylabel("Cumulative return (%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=130)
    plt.close(fig)


def _plot_profitable_months(monthly: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    for arm in PERIOD_ARM_LABELS:
        g = monthly[monthly["arm"] == arm]
        if g.empty:
            continue
        n = len(g)
        n_prof = int((g["monthly_return_pct"] > 0).sum())
        rows.append({"arm": arm, "pct": 100.0 * n_prof / n})
    df = pd.DataFrame(rows)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["steelblue" if a in PERIOD_FIXED_LABELS else "darkorange" for a in df["arm"]]
    bars = ax.bar(df["arm"], df["pct"], color=colors)
    for bar, v in zip(bars, df["pct"]):
        ax.annotate(f"{v:.0f}", xy=(bar.get_x() + bar.get_width() / 2, v), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8)
    ax.set_ylabel("% profitable months")
    ax.set_title("Monthly consistency — share of profitable months")
    ax.set_ylim(0, 110)
    fig.tight_layout()
    fig.savefig(out_dir / "profitable_months_share.png", dpi=120)
    plt.close(fig)


def _plot_yearly_bars(yearly: pd.DataFrame, out_dir: Path) -> None:
    # Period return by arm for Y1/Y2/Y3/FULL
    periods = [p for p in ("Y1", "Y2", "Y3", "FULL") if p in set(yearly["period"])]
    if not periods:
        return
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(PERIOD_ARM_LABELS))
    width = 0.18
    for i, per in enumerate(periods):
        vals = []
        for arm in PERIOD_ARM_LABELS:
            g = yearly[(yearly["arm"] == arm) & (yearly["period"] == per)]
            vals.append(float(g.iloc[0]["period_return_pct"]) if len(g) else 0.0)
        ax.bar(x + (i - len(periods) / 2) * width + width / 2, vals, width, label=per)
    ax.set_xticks(x)
    ax.set_xticklabels(PERIOD_ARM_LABELS)
    ax.set_ylabel("Period return %")
    ax.set_title("Yearly / full-period returns by arm")
    ax.axhline(0, color="#222", lw=0.8)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "yearly_returns_by_arm.png", dpi=130)
    plt.close(fig)
