"""Plot generation for selector experiment (T1–T12 + A–F)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from btcc.sim.selector_config import ALL_ARM_LABELS, FIXED_ARM_LABELS, SELECTOR_ARM_LABELS

# 18 distinct styles: numbered 1–18 (T1…T12, then A…F).
_ARM_COLORS = plt.cm.tab20(np.linspace(0, 0.95, len(ALL_ARM_LABELS)))
_ARM_MARKERS = ("o", "s", "^", "v", "D", "p", "*", "h", "X", "P", "d", "8", "+", "x", "1", "2", "3", "4")
_ARM_LINESTYLES = (
    "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-",
    (0, (5, 2)), (0, (5, 2)), (0, (5, 2)), (0, (5, 2)), (0, (5, 2)), (0, (5, 2)),
)


def _arm_line_number(arm: str) -> int:
    return ALL_ARM_LABELS.index(arm) + 1


def _style_for_arm(arm: str) -> dict[str, object]:
    i = ALL_ARM_LABELS.index(arm)
    is_selector = arm in SELECTOR_ARM_LABELS
    return {
        "color": _ARM_COLORS[i],
        "linestyle": _ARM_LINESTYLES[i],
        "marker": _ARM_MARKERS[i],
        "linewidth": 2.2 if is_selector else 1.6,
        "alpha": 1.0,
        "zorder": 3 if is_selector else 2,
    }


def _markevery(max_day: int) -> int:
    return max(1, max_day // 12)


def _right_label_x(max_day: int) -> float:
    """X position for end-of-line labels, just outside the day axis."""
    return max_day + max(2.0, max_day * 0.04)


def _add_right_arm_labels(
    ax,
    endpoints: list[tuple[str, float, float]],
    *,
    max_day: int,
) -> None:
    """Place numbered arm labels (e.g. '3 T3') to the right of the plot area."""
    label_x = _right_label_x(max_day)
    # Sort by y so labels don't overlap as badly; stagger slightly if needed.
    endpoints_sorted = sorted(endpoints, key=lambda e: e[2])
    n = len(endpoints_sorted)
    min_gap = 0.0
    if n > 1:
        ys = [e[2] for e in endpoints_sorted]
        span = max(abs(max(ys) - min(ys)), 1e-6)
        min_gap = span * 0.025
    adjusted_y: list[float] = []
    prev = -1e18
    for _, _, y in endpoints_sorted:
        y_adj = max(y, prev + min_gap)
        adjusted_y.append(y_adj)
        prev = y_adj
    y_map = {arm: y_adj for (arm, _, _), y_adj in zip(endpoints_sorted, adjusted_y)}
    for arm, _x_last, y_last in endpoints:
        st = _style_for_arm(arm)
        num = _arm_line_number(arm)
        ax.text(
            label_x,
            y_map.get(arm, y_last),
            f"{num} {arm}",
            fontsize=8,
            fontweight="bold" if arm in SELECTOR_ARM_LABELS else "normal",
            color=st["color"],
            va="center",
            ha="left",
            clip_on=False,
        )


def _plot_multi_arm_timeseries(
    ax,
    daily_cap: pd.DataFrame,
    *,
    max_day: int,
    y_fn,
    title: str,
    ylabel: str,
) -> None:
    """Draw all 18 arms with distinct shape/color and numbered right-side labels."""
    endpoints: list[tuple[str, float, float]] = []
    me = _markevery(max_day)
    x_right = _right_label_x(max_day)
    ax.set_xlim(1, x_right)
    for arm in ALL_ARM_LABELS:
        g = daily_cap[daily_cap["strategy_key"] == arm].sort_values("day_number")
        if g.empty:
            continue
        st = _style_for_arm(arm)
        y = y_fn(g)
        x = g["day_number"].astype(float)
        ax.plot(
            x,
            y,
            color=st["color"],
            linestyle=st["linestyle"],
            marker=st["marker"],
            markevery=me,
            markersize=5 if arm in SELECTOR_ARM_LABELS else 4,
            linewidth=st["linewidth"],
            alpha=st["alpha"],
            zorder=st["zorder"],
        )
        endpoints.append((arm, float(x.iloc[-1]), float(y.iloc[-1])))
    ax.set_xlabel("Day")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if endpoints:
        _add_right_arm_labels(ax, endpoints, max_day=max_day)


def generate_selector_plots(
    legs: pd.DataFrame,
    *,
    daily_cap: pd.DataFrame,
    trade_cap: pd.DataFrame,
    out_dir: Path,
    max_day: int,
    starting_capital_usd: float = 1000.0,
    opportunities: pd.DataFrame | None = None,
    selection: pd.DataFrame | None = None,
    regret: pd.DataFrame | None = None,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if legs.empty:
        return

    legs = legs.copy()
    if "arm_key" not in legs.columns:
        legs["arm_key"] = legs.get("strategy_key")

    _plot_cumulative_return(daily_cap, out_dir, max_day)
    _plot_drawdown(daily_cap, out_dir, max_day)
    _plot_win_rate(legs, out_dir)
    _plot_profit_factor(legs, out_dir)
    _plot_avg_pnl(legs, out_dir)
    _plot_trade_count(legs, out_dir, max_day)
    _plot_mfe_mae(legs, out_dir)
    _plot_regime_heatmap(legs, out_dir)
    _plot_s_band_heatmap(legs, opportunities, out_dir)
    _plot_ranking_over_time(daily_cap, out_dir, max_day)
    if selection is not None and not selection.empty:
        _plot_selection_frequency(selection, out_dir)
        _plot_switching_timeline(selection, out_dir, max_day)
    if regret is not None and not regret.empty:
        _plot_selection_regret(regret, out_dir)
    _plot_fixed_vs_selector(daily_cap, out_dir, max_day)


def _plot_cumulative_return(daily_cap: pd.DataFrame, out_dir: Path, max_day: int) -> None:
    if daily_cap.empty:
        return
    fig, ax = plt.subplots(figsize=(15, 8))
    fig.subplots_adjust(right=0.88)
    _plot_multi_arm_timeseries(
        ax,
        daily_cap,
        max_day=max_day,
        y_fn=lambda g: g["cumulative_return_pct"].astype(float),
        title="Cumulative return — T1–T12 + A–F",
        ylabel="Cumulative return (%)",
    )
    fig.savefig(out_dir / "cumulative_return_all.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_drawdown(daily_cap: pd.DataFrame, out_dir: Path, max_day: int) -> None:
    if daily_cap.empty:
        return
    fig, ax = plt.subplots(figsize=(15, 8))
    fig.subplots_adjust(right=0.88)

    def _dd(g: pd.DataFrame) -> pd.Series:
        eq = g["ending_value"].astype(float)
        peak = eq.cummax()
        return 100 * (eq / peak - 1.0)

    _plot_multi_arm_timeseries(
        ax,
        daily_cap,
        max_day=max_day,
        y_fn=_dd,
        title="Drawdown (%)",
        ylabel="Drawdown (%)",
    )
    fig.savefig(out_dir / "drawdown_all.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_win_rate(legs: pd.DataFrame, out_dir: Path) -> None:
    closed = legs[legs.get("closed", True) == True]  # noqa: E712
    wr = []
    for arm in ALL_ARM_LABELS:
        g = closed[closed["arm_key"] == arm]
        wr.append(100 * (pd.to_numeric(g["pnl_pct"], errors="coerce") > 0).mean() if len(g) else 0)
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["steelblue"] * 12 + ["darkorange"] * 6
    ax.bar(list(ALL_ARM_LABELS), wr, color=colors)
    ax.set_ylabel("Win rate (%)")
    ax.set_title("Win rate by arm")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_dir / "win_rate_all.png", dpi=120)
    plt.close(fig)


def _plot_profit_factor(legs: pd.DataFrame, out_dir: Path) -> None:
    pnl = pd.to_numeric(legs.get("pnl_usd_equiv"), errors="coerce").fillna(0)
    pfs = []
    for arm in ALL_ARM_LABELS:
        m = legs["arm_key"] == arm
        g = pnl[m]
        gp = g[g > 0].sum()
        gl = abs(g[g < 0].sum())
        pfs.append(gp / gl if gl > 1e-9 else 0)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(list(ALL_ARM_LABELS), pfs, color="darkorange")
    ax.set_ylabel("Profit factor")
    ax.set_title("Profit factor by arm")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_dir / "profit_factor_all.png", dpi=120)
    plt.close(fig)


def _plot_avg_pnl(legs: pd.DataFrame, out_dir: Path) -> None:
    avgs = []
    for arm in ALL_ARM_LABELS:
        g = legs[legs["arm_key"] == arm]
        avgs.append(100 * float(pd.to_numeric(g.get("pnl_pct"), errors="coerce").mean()) if len(g) else 0)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(list(ALL_ARM_LABELS), avgs, color="seagreen")
    ax.set_ylabel("Avg P/L per trade (%)")
    ax.set_title("Average P/L per trade")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_dir / "avg_pnl_per_trade_all.png", dpi=120)
    plt.close(fig)


def _plot_trade_count(legs: pd.DataFrame, out_dir: Path, max_day: int) -> None:
    if "entry_day_number" not in legs.columns:
        legs = legs.copy()
        legs["entry_day_number"] = legs.get("day_number")
    rows = []
    for arm in ALL_ARM_LABELS:
        g = legs[legs["arm_key"] == arm]
        if g.empty:
            continue
        c = g.groupby("entry_day_number").size()
        for d, n in c.items():
            if pd.notna(d) and 1 <= int(d) <= max_day:
                rows.append({"arm": arm, "day_number": int(d), "count": int(n)})
    if not rows:
        return
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(12, 6))
    for arm in ALL_ARM_LABELS[:4]:
        g = df[df["arm"] == arm]
        if not g.empty:
            ax.plot(g["day_number"], g["count"], label=arm, alpha=0.7)
    ax.set_xlabel("Day")
    ax.set_ylabel("Trades entered")
    ax.set_title("Trade count per day (sample arms)")
    ax.set_xlim(1, max_day)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "trade_count_per_day.png", dpi=120)
    plt.close(fig)


def _plot_mfe_mae(legs: pd.DataFrame, out_dir: Path) -> None:
    mfe = pd.to_numeric(legs.get("mfe_pct"), errors="coerce").dropna()
    mae = pd.to_numeric(legs.get("mae_pct"), errors="coerce").dropna()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    if len(mfe):
        axes[0].hist(100 * mfe, bins=40, color="steelblue", alpha=0.8)
    axes[0].set_title("MFE distribution (%)")
    if len(mae):
        axes[1].hist(100 * mae, bins=40, color="coral", alpha=0.8)
    axes[1].set_title("MAE distribution (%)")
    fig.tight_layout()
    fig.savefig(out_dir / "mfe_mae_distribution.png", dpi=120)
    plt.close(fig)
    if "realized_over_mfe" in legs.columns or "mfe_pct" in legs.columns:
        pnl = pd.to_numeric(legs.get("pnl_pct"), errors="coerce")
        mfe_v = pd.to_numeric(legs.get("mfe_pct"), errors="coerce")
        ratio = pnl / mfe_v.replace(0, np.nan)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(ratio.dropna(), bins=40, color="purple", alpha=0.7)
        ax.set_title("Realized return / MFE")
        fig.tight_layout()
        fig.savefig(out_dir / "realized_over_mfe.png", dpi=120)
        plt.close(fig)


def _plot_regime_heatmap(legs: pd.DataFrame, out_dir: Path) -> None:
    cf = legs[legs.get("is_counterfactual") == True] if "is_counterfactual" in legs.columns else legs  # noqa: E712
    if cf.empty or "regime" not in cf.columns:
        return
    pivot_rows = []
    for regime, g in cf.groupby("regime"):
        row = {"regime": regime}
        for arm in FIXED_ARM_LABELS:
            ga = g[g["arm_key"] == arm]
            row[arm] = 100 * float(pd.to_numeric(ga["pnl_pct"], errors="coerce").mean()) if len(ga) else 0.0
        pivot_rows.append(row)
    if not pivot_rows:
        return
    df = pd.DataFrame(pivot_rows).set_index("regime")
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(df.values, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=45)
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index)
    ax.set_title("Strategy × regime avg return (%) — counterfactual T1–T12")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "regime_strategy_heatmap.png", dpi=120)
    plt.close(fig)


def _plot_s_band_heatmap(legs: pd.DataFrame, opps: pd.DataFrame | None, out_dir: Path) -> None:
    cf = legs[legs.get("is_counterfactual") == True] if "is_counterfactual" in legs.columns else legs  # noqa: E712
    if cf.empty:
        return
    s_map = {}
    if opps is not None and not opps.empty:
        s_map = opps.set_index("opportunity_id")["S"].astype(float).to_dict()
    cf = cf.copy()
    cf["S"] = cf["opportunity_id"].map(s_map)
    cf = cf.dropna(subset=["S"])
    if cf.empty:
        return

    def band(s: float) -> str:
        if s < 0.65:
            return "0.60-0.65"
        if s < 0.70:
            return "0.65-0.70"
        if s < 0.80:
            return "0.70-0.80"
        return "0.80+"

    cf["s_band"] = cf["S"].map(band)
    pivot_rows = []
    for band, g in cf.groupby("s_band"):
        row = {"s_band": band}
        for arm in FIXED_ARM_LABELS:
            ga = g[g["arm_key"] == arm]
            row[arm] = 100 * float(pd.to_numeric(ga["pnl_pct"], errors="coerce").mean()) if len(ga) else 0.0
        pivot_rows.append(row)
    df = pd.DataFrame(pivot_rows).set_index("s_band")
    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(df.values, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=45)
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index)
    ax.set_title("Strategy × S-band avg return (%)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "s_band_strategy_heatmap.png", dpi=120)
    plt.close(fig)


def _plot_ranking_over_time(daily_cap: pd.DataFrame, out_dir: Path, max_day: int) -> None:
    if daily_cap.empty:
        return
    fig, ax = plt.subplots(figsize=(15, 8))
    fig.subplots_adjust(right=0.88)
    _plot_multi_arm_timeseries(
        ax,
        daily_cap,
        max_day=max_day,
        y_fn=lambda g: g["cumulative_return_pct"].astype(float),
        title="Strategy ranking proxy — cumulative return by arm",
        ylabel="Cumulative return (%)",
    )
    fig.savefig(out_dir / "strategy_ranking_over_time.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_selection_frequency(selection: pd.DataFrame, out_dir: Path) -> None:
    for sel in SELECTOR_ARM_LABELS:
        g = selection[selection["arm_label"] == sel]
        if g.empty:
            continue
        freq = g["selected_arm_label"].value_counts(normalize=True).sort_index() * 100
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(freq.index, freq.values, color="steelblue")
        ax.set_ylabel("Selection frequency (%)")
        ax.set_title(f"Selector {sel} — strategy selection frequency")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(out_dir / f"selection_frequency_{sel}.png", dpi=120)
        plt.close(fig)


def _plot_switching_timeline(selection: pd.DataFrame, out_dir: Path, max_day: int) -> None:
    for sel in SELECTOR_ARM_LABELS:
        g = selection[selection["arm_label"] == sel].sort_values("entry_ts")
        if g.empty:
            continue
        switches = g[g["switched"] == True]  # noqa: E712
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.scatter(g["day_number"], [sel] * len(g), c=g["selected_arm_label"].astype("category").cat.codes, cmap="tab20", s=8)
        if not switches.empty:
            ax.scatter(switches["day_number"], [sel] * len(switches), marker="x", c="red", s=30, label="switch")
        ax.set_xlabel("Day")
        ax.set_title(f"Selector {sel} — selection timeline")
        ax.set_xlim(1, max_day)
        fig.tight_layout()
        fig.savefig(out_dir / f"switching_timeline_{sel}.png", dpi=120)
        plt.close(fig)


def _plot_selection_regret(regret: pd.DataFrame, out_dir: Path) -> None:
    for sel in SELECTOR_ARM_LABELS:
        g = regret[regret["arm_label"] == sel]
        if g.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(g["regret_pct"] * 100, bins=30, color="indianred", alpha=0.8)
        ax.set_xlabel("Regret (%)")
        ax.set_title(f"Selector {sel} — selection regret distribution")
        fig.tight_layout()
        fig.savefig(out_dir / f"selection_regret_{sel}.png", dpi=120)
        plt.close(fig)


def _plot_fixed_vs_selector(daily_cap: pd.DataFrame, out_dir: Path, max_day: int) -> None:
    if daily_cap.empty:
        return
    best_fixed = None
    best_fixed_ret = -1e9
    for arm in FIXED_ARM_LABELS:
        g = daily_cap[daily_cap["strategy_key"] == arm]
        if g.empty:
            continue
        ret = float(g["cumulative_return_pct"].iloc[-1])
        if ret > best_fixed_ret:
            best_fixed_ret = ret
            best_fixed = arm
    fig, ax = plt.subplots(figsize=(15, 8))
    fig.subplots_adjust(right=0.88)
    _plot_multi_arm_timeseries(
        ax,
        daily_cap,
        max_day=max_day,
        y_fn=lambda g: g["cumulative_return_pct"].astype(float),
        title=f"Fixed vs dynamic selectors; best fixed ≈ {best_fixed}",
        ylabel="Cumulative return (%)",
    )
    fig.savefig(out_dir / "fixed_vs_selector_comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
