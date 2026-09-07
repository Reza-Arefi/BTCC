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


def _annotate_bars(
    ax,
    bars,
    values,
    *,
    fmt: str = "{:.1f}",
    fontsize: int = 8,
    rotation: int = 0,
    color: str | None = None,
    adjust_ylim: bool = True,
) -> None:
    """Write numeric labels just outside each bar tip (works for +/− values)."""
    vals = list(values)
    if len(vals) != len(bars):
        return
    finite = []
    for bar, v in zip(bars, vals):
        try:
            num = float(v)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(num):
            continue
        finite.append(num)
        x = bar.get_x() + bar.get_width() / 2.0
        if num >= 0:
            y = num
            va = "bottom"
        else:
            y = num
            va = "top"
        kwargs = {
            "ha": "center",
            "va": va,
            "fontsize": fontsize,
            "rotation": rotation,
            "clip_on": False,
        }
        if color is not None:
            kwargs["color"] = color
        # Small visual offset in points so labels sit just outside the tip.
        ax.annotate(
            fmt.format(num),
            xy=(x, y),
            xytext=(0, 3 if num >= 0 else -3),
            textcoords="offset points",
            **kwargs,
        )
    if adjust_ylim and finite:
        ymax = max(finite)
        ymin = min(finite)
        span = max(abs(ymax), abs(ymin), 1.0)
        lo = ymin - span * 0.12 if ymin < 0 else min(0.0, ymin)
        hi = ymax + span * 0.18
        cur_lo, cur_hi = ax.get_ylim()
        ax.set_ylim(min(cur_lo, lo), max(cur_hi, hi))


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


def _add_endpoint_y_guides(
    ax,
    endpoints: list[tuple[str, float, float]],
    *,
    max_day: int,
    value_fmt: str = "{:.1f}",
) -> None:
    """Gray horizontal guides from y-axis to each arm's current (end) value + value labels."""
    if not endpoints:
        return
    x0 = 1.0
    # Deduplicate nearly-identical y levels so clustered arms share one guide line.
    levels: list[float] = []
    for _, _, y in sorted(endpoints, key=lambda e: e[2]):
        if not levels or abs(y - levels[-1]) > 0.15:
            levels.append(y)
        # else keep existing shared level
    for y in levels:
        ax.axhline(
            y,
            color="#9aa0a6",
            linewidth=0.8,
            linestyle="-",
            alpha=0.55,
            zorder=1,
        )
    # Per-arm value label at left (staggered if needed) so y-axis is readable.
    endpoints_sorted = sorted(endpoints, key=lambda e: e[2])
    span = max(abs(endpoints_sorted[-1][2] - endpoints_sorted[0][2]), 1e-6)
    min_gap = span * 0.022
    prev = -1e18
    for arm, _x_last, y_last in endpoints_sorted:
        y_adj = max(y_last, prev + min_gap)
        prev = y_adj
        st = _style_for_arm(arm)
        ax.text(
            x0 + max_day * 0.002,
            y_adj,
            value_fmt.format(y_last),
            fontsize=7,
            color=st["color"],
            va="center",
            ha="left",
            zorder=4,
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.75,
            },
        )


def _plot_multi_arm_timeseries(
    ax,
    daily_cap: pd.DataFrame,
    *,
    max_day: int,
    y_fn,
    title: str,
    ylabel: str,
    endpoint_y_guides: bool = False,
    value_fmt: str = "{:.1f}",
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
    ax.grid(True, alpha=0.25)
    if endpoints and endpoint_y_guides:
        _add_endpoint_y_guides(ax, endpoints, max_day=max_day, value_fmt=value_fmt)
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
    _plot_win_loss_breakdown(legs, out_dir, max_day=max_day, starting_capital_usd=starting_capital_usd)
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
        endpoint_y_guides=True,
        value_fmt="{:.1f}",
    )
    path = out_dir / "cumulative_return_all.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    # Stable copy: long-running jobs may overwrite the main file from stale in-memory code.
    fig.savefig(out_dir / "cumulative_return_all_yguides.png", dpi=120, bbox_inches="tight")
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
    bars = ax.bar(list(ALL_ARM_LABELS), wr, color=colors)
    _annotate_bars(ax, bars, wr, fmt="{:.1f}", fontsize=7, rotation=90)
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
    bars = ax.bar(list(ALL_ARM_LABELS), pfs, color="darkorange")
    _annotate_bars(ax, bars, pfs, fmt="{:.2f}", fontsize=7, rotation=90)
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
    bars = ax.bar(list(ALL_ARM_LABELS), avgs, color="seagreen")
    _annotate_bars(ax, bars, avgs, fmt="{:.3f}", fontsize=7, rotation=90)
    ax.set_ylabel("Avg P/L per trade (%)")
    ax.set_title("Average P/L per trade")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_dir / "avg_pnl_per_trade_all.png", dpi=120)
    plt.close(fig)


def _win_loss_rows(legs: pd.DataFrame, *, starting_capital_usd: float) -> pd.DataFrame:
    rows = []
    pnl = pd.to_numeric(legs.get("pnl_usd_equiv"), errors="coerce")
    pct = pd.to_numeric(legs.get("pnl_pct"), errors="coerce")
    for arm in ALL_ARM_LABELS:
        g = legs[legs["arm_key"] == arm]
        if g.empty:
            continue
        gp = pnl.loc[g.index]
        pp = pct.loc[g.index]
        wins = g[pp > 0]
        losses = g[pp < 0]
        win_usd = float(gp.loc[wins.index].sum()) if len(wins) else 0.0
        loss_usd = float(gp.loc[losses.index].sum()) if len(losses) else 0.0
        net_usd = float(gp.sum())
        rows.append(
            {
                "arm": arm,
                "n_trades": int(len(g)),
                "n_wins": int(len(wins)),
                "n_losses": int(len(losses)),
                "wins_pct": 100.0 * win_usd / starting_capital_usd,
                "losses_pct": 100.0 * loss_usd / starting_capital_usd,
                "net_pct": 100.0 * net_usd / starting_capital_usd,
            }
        )
    return pd.DataFrame(rows)


def _label_outside_bar(ax, x, y, text, *, color: str, above: bool) -> None:
    """High-contrast numeric label sitting outside a bar tip."""
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(0, 10 if above else -10),
        textcoords="offset points",
        ha="center",
        va="bottom" if above else "top",
        fontsize=11,
        fontweight="bold",
        color=color,
        clip_on=False,
        zorder=6,
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": color,
            "linewidth": 1.1,
            "alpha": 0.97,
        },
    )


def _plot_win_loss_focus(sub: pd.DataFrame, out_dir: Path, *, max_day: int) -> None:
    """Readable T1/T4/A–F diverging win/loss chart plus a values table."""
    fig = plt.figure(figsize=(14.5, 9.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.4, 1.15], hspace=0.18)
    ax = fig.add_subplot(gs[0])
    x = np.arange(len(sub))
    wins = sub["wins_pct"].to_numpy(dtype=float)
    losses = sub["losses_pct"].to_numpy(dtype=float)

    ax.bar(x, wins, width=0.62, color="#2ca02c", alpha=0.88, zorder=2, label="Sum of winning trades")
    ax.bar(x, losses, width=0.62, color="#d62728", alpha=0.88, zorder=2, label="Sum of losing trades")
    ax.axhline(0, color="#111", lw=1.1, zorder=3)

    for i, (w, loss) in enumerate(zip(wins, losses)):
        _label_outside_bar(ax, i, w, f"{w:+.1f}%", color="#145a32", above=True)
        _label_outside_bar(ax, i, loss, f"{loss:.1f}%", color="#7b241c", above=False)

    ymax = float(np.nanmax(wins)) if len(wins) else 1.0
    ymin = float(np.nanmin(losses)) if len(losses) else -1.0
    span = max(abs(ymax), abs(ymin), 1.0)
    ax.set_ylim(ymin - span * 0.32, ymax + span * 0.32)
    ax.set_xticks(x)
    ax.set_xticklabels(sub["arm"].tolist(), fontsize=12, fontweight="bold")
    ax.set_ylabel("% of starting capital ($1,000)", fontsize=11)
    ax.set_title(
        f"Profit vs loss breakdown (day {max_day}/365)\n"
        "Green = total from winners · Red = total from losers · Net is in the table below",
        fontsize=13,
        pad=10,
    )
    ax.legend(loc="upper right", framealpha=0.95, fontsize=10)
    ax.grid(True, axis="y", alpha=0.28, zorder=1)
    ax.tick_params(axis="y", labelsize=10)

    ax_t = fig.add_subplot(gs[1])
    ax_t.axis("off")
    cell_text = []
    for _, row in sub.iterrows():
        cell_text.append(
            [
                str(row["arm"]),
                f"{row['wins_pct']:+.1f}%",
                f"{row['losses_pct']:.1f}%",
                f"{row['net_pct']:+.1f}%",
                f"{int(row['n_wins'])}",
                f"{int(row['n_losses'])}",
            ]
        )
    table = ax_t.table(
        cellText=cell_text,
        colLabels=["Arm", "Winners", "Losers", "Net", "Win count", "Loss count"],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.55)
    header_color = "#1f4e79"
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#c5d0da")
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", fontweight="bold")
        elif c == 0:
            cell.set_facecolor("#f4f7fa")
            cell.set_text_props(fontweight="bold")
        elif c == 1:
            cell.set_text_props(color="#145a32", fontweight="bold")
        elif c == 2:
            cell.set_text_props(color="#7b241c", fontweight="bold")
        elif c == 3:
            cell.set_text_props(color="#1f4e79", fontweight="bold")
            cell.set_facecolor("#eaf3fb")
    ax_t.set_title("Exact values (% of $1,000 starting capital)", fontsize=11, pad=2)

    fig.savefig(out_dir / "win_loss_breakdown_focus.png", dpi=160, bbox_inches="tight")
    fig.savefig(out_dir / "win_loss_net_pct_of_capital.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_win_loss_breakdown(
    legs: pd.DataFrame,
    out_dir: Path,
    *,
    max_day: int,
    starting_capital_usd: float,
) -> None:
    df = _win_loss_rows(legs, starting_capital_usd=starting_capital_usd)
    if df.empty:
        return

    focus = ["T1", "T4", "A", "B", "C", "D", "E", "F"]
    sub = df[df["arm"].isin(focus)].copy()
    sub["ord"] = sub["arm"].map({a: i for i, a in enumerate(focus)})
    sub = sub.sort_values("ord")
    _plot_win_loss_focus(sub, out_dir, max_day=max_day)

    fig, ax = plt.subplots(figsize=(15, 6.5))
    x = np.arange(len(df))
    w = 0.28
    b1 = ax.bar(x - w, df["wins_pct"], width=w, color="#2ca02c", label="Winners total % of start")
    b2 = ax.bar(x, df["losses_pct"], width=w, color="#d62728", label="Losers total % of start")
    b3 = ax.bar(x + w, df["net_pct"], width=w, color="#1f77b4", label="Net account return %")
    _annotate_bars(ax, b1, df["wins_pct"], fmt="{:+.0f}", fontsize=6, rotation=90, color="#2ca02c", adjust_ylim=False)
    _annotate_bars(ax, b2, df["losses_pct"], fmt="{:.0f}", fontsize=6, rotation=90, color="#d62728", adjust_ylim=False)
    _annotate_bars(ax, b3, df["net_pct"], fmt="{:+.0f}", fontsize=6, rotation=90, color="#1f77b4", adjust_ylim=False)
    all_vals = list(df["wins_pct"]) + list(df["losses_pct"]) + list(df["net_pct"])
    ymax, ymin = max(all_vals), min(all_vals)
    span = max(abs(ymax), abs(ymin), 1.0)
    ax.set_ylim(ymin - span * 0.15, ymax + span * 0.22)
    ax.axhline(0, color="#222", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(df["arm"])
    ax.set_ylabel("% of starting capital")
    ax.set_title(f"All arms — Win / Loss / Net (day {max_day}/365)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "win_loss_net_pct_of_capital_all.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(df))
    b1 = ax.bar(x - 0.2, df["n_wins"], width=0.4, color="#2ca02c", label="Winners")
    b2 = ax.bar(x + 0.2, df["n_losses"], width=0.4, color="#d62728", label="Losers")
    _annotate_bars(ax, b1, df["n_wins"], fmt="{:.0f}", fontsize=7, rotation=90, adjust_ylim=False)
    _annotate_bars(ax, b2, df["n_losses"], fmt="{:.0f}", fontsize=7, rotation=90, adjust_ylim=False)
    ymax = max(float(df["n_wins"].max()), float(df["n_losses"].max()))
    ax.set_ylim(0, ymax * 1.22)
    ax.set_xticks(x)
    ax.set_xticklabels(df["arm"])
    ax.set_ylabel("Number of trades")
    n_tr = int(df["n_trades"].iloc[0]) if len(df) else 0
    ax.set_title(f"Winners vs losers count — {n_tr} trades/arm, day {max_day}/365")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "win_loss_counts.png", dpi=140)
    plt.close(fig)


def _plot_trade_count(legs: pd.DataFrame, out_dir: Path, max_day: int) -> None:
    if "entry_day_number" not in legs.columns:
        legs = legs.copy()
        legs["entry_day_number"] = legs.get("day_number")
    # All arms see the same opportunity set — use unique opportunities/day (or T1 as proxy).
    if "opportunity_id" in legs.columns:
        daily = (
            legs.dropna(subset=["entry_day_number"])
            .assign(entry_day_number=lambda d: d["entry_day_number"].astype(int))
            .loc[lambda d: (d["entry_day_number"] >= 1) & (d["entry_day_number"] <= max_day)]
            .groupby("entry_day_number")["opportunity_id"]
            .nunique()
            .rename("count")
        )
    else:
        t1 = legs[legs["arm_key"] == "T1"] if "arm_key" in legs.columns else legs
        daily = (
            t1.dropna(subset=["entry_day_number"])
            .assign(entry_day_number=lambda d: d["entry_day_number"].astype(int))
            .loc[lambda d: (d["entry_day_number"] >= 1) & (d["entry_day_number"] <= max_day)]
            .groupby("entry_day_number")
            .size()
            .rename("count")
        )
    if daily.empty:
        return
    days = pd.RangeIndex(1, int(max_day) + 1, name="entry_day_number")
    series = daily.reindex(days, fill_value=0).astype(float)
    ma7 = series.rolling(7, min_periods=1).mean()
    ma30 = series.rolling(30, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(series.index, series.values, color="#9aa0a6", alpha=0.55, linewidth=1.0, label="Daily count")
    ax.plot(ma7.index, ma7.values, color="#1f77b4", linewidth=2.0, label="7-day MA")
    ax.plot(ma30.index, ma30.values, color="#d62728", linewidth=2.2, label="30-day MA")
    ax.set_xlabel("Day")
    ax.set_ylabel("Trades entered")
    ax.set_title("Trade count per day (unique opportunities) + moving averages")
    ax.set_xlim(1, max_day)
    ax.grid(True, alpha=0.3)
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
        bars = ax.bar(freq.index.astype(str), freq.values, color="steelblue")
        _annotate_bars(ax, bars, freq.values, fmt="{:.1f}")
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
