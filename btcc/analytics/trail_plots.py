"""Plot generation for trailing-exit experiment."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from btcc.sim.trail_config import TRAIL_STRATEGY_KEYS


def generate_trail_plots(
    legs: pd.DataFrame,
    *,
    daily_cap: pd.DataFrame,
    trade_cap: pd.DataFrame,
    out_dir: Path,
    max_day: int,
    benchmark_key: str = "trail_5",
    starting_capital_usd: float = 1000.0,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if legs.empty:
        return

    _plot_cumulative_return(daily_cap, out_dir, max_day, benchmark_key)
    _plot_drawdown(daily_cap, out_dir, max_day, benchmark_key)
    _plot_win_rate(legs, out_dir, max_day)
    _plot_profit_factor(legs, out_dir)
    _plot_avg_pnl(legs, out_dir)
    _plot_trade_count(legs, out_dir, max_day)
    _plot_exit_distribution(legs, out_dir)
    _plot_trail_activation(legs, out_dir)
    _plot_mfe_mae(legs, out_dir)
    _plot_regime_heatmaps(legs, out_dir, starting_capital_usd)
    _plot_ranking_over_time(daily_cap, out_dir, max_day)


def _plot_cumulative_return(daily_cap: pd.DataFrame, out_dir: Path, max_day: int, benchmark: str) -> None:
    if daily_cap.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    for sk in TRAIL_STRATEGY_KEYS:
        g = daily_cap[daily_cap["strategy_key"] == sk]
        if g.empty:
            continue
        lw = 2.5 if sk == benchmark else 1.0
        ax.plot(g["day_number"], g["cumulative_return_pct"], label=sk, linewidth=lw)
    ax.set_xlabel("Day")
    ax.set_ylabel("Cumulative return (%)")
    ax.set_title("Cumulative return — T1–T10 (T3 benchmark highlighted)")
    ax.set_xlim(1, max_day)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "cumulative_return_all.png", dpi=120)
    plt.close(fig)


def _plot_drawdown(daily_cap: pd.DataFrame, out_dir: Path, max_day: int, benchmark: str) -> None:
    if daily_cap.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    for sk in TRAIL_STRATEGY_KEYS:
        g = daily_cap[daily_cap["strategy_key"] == sk].sort_values("day_number")
        if g.empty:
            continue
        eq = g["ending_value"].astype(float)
        peak = eq.cummax()
        dd = 100 * (eq / peak - 1.0)
        lw = 2.5 if sk == benchmark else 1.0
        ax.plot(g["day_number"], dd, label=sk, linewidth=lw)
    ax.set_xlabel("Day")
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Drawdown (%)")
    ax.set_xlim(1, max_day)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "drawdown_all.png", dpi=120)
    plt.close(fig)


def _plot_win_rate(legs: pd.DataFrame, out_dir: Path, max_day: int) -> None:
    closed = legs[legs.get("closed", True) == True]  # noqa: E712
    if closed.empty:
        return
    wr = []
    for sk in TRAIL_STRATEGY_KEYS:
        g = closed[closed["strategy_key"] == sk]
        wr.append(100 * (pd.to_numeric(g["pnl_pct"], errors="coerce") > 0).mean() if len(g) else 0)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(list(TRAIL_STRATEGY_KEYS), wr, color="steelblue")
    ax.set_ylabel("Win rate (%)")
    ax.set_title("Win rate by strategy")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(out_dir / "win_rate_all.png", dpi=120)
    plt.close(fig)


def _plot_profit_factor(legs: pd.DataFrame, out_dir: Path) -> None:
    pfs = []
    pnl = pd.to_numeric(legs.get("pnl_usd_equiv"), errors="coerce").fillna(0)
    for sk in TRAIL_STRATEGY_KEYS:
        g = pnl[legs["strategy_key"] == sk]
        gp = g[g > 0].sum()
        gl = abs(g[g < 0].sum())
        pfs.append(gp / gl if gl > 1e-9 else 0)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(list(TRAIL_STRATEGY_KEYS), pfs, color="darkorange")
    ax.set_ylabel("Profit factor")
    ax.set_title("Profit factor by strategy")
    fig.tight_layout()
    fig.savefig(out_dir / "profit_factor_all.png", dpi=120)
    plt.close(fig)


def _plot_avg_pnl(legs: pd.DataFrame, out_dir: Path) -> None:
    avgs = []
    meds = []
    for sk in TRAIL_STRATEGY_KEYS:
        g = legs[legs["strategy_key"] == sk]
        s = 100 * pd.to_numeric(g.get("pnl_pct"), errors="coerce")
        avgs.append(float(s.mean()) if len(g) else 0)
        meds.append(float(s.median()) if len(g) else 0)
    labels = list(TRAIL_STRATEGY_KEYS)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(labels, avgs, color="seagreen")
    ax.set_ylabel("Avg P/L per trade (%)")
    ax.set_title("Average P/L per trade")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_dir / "avg_pnl_per_trade_all.png", dpi=120)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(labels, meds, color="teal")
    ax.set_ylabel("Median P/L per trade (%)")
    ax.set_title("Median P/L per trade")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_dir / "median_pnl_per_trade_all.png", dpi=120)
    plt.close(fig)


def _plot_trade_count(legs: pd.DataFrame, out_dir: Path, max_day: int) -> None:
    if "day_number" not in legs.columns:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    for sk in TRAIL_STRATEGY_KEYS:
        g = legs[(legs["strategy_key"] == sk) & (legs["day_number"] <= max_day)]
        if g.empty:
            continue
        c = g.groupby("day_number").size()
        ax.plot(c.index, c.values.cumsum(), label=sk, alpha=0.8)
    ax.set_xlabel("Day")
    ax.set_ylabel("Cumulative trades")
    ax.set_title("Number of trades over time")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "trade_count_over_time.png", dpi=120)
    plt.close(fig)


def _plot_exit_distribution(legs: pd.DataFrame, out_dir: Path) -> None:
    if "exit_reason" not in legs.columns:
        return
    n = len(TRAIL_STRATEGY_KEYS)
    ncols = 5
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.2 * nrows))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for i, sk in enumerate(TRAIL_STRATEGY_KEYS):
        ax = axes_flat[i]
        g = legs[legs["strategy_key"] == sk]["exit_reason"].value_counts()
        if len(g) == 0:
            ax.set_title(sk, fontsize=9)
            ax.axis("off")
            continue
        ax.pie(g.values, labels=g.index, autopct="%1.0f%%", textprops={"fontsize": 6})
        ax.set_title(sk, fontsize=9)
    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")
    fig.suptitle("Exit distribution (SL vs trailing)")
    fig.tight_layout()
    fig.savefig(out_dir / "exit_distribution.png", dpi=120)
    plt.close(fig)


def _plot_trail_activation(legs: pd.DataFrame, out_dir: Path) -> None:
    if "trail_activated" not in legs.columns:
        return
    rates = []
    for sk in TRAIL_STRATEGY_KEYS:
        g = legs[legs["strategy_key"] == sk]
        rates.append(100 * g["trail_activated"].astype(bool).mean() if len(g) else 0)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(list(TRAIL_STRATEGY_KEYS), rates, color="purple", alpha=0.7)
    ax.set_ylabel("Trail activation rate (%)")
    ax.set_title("Trail activation rate")
    fig.tight_layout()
    fig.savefig(out_dir / "trail_activation_rate.png", dpi=120)
    plt.close(fig)


def _plot_mfe_mae(legs: pd.DataFrame, out_dir: Path) -> None:
    if "mfe_pct" not in legs.columns or "mae_pct" not in legs.columns:
        return
    mfe = 100 * pd.to_numeric(legs["mfe_pct"], errors="coerce")
    mae = 100 * pd.to_numeric(legs["mae_pct"], errors="coerce")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(mfe.dropna(), bins=40, color="green", alpha=0.7)
    axes[0].set_title("MFE distribution (%)")
    axes[1].hist(mae.dropna(), bins=40, color="red", alpha=0.7)
    axes[1].set_title("MAE distribution (%)")
    axes[2].scatter(mae, mfe, s=4, alpha=0.3)
    axes[2].set_xlabel("MAE (%)")
    axes[2].set_ylabel("MFE (%)")
    axes[2].set_title("MFE vs MAE")
    fig.tight_layout()
    fig.savefig(out_dir / "mfe_mae_diagnostics.png", dpi=120)
    plt.close(fig)
    if "realized_fraction_of_mfe" in legs.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        rf = pd.to_numeric(legs["realized_fraction_of_mfe"], errors="coerce").dropna()
        ax.hist(rf, bins=40, color="teal", alpha=0.7)
        ax.set_title("Realized fraction of MFE")
        fig.tight_layout()
        fig.savefig(out_dir / "avg_pnl_vs_mfe.png", dpi=120)
        plt.close(fig)


def _plot_regime_heatmaps(legs: pd.DataFrame, out_dir: Path, capital: float) -> None:
    if "regime" not in legs.columns:
        return
    regimes = sorted(legs["regime"].dropna().unique())
    if not regimes:
        return
    pnl = pd.to_numeric(legs.get("pnl_usd_equiv"), errors="coerce").fillna(0)
    legs = legs.copy()
    legs["_pnl"] = pnl
    legs["_win"] = pd.to_numeric(legs["pnl_pct"], errors="coerce") > 0

    def _pf(g: pd.DataFrame) -> float:
        gp = g.loc[g["_pnl"] > 0, "_pnl"].sum()
        gl = abs(g.loc[g["_pnl"] < 0, "_pnl"].sum())
        return float(gp / gl) if gl > 1e-9 else float("nan")

    def _matrix(col: str) -> np.ndarray:
        mat = np.zeros((len(TRAIL_STRATEGY_KEYS), len(regimes)))
        for i, sk in enumerate(TRAIL_STRATEGY_KEYS):
            for j, reg in enumerate(regimes):
                g = legs[(legs["strategy_key"] == sk) & (legs["regime"] == reg)]
                if col == "ret":
                    mat[i, j] = 100 * g["_pnl"].sum() / capital if len(g) else np.nan
                elif col == "wr":
                    mat[i, j] = 100 * g["_win"].mean() if len(g) else np.nan
                elif col == "pf":
                    mat[i, j] = _pf(g) if len(g) else np.nan
                elif col == "avg":
                    mat[i, j] = 100 * pd.to_numeric(g["pnl_pct"], errors="coerce").mean() if len(g) else np.nan
                else:
                    mat[i, j] = len(g)
        return mat

    for name, col, title in (
        ("strategy_regime_return_heatmap", "ret", "Return (%)"),
        ("strategy_regime_winrate_heatmap", "wr", "Win rate (%)"),
        ("strategy_regime_pf_heatmap", "pf", "Profit factor"),
        ("strategy_regime_avg_pnl_heatmap", "avg", "Avg P/L (%)"),
        ("strategy_regime_count_heatmap", "n", "Trade count"),
    ):
        mat = _matrix(col)
        fig, ax = plt.subplots(figsize=(9, 11))
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn" if col != "n" else "Blues")
        ax.set_xticks(range(len(regimes)))
        ax.set_xticklabels(regimes, rotation=45, ha="right")
        ax.set_yticks(range(len(TRAIL_STRATEGY_KEYS)))
        ax.set_yticklabels(TRAIL_STRATEGY_KEYS)
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}.png", dpi=120)
        plt.close(fig)

    # CSV table: strategy × regime return (%) + overall
    rows = []
    for sk in TRAIL_STRATEGY_KEYS:
        row = {"strategy": sk}
        for reg in regimes:
            g = legs[(legs["strategy_key"] == sk) & (legs["regime"] == reg)]
            row[reg] = round(100 * g["_pnl"].sum() / capital, 2) if len(g) else None
        g_all = legs[legs["strategy_key"] == sk]
        row["Overall"] = round(100 * g_all["_pnl"].sum() / capital, 2) if len(g_all) else None
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "regime_performance_table.csv", index=False)


def _plot_ranking_over_time(daily_cap: pd.DataFrame, out_dir: Path, max_day: int) -> None:
    if daily_cap.empty:
        return
    days = sorted(daily_cap["day_number"].dropna().unique())
    if not days:
        return
    ranks = {sk: [] for sk in TRAIL_STRATEGY_KEYS}
    for d in days:
        snap = daily_cap[daily_cap["day_number"] == d]
        rets = {sk: snap[snap["strategy_key"] == sk]["cumulative_return_pct"].max() for sk in TRAIL_STRATEGY_KEYS}
        ordered = sorted(rets.keys(), key=lambda k: rets[k] if rets[k] is not None and not pd.isna(rets[k]) else -1e9, reverse=True)
        for rank, sk in enumerate(ordered, 1):
            ranks[sk].append(rank)
    fig, ax = plt.subplots(figsize=(12, 6))
    for sk in TRAIL_STRATEGY_KEYS:
        if ranks[sk]:
            ax.plot(days[: len(ranks[sk])], ranks[sk], label=sk, alpha=0.8)
    ax.set_xlabel("Day")
    ax.set_ylabel("Rank (1=best)")
    ax.set_title("Strategy ranking over time")
    ax.invert_yaxis()
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "strategy_ranking_over_time.png", dpi=120)
    plt.close(fig)
