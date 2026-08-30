"""Plot generators — always paired with saved metric CSVs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from btcc.sim.score import FACTOR_KEYS

ARM_COLORS = {"static": "#1f77b4", "equal": "#2ca02c", "adaptive": "#d62728"}


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _mark_init_day(ax, init_days: int | None) -> None:
    if init_days is None:
        return
    ax.axvline(int(init_days), color="#666666", ls="--", lw=1.2, alpha=0.8, label=f"Day {int(init_days)} init→daily")


def _series_x(df: pd.DataFrame, ts_col: str) -> tuple[Any, str]:
    """Prefer day_number x-axis; fall back to timestamp."""
    if df is not None and not df.empty and "day_number" in df.columns and df["day_number"].notna().any():
        return pd.to_numeric(df["day_number"], errors="coerce"), "Day"
    if df is not None and not df.empty and ts_col in df.columns:
        return pd.to_datetime(df[ts_col], utc=True, errors="coerce"), "Time"
    return pd.Series(dtype=float), "Day"


def plot_btc_accumulation_by_strategy(
    cum_by_arm: dict[str, pd.DataFrame],
    plots_dir: Path,
    *,
    init_days: int | None = None,
) -> list[Path]:
    paths = []
    strategies = sorted({
        sk for df in cum_by_arm.values() if df is not None and not df.empty
        for sk in df.get("strategy_key", pd.Series(dtype=str)).unique()
    })
    for sk in strategies:
        fig, ax = plt.subplots(figsize=(10, 5))
        for arm, df in cum_by_arm.items():
            if df is None or df.empty:
                continue
            g = df[df["strategy_key"] == sk].copy()
            if g.empty:
                continue
            if "day_number" in g.columns:
                g = g.sort_values("day_number")
                x, xlabel = g["day_number"], "Day"
            else:
                g = g.sort_values("exit_ts")
                x, xlabel = pd.to_datetime(g["exit_ts"]), "Time"
            ax.plot(x, g["cum_btc"], label=arm, color=ARM_COLORS.get(arm))
        _mark_init_day(ax, init_days)
        ax.set_title(f"BTC accumulation — {sk}")
        ax.set_xlabel(xlabel if 'xlabel' in dir() else "Day")
        # xlabel from last series; force Day when any arm has day_number
        if any(
            df is not None and not df.empty and "day_number" in df.columns
            for df in cum_by_arm.values()
        ):
            ax.set_xlabel("Day")
        ax.set_ylabel("Cumulative BTC PnL")
        ax.legend()
        ax.grid(True, alpha=0.3)
        p = plots_dir / f"btc_accumulation_{sk}.png"
        _save(fig, p)
        paths.append(p)
    return paths


def plot_rolling_win_rate(
    wr_by_arm: dict[str, pd.DataFrame],
    plots_dir: Path,
    *,
    init_days: int | None = None,
) -> list[Path]:
    paths = []
    strategies = sorted({
        sk for df in wr_by_arm.values() if df is not None and not df.empty
        for sk in df["strategy_key"].unique()
    })
    for sk in strategies:
        fig, ax = plt.subplots(figsize=(10, 5))
        for arm, df in wr_by_arm.items():
            if df is None or df.empty:
                continue
            g = df[df["strategy_key"] == sk].dropna(subset=["rolling_win_rate"]).copy()
            if g.empty:
                continue
            if "day_number" in g.columns:
                g = g.sort_values("day_number")
                ax.plot(g["day_number"], g["rolling_win_rate"], label=arm, color=ARM_COLORS.get(arm))
            else:
                ax.plot(pd.to_datetime(g["exit_ts"]), g["rolling_win_rate"], label=arm, color=ARM_COLORS.get(arm))
        _mark_init_day(ax, init_days)
        ax.set_title(f"Rolling 30d win rate — {sk}")
        ax.set_xlabel("Day" if any(
            d is not None and not d.empty and "day_number" in d.columns for d in wr_by_arm.values()
        ) else "Time")
        ax.set_ylabel("Win rate")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        p = plots_dir / f"rolling_win_rate_{sk}.png"
        _save(fig, p)
        paths.append(p)
    return paths


def plot_adaptive_weights(
    wh: pd.DataFrame,
    plots_dir: Path,
    *,
    init_days: int | None = None,
) -> Path | None:
    if wh.empty or "indicator" not in wh.columns:
        return None
    fig, ax = plt.subplots(figsize=(11, 6))
    for ind in FACTOR_KEYS:
        g = wh[wh["indicator"] == ind].copy()
        if g.empty:
            continue
        if "day_number" in g.columns and g["day_number"].notna().any():
            g = g.sort_values("day_number")
            ax.plot(g["day_number"], g["new_weight"], label=ind, marker="o", markersize=3)
        else:
            g = g.sort_values("update_timestamp")
            ax.plot(pd.to_datetime(g["update_timestamp"]), g["new_weight"], label=ind, marker="o", markersize=3)
    _mark_init_day(ax, init_days)
    ax.set_title("Adaptive indicator weights through time")
    ax.set_xlabel("Day" if "day_number" in wh.columns else "Time")
    ax.set_ylabel("Weight")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    p = plots_dir / "adaptive_weights.png"
    _save(fig, p)
    return p


def plot_indicator_correlations(corr_ts: pd.DataFrame, plots_dir: Path) -> Path | None:
    if corr_ts.empty:
        return None
    df = corr_ts.copy()
    # Full-sample rows use asof="end_of_sample" — plot as bar instead if no time axis
    if df["asof"].astype(str).eq("end_of_sample").all():
        fig, ax = plt.subplots(figsize=(9, 4))
        g = df.dropna(subset=["correlation"])
        ax.bar(g["indicator"].astype(str), g["correlation"])
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title("Predictive correlation (matured, full sample)")
        plt.xticks(rotation=30, ha="right")
        p = plots_dir / "indicator_predictive_correlation.png"
        _save(fig, p)
        return p
    fig, ax = plt.subplots(figsize=(11, 6))
    for ind in FACTOR_KEYS:
        g = df[df["indicator"] == ind].dropna(subset=["correlation"])
        g = g[g["asof"].astype(str) != "end_of_sample"]
        if g.empty:
            continue
        ax.plot(pd.to_datetime(g["asof"], utc=True), g["correlation"], label=ind)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Predictive correlation (matured only): indicator vs future_return_4h")
    ax.set_ylabel("corr")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    p = plots_dir / "indicator_predictive_correlation.png"
    _save(fig, p)
    return p


def plot_weight_vs_usefulness(df: pd.DataFrame, plots_dir: Path) -> Path | None:
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 6))
    for ind in FACTOR_KEYS:
        g = df[df["indicator"] == ind].dropna(subset=["weight", "correlation"])
        if g.empty:
            continue
        ax.scatter(g["correlation"], g["weight"], label=ind, alpha=0.7, s=20)
    ax.set_xlabel("Predictive correlation")
    ax.set_ylabel("Adaptive weight")
    ax.set_title("Weight vs predictive usefulness")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    p = plots_dir / "weight_vs_usefulness.png"
    _save(fig, p)
    return p


def plot_score_vs_return(df: pd.DataFrame, plots_dir: Path) -> list[Path]:
    paths = []
    if df.empty:
        return paths
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["S"], df["future_return_4h"], alpha=0.15, s=8)
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("S")
    ax.set_ylabel("future_return_4h")
    ax.set_title("Prediction score S vs realized 4h return (matured)")
    p = plots_dir / "score_vs_return_scatter.png"
    _save(fig, p)
    paths.append(p)
    # Binned mean return
    try:
        df = df.copy()
        df["bin"] = pd.cut(df["S"], bins=10)
        g = df.groupby("bin", observed=False)["future_return_4h"].mean()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(len(g)), g.values)
        ax.set_xticks(range(len(g)))
        ax.set_xticklabels([str(x) for x in g.index], rotation=45, ha="right", fontsize=7)
        ax.set_title("Mean 4h return by S bin")
        ax.axhline(0, color="black", lw=0.8)
        p2 = plots_dir / "score_vs_return_binned.png"
        _save(fig, p2)
        paths.append(p2)
    except Exception:
        pass
    return paths


def plot_accuracy_buckets(df: pd.DataFrame, plots_dir: Path) -> Path | None:
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(df["bucket"].astype(str), df["accuracy"].fillna(0))
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Prediction accuracy by score bucket (score_01=(S+1)/2)")
    plt.xticks(rotation=30, ha="right")
    p = plots_dir / "accuracy_by_score_bucket.png"
    _save(fig, p)
    return p


def plot_drawdown(
    dd_by_arm: dict[str, pd.DataFrame],
    plots_dir: Path,
    *,
    init_days: int | None = None,
) -> list[Path]:
    paths = []
    strategies = sorted({
        sk for df in dd_by_arm.values() if df is not None and not df.empty
        for sk in df["strategy_key"].unique()
    })
    use_day = any(
        d is not None and not d.empty and "day_number" in d.columns for d in dd_by_arm.values()
    )
    for sk in strategies:
        fig, ax = plt.subplots(figsize=(10, 4))
        for arm, df in dd_by_arm.items():
            if df is None or df.empty:
                continue
            g = df[df["strategy_key"] == sk].copy()
            if g.empty:
                continue
            if use_day and "day_number" in g.columns:
                g = g.sort_values("day_number")
                ax.plot(g["day_number"], g["drawdown_btc"], label=arm, color=ARM_COLORS.get(arm))
            else:
                ax.plot(pd.to_datetime(g["exit_ts"]), g["drawdown_btc"], label=arm, color=ARM_COLORS.get(arm))
        _mark_init_day(ax, init_days)
        ax.set_title(f"BTC drawdown — {sk}")
        ax.set_xlabel("Day" if use_day else "Time")
        ax.set_ylabel("Drawdown (BTC)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        p = plots_dir / f"drawdown_{sk}.png"
        _save(fig, p)
        paths.append(p)
    return paths


def plot_simultaneous(
    sim_df: pd.DataFrame,
    plots_dir: Path,
    max_open: int = 10,
    *,
    init_days: int | None = None,
) -> Path | None:
    if sim_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 4))
    if "day_number" in sim_df.columns and sim_df["day_number"].notna().any():
        g = sim_df.sort_values("day_number")
        ax.plot(g["day_number"], g["n_open"], color="#333")
        ax.set_xlabel("Day")
    else:
        ax.plot(pd.to_datetime(sim_df["timestamp"]), sim_df["n_open"], color="#333")
        ax.set_xlabel("Time")
    ax.axhline(max_open, color="red", ls="--", label=f"max_open={max_open}")
    _mark_init_day(ax, init_days)
    ax.set_title("Simultaneous open opportunities")
    ax.set_ylabel("n_open")
    ax.legend()
    ax.grid(True, alpha=0.3)
    p = plots_dir / "simultaneous_opportunities.png"
    _save(fig, p)
    return p


def plot_rejections(df: pd.DataFrame, plots_dir: Path) -> Path | None:
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(df["rejection_reason"].astype(str), df["count"])
    ax.set_title("Rejected signals by reason")
    plt.xticks(rotation=30, ha="right", fontsize=8)
    p = plots_dir / "rejected_signals.png"
    _save(fig, p)
    return p


def plot_learning_progress(df: pd.DataFrame, plots_dir: Path) -> Path | None:
    if df.empty:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    x = df["update_number"]
    axes[0, 0].plot(x, df["prediction_accuracy"], marker="o")
    axes[0, 0].set_title("Prediction accuracy")
    axes[0, 1].plot(x, df["win_rate"], marker="o", color="green")
    axes[0, 1].set_title("Win rate")
    axes[1, 0].plot(x, df["mean_future_return_4h"], marker="o", color="orange")
    axes[1, 0].set_title("Mean 4h return")
    axes[1, 1].plot(x, df["mean_trade_pnl_btc"], marker="o", color="purple")
    axes[1, 1].set_title("Mean trade PnL (BTC)")
    for ax in axes.ravel():
        ax.grid(True, alpha=0.3)
    fig.suptitle("Adaptive learning progress vs update number")
    p = plots_dir / "adaptive_learning_progress.png"
    _save(fig, p)
    return p


def plot_adaptive_advantage(adv: pd.DataFrame, plots_dir: Path, strategy_key: str) -> Path | None:
    if adv.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 4))
    ts = pd.to_datetime(adv["exit_ts"])
    if "vs_static" in adv.columns:
        ax.plot(ts, adv["vs_static"], label="Adaptive − Static")
    if "vs_equal" in adv.columns:
        ax.plot(ts, adv["vs_equal"], label="Adaptive − Equal")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(f"Adaptive advantage — {strategy_key}")
    ax.set_ylabel("Δ cumulative BTC")
    ax.legend()
    ax.grid(True, alpha=0.3)
    p = plots_dir / f"adaptive_advantage_{strategy_key}.png"
    _save(fig, p)
    return p


def plot_monthly(df: pd.DataFrame, plots_dir: Path, arm: str) -> Path | None:
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(df["month"].astype(str), df["btc_pnl"].fillna(0))
    ax.set_title(f"Monthly BTC PnL — {arm}")
    plt.xticks(rotation=45, ha="right")
    ax.axhline(0, color="black", lw=0.8)
    p = plots_dir / f"monthly_pnl_{arm}.png"
    _save(fig, p)
    return p


def plot_btcd_regimes(df: pd.DataFrame, plots_dir: Path) -> Path | None:
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df["regime"].astype(str), df["prediction_accuracy"].fillna(0))
    ax.set_ylim(0, 1)
    ax.set_title("Prediction accuracy by relative BTC.D regime")
    p = plots_dir / "btcd_regime_accuracy.png"
    _save(fig, p)
    return p


def plot_entry_policy_btc(
    legs: pd.DataFrame,
    plots_dir: Path,
) -> list[Path]:
    """BTC accumulation: NORMAL_FILTERED vs LATE_ENTRY_ALLOWED per exit strategy."""
    from btcc.analytics.metrics import cumulative_btc

    paths = []
    if legs.empty or "entry_policy" not in legs.columns:
        return paths
    for sk in sorted(legs["strategy_key"].unique()):
        fig, ax = plt.subplots(figsize=(10, 5))
        for pol, color in (("NORMAL_FILTERED", "#1f77b4"), ("LATE_ENTRY_ALLOWED", "#d62728")):
            sub = legs[(legs["strategy_key"] == sk) & (legs["entry_policy"] == pol)]
            cum = cumulative_btc(sub)
            if cum.empty:
                continue
            ax.plot(pd.to_datetime(cum["exit_ts"]), cum["cum_btc"], label=pol, color=color)
        ax.set_title(f"Entry policy BTC accumulation — {sk}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        p = plots_dir / f"entry_policy_btc_{sk}.png"
        _save(fig, p)
        paths.append(p)
    return paths


def plot_recovered_trades(legs: pd.DataFrame, plots_dir: Path) -> list[Path]:
    paths = []
    if legs.empty or "recovered_by_late_allowed" not in legs.columns:
        return paths
    rec = legs[legs["recovered_by_late_allowed"] == True]  # noqa: E712
    if rec.empty:
        return paths
    from btcc.analytics.metrics import cumulative_btc
    import numpy as np

    cum = cumulative_btc(rec)
    if not cum.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        for sk, g in cum.groupby("strategy_key"):
            ax.plot(pd.to_datetime(g["exit_ts"]), g["cum_btc"], label=sk)
        ax.set_title("Cumulative BTC — recovered late-allowed trades only")
        ax.legend()
        ax.grid(True, alpha=0.3)
        p = plots_dir / "recovered_trades_cum_btc.png"
        _save(fig, p)
        paths.append(p)

    # Recovered trade count over time
    r = rec.copy()
    r["_ts"] = pd.to_datetime(r.get("exit_ts", r.get("entry_ts")), utc=True, errors="coerce")
    r = r.dropna(subset=["_ts"]).sort_values("_ts")
    if not r.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        for sk, g in r.groupby("strategy_key"):
            ax.plot(g["_ts"], np.arange(1, len(g) + 1), label=sk)
        ax.set_title("Recovered late-allowed trades over time (count)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        p = plots_dir / "recovered_trades_count_over_time.png"
        _save(fig, p)
        paths.append(p)

    # Winning vs losing recovered (by strategy)
    fig, ax = plt.subplots(figsize=(8, 4))
    strategies = sorted(rec["strategy_key"].unique())
    wins, losses = [], []
    for sk in strategies:
        g = rec[rec["strategy_key"] == sk]
        pnl = pd.to_numeric(g.get("pnl_btc"), errors="coerce")
        wins.append(int((pnl > 0).sum()))
        losses.append(int((pnl <= 0).sum()))
    x = np.arange(len(strategies))
    ax.bar(x - 0.2, wins, 0.4, label="wins", color="#2ca02c")
    ax.bar(x + 0.2, losses, 0.4, label="losses", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.set_title("Recovered trades — wins vs losses")
    ax.legend()
    p = plots_dir / "recovered_trades_win_loss.png"
    _save(fig, p)
    paths.append(p)

    # Monthly BTC of recovered trades
    r2 = rec.copy()
    r2["_ts"] = pd.to_datetime(r2.get("exit_ts"), utc=True, errors="coerce")
    r2["pnl_btc"] = pd.to_numeric(r2.get("pnl_btc"), errors="coerce")
    r2 = r2.dropna(subset=["_ts", "pnl_btc"])
    if not r2.empty:
        r2["month"] = r2["_ts"].dt.tz_localize(None).dt.to_period("M").astype(str)
        pivot = r2.groupby(["month", "strategy_key"])["pnl_btc"].sum().unstack(fill_value=0.0)
        fig, ax = plt.subplots(figsize=(10, 4))
        pivot.plot(kind="bar", ax=ax)
        ax.set_title("Monthly BTC PnL — recovered late-allowed trades")
        ax.axhline(0, color="black", lw=0.8)
        ax.legend(title="strategy")
        p = plots_dir / "recovered_trades_monthly_btc.png"
        _save(fig, p)
        paths.append(p)
    return paths


def plot_entry_policy_win_rate(legs: pd.DataFrame, plots_dir: Path) -> Path | None:
    """Win rate: NORMAL_FILTERED vs LATE_ENTRY_ALLOWED per exit strategy."""
    if legs.empty or "entry_policy" not in legs.columns:
        return None
    import numpy as np

    rows = []
    for pol in ("NORMAL_FILTERED", "LATE_ENTRY_ALLOWED"):
        for sk, g in legs[legs["entry_policy"] == pol].groupby("strategy_key"):
            pnl = pd.to_numeric(g.get("pnl_btc"), errors="coerce").dropna()
            if len(pnl) == 0:
                continue
            rows.append({"policy": pol, "strategy": sk, "win_rate": float((pnl > 0).mean()), "n": len(pnl)})
    if not rows:
        return None
    t = pd.DataFrame(rows)
    strategies = sorted(t["strategy"].unique())
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(strategies))
    width = 0.35
    for i, (pol, color) in enumerate((("NORMAL_FILTERED", "#1f77b4"), ("LATE_ENTRY_ALLOWED", "#d62728"))):
        vals = [
            float(t[(t.policy == pol) & (t.strategy == sk)]["win_rate"].iloc[0])
            if len(t[(t.policy == pol) & (t.strategy == sk)]) else 0.0
            for sk in strategies
        ]
        ax.bar(x + i * width, vals, width, label=pol, color=color)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(strategies)
    ax.set_ylim(0, 1)
    ax.set_title("Win rate — NORMAL_FILTERED vs LATE_ENTRY_ALLOWED")
    ax.legend()
    p = plots_dir / "entry_policy_win_rate.png"
    _save(fig, p)
    return p


def plot_entry_classification_scores(pred: pd.DataFrame, plots_dir: Path) -> Path | None:
    """S distribution for normal / late-accepted / late-rejected classifications."""
    if pred.empty or "entry_classification" not in pred.columns or "S" not in pred.columns:
        return None
    classes = {
        "NORMAL_ENTRY": "#1f77b4",
        "LATE_ENTRY_ACCEPTED": "#d62728",
        "LATE_ENTRY_REJECTED": "#ff7f0e",
    }
    fig, ax = plt.subplots(figsize=(9, 4))
    plotted = False
    for cls, color in classes.items():
        s = pd.to_numeric(pred.loc[pred["entry_classification"] == cls, "S"], errors="coerce").dropna()
        if s.empty:
            continue
        ax.hist(s, bins=30, alpha=0.45, label=f"{cls} (n={len(s)})", color=color, density=True)
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.axvline(0.60, color="black", ls="--", lw=1, label="threshold 0.60")
    ax.set_title("Prediction score S by entry classification")
    ax.set_xlabel("S")
    ax.legend()
    p = plots_dir / "entry_classification_score_dist.png"
    _save(fig, p)
    return p


def plot_entry_future_returns(pred: pd.DataFrame, plots_dir: Path) -> Path | None:
    """Future 4h return distribution: normal entries vs recovered late entries."""
    if pred.empty or "future_return_4h" not in pred.columns or "entry_classification" not in pred.columns:
        return None
    fig, ax = plt.subplots(figsize=(9, 4))
    plotted = False
    for cls, color, label in (
        ("NORMAL_ENTRY", "#1f77b4", "normal entries"),
        ("LATE_ENTRY_ACCEPTED", "#d62728", "recovered late entries"),
    ):
        y = pd.to_numeric(
            pred.loc[pred["entry_classification"] == cls, "future_return_4h"],
            errors="coerce",
        ).dropna()
        if y.empty:
            continue
        ax.hist(y, bins=40, alpha=0.45, label=f"{label} (n={len(y)})", color=color, density=True)
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Future 4h return — normal vs recovered late entries")
    ax.set_xlabel("future_return_4h")
    ax.legend()
    p = plots_dir / "entry_future_return_dist.png"
    _save(fig, p)
    return p


def plot_win_loss_bars(wl_by_arm: dict[str, pd.DataFrame], plots_dir: Path) -> Path | None:
    # Stack net PnL by arm × strategy
    rows = []
    for arm, df in wl_by_arm.items():
        if df is None or df.empty:
            continue
        for _, r in df.iterrows():
            rows.append({"arm": arm, "strategy": r["strategy_key"], "net": r.get("net_btc_pnl")})
    if not rows:
        return None
    t = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    strategies = sorted(t["strategy"].unique())
    arms = [a for a in ("static", "equal", "adaptive") if a in set(t["arm"])]
    import numpy as np
    x = np.arange(len(strategies))
    width = 0.25
    for i, arm in enumerate(arms):
        vals = [
            float(t[(t.arm == arm) & (t.strategy == sk)]["net"].iloc[0])
            if len(t[(t.arm == arm) & (t.strategy == sk)]) else 0.0
            for sk in strategies
        ]
        ax.bar(x + i * width, vals, width, label=arm, color=ARM_COLORS.get(arm))
    ax.set_xticks(x + width)
    ax.set_xticklabels(strategies)
    ax.set_title("Net BTC PnL by arm × strategy")
    ax.legend()
    ax.axhline(0, color="black", lw=0.8)
    p = plots_dir / "win_loss_net_pnl.png"
    _save(fig, p)
    return p
