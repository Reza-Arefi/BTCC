"""Plot generators — always paired with saved metric CSVs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from btcc.sim.score import ACTIVE_SIGNAL_KEYS, FACTOR_KEYS

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
            # Display as 0–100% (underlying series remains 0–1)
            g = g.copy()
            g["_wr_pct"] = g["rolling_win_rate"].astype(float) * 100.0
            if "day_number" in g.columns:
                g = g.sort_values("day_number")
                ax.plot(g["day_number"], g["_wr_pct"], label=arm, color=ARM_COLORS.get(arm))
            else:
                ax.plot(pd.to_datetime(g["exit_ts"]), g["_wr_pct"], label=arm, color=ARM_COLORS.get(arm))
        _mark_init_day(ax, init_days)
        ax.set_title(f"Rolling 30d Win Rate (%) — {STRATEGY_SHORT.get(sk, sk)}")
        ax.set_xlabel("Day" if any(
            d is not None and not d.empty and "day_number" in d.columns for d in wr_by_arm.values()
        ) else "Time")
        ax.set_ylabel("Win Rate (%)")
        ax.set_ylim(0, 100)
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
    """Plot active signal weights only — BTC.D / btc_regime excluded (contextual)."""
    if wh.empty or "indicator" not in wh.columns:
        return None
    fig, ax = plt.subplots(figsize=(11, 6))
    for ind in ACTIVE_SIGNAL_KEYS:
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
    ax.set_title("Adaptive indicator weights through time (BTC.D excluded)")
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
    def _ind_label(ind: str) -> str:
        return f"{ind} (Context only)" if str(ind) == "btc_regime" else str(ind)

    if df["asof"].astype(str).eq("end_of_sample").all():
        fig, ax = plt.subplots(figsize=(9, 4))
        g = df.dropna(subset=["correlation"]).copy()
        labels = [_ind_label(i) for i in g["indicator"].astype(str)]
        ax.bar(labels, g["correlation"])
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
        if "day_number" in g.columns and g["day_number"].notna().any():
            g = g.sort_values("day_number")
            ax.plot(g["day_number"], g["correlation"], label=_ind_label(ind))
        else:
            ax.plot(pd.to_datetime(g["asof"], utc=True), g["correlation"], label=_ind_label(ind))
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Predictive correlation (matured only): indicator vs future_return_4h")
    ax.set_xlabel("Day" if "day_number" in df.columns else "Time")
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


def plot_compounded_capital(
    daily_by_label: dict[str, pd.DataFrame],
    plots_dir: Path,
    *,
    strategy_key: str,
    starting_capital_usd: float = 1000.0,
    init_days: int | None = 90,
) -> Path | None:
    """Cumulative return (%) from starting_capital vs Day — one figure per exit strategy."""
    fig, ax = plt.subplots(figsize=(11, 5))
    any_line = False
    for label, df in daily_by_label.items():
        if df is None or df.empty:
            continue
        g = df[df["strategy_key"] == strategy_key] if "strategy_key" in df.columns else df
        if g.empty or "day_number" not in g.columns:
            continue
        g = g.sort_values("day_number")
        if "cumulative_return_pct" in g.columns:
            y = g["cumulative_return_pct"]
        else:
            y = 100.0 * (g["ending_value"].astype(float) / float(starting_capital_usd) - 1.0)
        ax.plot(g["day_number"], y, label=label, lw=1.4)
        any_line = True
    if not any_line:
        plt.close(fig)
        return None
    _mark_init_day(ax, init_days)
    ax.axhline(0.0, color="#888", ls=":", lw=1, alpha=0.8)
    ax.set_xlabel("Day")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title(f"Cumulative Return (%) from ${starting_capital_usd:,.0f} — {strategy_key}")
    ax.legend(fontsize=8)
    p = plots_dir / f"compounded_capital_{strategy_key}.png"
    _save(fig, p)
    return p


def plot_cumulative_pl_pct(
    daily_by_label: dict[str, pd.DataFrame],
    plots_dir: Path,
    *,
    strategy_key: str,
    starting_capital_usd: float = 1000.0,
    init_days: int | None = 90,
) -> Path | None:
    """Cumulative P/L (%) vs Day — one figure per exit strategy."""
    fig, ax = plt.subplots(figsize=(11, 5))
    any_line = False
    for label, df in daily_by_label.items():
        if df is None or df.empty:
            continue
        g = df[df["strategy_key"] == strategy_key] if "strategy_key" in df.columns else df
        if g.empty or "day_number" not in g.columns:
            continue
        g = g.sort_values("day_number")
        ax.plot(g["day_number"], g["cumulative_return_pct"], label=label, lw=1.4)
        any_line = True
    if not any_line:
        plt.close(fig)
        return None
    _mark_init_day(ax, init_days)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Day")
    ax.set_ylabel("Cumulative P/L (%)")
    ax.set_title(f"Cumulative P/L (%) from ${starting_capital_usd:,.0f} — {strategy_key}")
    ax.legend(fontsize=8)
    p = plots_dir / f"cumulative_pl_pct_{strategy_key}.png"
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


STRATEGY_COLORS = {
    "strategy_1": "#1f77b4",
    "strategy_2": "#ff7f0e",
    "strategy_3": "#2ca02c",
    "strategy_4": "#d62728",
    "strategy_5": "#9467bd",
}
STRATEGY_SHORT = {
    "strategy_1": "S1",
    "strategy_2": "S2",
    "strategy_3": "S3",
    "strategy_4": "S4",
    "strategy_5": "S5",
}


def plot_s1_s5_portfolio(
    daily: pd.DataFrame,
    plots_dir: Path,
    *,
    title: str,
    filename: str,
    starting_capital_usd: float = 1000.0,
    init_days: int | None = 90,
) -> Path | None:
    """S1–S5 cumulative return (%) from independent $1,000 accounts (plot-only)."""
    if daily is None or daily.empty or "day_number" not in daily.columns:
        return None
    fig, ax = plt.subplots(figsize=(11, 5))
    any_line = False
    for sk in ("strategy_1", "strategy_2", "strategy_3", "strategy_4", "strategy_5"):
        g = daily[daily["strategy_key"] == sk].sort_values("day_number")
        if g.empty:
            continue
        if "cumulative_return_pct" in g.columns:
            y = g["cumulative_return_pct"]
        else:
            y = 100.0 * (g["ending_value"].astype(float) / float(starting_capital_usd) - 1.0)
        ax.plot(
            g["day_number"], y,
            label=STRATEGY_SHORT[sk], color=STRATEGY_COLORS[sk], lw=1.5,
        )
        any_line = True
    if not any_line:
        plt.close(fig)
        return None
    ax.axhline(0.0, color="#888", ls=":", lw=1.2, label="0% (start)")
    _mark_init_day(ax, init_days)
    ax.set_xlabel("Day")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    p = plots_dir / filename
    _save(fig, p)
    return p


def plot_s1_s5_cumulative_pl(
    daily: pd.DataFrame,
    plots_dir: Path,
    *,
    title: str,
    filename: str,
    init_days: int | None = 90,
) -> Path | None:
    if daily is None or daily.empty:
        return None
    fig, ax = plt.subplots(figsize=(11, 5))
    any_line = False
    for sk in ("strategy_1", "strategy_2", "strategy_3", "strategy_4", "strategy_5"):
        g = daily[daily["strategy_key"] == sk].sort_values("day_number")
        if g.empty:
            continue
        ax.plot(
            g["day_number"], g["cumulative_return_pct"],
            label=STRATEGY_SHORT[sk], color=STRATEGY_COLORS[sk], lw=1.5,
        )
        any_line = True
    if not any_line:
        plt.close(fig)
        return None
    ax.axhline(0, color="black", lw=0.8)
    ax.fill_between(ax.get_xlim(), 0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0, alpha=0.02, color="green")
    _mark_init_day(ax, init_days)
    ax.set_xlabel("Day")
    ax.set_ylabel("Cumulative P/L (%)")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    p = plots_dir / filename
    _save(fig, p)
    return p


def plot_s1_s5_drawdown_pct(
    dd: pd.DataFrame,
    plots_dir: Path,
    *,
    title: str,
    filename: str,
    init_days: int | None = 90,
) -> Path | None:
    if dd is None or dd.empty:
        return None
    fig, ax = plt.subplots(figsize=(11, 5))
    any_line = False
    for sk in ("strategy_1", "strategy_2", "strategy_3", "strategy_4", "strategy_5"):
        g = dd[dd["strategy_key"] == sk].sort_values("day_number")
        if g.empty:
            continue
        ax.plot(
            g["day_number"], g["drawdown_pct"],
            label=STRATEGY_SHORT[sk], color=STRATEGY_COLORS[sk], lw=1.4,
        )
        any_line = True
    if not any_line:
        plt.close(fig)
        return None
    _mark_init_day(ax, init_days)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Day")
    ax.set_ylabel("Drawdown (%)")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    p = plots_dir / filename
    _save(fig, p)
    return p


def plot_s1_s5_btc_pnl(
    btc_eq: pd.DataFrame,
    plots_dir: Path,
    *,
    title: str,
    filename: str,
    init_days: int | None = 90,
    starting_capital_usd: float = 1000.0,
) -> Path | None:
    """Cumulative trading P/L (%) vs each strategy's own initial account (plot-only).

    Prefers ``cumulative_return_pct`` when present (USD $1,000 account).
    Else normalizes cumulative BTC PnL to the strategy's initial BTC-equivalent stake.
    """
    if btc_eq is None or btc_eq.empty:
        return None
    fig, ax = plt.subplots(figsize=(11, 5))
    any_line = False
    for sk in ("strategy_1", "strategy_2", "strategy_3", "strategy_4", "strategy_5"):
        g = btc_eq[btc_eq["strategy_key"] == sk].sort_values("day_number")
        if g.empty:
            continue
        if "cumulative_return_pct" in g.columns:
            y = g["cumulative_return_pct"].astype(float)
        elif "cumulative_btc_pnl_pct" in g.columns:
            y = g["cumulative_btc_pnl_pct"].astype(float)
        elif "cumulative_btc_pnl" in g.columns and "ending_value_usd" in g.columns:
            # Fallback: % of initial USD capital via first-day USD / initial stake proxy
            init_usd = float(starting_capital_usd)
            # Prefer explicit initial BTC stake from first total_btc_equiv if available
            if "total_btc_equiv" in g.columns and g["total_btc_equiv"].notna().any():
                init_btc = float(g["total_btc_equiv"].dropna().iloc[0])
                # Undo first-day cum pnl already in total? Use price path:
                # trading PnL % ≈ 100 * cum_btc / (init_usd / first_price)
                first_px = None
                if "btc_price" in g.columns and g["btc_price"].notna().any():
                    first_px = float(g["btc_price"].dropna().iloc[0])
                init_btc_stake = (init_usd / first_px) if first_px and first_px > 0 else init_btc
            else:
                first_px = None
                if "btc_price" in g.columns and g["btc_price"].notna().any():
                    first_px = float(g["btc_price"].dropna().iloc[0])
                init_btc_stake = (init_usd / first_px) if first_px and first_px > 0 else None
            if not init_btc_stake:
                continue
            y = 100.0 * g["cumulative_btc_pnl"].astype(float) / float(init_btc_stake)
        else:
            continue
        ax.plot(
            g["day_number"], y,
            label=STRATEGY_SHORT[sk], color=STRATEGY_COLORS[sk], lw=1.4,
        )
        any_line = True
    if not any_line:
        plt.close(fig)
        return None
    _mark_init_day(ax, init_days)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Day")
    ax.set_ylabel("Cumulative P/L (%)")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    p = plots_dir / filename
    _save(fig, p)
    return p


def plot_s1_s5_btc_total_equiv(
    btc_eq: pd.DataFrame,
    plots_dir: Path,
    *,
    title: str,
    filename: str,
    init_days: int | None = 90,
) -> Path | None:
    """BTC-equivalent account cumulative return (%) — initial = 0% (plot-only)."""
    if btc_eq is None or btc_eq.empty:
        return None
    ycol = "total_btc_equiv_return_pct"
    if ycol not in btc_eq.columns and "total_btc_equiv" not in btc_eq.columns:
        return None
    fig, ax = plt.subplots(figsize=(11, 5))
    any_line = False
    for sk in ("strategy_1", "strategy_2", "strategy_3", "strategy_4", "strategy_5"):
        g = btc_eq[btc_eq["strategy_key"] == sk].sort_values("day_number").copy()
        if g.empty:
            continue
        if ycol in g.columns and g[ycol].notna().any():
            y = g[ycol].astype(float)
        else:
            base = g["total_btc_equiv"].astype(float)
            if base.isna().all():
                continue
            init_v = float(base.dropna().iloc[0])
            if not init_v:
                continue
            y = 100.0 * (base / init_v - 1.0)
        ax.plot(
            g["day_number"], y,
            label=STRATEGY_SHORT[sk], color=STRATEGY_COLORS[sk], lw=1.4,
        )
        any_line = True
    if not any_line:
        plt.close(fig)
        return None
    _mark_init_day(ax, init_days)
    ax.axhline(0.0, color="#888", ls=":", lw=1.0)
    ax.set_xlabel("Day")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    p = plots_dir / filename
    _save(fig, p)
    return p


def plot_cumulative_win_rate(
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
            g = df[df["strategy_key"] == sk].dropna(subset=["cum_win_rate"]).copy()
            if g.empty:
                continue
            g["_wr_pct"] = g["cum_win_rate"].astype(float) * 100.0
            if "day_number" in g.columns and g["day_number"].notna().any():
                g = g.sort_values("day_number")
                ax.plot(g["day_number"], g["_wr_pct"], label=arm, color=ARM_COLORS.get(arm))
            else:
                g = g.sort_values("exit_ts")
                ax.plot(pd.to_datetime(g["exit_ts"]), g["_wr_pct"], label=arm, color=ARM_COLORS.get(arm))
        _mark_init_day(ax, init_days)
        ax.set_ylim(0, 100)
        ax.set_title(f"Cumulative Win Rate (%) — {STRATEGY_SHORT.get(sk, sk)}")
        ax.set_xlabel("Day")
        ax.set_ylabel("Win Rate (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        p = plots_dir / f"cumulative_win_rate_{sk}.png"
        _save(fig, p)
        paths.append(p)
    return paths


def plot_expectancy_bars(exp: pd.DataFrame, plots_dir: Path, *, arm: str) -> Path | None:
    if exp is None or exp.empty:
        return None
    import numpy as np
    fig, ax = plt.subplots(figsize=(10, 4))
    strategies = [s for s in STRATEGY_SHORT if s in set(exp["strategy_key"])]
    pols = sorted(exp["entry_policy"].unique())
    x = np.arange(len(strategies))
    width = 0.35 if len(pols) > 1 else 0.5
    for i, pol in enumerate(pols):
        vals = []
        for sk in strategies:
            sub = exp[(exp.strategy_key == sk) & (exp.entry_policy == pol)]
            vals.append(float(sub["expectancy_btc"].iloc[0]) if len(sub) else 0.0)
        ax.bar(x + i * width, vals, width, label=pol)
    ax.set_xticks(x + width * (len(pols) - 1) / 2)
    ax.set_xticklabels([STRATEGY_SHORT[s] for s in strategies])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(f"Expectancy per Trade (BTC) — {arm}")
    ax.legend(fontsize=8)
    p = plots_dir / f"expectancy_bars_{arm}.png"
    _save(fig, p)
    return p


def plot_profit_factor_bars(exp: pd.DataFrame, plots_dir: Path, *, arm: str) -> Path | None:
    if exp is None or exp.empty or "profit_factor" not in exp.columns:
        return None
    import numpy as np
    fig, ax = plt.subplots(figsize=(10, 4))
    strategies = [s for s in STRATEGY_SHORT if s in set(exp["strategy_key"])]
    pols = sorted(exp["entry_policy"].unique())
    x = np.arange(len(strategies))
    width = 0.35 if len(pols) > 1 else 0.5
    for i, pol in enumerate(pols):
        vals = []
        for sk in strategies:
            sub = exp[(exp.strategy_key == sk) & (exp.entry_policy == pol)]
            v = sub["profit_factor"].iloc[0] if len(sub) else None
            vals.append(float(v) if v is not None and pd.notna(v) else 0.0)
        ax.bar(x + i * width, vals, width, label=pol)
    ax.set_xticks(x + width * (len(pols) - 1) / 2)
    ax.set_xticklabels([STRATEGY_SHORT[s] for s in strategies])
    ax.axhline(1.0, color="black", ls="--", lw=0.8)
    ax.set_title(f"Profit Factor — {arm}")
    ax.legend(fontsize=8)
    p = plots_dir / f"profit_factor_bars_{arm}.png"
    _save(fig, p)
    return p


def plot_expectancy_over_days(
    exp_day: pd.DataFrame,
    plots_dir: Path,
    *,
    filename: str,
    init_days: int | None = 90,
) -> Path | None:
    if exp_day is None or exp_day.empty:
        return None
    fig, ax = plt.subplots(figsize=(11, 5))
    any_line = False
    for sk in STRATEGY_SHORT:
        g = exp_day[exp_day["strategy_key"] == sk].sort_values("day_number")
        if g.empty:
            continue
        ax.plot(g["day_number"], g["expectancy_btc"], label=STRATEGY_SHORT[sk], color=STRATEGY_COLORS[sk])
        any_line = True
    if not any_line:
        plt.close(fig)
        return None
    _mark_init_day(ax, init_days)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Day")
    ax.set_ylabel("Expectancy (BTC / trade)")
    ax.set_title("Expectancy over Day (expanding)")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    p = plots_dir / filename
    _save(fig, p)
    return p


def plot_win_loss_counts(wl: pd.DataFrame, plots_dir: Path, *, arm: str) -> Path | None:
    if wl is None or wl.empty:
        return None
    import numpy as np
    fig, ax = plt.subplots(figsize=(10, 4))
    strategies = [s for s in STRATEGY_SHORT if s in set(wl["strategy_key"])]
    x = np.arange(len(strategies))
    wins = [
        int(wl[wl.strategy_key == sk]["winning_trades"].iloc[0])
        if len(wl[wl.strategy_key == sk]) else 0
        for sk in strategies
    ]
    losses = [
        int(wl[wl.strategy_key == sk]["losing_trades"].iloc[0])
        if len(wl[wl.strategy_key == sk]) else 0
        for sk in strategies
    ]
    ax.bar(x - 0.2, wins, 0.4, label="wins", color="#2ca02c")
    ax.bar(x + 0.2, losses, 0.4, label="losses", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels([STRATEGY_SHORT[s] for s in strategies])
    ax.set_title(f"Winning vs Losing Trades — {arm}")
    ax.legend()
    p = plots_dir / f"win_loss_counts_{arm}.png"
    _save(fig, p)
    return p


def plot_funnel(funnel: pd.DataFrame, plots_dir: Path, *, arm: str) -> Path | None:
    if funnel is None or funnel.empty:
        return None
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(funnel["stage"][::-1], funnel["count"][::-1], color="#1f77b4")
    ax.set_xlabel("Count")
    ax.set_title(f"Opportunity Funnel — {arm} (BTC.D not a rejection stage)")
    p = plots_dir / f"opportunity_funnel_{arm}.png"
    _save(fig, p)
    return p


def plot_normal_vs_late_summary(legs: pd.DataFrame, plots_dir: Path, *, arm: str) -> Path | None:
    if legs is None or legs.empty or "entry_policy" not in legs.columns:
        return None
    import numpy as np
    rows = []
    for pol, g in legs.groupby("entry_policy"):
        pnl = pd.to_numeric(g.get("pnl_btc"), errors="coerce").dropna()
        rows.append({
            "policy": pol,
            "n": len(g),
            "win_rate": float((pnl > 0).mean()) if len(pnl) else 0.0,
            "sum_btc": float(pnl.sum()) if len(pnl) else 0.0,
        })
    if not rows:
        return None
    t = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].bar(t["policy"], t["n"], color=["#1f77b4", "#d62728"][:len(t)])
    axes[0].set_title("Trades")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(t["policy"], t["win_rate"], color=["#1f77b4", "#d62728"][:len(t)])
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Win Rate")
    axes[1].tick_params(axis="x", rotation=20)
    axes[2].bar(t["policy"], t["sum_btc"], color=["#1f77b4", "#d62728"][:len(t)])
    axes[2].axhline(0, color="black", lw=0.8)
    axes[2].set_title("Sum BTC PnL")
    axes[2].tick_params(axis="x", rotation=20)
    fig.suptitle(f"NORMAL vs LATE — {arm}")
    p = plots_dir / f"normal_vs_late_{arm}.png"
    _save(fig, p)
    return p


def plot_btcd_over_days(
    btcd_days: pd.DataFrame,
    plots_dir: Path,
    *,
    init_days: int | None = 90,
) -> Path | None:
    if btcd_days is None or btcd_days.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(btcd_days["day_number"], btcd_days["btc_d_mean"], color="#17becf", lw=1.4)
    _mark_init_day(ax, init_days)
    ax.set_xlabel("Day")
    ax.set_ylabel("Relative BTC.D (%)")
    ax.set_title("BTC.D over Day — Context Only (not a trading filter)")
    ax.grid(True, alpha=0.3)
    p = plots_dir / "btcd_over_days_context_only.png"
    _save(fig, p)
    return p
