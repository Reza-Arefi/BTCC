"""Plots for results/live_no_trade_zones_2026-10-12 (run run_live_no_trade_zones_20261012.py first)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results" / "live_no_trade_zones_2026-10-12"
EXITS = REPO / "results" / "live_exit_compare_2026-10-12" / "per_trade_by_strategy.csv"
OUT = RES / "plots"
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
COHORT_COLOR = {"T30": "#1f77b4", "W2": "#d62728"}

FILTERS = {
    "no filter": lambda t: np.zeros(len(t), dtype=bool),
    "skip weekend": lambda t: (t["weekend"] == 1).to_numpy(),
    "skip BTC 24h >= +4%": lambda t: (t["btc_ret_24h"] >= 0.04).to_numpy(),
    "skip weekend + BTC 24h >= +4%": lambda t: ((t["weekend"] == 1) | (t["btc_ret_24h"] >= 0.04)).to_numpy(),
}
FILTER_STYLE = {
    "no filter": ("black", "-"),
    "skip weekend": ("#2ca02c", "-"),
    "skip BTC 24h >= +4%": ("#ff7f0e", "--"),
    "skip weekend + BTC 24h >= +4%": ("#9467bd", "-"),
}


def plot_weekday_hour(t: pd.DataFrame) -> None:
    t = t.assign(day=t["entry_utc"].dt.day_name().str[:3], hour4=(t["entry_utc"].dt.hour // 4) * 4)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for col, (key, order, label) in enumerate(
        [("day", DAYS, "weekday (UTC)"), ("hour4", list(range(0, 24, 4)), "entry hour block (UTC)")]
    ):
        g = t.groupby([key, "cohort"])["net_TRN_4"].agg(["mean", "size", lambda x: (x > 0).mean()])
        g.columns = ["mean", "n", "win"]
        x = np.arange(len(order))
        for k, c in enumerate(("T30", "W2")):
            sub = g.xs(c, level="cohort").reindex(order)
            axes[0, col].bar(x + (k - 0.5) * 0.4, sub["mean"] * 100, 0.4, color=COHORT_COLOR[c], label=f"{c} entries")
            axes[1, col].bar(x + (k - 0.5) * 0.4, sub["win"] * 100, 0.4, color=COHORT_COLOR[c])
            for xi, (n, v) in enumerate(zip(sub["n"], sub["mean"] * 100)):
                if pd.notna(n):
                    axes[0, col].text(xi + (k - 0.5) * 0.4, v + (0.15 if v >= 0 else -0.45), f"{int(n)}", ha="center", fontsize=8)
        ticks = order if key == "day" else [f"{h:02d}-{h + 4:02d}" for h in order]
        for row in (0, 1):
            axes[row, col].set_xticks(x, ticks)
            axes[row, col].grid(axis="y", alpha=0.3)
        axes[0, col].axhline(0, color="black", lw=0.8)
        axes[1, col].axhline(50, color="black", lw=0.8, ls=":")
        axes[0, col].set_title(f"Mean net return per trade (TRN_4 exit) by {label}  (numbers = trades)")
        axes[1, col].set_title(f"Win rate by {label}")
        axes[0, col].set_ylabel("mean net (%)")
        axes[1, col].set_ylabel("win rate (%)")
        if key == "day":
            for row in (0, 1):
                axes[row, col].axvspan(4.5, 6.5, color="0.9", zorder=0)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(OUT / "1_weekday_and_hour.png", dpi=130)
    plt.close(fig)


def plot_btc_context(t: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    for ax, (f, label) in zip(
        axes,
        [("btc_ret_24h", "BTC/USDT return, last 24h (%)"), ("btc_ret_4h", "BTC/USDT return, last 4h (%)"), ("btc_range_24h", "BTC 24h high-low range (%)")],
    ):
        for c in ("T30", "W2"):
            g = t[t["cohort"] == c]
            mk = np.where(g["weekend"] == 1, "x", "o")
            for m in ("o", "x"):
                s = g[mk == m]
                ax.scatter(s[f] * 100, s["net_TRN_4"] * 100, c=COHORT_COLOR[c], marker=m, s=30, alpha=0.75,
                           label=f"{c} {'weekend' if m == 'x' else 'weekday'}")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel(label)
        ax.set_ylabel("net return, TRN_4 exit (%)")
        ax.grid(alpha=0.3)
    axes[0].axvline(4, color="#ff7f0e", ls="--")
    axes[0].text(4.1, axes[0].get_ylim()[1] * 0.9, "BTC +4%/24h", color="#ff7f0e")
    axes[0].legend(fontsize=8)
    fig.suptitle("Trade result vs BTC context at entry (x = weekend entry)")
    fig.tight_layout()
    fig.savefig(OUT / "2_btc_context.png", dpi=130)
    plt.close(fig)


def equity(g: pd.DataFrame, net_col: str, start: pd.Timestamp) -> pd.Series:
    g = g.sort_values("exit_utc")
    eq = (1 + g["alloc"].to_numpy() * g[net_col].to_numpy()).cumprod()
    return pd.Series(np.r_[1.0, eq] * 100 - 100, index=pd.DatetimeIndex([start, *g["exit_utc"]]))


def plot_filtered_equity(t: pd.DataFrame, exits: pd.DataFrame) -> pd.DataFrame:
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for col, c in enumerate(("W2", "T30")):
        for row, ex in enumerate(("TRN_4", "LIVE")):
            ax = axes[row, col]
            e = exits[(exits["strategy"] == ex) & (exits["cohort"] == c)][["trade_id", "exit_utc", "net_pct"]]
            g = t[t["cohort"] == c].merge(e, on="trade_id")
            start = g["entry_utc"].min()
            for name, fn in FILTERS.items():
                keep = g[~fn(g)]
                s = equity(keep, "net_pct", start)
                color, ls = FILTER_STYLE[name]
                ax.step(s.index, s.to_numpy(), where="post", color=color, ls=ls, lw=2.2 if name == "no filter" else 1.8,
                        label=f"{name}: {len(keep)} trades, {s.iloc[-1]:+.1f}%")
                net = keep["net_pct"]
                dd = (1 + s / 100) / (1 + s / 100).cummax() - 1
                rows.append({"cohort": c, "exit": ex, "filter": name, "trades": len(keep), "wins": int((net > 0).sum()),
                             "losses": int((net < 0).sum()), "win_rate": float((net > 0).mean()),
                             "account_return_pct": float(s.iloc[-1]), "max_dd_pct": float(dd.min() * 100)})
            if c == "W2":
                ax.axvline(pd.Timestamp("2026-10-06 12:00", tz="UTC"), color="red", lw=1, alpha=0.4)
            for d in pd.date_range(start.floor("D"), g["exit_utc"].max().ceil("D"), freq="D"):
                if d.weekday() == 5:
                    ax.axvspan(d, d + pd.Timedelta(days=2), color="0.92", zorder=0)
            ax.axhline(0, color="black", lw=0.8)
            ax.set_title(f"{c} entries, {'TRN_4 exit (simulated)' if ex == 'TRN_4' else 'LIVE (actual bot exits)'}  - grey = weekends")
            ax.set_ylabel("account return, compounded (%)")
            ax.legend(fontsize=8, loc="upper left")
            ax.grid(alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.tight_layout()
    fig.savefig(OUT / "3_equity_with_filters.png", dpi=130)
    plt.close(fig)
    return pd.DataFrame(rows)


def plot_rule_map(rules: pd.DataFrame) -> None:
    r = rules.copy()
    r["T30_mean"] = r["T30_TRN_4_delta_sum_pct"] / 112
    r["W2_mean"] = r["W2_TRN_4_delta_sum_pct"] / 37
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.scatter(r["T30_mean"], r["W2_mean"], s=10, c=np.where(r["robust"], "#2ca02c", "0.75"), alpha=0.6)
    best = r[r["robust"]].drop_duplicates("feature").head(8)
    for _, x in best.iterrows():
        ax.annotate(x["rule"], (x["T30_mean"], x["W2_mean"]), xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("T30 entries: gain per trade from skipping (pp, TRN_4 exit)")
    ax.set_ylabel("W2 entries: gain per trade from skipping (pp, TRN_4 exit)")
    ax.set_title(f"All {len(r)} candidate skip-rules (green = helps both cohorts under TRN_4 and W2 exits)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "4_rule_map.png", dpi=130)
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t = pd.read_csv(RES / "trade_features.csv", parse_dates=["entry_utc", "live_exit_utc"])
    exits = pd.read_csv(EXITS, parse_dates=["entry_utc", "exit_utc"])
    rules = pd.read_csv(RES / "rules_all.csv")
    plot_weekday_hour(t)
    plot_btc_context(t)
    summary = plot_filtered_equity(t, exits)
    summary.to_csv(RES / "filter_summary.csv", index=False)
    plot_rule_map(rules)
    print(summary.round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
