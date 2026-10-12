"""Plots for results/live_exit_compare_2026-10-12 (run run_live_exit_compare_20261012.py first).

Equity is compounded at each simulated exit time: equity *= 1 + allocation x net return.
"""

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
RES = REPO / "results" / "live_exit_compare_2026-10-12"
OUT = RES / "plots"

HIGHLIGHT = {
    "LIVE": ("black", 2.8, "-"),
    "TRN_4": ("#d62728", 2.2, "-"),
    "TRN_5": ("#ff7f0e", 1.6, "-"),
    "TRN_3": ("#bcbd22", 1.6, "-"),
    "TRW_5": ("#9467bd", 1.6, "-"),
    "W2": ("#1f77b4", 2.2, "-"),
    "SL5_TP4": ("#17becf", 2.0, "--"),
    "T30": ("#7f7f7f", 2.0, "--"),
    "TRW_1": ("#2ca02c", 1.6, ":"),
}
LABEL = {
    "LIVE": "LIVE (actual bot)",
    "TRN_4": "TRN_4  SL5 / trail@4% / 0.5%",
    "TRN_5": "TRN_5  SL5 / trail@5% / 0.5%",
    "TRN_3": "TRN_3  SL5 / trail@3% / 0.5%",
    "TRW_5": "TRW_5  SL5 / trail@5% / 2%",
    "W2": "W2  SL5 / trail@4% / 2%",
    "SL5_TP4": "SL5_TP4  fixed TP +4% (live since Oct 6)",
    "T30": "T30  SL3 / trail@1% / 0.25%",
    "TRW_1": "TRW_1  SL5 / trail@1% / 2%",
}
FAMILY_COLOR = {"fixed trail": "#4c72b0", "fixed SL/TP": "#dd8452", "live": "black"}
COHORT_TITLE = {"W2": "W2 entries (37 trades, Sep 28 - Oct 12)", "T30": "T30 entries (112 trades, Sep 13 - Sep 21)"}


def equity_curve(g: pd.DataFrame, start: pd.Timestamp) -> pd.Series:
    g = g.sort_values("exit_utc")
    eq = (1.0 + g["alloc"].to_numpy() * g["net_pct"].to_numpy()).cumprod()
    s = pd.Series(np.r_[1.0, eq] * 100 - 100, index=pd.DatetimeIndex([start, *g["exit_utc"]]))
    return s


def plot_equity(detail: pd.DataFrame, cohort: str) -> None:
    d = detail[detail["cohort"] == cohort]
    start = d["entry_utc"].min()
    fig, (ax, axd) = plt.subplots(2, 1, figsize=(15, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    for name, g in d.groupby("strategy"):
        if name in HIGHLIGHT:
            continue
        ax.step(*_xy(equity_curve(g, start)), where="post", color="0.82", lw=0.7, zorder=1)
    ax.plot([], [], color="0.82", lw=0.7, label="other exits (38)")
    for name, (color, lw, ls) in HIGHLIGHT.items():
        s = equity_curve(d[d["strategy"] == name], start)
        ax.step(*_xy(s), where="post", color=color, lw=lw, ls=ls, label=f"{LABEL[name]}  ({s.iloc[-1]:+.1f}%)", zorder=3)
        dd = ((1 + s / 100) / (1 + s / 100).cummax() - 1) * 100
        axd.step(*_xy(dd), where="post", color=color, lw=lw * 0.8, ls=ls)
    if cohort == "W2":
        switch = pd.Timestamp("2026-10-06 12:00", tz="UTC")
        for a in (ax, axd):
            a.axvline(switch, color="red", lw=1, alpha=0.5)
        ax.text(switch, ax.get_ylim()[1] * 0.95, "  live switched to fixed TP +4%", color="red", va="top", fontsize=9)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Account return, compounded (%)")
    ax.set_title(f"Cumulative account return by exit - {COHORT_TITLE[cohort]}")
    ax.legend(loc="upper left", fontsize=8.5, ncol=2)
    ax.grid(alpha=0.3)
    axd.set_ylabel("Drawdown (%)")
    axd.grid(alpha=0.3)
    axd.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.tight_layout()
    fig.savefig(OUT / f"1_equity_{cohort}.png", dpi=130)
    plt.close(fig)


def _xy(s: pd.Series) -> tuple[pd.DatetimeIndex, np.ndarray]:
    return s.index, s.to_numpy()


def plot_ranking(summaries: dict[str, pd.DataFrame]) -> None:
    order = summaries["W2"].sort_values("equity_return_pct")["strategy"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(15, 13), sharey=True)
    for ax, (cohort, s) in zip(axes, summaries.items()):
        s = s.set_index("strategy").loc[order]
        colors = [FAMILY_COLOR[f] for f in s["family"]]
        ax.barh(range(len(s)), s["equity_return_pct"], color=colors)
        ax.axvline(0, color="black", lw=0.8)
        ax.axvline(s.loc["LIVE", "equity_return_pct"], color="black", ls="--", lw=1)
        ax.set_yticks(range(len(s)), s.index, fontsize=8)
        ax.set_title(COHORT_TITLE[cohort])
        ax.set_xlabel("Account return, simple sum (%)")
        ax.grid(axis="x", alpha=0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLOR.values()]
    axes[0].legend(handles, ["fixed trailing", "fixed SL / TP (no trail)", "LIVE (dashed line)"], loc="lower right")
    fig.suptitle("All 46 exits + live, sorted by W2-cohort result", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "2_ranking_all_exits.png", dpi=130)
    plt.close(fig)


def plot_heatmaps(summaries: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for col, (cohort, s) in enumerate(summaries.items()):
        r = s.set_index("strategy")["equity_return_pct"]
        grid = pd.DataFrame(
            [[r[f"SL{sl}_TP{tp}"] for tp in range(1, 6)] for sl in (2, 3, 4, 5)],
            index=[f"SL {sl}%" for sl in (2, 3, 4, 5)],
            columns=[f"TP +{tp}%" for tp in range(1, 6)],
        )
        _heat(axes[0, col], grid, f"Fixed SL / TP - {cohort}")
        sl5 = pd.DataFrame(
            [
                [r[f"SL5_TP{lv}"] for lv in range(1, 6)],
                [r[f"TRN_{lv}"] for lv in range(1, 6)],
                [r[f"TRW_{lv}"] if lv != 4 else r["W2"] for lv in range(1, 6)],
            ],
            index=["fixed TP", "trail 0.5%", "trail 2%"],
            columns=[f"+{lv}%" for lv in range(1, 6)],
        )
        _heat(axes[1, col], sl5, f"SL 5%: target / trail activation - {cohort}")
    axes[1, 0].set_xlabel("take-profit or trail activation level")
    axes[1, 1].set_xlabel("take-profit or trail activation level")
    fig.suptitle(f"Account return (%) by exit setting   |   LIVE: W2 cohort {summaries['W2'].set_index('strategy').loc['LIVE','equity_return_pct']:+.1f}%, "
                 f"T30 cohort {summaries['T30'].set_index('strategy').loc['LIVE','equity_return_pct']:+.1f}%", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "3_heatmaps_exit_settings.png", dpi=130)
    plt.close(fig)


def _heat(ax: plt.Axes, grid: pd.DataFrame, title: str) -> None:
    v = float(np.nanmax(np.abs(grid.to_numpy())))
    im = ax.imshow(grid.to_numpy(), cmap="RdYlGn", vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(grid.shape[1]), grid.columns)
    ax.set_yticks(range(grid.shape[0]), grid.index)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid.iat[i, j]:+.1f}", ha="center", va="center", fontsize=10)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.04)


def plot_per_trade(detail: pd.DataFrame) -> None:
    d = detail[detail["cohort"] == "W2"]
    names = ["LIVE", "TRN_4", "W2", "SL5_TP4", "TRW_1", "T30"]
    piv = d.pivot_table(index=["entry_utc", "symbol"], columns="strategy", values="net_pct").sort_index()[names] * 100
    mfe = d[d["strategy"] == "SL5_TP5"].set_index(["entry_utc", "symbol"])["mfe_pct"].sort_index() * 100
    fig, (ax, axm) = plt.subplots(1, 2, figsize=(15, 12), sharey=True, gridspec_kw={"width_ratios": [4, 1.2]})
    im = ax.imshow(piv.to_numpy(), cmap="RdYlGn", vmin=-6, vmax=6, aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.iat[i, j]:+.1f}", ha="center", va="center", fontsize=7.5)
    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(piv)), [f"{t:%m-%d %H:%M} {s.replace('BTC', '')}" for t, s in piv.index], fontsize=8)
    ax.xaxis.tick_top()
    switch = (piv.index.get_level_values(0) < pd.Timestamp("2026-10-06 12:00", tz="UTC")).sum() - 0.5
    ax.axhline(switch, color="red", lw=1.5)
    ax.set_title("Net return per W2 trade (%) by exit   (red line: live switched to fixed TP +4%)", pad=28)
    plt.colorbar(im, ax=ax, fraction=0.03)
    axm.barh(range(len(mfe)), mfe.to_numpy(), color=np.where(mfe.to_numpy() >= 4, "#2ca02c", "#c44e52"))
    axm.axvline(4, color="black", ls="--", lw=1)
    axm.axvline(1, color="0.5", ls=":", lw=1)
    axm.axhline(switch, color="red", lw=1.5)
    axm.set_title("Max gain reached (%)\n(green = reached +4%)")
    axm.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "4_per_trade_W2.png", dpi=130)
    plt.close(fig)


def plot_exit_reasons(detail: pd.DataFrame) -> None:
    names = ["TRN_4", "TRN_5", "TRN_3", "TRW_5", "W2", "SL5_TP4", "TRW_1", "T30"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    cmap = {"STOP_LOSS": "#c44e52", "TRAILING_STOP": "#2ca02c", "TAKE_PROFIT": "#55a868", "OPEN_MTM": "0.6"}
    for ax, cohort in zip(axes, ("W2", "T30")):
        d = detail[(detail["cohort"] == cohort) & detail["strategy"].isin(names)].copy()
        d["reason"] = d["exit_reason"].str.replace(r"_SAME_CANDLE.*|_TRAIL_ACTIVATION.*", "", regex=True)
        d["win"] = d["net_pct"] > 0
        d["bucket"] = np.where(
            d["reason"] == "TRAILING_STOP", np.where(d["win"], "TRAILING_STOP", "STOP_LOSS"), d["reason"]
        )
        ct = d.groupby(["strategy", "bucket"]).size().unstack(fill_value=0).reindex(names)
        bottom = np.zeros(len(ct))
        for b in ["TAKE_PROFIT", "TRAILING_STOP", "STOP_LOSS", "OPEN_MTM"]:
            if b in ct:
                ax.bar(ct.index, ct[b], bottom=bottom, color=cmap[b], label=b.replace("_", " ").lower())
                bottom += ct[b].to_numpy()
        ax.set_title(f"How trades closed - {COHORT_TITLE[cohort]}")
        ax.set_ylabel("trades")
        ax.tick_params(axis="x", rotation=30)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "5_exit_reasons.png", dpi=130)
    plt.close(fig)


def plot_risk_return(summaries: dict[str, pd.DataFrame], detail: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, (cohort, s) in zip(axes, summaries.items()):
        d = detail[detail["cohort"] == cohort]
        start = d["entry_utc"].min()
        mdd = {}
        for name, g in d.groupby("strategy"):
            e = 1 + equity_curve(g, start) / 100
            mdd[name] = float((e / e.cummax() - 1).min() * 100)
        s = s.assign(max_dd=s["strategy"].map(mdd))
        for _, r in s.iterrows():
            c = HIGHLIGHT.get(r["strategy"], (FAMILY_COLOR[r["family"]], 0, ""))[0]
            big = r["strategy"] in HIGHLIGHT
            ax.scatter(r["max_dd"], r["equity_return_pct"], color=c, s=70 if big else 22, alpha=1 if big else 0.6, zorder=3 if big else 2)
            if big:
                ax.annotate(r["strategy"], (r["max_dd"], r["equity_return_pct"]), xytext=(4, 4), textcoords="offset points", fontsize=9)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("Max drawdown (%)")
        ax.set_ylabel("Account return, simple sum (%)")
        ax.set_title(f"Return vs drawdown - {COHORT_TITLE[cohort]}")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "6_return_vs_drawdown.png", dpi=130)
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    detail = pd.read_csv(RES / "per_trade_by_strategy.csv", parse_dates=["entry_utc", "exit_utc"])
    summaries = {c: pd.read_csv(RES / f"summary_{c}.csv") for c in ("W2", "T30")}
    for cohort in ("W2", "T30"):
        plot_equity(detail, cohort)
    plot_ranking(summaries)
    plot_heatmaps(summaries)
    plot_per_trade(detail)
    plot_exit_reasons(detail)
    plot_risk_return(summaries, detail)
    print("\n".join(str(p) for p in sorted(OUT.glob("*.png"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
