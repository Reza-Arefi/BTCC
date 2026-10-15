"""HTF4H — equity of the accepted 4h setup over the full 19 months (2025-03 -> 2026-09).

Setup: x16 ("scaled") 4h score, NEW_CROSS S>=0.65, exit SL 25% / TP 30% / 5d max, 10 slots x 10% of realized
equity, one position per coin, 2h same-coin block, NO weekend block. Uses the step-6 trade/price cache.

    python -m htf_4h.plot_winner

Writes: results/htf_4h/winner/
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from htf_4h.common import BAR, RESULTS
from htf_4h.step3_exits import RT_FEE
from htf_4h.step4_portfolio import LIVE_GAP, mark_to_market, simulate, stats
from htf_4h.step6_lookback import OLD_BLOCK, PERIODS, SLOTS, build

OUT = RESULTS / "winner"
SCALE, THRESHOLD = "x16", 0.65
OOS_END = PERIODS["OOS"][1]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    _, t, r4 = build()
    start = PERIODS["ALL"][0]
    t = t[(t["scale"] == SCALE) & (t["threshold"] == THRESHOLD) & (t["entry_utc"] >= start)].copy()
    t["exit_utc"] = t["entry_utc"] + pd.to_timedelta(t["hold_d"], unit="D")
    t["exit_px"] = t["entry_px"] * (1 + t["net"] + RT_FEE)
    t = t.sort_values(["entry_utc", "S"], ascending=[True, False]).reset_index(drop=True)
    closes = {b: g.set_index(g["timestamp"] + BAR)["close"] for b, g in r4.groupby("base")}
    grid = pd.date_range(start, t["exit_utc"].max().ceil("4h"), freq="4h")

    tk, info = simulate(t, SLOTS, False, 0.0, OLD_BLOCK)
    curve = mark_to_market(tk, closes, grid, 0.0)
    tk_g, info_g = simulate(t, SLOTS, False, LIVE_GAP, OLD_BLOCK)
    curve_g = mark_to_market(tk_g, closes, grid, LIVE_GAP)
    st, st_g = stats(tk, info, curve), stats(tk_g, info_g, curve_g)
    tk.to_csv(OUT / "trades.csv", index=False)
    curve.rename("equity").to_csv(OUT / "equity_4h.csv")

    p = r4[r4["timestamp"] + BAR >= start].pivot(index="timestamp", columns="base", values="close").ffill()
    p.index = p.index + BAR
    basket = (1 + p.pct_change().mean(axis=1).fillna(0)).cumprod().reindex(grid).ffill()

    dd = curve / curve.cummax() - 1
    m = curve.resample("M").last()
    m = m / m.shift(1, fill_value=1.0) - 1
    m = m[m.index <= pd.Timestamp("2026-09-30", tz="UTC")]

    fig, axes = plt.subplots(3, 1, figsize=(17, 13), sharex=False, gridspec_kw={"height_ratios": [3, 1.2, 1.6]})
    ax = axes[0]
    ax.axvspan(start, OOS_END, color="#e8eef7", zorder=0)
    ax.text(start + pd.Timedelta(days=5), 0.97, "separate year (not used for tuning)", transform=ax.get_xaxis_transform(),
            va="top", fontsize=9, color="#335")
    ax.text(OOS_END + pd.Timedelta(days=5), 0.97, "tuning window", transform=ax.get_xaxis_transform(), va="top",
            fontsize=9, color="#335")
    ax.plot(curve.index, 100 * (curve - 1), color="#1a7f37", lw=1.6,
            label=f"strategy  {st['return_pct']:+.1f}%  (max DD {st['max_dd_pct']:.1f}%)")
    ax.plot(curve_g.index, 100 * (curve_g - 1), color="#1a7f37", lw=1, ls="--", alpha=0.7,
            label=f"with live gap 0.22pp/trade  {st_g['return_pct']:+.1f}%  (max DD {st_g['max_dd_pct']:.1f}%)")
    ax.plot(basket.index, 100 * (basket - 1), color="grey", lw=1, alpha=0.8,
            label=f"all alts vs BTC, equal weight (rebalanced)  {100 * (basket.iloc[-1] - 1):+.1f}%")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("cumulative return % (BTC terms)")
    ax.set_title(f"HTF4H accepted setup — x16 4h score, S>=0.65, SL25/TP30/5d, 10 slots x 10%, 2h coin block, "
                 f"no weekend block  |  {st['trades']} trades, win {100 * st['win_rate']:.0f}%, PF {st['pf']:.2f}")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.92))
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.fill_between(dd.index, 100 * dd, 0, color="#cf222e", alpha=0.35)
    ax.plot(dd.index, 100 * dd, color="#cf222e", lw=0.8)
    ax.set_ylabel("drawdown %")
    ax.grid(alpha=0.3)
    for a in axes[:2]:
        a.set_xlim(grid[0], grid[-1])
        a.xaxis.set_major_locator(mdates.MonthLocator())
        a.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))

    ax = axes[2]
    vals = 100 * m.to_numpy()
    ax.bar(range(len(m)), vals, color=np.where(vals >= 0, "#1a7f37", "#cf222e"))
    for i, v in enumerate(vals):
        ax.text(i, v + (0.6 if v >= 0 else -0.6), f"{v:+.1f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    ax.set_xticks(range(len(m)))
    ax.set_xticklabels([d.strftime("%y-%m") for d in m.index])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("monthly return %")
    ax.set_title(f"Monthly returns — {int((m > 0).sum())} of {len(m)} months positive")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "winner_equity_19m.png", dpi=130)
    plt.close(fig)

    summary = {"return_pct": st["return_pct"], "max_dd_pct": st["max_dd_pct"], "trades": st["trades"],
               "win_rate": st["win_rate"], "pf": st["pf"], "return_with_gap_pct": st_g["return_pct"],
               "max_dd_with_gap_pct": st_g["max_dd_pct"], "basket_pct": 100 * (basket.iloc[-1] - 1),
               "monthly_pct": {d.strftime("%Y-%m"): round(100 * v, 2) for d, v in m.items()},
               "exit_reasons": tk["reason"].value_counts().to_dict(), "avg_hold_d": float(tk["hold_d"].mean()),
               "pnl_by_coin_pct": (100 * (tk["size"] * tk["net"]).groupby(tk["base"]).sum()
                                   .sort_values(ascending=False)).round(2).to_dict(),
               "max_open": info["max_open"]}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
