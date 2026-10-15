"""HTF4H step 7 — slot count x size per slot for the accepted 4h setup (19 months).

Setup fixed: x16 4h score, S>=0.65, SL25/TP30/5d, one position per coin, 2h coin block, no weekend block.
Grid: max open positions N x size per position f (fraction of realized equity at entry). Spot, no leverage:
a position gets min(f x equity, free cash); if free cash < 1% of equity the signal is skipped.

    python -m htf_4h.step7_sizing

Writes: results/htf_4h/step7/
"""

from __future__ import annotations

import heapq
from itertools import product

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from htf_4h.common import BAR, RESULTS
from htf_4h.report_winner import signals
from htf_4h.step4_portfolio import LIVE_GAP, RT_FEE, mark_to_market
from htf_4h.step6_lookback import OLD_BLOCK, PERIODS

OUT = RESULTS / "step7"
SLOTS = (4, 5, 6, 8, 10, 12, 15, 20, 25)
SIZES = (0.04, 0.05, 0.0667, 0.08, 0.10, 0.125, 0.15, 0.20, 0.25)
MIN_SIZE = 0.01
CURRENT = (10, 0.10)


def simulate_cash(t: pd.DataFrame, slots: int, frac: float, extra_cost: float = 0.0) -> pd.DataFrame:
    equity = 1.0
    heap: list[tuple] = []
    open_coins: set[str] = set()
    last_close: dict[str, pd.Timestamp] = {}
    invested = 0.0
    taken = []
    for k, e in enumerate(t.itertuples()):
        while heap and heap[0][0] <= e.entry_utc:
            ex_ts, _, base, size, pnl = heapq.heappop(heap)
            equity += pnl
            invested -= size
            open_coins.discard(base)
            last_close[base] = ex_ts
        if e.base in open_coins or len(heap) >= slots:
            continue
        if e.base in last_close and e.entry_utc - last_close[e.base] < OLD_BLOCK:
            continue
        size = min(frac * equity, equity - invested)
        if size < MIN_SIZE * equity:
            continue
        net = e.net - extra_cost
        heapq.heappush(heap, (e.exit_utc, k, e.base, size, size * net))
        open_coins.add(e.base)
        invested += size
        taken.append({"base": e.base, "entry_utc": e.entry_utc, "exit_utc": e.exit_utc, "entry_px": e.entry_px,
                      "size": size, "net": net, "hold_d": e.hold_d, "reason": e.reason, "S": e.S})
    return pd.DataFrame(taken)


def period_stats(curve: pd.Series, a: pd.Timestamp, b: pd.Timestamp) -> dict:
    c = curve[(curve.index >= a) & (curve.index <= b)]
    c = c / c.iloc[0]
    dd = (c / c.cummax() - 1).min()
    daily = c.resample("D").last().pct_change().dropna()
    days = (c.index[-1] - c.index[0]).total_seconds() / 86400
    ret = c.iloc[-1] - 1
    cagr = c.iloc[-1] ** (365 / days) - 1
    return {"ret": 100 * ret, "dd": 100 * dd, "ret_dd": ret / abs(dd), "calmar": cagr / abs(dd),
            "sharpe": float(daily.mean() / daily.std() * np.sqrt(365))}


def heat(ax, p: pd.DataFrame, title: str, fmt: str, cmap: str = "RdYlGn", mark=CURRENT) -> None:
    v = p.to_numpy(float)
    im = ax.imshow(v, cmap=cmap, aspect="auto")
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            ax.text(j, i, format(v[i, j], fmt), ha="center", va="center", fontsize=7)
    ax.set_xticks(range(p.shape[1]))
    ax.set_xticklabels([f"{c*100:.3g}%" for c in p.columns])
    ax.set_yticks(range(p.shape[0]))
    ax.set_yticklabels(p.index)
    ax.set_xlabel("size per position")
    ax.set_ylabel("max open positions")
    if mark in [(r, c) for r in p.index for c in p.columns]:
        ax.add_patch(plt.Rectangle((list(p.columns).index(mark[1]) - 0.5, list(p.index).index(mark[0]) - 0.5), 1, 1,
                                   fill=False, ec="black", lw=2))
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.04)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 30)
    t, r4 = signals()
    closes = {b: g.set_index(g["timestamp"] + BAR)["close"] for b, g in r4.groupby("base")}
    grid = pd.date_range(PERIODS["ALL"][0], t["exit_utc"].max().ceil("4h"), freq="4h")
    rows = []
    for n, f in product(SLOTS, SIZES):
        for gap in (0.0, LIVE_GAP):
            tk = simulate_cash(t, n, f, gap)
            curve = mark_to_market(tk, closes, grid, gap)
            row = {"slots": n, "size": f, "max_invested": min(1.0, n * f), "live_gap": gap > 0, "trades": len(tk)}
            for per, (a, b) in PERIODS.items():
                for k, v in period_stats(curve, a, b).items():
                    row[f"{per}_{k}"] = v
            rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "sizing_grid.csv", index=False)

    r = res[~res["live_gap"]]
    piv = lambda col, d=r: d.pivot(index="slots", columns="size", values=col)
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    heat(axes[0][0], piv("ALL_ret"), "19-month return %", ".0f")
    heat(axes[0][1], piv("ALL_dd"), "19-month max drawdown %", ".1f")
    heat(axes[0][2], piv("ALL_ret_dd"), "19-month return / |max DD|", ".1f")
    heat(axes[1][0], piv("ALL_sharpe"), "19-month Sharpe (daily, annualised)", ".2f")
    heat(axes[1][1], piv("OOS_ret_dd"), "separate year: return / |max DD|", ".2f")
    heat(axes[1][2], piv("IS_ret_dd"), "tuning window: return / |max DD|", ".1f")
    fig.suptitle("HTF4H step 7 — max open positions x size per position (black box = current 10 x 10%); "
                 "spot: total invested capped at 100%")
    fig.tight_layout()
    fig.savefig(OUT / "step7_sizing.png", dpi=125)
    plt.close(fig)

    cols = ["slots", "size", "trades", "ALL_ret", "ALL_dd", "ALL_ret_dd", "ALL_sharpe", "OOS_ret", "OOS_dd",
            "OOS_ret_dd", "IS_ret", "IS_dd", "IS_ret_dd"]
    print("== top 15 by 19-month Sharpe ==")
    print(r.sort_values("ALL_sharpe", ascending=False)[cols].head(15).round(2).to_string(index=False))
    print("\n== top 15 by 19-month return/|DD| ==")
    print(r.sort_values("ALL_ret_dd", ascending=False)[cols].head(15).round(2).to_string(index=False))
    print("\n== fully invested diagonal (slots x size = 100%) ==")
    diag = r[(r["slots"] * r["size"]).round(2).between(0.99, 1.01)]
    print(diag.sort_values("slots")[cols].round(2).to_string(index=False))
    print("\n== 10 slots, every size ==")
    print(r[r["slots"] == 10][cols].round(2).to_string(index=False))
    print("\n== size 10%, every slot count ==")
    print(r[r["size"] == 0.10][cols].round(2).to_string(index=False))
    g = res[res["live_gap"]]
    print("\n== with live gap: selected ==")
    sel = g[g.apply(lambda x: (x["slots"], round(x["size"], 4)) in
                    {(10, 0.10), (8, 0.125), (6, 0.1667), (5, 0.2), (12, 0.0833), (15, 0.0667), (20, 0.05),
                     (10, 0.125), (10, 0.15), (8, 0.15), (12, 0.10), (15, 0.10)}, axis=1)]
    print(sel[cols].round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
