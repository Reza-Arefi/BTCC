"""HTF4H — full 19-month report of the accepted 4h setup (2025-03 -> 2026-09).

Setup: x16 4h score, NEW_CROSS S>=0.65, SL 25% / TP 30% / 5d max, 10 slots x 10% of realized equity,
one position per coin, 2h same-coin block, no weekend block. Trades come from the step-6 1m walk.

    python -m htf_4h.report_winner

Writes: results/htf_4h/winner/report_*.png, report_stats.json, trades_detail.csv
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from htf_4h.common import BAR, RESULTS, load_coin
from htf_4h.step3_exits import RT_FEE
from htf_4h.step4_portfolio import LIVE_GAP, mark_to_market, simulate, stats
from htf_4h.step6_lookback import DIRS, LOAD_END, LOAD_START, OLD_BLOCK, PERIODS, SLOTS, build

OUT = RESULTS / "winner"
SCALE, THRESHOLD = "x16", 0.65
START = PERIODS["ALL"][0]
OOS_END = PERIODS["OOS"][1]
N_BOOT = 3000
GREEN, RED, GREY, BLUE = "#1a7f37", "#cf222e", "#8c959f", "#0969da"


def signals() -> tuple[pd.DataFrame, pd.DataFrame]:
    _, t, r4 = build()
    t = t[(t["scale"] == SCALE) & (t["threshold"] == THRESHOLD) & (t["entry_utc"] >= START)].copy()
    t["exit_utc"] = t["entry_utc"] + pd.to_timedelta(t["hold_d"], unit="D")
    t["exit_px"] = t["entry_px"] * (1 + t["net"] + RT_FEE)
    return t.sort_values(["entry_utc", "S"], ascending=[True, False]).reset_index(drop=True), r4


def _excursions(args) -> list[dict]:
    base, trades = args
    d = load_coin(base, DIRS, LOAD_START, LOAD_END)
    if d is None:
        return []
    r = d["rel1m"].set_index("timestamp").sort_index()
    out = []
    for e in trades.itertuples():
        w = r.loc[e.entry_utc : e.exit_utc - pd.Timedelta(minutes=1)]
        if w.empty:
            continue
        out.append({"base": base, "entry_utc": e.entry_utc, "mfe": w["high"].max() / e.entry_px - 1,
                    "mae": w["low"].min() / e.entry_px - 1})
    return out


def excursions(tk: pd.DataFrame) -> pd.DataFrame:
    jobs = [(b, g[["entry_utc", "exit_utc", "entry_px"]]) for b, g in tk.groupby("base")]
    rows = []
    with ProcessPoolExecutor(max_workers=18) as ex:
        for part in ex.map(_excursions, jobs):
            rows += part
    return tk.merge(pd.DataFrame(rows), on=["base", "entry_utc"], how="left")


def open_positions(tk: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.Series:
    gi = grid.asi8
    cnt = np.zeros(len(grid), dtype=int)
    for r in tk.itertuples():
        a = np.searchsorted(gi, r.entry_utc.value, side="left")
        b = np.searchsorted(gi, r.exit_utc.value, side="left")
        cnt[a:b] += 1
    return pd.Series(cnt, index=grid)


def basket_curve(r4: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.Series:
    p = r4.pivot(index="timestamp", columns="base", values="close").sort_index().ffill()
    p.index = p.index + BAR
    p = p[p.index >= START]
    return (1 + p.pct_change().mean(axis=1).fillna(0)).cumprod().reindex(grid).ffill().fillna(1.0)


def monthly(curve: pd.Series) -> pd.Series:
    m = curve.resample("M").last()
    return (m / m.shift(1, fill_value=1.0) - 1)[lambda x: x.index <= pd.Timestamp("2026-09-30", tz="UTC")]


def risk_stats(curve: pd.Series, tk: pd.DataFrame) -> dict:
    daily = curve.resample("D").last().dropna()
    dr = daily.pct_change().dropna()
    days = (curve.index[-1] - curve.index[0]).total_seconds() / 86400
    dd = curve / curve.cummax() - 1
    under = dd < 0
    runs = (under != under.shift()).cumsum()
    longest_dd_days = float(under.groupby(runs).sum().max() * 4 / 24) if under.any() else 0.0
    loss = (tk["net"] <= 0).astype(int)
    streak = loss.groupby((loss != loss.shift()).cumsum()).sum().max()
    cagr = curve.iloc[-1] ** (365 / days) - 1
    return {
        "total_return_pct": 100 * (curve.iloc[-1] - 1), "cagr_pct": 100 * cagr,
        "max_dd_pct": 100 * dd.min(), "longest_underwater_days": longest_dd_days,
        "sharpe_daily_ann": float(dr.mean() / dr.std() * np.sqrt(365)),
        "sortino_daily_ann": float(dr.mean() / dr[dr < 0].std() * np.sqrt(365)),
        "calmar": float(cagr / abs(dd.min())),
        "best_day_pct": 100 * dr.max(), "worst_day_pct": 100 * dr.min(),
        "trades": len(tk), "win_rate": float((tk["net"] > 0).mean()),
        "avg_trade_pct": 100 * tk["net"].mean(), "median_trade_pct": 100 * tk["net"].median(),
        "avg_win_pct": 100 * tk.loc[tk["net"] > 0, "net"].mean(), "avg_loss_pct": 100 * tk.loc[tk["net"] <= 0, "net"].mean(),
        "best_trade_pct": 100 * tk["net"].max(), "worst_trade_pct": 100 * tk["net"].min(),
        "profit_factor": float(tk.loc[tk["net"] > 0, "net"].sum() / -tk.loc[tk["net"] < 0, "net"].sum()),
        "longest_losing_streak": int(streak), "avg_hold_days": float(tk["hold_d"].mean()),
    }


def bootstrap(tk: pd.DataFrame, n: int = N_BOOT, seed: int = 7) -> pd.DataFrame:
    """Resample trades with replacement, apply them one after another at 10% of equity each."""
    rng = np.random.default_rng(seed)
    r = tk["net"].to_numpy() / SLOTS
    out = []
    for _ in range(n):
        eq = np.cumprod(1 + rng.choice(r, size=len(r), replace=True))
        out.append({"final_pct": 100 * (eq[-1] - 1), "max_dd_pct": 100 * (eq / np.maximum.accumulate(eq) - 1).min()})
    return pd.DataFrame(out)


def fmt_month_axis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))


def fig_equity(curve, curve_g, basket, openpos, st, path):
    fig, axes = plt.subplots(3, 1, figsize=(17, 12), sharex=True, gridspec_kw={"height_ratios": [3, 1.2, 1]})
    ax = axes[0]
    ax.axvspan(START, OOS_END, color="#eef3fa", zorder=0)
    ax.plot(curve.index, 100 * (curve - 1), color=GREEN, lw=1.7, label=f"strategy {100*(curve.iloc[-1]-1):+.1f}%")
    ax.plot(curve_g.index, 100 * (curve_g - 1), color=GREEN, lw=1, ls="--", alpha=0.7,
            label=f"with live gap 0.22pp/trade {100*(curve_g.iloc[-1]-1):+.1f}%")
    ax.plot(basket.index, 100 * (basket - 1), color=GREY, lw=1.1,
            label=f"all alts vs BTC, equal weight {100*(basket.iloc[-1]-1):+.1f}%")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("cumulative return % (BTC)")
    ax.set_title(f"Accumulation — CAGR {st['cagr_pct']:.0f}%, Sharpe {st['sharpe_daily_ann']:.2f}, "
                 f"max DD {st['max_dd_pct']:.1f}%, Calmar {st['calmar']:.2f}   (shaded = separate year)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    dd = curve / curve.cummax() - 1
    ax = axes[1]
    ax.fill_between(dd.index, 100 * dd, 0, color=RED, alpha=0.35)
    ax.plot(dd.index, 100 * dd, color=RED, lw=0.8)
    i = dd.idxmin()
    ax.annotate(f"{100*dd.min():.1f}%", (i, 100 * dd.min()), xytext=(10, -2), textcoords="offset points", fontsize=9)
    ax.set_ylabel("drawdown %")
    ax.set_title(f"Drawdown — longest time under water {st['longest_underwater_days']:.0f} days")
    ax.grid(alpha=0.3)
    ax = axes[2]
    ax.fill_between(openpos.index, openpos.to_numpy(), step="post", color=BLUE, alpha=0.4)
    ax.axhline(SLOTS, color="black", ls=":", lw=1)
    ax.set_ylabel("open positions")
    ax.set_ylim(0, SLOTS + 1)
    ax.set_title(f"Exposure — average {openpos.mean():.1f} of {SLOTS} slots in use "
                 f"({100*openpos.mean()/SLOTS:.0f}% of capital), all 10 used {100*(openpos >= SLOTS).mean():.1f}% of the time")
    ax.grid(alpha=0.3)
    fmt_month_axis(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_monthly(curve, basket, tk, path):
    m, mb = monthly(curve), monthly(basket)
    tk = tk.assign(month=tk["entry_utc"].dt.tz_localize(None).dt.to_period("M"))
    per = tk.groupby("month").agg(n=("net", "size"), win=("net", lambda x: (x > 0).mean()))
    labels = [d.strftime("%y-%m") for d in m.index]
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0][0]
    x = np.arange(len(m))
    ax.bar(x - 0.2, 100 * m.to_numpy(), width=0.4, color=np.where(m >= 0, GREEN, RED), label="strategy")
    ax.bar(x + 0.2, 100 * mb.reindex(m.index).to_numpy(), width=0.4, color=GREY, label="alts vs BTC basket")
    for i, v in enumerate(100 * m.to_numpy()):
        ax.text(i - 0.2, v + (0.7 if v >= 0 else -0.7), f"{v:+.1f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(f"Monthly return % — {int((m > 0).sum())}/{len(m)} positive months")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    ax = axes[0][1]
    mb2 = mb.reindex(m.index)
    ax.scatter(100 * mb2, 100 * m, color=np.where(m >= 0, GREEN, RED), s=45)
    for d, a, b in zip(m.index, 100 * mb2, 100 * m):
        ax.annotate(d.strftime("%y-%m"), (a, b), fontsize=7, xytext=(3, 3), textcoords="offset points")
    k = np.polyfit(100 * mb2, 100 * m, 1)
    xs = np.linspace(100 * mb2.min(), 100 * mb2.max(), 10)
    ax.plot(xs, np.polyval(k, xs), color="black", lw=1, ls="--")
    corr = np.corrcoef(mb2, m)[0, 1]
    ax.axhline(0, color="black", lw=0.6)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("alt basket vs BTC, month %")
    ax.set_ylabel("strategy, month %")
    ax.set_title(f"Strategy vs alt market (corr {corr:.2f}, slope {k[0]:.2f})")
    ax.grid(alpha=0.3)

    ax = axes[1][0]
    per = per.reindex(pd.PeriodIndex([d.strftime("%Y-%m") for d in m.index], freq="M"))
    ax.bar(x, per["n"], color=BLUE, alpha=0.6, label="trades")
    ax.set_ylabel("trades / month")
    ax2 = ax.twinx()
    ax2.plot(x, 100 * per["win"], color="black", marker="o", label="win rate")
    ax2.axhline(100 * (tk["net"] > 0).mean(), color="black", ls=":", lw=1)
    ax2.set_ylabel("win rate %")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Trades and win rate per month")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1][1]
    roll = curve / curve.shift(180) - 1
    ax.plot(roll.index, 100 * roll, color=BLUE, lw=1)
    ax.fill_between(roll.index, 100 * roll, 0, where=roll >= 0, color=GREEN, alpha=0.25)
    ax.fill_between(roll.index, 100 * roll, 0, where=roll < 0, color=RED, alpha=0.25)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Rolling 30-day return %")
    fmt_month_axis(ax)
    ax.tick_params(axis="x", rotation=60, labelsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_trades(tk, path):
    fig, axes = plt.subplots(2, 3, figsize=(19, 10))
    ax = axes[0][0]
    v = 100 * tk["net"]
    ax.hist(v[v > 0], bins=np.arange(-27, 32, 1), color=GREEN, alpha=0.8)
    ax.hist(v[v <= 0], bins=np.arange(-27, 32, 1), color=RED, alpha=0.8)
    ax.axvline(v.mean(), color="black", ls="--", lw=1)
    ax.set_title(f"Trade net return % — mean {v.mean():+.2f}, median {v.median():+.2f}")
    ax.set_xlabel("net % per trade")
    ax.grid(alpha=0.3)

    ax = axes[0][1]
    rs = tk.groupby("reason")["net"].agg(["size", "mean"])
    ax.bar(rs.index, rs["size"], color=[{"TP": GREEN, "SL": RED, "TIME": BLUE}.get(i, GREY) for i in rs.index])
    for i, (n, mu) in enumerate(zip(rs["size"], rs["mean"])):
        ax.text(i, n, f"{n} trades\navg {100*mu:+.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_title("Exit reasons")
    ax.set_ylim(0, rs["size"].max() * 1.2)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[0][2]
    ax.scatter(100 * tk["mae"], 100 * tk["mfe"], c=np.where(tk["net"] > 0, GREEN, RED), s=10, alpha=0.6)
    ax.axhline(30, color=GREEN, ls=":", lw=1)
    ax.axvline(-25, color=RED, ls=":", lw=1)
    ax.set_xlabel("worst point during trade (MAE %)")
    ax.set_ylabel("best point during trade (MFE %)")
    ax.set_title(f"Excursions — median MFE {100*tk['mfe'].median():+.1f}%, median MAE {100*tk['mae'].median():.1f}%")
    ax.grid(alpha=0.3)

    ax = axes[1][0]
    bins = [0.65, 0.67, 0.69, 0.71, 0.74, 1.0]
    g = tk.groupby(pd.cut(tk["S"], bins, right=False), observed=True)["net"].agg(["size", "mean", lambda x: (x > 0).mean()])
    g.columns = ["n", "mean", "win"]
    ax.bar(range(len(g)), 100 * g["mean"], color=np.where(g["mean"] >= 0, GREEN, RED))
    for i, (n, w) in enumerate(zip(g["n"], g["win"])):
        ax.text(i, 0, f"n={n}\nwin {100*w:.0f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels([f"{iv.left:.2f}-{iv.right:.2f}" for iv in g.index])
    ax.set_title("Avg net % by S at entry")
    ax.axhline(0, color="black", lw=0.8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1][1]
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    g = tk.groupby(tk["entry_utc"].dt.dayofweek)["net"].agg(["size", "mean"]).reindex(range(7))
    ax.bar(days, 100 * g["mean"], color=np.where(g["mean"] >= 0, GREEN, RED))
    for i, n in enumerate(g["size"]):
        ax.text(i, 0, f"n={n}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Avg net % by entry weekday (UTC)")
    ax.axhline(0, color="black", lw=0.8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1][2]
    g = tk.groupby(tk["entry_utc"].dt.hour)["net"].agg(["size", "mean"])
    ax.bar([f"{h:02d}:00" for h in g.index], 100 * g["mean"], color=np.where(g["mean"] >= 0, GREEN, RED))
    for i, n in enumerate(g["size"]):
        ax.text(i, 0, f"n={n}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Avg net % by entry hour (UTC)")
    ax.axhline(0, color="black", lw=0.8)
    ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Trade-level view (578 trades)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_robust(tk, boot, excl, path):
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    ax = axes[0]
    pnl = (100 * tk["size"] * tk["net"]).groupby(tk["base"]).sum().sort_values()
    ax.barh(pnl.index, pnl.to_numpy(), color=np.where(pnl >= 0, GREEN, RED))
    ax.set_title("P&L contribution by coin (% of starting equity)")
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(alpha=0.3, axis="x")

    ax = axes[1]
    ax.scatter(boot["max_dd_pct"], boot["final_pct"], s=4, alpha=0.3, color=BLUE)
    q = boot.quantile([0.05, 0.5, 0.95])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("max drawdown % (trades in sequence)")
    ax.set_ylabel("final return %")
    ax.set_title(f"Bootstrap of the trades ({len(boot)} runs)\nfinal return 5/50/95%: "
                 f"{q['final_pct'][0.05]:+.0f} / {q['final_pct'][0.5]:+.0f} / {q['final_pct'][0.95]:+.0f}%   "
                 f"P(loss) {100*(boot['final_pct'] < 0).mean():.1f}%\nmax DD 5/50/95%: "
                 f"{q['max_dd_pct'][0.05]:.0f} / {q['max_dd_pct'][0.5]:.0f} / {q['max_dd_pct'][0.95]:.0f}%")
    ax.grid(alpha=0.3)

    ax = axes[2]
    for (name, c), col in zip(excl.items(), plt.cm.viridis(np.linspace(0, 0.85, len(excl)))):
        ax.plot(c.index, 100 * (c - 1), color=col, lw=1.3, label=f"{name}: {100*(c.iloc[-1]-1):+.0f}%")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Dependence on the best coins (portfolio re-run without them)")
    ax.legend(fontsize=8)
    fmt_month_axis(ax)
    ax.tick_params(axis="x", rotation=60, labelsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t, r4 = signals()
    closes = {b: g.set_index(g["timestamp"] + BAR)["close"] for b, g in r4.groupby("base")}
    grid = pd.date_range(START, t["exit_utc"].max().ceil("4h"), freq="4h")

    tk, info = simulate(t, SLOTS, False, 0.0, OLD_BLOCK)
    curve = mark_to_market(tk, closes, grid, 0.0)
    tk_g, info_g = simulate(t, SLOTS, False, LIVE_GAP, OLD_BLOCK)
    curve_g = mark_to_market(tk_g, closes, grid, LIVE_GAP)
    tk = excursions(tk)
    tk.to_csv(OUT / "trades_detail.csv", index=False)
    st = risk_stats(curve, tk)
    st_g = risk_stats(curve_g, tk_g)
    basket = basket_curve(r4, grid)
    openpos = open_positions(tk, grid)
    boot = bootstrap(tk)

    pnl = (tk["size"] * tk["net"]).groupby(tk["base"]).sum().sort_values(ascending=False)
    excl = {"all coins": curve}
    for k in (1, 2, 4, 8):
        drop = set(pnl.index[:k])
        tk_x, _ = simulate(t[~t["base"].isin(drop)], SLOTS, False, 0.0, OLD_BLOCK)
        excl[f"without top {k} ({', '.join(pnl.index[:k]) if k <= 2 else 'incl. ' + pnl.index[0]})"] = \
            mark_to_market(tk_x, closes, grid, 0.0)

    fig_equity(curve, curve_g, basket, openpos, st, OUT / "report_1_equity.png")
    fig_monthly(curve, basket, tk, OUT / "report_2_monthly.png")
    fig_trades(tk, OUT / "report_3_trades.png")
    fig_robust(tk, boot, excl, OUT / "report_4_robustness.png")

    out = {"strategy": st, "with_live_gap": st_g, "alt_basket_pct": 100 * (basket.iloc[-1] - 1),
           "avg_open_positions": float(openpos.mean()), "pct_time_all_slots_full": float(100 * (openpos >= SLOTS).mean()),
           "bootstrap": boot.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(1).to_dict(),
           "bootstrap_p_loss": float((boot["final_pct"] < 0).mean()),
           "without_top_coins_pct": {k: round(100 * (c.iloc[-1] - 1), 1) for k, c in excl.items()},
           "skips": {k: v for k, v in info.items() if k.startswith("skip_")}}
    (OUT / "report_stats.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(json.dumps(out, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
