"""HTF4H step 8 — 3-year backtest of the accepted 4h setup (2023-10 -> 2026-09, 36 months).

Setup (frozen): x16 4h score (history 400 bars), NEW_CROSS S>=0.65, SL 25% / TP 30% / 5d max, max 6 open
positions x 1/6 of realized equity, one position per coin, 2h same-coin block, no weekend block.
Data: data/oos_2025_candles (2023-08 .. 2026-02, RENDER spliced with RNDR before 2024-07-26) + main store.
The 17 months 2023-10 -> 2025-02 were never used for any choice: they are a clean out-of-sample test.

    python -m htf_4h.step8_3y

Writes: results/htf_4h/step8_3y/
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from btcc.research.early_breakout_entry import score_variant
from htf_4h.common import BAR, RESULTS, bases, load_coin, weights
from htf_4h.report_winner import open_positions, risk_stats
from htf_4h.step3_exits import DAY_NS, RT_FEE, _eval_trade
from htf_4h.step4_portfolio import LIVE_GAP, mark_to_market
from htf_4h.step6_lookback import DIRS, EXIT, HISTORY, HOLD_NS, OLD_BLOCK, THRESHOLDS, _signals
from htf_4h.step7_sizing import simulate_cash

OUT = RESULTS / "step8_3y"
LOAD_START = pd.Timestamp("2023-08-01 00:00", tz="UTC")
LOAD_END = pd.Timestamp("2026-09-24 23:00", tz="UTC")
START = pd.Timestamp("2023-10-01", tz="UTC")
THRESHOLD = 0.65
SLOTS, SIZE = 6, 1 / 6
REF = (10, 0.10)
SEGMENTS = {
    "new 17m (never used)": (START, pd.Timestamp("2025-03-01", tz="UTC")),
    "separate year": (pd.Timestamp("2025-03-01", tz="UTC"), pd.Timestamp("2026-03-01", tz="UTC")),
    "tuning window": (pd.Timestamp("2026-03-01", tz="UTC"), pd.Timestamp("2026-09-25", tz="UTC")),
    "ALL 36m": (START, pd.Timestamp("2026-09-25", tz="UTC")),
}
GREEN, RED, GREY, BLUE = "#1a7f37", "#cf222e", "#8c959f", "#0969da"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("htf4h.step8")


def _coin(base: str) -> dict | None:
    d = load_coin(base, DIRS, LOAD_START, LOAD_END)
    if d is None or len(d["rel4"]) <= HISTORY:
        return None
    w = weights()
    r4, a4, b4 = d["rel4"], d["alt4"], d["btc4"]
    rows = []
    for i in range(HISTORY - 1, len(r4)):
        lo = i + 1 - HISTORY
        ts = r4["timestamp"].iloc[i]
        nxt = i + 1 < len(r4) and r4["timestamp"].iloc[i + 1] - ts == BAR
        rows.append({"base": base, "timestamp": ts, "close_ts": ts + BAR,
                     "next_open": float(r4["open"].iloc[i + 1]) if nxt else np.nan,
                     "S": float(score_variant(alt_btc=r4.iloc[lo : i + 1], alt_usdt=a4.iloc[lo : i + 1],
                                              btc_usdt=b4.iloc[lo : i + 1], weights=w, interval="15m",
                                              use_e2=True)["S"])})
    s = pd.DataFrame(rows)
    rel = d["rel1m"].sort_values("timestamp")
    ts_all = rel["timestamp"].astype("int64").to_numpy()
    hi_all, lo_all, cl_all = (rel[k].to_numpy(float) for k in ("high", "low", "close"))
    data_end_ns = int(ts_all[-1])
    trades = []
    for thr in THRESHOLDS:
        for e in _signals(s, thr).itertuples():
            entry_ns = int(e.close_ts.value)
            i0 = int(np.searchsorted(ts_all, entry_ns, side="left"))
            i1 = int(np.searchsorted(ts_all, entry_ns + HOLD_NS, side="left"))
            if i0 >= i1:
                continue
            hi, lo_, cl = hi_all[i0:i1], lo_all[i0:i1], cl_all[i0:i1]
            entry = float(e.next_open)
            cmax = np.maximum.accumulate(np.maximum(hi, entry))
            neg_cmin = -np.minimum.accumulate(np.minimum(lo_, entry))
            res = _eval_trade(ts_all[i0:i1], hi, lo_, cl, cmax, neg_cmin, entry, entry_ns, data_end_ns, {}, EXIT)
            if res is None:
                continue
            k, px, reason = res
            trades.append({"threshold": thr, "base": base, "entry_utc": e.close_ts, "entry_px": entry, "S": float(e.S),
                           "net": px / entry - 1 - RT_FEE, "reason": reason,
                           "hold_d": (int(ts_all[i0 + k]) - entry_ns) / DAY_NS})
    return {"s": s, "trades": pd.DataFrame(trades), "rel4": r4[["timestamp", "close"]].assign(base=base),
            "first_bar": str(r4["timestamp"].iloc[0])}


def build(workers: int = 18):
    paths = {k: OUT / f"{k}.parquet" for k in ("s", "trades", "rel4")}
    if all(p.exists() for p in paths.values()):
        out = {k: pd.read_parquet(p) for k, p in paths.items()}
    else:
        parts: dict[str, list] = {k: [] for k in paths}
        first = {}
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_coin, b): b for b in bases()}
            for n, fut in enumerate(as_completed(futs), 1):
                res = fut.result()
                log.info("%s done (%d/%d) %.0fs%s", futs[fut], n, len(futs), time.time() - t0, "" if res else " NO DATA")
                if res:
                    first[futs[fut]] = res["first_bar"]
                    for k in parts:
                        parts[k].append(res[k])
        out = {k: pd.concat(v, ignore_index=True) for k, v in parts.items()}
        for k, p in paths.items():
            out[k].to_parquet(p, index=False)
        (OUT / "coin_first_bar.json").write_text(json.dumps(first, indent=2), encoding="utf-8")
    for df in out.values():
        for col in ("timestamp", "close_ts", "entry_utc"):
            if col in df:
                df[col] = pd.to_datetime(df[col], utc=True)
    return out["s"], out["trades"], out["rel4"]


def check_overlap(s: pd.DataFrame) -> dict:
    p = RESULTS / "step6" / "s.parquet"
    if not p.exists():
        return {}
    old = pd.read_parquet(p, columns=["base", "timestamp", "x16"])
    old["timestamp"] = pd.to_datetime(old["timestamp"], utc=True)
    m = s.merge(old, on=["base", "timestamp"])
    return {"overlap_bars": int(len(m)), "max_abs_dS": float((m["S"] - m["x16"]).abs().max()),
            "zone_agree": float(((m["S"] >= THRESHOLD) == (m["x16"] >= THRESHOLD)).mean())}


def seg_stats(curve: pd.Series, a, b) -> dict:
    c = curve[(curve.index >= a) & (curve.index <= b)]
    c = c / c.iloc[0]
    dd = (c / c.cummax() - 1).min()
    days = (c.index[-1] - c.index[0]).total_seconds() / 86400
    dr = c.resample("D").last().pct_change().dropna()
    m = c.resample("M").last()
    m = m / m.shift(1, fill_value=1.0) - 1
    return {"return_pct": 100 * (c.iloc[-1] - 1), "cagr_pct": 100 * (c.iloc[-1] ** (365 / days) - 1),
            "max_dd_pct": 100 * dd, "sharpe": float(dr.mean() / dr.std() * np.sqrt(365)),
            "pos_months": f"{int((m > 0).sum())}/{len(m)}"}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    s, t_all, r4 = build()
    chk = check_overlap(s)
    log.info("overlap check vs step 6: %s", chk)
    t = t_all[(t_all["threshold"] == THRESHOLD) & (t_all["entry_utc"] >= START)].copy()
    t["exit_utc"] = t["entry_utc"] + pd.to_timedelta(t["hold_d"], unit="D")
    t["exit_px"] = t["entry_px"] * (1 + t["net"] + RT_FEE)
    t = t.sort_values(["entry_utc", "S"], ascending=[True, False]).reset_index(drop=True)
    closes = {b: g.set_index(g["timestamp"] + BAR)["close"] for b, g in r4.groupby("base")}
    grid = pd.date_range(START, t["exit_utc"].max().ceil("4h"), freq="4h")

    curves, tks = {}, {}
    for name, (n, f, gap) in {"6 x 16.7%": (SLOTS, SIZE, 0.0), "6 x 16.7% + live gap": (SLOTS, SIZE, LIVE_GAP),
                              "10 x 10% (previous)": (*REF, 0.0)}.items():
        tk = simulate_cash(t, n, f, gap)
        tks[name] = tk
        curves[name] = mark_to_market(tk, closes, grid, gap)
    main_c, main_t = curves["6 x 16.7%"], tks["6 x 16.7%"]
    main_t.to_csv(OUT / "trades.csv", index=False)
    main_c.rename("equity").to_csv(OUT / "equity_4h.csv")

    p = r4.pivot(index="timestamp", columns="base", values="close").sort_index().ffill()
    p.index = p.index + BAR
    p = p[p.index >= START]
    basket = (1 + p.pct_change().mean(axis=1).fillna(0)).cumprod().reindex(grid).ffill().fillna(1.0)
    curves["alts vs BTC basket"] = basket

    seg = {name: {k: seg_stats(c, a, b) for k, (a, b) in SEGMENTS.items()} for name, c in curves.items()}
    st = risk_stats(main_c, main_t)
    openpos = open_positions(main_t, grid)
    m = main_c.resample("M").last()
    m = (m / m.shift(1, fill_value=1.0) - 1)[lambda x: x.index <= pd.Timestamp("2026-09-30", tz="UTC")]
    mb = basket.resample("M").last()
    mb = (mb / mb.shift(1, fill_value=1.0) - 1).reindex(m.index)
    yearly = main_c.resample("Y").last()
    yearly = 100 * (yearly / yearly.shift(1, fill_value=1.0) - 1)

    # ---------------------------------------------------------------- figure 1: equity / dd / exposure
    fig, axes = plt.subplots(3, 1, figsize=(18, 13), sharex=True, gridspec_kw={"height_ratios": [3, 1.2, 1]})
    ax = axes[0]
    shade = {"new 17m (never used)": "#fff4e0", "separate year": "#eef3fa", "tuning window": "#ffffff"}
    for k, col in shade.items():
        a, b = SEGMENTS[k]
        ax.axvspan(a, b, color=col, zorder=0)
        ax.text(a + pd.Timedelta(days=6), 0.98, k, transform=ax.get_xaxis_transform(), va="top", fontsize=9, color="#444")
    style = {"6 x 16.7%": (GREEN, "-", 1.8), "6 x 16.7% + live gap": (GREEN, "--", 1.0),
             "10 x 10% (previous)": (BLUE, "-", 1.0), "alts vs BTC basket": (GREY, "-", 1.0)}
    for name, c in curves.items():
        col, ls, lw = style[name]
        ax.plot(c.index, 100 * (c - 1), color=col, ls=ls, lw=lw,
                label=f"{name}: {100 * (c.iloc[-1] - 1):+.0f}%  (max DD {seg[name]['ALL 36m']['max_dd_pct']:.1f}%)")
    ax.set_yscale("symlog", linthresh=100)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("cumulative return % (BTC, symlog)")
    ax.set_title(f"HTF4H 3-year backtest — x16 4h score, S>=0.65, SL25/TP30/5d, max 6 open x 16.7%, 2h coin block, "
                 f"no weekend block | CAGR {st['cagr_pct']:.0f}%, Sharpe {st['sharpe_daily_ann']:.2f}, "
                 f"{st['trades']} trades, win {100 * st['win_rate']:.0f}%, PF {st['profit_factor']:.2f}")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.93), fontsize=9)
    ax.grid(alpha=0.3)
    dd = main_c / main_c.cummax() - 1
    ax = axes[1]
    ax.fill_between(dd.index, 100 * dd, 0, color=RED, alpha=0.35)
    ax.plot(dd.index, 100 * dd, color=RED, lw=0.8)
    ax.annotate(f"{100 * dd.min():.1f}%", (dd.idxmin(), 100 * dd.min()), xytext=(8, 0), textcoords="offset points")
    ax.set_ylabel("drawdown %")
    ax.set_title(f"Drawdown (6 x 16.7%) — longest under water {st['longest_underwater_days']:.0f} days")
    ax.grid(alpha=0.3)
    ax = axes[2]
    ax.fill_between(openpos.index, openpos.to_numpy(), step="post", color=BLUE, alpha=0.4)
    ax.set_ylim(0, SLOTS + 0.5)
    ax.set_ylabel("open positions")
    ax.set_title(f"Exposure — average {openpos.mean():.1f} of {SLOTS} open ({100 * openpos.mean() / SLOTS:.0f}% of capital)")
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    fig.tight_layout()
    fig.savefig(OUT / "3y_1_equity.png", dpi=125)
    plt.close(fig)

    # ---------------------------------------------------------------- figure 2: monthly / yearly / segments
    fig, axes = plt.subplots(2, 2, figsize=(20, 10), gridspec_kw={"width_ratios": [2.6, 1]})
    ax = axes[0][0]
    x = np.arange(len(m))
    ax.bar(x - 0.2, 100 * m.to_numpy(), width=0.4, color=np.where(m >= 0, GREEN, RED), label="strategy")
    ax.bar(x + 0.2, 100 * mb.to_numpy(), width=0.4, color=GREY, label="alts vs BTC basket")
    for i, v in enumerate(100 * m.to_numpy()):
        ax.text(i - 0.2, v + (0.8 if v >= 0 else -0.8), f"{v:+.0f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels([d.strftime("%y-%m") for d in m.index], rotation=60, fontsize=8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(f"Monthly return % — {int((m > 0).sum())}/{len(m)} positive")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    ax = axes[0][1]
    ax.scatter(100 * mb, 100 * m, color=np.where(m >= 0, GREEN, RED), s=35)
    k = np.polyfit(100 * mb, 100 * m, 1)
    xs = np.linspace(100 * mb.min(), 100 * mb.max(), 10)
    ax.plot(xs, np.polyval(k, xs), color="black", ls="--", lw=1)
    ax.axhline(0, color="black", lw=0.6)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("alt basket vs BTC, month %")
    ax.set_ylabel("strategy, month %")
    ax.set_title(f"Strategy vs alt market (corr {np.corrcoef(mb, m)[0, 1]:.2f}, slope {k[0]:.2f})")
    ax.grid(alpha=0.3)
    ax = axes[1][0]
    names = list(SEGMENTS)
    xs = np.arange(len(names))
    for j, (name, col) in enumerate(zip(["6 x 16.7%", "6 x 16.7% + live gap", "10 x 10% (previous)", "alts vs BTC basket"],
                                        [GREEN, "#7fbf8f", BLUE, GREY])):
        vals = [seg[name][k]["return_pct"] for k in names]
        ax.bar(xs + (j - 1.5) * 0.2, vals, width=0.2, color=col, label=name)
        for xi, v, k2 in zip(xs, vals, names):
            ax.text(xi + (j - 1.5) * 0.2, v, f"{v:+.0f}%\nDD{seg[name][k2]['max_dd_pct']:.0f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=6.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(names)
    ax.set_yscale("symlog", linthresh=100)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Return and max DD by period")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    ax = axes[1][1]
    ax.bar([str(d.year) for d in yearly.index], yearly.to_numpy(), color=np.where(yearly >= 0, GREEN, RED))
    for i, v in enumerate(yearly.to_numpy()):
        ax.text(i, v, f"{v:+.0f}%", ha="center", va="bottom" if v >= 0 else "top")
    ax.set_title("Calendar-year return % (2023 = Oct-Dec, 2026 = Jan-Sep)")
    ax.axhline(0, color="black", lw=0.8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "3y_2_monthly.png", dpi=125)
    plt.close(fig)

    # ---------------------------------------------------------------- figure 3: trades / coins / rolling
    tk = main_t
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    ax = axes[0]
    v = 100 * tk["net"]
    bins = np.arange(-27, 32, 1)
    ax.hist(v[v > 0], bins=bins, color=GREEN)
    ax.hist(v[v <= 0], bins=bins, color=RED)
    rs = tk["reason"].value_counts()
    ax.set_title(f"Trade net % — mean {v.mean():+.2f}, median {v.median():+.2f}\n"
                 + ", ".join(f"{k} {n}" for k, n in rs.items()))
    ax.grid(alpha=0.3)
    ax = axes[1]
    pnl = (100 * tk["size"] * tk["net"]).groupby(tk["base"]).sum().sort_values()
    ax.barh(pnl.index, pnl.to_numpy(), color=np.where(pnl >= 0, GREEN, RED))
    ax.set_title("P&L by coin (sum of size x net, % of starting equity)")
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(alpha=0.3, axis="x")
    ax = axes[2]
    roll = main_c / main_c.shift(6 * 90) - 1
    ax.plot(roll.index, 100 * roll, color=BLUE, lw=1)
    ax.fill_between(roll.index, 100 * roll, 0, where=roll >= 0, color=GREEN, alpha=0.25)
    ax.fill_between(roll.index, 100 * roll, 0, where=roll < 0, color=RED, alpha=0.25)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(f"Rolling 90-day return % — negative {100 * (roll < 0).mean():.0f}% of the time")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "3y_3_trades.png", dpi=125)
    plt.close(fig)

    out = {"overlap_check": chk, "risk_stats_6x16.7": st, "segments": seg,
           "monthly_pct": {d.strftime("%Y-%m"): round(100 * v, 2) for d, v in m.items()},
           "yearly_pct": {str(d.year): round(v, 2) for d, v in yearly.items()},
           "signals": int(len(t)), "coins_with_trades": int(tk["base"].nunique())}
    (OUT / "summary.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    pd.set_option("display.width", 220)
    print(json.dumps({"overlap_check": chk, "yearly_pct": out["yearly_pct"], "signals": out["signals"]}, indent=1))
    print(pd.DataFrame({(n, k): v for n, d in seg.items() for k, v in d.items()}).T.to_string())
    print(json.dumps(st, indent=1, default=float))
    print(" ".join(f"{k}:{v:+.1f}" for k, v in out["monthly_pct"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
