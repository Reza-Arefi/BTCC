"""HTF4H step 4 — portfolio simulation of the 4h signals with the step-3 exit candidates.

Rules (mirrors live, adapted to 4h): signals in entry-time order, one position per coin, 2h same-coin
block after a close, N slots each sized at equity/N of REALIZED equity at entry (whole account split
evenly; 20 x 10% would need 200% on spot), optional weekend entry block FRI 12:00 -> MON 00:00 UTC.
Exits/returns come from step 3 (1m walk). Equity is marked to market on every 4h close for drawdown.

    python -m htf_4h.step4_portfolio

Writes: results/htf_4h/step4/
"""

from __future__ import annotations

import heapq
import logging
from itertools import product

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from binance_btc_bot.strategy.entry_blackout import EntryBlackout
from htf_4h.common import BAR, FEE_PER_SIDE, RESULTS, new_cross_signals

OUT = RESULTS / "step4"
PANEL = RESULTS / "step1" / "s_panel_4h_h150.parquet"
TRADES = RESULTS / "step3" / "trades.parquet"
SUMMARY3 = RESULTS / "step3" / "summary.csv"
REL4 = RESULTS / "step2" / "rel4.parquet"
RT_FEE = 2 * FEE_PER_SIDE
LIVE_GAP = 0.0022
SAME_COIN_BLOCK = pd.Timedelta(hours=2)
WEEKEND = EntryBlackout.from_config({"enabled": True, "timezone": "UTC", "start": "FRI 12:00", "end": "MON 00:00"})
SETS = [("scaled", 0.60), ("scaled", 0.65), ("scaled", 0.70)]
SLOTS = (10, 15, 20)
EXITS = {
    "TRAIL SL20 A10/T5 14d": "Trail SL20 +10/5 14d",
    "TRAIL SL20 A10/T5 7d": "Trail SL20 +10/5 7d",
    "TRAIL SL25 A20/T5 14d": "Trail SL25 +20/5 14d",
    "FIX SL25 TP30 5d": "SL25 TP30 5d",
    "SDROP SL10 A10/T5 S<0 14d": "SL10 trail, exit S<0",
    "SDROP SL20 A10/T5 S<0.35 14d": "SL20 trail, exit S<0.35",
    "LIVE SL5 A4/T0.5 14d": "Live 15m exit",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("htf4h.step4")


def load_trades() -> pd.DataFrame:
    s = pd.read_parquet(PANEL)
    for col in ("timestamp", "close_ts"):
        s[col] = pd.to_datetime(s[col], utc=True)
    px = []
    for v, thr in SETS:
        sg = new_cross_signals(s[s["variant"] == v], thr)
        sg["set"] = f"{v}_{thr:.2f}"
        px.append(sg[["set", "base", "entry_utc", "entry_px", "S"]])
    px = pd.concat(px, ignore_index=True)
    lab = pd.read_csv(SUMMARY3)[["cfg", "label"]].drop_duplicates()
    lab = lab[lab["label"].isin(EXITS)]
    t = pd.read_parquet(TRADES)
    t["entry_utc"] = pd.to_datetime(t["entry_utc"], utc=True)
    t = t.merge(lab, on="cfg").merge(px, on=["set", "base", "entry_utc"])
    t["exit_utc"] = t["entry_utc"] + pd.to_timedelta(t["hold_d"], unit="D")
    t["exit_px"] = t["entry_px"] * (1 + t["net"] + RT_FEE)
    t["exit"] = t["label"].map(EXITS)
    return t.sort_values(["entry_utc", "S"], ascending=[True, False]).reset_index(drop=True)


def simulate(t: pd.DataFrame, slots: int, blackout: bool, extra_cost: float = 0.0,
             coin_block: pd.Timedelta = SAME_COIN_BLOCK) -> tuple[pd.DataFrame, dict]:
    equity = 1.0
    frac = 1.0 / slots
    heap: list[tuple] = []
    open_coins: dict[str, int] = {}
    last_close: dict[str, pd.Timestamp] = {}
    taken, skip = [], {"slots_full": 0, "coin_open": 0, "coin_block": 0, "weekend": 0}
    max_open = 0
    for k, e in enumerate(t.itertuples()):
        while heap and heap[0][0] <= e.entry_utc:
            ex_ts, _, base, pnl = heapq.heappop(heap)
            equity += pnl
            open_coins.pop(base, None)
            last_close[base] = ex_ts
        if blackout and WEEKEND.is_blocked(e.entry_utc.to_pydatetime()):
            skip["weekend"] += 1
            continue
        if e.base in open_coins:
            skip["coin_open"] += 1
            continue
        if e.base in last_close and e.entry_utc - last_close[e.base] < coin_block:
            skip["coin_block"] += 1
            continue
        if len(heap) >= slots:
            skip["slots_full"] += 1
            continue
        size = equity * frac
        net = e.net - extra_cost
        heapq.heappush(heap, (e.exit_utc, k, e.base, size * net))
        open_coins[e.base] = k
        max_open = max(max_open, len(heap))
        taken.append({"base": e.base, "entry_utc": e.entry_utc, "exit_utc": e.exit_utc, "entry_px": e.entry_px,
                      "exit_px": e.exit_px, "size": size, "net": net, "reason": e.reason, "hold_d": e.hold_d,
                      "S": e.S})
    while heap:
        _, _, _, pnl = heapq.heappop(heap)
        equity += pnl
    tk = pd.DataFrame(taken)
    return tk, {"final_equity": equity, "max_open": max_open, **{f"skip_{a}": b for a, b in skip.items()}}


def mark_to_market(tk: pd.DataFrame, closes: dict[str, pd.Series], grid: pd.DatetimeIndex, extra_cost: float) -> pd.Series:
    """Equity on every 4h close: realized P&L + open positions valued at the 4h close (after exit fee)."""
    eq = np.ones(len(grid))
    gi = grid.asi8
    for r in tk.itertuples():
        i_exit = np.searchsorted(gi, r.exit_utc.value, side="left")
        eq[i_exit:] += r.size * r.net
        i_in = np.searchsorted(gi, r.entry_utc.value, side="right")
        if i_in < i_exit:
            c = closes[r.base].reindex(grid[i_in:i_exit]).ffill().to_numpy()
            c = np.where(np.isnan(c), r.entry_px, c)
            eq[i_in:i_exit] += r.size * (c / r.entry_px - 1 - RT_FEE - extra_cost)
    return pd.Series(eq, index=grid)


def stats(tk: pd.DataFrame, info: dict, curve: pd.Series) -> dict:
    dd = curve / curve.cummax() - 1
    m = curve.resample("M").last()
    m = m / m.shift(1, fill_value=1.0) - 1
    gp, gl = tk.loc[tk["net"] > 0, "net"].sum(), -tk.loc[tk["net"] < 0, "net"].sum()
    days = (curve.index[-1] - curve.index[0]).total_seconds() / 86400
    avg_open = (tk["hold_d"].sum()) / days
    return {"return_pct": 100 * (curve.iloc[-1] - 1), "max_dd_pct": 100 * dd.min(), "trades": len(tk),
            "win_rate": float((tk["net"] > 0).mean()), "pf": gp / gl if gl else np.nan,
            "avg_net_pct": 100 * tk["net"].mean(), "avg_open": avg_open, **info,
            "pos_months": int((m > 0).sum()), "months": len(m),
            **{f"m_{d:%y-%m}": 100 * v for d, v in m.items()}}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    t = load_trades()
    r4 = pd.read_parquet(REL4)
    r4["timestamp"] = pd.to_datetime(r4["timestamp"], utc=True)
    closes = {b: g.set_index(g["timestamp"] + BAR)["close"] for b, g in r4.groupby("base")}
    start = t["entry_utc"].min().floor("D")
    grid = pd.date_range(start, t["exit_utc"].max().ceil("4h"), freq="4h")

    bench = r4[r4["timestamp"] + BAR >= start].groupby("base")["close"].agg(["first", "last"])
    bench_ret = 100 * float((bench["last"] / bench["first"] - 1).mean())

    rows, curves = [], {}
    for (v, thr), ex, n, bo in product(SETS, EXITS.values(), SLOTS, (True, False)):
        st = f"{v}_{thr:.2f}"
        sub = t[(t["set"] == st) & (t["exit"] == ex)]
        for gap in ((0.0, LIVE_GAP) if bo else (0.0,)):
            tk, info = simulate(sub, n, bo, gap)
            curve = mark_to_market(tk, closes, grid, gap)
            rows.append({"set": st, "exit": ex, "slots": n, "weekend_block": bo, "live_gap": gap > 0,
                         **stats(tk, info, curve)})
            curves[(st, ex, n, bo, gap > 0)] = curve
            if (st, n, bo, gap) == ("scaled_0.65", 20, True, 0.0):
                tk.to_csv(OUT / f"trades_{ex.replace(' ', '_').replace('/', '-').replace('<', 'lt')}.csv", index=False)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "portfolio_results.csv", index=False)

    cols = ["set", "exit", "slots", "return_pct", "max_dd_pct", "trades", "win_rate", "pf", "avg_net_pct", "avg_open",
            "max_open", "skip_slots_full", "skip_coin_open", "skip_weekend", "pos_months"]
    base = res[res["weekend_block"] & ~res["live_gap"]]
    print(f"Equal-weight buy&hold of all coins vs BTC over the window: {bench_ret:+.1f}%\n")
    print("== weekend block ON, no live gap — scaled 0.65 ==")
    print(base[base["set"] == "scaled_0.65"].sort_values(["exit", "slots"])[cols].round(3).to_string(index=False))
    print("\n== best 12 overall (weekend ON, no gap) by return/|DD| ==")
    base = base.assign(ret_dd=base["return_pct"] / base["max_dd_pct"].abs())
    print(base.sort_values("ret_dd", ascending=False)[cols + ["ret_dd"]].head(12).round(3).to_string(index=False))
    print("\n== weekend block effect (return / DD), scaled 0.65 ==")
    w = res[(res["set"] == "scaled_0.65") & ~res["live_gap"]].pivot_table(
        index=["exit", "slots"], columns="weekend_block", values=["return_pct", "max_dd_pct"])
    print(w.round(1).to_string())
    print("\n== with live execution gap (-0.22pp/trade), weekend ON, scaled 0.65 ==")
    g = res[(res["set"] == "scaled_0.65") & res["live_gap"]]
    print(g.sort_values(["exit", "slots"])[["exit", "slots", "return_pct", "max_dd_pct", "trades"]].round(2).to_string(index=False))
    mcols = [c for c in res.columns if c.startswith("m_")]
    print("\n== monthly %, scaled 0.65, 20 slots, weekend ON, no gap ==")
    mm = res[(res["set"] == "scaled_0.65") & (res["slots"] == 20) & res["weekend_block"] & ~res["live_gap"]]
    print(mm[["exit"] + mcols].round(2).to_string(index=False))

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    for ex, col in zip(EXITS.values(), plt.cm.tab10(np.arange(len(EXITS)))):
        c = curves[("scaled_0.65", ex, 20, True, False)]
        axes[0].plot(c.index, 100 * (c - 1), color=col, lw=1.3, label=ex)
        axes[1].plot(c.index, 100 * (c / c.cummax() - 1), color=col, lw=1)
    axes[0].set_title("Scaled S>=0.65, 20 slots x 5%, weekend block ON — equity (mark-to-market every 4h, BTC terms)")
    axes[0].set_ylabel("return %")
    axes[1].set_ylabel("drawdown %")
    axes[0].legend(fontsize=8)
    for a in axes:
        a.grid(alpha=0.3)
        a.axhline(0, color="black", lw=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "step4_equity.png", dpi=130)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
