"""HTF4H out-of-sample year: 2025-03-01 -> 2026-02-28 (data from htf_4h.oos_download, no overlap with tuning).

Frozen from the in-sample study: scaled 4h score (history 150 bars), NEW_CROSS at S>=0.65 (0.60/0.70 shown),
the seven step-4 exits, portfolio rules from step 4 (one per coin, 2h block, N slots x equity/N, weekend
block on/off, live gap 0.22pp). Nothing is re-tuned here.

    python -m htf_4h.oos_1y

Writes: results/htf_4h/oos_1y/
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import htf_4h.step2_entry as step2
from htf_4h.common import BAR, RESULTS, bases, load_coin, new_cross_signals, score_coin_4h
from htf_4h.oos_download import OOS_DIR
from htf_4h.step3_exits import DAY_NS, MAX_HOLD_D, RT_FEE, _eval_trade, exit_grid
from htf_4h.step4_portfolio import EXITS, LIVE_GAP, mark_to_market, simulate, stats

OUT = RESULTS / "oos_1y"
LOAD_START = pd.Timestamp("2025-02-01 00:00", tz="UTC")
OOS_START = pd.Timestamp("2025-03-01 00:00", tz="UTC")
OOS_END = pd.Timestamp("2026-02-28 23:59", tz="UTC")
HISTORY = 150
VARIANT = "scaled"
THRESHOLDS = (0.60, 0.65, 0.70)
MAIN_THR = 0.65
SLOTS = (10, 20)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("htf4h.oos")


def _coin(base: str, grid: list[dict]) -> dict | None:
    s = score_coin_4h(base, VARIANT, HISTORY, OOS_DIR, LOAD_START, OOS_END)
    d = load_coin(base, OOS_DIR, LOAD_START, OOS_END)
    if s.empty or d is None:
        return None
    rel = d["rel1m"].sort_values("timestamp")
    ts_all = rel["timestamp"].astype("int64").to_numpy()
    hi_all, lo_all, cl_all = (rel[k].to_numpy(float) for k in ("high", "low", "close"))
    data_end_ns = int(ts_all[-1])
    levels = sorted({c["sdrop"] for c in grid if c["sdrop"] is not None})
    rows = []
    for thr in THRESHOLDS:
        sig = new_cross_signals(s, thr)
        sig = sig[sig["entry_utc"] >= OOS_START]
        for e in sig.itertuples():
            entry_ns = int(e.entry_utc.value)
            i0 = int(np.searchsorted(ts_all, entry_ns, side="left"))
            i1 = int(np.searchsorted(ts_all, entry_ns + MAX_HOLD_D * DAY_NS, side="left"))
            if i0 >= i1:
                continue
            ts, hi, lo, cl = ts_all[i0:i1], hi_all[i0:i1], lo_all[i0:i1], cl_all[i0:i1]
            entry = float(e.entry_px)
            cmax = np.maximum.accumulate(np.maximum(hi, entry))
            neg_cmin = -np.minimum.accumulate(np.minimum(lo, entry))
            after = s[s["close_ts"] > e.entry_utc]
            sdrop_ns = {}
            for lv in levels:
                below = after.loc[after["S"] < lv, "close_ts"]
                if len(below):
                    sdrop_ns[lv] = int(below.iloc[0].value)
            for c in grid:
                res = _eval_trade(ts, hi, lo, cl, cmax, neg_cmin, entry, entry_ns, data_end_ns, sdrop_ns, c)
                if res is None:
                    continue
                i, px, reason = res
                rows.append({"set": f"{VARIANT}_{thr:.2f}", "base": base, "entry_utc": e.entry_utc, "entry_px": entry,
                             "S": float(e.S), "label": c["label"], "net": px / entry - 1 - RT_FEE, "reason": reason,
                             "hold_d": (int(ts[i]) - entry_ns) / DAY_NS})
    r4 = d["rel4"][["timestamp", "open", "high", "low", "close"]].assign(base=base)
    return {"s": s, "rel4": r4, "trades": pd.DataFrame(rows)}


def build(workers: int = 18) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = [OUT / f"{k}.parquet" for k in ("s_panel", "rel4", "trades")]
    if all(p.exists() for p in paths):
        out = [pd.read_parquet(p) for p in paths]
    else:
        grid = [c for c in exit_grid() if c["label"] in EXITS]
        parts: dict[str, list] = {"s": [], "rel4": [], "trades": []}
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_coin, b, grid): b for b in bases()}
            for n, fut in enumerate(as_completed(futs), 1):
                res = fut.result()
                log.info("%s done (%d/%d) %.0fs%s", futs[fut], n, len(futs), time.time() - t0, "" if res else " NO DATA")
                if res:
                    for k in parts:
                        parts[k].append(res[k])
        out = [pd.concat(parts[k], ignore_index=True) for k in ("s", "rel4", "trades")]
        for df, p in zip(out, paths):
            df.to_parquet(p, index=False)
    for df in out:
        for col in ("timestamp", "close_ts", "entry_utc"):
            if col in df:
                df[col] = pd.to_datetime(df[col], utc=True)
    return tuple(out)


def entry_quality(s: pd.DataFrame, r4: pd.DataFrame) -> pd.DataFrame:
    step2.WINDOW_START = OOS_START
    f = step2.forward_table(r4.sort_values(["base", "timestamp"]).reset_index(drop=True))
    xs = step2.baseline_same_bar(f)
    rows = []
    for thr in THRESHOLDS:
        sg = new_cross_signals(s, thr)
        sg = sg[sg["entry_utc"] >= OOS_START]
        m = sg[["base", "entry_utc"]].merge(f, on=["base", "entry_utc"]).merge(xs, on="entry_utc")
        for h in ("12h", "1d", "3d", "5d", "7d", "10d"):
            x = m[f"ret_{h}"].dropna()
            e = (m[f"ret_{h}"] - m[f"base_ret_{h}"]).dropna()
            rows.append({"threshold": thr, "horizon": h, "n": len(x), "mean_net_pct": 100 * (x.mean() - RT_FEE),
                         "median_net_pct": 100 * (x.median() - RT_FEE), "excess_pct": 100 * e.mean(),
                         "excess_t": e.mean() / (e.std(ddof=1) / np.sqrt(len(e))),
                         "all_bars_net_pct": 100 * (f[f"ret_{h}"].mean() - RT_FEE)})
    return pd.DataFrame(rows)


def per_trade(t: pd.DataFrame) -> pd.DataFrame:
    g = t.groupby(["set", "label"])
    out = g["net"].agg(n="size", mean_net_pct=lambda x: 100 * x.mean(), median_net_pct=lambda x: 100 * x.median(),
                       win_rate=lambda x: (x > 0).mean(),
                       pf=lambda x: x[x > 0].sum() / -x[x < 0].sum()).reset_index()
    out["hold_d"] = g["hold_d"].mean().to_numpy()
    out["exit"] = out["label"].map(EXITS)
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    s, r4, t = build()
    s = s[s["variant"] == VARIANT]
    log.info("S rows=%d coins=%d  %s -> %s", len(s), s["base"].nunique(), s["close_ts"].min(), s["close_ts"].max())

    eq = entry_quality(s, r4)
    eq.to_csv(OUT / "entry_quality.csv", index=False)
    pt = per_trade(t)
    pt.to_csv(OUT / "per_trade.csv", index=False)

    t = t.copy()
    t["exit"] = t["label"].map(EXITS)
    t["exit_utc"] = t["entry_utc"] + pd.to_timedelta(t["hold_d"], unit="D")
    t["exit_px"] = t["entry_px"] * (1 + t["net"] + RT_FEE)
    t = t.sort_values(["entry_utc", "S"], ascending=[True, False]).reset_index(drop=True)
    closes = {b: g.set_index(g["timestamp"] + BAR)["close"] for b, g in r4.groupby("base")}
    grid = pd.date_range(OOS_START, t["exit_utc"].max().ceil("4h"), freq="4h")

    rows, curves = [], {}
    for thr, ex, n, bo in product(THRESHOLDS, EXITS.values(), SLOTS, (True, False)):
        st = f"{VARIANT}_{thr:.2f}"
        sub = t[(t["set"] == st) & (t["exit"] == ex)]
        for gap in ((0.0, LIVE_GAP) if bo else (0.0,)):
            tk, info = simulate(sub, n, bo, gap)
            curve = mark_to_market(tk, closes, grid, gap)
            rows.append({"set": st, "exit": ex, "slots": n, "weekend_block": bo, "live_gap": gap > 0,
                         **stats(tk, info, curve)})
            curves[(st, ex, n, bo, gap > 0)] = curve
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "portfolio_results.csv", index=False)

    p = r4[r4["timestamp"] >= OOS_START].pivot(index="timestamp", columns="base", values="close").ffill()
    basket = (p / p.bfill().iloc[0]).mean(axis=1)
    bm = basket.resample("M").last()
    bm = 100 * (bm / bm.shift(1, fill_value=1.0) - 1)

    print(f"window {OOS_START:%Y-%m-%d} -> {OOS_END:%Y-%m-%d}, coins {s['base'].nunique()}")
    print(f"equal-weight alt basket vs BTC: {100*(basket.iloc[-1]-1):+.1f}%  max DD {100*(basket/basket.cummax()-1).min():.1f}%")
    print("\n== entry quality (vs all coins entered at the same 4h bar) ==")
    print(eq.round(3).to_string(index=False))
    print("\n== per trade (no portfolio) ==")
    print(pt.sort_values(["set", "mean_net_pct"], ascending=[True, False]).round(3).to_string(index=False))
    cols = ["set", "exit", "slots", "return_pct", "max_dd_pct", "trades", "win_rate", "pf", "avg_net_pct",
            "max_open", "skip_slots_full", "skip_weekend", "pos_months", "months"]
    for bo in (True, False):
        print(f"\n== portfolio, weekend block {'ON' if bo else 'OFF'}, no live gap ==")
        x = res[(res["weekend_block"] == bo) & ~res["live_gap"]]
        print(x.sort_values(["set", "exit", "slots"])[cols].round(2).to_string(index=False))
    print("\n== live gap (-0.22pp/trade), weekend ON, S>=0.65 ==")
    x = res[res["live_gap"] & (res["set"] == f"{VARIANT}_{MAIN_THR:.2f}")]
    print(x.sort_values(["exit", "slots"])[["exit", "slots", "return_pct", "max_dd_pct"]].round(2).to_string(index=False))
    mcols = [c for c in res.columns if c.startswith("m_")]
    print("\n== monthly %, S>=0.65, 10 slots, weekend ON, no gap (month = calendar month number) ==")
    mm = res[(res["set"] == f"{VARIANT}_{MAIN_THR:.2f}") & (res["slots"] == 10) & res["weekend_block"] & ~res["live_gap"]]
    print(mm[["exit"] + mcols].round(2).to_string(index=False))
    print("basket   " + "  ".join(f"{d:%y-%m}:{v:+.1f}" for d, v in bm.items()))

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    for ex, col in zip(EXITS.values(), plt.cm.tab10(np.arange(len(EXITS)))):
        c = curves[(f"{VARIANT}_{MAIN_THR:.2f}", ex, 10, True, False)]
        axes[0].plot(c.index, 100 * (c - 1), color=col, lw=1.3, label=ex)
        axes[1].plot(c.index, 100 * (c / c.cummax() - 1), color=col, lw=1)
    axes[0].plot(basket.index, 100 * (basket - 1), color="grey", lw=1, ls="--", label="alt basket buy&hold")
    axes[0].set_title("OUT-OF-SAMPLE 2025-03 -> 2026-02 — scaled S>=0.65, 10 slots x 10%, weekend block ON (BTC terms)")
    axes[0].set_ylabel("return %")
    axes[1].set_ylabel("drawdown %")
    axes[0].legend(fontsize=8)
    for a in axes:
        a.grid(alpha=0.3)
        a.axhline(0, color="black", lw=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "oos_equity.png", dpi=130)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
