"""HTF4H step 5 — tight trailing exits on the 4h signals, both periods, monthly.

Signals: scaled 4h S>=0.65 NEW_CROSS (frozen). Periods:
  IS   2026-03-26 -> 2026-09-24 (tuning window, step 1 panel, cached 1m rel)
  OOS  2025-03-01 -> 2026-02-28 (htf_4h.oos_1y panel, data/oos_2025_candles)
Exits: SL {10, 20}% x activation {2,3,4,5,8,10}% x trail {0.5,1,1.5,2,3}% (trail < activation), 14d max,
plus the step-4 reference exits. Portfolio: 10 slots x 10%, weekend block ON (frozen), live gap shown.

    python -m htf_4h.step5_tight_trail

Writes: results/htf_4h/step5/
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

from htf_4h.common import BAR, RESULTS, load_coin, new_cross_signals
from htf_4h.oos_1y import LOAD_START, OOS_END, OOS_START
from htf_4h.oos_download import OOS_DIR
from htf_4h.step3_exits import DAY_NS, MAX_HOLD_D, REL1M, RT_FEE, _eval_trade, _label, exit_grid
from htf_4h.step4_portfolio import EXITS, LIVE_GAP, mark_to_market, simulate, stats

OUT = RESULTS / "step5"
THRESHOLD = 0.65
SLOTS = 10
PERIODS = {
    "IS": {"panel": RESULTS / "step1" / "s_panel_4h_h150.parquet", "rel4": RESULTS / "step2" / "rel4.parquet",
           "start": pd.Timestamp("2026-03-26", tz="UTC")},
    "OOS": {"panel": RESULTS / "oos_1y" / "s_panel.parquet", "rel4": RESULTS / "oos_1y" / "rel4.parquet",
            "start": OOS_START},
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("htf4h.step5")


def tight_grid() -> list[dict]:
    g = []
    for sl, act, dist in product((0.10, 0.20), (0.02, 0.03, 0.04, 0.05, 0.08, 0.10), (0.005, 0.01, 0.015, 0.02, 0.03)):
        if dist >= act:
            continue
        g.append({"family": "TIGHT", "sl": sl, "tp": None, "act": act, "dist": dist, "sdrop": None, "hold_d": MAX_HOLD_D})
    g += [c for c in exit_grid() if c["label"] in EXITS]
    for k, c in enumerate(g):
        c["cfg"] = k
        c["label"] = _label(c)
    return g


def _short(c: dict) -> str:
    if c["family"] == "TIGHT":
        return f"SL{c['sl']*100:g} +{c['act']*100:g}/{c['dist']*100:g}"
    return EXITS[c["label"]]


def _rel1m(period: str, base: str) -> pd.DataFrame | None:
    if period == "IS":
        r = pd.read_parquet(REL1M, filters=[("base", "==", base)], columns=["timestamp", "high", "low", "close"])
        if not r.empty:
            return r
        d = load_coin(base)
    else:
        d = load_coin(base, OOS_DIR, LOAD_START, OOS_END)
    return None if d is None else d["rel1m"]


def _coin_task(period: str, base: str, sig: pd.DataFrame, s_coin: pd.DataFrame, grid: list[dict]) -> list[dict]:
    r = _rel1m(period, base)
    if r is None or r.empty:
        return []
    r = r.sort_values("timestamp")
    ts_all = pd.to_datetime(r["timestamp"], utc=True).astype("int64").to_numpy()
    hi_all, lo_all, cl_all = (r[k].to_numpy(float) for k in ("high", "low", "close"))
    data_end_ns = int(ts_all[-1])
    levels = sorted({c["sdrop"] for c in grid if c["sdrop"] is not None})
    rows = []
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
        after = s_coin[s_coin["close_ts"] > e.entry_utc]
        sdrop_ns = {lv: int(after.loc[after["S"] < lv, "close_ts"].iloc[0].value)
                    for lv in levels if (after["S"] < lv).any()}
        for c in grid:
            res = _eval_trade(ts, hi, lo, cl, cmax, neg_cmin, entry, entry_ns, data_end_ns, sdrop_ns, c)
            if res is None:
                continue
            i, px, reason = res
            rows.append({"period": period, "base": base, "entry_utc": e.entry_utc, "entry_px": entry, "S": float(e.S),
                         "cfg": c["cfg"], "net": px / entry - 1 - RT_FEE, "reason": reason,
                         "hold_d": (int(ts[i]) - entry_ns) / DAY_NS})
    return rows


def walk(grid: list[dict], workers: int = 18) -> pd.DataFrame:
    path = OUT / "trades.parquet"
    if path.exists():
        t = pd.read_parquet(path)
        t["entry_utc"] = pd.to_datetime(t["entry_utc"], utc=True)
        return t
    jobs = []
    for period, p in PERIODS.items():
        s = pd.read_parquet(p["panel"])
        for col in ("timestamp", "close_ts"):
            s[col] = pd.to_datetime(s[col], utc=True)
        s = s[s["variant"] == "scaled"]
        sig = new_cross_signals(s, THRESHOLD)
        sig = sig[sig["entry_utc"] >= p["start"]]
        for b, g in sig.groupby("base"):
            jobs.append((period, b, g[["base", "entry_utc", "entry_px", "S"]], s[s["base"] == b][["close_ts", "S"]]))
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_coin_task, pr, b, g, sc, grid): (pr, b) for pr, b, g, sc in jobs}
        for n, fut in enumerate(as_completed(futs), 1):
            rows += fut.result()
            if n % 10 == 0 or n == len(futs):
                log.info("walked %d/%d %.0fs", n, len(futs), time.time() - t0)
    t = pd.DataFrame(rows)
    t.to_parquet(path, index=False)
    return t


def portfolio(t: pd.DataFrame, grid: list[dict]) -> pd.DataFrame:
    names = {c["cfg"]: _short(c) for c in grid}
    t = t.copy()
    t["exit_utc"] = t["entry_utc"] + pd.to_timedelta(t["hold_d"], unit="D")
    t["exit_px"] = t["entry_px"] * (1 + t["net"] + RT_FEE)
    t = t.sort_values(["entry_utc", "S"], ascending=[True, False]).reset_index(drop=True)
    rows = []
    for period, p in PERIODS.items():
        r4 = pd.read_parquet(p["rel4"])
        r4["timestamp"] = pd.to_datetime(r4["timestamp"], utc=True)
        closes = {b: g.set_index(g["timestamp"] + BAR)["close"] for b, g in r4.groupby("base")}
        tp = t[t["period"] == period]
        grid_ts = pd.date_range(p["start"], tp["exit_utc"].max().ceil("4h"), freq="4h")
        for c in grid:
            sub = tp[tp["cfg"] == c["cfg"]]
            for gap in (0.0, LIVE_GAP):
                tk, info = simulate(sub, SLOTS, True, gap)
                curve = mark_to_market(tk, closes, grid_ts, gap)
                st = stats(tk, info, curve)
                rows.append({"period": period, "cfg": c["cfg"], "exit": names[c["cfg"]], "family": c["family"],
                             "sl": c["sl"], "act": c["act"], "dist": c["dist"], "live_gap": gap > 0,
                             "avg_hold_d": tk["hold_d"].mean(), **st})
    return pd.DataFrame(rows)


def heatmap(res: pd.DataFrame, order: list[str], path, title: str) -> None:
    mcols = sorted([c for c in res.columns if c.startswith("m_")])
    blocks = []
    for period in ("OOS", "IS"):
        x = res[(res["period"] == period) & ~res["live_gap"]].set_index("exit").reindex(order)
        cols = [c for c in mcols if x[c].notna().any()]
        blocks.append((period, x[cols + ["return_pct", "max_dd_pct"]]))
    fig, axes = plt.subplots(1, 2, figsize=(26, 0.32 * len(order) + 2.5),
                             gridspec_kw={"width_ratios": [blocks[0][1].shape[1], blocks[1][1].shape[1]]})
    for ax, (period, x) in zip(axes, blocks):
        v = x.to_numpy(float)
        months = v[:, :-2]
        im = ax.imshow(np.clip(months, -10, 10), cmap="RdYlGn", vmin=-10, vmax=10, aspect="auto")
        tot = np.full_like(v, np.nan)
        tot[:, -2:] = v[:, -2:]
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                if np.isnan(v[i, j]):
                    continue
                bold = "bold" if j >= v.shape[1] - 2 else "normal"
                ax.text(j, i, f"{v[i, j]:.1f}", ha="center", va="center", fontsize=7, fontweight=bold)
        ax.set_xticks(range(v.shape[1]))
        ax.set_xticklabels([c.replace("m_", "") for c in x.columns[:-2]] + ["TOTAL", "MAX DD"], rotation=60, fontsize=8)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order if period == "OOS" else [""] * len(order), fontsize=8)
        ax.axvline(v.shape[1] - 2.5, color="black", lw=1.5)
        ax.set_title(f"{period}: monthly return % (10 slots x 10%, weekend block ON)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 280)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 200)
    grid = tight_grid()
    log.info("configs=%d", len(grid))
    t = walk(grid)
    res = portfolio(t, grid)
    res.to_csv(OUT / "portfolio_results.csv", index=False)

    base = res[~res["live_gap"]]
    piv = base.pivot(index="exit", columns="period", values=["return_pct", "max_dd_pct", "trades", "win_rate", "pf", "avg_hold_d"])
    gap = res[res["live_gap"]].pivot(index="exit", columns="period", values="return_pct").add_suffix("_gap")
    tab = pd.concat([piv["return_pct"].add_prefix("ret_"), piv["max_dd_pct"].add_prefix("dd_"),
                     piv["win_rate"].add_prefix("win_"), piv["pf"].add_prefix("pf_"),
                     piv["avg_hold_d"]["OOS"].rename("hold_d_OOS"), piv["trades"]["OOS"].rename("trades_OOS"),
                     gap], axis=1)
    tab["worst_ret"] = tab[["ret_IS", "ret_OOS"]].min(axis=1)
    tab = tab.sort_values("ret_OOS", ascending=False)
    tab.to_csv(OUT / "comparison.csv")
    print("== all exits, sorted by OOS return (10 slots, weekend ON) ==")
    print(tab.round(2).to_string())

    mcols = sorted([c for c in res.columns if c.startswith("m_")])
    order = tab.index.tolist()
    heatmap(res, order, OUT / "monthly_all.png", "HTF4H step 5 — all exits, sorted by OOS total return")
    top = tab.head(10).index.tolist() + [e for e in EXITS.values() if e not in tab.head(10).index]
    for period in ("OOS", "IS"):
        x = base[base["period"] == period].set_index("exit").reindex(top)
        cols = [c for c in mcols if x[c].notna().any()]
        print(f"\n== monthly %, {period}: top 10 by OOS + references ==")
        print(x[cols + ["return_pct", "max_dd_pct"]].round(1).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
