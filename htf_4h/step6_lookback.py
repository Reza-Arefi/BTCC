"""HTF4H step 6 — momentum look-back length of the 4h score, winner exit only.

Score = live E2 score on 4h ALT/BTC bars; the E2 momentum horizons are defined in hours and converted to
bars through ``interval``. Passing interval X on 4h bars stretches every momentum horizon by 4h/X:
  4h x1 (4h..1d)  2h x2  1h x4  30m x8  15m x16 (current "scaled", 4h..16d)  5m x48 (12h..48d)
Trend / volume / volatility / RSI / structure stay in 4h bars (unchanged).

Continuous data 2024-12 -> 2026-09-24 (data/oos_2025_candles + main store), history 400 bars for every
variant so all share the same scored window. Periods reported:
  OOS 2025-03-01 -> 2026-02-28   IS 2026-03-01 -> 2026-09-24   ALL = both

Exit: SL 25% / TP 30% / 5d max (the out-of-sample winner).
Portfolio: 10 slots x 10% of realized equity, one position per coin, same-coin block 2 DAYS after a close,
NO weekend block. Old rules (2h block + weekend block) shown for the current x16 score as reference.

    python -m htf_4h.step6_lookback

Writes: results/htf_4h/step6/
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from btcc.factors.btc_regime import btc_regime_factor
from btcc.factors.structure import structure_factor
from btcc.factors.trend import trend_factor
from btcc.factors.volume import volume_factor
from btcc.research.altbtc_breakout.data import DEFAULT_CANDLE_DIR
from btcc.research.early_breakout_entry import score_variant
from htf_4h.common import BAR, RESULTS, load_coin, bases, weights
from htf_4h.oos_download import OOS_DIR
from htf_4h.step3_exits import DAY_NS, RT_FEE, _eval_trade
from htf_4h.step4_portfolio import LIVE_GAP, mark_to_market, simulate, stats

OUT = RESULTS / "step6"
DIRS = (OOS_DIR, DEFAULT_CANDLE_DIR)
LOAD_START = pd.Timestamp("2024-12-01 00:00", tz="UTC")
LOAD_END = pd.Timestamp("2026-09-24 23:00", tz="UTC")
HISTORY = 400
SCALES = {"x1": "4h", "x2": "2h", "x4": "1h", "x8": "30m", "x16": "15m", "x48": "5m"}
CURRENT = "x16"
THRESHOLDS = (0.55, 0.60, 0.65, 0.70, 0.75)
MAIN_THR = 0.65
EXIT = {"family": "FIX", "sl": 0.25, "tp": 0.30, "act": None, "dist": None, "sdrop": None, "hold_d": 5}
HOLD_NS = EXIT["hold_d"] * DAY_NS
SLOTS = 10
NEW_BLOCK = pd.Timedelta(days=2)
OLD_BLOCK = pd.Timedelta(hours=2)
PERIODS = {
    "OOS": (pd.Timestamp("2025-03-01", tz="UTC"), pd.Timestamp("2026-03-01", tz="UTC")),
    "IS": (pd.Timestamp("2026-03-01", tz="UTC"), pd.Timestamp("2026-09-25", tz="UTC")),
    "ALL": (pd.Timestamp("2025-03-01", tz="UTC"), pd.Timestamp("2026-09-25", tz="UTC")),
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("htf4h.step6")


def _signals(s: pd.DataFrame, thr: float) -> pd.DataFrame:
    zone = s["S"] >= thr
    contiguous = s["timestamp"].diff() == BAR
    prev = zone.shift(1, fill_value=False) & contiguous
    return s[zone & ~prev & contiguous & s["next_open"].notna()]


def _coin(base: str) -> dict | None:
    d = load_coin(base, DIRS, LOAD_START, LOAD_END)
    if d is None or len(d["rel4"]) <= HISTORY:
        return None
    w = weights()
    r4, a4, b4 = d["rel4"], d["alt4"], d["btc4"]
    rows = []
    for i in range(HISTORY - 1, len(r4)):
        lo = i + 1 - HISTORY
        ab, au, bu = r4.iloc[lo : i + 1], a4.iloc[lo : i + 1], b4.iloc[lo : i + 1]
        shared = {"trend": trend_factor(ab), "volu": volume_factor(au, ab), "struct": structure_factor(ab),
                  "regime": btc_regime_factor(bu, None, {}, "4h")}
        ts = r4["timestamp"].iloc[i]
        nxt = i + 1 < len(r4) and r4["timestamp"].iloc[i + 1] - ts == BAR
        row = {"base": base, "timestamp": ts, "close_ts": ts + BAR,
               "next_open": float(r4["open"].iloc[i + 1]) if nxt else np.nan}
        for name, iv in SCALES.items():
            row[name] = float(score_variant(alt_btc=ab, alt_usdt=au, btc_usdt=bu, weights=w, interval=iv,
                                            use_e2=True, shared=shared)["S"])
        rows.append(row)
    s = pd.DataFrame(rows)

    rel = d["rel1m"].sort_values("timestamp")
    ts_all = rel["timestamp"].astype("int64").to_numpy()
    hi_all, lo_all, cl_all = (rel[k].to_numpy(float) for k in ("high", "low", "close"))
    data_end_ns = int(ts_all[-1])
    trades = []
    for name in SCALES:
        sv = s[["timestamp", "close_ts", "next_open"]].assign(S=s[name])
        for thr in THRESHOLDS:
            for e in _signals(sv, thr).itertuples():
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
                trades.append({"scale": name, "threshold": thr, "base": base, "entry_utc": e.close_ts,
                               "entry_px": entry, "S": float(e.S), "net": px / entry - 1 - RT_FEE, "reason": reason,
                               "hold_d": (int(ts_all[i0 + k]) - entry_ns) / DAY_NS})
    r4o = r4[["timestamp", "close"]].assign(base=base)
    return {"s": s, "trades": pd.DataFrame(trades), "rel4": r4o}


def build(workers: int = 18):
    paths = {k: OUT / f"{k}.parquet" for k in ("s", "trades", "rel4")}
    if all(p.exists() for p in paths.values()):
        out = {k: pd.read_parquet(p) for k, p in paths.items()}
    else:
        parts: dict[str, list] = {k: [] for k in paths}
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_coin, b): b for b in bases()}
            for n, fut in enumerate(as_completed(futs), 1):
                res = fut.result()
                log.info("%s done (%d/%d) %.0fs%s", futs[fut], n, len(futs), time.time() - t0, "" if res else " NO DATA")
                if res:
                    for k in parts:
                        parts[k].append(res[k])
        out = {k: pd.concat(v, ignore_index=True) for k, v in parts.items()}
        for k, p in paths.items():
            out[k].to_parquet(p, index=False)
    for df in out.values():
        for col in ("timestamp", "close_ts", "entry_utc"):
            if col in df:
                df[col] = pd.to_datetime(df[col], utc=True)
    return out["s"], out["trades"], out["rel4"]


def run_portfolios(t: pd.DataFrame, r4: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    t["exit_utc"] = t["entry_utc"] + pd.to_timedelta(t["hold_d"], unit="D")
    t["exit_px"] = t["entry_px"] * (1 + t["net"] + RT_FEE)
    t = t.sort_values(["entry_utc", "S"], ascending=[True, False]).reset_index(drop=True)
    closes = {b: g.set_index(g["timestamp"] + BAR)["close"] for b, g in r4.groupby("base")}
    rows = []
    rules = [("new", NEW_BLOCK, False)]
    for (scale, thr), sub in t.groupby(["scale", "threshold"]):
        for period, (a, b) in PERIODS.items():
            p = sub[(sub["entry_utc"] >= a) & (sub["entry_utc"] < b)]
            grid = pd.date_range(a, max(b, p["exit_utc"].max()).ceil("4h"), freq="4h")
            cur_rules = rules + ([("old", OLD_BLOCK, True)] if scale == CURRENT else [])
            for rule, block, bo in cur_rules:
                for gap in (0.0, LIVE_GAP):
                    tk, info = simulate(p, SLOTS, bo, gap, block)
                    curve = mark_to_market(tk, closes, grid, gap)
                    rows.append({"scale": scale, "threshold": thr, "period": period, "rules": rule,
                                 "live_gap": gap > 0, "signals": len(p), **stats(tk, info, curve)})
    return pd.DataFrame(rows)


def entry_edge(s: pd.DataFrame, r4: pd.DataFrame) -> pd.DataFrame:
    """Mean 5d forward ALT/BTC return of each signal minus all coins entered at the same 4h open."""
    c = r4.pivot(index="timestamp", columns="base", values="close").sort_index()
    fwd = c.shift(-30) / c - 1
    base_fwd = fwd.mean(axis=1)
    rows = []
    for name in SCALES:
        sv = s[["base", "timestamp", "close_ts", "next_open"]].assign(S=s[name]).sort_values(["base", "timestamp"])
        for thr in THRESHOLDS:
            sig = pd.concat([_signals(g, thr) for _, g in sv.groupby("base")])
            for period, (a, b) in PERIODS.items():
                x = sig[(sig["close_ts"] >= a) & (sig["close_ts"] < b)]
                bar = x["timestamp"]
                r = np.array([fwd.at[t0, bs] if t0 in fwd.index else np.nan for t0, bs in zip(bar, x["base"])])
                bl = base_fwd.reindex(bar).to_numpy()
                e = pd.Series(r - bl).dropna()
                rows.append({"scale": name, "threshold": thr, "period": period, "signals": len(x),
                             "excess_5d_pct": 100 * e.mean(), "t": e.mean() / (e.std(ddof=1) / np.sqrt(len(e)))})
    return pd.DataFrame(rows)


def plots(res: pd.DataFrame) -> None:
    x = res[(res["rules"] == "new") & ~res["live_gap"]]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, period in zip(axes, PERIODS):
        p = x[x["period"] == period].pivot(index="scale", columns="threshold", values="return_pct").reindex(list(SCALES))
        v = p.to_numpy(float)
        lim = np.nanmax(np.abs(v))
        im = ax.imshow(v, cmap="RdYlGn", vmin=-lim, vmax=lim, aspect="auto")
        dd = x[x["period"] == period].pivot(index="scale", columns="threshold", values="max_dd_pct").reindex(list(SCALES))
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                ax.text(j, i, f"{v[i, j]:+.0f}%\nDD {dd.iloc[i, j]:.0f}", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(p.shape[1]))
        ax.set_xticklabels([f"S>={c}" for c in p.columns])
        ax.set_yticks(range(p.shape[0]))
        ax.set_yticklabels([f"{k} ({SCALES[k]})" for k in p.index])
        ax.set_title(f"{period}: portfolio return / max DD")
        plt.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle("HTF4H step 6 — momentum look-back scale x threshold (SL25 TP30 5d, 10x10%, 2-day coin block, no weekend block)")
    fig.tight_layout()
    fig.savefig(OUT / "step6_lookback.png", dpi=130)
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 280)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 300)
    s, t, r4 = build()
    log.info("S rows=%d coins=%d %s -> %s; trades=%d", len(s), s["base"].nunique(), s["close_ts"].min(),
             s["close_ts"].max(), len(t))
    res = run_portfolios(t, r4)
    res.to_csv(OUT / "portfolio_results.csv", index=False)
    edge = entry_edge(s, r4)
    edge.to_csv(OUT / "entry_edge.csv", index=False)
    plots(res)

    new = res[(res["rules"] == "new") & ~res["live_gap"]]
    cols = ["return_pct", "max_dd_pct", "trades", "win_rate", "pf", "pos_months"]
    tab = new.pivot_table(index=["scale", "threshold"], columns="period", values=cols)
    tab = tab.reindex(columns=["OOS", "IS", "ALL"], level=1)
    print("== new rules (2-day block, no weekend block), SL25 TP30 5d, 10 x 10% ==")
    print(tab[["return_pct", "max_dd_pct", "trades"]].round(1).to_string())
    print(tab[["win_rate", "pf", "pos_months"]].round(2).to_string())
    print("\n== with live gap 0.22pp/trade, return % ==")
    g = res[(res["rules"] == "new") & res["live_gap"]].pivot_table(index=["scale", "threshold"], columns="period",
                                                                    values="return_pct")
    print(g[["OOS", "IS", "ALL"]].round(1).to_string())
    print("\n== rule change on the current x16 score (old = 2h block + weekend block) ==")
    o = res[(res["scale"] == CURRENT) & ~res["live_gap"]].pivot_table(index=["threshold", "rules"], columns="period",
                                                                      values=["return_pct", "max_dd_pct"])
    print(o.round(1).to_string())
    print("\n== entry edge: 5d excess vs all coins at the same bar (%, t) ==")
    e = edge.pivot_table(index=["scale", "threshold"], columns="period", values=["excess_5d_pct", "t", "signals"])
    print(e.round(2).to_string())
    mcols = sorted(c for c in res.columns if c.startswith("m_"))
    print("\n== monthly %, ALL period, S>=0.65, new rules ==")
    m = new[(new["period"] == "ALL") & (new["threshold"] == MAIN_THR)].set_index("scale").reindex(list(SCALES))
    print(m[[c for c in mcols if m[c].notna().any()] + ["return_pct", "max_dd_pct"]].round(1).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
