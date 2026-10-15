"""HTF4H step 3 — exit grid on 1m ALT/BTC for the 4h signals (per signal, no portfolio yet).

Signal sets: scaled S>=0.60/0.65/0.70 and literal S>=0.65 (NEW_CROSS, entry at next 4h open).
Exit families (fees 0.1%/side, SL wins when SL and TP/activation touch in the same 1m bar,
trail stop uses the peak including the current 1m high — same rules as _fast_trail_walk):
  FIX    SL x TP x max hold
  TRAIL  SL x activation x trail distance x max hold
  SDROP  SL + exit at the first 4h close with S < L (optional trail), max hold 14d
  LIVE   the live 15m TRN4 exit (SL 5%, act 4%, trail 0.5%) for reference

    python -m htf_4h.step3_exits

Writes: results/htf_4h/step3/
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

from htf_4h.common import FEE_PER_SIDE, MAX_SL, RESULTS, load_coin, new_cross_signals

OUT = RESULTS / "step3"
PANEL = RESULTS / "step1" / "s_panel_4h_h150.parquet"
REL1M = RESULTS.parent / "candidate_trn4_wknd_local_7m" / "rel1m.parquet"
RT_FEE = 2 * FEE_PER_SIDE
DAY_NS = 86_400 * 10**9
MAX_HOLD_D = 14
SETS = [("scaled", 0.60), ("scaled", 0.65), ("scaled", 0.70), ("literal", 0.65)]
MAIN_SET = "scaled_0.65"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("htf4h.step3")


def exit_grid() -> list[dict]:
    g = []
    for sl, tp, hd in product((0.08, 0.10, 0.12, 0.15, 0.20, 0.25), (0.10, 0.15, 0.20, 0.30, 0.40, None), (3, 5, 7, 10)):
        g.append({"family": "FIX", "sl": sl, "tp": tp, "act": None, "dist": None, "sdrop": None, "hold_d": hd})
    for sl, act, dist, hd in product((0.10, 0.15, 0.20, 0.25), (0.05, 0.08, 0.10, 0.15, 0.20), (0.03, 0.05, 0.07, 0.10), (7, 14)):
        g.append({"family": "TRAIL", "sl": sl, "tp": None, "act": act, "dist": dist, "sdrop": None, "hold_d": hd})
    for sl, lv, tr in product((0.10, 0.15, 0.20, 0.25), (0.0, 0.2, 0.35, 0.5), (None, (0.10, 0.05))):
        g.append({"family": "SDROP", "sl": sl, "tp": None, "act": tr[0] if tr else None, "dist": tr[1] if tr else None,
                  "sdrop": lv, "hold_d": MAX_HOLD_D})
    g.append({"family": "LIVE", "sl": 0.05, "tp": None, "act": 0.04, "dist": 0.005, "sdrop": None, "hold_d": MAX_HOLD_D})
    for k, c in enumerate(g):
        c["cfg"] = k
        c["label"] = _label(c)
    assert all(c["sl"] <= MAX_SL for c in g)
    return g


def _label(c: dict) -> str:
    p = [c["family"], f"SL{c['sl']*100:g}"]
    if c["tp"] is not None:
        p.append(f"TP{c['tp']*100:g}")
    if c["act"] is not None:
        p.append(f"A{c['act']*100:g}/T{c['dist']*100:g}")
    if c["sdrop"] is not None:
        p.append(f"S<{c['sdrop']:g}")
    p.append(f"{c['hold_d']}d")
    return " ".join(p)


def _eval_trade(ts, hi, lo, cl, cmax, neg_cmin, entry, entry_ns, data_end_ns, sdrop_ns: dict, c: dict):
    n = len(ts)
    cap_ns = entry_ns + c["hold_d"] * DAY_NS
    if c["sdrop"] is not None:
        cap_ns = min(cap_ns, sdrop_ns.get(c["sdrop"], cap_ns))
    end = int(np.searchsorted(ts, cap_ns, side="left"))
    if end == 0:
        return None
    sl_px = entry * (1 - c["sl"])
    i_sl = int(np.searchsorted(neg_cmin, -sl_px, side="left"))
    if c["tp"] is not None:
        i_tp = int(np.searchsorted(cmax, entry * (1 + c["tp"]), side="left"))
        if i_sl < end and i_sl <= i_tp:
            return i_sl, sl_px, "SL"
        if i_tp < end:
            return i_tp, entry * (1 + c["tp"]), "TP"
    elif c["act"] is not None:
        ia = int(np.searchsorted(cmax, entry * (1 + c["act"]), side="left"))
        if i_sl < end and i_sl <= ia:
            return i_sl, sl_px, "SL"
        if ia < end:
            stop = np.maximum(sl_px, cmax[ia:end] * (1 - c["dist"]))
            hit = np.flatnonzero(lo[ia:end] <= stop)
            if len(hit):
                j = int(hit[0])
                return ia + j, float(stop[j]), "TRAIL"
    elif i_sl < end:
        return i_sl, sl_px, "SL"
    reason = "SDROP" if c["sdrop"] is not None and cap_ns < entry_ns + c["hold_d"] * DAY_NS else "TIME"
    if end == n and ts[-1] < cap_ns - 60 * 10**9 and cap_ns > data_end_ns:
        reason = "END_OF_DATA"
    return end - 1, float(cl[end - 1]), reason


def _coin_task(base: str, sig: pd.DataFrame, s_coin: pd.DataFrame, grid: list[dict]) -> list[tuple]:
    r = pd.read_parquet(REL1M, filters=[("base", "==", base)], columns=["timestamp", "high", "low", "close"])
    if r.empty:
        d = load_coin(base)
        if d is None:
            return []
        r = d["rel1m"]
    r = r.sort_values("timestamp")
    ts_all = r["timestamp"].astype("int64").to_numpy()
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
        sv = s_coin[(s_coin["variant"] == e.variant) & (s_coin["close_ts"] > e.entry_utc)]
        sdrop_ns = {}
        for lv in levels:
            below = sv.loc[sv["S"] < lv, "close_ts"]
            if len(below):
                sdrop_ns[lv] = int(below.iloc[0].value)
        for c in grid:
            res = _eval_trade(ts, hi, lo, cl, cmax, neg_cmin, entry, entry_ns, data_end_ns, sdrop_ns, c)
            if res is None:
                continue
            i, px, reason = res
            rows.append((e.set, base, e.entry_utc, c["cfg"], px / entry - 1 - RT_FEE, reason,
                         (int(ts[i]) - entry_ns) / DAY_NS))
    return rows


def run(sig: pd.DataFrame, s: pd.DataFrame, grid: list[dict], workers: int = 18) -> pd.DataFrame:
    path = OUT / "trades.parquet"
    if path.exists():
        return pd.read_parquet(path)
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_coin_task, b, g, s[s["base"] == b][["variant", "close_ts", "S"]], grid): b
                for b, g in sig.groupby("base")}
        for n, fut in enumerate(as_completed(futs), 1):
            rows += fut.result()
            log.info("walked %s (%d/%d) %.0fs", futs[fut], n, len(futs), time.time() - t0)
    t = pd.DataFrame(rows, columns=["set", "base", "entry_utc", "cfg", "net", "reason", "hold_d"])
    t.to_parquet(path, index=False)
    return t


def summarize(t: pd.DataFrame, grid: list[dict]) -> pd.DataFrame:
    t = t.copy()
    t["month"] = pd.to_datetime(t["entry_utc"], utc=True).dt.strftime("%Y-%m")
    g = t.groupby(["set", "cfg"])
    out = g.agg(n=("net", "size"), mean_net_pct=("net", "mean"), median_net_pct=("net", "median"),
                win_rate=("net", lambda x: (x > 0).mean()), hold_d=("hold_d", "mean")).reset_index()
    pos = g["net"].apply(lambda x: x[x > 0].sum()).rename("gp")
    neg = g["net"].apply(lambda x: -x[x < 0].sum()).rename("gl")
    out = out.merge(pos.reset_index(), on=["set", "cfg"]).merge(neg.reset_index(), on=["set", "cfg"])
    out["pf"] = out["gp"] / out["gl"].replace(0, np.nan)
    mm = t.groupby(["set", "cfg", "month"])["net"].mean().reset_index()
    out = out.merge(mm.groupby(["set", "cfg"])["net"].agg(pos_months=lambda x: int((x > 0).sum()),
                                                         worst_month_pct="min").reset_index(), on=["set", "cfg"])
    rs = t.groupby(["set", "cfg"])["reason"].value_counts(normalize=True).unstack(fill_value=0).add_prefix("x_").reset_index()
    out = out.merge(rs, on=["set", "cfg"])
    out["mean_net_pct"] *= 100
    out["median_net_pct"] *= 100
    out["worst_month_pct"] *= 100
    out["pnl_per_day_pct"] = out["mean_net_pct"] / out["hold_d"]
    meta = pd.DataFrame(grid)[["cfg", "family", "label", "sl", "tp", "act", "dist", "sdrop", "hold_d"]]
    return out.drop(columns=["gp", "gl"]).merge(meta.rename(columns={"hold_d": "max_hold_d"}), on="cfg")


def plots(summ: pd.DataFrame) -> None:
    m = summ[summ["set"] == MAIN_SET]
    fig, axes = plt.subplots(2, 2, figsize=(17, 11))

    ax = axes[0][0]
    f = m[(m["family"] == "FIX") & (m["max_hold_d"] == 7)].copy()
    f["tp_lbl"] = f["tp"].map(lambda x: "none" if pd.isna(x) else f"{x*100:g}%")
    p = f.pivot(index="sl", columns="tp_lbl", values="mean_net_pct")
    p = p[[c for c in ("10%", "15%", "20%", "30%", "40%", "none") if c in p.columns]]
    _heat(ax, p, "FIX, 7d max hold: mean net % per trade (rows SL, cols TP)", [f"{x*100:g}%" for x in p.index])

    ax = axes[0][1]
    f = m[(m["family"] == "TRAIL") & (m["sl"] == 0.20) & (m["max_hold_d"] == 14)]
    p = f.pivot(index="act", columns="dist", values="mean_net_pct")
    p.columns = [f"trail {x*100:g}%" for x in p.columns]
    _heat(ax, p, "TRAIL, SL 20%, 14d max: mean net % (rows activation)", [f"act {x*100:g}%" for x in p.index])

    ax = axes[1][0]
    f = m[(m["family"] == "FIX") & m["tp"].isna()]
    for sl, col in zip(sorted(f["sl"].unique()), plt.cm.viridis(np.linspace(0, 0.9, f["sl"].nunique()))):
        x = f[f["sl"] == sl].sort_values("max_hold_d")
        ax.plot(x["max_hold_d"], x["mean_net_pct"], marker="o", color=col, label=f"SL {sl*100:g}%")
    ax.set_xlabel("max hold (days), no TP")
    ax.set_ylabel("mean net % per trade")
    ax.set_title("SL + time exit only")
    ax.axhline(0, color="black", lw=0.8)
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1][1]
    top = summ[summ["family"] != "LIVE"].sort_values("mean_net_pct", ascending=False)
    labels = top[top["set"] == MAIN_SET].head(12)["label"].tolist()
    live = summ[summ["family"] == "LIVE"]["label"].iloc[0]
    labels.append(live)
    sets = [f"{v}_{t:.2f}" for v, t in SETS]
    x = np.arange(len(labels))
    for k, (st, col) in enumerate(zip(sets, ("#9ecae1", "#d62728", "#fc9272", "#1f77b4"))):
        vals = summ[summ["set"] == st].set_index("label").reindex(labels)["mean_net_pct"]
        ax.bar(x + (k - 1.5) * 0.2, vals, width=0.2, color=col, label=st)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(f"Top 12 exits for {MAIN_SET} (+ live TRN4 exit) across signal sets")
    ax.set_ylabel("mean net % per trade")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.suptitle("HTF4H step 3 — exits on 4h signals (1m walk, per signal, no portfolio)")
    fig.tight_layout()
    fig.savefig(OUT / "step3_exits.png", dpi=130)
    plt.close(fig)


def _heat(ax, p: pd.DataFrame, title: str, ylabels: list[str]) -> None:
    v = p.to_numpy(float)
    lim = np.nanmax(np.abs(v))
    im = ax.imshow(v, cmap="RdYlGn", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(p.shape[1]))
    ax.set_xticklabels(p.columns)
    ax.set_yticks(range(p.shape[0]))
    ax.set_yticklabels(ylabels)
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            ax.text(j, i, f"{v[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.04)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    s = pd.read_parquet(PANEL)
    for col in ("timestamp", "close_ts"):
        s[col] = pd.to_datetime(s[col], utc=True)
    parts = []
    for v, thr in SETS:
        sg = new_cross_signals(s[s["variant"] == v], thr)
        sg["set"] = f"{v}_{thr:.2f}"
        parts.append(sg[["set", "variant", "base", "entry_utc", "entry_px", "S"]])
    sig = pd.concat(parts, ignore_index=True)
    grid = exit_grid()
    log.info("signals=%d configs=%d", len(sig), len(grid))
    t = run(sig, s, grid)
    summ = summarize(t, grid)
    summ.to_csv(OUT / "summary.csv", index=False)
    plots(summ)

    cols = ["set", "label", "n", "mean_net_pct", "median_net_pct", "win_rate", "pf", "hold_d", "pnl_per_day_pct",
            "pos_months", "worst_month_pct"] + [c for c in summ.columns if c.startswith("x_")]
    main_ = summ[summ["set"] == MAIN_SET].sort_values("mean_net_pct", ascending=False)
    print(f"== {MAIN_SET}: top 25 by mean net per trade ==")
    print(main_[cols].head(25).round(3).to_string(index=False))
    print("\n== best per family ==")
    print(main_.groupby("family").head(3)[cols].round(3).to_string(index=False))
    print("\n== live TRN4 exit on every set ==")
    print(summ[summ["family"] == "LIVE"][cols].round(3).to_string(index=False))
    print("\n== best 3 per signal set ==")
    print(summ.sort_values("mean_net_pct", ascending=False).groupby("set").head(3)[cols].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
