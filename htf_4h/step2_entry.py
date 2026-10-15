"""HTF4H step 2 — entry quality on 4h, independent of any exit.

For every NEW_CROSS signal (both variants, thresholds 0.55-0.75) entered at the next 4h open:
  forward ALT/BTC return after 4h .. 10d, MFE (best high) and MAE (worst low) over the same window,
  versus a baseline of entering at EVERY 4h bar of the same coin in the same month.
Entries sit on 4h boundaries, so every window is a whole number of 4h bars and the 4h high/low
give the exact 1m extremes. Ordering of SL/TP touches is left to step 3 (1m walk).

    python -m htf_4h.step2_entry

Writes: results/htf_4h/step2/
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from htf_4h.common import BAR, FEE_PER_SIDE, RESULTS, THRESHOLD, VARIANTS, bases, load_coin, new_cross_signals

OUT = RESULTS / "step2"
PANEL = RESULTS / "step1" / "s_panel_4h_h150.parquet"
REL4 = OUT / "rel4.parquet"
WINDOW_START = pd.Timestamp("2026-03-26 00:00", tz="UTC")
HORIZONS = {"4h": 1, "12h": 3, "1d": 6, "2d": 12, "3d": 18, "5d": 30, "7d": 42, "10d": 60}
THRESHOLDS = (0.55, 0.60, 0.65, 0.70, 0.75)
RT_FEE = 2 * FEE_PER_SIDE
MFE_LEVELS = (0.05, 0.10, 0.15, 0.20, 0.30)
MAE_LEVELS = (0.05, 0.10, 0.15, 0.20, 0.25)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("htf4h.step2")


def _rel4(base: str) -> pd.DataFrame:
    d = load_coin(base)
    if d is None:
        return pd.DataFrame()
    r = d["rel4"][["timestamp", "open", "high", "low", "close"]].copy()
    r["base"] = base
    return r


def load_rel4(workers: int = 18) -> pd.DataFrame:
    if REL4.exists():
        r = pd.read_parquet(REL4)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            r = pd.concat([x for x in ex.map(_rel4, bases()) if not x.empty], ignore_index=True)
        r.to_parquet(REL4, index=False)
    r["timestamp"] = pd.to_datetime(r["timestamp"], utc=True)
    return r.sort_values(["base", "timestamp"]).reset_index(drop=True)


def forward_table(r: pd.DataFrame) -> pd.DataFrame:
    """Forward stats for an entry at the OPEN of every 4h bar (row = entry bar)."""
    out = []
    for base, g in r.groupby("base", sort=False):
        ts = g["timestamp"].to_numpy()
        o, h, l, c = (g[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        n = len(g)
        res = {"base": base, "entry_utc": g["timestamp"].to_numpy(), "entry_px": o}
        for name, hb in HORIZONS.items():
            ret = np.full(n, np.nan)
            mfe = np.full(n, np.nan)
            mae = np.full(n, np.nan)
            for k in range(n - hb + 1):
                if ts[k + hb - 1] - ts[k] != np.timedelta64(int((hb - 1) * 4), "h"):
                    continue
                ret[k] = c[k + hb - 1] / o[k] - 1
                mfe[k] = h[k : k + hb].max() / o[k] - 1
                mae[k] = l[k : k + hb].min() / o[k] - 1
            res[f"ret_{name}"] = ret
            res[f"mfe_{name}"] = mfe
            res[f"mae_{name}"] = mae
        out.append(pd.DataFrame(res))
    f = pd.concat(out, ignore_index=True)
    f["entry_utc"] = pd.to_datetime(f["entry_utc"], utc=True)
    f = f[f["entry_utc"] >= WINDOW_START].reset_index(drop=True)
    f["month"] = f["entry_utc"].dt.strftime("%Y-%m")
    return f


def baseline_same_bar(f: pd.DataFrame) -> pd.DataFrame:
    """Mean forward return of all coins entered at the same 4h open (no look-ahead on the coin)."""
    cols = [f"ret_{h}" for h in HORIZONS]
    return f.groupby("entry_utc")[cols].mean().add_prefix("base_").reset_index()


def baseline_by_coin_month(f: pd.DataFrame) -> pd.DataFrame:
    """Biased: the month mean includes the run-up that produced the signal. Reported for reference only."""
    cols = [f"ret_{h}" for h in HORIZONS]
    return f.groupby(["base", "month"])[cols].mean().add_prefix("cm_").reset_index()


def attach(sig: pd.DataFrame, f: pd.DataFrame, base_xs: pd.DataFrame, base_cm: pd.DataFrame) -> pd.DataFrame:
    m = sig[["variant", "base", "entry_utc", "S"]].merge(f, on=["base", "entry_utc"], how="left")
    m = m.merge(base_xs, on="entry_utc", how="left").merge(base_cm, on=["base", "month"], how="left")
    for h in HORIZONS:
        m[f"net_{h}"] = m[f"ret_{h}"] - RT_FEE
        m[f"exc_{h}"] = m[f"ret_{h}"] - m[f"base_ret_{h}"]
        m[f"exc_cm_{h}"] = m[f"ret_{h}"] - m[f"cm_ret_{h}"]
    return m


def summarize(m: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (v, thr), g in m.groupby(["variant", "threshold"]):
        for h in HORIZONS:
            x = g[f"ret_{h}"].dropna()
            e = g[f"exc_{h}"].dropna()
            if len(x) < 5:
                continue
            rows.append({
                "variant": v, "threshold": thr, "horizon": h, "n": len(x),
                "mean_net_pct": 100 * (x.mean() - RT_FEE), "median_net_pct": 100 * (x.median() - RT_FEE),
                "win_rate_net": float((x > RT_FEE).mean()),
                "baseline_pct": 100 * g.loc[x.index, f"base_ret_{h}"].mean(),
                "excess_pct": 100 * e.mean(), "excess_t": e.mean() / (e.std(ddof=1) / np.sqrt(len(e))) if len(e) > 2 else np.nan,
                "excess_vs_coin_month_pct": 100 * g[f"exc_cm_{h}"].mean(),
                "mfe_med_pct": 100 * g[f"mfe_{h}"].median(), "mae_med_pct": 100 * g[f"mae_{h}"].median(),
            })
    for h in HORIZONS:
        x = f[f"ret_{h}"].dropna()
        rows.append({"variant": "ALL BARS", "threshold": np.nan, "horizon": h, "n": len(x),
                     "mean_net_pct": 100 * (x.mean() - RT_FEE), "median_net_pct": 100 * (x.median() - RT_FEE),
                     "win_rate_net": float((x > RT_FEE).mean()), "baseline_pct": 100 * x.mean(), "excess_pct": 0.0,
                     "excess_t": np.nan, "mfe_med_pct": 100 * f[f"mfe_{h}"].median(),
                     "mae_med_pct": 100 * f[f"mae_{h}"].median()})
    return pd.DataFrame(rows)


def excursions(m: pd.DataFrame, f: pd.DataFrame, horizon: str) -> pd.DataFrame:
    rows = []
    groups = [(f"{v} S>={THRESHOLD}", g) for v, g in m[m["threshold"] == THRESHOLD].groupby("variant")]
    groups.append(("ALL BARS", f))
    for name, g in groups:
        mfe = g[f"mfe_{horizon}"].dropna()
        mae = g[f"mae_{horizon}"].dropna()
        row = {"group": name, "horizon": horizon, "n": len(mfe)}
        for q in (0.25, 0.5, 0.75, 0.9):
            row[f"mfe_p{int(q*100)}"] = 100 * mfe.quantile(q)
            row[f"mae_p{int(q*100)}"] = 100 * mae.quantile(1 - q)
        for lv in MFE_LEVELS:
            row[f"P(mfe>={int(lv*100)}%)"] = float((mfe >= lv).mean())
        for lv in MAE_LEVELS:
            row[f"P(mae<=-{int(lv*100)}%)"] = float((mae <= -lv).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def monthly(m: pd.DataFrame, f: pd.DataFrame, horizons=("1d", "3d", "5d", "10d")) -> pd.DataFrame:
    g = m[m["threshold"] == THRESHOLD]
    rows = []
    for (v, mo), x in g.groupby(["variant", "month"]):
        row = {"variant": v, "month": mo, "n": len(x)}
        for h in horizons:
            row[f"net_{h}_pct"] = 100 * x[f"net_{h}"].mean()
            row[f"exc_{h}_pct"] = 100 * x[f"exc_{h}"].mean()
        rows.append(row)
    for mo, x in f.groupby("month"):
        row = {"variant": "ALL BARS", "month": mo, "n": len(x)}
        for h in horizons:
            row[f"net_{h}_pct"] = 100 * (x[f"ret_{h}"].mean() - RT_FEE)
            row[f"exc_{h}_pct"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def bar_ranges(r: pd.DataFrame) -> pd.DataFrame:
    r = r[r["timestamp"] >= WINDOW_START].copy()
    r["range_pct"] = 100 * (r["high"] / r["low"] - 1)
    d = r.set_index("timestamp").groupby("base").resample("1D").agg({"high": "max", "low": "min"}).dropna()
    d["range_pct"] = 100 * (d["high"] / d["low"] - 1)
    out = pd.DataFrame({
        "range4h_median_pct": r.groupby("base")["range_pct"].median(),
        "range4h_p90_pct": r.groupby("base")["range_pct"].quantile(0.9),
        "range1d_median_pct": d.groupby("base")["range_pct"].median(),
    }).sort_values("range4h_median_pct", ascending=False)
    return out


def plots(summ: pd.DataFrame, m: pd.DataFrame, f: pd.DataFrame, mon: pd.DataFrame) -> None:
    hs = list(HORIZONS)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for ax, v in zip(axes[0], VARIANTS):
        for thr, col in zip(THRESHOLDS, plt.cm.viridis(np.linspace(0, 0.9, len(THRESHOLDS)))):
            s = summ[(summ["variant"] == v) & (summ["threshold"] == thr)].set_index("horizon").reindex(hs)
            ax.plot(hs, s["excess_pct"], marker="o", color=col, label=f"S>={thr} (n={int(s['n'].iloc[0])})")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"{v}: mean excess return vs all coins entered at the same 4h bar")
        ax.set_ylabel("% (signal - baseline)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    ax = axes[1][0]
    for v, col in zip(VARIANTS, ("#1f77b4", "#d62728")):
        g = m[(m["variant"] == v) & (m["threshold"] == THRESHOLD)]
        ax.scatter(100 * g["mae_5d"], 100 * g["mfe_5d"], s=12, alpha=0.6, color=col, label=f"{v} S>=0.65")
    ax.set_xlabel("MAE over 5d (%)")
    ax.set_ylabel("MFE over 5d (%)")
    ax.axvline(-25, color="grey", ls=":", lw=1)
    ax.set_title("Best vs worst excursion within 5 days of entry")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1][1]
    p = mon[mon["variant"] != "ALL BARS"].pivot(index="month", columns="variant", values="net_5d_pct")
    p["ALL BARS"] = mon[mon["variant"] == "ALL BARS"].set_index("month")["net_5d_pct"]
    p.plot(kind="bar", ax=ax, color=["#1f77b4", "#d62728", "grey"])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Mean 5-day net return per signal by month (S>=0.65)")
    ax.set_ylabel("%")
    ax.grid(alpha=0.3, axis="y")
    fig.suptitle("HTF4H step 2 — 4h entry quality")
    fig.tight_layout()
    fig.savefig(OUT / "step2_entry.png", dpi=130)
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    s = pd.read_parquet(PANEL)
    for col in ("timestamp", "close_ts"):
        s[col] = pd.to_datetime(s[col], utc=True)
    r = load_rel4()
    f = forward_table(r)
    base_xs = baseline_same_bar(f)
    base_cm = baseline_by_coin_month(f)
    parts = []
    for thr in THRESHOLDS:
        sg = new_cross_signals(s, thr)
        x = attach(sg, f, base_xs, base_cm)
        x["threshold"] = thr
        parts.append(x)
    m = pd.concat(parts, ignore_index=True)
    m.to_csv(OUT / "signal_forward.csv", index=False)

    summ = summarize(m, f)
    summ.to_csv(OUT / "summary.csv", index=False)
    exc = pd.concat([excursions(m, f, h) for h in ("1d", "3d", "5d", "10d")], ignore_index=True)
    exc.to_csv(OUT / "excursions.csv", index=False)
    mon = monthly(m, f)
    mon.to_csv(OUT / "monthly.csv", index=False)
    rng = bar_ranges(r)
    rng.to_csv(OUT / "bar_ranges.csv")
    plots(summ, m, f, mon)

    print("== summary (S>=0.65 + all bars) ==")
    print(summ[(summ["threshold"] == THRESHOLD) | (summ["variant"] == "ALL BARS")].round(3).to_string(index=False))
    print("\n== excess by threshold, 1d/3d/5d/10d ==")
    print(summ[summ["horizon"].isin(["1d", "3d", "5d", "10d"]) & (summ["variant"] != "ALL BARS")]
          .pivot_table(index=["variant", "threshold"], columns="horizon", values=["n", "excess_pct", "excess_t"])
          .round(2).to_string())
    print("\n== excursions ==")
    print(exc.round(3).to_string(index=False))
    print("\n== monthly ==")
    print(mon.round(2).to_string(index=False))
    print("\n== bar ranges (median over coins) ==")
    print(rng.median().round(2).to_string())
    print(rng.round(2).head(8).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
