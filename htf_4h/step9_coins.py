"""HTF4H step 9 — which coins help / hurt over 3 years, and does filtering coins survive honest testing?

Setup frozen (step 8): x16 4h score, S>=0.65, SL25/TP30/5d, max 6 open x 1/6, 2h coin block, no weekend block.

Per coin: every signal (signal quality) and the trades the portfolio actually took (contribution).
Filters tested on the 6 x 16.7% portfolio:
  hindsight   drop coins whose 36-month mean net per signal < 0         (upper bound, look-ahead)
  cross-val   choose drops on one half, apply on the other half          (honest)
  rolling     skip a signal if the coin's PAST closed signals (last 365d, >= MIN_N) have mean net < 0 (honest, causal)

    python -m htf_4h.step9_coins

Writes: results/htf_4h/step9_coins/
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
from htf_4h.step4_portfolio import LIVE_GAP, mark_to_market
from htf_4h.step7_sizing import simulate_cash
from htf_4h.step8_3y import SEGMENTS, SIZE, SLOTS, START, THRESHOLD, build, seg_stats

OUT = RESULTS / "step9_coins"
HALF = pd.Timestamp("2025-04-01", tz="UTC")
ROLL_DAYS = 365
MIN_N = (3, 5, 8)
GREEN, RED, GREY, BLUE = "#1a7f37", "#cf222e", "#8c959f", "#0969da"


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    _, t_all, r4 = build()
    t = t_all[(t_all["threshold"] == THRESHOLD) & (t_all["entry_utc"] >= START)].copy()
    t["exit_utc"] = t["entry_utc"] + pd.to_timedelta(t["hold_d"], unit="D")
    t["exit_px"] = t["entry_px"] * (1 + t["net"] + RT_FEE)
    t["year"] = t["entry_utc"].dt.year
    t["segment"] = None
    for k, (a, b) in SEGMENTS.items():
        if k != "ALL 36m":
            t.loc[(t["entry_utc"] >= a) & (t["entry_utc"] < b), "segment"] = k
    return t.sort_values(["entry_utc", "S"], ascending=[True, False]).reset_index(drop=True), r4


def coin_table(t: pd.DataFrame, taken: pd.DataFrame) -> pd.DataFrame:
    g = t.groupby("base")["net"]
    tab = pd.DataFrame({
        "signals": g.size(), "mean_net_pct": 100 * g.mean(), "median_net_pct": 100 * g.median(),
        "win_rate": g.apply(lambda x: (x > 0).mean()),
        "pf": g.apply(lambda x: x[x > 0].sum() / max(-x[x < 0].sum(), 1e-9)),
        "t_stat": g.apply(lambda x: x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 else np.nan),
        "tp_hits": t[t["reason"] == "TP"].groupby("base").size(),
    }).fillna({"tp_hits": 0})
    seg = t.groupby(["base", "segment"])["net"].mean().unstack() * 100
    tab = tab.join(seg.add_prefix("seg: "))
    yr = t.groupby(["base", "year"])["net"].mean().unstack() * 100
    tab = tab.join(yr.add_prefix("y"))
    tab["neg_segments"] = (seg < 0).sum(axis=1)
    tab["portfolio_trades"] = taken.groupby("base").size()
    tab["portfolio_pnl_pct"] = 100 * (taken["size"] * taken["net"]).groupby(taken["base"]).sum()
    return tab.sort_values("portfolio_pnl_pct", ascending=False)


def rolling_filter(t: pd.DataFrame, min_n: int, days: int = ROLL_DAYS) -> pd.Series:
    """Keep a signal unless the coin's signals that CLOSED before this entry (within `days`) average < 0."""
    keep = np.ones(len(t), dtype=bool)
    for base, g in t.groupby("base"):
        ent = g["entry_utc"].to_numpy()
        ex = g["exit_utc"].to_numpy()
        net = g["net"].to_numpy()
        for j, idx in enumerate(g.index):
            m = (ex < ent[j]) & (ex >= ent[j] - np.timedelta64(days, "D"))
            if m.sum() >= min_n and net[m].mean() < 0:
                keep[t.index.get_loc(idx)] = False
    return pd.Series(keep, index=t.index)


def run(t: pd.DataFrame, closes, grid, gap: float = 0.0) -> pd.Series:
    tk = simulate_cash(t, SLOTS, SIZE, gap)
    return mark_to_market(tk, closes, grid, gap)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 100)
    t, r4 = load()
    closes = {b: g.set_index(g["timestamp"] + BAR)["close"] for b, g in r4.groupby("base")}
    grid = pd.date_range(START, t["exit_utc"].max().ceil("4h"), freq="4h")

    taken = simulate_cash(t, SLOTS, SIZE)
    tab = coin_table(t, taken)
    tab.to_csv(OUT / "coins.csv")

    # --------------------------------------------------------------------------- filters
    variants: dict[str, pd.DataFrame] = {"no filter": t}
    bad_all = tab.index[tab["mean_net_pct"] < 0].tolist()
    variants[f"hindsight: drop {len(bad_all)} coins with 36m mean<0"] = t[~t["base"].isin(bad_all)]
    first, second = t[t["entry_utc"] < HALF], t[t["entry_utc"] >= HALF]
    bad_first = first.groupby("base")["net"].mean().loc[lambda x: x < 0].index.tolist()
    bad_second = second.groupby("base")["net"].mean().loc[lambda x: x < 0].index.tolist()
    variants[f"cross-val: drops from 1st half ({len(bad_first)}) applied to 2nd half"] = \
        pd.concat([first, second[~second["base"].isin(bad_first)]])
    variants[f"cross-val: drops from 2nd half ({len(bad_second)}) applied to 1st half"] = \
        pd.concat([first[~first["base"].isin(bad_second)], second])
    for n in MIN_N:
        keep = rolling_filter(t, n)
        variants[f"rolling 365d, min {n} closed signals"] = t[keep]
    nonalt = ["PAXG", "XAUT", "WBTC"]
    variants["drop gold/BTC tokens (PAXG, XAUT, WBTC)"] = t[~t["base"].isin(nonalt)]

    rows, curves = [], {}
    for name, sub in variants.items():
        sub = sub.sort_values(["entry_utc", "S"], ascending=[True, False]).reset_index(drop=True)
        for gap in (0.0, LIVE_GAP):
            c = run(sub, closes, grid, gap)
            if gap == 0.0:
                curves[name] = c
            row = {"filter": name, "live_gap": gap > 0, "signals": len(sub)}
            for k, (a, b) in SEGMENTS.items():
                st = seg_stats(c, a, b)
                row[f"{k} ret"] = st["return_pct"]
                row[f"{k} dd"] = st["max_dd_pct"]
            rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "filters.csv", index=False)

    # --------------------------------------------------------------------------- plots
    fig, axes = plt.subplots(1, 3, figsize=(22, 10), gridspec_kw={"width_ratios": [1, 1, 1.4]})
    order = tab.sort_values("portfolio_pnl_pct").index
    ax = axes[0]
    v = tab.loc[order, "portfolio_pnl_pct"]
    ax.barh(order, v, color=np.where(v >= 0, GREEN, RED))
    ax.set_title("Portfolio P&L by coin, 36 months\n(sum of size x net, % of starting equity)")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(alpha=0.3, axis="x")
    ax = axes[1]
    v = tab.loc[order, "mean_net_pct"]
    err = tab.loc[order, "mean_net_pct"] / tab.loc[order, "t_stat"].replace(0, np.nan)
    ax.barh(order, v, xerr=1.0 * err.abs(), color=np.where(v >= 0, GREEN, RED), alpha=0.8, ecolor="black", capsize=2)
    for i, (b, n) in enumerate(tab.loc[order, "signals"].items()):
        ax.text(ax.get_xlim()[1] if False else v.iloc[i], i, f" n={n}", va="center", fontsize=6)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Mean net % per signal (all signals) ± 1 std. error")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(alpha=0.3, axis="x")
    ax = axes[2]
    cols = [c for c in tab.columns if c.startswith("seg: ")] + [c for c in tab.columns if c.startswith("y")]
    h = tab.loc[order, cols].to_numpy(float)
    im = ax.imshow(np.clip(h, -6, 6), cmap="RdYlGn", aspect="auto", vmin=-6, vmax=6)
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            if not np.isnan(h[i, j]):
                ax.text(j, i, f"{h[i, j]:+.1f}", ha="center", va="center", fontsize=6)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c.replace("seg: ", "").replace("y", "") for c in cols], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=8)
    ax.axvline(2.5, color="black", lw=1.5)
    ax.set_title("Mean net % per signal by period (left) and calendar year (right)")
    plt.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(OUT / "coins_1_breakdown.png", dpi=125)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(17, 8))
    for (name, c), col in zip(curves.items(), plt.cm.tab10(np.arange(len(curves)))):
        ax.plot(c.index, 100 * (c - 1), lw=2.0 if name == "no filter" else 1.1, color="black" if name == "no filter" else col,
                ls="--" if name.startswith("hindsight") else "-", label=f"{name}: {100 * (c.iloc[-1] - 1):+.0f}%")
    ax.set_yscale("symlog", linthresh=100)
    ax.axvline(HALF, color="grey", ls=":", lw=1)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Coin filters on the 6 x 16.7% portfolio (dashed = hindsight, not achievable)")
    ax.legend(fontsize=8, loc="upper left")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "coins_2_filters.png", dpi=125)
    plt.close(fig)

    show = ["signals", "mean_net_pct", "win_rate", "pf", "t_stat", "tp_hits", "neg_segments", "portfolio_trades",
            "portfolio_pnl_pct"] + cols
    print("== per coin (sorted by portfolio P&L) ==")
    print(tab[show].round(2).to_string())
    print("\n== filters (6 x 16.7%) ==")
    print(res.round(1).to_string(index=False))
    print("\nhindsight drops:", bad_all)
    print("1st-half drops:", bad_first)
    print("2nd-half drops:", bad_second)
    (OUT / "drops.json").write_text(json.dumps({"hindsight": bad_all, "first_half": bad_first,
                                                 "second_half": bad_second}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
