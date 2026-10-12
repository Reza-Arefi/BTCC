"""Compare three TRN_4-based candidates on all live entries (2026-09-13 -> 2026-10-12), weekend entry block.

  A  TRN_4, SL 5%                                  (current candidate)
  B  TRN_4, SL 4%
  C  TRN_4, SL 5%; after touching -4% the take-profit moves to -2% (rescue)

Exit walker / fees from run_trn4_rescue_lateentry_20261012.py; portfolio from run_live_trn4_weekend_sim_20261012.py.
Checks: portfolio N=4 (main), trade-by-trade on the same weekday entries, and slot sensitivity (N=3, 5, unlimited).

Writes: results/three_candidates_2026-10-12/
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import binance_btc_bot.tools.run_live_trn4_weekend_sim_20261012 as port  # noqa: E402
import binance_btc_bot.tools.run_trn4_rescue_lateentry_20261012 as rescue  # noqa: E402

OUT = REPO / "results" / "three_candidates_2026-10-12"
CANDIDATES = {
    "A: SL5": {"sl": 0.05, "trigger": None, "target": None},
    "B: SL4": {"sl": 0.04, "trigger": None, "target": None},
    "C: SL5, -4% -> TP -2%": {"sl": 0.05, "trigger": 0.04, "target": -0.02},
}
COLORS = {"A: SL5": "#d62728", "B: SL4": "#1f77b4", "C: SL5, -4% -> TP -2%": "#2ca02c"}


def load_entries() -> pd.DataFrame:
    live = pd.read_csv(rescue.EXIT_RES / "per_trade_by_strategy.csv", parse_dates=["entry_utc"])
    t = live[live["strategy"] == "LIVE"][["trade_id", "cohort", "symbol", "entry_utc", "entry_px", "alloc", "btc_value"]]
    t = t.merge(pd.read_csv(rescue.FEATURES)[["trade_id", "late_entry_score"]], on="trade_id")
    return t.sort_values("entry_utc").reset_index(drop=True)


def portfolio(d: pd.DataFrame, nmax: int) -> tuple[dict[str, dict], dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    port.NMAX = nmax
    res, curves, logs = {}, {}, {}
    for name in CANDIDATES:
        t = d[d["strategy"] == name].sort_values("entry_utc").reset_index(drop=True)
        log, curve = port.simulate(t, block_weekend=True)
        res[name], curves[name], logs[name] = port.stats(log, curve), curve, log
    return res, curves, logs


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rescue.EXITS = CANDIDATES
    d = rescue.replay_all(load_entries())
    d.to_csv(OUT / "per_trade.csv", index=False)

    main_res, curves, logs = portfolio(d, 4)
    summary = pd.DataFrame(main_res).T
    for name in CANDIDATES:
        tk = logs[name][logs[name]["status"] == "TAKEN"]
        for p, lo, hi in (("Sep 13-21", "2026-09-01", "2026-09-25"), ("Sep 28-Oct 12", "2026-09-25", "2026-11-01")):
            g = tk[(tk["entry_utc"] >= pd.Timestamp(lo, tz="UTC")) & (tk["entry_utc"] < pd.Timestamp(hi, tz="UTC"))]
            summary.loc[name, f"pnl_{p}_pct"] = float(g["pnl"].sum() * 100)
    summary.to_csv(OUT / "summary_portfolio_n4.csv")

    wk = d[d["entry_utc"].dt.weekday < 5]
    piv = wk.pivot_table(index="trade_id", columns="strategy", values="net_pct")[list(CANDIDATES)]
    cohort = wk.drop_duplicates("trade_id").set_index("trade_id")["cohort"]
    per_trade = []
    for name in CANDIDATES:
        for c in ("T30", "W2", "all"):
            x = piv[name] if c == "all" else piv.loc[cohort[cohort == c].index, name]
            per_trade.append({"candidate": name, "entries": c, "n": len(x), "win_rate": float((x > 0).mean()),
                              "mean_net_pct": float(x.mean() * 100), "sum_net_pct": float(x.sum() * 100)})
    per_trade = pd.DataFrame(per_trade)
    per_trade.to_csv(OUT / "per_trade_same_entries.csv", index=False)

    diffs = []
    base = piv["A: SL5"]
    for name in list(CANDIDATES)[1:]:
        delta = piv[name] - base
        changed = delta[delta.abs() > 1e-9]
        rng = np.random.default_rng(5)
        boot = delta.to_numpy()[rng.integers(0, len(delta), size=(10000, len(delta)))].mean(axis=1)
        diffs.append({"candidate": name, "trades_changed": len(changed),
                      "losers_made_smaller": int(((base < 0) & (delta > 0)).sum()),
                      "winners_turned_losers": int(((base > 0) & (piv[name] < 0)).sum()),
                      "total_change_pp": float(delta.sum() * 100), "mean_change_pp": float(delta.mean() * 100),
                      "p_better_than_A": float((boot > 0).mean())})
    diffs = pd.DataFrame(diffs)
    diffs.to_csv(OUT / "vs_A_trade_by_trade.csv", index=False)

    sens = []
    for n in (3, 4, 5, 99):
        r, _, _ = portfolio(d, n)
        for name, s in r.items():
            sens.append({"slots": "unlimited" if n == 99 else n, "candidate": name, "taken": s["taken"],
                         "return_pct": s["total_return_pct"], "max_dd_pct": s["max_drawdown_pct"], "profit_factor": s["profit_factor"]})
    sens = pd.DataFrame(sens)
    sens.to_csv(OUT / "slot_sensitivity.csv", index=False)
    port.NMAX = 4

    plot(curves, summary, piv, cohort, sens)
    pd.set_option("display.width", 250)
    cols = ["taken", "wins", "losses", "win_rate", "avg_win_pct", "avg_loss_pct", "profit_factor", "total_return_pct",
            "max_drawdown_pct", "max_consec_losses", "avg_hold_h", "pnl_Sep 13-21_pct", "pnl_Sep 28-Oct 12_pct"]
    print(summary[cols].astype(float).round(2).to_string())
    print(per_trade.round(3).to_string(index=False))
    print(diffs.round(3).to_string(index=False))
    print(sens.pivot(index="slots", columns="candidate", values="return_pct").round(1).to_string())
    print(sens.pivot(index="slots", columns="candidate", values="max_dd_pct").round(1).to_string())
    return 0


def plot(curves, summary, piv, cohort, sens) -> None:
    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1])
    ax = fig.add_subplot(gs[0, :])
    for name, c in curves.items():
        s = summary.loc[name]
        ax.step(c["ts"], (c["equity"] - 1) * 100, where="post", color=COLORS[name], lw=2.2,
                label=f"{name}: {s['total_return_pct']:+.1f}%, DD {s['max_drawdown_pct']:.1f}%, PF {s['profit_factor']:.2f}, "
                      f"{int(s['taken'])} trades, win {s['win_rate']:.0%}")
    start, end = curves["A: SL5"]["ts"].min(), max(c["ts"].max() for c in curves.values())
    for day in pd.date_range(start.floor("D"), end.ceil("D"), freq="D"):
        if day.weekday() == 5:
            ax.axvspan(day, day + pd.Timedelta(days=2), color="0.92", zorder=0)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("account return, compounded (%)")
    ax.set_title("Three TRN_4 candidates, all live entries, weekend block, N=4 x 25%  (grey = weekends)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    ax2 = fig.add_subplot(gs[1, 0])
    base = piv["A: SL5"]
    order = piv.index[np.argsort(base.to_numpy())]
    for name, off in (("B: SL4", -0.2), ("C: SL5, -4% -> TP -2%", 0.2)):
        delta = (piv.loc[order, name] - base.loc[order]) * 100
        idx = np.flatnonzero(delta.abs().to_numpy() > 1e-9)
        ax2.bar(idx + off, delta.to_numpy()[idx], width=0.4, color=COLORS[name], label=f"{name} minus A")
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xlabel("weekday entries, sorted by A's result (left = A's worst)")
    ax2.set_ylabel("difference vs A (pp)")
    ax2.set_title("Trade-by-trade difference vs A (only trades that changed)")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 1])
    slots = ["3", "4", "5", "unlimited"]
    sens = sens.assign(slots=sens["slots"].astype(str))
    x = np.arange(len(slots))
    for k, name in enumerate(CANDIDATES):
        v = sens[sens["candidate"] == name].set_index("slots").loc[slots, "return_pct"]
        ax3.bar(x + (k - 1) * 0.27, v, 0.27, color=COLORS[name], label=name)
    ax3.set_xticks(x, [f"N={s}" for s in slots])
    ax3.set_ylabel("account return (%)")
    ax3.set_title("Slot sensitivity (25% per trade; unlimited = every weekday entry taken)")
    ax3.legend(fontsize=9)
    ax3.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "three_candidates.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
