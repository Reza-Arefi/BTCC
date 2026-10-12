"""TRN_4 + weekend block: loss-rescue exits and late-entry filter, on all live entries (2026-09-13 -> 2026-10-12).

Exit walker on native ALTBTC 1m (cache from run_live_exit_compare_20261012.py), 0.1%/side fees.

TRN_4: SL -5%, trail arms when high >= +4%, stop = peak x (1 - 0.5%). SL first if SL and activation share a bar.

Rescue RES_{X}_{T}: while the trail is not armed, if price touches -X% the trade enters rescue mode from the
next bar: the +4% trail is replaced by a take-profit at T (relative to entry); the -5% SL stays.
In rescue mode SL is checked before the take-profit within a bar. T = +0.2% is break-even after fees.

Tight SL comparison: TRN_4 geometry with the hard SL at -3% / -4% (cut the loss instead of waiting).

Late-entry filter (applied before the portfolio): original rule blocks late_entry_score >= 0.55
(HIGH / VERY_HIGH class, btcc.sim.entry_policy); softer variants block >= 0.65 / >= 0.70.

Portfolio: weekend entry block, N=4, 25% of realized equity, one position per coin, 2h same-coin block
(simulate() from run_live_trn4_weekend_sim_20261012).

Writes: results/trn4_rescue_lateentry_2026-10-12/
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from binance_btc_bot.tools.run_live_trn4_weekend_sim_20261012 import simulate, stats  # noqa: E402

EXIT_RES = REPO / "results" / "live_exit_compare_2026-10-12"
FEATURES = REPO / "results" / "live_no_trade_zones_2026-10-12" / "trade_features.csv"
CACHE = REPO / "data" / "live_replay_1m_native"
OUT = REPO / "results" / "trn4_rescue_lateentry_2026-10-12"
FEE_RT = 0.002
MAX_HOLD = pd.Timedelta(days=7)
ACT, DIST = 0.04, 0.005

EXITS: dict[str, dict[str, float | None]] = {
    "TRN_4": {"sl": 0.05, "trigger": None, "target": None},
    **{
        f"RES_{int(x * 100)}_{'BE' if t > 0 else f'm{int(-t * 100)}'}": {"sl": 0.05, "trigger": x, "target": t}
        for x in (0.02, 0.03, 0.04)
        for t in (0.002, -0.01, -0.02)
        if -t < x
    },
    "TRN_4_SL4": {"sl": 0.04, "trigger": None, "target": None},
    "TRN_4_SL3": {"sl": 0.03, "trigger": None, "target": None},
}
LATE = {"no late filter": None, "late >= 0.55 (original rule)": 0.55, "late >= 0.65": 0.65, "late >= 0.70": 0.70}


def walk(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, entry: float, sl: float,
         trigger: float | None, target: float | None) -> tuple[int, float, str, bool]:
    """Returns (exit bar index, exit price, reason, rescue_was_armed)."""
    sl_px = entry * (1 - sl)
    act_px = entry * (1 + ACT)
    trig_px = entry * (1 - trigger) if trigger is not None else -np.inf
    tgt_px = entry * (1 + target) if target is not None else np.inf
    trailing = rescue = False
    peak = stop = 0.0
    for i in range(len(h)):
        if rescue:
            if l[i] <= sl_px:
                return i, sl_px, "STOP_LOSS", True
            if h[i] >= tgt_px:
                return i, tgt_px, "RESCUE_TP", True
            continue
        if not trailing:
            if l[i] <= sl_px:
                return i, sl_px, "STOP_LOSS", False
            if l[i] <= trig_px:
                rescue = True
                continue
            if h[i] >= act_px:
                trailing = True
                peak = h[i]
                stop = peak * (1 - DIST)
        if trailing:
            peak = max(peak, h[i])
            stop = max(stop, peak * (1 - DIST), sl_px)
            if l[i] <= stop:
                return i, stop, "TRAILING_STOP", False
    return len(c) - 1, float(c[-1]), "OPEN_MTM", rescue


def replay_all(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, t in trades.iterrows():
        m = pd.read_parquet(CACHE / f"{t['symbol']}.parquet")
        e = t["entry_utc"]
        bars = m[(m["timestamp"] >= e.floor("min") + pd.Timedelta(minutes=1)) & (m["timestamp"] < e + MAX_HOLD)]
        arr = {k: bars[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close")}
        ts = bars["timestamp"].to_numpy()
        entry = float(t["entry_px"])
        for name, p in EXITS.items():
            i, px, reason, armed = walk(arr["open"], arr["high"], arr["low"], arr["close"], entry, p["sl"], p["trigger"], p["target"])
            gross = px / entry - 1
            rows.append({
                "trade_id": t["trade_id"], "cohort": t["cohort"], "symbol": t["symbol"], "entry_utc": e,
                "exit_utc": pd.Timestamp(ts[i]), "strategy": name, "exit_reason": reason, "rescue_armed": armed,
                "gross_pct": gross, "net_pct": gross - FEE_RT, "alloc": t["alloc"], "btc_value": t["btc_value"],
                "late_entry_score": t["late_entry_score"], "mfe_pct": float(arr["high"][: i + 1].max() / entry - 1),
            })
    return pd.DataFrame(rows)


def per_trade_effect(d: pd.DataFrame) -> pd.DataFrame:
    """Exit-level effect of each rescue vs TRN_4 on the trades actually taken by the TRN_4 + weekend portfolio."""
    base = d[d["strategy"] == "TRN_4"].set_index("trade_id")
    rows = []
    for name, g in d.groupby("strategy", sort=False):
        g = g.set_index("trade_id")
        b = base.loc[g.index, "net_pct"]
        armed = g["rescue_armed"]
        rows.append({
            "strategy": name,
            "rescue_armed": int(armed.sum()),
            "saved_losers": int(((b < 0) & (g["net_pct"] > b + 1e-9)).sum()),
            "killed_winners": int(((b > 0) & (g["net_pct"] < 0)).sum()),
            "winners_cut_smaller": int(((b > 0) & (g["net_pct"] < b - 1e-9)).sum()),
            "sum_change_pp": float((g["net_pct"] - b).sum() * 100),
        })
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    live = pd.read_csv(EXIT_RES / "per_trade_by_strategy.csv", parse_dates=["entry_utc"])
    trades = live[live["strategy"] == "LIVE"][["trade_id", "cohort", "symbol", "entry_utc", "entry_px", "alloc", "btc_value"]]
    feats = pd.read_csv(FEATURES)[["trade_id", "late_entry_score"]]
    trades = trades.merge(feats, on="trade_id").sort_values("entry_utc").reset_index(drop=True)

    d = replay_all(trades)
    d.to_csv(OUT / "per_trade_exits.csv", index=False)
    check = d[d["strategy"] == "TRN_4"].merge(
        live[live["strategy"] == "TRN_4"][["trade_id", "gross_pct"]], on="trade_id", suffixes=("", "_engine"))
    max_dev = float((check["gross_pct"] - check["gross_pct_engine"]).abs().max() * 100)
    print(f"TRN_4 walker vs research engine: max abs diff {max_dev:.4f} pp over {len(check)} trades")

    rows, curves, logs = [], {}, {}
    for late_name, thr in LATE.items():
        for ex in EXITS:
            t = d[d["strategy"] == ex].sort_values("entry_utc").reset_index(drop=True)
            blocked = int((t["late_entry_score"] >= thr).sum()) if thr is not None else 0
            if thr is not None:
                t = t[t["late_entry_score"] < thr].reset_index(drop=True)
            log, curve = simulate(t, block_weekend=True)
            s = stats(log, curve)
            rows.append({"late_filter": late_name, "exit": ex, "late_blocked": blocked, **s})
            curves[(late_name, ex)], logs[(late_name, ex)] = curve, log
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "summary.csv", index=False)

    taken_ids = logs[("no late filter", "TRN_4")].query("status == 'TAKEN'")["trade_id"]
    effect = per_trade_effect(d[d["trade_id"].isin(taken_ids)])
    effect.to_csv(OUT / "rescue_effect_on_taken_trades.csv", index=False)

    plot_rescue(summary, curves)
    plot_late(summary, curves, d)

    pd.set_option("display.width", 250)
    cols = ["late_filter", "exit", "late_blocked", "taken", "wins", "losses", "win_rate", "avg_win_pct", "avg_loss_pct",
            "profit_factor", "total_return_pct", "max_drawdown_pct", "max_consec_losses"]
    print(summary[cols].round(2).to_string(index=False))
    print(effect.round(2).to_string(index=False))
    return 0


def plot_rescue(summary: pd.DataFrame, curves: dict[tuple[str, str], pd.DataFrame]) -> None:
    s = summary[summary["late_filter"] == "no late filter"].set_index("exit")
    fig, axes = plt.subplots(1, 2, figsize=(17, 6), gridspec_kw={"width_ratios": [1.1, 1.6]})
    order = list(EXITS)
    colors = ["#d62728" if e == "TRN_4" else ("#7f7f7f" if "SL" in e and not e.startswith("RES") else "#1f77b4") for e in order]
    axes[0].barh(order[::-1], s.loc[order[::-1], "total_return_pct"], color=colors[::-1])
    for i, e in enumerate(order[::-1]):
        v = s.loc[e]
        axes[0].text(v["total_return_pct"] + 0.3, i, f"{v['total_return_pct']:+.1f}%  DD {v['max_drawdown_pct']:.1f}%  PF {v['profit_factor']:.2f}",
                     va="center", fontsize=8)
    axes[0].axvline(0, color="black", lw=0.8)
    axes[0].set_xlabel("account return (%)")
    axes[0].set_title("TRN_4 + weekend: rescue variants (blue) and tighter SL (grey)")
    axes[0].set_xlim(min(0, s["total_return_pct"].min()) - 1, s["total_return_pct"].max() + 14)
    best = s.drop("TRN_4").sort_values("total_return_pct", ascending=False).index[:3].tolist()
    for e in ["TRN_4", *best, "RES_3_BE", "TRN_4_SL3"]:
        c = curves[("no late filter", e)]
        axes[1].step(c["ts"], (c["equity"] - 1) * 100, where="post", lw=2.6 if e == "TRN_4" else 1.5,
                     color="#d62728" if e == "TRN_4" else None, label=f"{e}: {s.loc[e, 'total_return_pct']:+.1f}%")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_ylabel("account return, compounded (%)")
    axes[1].set_title("Equity: TRN_4 vs best rescue variants")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.tight_layout()
    fig.savefig(OUT / "1_rescue_variants.png", dpi=130)
    plt.close(fig)


def plot_late(summary: pd.DataFrame, curves: dict[tuple[str, str], pd.DataFrame], d: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(17, 6))
    t = d[d["strategy"] == "TRN_4"]
    wk = t["entry_utc"].dt.weekday < 5
    for ok, c, lab in ((True, "#2ca02c", "win"), (False, "#d62728", "loss")):
        m = wk & ((t["net_pct"] > 0) == ok)
        axes[0].hist(t.loc[m, "late_entry_score"], bins=np.arange(0.45, 0.8, 0.02), alpha=0.6, color=c, label=f"TRN_4 {lab} (weekday entries)")
    for thr, ls in ((0.55, "-"), (0.65, "--"), (0.70, ":")):
        axes[0].axvline(thr, color="black", ls=ls, lw=1, label=f"block >= {thr}")
    axes[0].set_xlabel("late-entry score at entry")
    axes[0].set_ylabel("trades")
    axes[0].set_title("Late-entry score: winners vs losers (most entries are >= 0.55)")
    axes[0].legend(fontsize=8)
    for late_name in LATE:
        c = curves[(late_name, "TRN_4")]
        r = summary[(summary["late_filter"] == late_name) & (summary["exit"] == "TRN_4")].iloc[0]
        axes[1].step(c["ts"], (c["equity"] - 1) * 100, where="post", lw=2.4 if late_name == "no late filter" else 1.5,
                     label=f"{late_name}: {r['total_return_pct']:+.1f}%, {int(r['taken'])} trades, DD {r['max_drawdown_pct']:.1f}%")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_title("TRN_4 + weekend, with / without late-entry filter")
    axes[1].set_ylabel("account return, compounded (%)")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.tight_layout()
    fig.savefig(OUT / "2_late_entry_filter.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
