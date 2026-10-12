"""Portfolio simulation of all live entries (2026-09-13 -> 2026-10-12) with TRN_4 exit + weekend entry block.

Entries are the live fills from the 2026-10-12 pack. Per-trade exits come from
results/live_exit_compare_2026-10-12/per_trade_by_strategy.csv (1m native ALTBTC replay, 0.1%/side fees).

Portfolio rules (current live production): N=4 slots, 25% of realized equity per entry (compounding),
one position per coin, no re-entry on a coin within 2h of its close. Weekend block = no NEW entries
Sat 00:00 - Sun 23:59 UTC; open positions keep running. Entries the live bot never took cannot be added,
so skipped entries are never replaced.

Scenarios (same portfolio rules for all):
  LIVE                 actual live exits
  LIVE + weekend       actual live exits, weekend block
  TRN_4                TRN_4 exit (SL 5%, trail arms +4%, 0.5% distance)
  TRN_4 + weekend      the proposed setup
  TRN_4 + weekend, live slippage   proposed setup with an extra per-trade cost equal to the
                       mean live-vs-sim execution gap measured on the matched live exits

Writes: results/live_trn4_weekend_sim_2026-10-12/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
EXIT_RES = REPO / "results" / "live_exit_compare_2026-10-12"
OUT = REPO / "results" / "live_trn4_weekend_sim_2026-10-12"
NMAX = 4
ALLOC = 0.25
SAME_COIN_BLOCK = pd.Timedelta(hours=2)
FINAL = "TRN_4 + weekend"
SCENARIOS = {
    "LIVE": ("LIVE", False, 0.0),
    "LIVE + weekend": ("LIVE", True, 0.0),
    "TRN_4": ("TRN_4", False, 0.0),
    "TRN_4 + weekend": ("TRN_4", True, 0.0),
    "TRN_4 + weekend, live slippage": ("TRN_4", True, None),
}
STYLE = {
    "LIVE": ("black", "-", 2.4),
    "LIVE + weekend": ("0.45", "--", 1.8),
    "TRN_4": ("#ff7f0e", "--", 1.8),
    "TRN_4 + weekend": ("#d62728", "-", 2.6),
    "TRN_4 + weekend, live slippage": ("#9467bd", ":", 2.0),
}


def live_slippage_pp() -> float:
    v = pd.read_csv(EXIT_RES / "live_vs_sim_validation.csv")
    return float((v["live_gross_pct"] - v["sim_gross_pct"]).mean() / 100)


def load_trades(exit_key: str, extra_cost: float) -> pd.DataFrame:
    d = pd.read_csv(EXIT_RES / "per_trade_by_strategy.csv", parse_dates=["entry_utc", "exit_utc"])
    t = d[d["strategy"] == exit_key].copy()
    t["net_pct"] = t["net_pct"] - extra_cost
    return t.sort_values("entry_utc").reset_index(drop=True)


def simulate(
    t: pd.DataFrame, block_weekend: bool, is_blocked: Callable[[pd.Timestamp], bool] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """is_blocked overrides the default Sat 00:00 - Sun 23:59 UTC window when block_weekend is True."""
    blocked = is_blocked or (lambda ts: ts.weekday() >= 5)
    equity = 1.0
    open_pos: list[dict[str, Any]] = []
    last_close: dict[str, pd.Timestamp] = {}
    curve = [(t["entry_utc"].min(), equity, 0)]
    log: list[dict[str, Any]] = []

    def close_until(ts: pd.Timestamp) -> None:
        nonlocal equity
        for p in sorted([p for p in open_pos if p["exit_utc"] <= ts], key=lambda p: p["exit_utc"]):
            equity += p["size"] * p["net_pct"]
            last_close[p["symbol"]] = p["exit_utc"]
            open_pos.remove(p)
            curve.append((p["exit_utc"], equity, len(open_pos)))

    for _, r in t.iterrows():
        e = r["entry_utc"]
        close_until(e)
        status = "TAKEN"
        if block_weekend and blocked(e):
            status = "SKIP_WEEKEND"
        elif any(p["symbol"] == r["symbol"] for p in open_pos):
            status = "SKIP_COIN_OPEN"
        elif r["symbol"] in last_close and e - last_close[r["symbol"]] < SAME_COIN_BLOCK:
            status = "SKIP_COIN_2H"
        elif len(open_pos) >= NMAX:
            status = "SKIP_SLOTS_FULL"
        row = {**r.to_dict(), "status": status, "size": np.nan, "pnl": np.nan, "equity_at_entry": equity}
        if status == "TAKEN":
            size = ALLOC * equity
            open_pos.append({"symbol": r["symbol"], "exit_utc": r["exit_utc"], "size": size, "net_pct": r["net_pct"]})
            row.update(size=size, pnl=size * r["net_pct"])
            curve.append((e, equity, len(open_pos)))
        log.append(row)
    close_until(pd.Timestamp.max.tz_localize("UTC"))
    return pd.DataFrame(log), pd.DataFrame(curve, columns=["ts", "equity", "open_positions"])


def stats(log: pd.DataFrame, curve: pd.DataFrame) -> dict[str, Any]:
    tk = log[log["status"] == "TAKEN"]
    net = tk["net_pct"]
    wins, losses = net[net > 0], net[net < 0]
    eq = curve["equity"]
    consec = best = 0
    for x in tk.sort_values("exit_utc")["net_pct"]:
        consec = consec + 1 if x < 0 else 0
        best = max(best, consec)
    avg_w, avg_l = float(wins.mean()), float(-losses.mean())
    return {
        "entries_available": len(log),
        "taken": len(tk),
        "skip_weekend": int((log["status"] == "SKIP_WEEKEND").sum()),
        "skip_slots_full": int((log["status"] == "SKIP_SLOTS_FULL").sum()),
        "skip_same_coin": int(log["status"].isin(["SKIP_COIN_OPEN", "SKIP_COIN_2H"]).sum()),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": float((net > 0).mean()),
        "breakeven_win_rate": avg_l / (avg_w + avg_l),
        "avg_win_pct": avg_w * 100,
        "avg_loss_pct": -avg_l * 100,
        "profit_factor": float(wins.sum() / -losses.sum()),
        "total_return_pct": float((eq.iloc[-1] - 1) * 100),
        "max_drawdown_pct": float((eq / eq.cummax() - 1).min() * 100),
        "max_consec_losses": best,
        "worst_trade_pct": float(net.min() * 100),
        "best_trade_pct": float(net.max() * 100),
        "avg_hold_h": float(((tk["exit_utc"] - tk["entry_utc"]).dt.total_seconds() / 3600).mean()),
        "open_mtm_at_end": int((tk["exit_reason"] == "OPEN_MTM").sum()),
    }


def period_of(ts: pd.Series) -> pd.Series:
    return np.where(ts < pd.Timestamp("2026-09-25", tz="UTC"), "Sep 13-21 (T30 period)", "Sep 28-Oct 12 (W2 period)")


def plot_equity(curves: dict[str, pd.DataFrame], summary: pd.DataFrame) -> None:
    fig, (ax, axd) = plt.subplots(2, 1, figsize=(16, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
    for name, c in curves.items():
        color, ls, lw = STYLE[name]
        s = summary.loc[name]
        r = (c["equity"] - 1) * 100
        ax.step(c["ts"], r, where="post", color=color, ls=ls, lw=lw,
                label=f"{name}: {s['total_return_pct']:+.1f}%, max DD {s['max_drawdown_pct']:.1f}%, {int(s['taken'])} trades")
        axd.step(c["ts"], (c["equity"] / c["equity"].cummax() - 1) * 100, where="post", color=color, ls=ls, lw=lw * 0.8)
    start, end = curves["LIVE"]["ts"].min(), max(c["ts"].max() for c in curves.values())
    for d in pd.date_range(start.floor("D"), end.ceil("D"), freq="D"):
        if d.weekday() == 5:
            for a in (ax, axd):
                a.axvspan(d, d + pd.Timedelta(days=2), color="0.92", zorder=0)
    for a in (ax, axd):
        a.axvspan(pd.Timestamp("2026-09-22", tz="UTC"), pd.Timestamp("2026-09-28", tz="UTC"), color="#fff3cd", zorder=0)
    ax.text(pd.Timestamp("2026-09-22 06:00", tz="UTC"), ax.get_ylim()[1] * 0.92, "bot not trading\n(T30 -> W2 switch)", fontsize=9)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("account return, compounded (%)")
    ax.set_title("All live entries, N=4 x 25%: TRN_4 exit + weekend block vs live  (grey = weekends, entries blocked)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    axd.set_ylabel("drawdown (%)")
    axd.grid(alpha=0.3)
    axd.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.tight_layout()
    fig.savefig(OUT / "1_equity_scenarios.png", dpi=130)
    plt.close(fig)


def plot_trades(final: pd.DataFrame, trn4_all: pd.DataFrame) -> None:
    allr = trn4_all.set_index("trade_id")
    f = final.copy()
    f["would_be"] = f["trade_id"].map(allr["net_pct"]) * 100
    fig, ax = plt.subplots(figsize=(17, 6))
    x = np.arange(len(f))
    colors = {"TAKEN": None, "SKIP_WEEKEND": "#bdbdbd", "SKIP_SLOTS_FULL": "#c6dbef", "SKIP_COIN_OPEN": "#fdd0a2", "SKIP_COIN_2H": "#fdd0a2"}
    for i, r in f.iterrows():
        if r["status"] == "TAKEN":
            c = "#2ca02c" if r["net_pct"] > 0 else "#d62728"
            ax.bar(i, r["net_pct"] * 100, color=c, width=0.85)
        else:
            ax.bar(i, r["would_be"], color=colors[r["status"]], width=0.85, hatch="//", edgecolor="white")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylim(-7, 12)
    for i, r in f.iterrows():
        v = r["net_pct"] * 100 if r["status"] == "TAKEN" else r["would_be"]
        if v > 12:
            ax.text(i, 11.2, f"{v:+.0f}%", ha="center", fontsize=8)
    switch = int((f["entry_utc"] < pd.Timestamp("2026-09-25", tz="UTC")).sum()) - 0.5
    ax.axvline(switch, color="black", ls="--", lw=1)
    ax.text(switch + 0.5, 10, "W2 period ->", fontsize=9)
    ticks = list(range(0, len(f), 6))
    ax.set_xticks(ticks, [f"{f.loc[i, 'entry_utc']:%m-%d}" for i in ticks], rotation=45, fontsize=8)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#2ca02c"), plt.Rectangle((0, 0), 1, 1, color="#d62728"),
        plt.Rectangle((0, 0), 1, 1, color="#bdbdbd", hatch="//"), plt.Rectangle((0, 0), 1, 1, color="#c6dbef", hatch="//"),
        plt.Rectangle((0, 0), 1, 1, color="#fdd0a2", hatch="//"),
    ]
    ax.legend(handles, ["taken: win", "taken: loss", "skipped: weekend (what it would have done)", "skipped: 4 slots full", "skipped: same coin"],
              fontsize=8, loc="upper right")
    ax.set_ylabel("net return per trade (%)")
    ax.set_title(f"Every live entry under {FINAL} (chronological)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "2_trades_chronological.png", dpi=130)
    plt.close(fig)


def plot_distribution_and_slots(logs: dict[str, pd.DataFrame], curves: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    bins = np.arange(-9, 16, 0.5)
    for name in ("LIVE", FINAL):
        tk = logs[name][logs[name]["status"] == "TAKEN"]
        axes[0].hist(tk["net_pct"] * 100, bins=bins, alpha=0.55, color=STYLE[name][0], label=f"{name} ({len(tk)} trades)")
    axes[0].axvline(0, color="black", lw=0.8)
    axes[0].set_xlabel("net return per trade (%)")
    axes[0].set_ylabel("trades")
    axes[0].set_title("Trade return distribution")
    axes[0].legend(fontsize=8)

    f = logs[FINAL][logs[FINAL]["status"] == "TAKEN"].copy()
    f["day"] = f["exit_utc"].dt.floor("D")
    daily = f.groupby("day")["pnl"].sum() * 100
    axes[1].bar(daily.index, daily.to_numpy(), color=np.where(daily.to_numpy() >= 0, "#2ca02c", "#d62728"), width=0.8)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_title(f"{FINAL}: realized P&L per day (% of starting equity)")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[1].tick_params(axis="x", rotation=45)

    c = curves[FINAL]
    axes[2].step(c["ts"], c["open_positions"], where="post", color="#d62728")
    axes[2].axhline(NMAX, color="black", ls=":", lw=1)
    axes[2].set_title(f"{FINAL}: open positions (max {NMAX})")
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[2].tick_params(axis="x", rotation=45)
    for a in axes:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "3_distribution_daily_slots.png", dpi=130)
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    slip = live_slippage_pp()
    logs, curves, rows, periods = {}, {}, {}, []
    for name, (exit_key, wk, cost) in SCENARIOS.items():
        extra = -slip if cost is None else cost
        log, curve = simulate(load_trades(exit_key, extra), wk)
        logs[name], curves[name] = log, curve
        rows[name] = stats(log, curve)
        tk = log[log["status"] == "TAKEN"].assign(period=lambda x: period_of(x["entry_utc"]))
        for p, g in tk.groupby("period"):
            periods.append({"scenario": name, "period": p, "taken": len(g), "win_rate": float((g["net_pct"] > 0).mean()),
                            "pnl_pct_of_start_equity": float(g["pnl"].sum() * 100)})
        log.to_csv(OUT / f"trades_{name.replace(' ', '_').replace(',', '').replace('+', 'plus')}.csv", index=False)
    summary = pd.DataFrame(rows).T
    summary.to_csv(OUT / "summary.csv")
    pd.DataFrame(periods).to_csv(OUT / "summary_by_period.csv", index=False)

    final = logs[FINAL]
    tk = final[final["status"] == "TAKEN"]
    reasons = tk["exit_reason"].str.replace(r"_SAME_CANDLE.*|_TRAIL_ACTIVATION.*", "", regex=True)
    losers = tk[tk["net_pct"] < 0][["entry_utc", "symbol", "net_pct", "mfe_pct", "exit_reason", "hold_h"]]
    extra = {
        "live_slippage_pp_per_trade": slip * 100,
        "final_exit_reasons": reasons.value_counts().to_dict(),
        "final_losers_mfe_pct": (losers["mfe_pct"] * 100).describe().round(2).to_dict(),
        "final_losses_by_symbol": losers["symbol"].value_counts().to_dict(),
        "final_losses_by_weekday": losers["entry_utc"].dt.day_name().value_counts().to_dict(),
    }
    (OUT / "final_details.json").write_text(json.dumps(extra, indent=2, default=str), encoding="utf-8")

    plot_equity(curves, summary)
    plot_trades(final, load_trades("TRN_4", 0.0))
    plot_distribution_and_slots(logs, curves)

    pd.set_option("display.width", 250)
    print(summary.astype(float).round(2).T.to_string())
    print(pd.DataFrame(periods).round(3).to_string(index=False))
    print(json.dumps(extra, indent=1, default=str))
    print(losers.assign(net_pct=losers["net_pct"] * 100, mfe_pct=losers["mfe_pct"] * 100).round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
