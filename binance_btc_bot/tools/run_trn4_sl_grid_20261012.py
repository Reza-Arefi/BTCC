"""TRN_4 trail (arms +4%, 0.5% distance) with hard SL 2%..5%, weekend block, N=4 x 25%, all live entries.

Uses the exit walker / portfolio from run_trn4_rescue_lateentry_20261012.py.
Writes: results/trn4_rescue_lateentry_2026-10-12/summary_sl_grid.csv, per_trade_sl_grid.csv, 3_sl_grid.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import binance_btc_bot.tools.run_trn4_rescue_lateentry_20261012 as rescue  # noqa: E402
from binance_btc_bot.tools.run_live_trn4_weekend_sim_20261012 import simulate, stats  # noqa: E402

SL_LEVELS = (2, 2.5, 3, 3.5, 4, 4.5, 5)


def main() -> int:
    rescue.EXITS = {f"SL{s:g}": {"sl": s / 100, "trigger": None, "target": None} for s in SL_LEVELS}
    live = pd.read_csv(rescue.EXIT_RES / "per_trade_by_strategy.csv", parse_dates=["entry_utc"])
    trades = live[live["strategy"] == "LIVE"][["trade_id", "cohort", "symbol", "entry_utc", "entry_px", "alloc", "btc_value"]]
    trades = trades.merge(pd.read_csv(rescue.FEATURES)[["trade_id", "late_entry_score"]], on="trade_id")
    trades = trades.sort_values("entry_utc").reset_index(drop=True)
    d = rescue.replay_all(trades)
    d.to_csv(rescue.OUT / "per_trade_sl_grid.csv", index=False)

    rows, curves = [], {}
    for ex, g in d.groupby("strategy", sort=False):
        t = g.sort_values("entry_utc").reset_index(drop=True)
        log, curve = simulate(t, block_weekend=True)
        s = stats(log, curve)
        curves[ex] = curve
        weekday = t[t["entry_utc"].dt.weekday < 5]
        tk = log[log["status"] == "TAKEN"]
        rows.append({
            "stop_loss": ex,
            "taken": s["taken"],
            "wins": s["wins"],
            "losses": s["losses"],
            "win_rate": s["win_rate"],
            "avg_win_pct": s["avg_win_pct"],
            "avg_loss_pct": s["avg_loss_pct"],
            "profit_factor": s["profit_factor"],
            "return_pct": s["total_return_pct"],
            "max_dd_pct": s["max_drawdown_pct"],
            "max_consec_losses": s["max_consec_losses"],
            "trail_exits": int((tk["exit_reason"] == "TRAILING_STOP").sum()),
            "mean_trade_T30_weekday_pct": float(weekday.loc[weekday["cohort"] == "T30", "net_pct"].mean() * 100),
            "mean_trade_W2_weekday_pct": float(weekday.loc[weekday["cohort"] == "W2", "net_pct"].mean() * 100),
        })
    r = pd.DataFrame(rows)
    r.to_csv(rescue.OUT / "summary_sl_grid.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(17, 6))
    x = range(len(r))
    axes[0].bar(x, r["return_pct"], color=["#d62728" if s == "SL5" else "#1f77b4" for s in r["stop_loss"]])
    for i, row in r.iterrows():
        axes[0].text(i, row["return_pct"] + 0.4, f"{row['return_pct']:+.1f}%\nwin {row['win_rate']:.0%}\nDD {row['max_dd_pct']:.1f}%",
                     ha="center", fontsize=8)
    axes[0].set_xticks(list(x), [f"SL {s:g}%" for s in SL_LEVELS])
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_ylim(min(0, r["return_pct"].min()) - 2, r["return_pct"].max() + 8)
    axes[0].set_ylabel("account return (%)")
    axes[0].set_title("TRN_4 trail (+4% / 0.5%) + weekend block: hard stop-loss size")
    for ex, c in curves.items():
        axes[1].step(c["ts"], (c["equity"] - 1) * 100, where="post", lw=2.6 if ex == "SL5" else 1.4,
                     color="#d62728" if ex == "SL5" else None, label=f"{ex}: {r.set_index('stop_loss').loc[ex, 'return_pct']:+.1f}%")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_ylabel("account return, compounded (%)")
    axes[1].set_title("Equity by stop-loss size")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.tight_layout()
    fig.savefig(rescue.OUT / "3_sl_grid.png", dpi=130)
    plt.close(fig)

    pd.set_option("display.width", 250)
    print(r.round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
