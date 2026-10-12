"""Candidate A (TRN_4, SL 5%) with different weekend entry-block windows, all live entries.

Window = block new entries from Friday <start> UTC (or Saturday 00:00) until Sunday <end> UTC (or Monday 00:00).
Portfolio: N=4 x 25%, one position per coin, 2h same-coin block (run_live_trn4_weekend_sim_20261012.simulate).
Also reports TRN_4 results of the Friday / Sunday entry slices on their own (no portfolio).

Writes: results/weekend_window_grid_2026-10-12/
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from binance_btc_bot.tools.run_live_trn4_weekend_sim_20261012 import load_trades, simulate, stats  # noqa: E402

OUT = REPO / "results" / "weekend_window_grid_2026-10-12"
FRI_START = (12, 15, 18, 21, 24)  # 24 = no Friday block (start Saturday 00:00)
SUN_END = (12, 15, 18, 21, 24)  # 24 = block all of Sunday (end Monday 00:00)
LOCAL_OFFSET_H = -3


def blocker(fri_start: int, sun_end: int):
    def is_blocked(ts: pd.Timestamp) -> bool:
        wd, h = ts.weekday(), ts.hour
        return (wd == 4 and h >= fri_start) or wd == 5 or (wd == 6 and h < sun_end)

    return is_blocked


def label(fri_start: int, sun_end: int) -> str:
    a = "Sat 00:00" if fri_start == 24 else f"Fri {fri_start:02d}:00"
    b = "Mon 00:00" if sun_end == 24 else f"Sun {sun_end:02d}:00"
    return f"{a} -> {b}"


def slice_stats(t: pd.DataFrame) -> pd.DataFrame:
    t = t.assign(wd=t["entry_utc"].dt.weekday, h=t["entry_utc"].dt.hour)
    slices = {
        "Fri 00-12": (t["wd"] == 4) & (t["h"] < 12),
        "Fri 12-15": (t["wd"] == 4) & t["h"].between(12, 14),
        "Fri 15-18": (t["wd"] == 4) & t["h"].between(15, 17),
        "Fri 18-21": (t["wd"] == 4) & t["h"].between(18, 20),
        "Fri 21-24": (t["wd"] == 4) & (t["h"] >= 21),
        "Sat (all day)": t["wd"] == 5,
        "Sun 00-12": (t["wd"] == 6) & (t["h"] < 12),
        "Sun 12-15": (t["wd"] == 6) & t["h"].between(12, 14),
        "Sun 15-18": (t["wd"] == 6) & t["h"].between(15, 17),
        "Sun 18-21": (t["wd"] == 6) & t["h"].between(18, 20),
        "Sun 21-24": (t["wd"] == 6) & (t["h"] >= 21),
        "Mon-Thu": t["wd"] <= 3,
    }
    rows = []
    for name, m in slices.items():
        g = t[m]
        rows.append({"slice_utc": name, "entries": len(g), "wins": int((g["net_pct"] > 0).sum()),
                     "losses": int((g["net_pct"] < 0).sum()),
                     "win_rate": float((g["net_pct"] > 0).mean()) if len(g) else np.nan,
                     "mean_net_pct": float(g["net_pct"].mean() * 100) if len(g) else np.nan,
                     "sum_net_pct": float(g["net_pct"].sum() * 100)})
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t = load_trades("TRN_4", 0.0)
    rows = []
    _, base_curve = simulate(t, block_weekend=False)
    for fs in FRI_START:
        for se in SUN_END:
            log, curve = simulate(t, block_weekend=True, is_blocked=blocker(fs, se))
            s = stats(log, curve)
            rows.append({"window_utc": label(fs, se), "fri_start": fs, "sun_end": se,
                         "blocked_hours_per_week": (24 - fs) + 24 + se, **s})
    grid = pd.DataFrame(rows)
    nb_log, nb_curve = simulate(t, block_weekend=False)
    nb = stats(nb_log, nb_curve)
    grid.to_csv(OUT / "window_grid.csv", index=False)
    sl = slice_stats(t)
    sl.to_csv(OUT / "entry_slices_trn4.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), gridspec_kw={"width_ratios": [1.1, 1.1, 1.3]})
    for ax, col, title, cmap in ((axes[0], "total_return_pct", "Account return (%)", "RdYlGn"),
                                 (axes[1], "max_drawdown_pct", "Worst drawdown (%)", "RdYlGn")):
        m = grid.pivot(index="fri_start", columns="sun_end", values=col)
        im = ax.imshow(m.to_numpy(), cmap=cmap, aspect="auto")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                bold = (m.index[i], m.columns[j]) == (24, 24)
                ax.text(j, i, f"{m.iat[i, j]:+.1f}", ha="center", va="center", fontsize=10, fontweight="bold" if bold else None)
        ax.set_xticks(range(m.shape[1]), ["Sun 12", "Sun 15", "Sun 18", "Sun 21", "Mon 00"])
        ax.set_yticks(range(m.shape[0]), ["Fri 12", "Fri 15", "Fri 18", "Fri 21", "Sat 00"])
        ax.set_xlabel("block ends (UTC)")
        ax.set_ylabel("block starts (UTC)")
        ax.set_title(f"{title}  - bold = current Sat 00 -> Mon 00\nno block: {nb['total_return_pct']:+.1f}% / DD {nb['max_drawdown_pct']:.1f}%")
        plt.colorbar(im, ax=ax, fraction=0.04)
    ax = axes[2]
    s = sl.set_index("slice_utc")
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in s["mean_net_pct"].fillna(0)]
    ax.bar(range(len(s)), s["mean_net_pct"].fillna(0), color=colors)
    for i, (n, w) in enumerate(zip(s["entries"], s["win_rate"])):
        ax.text(i, (s["mean_net_pct"].fillna(0).iloc[i]) + (0.15 if s["mean_net_pct"].fillna(0).iloc[i] >= 0 else -0.5),
                f"n={n}\n{w:.0%}" if n else "n=0", ha="center", fontsize=8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(s)), s.index, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("mean net return per entry, TRN_4 (%)")
    ax.set_title("Entry time slices (UTC): mean result, entries, win rate")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "weekend_window_grid.png", dpi=130)
    plt.close(fig)

    pd.set_option("display.width", 250)
    cols = ["window_utc", "blocked_hours_per_week", "taken", "skip_weekend", "wins", "losses", "win_rate", "profit_factor",
            "total_return_pct", "max_drawdown_pct", "max_consec_losses"]
    print(f"no block: {nb['total_return_pct']:.2f}% DD {nb['max_drawdown_pct']:.2f}%")
    print(grid[cols].sort_values("total_return_pct", ascending=False).round(2).to_string(index=False))
    print(sl.round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
