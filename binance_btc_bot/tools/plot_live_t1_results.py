"""Generate LIVE T1 result plots (accumulation, trades, W/L, drawdown).

Reads closed trades from data/binance_btc_bot/bot.sqlite3.
Does not place orders or touch the live process.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

BRT = ZoneInfo("America/Sao_Paulo")
REPO = Path(__file__).resolve().parents[2]
DB_PATH = REPO / "data" / "binance_btc_bot" / "bot.sqlite3"
OUT_DIR = REPO / "data" / "binance_btc_bot" / "plots_live_t1"

# Arm equity used for % of capital (Stage-8 re-arm snapshot).
STARTING_EQUITY_BTC = 0.00783355


def _parse_ts(v) -> datetime | None:
    if v is None or v == "":
        return None
    try:
        if isinstance(v, (int, float)) or (isinstance(v, str) and v.replace(".", "", 1).isdigit()):
            return datetime.fromtimestamp(float(v), tz=timezone.utc)
        s = str(v).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def load_scored_trades(db_path: Path = DB_PATH) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """
        SELECT trade_id, symbol, strategy, status, entry_time, exit_time,
               entry_price, exit_price, btc_value, realized_pnl_btc,
               realized_pnl_btc_equivalent
        FROM trades
        WHERE upper(status)='CLOSED'
        ORDER BY exit_time, entry_time
        """,
        con,
    )
    con.close()
    if df.empty:
        return df

    df["pnl_btc"] = pd.to_numeric(df["realized_pnl_btc"], errors="coerce")
    alt = pd.to_numeric(df["realized_pnl_btc_equivalent"], errors="coerce")
    df["pnl_btc"] = df["pnl_btc"].fillna(alt)
    # Drop stale rows with no authoritative PnL (e.g. repaired UNI without exit price).
    df = df.dropna(subset=["pnl_btc"]).copy()

    df["exit_dt"] = df["exit_time"].map(_parse_ts)
    df["entry_dt"] = df["entry_time"].map(_parse_ts)
    df = df.dropna(subset=["exit_dt"]).sort_values("exit_dt").reset_index(drop=True)

    df["strategy"] = df["strategy"].fillna("T1").astype(str).str.upper()
    df.loc[df["strategy"].isin(["", "NONE", "NULL"]), "strategy"] = "T1"

    df["pnl_pct_equity"] = 100.0 * df["pnl_btc"] / float(STARTING_EQUITY_BTC)
    df["is_win"] = df["pnl_btc"] > 0
    df["is_loss"] = df["pnl_btc"] < 0
    df["is_flat"] = df["pnl_btc"] == 0

    df["exit_brt"] = df["exit_dt"].map(lambda d: d.astimezone(BRT))
    df["day_brt"] = df["exit_brt"].map(lambda d: d.date())
    first_day = df["day_brt"].min()
    df["day_number"] = df["day_brt"].map(lambda d: (d - first_day).days + 1)
    return df


def _save(fig, path: Path, *, use_tight: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if use_tight:
        fig.tight_layout()
    else:
        fig.subplots_adjust(left=0.1, right=0.96, top=0.88, bottom=0.08, hspace=0.45)
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_cumulative_return(df: pd.DataFrame, out: Path) -> Path:
    g = df.sort_values("exit_dt").copy()
    g["cum_pct"] = g["pnl_pct_equity"].cumsum()
    daily = g.groupby("day_number", as_index=False).agg(cum_pct=("cum_pct", "last"))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        daily["day_number"],
        daily["cum_pct"],
        color="#1f4e79",
        marker="o",
        lw=2.2,
        ms=7,
        label="1 T1",
    )
    final = float(daily["cum_pct"].iloc[-1])
    ax.axhline(final, color="#7eb6d9", lw=1.2, alpha=0.9)
    ax.text(
        0.02,
        final,
        f"{final:.2f}",
        transform=ax.get_yaxis_transform(),
        color="#5b9bd5",
        va="bottom",
        fontsize=9,
    )
    ax.text(
        float(daily["day_number"].max()) + 0.05,
        final,
        "1 T1",
        color="#5b9bd5",
        va="center",
        fontsize=9,
    )
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_xlabel("Day")
    ax.set_ylabel("Cumulative return (%)")
    ax.set_title("Cumulative return — LIVE T1")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=9)
    return _save(fig, out / "cumulative_return_t1.png")


def plot_drawdown(df: pd.DataFrame, out: Path) -> Path:
    g = df.sort_values("exit_dt").copy()
    g["cum_pct"] = g["pnl_pct_equity"].cumsum()
    g["peak"] = g["cum_pct"].cummax()
    g["dd_pct"] = g["cum_pct"] - g["peak"]
    daily = g.groupby("day_number", as_index=False).agg(dd_pct=("dd_pct", "min"))

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(daily["day_number"], daily["dd_pct"], color="#2f6fb5", marker="o", lw=2, ms=6, label="1 T1")
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_xlabel("Day")
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Drawdown (%) — LIVE T1")
    ax.grid(True, alpha=0.35)
    ax.text(
        daily["day_number"].max() + 0.05,
        float(daily["dd_pct"].iloc[-1]),
        "1 T1",
        color="#2f6fb5",
        fontsize=9,
        va="center",
    )
    return _save(fig, out / "drawdown_t1.png")


def plot_trade_count_per_day(df: pd.DataFrame, out: Path) -> Path:
    daily = df.groupby("day_number", as_index=False).size().rename(columns={"size": "trades"})
    # Fill missing days
    days = np.arange(1, int(daily["day_number"].max()) + 1)
    full = pd.DataFrame({"day_number": days}).merge(daily, on="day_number", how="left").fillna(0)
    full["trades"] = full["trades"].astype(int)
    full["ma7"] = full["trades"].rolling(7, min_periods=1).mean()
    full["ma30"] = full["trades"].rolling(30, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(full["day_number"], full["trades"], color="#b0b0b0", lw=1.4, label="Daily count")
    ax.plot(full["day_number"], full["ma7"], color="#1f77b4", lw=2.2, label="7-day MA")
    ax.plot(full["day_number"], full["ma30"], color="#8B0000", lw=2.2, label="30-day MA")
    ax.set_xlabel("Day")
    ax.set_ylabel("Trades entered")
    ax.set_title("Trade count per day + moving averages — LIVE T1")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper right")
    return _save(fig, out / "trade_count_per_day_t1.png")


def plot_winners_vs_losers_count(df: pd.DataFrame, out: Path) -> Path:
    wins = int(df["is_win"].sum())
    losses = int(df["is_loss"].sum())
    flats = int(df["is_flat"].sum())
    n = wins + losses  # sample titles use win+loss as trades/arm

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.array([0.0])
    w = 0.28
    bars_w = ax.bar(x - w / 2, [wins], w, color="#2ca02c", label="Winners")
    bars_l = ax.bar(x + w / 2, [losses], w, color="#d62728", label="Losers")
    for b, v in zip(bars_w, [wins]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, str(v), ha="center", va="bottom", fontsize=11, rotation=90)
    for b, v in zip(bars_l, [losses]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, str(v), ha="center", va="bottom", fontsize=11, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(["T1"])
    ax.set_ylabel("Number of trades")
    title = f"Winners vs losers count — {n} scored trades (excl. flat={flats})"
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.35)
    ax.set_ylim(0, max(wins, losses, 1) * 1.25)
    return _save(fig, out / "winners_vs_losers_count_t1.png")


def plot_profit_vs_loss_breakdown(df: pd.DataFrame, out: Path) -> Path:
    win_sum = float(df.loc[df["is_win"], "pnl_pct_equity"].sum())
    loss_sum = float(df.loc[df["is_loss"], "pnl_pct_equity"].sum())
    net = win_sum + loss_sum
    wins = int(df["is_win"].sum())
    losses = int(df["is_loss"].sum())
    day_n = int(df["day_number"].max())

    fig = plt.figure(figsize=(8.5, 6.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.2], hspace=0.35)
    ax = fig.add_subplot(gs[0])

    ax.axhline(0, color="black", lw=1.2)
    ax.bar([0], [win_sum], width=0.45, color="#2ca02c", label="Sum of winning trades")
    ax.bar([0], [loss_sum], width=0.45, color="#d62728", label="Sum of losing trades")
    ax.text(0, win_sum + 0.05 * max(abs(win_sum), abs(loss_sum), 0.1), f"+{win_sum:.2f}%", ha="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#2ca02c"))
    ax.text(0, loss_sum - 0.05 * max(abs(win_sum), abs(loss_sum), 0.1), f"{loss_sum:.2f}%", ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#d62728"))
    ax.set_xticks([0])
    ax.set_xticklabels(["T1"])
    ax.set_ylabel(f"% of starting equity ({STARTING_EQUITY_BTC:.5f} BTC)")
    ax.set_title(f"Profit vs loss breakdown (day {day_n})")
    ax.text(
        0.5,
        1.02,
        "Green = total from winners · Red = total from losers · Net is in the table below",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color="#444",
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.35)

    ax_t = fig.add_subplot(gs[1])
    ax_t.axis("off")
    cell_text = [[
        "T1",
        f"+{win_sum:.2f}%",
        f"{loss_sum:.2f}%",
        f"{net:+.2f}%",
        str(wins),
        str(losses),
    ]]
    table = ax_t.table(
        cellText=cell_text,
        colLabels=["Arm", "Winners", "Losers", "Net", "Win count", "Loss count"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.15, 1.6)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1f4e79")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#e8f1fb" if c % 2 == 0 else "white")
            if c == 1:
                cell.get_text().set_color("#2ca02c")
            elif c == 2:
                cell.get_text().set_color("#d62728")
            elif c == 3:
                cell.get_text().set_color("#1f77b4")

    ax_t.set_title("Exact values (% of starting equity)", fontsize=9, pad=2)
    return _save(fig, out / "profit_vs_loss_breakdown_t1.png", use_tight=False)


def plot_win_rate(df: pd.DataFrame, out: Path) -> Path:
    wins = int(df["is_win"].sum())
    losses = int(df["is_loss"].sum())
    n = wins + losses
    wr = 100.0 * wins / n if n else 0.0

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(["T1"], [wr], color="#4c78a8", width=0.55)
    ax.text(0, wr + 1.5, f"{wr:.1f}%", ha="center", va="bottom", fontsize=11)
    ax.set_ylim(0, max(55, wr + 10))
    ax.set_ylabel("Win rate %")
    ax.set_title("Win rate by arm — LIVE T1")
    ax.grid(True, axis="y", alpha=0.35)
    return _save(fig, out / "win_rate_t1.png")


def plot_accumulation_per_trade_pct(df: pd.DataFrame, out: Path) -> Path:
    """Per-trade cumulative return (%) of starting equity — same series as BTC path."""
    g = df.sort_values("exit_dt").copy()
    g["cum_pct"] = g["pnl_pct_equity"].cumsum()
    xs = list(range(1, len(g) + 1))
    ys = g["cum_pct"].tolist()

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(xs, ys, color="#1f4e79", marker="o", lw=1.8, ms=4, label="1 T1")
    ax.axhline(0, color="#888", lw=0.8)
    final = float(ys[-1])
    ax.axhline(final, color="#7eb6d9", lw=1.0, alpha=0.85)
    ax.text(0.02, final, f"{final:.2f}", transform=ax.get_yaxis_transform(), color="#5b9bd5", fontsize=9, va="bottom")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Cumulative return (%)")
    ax.set_title("Cumulative return — LIVE T1 (per trade)")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=9)
    return _save(fig, out / "btc_accumulation_per_trade_t1.png")


def plot_avg_pnl_per_trade(df: pd.DataFrame, out: Path) -> Path:
    """Average P/L per trade (%) — T1 live value; other sample arms shown at 0."""
    arms = [f"T{i}" for i in range(1, 13)] + list("ABCDEF")
    avg_t1 = float(df["pnl_pct_equity"].mean()) if len(df) else 0.0
    vals = {a: 0.0 for a in arms}
    vals["T1"] = avg_t1

    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(arms))
    colors = []
    for a in arms:
        if a != "T1":
            colors.append("#cccccc")
        elif avg_t1 >= 0:
            colors.append("#2e8b57")
        else:
            colors.append("#d62728")
    heights = [vals[a] for a in arms]
    # For negative avg, still draw bar from 0
    ax.bar(x, heights, color=colors, width=0.7)
    ax.axhline(0, color="black", lw=0.8)
    for i, a in enumerate(arms):
        v = vals[a]
        label = f"{v:.3f}"
        y = v if v >= 0 else v
        va = "bottom" if v >= 0 else "top"
        offset = 0.002 if v >= 0 else -0.002
        if a == "T1" or abs(v) > 1e-12:
            ax.text(i, y + offset, label, ha="center", va=va, fontsize=8, rotation=90)
        else:
            ax.text(i, 0.002, "0.000", ha="center", va="bottom", fontsize=7, rotation=90, color="#888")
    ax.set_xticks(x)
    ax.set_xticklabels(arms, rotation=45, ha="right")
    ax.set_ylabel("Avg P/L per trade (%)")
    ax.set_title("Average P/L per trade — LIVE T1")
    ax.grid(True, axis="y", alpha=0.35)
    # Headroom for labels
    ymax = max(0.05, abs(avg_t1) * 1.8, 0.01)
    if avg_t1 >= 0:
        ax.set_ylim(0, ymax)
    else:
        ax.set_ylim(-ymax, ymax * 0.15)
    return _save(fig, out / "avg_pnl_per_trade_t1.png")


def main() -> None:
    df = load_scored_trades()
    if df.empty:
        raise SystemExit("No scored closed trades found.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_cumulative_return(df, OUT_DIR),
        plot_accumulation_per_trade_pct(df, OUT_DIR),
        plot_avg_pnl_per_trade(df, OUT_DIR),
        plot_drawdown(df, OUT_DIR),
        plot_trade_count_per_day(df, OUT_DIR),
        plot_winners_vs_losers_count(df, OUT_DIR),
        plot_profit_vs_loss_breakdown(df, OUT_DIR),
        plot_win_rate(df, OUT_DIR),
    ]

    summary = {
        "n_scored": int(len(df)),
        "wins": int(df["is_win"].sum()),
        "losses": int(df["is_loss"].sum()),
        "flats": int(df["is_flat"].sum()),
        "net_btc": float(df["pnl_btc"].sum()),
        "net_pct": float(df["pnl_pct_equity"].sum()),
        "avg_pnl_pct_per_trade": float(df["pnl_pct_equity"].mean()),
        "starting_equity_btc": STARTING_EQUITY_BTC,
        "days": int(df["day_number"].max()),
        "plots": [str(p) for p in paths],
    }
    (OUT_DIR / "summary.json").write_text(
        __import__("json").dumps(summary, indent=2),
        encoding="utf-8",
    )
    print("Wrote plots to", OUT_DIR)
    for p in paths:
        print(" ", p.name)
    print(
        f"Summary: {summary['wins']}W/{summary['losses']}L "
        f"net={summary['net_btc']:.8f} BTC ({summary['net_pct']:.2f}%) "
        f"avg/trade={summary['avg_pnl_pct_per_trade']:.3f}% "
        f"over {summary['days']} day(s)"
    )


if __name__ == "__main__":
    main()
