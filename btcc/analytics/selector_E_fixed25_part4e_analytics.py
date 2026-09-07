"""Full analytics package for Part 4E dedicated Selector E fixed-25% run."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from btcc.sim.selector_E_adaptive_sizing import STARTING_CAPITAL_USD
from btcc.sim.selector_E_fixed25_part4e import PART4C_FIXED25_TARGETS, PART4C_TOL

logger = logging.getLogger(__name__)


def _md_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if df is None or df.empty:
        return "_(empty)_"
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(str(c) for c in cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append("—" if not np.isfinite(v) else format(float(v), floatfmt))
            elif v is None:
                cells.append("—")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _max_streak(mask: pd.Series) -> int:
    if mask.empty:
        return 0
    best = cur = 0
    for v in mask.astype(bool):
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def enrich_trade_level(trades_25: pd.DataFrame, trades_ref: pd.DataFrame) -> pd.DataFrame:
    ex = trades_25.copy()
    ex["opportunity_id"] = ex["opportunity_id"].astype(str)
    ref = trades_ref.copy()
    ref["opportunity_id"] = ref["opportunity_id"].astype(str)
    cols = [
        c
        for c in (
            "opportunity_id",
            "symbol",
            "strategy_key",
            "exit_reason",
            "entry_fill_price",
            "exit_fill_price",
            "fees_btc",
            "slippage_btc_approx",
            "gross_pnl_btc_approx",
            "net_pnl_btc",
            "pnl_pct",
            "holding_hours",
        )
        if c in ref.columns
    ]
    m = ex.merge(ref[cols], on="opportunity_id", how="left", suffixes=("", "_ref"))
    # Prefer ref prices/reasons when present
    if "exit_reason_ref" in m.columns:
        m["exit_reason"] = m["exit_reason_ref"].fillna(m.get("exit_reason"))
    if "strategy_key_ref" in m.columns:
        m["selected_T"] = m["strategy_key_ref"].fillna(m.get("strategy_key"))
    else:
        m["selected_T"] = m.get("strategy_key")
    m["selector"] = "E"
    m["entry_price"] = m.get("entry_fill_price")
    m["exit_price"] = m.get("exit_fill_price")
    m["entry_equity"] = m.get("equity_before")
    m["position_size"] = m.get("actual_notional")
    m["net_pnl"] = m.get("pnl_usd")
    m["trade_return_pct"] = m.get("pnl_pct") * 100.0
    m["concurrent_positions"] = m.get("n_open_after_entry")
    m["open_exposure"] = m.get("exposure_after_entry")
    # equity after already on executed exits
    if "equity_after" not in m.columns:
        m["equity_after"] = np.nan
    m = m.sort_values("exit_ts").reset_index(drop=True)
    m.insert(0, "trade_id", np.arange(1, len(m) + 1))
    return m


def build_portfolio_stats(ex: pd.DataFrame, summary: dict[str, Any]) -> dict[str, Any]:
    executed = ex[ex["executed"] == True].copy() if "executed" in ex.columns else ex  # noqa: E712
    executed = executed.sort_values("exit_ts")
    pnl = executed["pnl_usd"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    eq = STARTING_CAPITAL_USD + pnl.cumsum()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    # drawdown duration (trades)
    in_dd = dd < -1e-12
    max_dd_dur_trades = _max_streak(in_dd)

    stats = {
        "starting_equity_usd": STARTING_CAPITAL_USD,
        "final_equity_usd": float(eq.iloc[-1]) if len(eq) else STARTING_CAPITAL_USD,
        "total_return_pct": float(100.0 * (eq.iloc[-1] / STARTING_CAPITAL_USD - 1.0)) if len(eq) else 0.0,
        "n_opportunities": int(summary.get("n_opportunities", len(ex))),
        "n_executed": int(len(executed)),
        "n_fully_allocated": int((~executed["exposure_limited"]).sum()) if len(executed) else 0,
        "n_partially_allocated": int(executed["exposure_limited"].sum()) if len(executed) else 0,
        "n_skipped": int((ex["executed"] == False).sum()) if "executed" in ex.columns else 0,  # noqa: E712
        "win_rate_pct": float(100.0 * (pnl > 0).mean()) if len(pnl) else 0.0,
        "n_winners": int((pnl > 0).sum()),
        "n_losers": int((pnl < 0).sum()),
        "avg_trade_pnl_usd": float(pnl.mean()) if len(pnl) else 0.0,
        "median_trade_pnl_usd": float(pnl.median()) if len(pnl) else 0.0,
        "avg_trade_return_pct": float(executed["pnl_pct"].mean() * 100.0) if len(executed) else 0.0,
        "median_trade_return_pct": float(executed["pnl_pct"].median() * 100.0) if len(executed) else 0.0,
        "avg_winning_trade_usd": float(wins.mean()) if len(wins) else 0.0,
        "avg_losing_trade_usd": float(losses.mean()) if len(losses) else 0.0,
        "largest_winner_usd": float(wins.max()) if len(wins) else 0.0,
        "largest_loser_usd": float(losses.min()) if len(losses) else 0.0,
        "gross_profit_usd": float(wins.sum()) if len(wins) else 0.0,
        "gross_loss_usd": float(-losses.sum()) if len(losses) else 0.0,
        "profit_factor": float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else float("inf"),
        "expectancy_usd": float(pnl.mean()) if len(pnl) else 0.0,
        "max_consecutive_wins": _max_streak(pnl > 0),
        "max_consecutive_losses": _max_streak(pnl < 0),
        "max_drawdown_pct": float(100.0 * dd.min()) if len(dd) else 0.0,
        "max_drawdown_duration_trades": int(max_dd_dur_trades),
        "mean_actual_allocation": float(executed["actual_allocation"].mean()) if len(executed) else 0.0,
        "median_actual_allocation": float(executed["actual_allocation"].median()) if len(executed) else 0.0,
        "max_actual_allocation": float(executed["actual_allocation"].max()) if len(executed) else 0.0,
        "avg_open_exposure": float(executed["exposure_after_entry"].mean()) if len(executed) else 0.0,
        "max_open_exposure": float(executed["exposure_after_entry"].max()) if len(executed) else 0.0,
        "avg_concurrent_positions": float(executed["n_open_after_entry"].mean()) if len(executed) else 0.0,
        "max_concurrent_positions": int(executed["n_open_after_entry"].max()) if len(executed) else 0,
        "n_exposure_constrained": int(executed["exposure_limited"].sum()) if len(executed) else 0,
        "pct_exposure_constrained": float(100.0 * executed["exposure_limited"].mean()) if len(executed) else 0.0,
        "requested_allocation": 0.25,
    }
    return stats


def build_daily(ex: pd.DataFrame) -> pd.DataFrame:
    executed = ex[ex["executed"] == True].copy()  # noqa: E712
    executed["entry_ts"] = pd.to_datetime(executed["entry_ts"], utc=True)
    executed["exit_ts"] = pd.to_datetime(executed["exit_ts"], utc=True)
    executed = executed.sort_values("exit_ts")

    # Build day index spanning first entry to last exit
    start = executed["entry_ts"].min().floor("D")
    end = executed["exit_ts"].max().floor("D")
    days = pd.date_range(start, end, freq="D", tz="UTC")

    equity = STARTING_CAPITAL_USD
    peak = equity
    # Precompute exits by day and entries by day
    executed["exit_day"] = executed["exit_ts"].dt.floor("D")
    executed["entry_day"] = executed["entry_ts"].dt.floor("D")
    pnl_by_day = executed.groupby("exit_day")["pnl_usd"].sum()
    entries_by_day = executed.groupby("entry_day").size()
    exits_by_day = executed.groupby("exit_day").size()

    rows = []
    for d in days:
        # open positions / exposure snapshot at end of day (after applying exits that day)
        day_pnl = float(pnl_by_day.get(d, 0.0))
        start_eq = equity
        equity = start_eq + day_pnl
        peak = max(peak, equity)
        dd = 100.0 * (equity / peak - 1.0) if peak else 0.0

        # concurrent at end of day: entered on/before d, exit after d
        open_mask = (executed["entry_ts"] <= d + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)) & (
            executed["exit_ts"] > d + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        )
        # simpler: entry_day <= d and exit_day > d
        open_mask = (executed["entry_day"] <= d) & (executed["exit_day"] > d)
        open_pos = executed[open_mask]
        open_n = int(len(open_pos))
        open_exp = float(open_pos["actual_notional"].sum() / equity) if equity > 0 and open_n else 0.0

        rows.append(
            {
                "date": d,
                "starting_equity": start_eq,
                "ending_equity": equity,
                "daily_pnl": day_pnl,
                "daily_return_pct": 100.0 * day_pnl / start_eq if start_eq else 0.0,
                "number_of_entries": int(entries_by_day.get(d, 0)),
                "number_of_exits": int(exits_by_day.get(d, 0)),
                "open_positions": open_n,
                "open_exposure": open_exp,
                "drawdown_pct": dd,
            }
        )
    daily = pd.DataFrame(rows)
    return daily


def daily_summary(daily: pd.DataFrame) -> dict[str, Any]:
    r = daily["daily_return_pct"]
    return {
        "n_days": int(len(daily)),
        "profitable_days": int((daily["daily_pnl"] > 0).sum()),
        "losing_days": int((daily["daily_pnl"] < 0).sum()),
        "flat_days": int((daily["daily_pnl"] == 0).sum()),
        "best_day_pct": float(r.max()) if len(r) else 0.0,
        "worst_day_pct": float(r.min()) if len(r) else 0.0,
        "best_day": str(daily.loc[r.idxmax(), "date"]) if len(r) else None,
        "worst_day": str(daily.loc[r.idxmin(), "date"]) if len(r) else None,
        "avg_daily_return_pct": float(r.mean()) if len(r) else 0.0,
        "median_daily_return_pct": float(r.median()) if len(r) else 0.0,
        "daily_volatility_pct": float(r.std(ddof=1)) if len(r) > 1 else 0.0,
    }


def build_monthly(ex: pd.DataFrame) -> pd.DataFrame:
    executed = ex[ex["executed"] == True].copy()  # noqa: E712
    executed["exit_ts"] = pd.to_datetime(executed["exit_ts"], utc=True)
    executed = executed.sort_values("exit_ts")
    executed["month"] = executed["exit_ts"].dt.tz_localize(None).dt.to_period("M").astype(str)
    equity = STARTING_CAPITAL_USD
    rows = []
    months = list(executed.groupby("month", sort=True))
    for i, (month, g) in enumerate(months):
        start = equity
        pnl = float(g["pnl_usd"].sum())
        equity = start + pnl
        path = start + g["pnl_usd"].cumsum()
        peak = path.cummax()
        dd = 100.0 * (path / peak - 1.0)
        partial = "partial_first" if i == 0 else ("partial_last" if i == len(months) - 1 else "full")
        # crude: first/last may be incomplete calendar months
        rows.append(
            {
                "month": month,
                "month_label": partial,
                "starting_equity": start,
                "ending_equity": equity,
                "monthly_pnl": pnl,
                "monthly_return_pct": 100.0 * pnl / start if start else 0.0,
                "trades": int(len(g)),
                "wins": int((g["pnl_usd"] > 0).sum()),
                "losses": int((g["pnl_usd"] < 0).sum()),
                "max_drawdown_pct": float(dd.min()) if len(dd) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def check_part4c_match(stats: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "final_equity_usd": abs(stats["final_equity_usd"] - PART4C_FIXED25_TARGETS["final_equity_usd"])
        <= PART4C_TOL["equity"],
        "total_return_pct": abs(stats["total_return_pct"] - PART4C_FIXED25_TARGETS["total_return_pct"])
        <= PART4C_TOL["return_pp"],
        "max_drawdown_pct": abs(stats["max_drawdown_pct"] - PART4C_FIXED25_TARGETS["max_drawdown_pct"])
        <= PART4C_TOL["dd_pp"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "part4e": {
            "final_equity_usd": stats["final_equity_usd"],
            "total_return_pct": stats["total_return_pct"],
            "max_drawdown_pct": stats["max_drawdown_pct"],
        },
        "part4c_targets": PART4C_FIXED25_TARGETS,
    }


def plot_all(
    out_dir: Path,
    *,
    trade_level: pd.DataFrame,
    daily: pd.DataFrame,
    monthly: pd.DataFrame,
    stats10: dict[str, Any],
    stats25: dict[str, Any],
) -> dict[str, str]:
    plots = out_dir / "plots"
    paths = {}
    executed = trade_level[trade_level.get("executed", True) == True].copy()  # noqa: E712
    executed["exit_ts"] = pd.to_datetime(executed["exit_ts"], utc=True)
    executed = executed.sort_values("exit_ts")
    eq = executed["equity_after"] if "equity_after" in executed.columns else (
        STARTING_CAPITAL_USD + executed["pnl_usd"].cumsum()
    )
    ret = 100.0 * (eq / STARTING_CAPITAL_USD - 1.0)
    peak = eq.cummax()
    dd = 100.0 * (eq / peak - 1.0)
    ts = executed["exit_ts"]

    # 1 equity
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ts, eq, color="#1f77b4", lw=1.5)
    ax.axhline(STARTING_CAPITAL_USD, color="grey", ls="--", lw=0.8)
    ax.set_title("Selector E FIXED_25 — Equity curve")
    ax.set_ylabel("Equity (USD)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = plots / "equity_curve.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["equity_curve.png"] = str(p)

    # 2 cumulative return
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ts, ret, color="#2ca02c", lw=1.5)
    ax.axhline(0, color="grey", ls="--", lw=0.8)
    ax.set_title("Selector E FIXED_25 — Cumulative return %")
    ax.set_ylabel("Cumulative return %")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = plots / "cumulative_return.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["cumulative_return.png"] = str(p)

    # 3 drawdown
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(ts, dd, 0, color="#d62728", alpha=0.5)
    ax.plot(ts, dd, color="#d62728", lw=1.0)
    ax.set_title("Selector E FIXED_25 — Drawdown %")
    ax.set_ylabel("Drawdown %")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = plots / "drawdown_curve.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["drawdown_curve.png"] = str(p)

    # 4 daily pnl
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = np.where(daily["daily_return_pct"] >= 0, "#2ca02c", "#d62728")
    ax.bar(pd.to_datetime(daily["date"]), daily["daily_return_pct"], color=colors, width=1.0)
    ax.set_title("Daily return %")
    ax.set_ylabel("Daily return %")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p = plots / "daily_pnl.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["daily_pnl.png"] = str(p)

    # 5 monthly
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = np.where(monthly["monthly_return_pct"] >= 0, "#2ca02c", "#d62728")
    ax.bar(monthly["month"], monthly["monthly_return_pct"], color=colors)
    ax.axhline(0, color="grey", ls="--", lw=0.8)
    ax.set_title("Monthly returns %")
    ax.set_ylabel("Monthly return %")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p = plots / "monthly_returns.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["monthly_returns.png"] = str(p)

    # 6 trade pnl distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(executed["pnl_pct"] * 100.0, bins=50, color="steelblue", edgecolor="white")
    ax.axvline(0, color="black", ls="--", lw=0.8)
    ax.set_title("Trade return distribution (%)")
    ax.set_xlabel("Trade return %")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p = plots / "trade_pnl_distribution.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["trade_pnl_distribution.png"] = str(p)

    # 7 outcomes over time
    fig, ax = plt.subplots(figsize=(12, 4))
    win = executed["pnl_usd"] > 0
    ax.scatter(ts[win], executed.loc[win, "pnl_usd"], s=10, c="#2ca02c", alpha=0.6, label="Win")
    ax.scatter(ts[~win], executed.loc[~win, "pnl_usd"], s=10, c="#d62728", alpha=0.6, label="Loss")
    ax.axhline(0, color="grey", ls="--", lw=0.8)
    ax.set_title("Trade P/L over time")
    ax.set_ylabel("PnL (USD)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = plots / "trade_outcomes_over_time.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["trade_outcomes_over_time.png"] = str(p)

    # 8 exposure
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ts, executed["actual_allocation"] * 100, lw=1.0, label="Actual allocation %")
    if "exposure_after_entry" in executed.columns:
        ax.plot(ts, executed["exposure_after_entry"] * 100, lw=1.0, alpha=0.8, label="Open exposure %")
    ax.axhline(25, color="grey", ls="--", lw=0.8)
    ax.set_title("Allocation / exposure through time")
    ax.set_ylabel("%")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = plots / "allocation_exposure.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["allocation_exposure.png"] = str(p)

    # 9 concurrent
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.step(ts, executed["n_open_after_entry"], where="post", color="#9467bd")
    ax.set_title("Concurrent open positions at entry")
    ax.set_ylabel("Open positions")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = plots / "concurrent_positions.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["concurrent_positions.png"] = str(p)

    # 10 rolling returns from daily
    fig, ax = plt.subplots(figsize=(12, 5))
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.set_index("date")
    # compound rolling from daily returns
    r = d["daily_return_pct"] / 100.0
    roll7 = (1.0 + r).rolling(7).apply(lambda x: np.prod(x) - 1.0, raw=True) * 100.0
    roll30 = (1.0 + r).rolling(30).apply(lambda x: np.prod(x) - 1.0, raw=True) * 100.0
    ax.plot(roll7.index, roll7, label="Rolling 7d return %", lw=1.2)
    ax.plot(roll30.index, roll30, label="Rolling 30d return %", lw=1.2)
    ax.axhline(0, color="grey", ls="--", lw=0.8)
    ax.set_title("Rolling performance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = plots / "rolling_performance.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["rolling_performance.png"] = str(p)

    # comparison bar 10 vs 25
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["Final equity", "Return %", "|Max DD| %"]
    v10 = [stats10["final_equity_usd"], stats10["total_return_pct"], abs(stats10["max_drawdown_pct"])]
    v25 = [stats25["final_equity_usd"], stats25["total_return_pct"], abs(stats25["max_drawdown_pct"])]
    # normalize return/dd for dual axis is messy; two panels
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(["FIXED_10", "FIXED_25"], [stats10["final_equity_usd"], stats25["final_equity_usd"]], color=["#7f7f7f", "#1f77b4"])
    axes[0].set_title("Final equity")
    axes[0].set_ylabel("USD")
    axes[1].bar(["FIXED_10", "FIXED_25"], [stats10["total_return_pct"], stats25["total_return_pct"]], color=["#7f7f7f", "#1f77b4"])
    axes[1].set_title("Total return %")
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p = plots / "compare_fixed10_vs_fixed25.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["compare_fixed10_vs_fixed25.png"] = str(p)

    return paths


def write_report(
    out_dir: Path,
    *,
    result: dict[str, Any],
    stats25: dict[str, Any],
    stats10: dict[str, Any],
    daily_stats: dict[str, Any],
    monthly: pd.DataFrame,
    part4c_check: dict[str, Any],
    plot_paths: dict[str, str],
) -> Path:
    lines = []
    lines.append("# RESEARCH REPORT — Selector E Fixed 25% (Part 4E)")
    lines.append("")
    lines.append(f"**Generated:** {result.get('generated_at')}")
    lines.append(f"**Reference:** `{result.get('ref_dir')}`")
    lines.append(f"**Results:** `{out_dir}`")
    lines.append("")
    lines.append("## Strategy")
    lines.append("")
    lines.append("- Selector E = `rank_ewma`, half-life 7d, `S >= 0.60`")
    lines.append("- Fixed allocation = **25%** (Part 4C / frozen-reference accounting)")
    lines.append(f"- {result.get('sizing_convention')}")
    lines.append("")
    lines.append("## Reproducibility")
    lines.append("")
    lines.append(f"- FIXED_10 identity vs immutable reference: **{result['fixed10_check']['passed']}**")
    lines.append(f"- Opportunity/trade identity for FIXED_25 stream: **{result['identity']['passed']}**")
    lines.append(f"- Part 4C FIXED_25 reproduction: **{part4c_check['passed']}**")
    lines.append("")
    lines.append(_md_table(pd.DataFrame([
        {
            "metric": k,
            "part4e": part4c_check["part4e"][k],
            "part4c_target": part4c_check["part4c_targets"][k],
            "ok": part4c_check["checks"][k],
        }
        for k in ("final_equity_usd", "total_return_pct", "max_drawdown_pct")
    ])))
    lines.append("")
    lines.append("## Portfolio statistics (FIXED_25)")
    lines.append("")
    lines.append(_md_table(pd.DataFrame([stats25]).T.reset_index().rename(columns={"index": "metric", 0: "value"})))
    lines.append("")
    lines.append("## Daily summary")
    lines.append("")
    lines.append(_md_table(pd.DataFrame([daily_stats]).T.reset_index().rename(columns={"index": "metric", 0: "value"})))
    lines.append("")
    lines.append("## Monthly results")
    lines.append("")
    lines.append(_md_table(monthly))
    lines.append("")
    lines.append("## Comparison vs FIXED_10")
    lines.append("")
    lines.append("| Metric | FIXED_10 | FIXED_25 |")
    lines.append("| --- | ---: | ---: |")
    for k in (
        "final_equity_usd",
        "total_return_pct",
        "max_drawdown_pct",
        "n_executed",
        "profit_factor",
        "win_rate_pct",
        "pct_exposure_constrained",
    ):
        lines.append(f"| {k} | {stats10.get(k)} | {stats25.get(k)} |")
    lines.append("")
    lines.append("## Answers")
    lines.append("")
    lines.append(
        f"**A.** Over this historical year, $1,000 became **${stats25['final_equity_usd']:.2f}** "
        "with 25% allocation per trade."
    )
    lines.append("")
    lines.append(
        f"**B.** Historical backtest cumulative return ≈ **+{stats25['total_return_pct']:.2f}%** "
        "(not a forecast of future annual return)."
    )
    lines.append("")
    lines.append(f"**C.** Maximum drawdown ≈ **{stats25['max_drawdown_pct']:.3f}%**.")
    lines.append("")
    lines.append(
        f"**D.** Exposure-constrained trades: **{stats25['n_exposure_constrained']}** "
        f"({stats25['pct_exposure_constrained']:.3f}%)."
    )
    lines.append("")
    lines.append(
        f"**E.** Skipped: **{stats25['n_skipped']}**; partial: **{stats25['n_partially_allocated']}**; "
        f"full: **{stats25['n_fully_allocated']}**."
    )
    lines.append("")
    lines.append(
        f"**F.** vs FIXED_10 (+{stats10['total_return_pct']:.2f}%, DD {stats10['max_drawdown_pct']:.3f}%): "
        f"FIXED_25 adds **{stats25['total_return_pct']-stats10['total_return_pct']:+.2f} pp** return and "
        f"**{stats25['max_drawdown_pct']-stats10['max_drawdown_pct']:+.3f} pp** drawdown."
    )
    lines.append("")
    lines.append(
        f"**G.** Dedicated run reproduces Part 4C FIXED_25: **{part4c_check['passed']}**."
    )
    lines.append("")
    lines.append(
        "**H.** Part 4A `7D_INVERSE` (+156.2%, avg alloc ~21%) sits between FIXED_20 (~+140%) and "
        "FIXED_25 (~+175%), consistent with larger average size rather than unique adaptive skill."
    )
    lines.append("")
    lines.append("## Interpretation caution")
    lines.append("")
    lines.append(
        f"Over this **specific 365-day historical MEXC period**, the backtest produced approximately "
        f"**+{stats25['total_return_pct']:.1f}%** cumulative return when allocating 25% per trade. "
        "This is a historical backtest result, **not** a statement that 25% allocation will produce "
        "+175% per year going forward."
    )
    lines.append("")
    lines.append("## Paper/live")
    lines.append("")
    lines.append("**RESEARCH ONLY — NO DEPLOYMENT.** Paper allocation remains 10%.")
    lines.append("")
    lines.append("## Plots")
    lines.append("")
    for name, path in plot_paths.items():
        lines.append(f"- `{name}` → `{path}`")
    lines.append("")

    path = out_dir / "RESEARCH_REPORT_selector_E_fixed25_365d.md"
    path.write_text("\n".join(lines))
    return path


def emit_all(out_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    out_dir = Path(out_dir)
    t25 = result["trades_25"]
    t10 = result["trades_10"]
    s25 = result["summary_25"]
    s10 = result["summary_10"]

    if not result["identity"]["passed"]:
        raise RuntimeError(f"Identity check failed: {result['identity']}")
    if not result["fixed10_check"]["passed"]:
        raise RuntimeError("FIXED_10 does not match reference")

    trade_level = enrich_trade_level(t25, result["trades_ref"])
    stats25 = build_portfolio_stats(t25, s25)
    # rebuild FIXED_10 stats similarly for fair compare
    stats10 = build_portfolio_stats(t10, s10)
    daily = build_daily(t25)
    dstats = daily_summary(daily)
    # attach worst week from daily
    if len(daily) >= 7:
        r = daily["daily_return_pct"] / 100.0
        week = (1 + r).rolling(7).apply(lambda x: np.prod(x) - 1.0, raw=True) * 100.0
        dstats["worst_week_pct"] = float(week.min())
        dstats["best_week_pct"] = float(week.max())
    monthly = build_monthly(t25)
    if len(monthly):
        dstats["worst_month_pct"] = float(monthly["monthly_return_pct"].min())
        dstats["best_month_pct"] = float(monthly["monthly_return_pct"].max())
        stats25["worst_month"] = str(monthly.loc[monthly["monthly_return_pct"].idxmin(), "month"])
        stats25["worst_month_pct"] = float(monthly["monthly_return_pct"].min())

    part4c_check = check_part4c_match(stats25)
    if not part4c_check["passed"]:
        logger.error("Part 4C FIXED_25 mismatch: %s", part4c_check)

    # persist
    trade_level.to_csv(out_dir / "trade_level_results.csv", index=False)
    t25.to_csv(out_dir / "trades_FIXED_25.csv", index=False)
    daily.to_csv(out_dir / "daily_equity.csv", index=False)
    monthly.to_csv(out_dir / "monthly_results.csv", index=False)
    pd.DataFrame([stats25]).to_csv(out_dir / "portfolio_stats.csv", index=False)
    pd.DataFrame([stats10]).to_csv(out_dir / "portfolio_stats_FIXED_10.csv", index=False)
    pd.DataFrame([dstats]).to_csv(out_dir / "daily_summary.csv", index=False)

    plot_paths = plot_all(
        out_dir,
        trade_level=trade_level,
        daily=daily,
        monthly=monthly,
        stats10=stats10,
        stats25=stats25,
    )
    report = write_report(
        out_dir,
        result=result,
        stats25=stats25,
        stats10=stats10,
        daily_stats=dstats,
        monthly=monthly,
        part4c_check=part4c_check,
        plot_paths=plot_paths,
    )

    manifest = {
        "experiment": "selector_E_fixed25_365d_part4e",
        "out_dir": str(out_dir),
        "identity_passed": result["identity"]["passed"],
        "fixed10_passed": result["fixed10_check"]["passed"],
        "part4c_match": part4c_check,
        "stats25": stats25,
        "stats10": stats10,
        "daily_summary": dstats,
        "report": str(report),
        "plots": plot_paths,
        "sizing_convention": result.get("sizing_convention"),
        "generated_at": result.get("generated_at"),
    }
    (out_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (out_dir / "identity_check.json").write_text(json.dumps(result["identity"], indent=2))
    (out_dir / "part4c_reproduction_check.json").write_text(json.dumps(part4c_check, indent=2))
    return manifest
