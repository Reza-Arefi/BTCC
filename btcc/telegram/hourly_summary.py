"""Hourly paper-trading Telegram summary stats for Selector E live."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd


def _is_cf_mask(legs: pd.DataFrame) -> pd.Series:
    if legs.empty or "is_counterfactual" not in legs.columns:
        return pd.Series(False, index=legs.index)
    return legs["is_counterfactual"].astype(str).str.lower().isin(["true", "1"])


def e_closed_legs(legs: pd.DataFrame) -> pd.DataFrame:
    if legs is None or legs.empty:
        return pd.DataFrame()
    is_cf = _is_cf_mask(legs)
    arm = legs["arm_key"] if "arm_key" in legs.columns else pd.Series("", index=legs.index)
    closed = legs["closed"].astype(str).str.lower().isin(["true", "1"]) if "closed" in legs.columns else True
    return legs[(~is_cf) & (arm == "E") & closed].copy()


def compute_hourly_paper_stats(
    *,
    legs: pd.DataFrame,
    open_opps: list[dict[str, Any]],
    starting_equity_btc: float,
    current_equity_btc: float,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Aggregate Selector-E paper stats since startup for the hourly Telegram digest."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    e_closed = e_closed_legs(legs)
    n_closed = int(len(e_closed))
    pnl = pd.to_numeric(e_closed["pnl_pct"], errors="coerce").fillna(0.0) if n_closed else pd.Series(dtype=float)
    winners = pnl[pnl > 0]
    losers = pnl[pnl < 0]
    flats = pnl[pnl == 0]

    open_book = [o for o in (open_opps or []) if o.get("status") in ("OPEN", "PENDING_ENTRY")]
    n_open = len(open_book)

    # Entries this UTC hour (open book + closed E with entry_ts in hour)
    hour_start = now.replace(minute=0, second=0, microsecond=0)
    opened_this_hour = 0
    for o in open_book:
        ts = pd.to_datetime(o.get("entry_fill_ts") or o.get("opened_ts") or o.get("signal_timestamp"), utc=True, errors="coerce")
        if pd.notna(ts) and ts >= hour_start:
            opened_this_hour += 1
    if n_closed and "entry_ts" in e_closed.columns:
        entry_ts = pd.to_datetime(e_closed["entry_ts"], utc=True, errors="coerce")
        opened_this_hour += int(((entry_ts >= hour_start) & (entry_ts <= now)).sum())

    n_entered = n_closed + n_open
    start = float(starting_equity_btc)
    cur = float(current_equity_btc)
    total_profit_btc = cur - start
    total_return_pct = (100.0 * (cur / start - 1.0)) if start > 0 else 0.0
    cum_pos_pct = float(winners.sum() * 100.0) if len(winners) else 0.0
    cum_neg_pct = float(losers.sum() * 100.0) if len(losers) else 0.0

    return {
        "asof_utc": now,
        "hour_bucket": hour_start.strftime("%Y-%m-%dT%H"),
        "n_entered": n_entered,
        "n_closed": n_closed,
        "n_winners": int(len(winners)),
        "n_losers": int(len(losers)),
        "n_flats": int(len(flats)),
        "cum_positive_pct": cum_pos_pct,
        "cum_negative_pct": cum_neg_pct,
        "total_profit_btc": total_profit_btc,
        "total_return_pct": total_return_pct,
        "starting_equity_btc": start,
        "current_equity_btc": cur,
        "n_open": n_open,
        "opened_this_hour": opened_this_hour,
    }


def format_hourly_paper_message(stats: dict[str, Any], *, version: dict[str, Any] | None = None) -> str:
    v = version or {}
    asof = stats["asof_utc"]
    if isinstance(asof, datetime):
        asof_s = asof.strftime("%Y-%m-%d %H:%M UTC")
    else:
        asof_s = str(asof)
    lines = [
        "⏱ HOURLY PAPER REPORT",
        "",
        f"As of: {asof_s}",
        f"Hour bucket: {stats.get('hour_bucket')}",
        "",
        f"Trades entered (since start): {stats['n_entered']}",
        f"  Closed: {stats['n_closed']} | Open now: {stats['n_open']}",
        f"Winners: {stats['n_winners']}",
        f"Losers: {stats['n_losers']}",
        f"Cumulative +% (sum winners): {stats['cum_positive_pct']:+.3f}%",
        f"Cumulative −% (sum losers): {stats['cum_negative_pct']:+.3f}%",
        "",
        f"Total profit: {stats['total_profit_btc']:+.8f} BTC ({stats['total_return_pct']:+.3f}%)",
        f"Starting equity: {stats['starting_equity_btc']:.8f} BTC",
        f"Current equity: {stats['current_equity_btc']:.8f} BTC",
        "",
        f"Open trades now: {stats['n_open']}",
        f"Opened this hour: {stats['opened_this_hour']}",
        "",
        "PAPER ONLY — NO REAL ORDERS",
        f"Bot: {v.get('bot_version')} | Selector: {v.get('selector_version')}",
    ]
    return "\n".join(str(x) for x in lines)
