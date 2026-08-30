"""Daily summary snapshots (append-only dated files)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def write_daily_snapshot(
    daily_dir: Path,
    *,
    arm: str,
    pred: pd.DataFrame,
    legs: pd.DataFrame,
    opp: pd.DataFrame,
    updates: list[dict],
    sim_stats: dict[str, Any],
    wl: pd.DataFrame,
    dd_stats: pd.DataFrame,
    telegram: bool = False,
    extra: dict[str, Any] | None = None,
) -> Path:
    daily_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    date = now.strftime("%Y-%m-%d")
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    # Never overwrite: unique filename per write
    base = f"daily_{arm}_{stamp}"

    today_pred = pred
    if not pred.empty and "timestamp" in pred.columns:
        ts = pd.to_datetime(pred["timestamp"], utc=True, errors="coerce")
        today_pred = pred[ts.dt.strftime("%Y-%m-%d") == date]

    signals_today = int(today_pred["signal_generated"].sum()) if not today_pred.empty and "signal_generated" in today_pred.columns else 0
    trades_today = int(today_pred["trade_opened"].sum()) if not today_pred.empty and "trade_opened" in today_pred.columns else 0
    rejected_today = int(today_pred["rejection_reason"].notna().sum()) if not today_pred.empty and "rejection_reason" in today_pred.columns else 0

    # Prediction accuracy on matured rows
    acc = None
    if not pred.empty and "future_return_4h" in pred.columns and "S" in pred.columns:
        m = pred.dropna(subset=["future_return_4h", "S"])
        if len(m):
            acc = float(((m["S"] >= 0) == (m["future_return_4h"] > 0)).mean())

    # 30d accuracy
    acc_30 = None
    if not pred.empty and "timestamp" in pred.columns:
        ts = pd.to_datetime(pred["timestamp"], utc=True, errors="coerce")
        cut = now - pd.Timedelta(days=30)
        m = pred[(ts >= cut) & pred["future_return_4h"].notna() & pred["S"].notna()] if "future_return_4h" in pred.columns else pred.iloc[0:0]
        if len(m):
            acc_30 = float(((m["S"] >= 0) == (m["future_return_4h"] > 0)).mean())

    update_n = None
    learn_window = None
    largest_up = largest_down = None
    if updates:
        ok = [u for u in updates if u.get("status") == "OK"]
        if ok:
            last = max(ok, key=lambda u: int(u.get("update_number") or 0))
            update_n = last.get("update_number")
            learn_window = f"{last.get('learning_window_start')} → {last.get('learning_window_end')}"
            old_w = last.get("old_weights") or {}
            new_w = last.get("weights") or {}
            deltas = {k: float(new_w.get(k, 0)) - float(old_w.get(k, 0)) for k in set(old_w) | set(new_w)}
            if deltas:
                largest_up = max(deltas.items(), key=lambda kv: kv[1])
                largest_down = min(deltas.items(), key=lambda kv: kv[1])

    payload: dict[str, Any] = {
        "date_utc": date,
        "generated_utc": now.isoformat(),
        "arm": arm,
        "telegram_would_send": bool(telegram),
        "adaptive_update_number": update_n,
        "open_opportunities": sim_stats.get("max_simultaneous"),
        "avg_simultaneous": sim_stats.get("avg_simultaneous"),
        "max_open_rejects": sim_stats.get("n_rejected_max_open_trades"),
        "signals_today": signals_today,
        "trades_today": trades_today,
        "rejected_today": rejected_today,
        "prediction_accuracy": acc,
        "prediction_accuracy_30d": acc_30,
        "learning_window": learn_window,
        "largest_weight_increase": largest_up,
        "largest_weight_decrease": largest_down,
        "win_loss": wl.to_dict(orient="records") if wl is not None and not wl.empty else [],
        "drawdown": dd_stats.to_dict(orient="records") if dd_stats is not None and not dd_stats.empty else [],
        "extra": extra or {},
    }
    json_path = daily_dir / f"{base}.json"
    md_path = daily_dir / f"{base}.md"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    lines = [
        f"# Daily summary — {arm} — {date}",
        "",
        f"- Generated UTC: {now.isoformat()}",
        f"- Adaptive update #: {update_n}",
        f"- Signals today: {signals_today}",
        f"- Trades today: {trades_today}",
        f"- Rejected today: {rejected_today}",
        f"- Prediction accuracy (all matured): {acc}",
        f"- Prediction accuracy 30d: {acc_30}",
        f"- Learning window: {learn_window}",
        f"- Largest weight ↑: {largest_up}",
        f"- Largest weight ↓: {largest_down}",
        f"- Simult. max / avg: {sim_stats.get('max_simultaneous')} / {sim_stats.get('avg_simultaneous')}",
        f"- MAX_OPEN rejects: {sim_stats.get('n_rejected_max_open_trades')}",
        "",
        "## Win/loss by strategy",
        "",
    ]
    if wl is not None and not wl.empty:
        lines.append(wl.to_string(index=False))
    lines += ["", "## Drawdown", ""]
    if dd_stats is not None and not dd_stats.empty:
        lines.append(dd_stats.to_string(index=False))
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path
