"""Append-only daily checkpoints and FINAL_CHECKPOINT for walk-forward backtests."""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def day_dir(out_dir: Path, day_number: int) -> Path:
    d = out_dir / "daily_checkpoints" / f"day_{int(day_number):03d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def summarize_day(
    *,
    day_number: int,
    simulated_timestamp: str,
    pred_df: pd.DataFrame,
    opp_df: pd.DataFrame,
    legs_df: pd.DataFrame,
    weight_hist: pd.DataFrame,
    update_log: list[dict[str, Any]],
    schedule_active: dict[str, Any] | None,
    open_count: int,
    weight_mode: str,
    init_days: int,
    starting_capital_usd: float = 1000.0,
) -> dict[str, Any]:
    """Machine-readable daily summary (as-of end of day_number)."""
    day_pred = (
        pred_df[pred_df["day_number"] == day_number]
        if not pred_df.empty and "day_number" in pred_df.columns
        else pred_df.iloc[0:0]
    )
    rejected_all: dict[str, int] = {}
    if not pred_df.empty and "rejection_reason" in pred_df.columns:
        rejected_all = (
            pred_df["rejection_reason"].fillna("NONE").value_counts().astype(int).to_dict()
        )
    # Trading-facing rejections: strip BTC.D / health diagnostic labels
    rejected_trading = {
        k: v for k, v in rejected_all.items()
        if not any(tok in str(k).upper() for tok in ("BTC_D", "DIAG_", "HEALTH"))
    }
    summary: dict[str, Any] = {
        "day_number": int(day_number),
        "timestamp": simulated_timestamp,
        "weight_mode": weight_mode,
        "phase": "init" if day_number <= init_days else "daily",
        "n_predictions_today": int(len(day_pred)),
        "n_predictions_cum": int(len(pred_df)),
        "n_opportunities_cum": int(len(opp_df)),
        "n_legs_cum": int(len(legs_df)),
        "active_opportunities": int(open_count),
        "active_weight_version": (schedule_active or {}).get("version_id"),
        "current_weights": (schedule_active or {}).get("weights"),
        "n_weight_updates": len(update_log),
        "rejected_signals": rejected_trading,
        "rejected_signals_all": rejected_all,
    }
    if not day_pred.empty and "rejection_reason" in day_pred.columns:
        today_rej = (
            day_pred["rejection_reason"].fillna("NONE").value_counts().astype(int).to_dict()
        )
        summary["rejected_signals_today"] = {
            k: v for k, v in today_rej.items()
            if not any(tok in str(k).upper() for tok in ("BTC_D", "DIAG_", "HEALTH"))
        }

    # Did a weight update land on this simulated calendar day?
    weight_update_today = False
    try:
        day_date = pd.Timestamp(simulated_timestamp)
        if day_date.tzinfo is None:
            day_date = day_date.tz_localize("UTC")
        else:
            day_date = day_date.tz_convert("UTC")
        day_date = day_date.normalize()
        for u in update_log or []:
            ca = u.get("calculated_at") or u.get("effective_from")
            if not ca:
                continue
            ts = pd.Timestamp(ca)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            if ts.normalize() == day_date:
                weight_update_today = True
                break
    except Exception:
        weight_update_today = False
    summary["weight_update_occurred_today"] = bool(
        weight_mode == "adaptive" and weight_update_today
    )

    def _slice_metrics(
        legs: pd.DataFrame,
        opp: pd.DataFrame,
        *,
        policy: str | None = None,
        strategy: str | None = None,
    ) -> dict[str, Any]:
        g = legs
        o = opp
        if not g.empty and policy is not None and "entry_policy" in g.columns:
            g = g[g["entry_policy"] == policy]
        if not g.empty and strategy is not None and "strategy_key" in g.columns:
            g = g[g["strategy_key"] == strategy]
        if not o.empty and policy is not None and "entry_policy" in o.columns:
            o = o[o["entry_policy"] == policy]
        if not o.empty and strategy is not None and "strategy_key" in o.columns:
            o = o[o["strategy_key"] == strategy]
        closed = g
        if not closed.empty and "closed" in closed.columns:
            closed = closed[closed["closed"] == True]  # noqa: E712
        pnl = pd.to_numeric(closed["pnl_btc"], errors="coerce") if not closed.empty and "pnl_btc" in closed.columns else pd.Series(dtype=float)
        wins = int((pnl > 0).sum()) if len(pnl) else 0
        losses = int((pnl <= 0).sum()) if len(pnl) else 0
        n = int(len(closed))
        gp = float(pnl[pnl > 0].sum()) if len(pnl) else 0.0
        gl = float((-pnl[pnl < 0]).sum()) if len(pnl) else 0.0
        cum = float(pnl.sum()) if len(pnl) else 0.0
        # Drawdown from cumulative path
        dd = 0.0
        if len(pnl):
            c = pnl.fillna(0.0).cumsum()
            peak = c.cummax()
            dd = float((peak - c).max()) if len(c) else 0.0
        hold = None
        if not closed.empty and "entry_ts" in closed.columns and "exit_ts" in closed.columns:
            et = pd.to_datetime(closed["entry_ts"], utc=True, errors="coerce")
            xt = pd.to_datetime(closed["exit_ts"], utc=True, errors="coerce")
            hrs = (xt - et).dt.total_seconds() / 3600.0
            hold = float(hrs.mean()) if hrs.notna().any() else None
        return {
            "opportunities": int(len(o)),
            "trades": n,
            "wins": wins,
            "losses": losses,
            "win_rate": (wins / n) if n else None,
            "average_return": float(pnl.mean()) if n else None,
            "BTC_PnL": cum,
            "cumulative_BTC": cum,
            "drawdown": dd,
            "profit_factor": (gp / gl) if gl > 0 else (None if gp == 0 else float("inf")),
            "average_holding_time": hold,
            "active_opportunities": int(open_count) if policy is None and strategy is None else None,
            "rejected_signals": rejected_all if policy is None and strategy is None else None,
        }

    policies = ["NORMAL_FILTERED", "LATE_ENTRY_ALLOWED"]
    strategies = [
        "strategy_1", "strategy_2", "strategy_3", "strategy_4", "strategy_5",
        "S1", "S2", "S3", "S4", "S5",
    ]
    # Normalize strategy keys present in data
    present_sk: list[str] = []
    if not legs_df.empty and "strategy_key" in legs_df.columns:
        present_sk = sorted({str(x) for x in legs_df["strategy_key"].dropna().unique()})
    if not present_sk and not opp_df.empty and "strategy_key" in opp_df.columns:
        present_sk = sorted({str(x) for x in opp_df["strategy_key"].dropna().unique()})
    if not present_sk:
        present_sk = [
            "strategy_1", "strategy_2", "strategy_3", "strategy_4", "strategy_5",
        ]

    by: dict[str, Any] = {}
    present_pol: list[str] = []
    if not legs_df.empty and "entry_policy" in legs_df.columns:
        present_pol = sorted({str(x) for x in legs_df["entry_policy"].dropna().unique()})
    if not present_pol and not opp_df.empty and "entry_policy" in opp_df.columns:
        present_pol = sorted({str(x) for x in opp_df["entry_policy"].dropna().unique()})
    if not present_pol:
        present_pol = policies

    for pol in present_pol:
        by[str(pol)] = {}
        for sk in present_sk:
            by[str(pol)][str(sk)] = _slice_metrics(legs_df, opp_df, policy=str(pol), strategy=str(sk))
    summary["by_entry_policy_strategy"] = by
    summary["totals"] = _slice_metrics(legs_df, opp_df)

    if weight_mode == "adaptive":
        last = update_log[-1] if update_log else {}
        summary["adaptive"] = {
            "weight_version": (schedule_active or {}).get("version_id") or last.get("update_number"),
            "weights": (schedule_active or {}).get("weights") or last.get("weights"),
            "learning_window": (
                f"{last.get('learning_window_start')} → {last.get('learning_window_end')}"
                if last.get("learning_window_start")
                else None
            ),
            "learning_window_start": last.get("learning_window_start"),
            "learning_window_end": last.get("learning_window_end"),
            "matured_sample_count": last.get("n_samples"),
            "phase": last.get("phase"),
            "status": last.get("status"),
            "weight_update_occurred_today": bool(summary.get("weight_update_occurred_today")),
        }
    try:
        from btcc.analytics.capital import capital_daily_series

        cap = capital_daily_series(
            legs_df,
            starting_capital_usd=float(starting_capital_usd),
            max_day=int(day_number),
        )
        if not cap.empty:
            # Keep only the current day rows for the summary payload
            today = cap[cap["day_number"] == int(day_number)]
            summary["capital"] = {
                "starting_capital_usd": float(starting_capital_usd),
                "asof_day": int(day_number),
                "accounts": today.to_dict(orient="records"),
                "todays_realized_PnL_usd_sum": float(today["daily_PnL"].sum()) if not today.empty else 0.0,
            }
    except Exception:
        pass
    return summary


def write_daily_checkpoint(
    out_dir: Path,
    *,
    day_number: int,
    simulated_timestamp: str,
    pred_rows: list[dict],
    opp_rows: list[dict],
    leg_rows: list[dict],
    weight_hist: list[dict],
    update_log: list[dict],
    schedule_history: dict[str, Any],
    schedule_active: dict[str, Any] | None,
    open_books_snapshot: dict[str, Any],
    weight_mode: str,
    init_days: int,
    eval_start: str,
    eval_end: str,
    coverage: dict[str, Any] | None = None,
    refresh_analytics: bool = True,
    telegram_enabled: bool = False,
    starting_capital_usd: float = 1000.0,
) -> Path:
    """Persist append-only day checkpoint + refresh as-of analytics/plots."""
    if telegram_enabled:
        raise RuntimeError("Backtest daily checkpoints must keep Telegram OFF")

    out_dir = Path(out_dir)
    ddir = day_dir(out_dir, day_number)

    pred_df = pd.DataFrame(pred_rows)
    opp_df = pd.DataFrame(opp_rows)
    legs_df = pd.DataFrame(leg_rows)
    wh_df = pd.DataFrame(weight_hist)

    # Filter to as-of this day (no future leakage into daily artifacts)
    if not pred_df.empty and "day_number" in pred_df.columns:
        pred_asof = pred_df[pred_df["day_number"] <= day_number].copy()
    else:
        pred_asof = pred_df
    if not opp_df.empty and "day_number" in opp_df.columns:
        opp_asof = opp_df[opp_df["day_number"] <= day_number].copy()
    else:
        opp_asof = opp_df
    if not legs_df.empty and "day_number" in legs_df.columns:
        legs_asof = legs_df[legs_df["day_number"] <= day_number].copy()
    else:
        legs_asof = legs_df

    open_count = sum(len(v) for v in (open_books_snapshot or {}).values())
    daily = summarize_day(
        day_number=day_number,
        simulated_timestamp=simulated_timestamp,
        pred_df=pred_asof,
        opp_df=opp_asof,
        legs_df=legs_asof,
        weight_hist=wh_df,
        update_log=update_log,
        schedule_active=schedule_active,
        open_count=open_count,
        weight_mode=weight_mode,
        init_days=init_days,
        starting_capital_usd=starting_capital_usd,
    )
    daily["eval_start"] = eval_start
    daily["eval_end_requested"] = eval_end
    daily["open_books"] = open_books_snapshot
    daily["pair_coverage"] = coverage or {}

    meta = {
        "day_number": day_number,
        "simulated_timestamp": simulated_timestamp,
        "written_utc": datetime.now(timezone.utc).isoformat(),
        "weight_mode": weight_mode,
        "active_weight_version": (schedule_active or {}).get("version_id"),
        "current_weights": (schedule_active or {}).get("weights"),
        "init_days": init_days,
        "eval_start": eval_start,
    }
    (ddir / "checkpoint_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    (ddir / "daily_summary.json").write_text(json.dumps(daily, indent=2, default=str), encoding="utf-8")
    (ddir / "weight_schedule.json").write_text(json.dumps(schedule_history, indent=2, default=str), encoding="utf-8")
    (ddir / "weight_updates.json").write_text(json.dumps(update_log, indent=2, default=str), encoding="utf-8")
    # Do NOT duplicate full prediction/leg CSVs into every day folder (quadratic storage).
    # Root CSVs are rewritten each day with day_number columns → reconstruct any day N by filter.
    (ddir / "RECONSTRUCT.txt").write_text(
        "Filter out_dir/predictions.csv (and opportunities/strategy_legs) where day_number <= "
        f"{int(day_number)}. Weight schedule/updates for this day are stored alongside.\n",
        encoding="utf-8",
    )

    # Running root snapshots (overwrite latest view; day folders remain append-only summaries)
    _write_df(pred_asof, out_dir / "predictions.csv")
    _write_df(opp_asof, out_dir / "opportunities.csv")
    _write_df(legs_asof, out_dir / "strategy_legs.csv")
    _write_df(wh_df, out_dir / "weight_history.csv")
    (out_dir / "weight_updates.json").write_text(json.dumps(update_log, indent=2, default=str), encoding="utf-8")
    (out_dir / "weight_schedule.json").write_text(json.dumps(schedule_history, indent=2, default=str), encoding="utf-8")

    # Append index row
    idx_path = out_dir / "daily_checkpoints" / "index.jsonl"
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    with idx_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "day_number": day_number,
            "timestamp": simulated_timestamp,
            "path": str(ddir.relative_to(out_dir)),
            "n_predictions": len(pred_asof),
            "n_opportunities": len(opp_asof),
            "n_legs": len(legs_asof),
        }, default=str) + "\n")

    if refresh_analytics:
        try:
            from btcc.analytics.pipeline import build_arm_analytics_asof

            build_arm_analytics_asof(
                out_dir,
                arm=weight_mode,
                day_number=day_number,
                init_days=init_days,
                eval_start=eval_start,
                telegram_enabled=False,
                analytics_root=out_dir / "analytics",
                starting_capital_usd=starting_capital_usd,
            )
            # Snapshot as-of plots into the day folder (append-only history).
            src_plots = out_dir / "analytics" / "plots" / weight_mode
            if src_plots.exists():
                dst_plots = ddir / "plots"
                if dst_plots.exists():
                    shutil.rmtree(dst_plots)
                shutil.copytree(src_plots, dst_plots)
        except Exception:
            logger.exception("Daily analytics failed for day=%s", day_number)

    logger.info(
        "Daily checkpoint day=%03d ts=%s preds=%d opps=%d → %s",
        day_number, simulated_timestamp, len(pred_asof), len(opp_asof), ddir,
    )
    return ddir


def write_final_checkpoint(
    out_dir: Path,
    *,
    summary: dict[str, Any],
    fingerprint: dict[str, Any] | None = None,
    last_day_number: int | None = None,
    git_commit: str | None = None,
) -> Path:
    """Create FINAL_CHECKPOINT for live handoff (formal, not an arbitrary weights file)."""
    out_dir = Path(out_dir)
    final = out_dir / "FINAL_CHECKPOINT"
    if final.exists():
        shutil.rmtree(final)
    final.mkdir(parents=True, exist_ok=True)

    # Prefer last daily checkpoint contents
    dc = out_dir / "daily_checkpoints"
    last_day_dir = None
    if last_day_number is not None:
        cand = dc / f"day_{int(last_day_number):03d}"
        if cand.exists():
            last_day_dir = cand
    if last_day_dir is None and dc.exists():
        days = sorted(dc.glob("day_*"))
        if days:
            last_day_dir = days[-1]

    for name in (
        "predictions.csv",
        "opportunities.csv",
        "strategy_legs.csv",
        "weight_history.csv",
        "weight_updates.json",
        "weight_schedule.json",
        "summary.json",
        "report.md",
        "fingerprint.json",
        "pair_coverage.json",
        "fixed_weights.json",
    ):
        src = out_dir / name
        if src.exists():
            shutil.copy2(src, final / name)

    if last_day_dir is not None:
        (final / "source_daily_checkpoint").write_text(str(last_day_dir.name), encoding="utf-8")
        meta_p = last_day_dir / "checkpoint_meta.json"
        if meta_p.exists():
            shutil.copy2(meta_p, final / "last_day_checkpoint_meta.json")

    handoff = {
        "checkpoint_kind": "FINAL_CHECKPOINT",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_dir": str(out_dir),
        "last_processed_day": last_day_number,
        "last_processed_timestamp": summary.get("eval_end"),
        "final_weight_version": None,
        "final_weights": summary.get("seed_weights"),
        "git_commit": git_commit or (fingerprint or {}).get("git_commit"),
        "fingerprint": fingerprint,
        "summary_keys": sorted(summary.keys()),
        "initialized_from_historical_checkpoint": True,
    }
    # Prefer last adaptive weights from schedule
    sched_path = final / "weight_schedule.json"
    if sched_path.exists():
        sched = json.loads(sched_path.read_text(encoding="utf-8"))
        if isinstance(sched, list):
            updates = sched
        elif isinstance(sched, dict):
            updates = sched.get("updates") or sched.get("versions") or []
        else:
            updates = []
        if isinstance(updates, list) and updates:
            last = updates[-1]
            handoff["final_weight_version"] = last.get("version_id")
            handoff["final_weights"] = last.get("weights")
            handoff["final_weights_effective_from"] = last.get("effective_from")
            handoff["final_weights_calculated_at"] = last.get("calculated_at")

    (final / "handoff.json").write_text(json.dumps(handoff, indent=2, default=str), encoding="utf-8")
    (final / "README.md").write_text(
        "\n".join([
            "# FINAL_CHECKPOINT",
            "",
            "Formal handoff from historical walk-forward to live simulation.",
            "Do not replace with an arbitrary weights.json.",
            "",
            f"- git_commit: `{handoff.get('git_commit')}`",
            f"- last_processed_day: `{handoff.get('last_processed_day')}`",
            f"- final_weight_version: `{handoff.get('final_weight_version')}`",
            "",
        ]),
        encoding="utf-8",
    )
    logger.info("FINAL_CHECKPOINT written → %s", final)
    return final


def verify_final_checkpoint(final_dir: Path) -> dict[str, Any]:
    """Consistency checks for live initialization."""
    final_dir = Path(final_dir)
    result = {"ok": True, "errors": [], "path": str(final_dir)}
    handoff_p = final_dir / "handoff.json"
    if not handoff_p.exists():
        result["ok"] = False
        result["errors"].append("missing handoff.json")
        return result
    handoff = json.loads(handoff_p.read_text(encoding="utf-8"))
    if handoff.get("checkpoint_kind") != "FINAL_CHECKPOINT":
        result["ok"] = False
        result["errors"].append("invalid checkpoint_kind")
    if not handoff.get("git_commit"):
        result["errors"].append("missing git_commit")
    if not (final_dir / "weight_schedule.json").exists() and not handoff.get("final_weights"):
        result["ok"] = False
        result["errors"].append("missing weights")
    fp = final_dir / "fingerprint.json"
    if fp.exists() and handoff.get("git_commit"):
        fpj = json.loads(fp.read_text(encoding="utf-8"))
        if fpj.get("git_commit") and fpj.get("git_commit") != handoff.get("git_commit"):
            result["ok"] = False
            result["errors"].append("git_commit mismatch vs fingerprint.json")
    result["handoff"] = {
        "git_commit": handoff.get("git_commit"),
        "last_processed_day": handoff.get("last_processed_day"),
        "final_weight_version": handoff.get("final_weight_version"),
    }
    if result["errors"] and result["ok"]:
        # soft warnings only
        pass
    result["ok"] = result["ok"] and not any("mismatch" in e or e.startswith("missing weights") or e.startswith("invalid") for e in result["errors"])
    return result
