"""Analytics pipeline — shared entry points for backtest ABC and live."""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from btcc.analytics import metrics as M
from btcc.analytics import plots as P
from btcc.analytics.capital import capital_daily_series
from btcc.analytics.daily import write_daily_snapshot

logger = logging.getLogger(__name__)

ARM_ORDER = ("static", "equal", "adaptive")


def _ensure_layout(root: Path) -> dict[str, Path]:
    dirs = {
        "root": root,
        "metrics": root / "metrics",
        "plots": root / "plots",
        "daily_snapshots": root / "daily_snapshots",
        "reports": root / "reports",
        "predictions": root / "predictions",
        "opportunities": root / "opportunities",
        "weights": root / "weights",
        "data_summary": root / "data_summary",
        "config_snapshot": root / "config_snapshot",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def build_arm_analytics(
    arm_dir: Path | str,
    *,
    arm: str,
    analytics_root: Path | str | None = None,
    max_open: int = 10,
    telegram_enabled: bool = False,
    init_days: int | None = 90,
    starting_capital_usd: float = 1000.0,
) -> Path:
    """Build metrics+plots for one arm into analytics_root (or arm_dir/analytics)."""
    arm_dir = Path(arm_dir)
    root = Path(analytics_root) if analytics_root else arm_dir / "analytics"
    dirs = _ensure_layout(root)
    tables = M.load_arm_tables(arm_dir)
    pred, opp, legs, wh = tables["predictions"], tables["opportunities"], tables["legs"], tables["weight_history"]

    # Organize source copies
    for name, df in (("predictions", pred), ("opportunities", opp), ("weights", wh)):
        if not df.empty:
            _save_df(df, dirs[name] / f"{arm}_{name}.csv")
    if not legs.empty:
        _save_df(legs, dirs["opportunities"] / f"{arm}_strategy_legs.csv")

    # Metrics
    cum = M.cumulative_btc(legs)
    wr = M.rolling_win_rate(legs, window_days=30)
    wh_ts = M.weight_timeseries(wh)
    corr_full = M.indicator_predictive_correlation(pred)
    corr_roll = M.rolling_indicator_correlation(pred, window_days=30, step_days=7)
    wvu = M.weight_vs_usefulness(wh, corr_roll) if arm == "adaptive" else pd.DataFrame()
    svr = M.score_vs_return(pred)
    buckets = M.accuracy_by_score_bucket(pred)
    wl = M.win_loss_summary(legs)
    dd = M.drawdown_series(legs)
    dd_stats = M.drawdown_stats(dd)
    sim = M.simultaneous_open_series(opp, legs, max_open=max_open)
    sim_stats = M.simultaneous_stats(sim, pred)
    rej = M.rejection_breakdown(pred)
    monthly = M.monthly_performance(pred, legs, opp)
    btcd = M.btcd_regime_analysis(pred)

    updates = []
    upd_path = arm_dir / "weight_updates.json"
    if upd_path.exists():
        updates = json.loads(upd_path.read_text(encoding="utf-8"))
    learn = M.learning_progress(pred, legs, updates) if arm == "adaptive" else pd.DataFrame()

    metric_files = {
        "cumulative_btc": cum,
        "rolling_win_rate": wr,
        "weight_timeseries": wh_ts,
        "indicator_correlation_full": corr_full,
        "indicator_correlation_rolling": corr_roll,
        "weight_vs_usefulness": wvu,
        "score_vs_return": svr,
        "accuracy_by_score_bucket": buckets,
        "win_loss_summary": wl,
        "drawdown_series": dd,
        "drawdown_stats": dd_stats,
        "simultaneous_open": sim,
        "rejection_breakdown": rej,
        "monthly_performance": monthly,
        "btcd_regime": btcd,
        "learning_progress": learn,
    }
    for name, df in metric_files.items():
        if df is not None and not df.empty:
            _save_df(df, dirs["metrics"] / f"{arm}_{name}.csv")

    (dirs["metrics"] / f"{arm}_simultaneous_stats.json").write_text(
        json.dumps(sim_stats, indent=2), encoding="utf-8"
    )
    (dirs["data_summary"] / f"{arm}_summary_meta.json").write_text(
        json.dumps({
            "arm": arm,
            "telegram_enabled": telegram_enabled,
            "n_predictions": len(pred),
            "n_opportunities": len(opp),
            "n_legs": len(legs),
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        }, indent=2),
        encoding="utf-8",
    )

    # Plots (arm-local) — Day-number x-axis; mark init→daily transition when known
    plot_dir = dirs["plots"] / arm
    plot_dir.mkdir(parents=True, exist_ok=True)
    P.plot_btc_accumulation_by_strategy({arm: cum}, plot_dir, init_days=init_days)
    P.plot_rolling_win_rate({arm: wr}, plot_dir, init_days=init_days)
    if arm == "adaptive":
        P.plot_adaptive_weights(wh_ts, plot_dir, init_days=init_days)
        P.plot_weight_vs_usefulness(wvu, plot_dir)
        P.plot_learning_progress(learn, plot_dir)
    P.plot_indicator_correlations(corr_roll if not corr_roll.empty else corr_full, plot_dir)
    P.plot_score_vs_return(svr, plot_dir)
    P.plot_accuracy_buckets(buckets, plot_dir)
    P.plot_drawdown({arm: dd}, plot_dir, init_days=init_days)
    P.plot_simultaneous(sim, plot_dir, max_open=max_open, init_days=init_days)
    P.plot_rejections(rej, plot_dir)
    P.plot_monthly(monthly, plot_dir, arm)
    P.plot_btcd_regimes(btcd, plot_dir)
    P.plot_win_loss_bars({arm: wl}, plot_dir)
    P.plot_entry_policy_btc(legs, plot_dir)
    P.plot_entry_policy_win_rate(legs, plot_dir)
    P.plot_recovered_trades(legs, plot_dir)
    P.plot_entry_classification_scores(pred, plot_dir)
    P.plot_entry_future_returns(pred, plot_dir)

    # $1,000 compounded capital curves (per entry policy × strategy within this arm)
    cap_daily = capital_daily_series(legs, starting_capital_usd=starting_capital_usd)
    if not cap_daily.empty:
        _save_df(cap_daily, dirs["metrics"] / f"{arm}_capital_daily.csv")
        for pol in sorted(cap_daily["entry_policy"].dropna().unique()):
            sub = cap_daily[cap_daily["entry_policy"] == pol]
            label = f"{arm}/{pol}"
            for sk in ("strategy_1", "strategy_2", "strategy_3"):
                P.plot_compounded_capital(
                    {label: sub}, plot_dir, strategy_key=sk,
                    starting_capital_usd=starting_capital_usd, init_days=init_days,
                )
                P.plot_cumulative_pl_pct(
                    {label: sub}, plot_dir, strategy_key=sk,
                    starting_capital_usd=starting_capital_usd, init_days=init_days,
                )

    write_daily_snapshot(
        dirs["daily_snapshots"],
        arm=arm,
        pred=pred,
        legs=legs,
        opp=opp,
        updates=updates,
        sim_stats=sim_stats,
        wl=wl,
        dd_stats=dd_stats,
        telegram=False,
    )
    logger.info("Arm analytics written → %s (telegram=%s)", root, telegram_enabled)
    return root


def build_arm_analytics_asof(
    arm_dir: Path | str,
    *,
    arm: str,
    day_number: int,
    init_days: int,
    eval_start: str | None = None,
    analytics_root: Path | str | None = None,
    max_open: int = 10,
    telegram_enabled: bool = False,
    starting_capital_usd: float = 1000.0,
) -> Path:
    """Rebuild arm analytics/plots using only data with day_number <= day_number."""
    if telegram_enabled:
        raise RuntimeError("Backtest analytics must not enable Telegram")
    from btcc.sim.day_axis import attach_day_number

    arm_dir = Path(arm_dir)
    root = Path(analytics_root) if analytics_root else arm_dir / "analytics"
    dirs = _ensure_layout(root)
    tables = M.load_arm_tables(arm_dir)
    pred, opp, legs, wh = tables["predictions"], tables["opportunities"], tables["legs"], tables["weight_history"]

    if eval_start:
        if not pred.empty and "day_number" not in pred.columns and "timestamp" in pred.columns:
            pred = attach_day_number(pred, "timestamp", eval_start)
        if not opp.empty and "day_number" not in opp.columns:
            ts_col = "opened_ts" if "opened_ts" in opp.columns else "signal_timestamp"
            if ts_col in opp.columns:
                opp = attach_day_number(opp, ts_col, eval_start)
        if not legs.empty and "day_number" not in legs.columns and "exit_ts" in legs.columns:
            legs = attach_day_number(legs, "exit_ts", eval_start)
        if not wh.empty and "day_number" not in wh.columns and "update_timestamp" in wh.columns:
            wh = attach_day_number(wh, "update_timestamp", eval_start)

    if not pred.empty and "day_number" in pred.columns:
        pred = pred[pred["day_number"] <= day_number].copy()
    if not opp.empty and "day_number" in opp.columns:
        opp = opp[opp["day_number"] <= day_number].copy()
    if not legs.empty and "day_number" in legs.columns:
        legs = legs[legs["day_number"] <= day_number].copy()
    if not wh.empty and "day_number" in wh.columns:
        wh = wh[wh["day_number"] <= day_number].copy()

    # Temporarily write filtered tables into a scratch view under analytics for metric loaders
    # Metrics functions take DataFrames directly below.
    cum = M.cumulative_btc(legs)
    wr = M.rolling_win_rate(legs, window_days=30)
    wh_ts = M.weight_timeseries(wh)
    corr_full = M.indicator_predictive_correlation(pred)
    corr_roll = M.rolling_indicator_correlation(pred, window_days=30, step_days=7)
    wvu = M.weight_vs_usefulness(wh, corr_roll) if arm == "adaptive" else pd.DataFrame()
    svr = M.score_vs_return(pred)
    buckets = M.accuracy_by_score_bucket(pred)
    wl = M.win_loss_summary(legs)
    dd = M.drawdown_series(legs)
    dd_stats = M.drawdown_stats(dd)
    sim = M.simultaneous_open_series(opp, legs, max_open=max_open)
    if eval_start and not sim.empty:
        sim = attach_day_number(sim, "timestamp", eval_start)
        sim = sim[sim["day_number"] <= day_number] if "day_number" in sim.columns else sim
    sim_stats = M.simultaneous_stats(sim, pred)
    rej = M.rejection_breakdown(pred)
    monthly = M.monthly_performance(pred, legs, opp)
    btcd = M.btcd_regime_analysis(pred)
    updates = []
    upd_path = arm_dir / "weight_updates.json"
    if upd_path.exists():
        updates = json.loads(upd_path.read_text(encoding="utf-8"))
    learn = M.learning_progress(pred, legs, updates) if arm == "adaptive" else pd.DataFrame()

    for name, df in (
        ("cumulative_btc", cum),
        ("rolling_win_rate", wr),
        ("weight_timeseries", wh_ts),
        ("drawdown_series", dd),
        ("simultaneous_open", sim),
        ("rejection_breakdown", rej),
    ):
        if df is not None and not df.empty:
            _save_df(df, dirs["metrics"] / f"{arm}_{name}.csv")

    plot_dir = dirs["plots"] / arm
    plot_dir.mkdir(parents=True, exist_ok=True)
    P.plot_btc_accumulation_by_strategy({arm: cum}, plot_dir, init_days=init_days)
    P.plot_rolling_win_rate({arm: wr}, plot_dir, init_days=init_days)
    if arm == "adaptive":
        P.plot_adaptive_weights(wh_ts, plot_dir, init_days=init_days)
        P.plot_weight_vs_usefulness(wvu, plot_dir)
        P.plot_learning_progress(learn, plot_dir)
    P.plot_indicator_correlations(corr_roll if not corr_roll.empty else corr_full, plot_dir)
    P.plot_score_vs_return(svr, plot_dir)
    P.plot_accuracy_buckets(buckets, plot_dir)
    P.plot_drawdown({arm: dd}, plot_dir, init_days=init_days)
    P.plot_simultaneous(sim, plot_dir, max_open=max_open, init_days=init_days)
    P.plot_rejections(rej, plot_dir)
    P.plot_monthly(monthly, plot_dir, arm)
    P.plot_btcd_regimes(btcd, plot_dir)
    P.plot_win_loss_bars({arm: wl}, plot_dir)
    P.plot_entry_policy_btc(legs, plot_dir)
    P.plot_entry_policy_win_rate(legs, plot_dir)
    P.plot_recovered_trades(legs, plot_dir)

    cap_daily = capital_daily_series(
        legs, starting_capital_usd=starting_capital_usd, max_day=day_number,
    )
    if not cap_daily.empty:
        _save_df(cap_daily, dirs["metrics"] / f"{arm}_capital_daily.csv")
        for pol in sorted(cap_daily["entry_policy"].dropna().unique()):
            sub = cap_daily[cap_daily["entry_policy"] == pol]
            label = f"{arm}/{pol}"
            for sk in ("strategy_1", "strategy_2", "strategy_3"):
                P.plot_compounded_capital(
                    {label: sub}, plot_dir, strategy_key=sk,
                    starting_capital_usd=starting_capital_usd, init_days=init_days,
                )
                P.plot_cumulative_pl_pct(
                    {label: sub}, plot_dir, strategy_key=sk,
                    starting_capital_usd=starting_capital_usd, init_days=init_days,
                )

    snap = {
        "day_number": day_number,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "arm": arm,
        "init_days": init_days,
        "n_predictions": len(pred),
        "n_opportunities": len(opp),
        "n_legs": len(legs),
        "sim_stats": sim_stats,
        "telegram_enabled": False,
    }
    snap_dir = dirs["daily_snapshots"] / f"day_{int(day_number):03d}"
    snap_dir.mkdir(parents=True, exist_ok=True)
    (snap_dir / "snapshot.json").write_text(json.dumps(snap, indent=2, default=str), encoding="utf-8")
    logger.info("As-of analytics day=%03d arm=%s → %s", day_number, arm, root)
    return root


def build_abc_analytics(
    cmp_dir: Path | str,
    *,
    max_open: int = 10,
    telegram_enabled: bool = False,
    init_days: int | None = 90,
    starting_capital_usd: float = 1000.0,
) -> Path:
    """Build comparison analytics for an ABC result directory."""
    cmp_dir = Path(cmp_dir)
    root = cmp_dir / "analytics"
    dirs = _ensure_layout(root)

    if telegram_enabled:
        raise RuntimeError("Backtest analytics must not enable Telegram (telegram_enabled=False required)")

    arm_dirs: dict[str, Path] = {}
    for arm in ARM_ORDER:
        matches = list(cmp_dir.glob(f"adaptive_v2_{arm}_*"))
        if not matches:
            logger.warning("Missing arm dir for %s under %s", arm, cmp_dir)
            continue
        arm_dirs[arm] = sorted(matches)[-1]
        # Prefer init_days recorded on the arm summary when available
        arm_init = init_days
        sum_path = arm_dirs[arm] / "summary.json"
        if sum_path.exists():
            try:
                sj = json.loads(sum_path.read_text(encoding="utf-8"))
                if sj.get("init_days") is not None:
                    arm_init = int(sj["init_days"])
            except Exception:
                pass
        build_arm_analytics(
            arm_dirs[arm],
            arm=arm,
            analytics_root=root / "arms" / arm,
            max_open=max_open,
            telegram_enabled=False,
            init_days=arm_init,
            starting_capital_usd=starting_capital_usd,
        )

    # Comparison overlays
    cum_by_arm = {}
    wr_by_arm = {}
    dd_by_arm = {}
    wl_by_arm = {}
    for arm, ad in arm_dirs.items():
        tables = M.load_arm_tables(ad)
        cum_by_arm[arm] = M.cumulative_btc(tables["legs"])
        wr_by_arm[arm] = M.rolling_win_rate(tables["legs"])
        dd_by_arm[arm] = M.drawdown_series(tables["legs"])
        wl_by_arm[arm] = M.win_loss_summary(tables["legs"])
        _save_df(cum_by_arm[arm], dirs["metrics"] / f"compare_cumulative_btc_{arm}.csv")

    P.plot_btc_accumulation_by_strategy(cum_by_arm, dirs["plots"], init_days=init_days)
    P.plot_rolling_win_rate(wr_by_arm, dirs["plots"], init_days=init_days)
    P.plot_drawdown(dd_by_arm, dirs["plots"], init_days=init_days)
    P.plot_win_loss_bars(wl_by_arm, dirs["plots"])

    for sk in ("strategy_1", "strategy_2", "strategy_3"):
        adv = M.adaptive_advantage_series(cum_by_arm, sk)
        if not adv.empty:
            _save_df(adv, dirs["metrics"] / f"adaptive_advantage_{sk}.csv")
            P.plot_adaptive_advantage(adv, dirs["plots"], sk)

    # Cross-arm capital overlays: Static/Equal/Adaptive × Normal/Late per strategy
    cap_labels: dict[str, pd.DataFrame] = {}
    for arm, ad in arm_dirs.items():
        tables = M.load_arm_tables(ad)
        cd = capital_daily_series(tables["legs"], starting_capital_usd=starting_capital_usd)
        if cd.empty:
            continue
        _save_df(cd, dirs["metrics"] / f"compare_capital_daily_{arm}.csv")
        for pol in sorted(cd["entry_policy"].dropna().unique()):
            cap_labels[f"{arm}/{pol}"] = cd[cd["entry_policy"] == pol]
    if cap_labels:
        for sk in ("strategy_1", "strategy_2", "strategy_3"):
            P.plot_compounded_capital(
                cap_labels, dirs["plots"], strategy_key=sk,
                starting_capital_usd=starting_capital_usd, init_days=init_days,
            )
            P.plot_cumulative_pl_pct(
                cap_labels, dirs["plots"], strategy_key=sk,
                starting_capital_usd=starting_capital_usd, init_days=init_days,
            )

    # Combined daily snapshot (all arms)
    write_daily_snapshot(
        dirs["daily_snapshots"],
        arm="abc",
        pred=pd.DataFrame(),
        legs=pd.DataFrame(),
        opp=pd.DataFrame(),
        updates=[],
        sim_stats={},
        wl=pd.DataFrame(),
        dd_stats=pd.DataFrame(),
        telegram=False,
        extra={
            "arms": list(arm_dirs.keys()),
            "note": "Per-arm daily detail under analytics/arms/*/daily_snapshots/",
            "telegram_policy": "DISABLED_IN_BACKTEST",
        },
    )

    report = {
        "mode": "backtest_abc",
        "telegram": "DISABLED",
        "cmp_dir": str(cmp_dir),
        "analytics_root": str(root),
        "arms": {k: str(v) for k, v in arm_dirs.items()},
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "plots": sorted(p.name for p in dirs["plots"].glob("*.png")),
    }
    (dirs["reports"] / "analytics_index.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (dirs["reports"] / "analytics_index.md").write_text(
        _index_md(report, dirs), encoding="utf-8"
    )
    logger.info("ABC analytics complete → %s", root)
    return root


def update_live_analytics(
    *,
    live_root: Path | str | None = None,
    sim_storage: dict[str, Any] | None = None,
    max_open: int = 10,
    send_daily_telegram: bool = True,
    tg_send=None,
) -> Path:
    """Refresh live analytics from persistent sim CSVs (append-only snapshots)."""
    from btcc.sim.config import load_sim_config

    sim = load_sim_config()
    storage = sim_storage or (sim.get("storage") or {})
    project = Path(sim.get("_root") or Path(__file__).resolve().parents[2])
    live_root = Path(live_root) if live_root else project / "data" / "live_analytics"
    dirs = _ensure_layout(live_root)

    # Materialize a pseudo arm dir from live store files
    arm_dir = dirs["root"] / "_live_source"
    arm_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "predictions.csv": storage.get("predictions_path", "data/sim/predictions.csv"),
        "opportunities.csv": storage.get("opportunities_path", "data/sim/opportunities.csv"),
        "strategy_legs.csv": storage.get("strategy_legs_path", "data/sim/strategy_legs.csv"),
        "weight_history.csv": storage.get("weight_history_path", "data/sim/weight_history.csv"),
    }
    for dest, src in mapping.items():
        sp = Path(src)
        if not sp.is_absolute():
            sp = project / sp
        if sp.exists():
            shutil.copy2(sp, arm_dir / dest)

    # weight updates log
    dup = storage.get("daily_update_log_path")
    updates = []
    if dup:
        dp = Path(dup)
        if not dp.is_absolute():
            dp = project / dp
        if dp.exists():
            for line in dp.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        updates.append(json.loads(line))
                    except Exception:
                        pass
            (arm_dir / "weight_updates.json").write_text(json.dumps(updates, indent=2, default=str), encoding="utf-8")

    build_arm_analytics(
        arm_dir,
        arm="adaptive",
        analytics_root=live_root,
        max_open=max_open,
        telegram_enabled=True,
    )

    # Dated snapshot copy (never overwrite)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    snap = dirs["daily_snapshots"] / f"live_refresh_{stamp}"
    snap.mkdir(parents=True, exist_ok=True)
    for sub in ("metrics", "plots"):
        src = dirs[sub]
        if src.exists():
            dst = snap / sub
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

    if send_daily_telegram and tg_send:
        summary_path = sorted(dirs["daily_snapshots"].glob("daily_*.md"))
        if summary_path:
            text = summary_path[-1].read_text(encoding="utf-8")[:3500]
            try:
                tg_send(f"BTCC daily analytics summary\n\n{text}")
            except Exception as e:
                logger.error("Daily telegram failed: %s", e)

    return live_root


def _index_md(report: dict, dirs: dict[str, Path]) -> str:
    lines = [
        "# Analytics index",
        "",
        f"- Mode: `{report.get('mode')}`",
        f"- Telegram: **{report.get('telegram')}**",
        f"- Generated: {report.get('generated_utc')}",
        "",
        "## Comparison plots",
        "",
    ]
    for p in sorted(dirs["plots"].glob("*.png")):
        lines.append(f"- `{p.name}`")
    lines += ["", "## Per-arm analytics", ""]
    for arm in ARM_ORDER:
        lines.append(f"- `arms/{arm}/`")
    return "\n".join(lines)
