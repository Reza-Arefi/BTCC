"""365-day trailing-exit experiment backtest — static entry, T1–T10 parallel exits.

Historical only. No weight updates. No live handoff.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from btcc.backtest.config import load_backtest_config
from btcc.backtest.data_loader import download_panels
from btcc.backtest.dominance_history import HistoricalDominanceSeries
from btcc.backtest.predict import predict_coin_at_bar
from btcc.sim.accounting import CostModel, close_long_alt_btc
from btcc.sim.checkpoint import verify_final_checkpoint, write_daily_checkpoint, write_final_checkpoint
from btcc.sim.day_axis import day_number_at
from btcc.sim.exits import (
    leg_to_record,
    open_opportunity_legs,
    process_bars_until_closed,
    specs_from_config,
)
from btcc.sim.fingerprint import write_run_fingerprint
from btcc.sim.health import evaluate_health
from btcc.sim.regime import classify_regime
from btcc.sim.score import FACTOR_KEYS, combined_score, extract_factor_scores, static_factor_weights
from btcc.sim.state_machine import CrossingStateMachine
from btcc.sim.trail_config import load_trail_experiment_config, validate_trail_strategies
from btcc.sim.trail_entry import evaluate_trail_entry
from btcc.sim.backtest import _advance_book, _experiment_label, _normal_pair_coverage_days, _summarize

logger = logging.getLogger(__name__)


def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def run_trail_experiment_backtest(
    *,
    days: int = 365,
    force_download: bool = False,
    out_root: Path | None = None,
    sim_cfg: dict[str, Any] | None = None,
) -> Path:
    """Run the trailing-exit geometry experiment."""
    sim = dict(sim_cfg or load_trail_experiment_config())
    errs = validate_trail_strategies(sim)
    if errs:
        raise ValueError("Trail strategy validation failed: " + "; ".join(errs))

    bt_cfg = load_backtest_config()
    cfg = {**bt_cfg, "sim": sim}
    root = Path(bt_cfg.get("_root") or Path(__file__).resolve().parents[2])
    out_root = Path(out_root) if out_root else root / "results"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    lo = float(sim["long_threshold"])
    hi = float(sim["upper_threshold"])
    out_dir = out_root / f"trail_compare_{days}d_S{lo:.2f}_{hi:.2f}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    interval = bt_cfg["backtest"]["interval"]
    min_warmup = int(bt_cfg["backtest"]["min_warmup_bars"])
    horizon_h = int(sim.get("primary_horizon_hours", 4))
    bars_4h = horizon_h * 4

    weights = static_factor_weights(cfg)
    (out_dir / "fixed_weights.json").write_text(
        json.dumps({"weight_mode": "trail_static", "weights": weights}, indent=2),
        encoding="utf-8",
    )

    panels = download_panels(cfg, days, min_warmup, force=force_download)
    btc_df = panels["btc"].copy()
    btc_df["timestamp"] = pd.to_datetime(btc_df["timestamp"], utc=True)
    eval_start = _utc(panels["window"]["eval_start"])
    eval_end = _utc(panels["window"]["eval_end"])
    btc_close = btc_df.set_index("timestamp")["close"]

    dom_series = HistoricalDominanceSeries.fetch_coingecko(
        days=int(days),
        cache_dir=bt_cfg["backtest_data"]["dominance_cache"],
        force=force_download,
    )

    ts_all = pd.to_datetime(btc_df["timestamp"], utc=True)
    decision_indices = [
        i for i in btc_df.index.tolist()
        if i >= min_warmup and i + 1 < len(btc_df)
        and ts_all.iloc[i] >= eval_start and ts_all.iloc[i] <= eval_end
    ]

    sm = CrossingStateMachine(
        long_threshold=lo,
        upper_threshold=hi,
        max_open=int(sim.get("max_open_opportunities", 10)),
        one_per_pair=bool(sim.get("one_opportunity_per_pair", True)),
    )
    open_book: dict[str, dict[str, Any]] = {}
    costs = CostModel(
        fee_rate_per_side=float(sim.get("fee_rate_per_side", 0.001)),
        slippage_rate_per_side=float(sim.get("slippage_rate_per_side", 0.0005)),
    )
    specs = specs_from_config(sim)
    regime_rules = (sim.get("trail_experiment") or {}).get("regime", {}).get("rules") or {}

    pred_rows: list[dict] = []
    opp_rows: list[dict] = []
    leg_rows: list[dict] = []
    prev_day_number: int | None = None
    last_flushed_day: int | None = None
    max_simultaneous = 0
    max_open_rejects = 0
    own_cache: dict = {}

    def _flush_day(day_n: int, ts_end: pd.Timestamp) -> None:
        nonlocal last_flushed_day
        if last_flushed_day == day_n:
            return
        open_snap = {
            oid: {
                **{k: v for k, v in opp.items() if k != "legs"},
                "legs": [leg_to_record(l, oid) for l in opp["legs"] if not l.closed],
            }
            for oid, opp in open_book.items()
        }
        for rec in leg_rows:
            if rec.get("day_number") is None and rec.get("exit_ts"):
                try:
                    rec["day_number"] = day_number_at(rec["exit_ts"], eval_start)
                except Exception:
                    rec["day_number"] = day_n
        write_daily_checkpoint(
            out_dir,
            day_number=day_n,
            simulated_timestamp=str(ts_end),
            pred_rows=pred_rows,
            opp_rows=opp_rows,
            leg_rows=leg_rows,
            weight_hist=[],
            update_log=[],
            schedule_history={"updates": []},
            schedule_active={"version_id": "trail_static", "weights": weights},
            open_books_snapshot={"COMMON": open_snap},
            weight_mode="trail_static",
            init_days=0,
            eval_start=str(eval_start),
            eval_end=str(eval_end),
            coverage=panels.get("coverage"),
            refresh_analytics=True,
            telegram_enabled=False,
            starting_capital_usd=float(sim.get("starting_capital_usd", 1000.0)),
        )
        last_flushed_day = day_n

    for n_done, i in enumerate(decision_indices):
        t = _utc(btc_df.iloc[i]["timestamp"])
        day_n = day_number_at(t, eval_start)
        if prev_day_number is not None and day_n != prev_day_number:
            _flush_day(prev_day_number, t)
        prev_day_number = day_n

        _advance_book(open_book, sm, panels, btc_close, costs, sim, t, leg_rows)
        max_simultaneous = max(max_simultaneous, sm.n_open())

        btc_hist = btc_df.iloc[: i + 1].copy()
        dom_pct, dom_obs, _ = dom_series.observation_at(t)
        dom_changes = dom_series.dom_changes_at(t)
        health = evaluate_health(
            sim_cfg=sim,
            dominance_pct=dom_pct,
            dominance_ts=dom_obs.to_pydatetime() if hasattr(dom_obs, "to_pydatetime") else dom_obs,
            dominance_source=(dom_series.meta or {}).get("source"),
            decision_candle_ts=t.to_pydatetime(),
            now=t.to_pydatetime(),
        )

        for base, coin in panels["coins"].items():
            rel_full = coin["rel"].copy()
            rel_full["timestamp"] = pd.to_datetime(rel_full["timestamp"], utc=True)
            rel_hist = rel_full[rel_full["timestamp"] <= t]
            if len(rel_hist) < 100:
                continue
            alt_vol = coin["alt_for_volume"].copy()
            alt_vol["timestamp"] = pd.to_datetime(alt_vol["timestamp"], utc=True)
            alt_vol_hist = alt_vol[alt_vol["timestamp"] <= t]

            cache_key = (str(t), coin["symbol"])
            if cache_key in own_cache:
                factor_scores = dict(own_cache[cache_key]["factor_scores"])
                late_score = own_cache[cache_key].get("late_entry_score")
                late_class = own_cache[cache_key].get("late_entry_class")
                factors_raw = own_cache[cache_key].get("factors_raw") or {}
            else:
                row = predict_coin_at_bar(rel_hist, alt_vol_hist, btc_hist, dom_pct, dom_changes, cfg, interval)
                if row is None:
                    continue
                factor_scores = extract_factor_scores(row["factors"])
                late_score = row.get("late_entry_score")
                late_class = row.get("late_entry_class")
                factors_raw = row.get("factors") or {}
                own_cache[cache_key] = {
                    "factor_scores": dict(factor_scores),
                    "late_entry_score": late_score,
                    "late_entry_class": late_class,
                    "factors_raw": factors_raw,
                }

            scored = combined_score(factor_scores, weights)
            s_val = float(scored["S"])
            pair = coin["symbol"]
            logical_pair = coin.get("logical_pair", base)
            resolved_market = coin.get("resolved_market", pair)

            decision = sm.evaluate(pair, s_val)
            ep = evaluate_trail_entry(
                sm_decision=decision,
                health_allow_new_trades=bool(health.allow_new_trades),
            )
            trade_opened = False
            rejection = ep["rejection_reason"]
            opportunity_id = None
            regime_info = classify_regime(factors_raw, rules=regime_rules)

            if ep["trade_suggested"]:
                future = rel_full[rel_full["timestamp"] > t]
                if future.empty:
                    rejection = "NO_NEXT_BAR"
                else:
                    entry_bar = future.iloc[0]
                    entry_mid = float(entry_bar["open"])
                    entry_ts = _utc(entry_bar["timestamp"])
                    btc_hist_px = btc_df[btc_df["timestamp"] <= entry_ts]
                    btc_usdt = float(btc_hist_px["close"].iloc[-1]) if not btc_hist_px.empty else float(btc_hist["close"].iloc[-1])
                    opportunity_id = f"opp_{uuid.uuid4().hex[:12]}"
                    legs = open_opportunity_legs(
                        alt_btc_entry_mid=entry_mid,
                        btc_usdt=btc_usdt,
                        notional_usd=float(sim.get("notional_usd", 100.0)),
                        costs=costs,
                        specs=specs,
                        entry_ts=entry_ts,
                    )
                    open_book[opportunity_id] = {
                        "opportunity_id": opportunity_id,
                        "symbol": pair,
                        "base": base,
                        "logical_pair": logical_pair,
                        "resolved_market": resolved_market,
                        "S": s_val,
                        "opened_ts": str(t),
                        "entry_fill_ts": str(entry_ts),
                        "entry_alt_btc_mid": entry_mid,
                        "legs": legs,
                        "last_processed_ts": str(entry_ts),
                        "regime": regime_info["regime"],
                        "regime_adx": regime_info["adx"],
                        "regime_natr": regime_info["natr"],
                    }
                    sm.register_open(pair, opportunity_id)
                    trade_opened = True
                    rejection = None
                    opp_rows.append({
                        "opportunity_id": opportunity_id,
                        "opened_ts": str(t),
                        "signal_timestamp": str(t),
                        "day_number": day_n,
                        "symbol": pair,
                        "base": base,
                        "logical_pair": logical_pair,
                        "resolved_market": resolved_market,
                        "S": s_val,
                        "long_threshold": lo,
                        "upper_threshold": hi,
                        "entry_fill_ts": str(entry_ts),
                        "entry_alt_btc_mid": entry_mid,
                        "entry_btc_usdt": btc_usdt,
                        "notional_usd": float(sim.get("notional_usd", 100.0)),
                        "status": "OPEN",
                        "weight_mode": "trail_static",
                        "entry_classification": ep["entry_classification"],
                        "late_entry_score": late_score,
                        "late_entry_class": late_class,
                        "regime": regime_info["regime"],
                        "regime_adx": regime_info["adx"],
                        "regime_natr": regime_info["natr"],
                        **{f"weight_{k}": weights.get(k) for k in FACTOR_KEYS},
                    })

            if rejection == "MAX_OPEN_TRADES":
                max_open_rejects += 1

            future_return_4h = None
            outcome_timestamp = None
            idxs = rel_full.index[rel_full["timestamp"] == t]
            if len(idxs):
                pos = rel_full.index.get_loc(idxs[0])
                if isinstance(pos, int) and pos + bars_4h < len(rel_full):
                    px0 = float(rel_full.iloc[pos]["close"])
                    px1 = float(rel_full.iloc[pos + bars_4h]["close"])
                    if px0:
                        future_return_4h = px1 / px0 - 1.0
                        outcome_timestamp = str(rel_full.iloc[pos + bars_4h]["timestamp"])

            pred_rows.append({
                "timestamp": str(t),
                "day_number": day_n,
                "symbol": pair,
                "base": base,
                "S": s_val,
                "long_threshold": lo,
                "upper_threshold": hi,
                "signal_generated": bool(decision["signal_generated"]),
                "trade_opened": trade_opened,
                "rejection_reason": rejection,
                "entry_classification": ep["entry_classification"],
                "late_entry_score": late_score,
                "late_entry_class": late_class,
                "opportunity_id": opportunity_id,
                "regime": regime_info["regime"],
                "future_return_4h": future_return_4h,
                "outcome_timestamp": outcome_timestamp,
                "health_allow_new_trades": bool(health.allow_new_trades),
                "btc_dominance": dom_pct,
                "btc_d_status": health.btc_d_status,
                **{f"weight_{k}": weights.get(k) for k in FACTOR_KEYS},
            })

        if n_done and n_done % 250 == 0:
            logger.info("Trail sim %d/%d day=%s open=%d opps=%d", n_done, len(decision_indices), day_n, sm.n_open(), len(opp_rows))

    if prev_day_number is not None and decision_indices:
        _flush_day(prev_day_number, _utc(btc_df.iloc[decision_indices[-1]]["timestamp"]))

    for oid, opp in list(open_book.items()):
        coin = panels["coins"].get(opp["base"])
        mid = float(coin["rel"].iloc[-1]["close"]) if coin else opp["entry_alt_btc_mid"]
        btc_usdt = float(btc_df.iloc[-1]["close"])
        for leg in opp["legs"]:
            if not leg.closed:
                leg.closed = True
                leg.exit_reason = "END_OF_BACKTEST"
                leg.exit_ts = btc_df.iloc[-1]["timestamp"]
                leg.exit_result = close_long_alt_btc(position=leg.position, alt_btc_mid=mid, btc_usdt=btc_usdt, costs=costs)
            rec = leg_to_record(leg, oid)
            rec["regime"] = opp.get("regime")
            rec["day_number"] = day_number_at(rec.get("exit_ts") or eval_end, eval_start)
            leg_rows.append(rec)
        sm.register_close(oid, opp.get("symbol"))
    open_book.clear()

    pred_df = pd.DataFrame(pred_rows)
    opp_df = pd.DataFrame(opp_rows)
    legs_df = pd.DataFrame(leg_rows)
    if not legs_df.empty and not opp_df.empty:
        meta = opp_df.set_index("opportunity_id")[[c for c in ("regime", "S", "late_entry_score") if c in opp_df.columns]]
        for c in meta.columns:
            if c not in legs_df.columns:
                legs_df[c] = legs_df["opportunity_id"].map(meta[c])

    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    opp_df.to_csv(out_dir / "opportunities.csv", index=False)
    legs_df.to_csv(out_dir / "strategy_legs.csv", index=False)

    summary = _summarize(pred_df, legs_df, lo, days, dom_series, max_simultaneous=max_simultaneous, max_open_rejects=max_open_rejects)
    summary.update({
        "experiment_kind": "trail_exit_v1",
        "weight_mode": "trail_static",
        "upper_threshold": hi,
        "benchmark_strategy_key": sim.get("benchmark_strategy_key", "trail_3"),
        "n_opportunities": len(opp_df),
        "requested_days": days,
        "actual_normal_pair_coverage_days": _normal_pair_coverage_days(panels),
        "experiment_label": _experiment_label(days, panels),
    })
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    write_run_fingerprint(
        out_dir, root=root, sim_cfg=sim, signal_cfg=cfg,
        meta=summary,
        candle_dir=Path(bt_cfg["backtest_data"]["candle_dir"]),
        dominance_cache_dir=Path(bt_cfg["backtest_data"]["dominance_cache"]),
    )
    try:
        import subprocess
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception:
        git_commit = None
    fp = json.loads((out_dir / "fingerprint.json").read_text(encoding="utf-8")) if (out_dir / "fingerprint.json").exists() else None
    final = write_final_checkpoint(
        out_dir, summary=summary, fingerprint=fp,
        last_day_number=last_flushed_day or prev_day_number,
        git_commit=git_commit,
    )
    v = verify_final_checkpoint(final)
    summary["final_checkpoint"] = {"path": str(final), "verify": v, "status": "COMPLETED"}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info("Trail experiment complete → %s verify=%s", out_dir, v.get("ok"))
    return out_dir
