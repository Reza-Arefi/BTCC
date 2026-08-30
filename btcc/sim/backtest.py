"""Walk-forward Adaptive V2 backtest — 90d init + daily rolling-90d adaptation.

Protocol (NOT 4×90d folds):
  [warmup] → [init ~90d learning] → initial weights effective_from next bar
           → [daily phase ~275d] 23:00 America/Sao_Paulo updates
             with effective_from = NEXT decision bar after calculation

Look-ahead controls:
  - features ≤ t
  - BTC.D last ≤ t (relative top-N, no present-day scale)
  - learning only matured outcomes: pred_ts + 4h ≤ update_ts
  - entry = next bar open
  - same-candle TP+SL → SL/TRAILING first
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from btcc.backtest.config import load_backtest_config
from btcc.backtest.data_loader import compute_window, download_panels
from btcc.backtest.dominance_history import HistoricalDominanceSeries
from btcc.backtest.predict import predict_coin_at_bar
from btcc.sim.accounting import CostModel, close_long_alt_btc
from btcc.sim.checkpoint import verify_final_checkpoint, write_daily_checkpoint, write_final_checkpoint
from btcc.sim.config import load_sim_config
from btcc.sim.day_axis import day_number_at
from btcc.sim.exits import (
    leg_to_record,
    open_opportunity_legs,
    process_bars_until_closed,
    specs_from_config,
)
from btcc.sim.fingerprint import write_run_fingerprint
from btcc.sim.maturity import filter_matured_for_learning
from btcc.sim.score import (
    FACTOR_KEYS,
    combined_score,
    equal_factor_weights,
    extract_factor_scores,
    normalize_weights,
    static_factor_weights,
)
from btcc.sim.state_machine import CrossingStateMachine
from btcc.sim.weight_schedule import WeightSchedule, WeightVersion
from btcc.sim.weights import estimate_weights_from_window, filter_rolling_window
from btcc.sim.entry_policy import (
    ENTRY_POLICIES,
    POLICY_LATE_ALLOWED,
    POLICY_NORMAL,
    evaluate_entry_policy,
)
from btcc.sim.health import evaluate_health

logger = logging.getLogger(__name__)


def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _next_bar_ts(btc_df: pd.DataFrame, i: int) -> pd.Timestamp | None:
    if i + 1 >= len(btc_df):
        return None
    return _utc(btc_df.iloc[i + 1]["timestamp"])


def run_adaptive_sim_backtest(
    signal_cfg: dict[str, Any] | None = None,
    *,
    days: int = 365,
    sim_cfg: dict[str, Any] | None = None,
    force_download: bool = False,
    long_threshold: float | None = None,
    out_root: Path | None = None,
    weight_mode: str = "adaptive",
    factor_cache: dict[tuple[str, str], dict[str, float]] | None = None,
    run_tag: str | None = None,
) -> Path:
    """Run paper sim backtest for one weight arm.

    weight_mode:
      - ``static``: fixed signal_config factors.weights (never updated)
      - ``equal``: fixed 1/N over FACTOR_KEYS (never updated)
      - ``adaptive``: 90d init + daily rolling 90d
    """
    if weight_mode not in ("static", "equal", "adaptive"):
        raise ValueError(f"Unknown weight_mode={weight_mode!r}")

    bt_cfg = load_backtest_config()
    cfg = bt_cfg
    if signal_cfg:
        cfg = {
            **bt_cfg,
            **{k: signal_cfg[k] for k in ("universe", "factors", "probability", "late_entry", "safety") if k in signal_cfg},
        }
        cfg["_root"] = signal_cfg.get("_root", bt_cfg.get("_root"))

    sim = dict(sim_cfg or load_sim_config())
    if long_threshold is not None:
        sim["long_threshold"] = float(long_threshold)

    root = Path(cfg.get("_root") or Path(__file__).resolve().parents[2])
    out_root = Path(out_root) if out_root else root / "results"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    thr_tag = f"{float(sim['long_threshold']):.2f}".replace(".", "p")
    tag = run_tag or weight_mode
    out_dir = out_root / f"adaptive_v2_{tag}_{days}d_thr{thr_tag}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    wf = sim.get("walk_forward") or {}
    init_days = int(wf.get("init_days", 90))
    roll_days = int(wf.get("daily_rolling_window_days", wf.get("rolling_window_days", 90)))
    interval = cfg["backtest"]["interval"]
    min_warmup = int(cfg["backtest"]["min_warmup_bars"])
    horizon_h = int(sim.get("primary_horizon_hours", 4))
    bars_4h = horizon_h * 4
    adapt = weight_mode == "adaptive"

    logger.info(
        "Sim backtest mode=%s days=%d init_days=%d thr=%.2f force=%s",
        weight_mode, days, init_days, sim["long_threshold"], force_download,
    )
    cfg = {**cfg, "sim": sim}
    panels = download_panels(cfg, days, min_warmup, force=force_download)
    btc_df = panels["btc"].copy()
    btc_df["timestamp"] = pd.to_datetime(btc_df["timestamp"], utc=True)
    eval_start = _utc(panels["window"]["eval_start"])
    eval_end = _utc(panels["window"]["eval_end"])
    init_end = eval_start + pd.Timedelta(days=init_days)

    logger.info("Fetching BTC.D for %d days (relative, no present-day calibration)...", days)
    dom_series = HistoricalDominanceSeries.fetch_coingecko(
        days=int(days),
        cache_dir=cfg["backtest_data"]["dominance_cache"],
        force=force_download,
    )
    if (dom_series.meta or {}).get("calibration", "").startswith("scaled_to_global"):
        raise RuntimeError(
            "Refusing present-day BTC.D calibration in backtest. "
            f"meta={dom_series.meta}"
        )

    ts_all = pd.to_datetime(btc_df["timestamp"], utc=True)
    decision_indices = [
        i for i in btc_df.index.tolist()
        if i >= min_warmup
        and i + 1 < len(btc_df)
        and ts_all.iloc[i] >= eval_start
        and ts_all.iloc[i] <= eval_end
    ]

    if weight_mode == "static":
        seed_w = static_factor_weights(cfg)
        schedule = WeightSchedule(seed_w, version_id="static_config_weights")
    elif weight_mode == "equal":
        seed_w = equal_factor_weights()
        schedule = WeightSchedule(seed_w, version_id="equal_1_over_n")
    else:
        seed_w = static_factor_weights(cfg)
        schedule = WeightSchedule(dict(seed_w), version_id="config_pre_init")

    (out_dir / "fixed_weights.json").write_text(
        json.dumps({
            "weight_mode": weight_mode,
            "weights": seed_w if weight_mode != "adaptive" else {
                "pre_init_seed": seed_w,
                "note": "Adaptive updates via schedule; see weight_schedule.json",
            },
            "source": (
                "signal_config.factors.weights" if weight_mode == "static"
                else ("1/N FACTOR_KEYS" if weight_mode == "equal" else "adaptive_schedule")
            ),
        }, indent=2),
        encoding="utf-8",
    )

    # Separate state machines / books per ENTRY POLICY (exit strategies S1–S3 shared per opp)
    policies = list(ENTRY_POLICIES)
    sms = {
        p: CrossingStateMachine(
            long_threshold=float(sim["long_threshold"]),
            max_open=int(sim.get("max_open_opportunities", 10)),
            one_per_pair=bool(sim.get("one_opportunity_per_pair", True)),
        )
        for p in policies
    }
    open_books: dict[str, dict[str, dict[str, Any]]] = {p: {} for p in policies}
    costs = CostModel(
        fee_rate_per_side=float(sim.get("fee_rate_per_side", 0.001)),
        slippage_rate_per_side=float(sim.get("slippage_rate_per_side", 0.0005)),
    )
    specs = specs_from_config(sim)
    wu = sim.get("weight_update") or {}
    tz = ZoneInfo(str(wu.get("timezone", "America/Sao_Paulo")))
    late_thr = float((cfg.get("late_entry") or {}).get("alert_threshold", 0.75))

    pred_rows: list[dict[str, Any]] = []
    opp_rows: list[dict[str, Any]] = []
    leg_rows: list[dict[str, Any]] = []
    weight_hist: list[dict[str, Any]] = []
    update_log: list[dict[str, Any]] = []
    btc_close = btc_df.set_index("timestamp")["close"]
    own_cache = factor_cache if factor_cache is not None else {}

    init_done = False if adapt else True
    last_daily_local_date = None
    daily_update_number = 0
    max_simultaneous = {p: 0 for p in policies}
    max_open_rejects = {p: 0 for p in policies}
    n_recovered = 0
    prev_day_number: int | None = None
    last_flushed_day: int | None = None

    def _flush_day(day_n: int, ts_end) -> None:
        nonlocal last_flushed_day
        if day_n is None or day_n == last_flushed_day:
            return
        active_v = schedule.active_at(ts_end)
        open_snap = {
            pol: {
                oid: {
                    "opportunity_id": oid,
                    "symbol": opp.get("symbol"),
                    "base": opp.get("base"),
                    "logical_pair": opp.get("logical_pair"),
                    "resolved_market": opp.get("resolved_market"),
                    "opened_ts": opp.get("opened_ts"),
                    "entry_policy": opp.get("entry_policy"),
                    "S": opp.get("S"),
                    "weight_version_id": opp.get("weight_version_id"),
                    "n_legs_open": sum(1 for leg in opp.get("legs") or [] if not getattr(leg, "closed", True)),
                }
                for oid, opp in book.items()
            }
            for pol, book in open_books.items()
        }
        # Stamp day_number on legs missing it
        for rec in leg_rows:
            if rec.get("day_number") is None and rec.get("exit_ts") is not None:
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
            weight_hist=weight_hist,
            update_log=update_log,
            schedule_history={"updates": schedule.history()},
            schedule_active=active_v.to_dict(),
            open_books_snapshot=open_snap,
            weight_mode=weight_mode,
            init_days=init_days,
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
        t_local = t.tz_convert(tz)
        phase = "init" if t < init_end else "daily"
        day_n = day_number_at(t, eval_start)
        if prev_day_number is not None and day_n != prev_day_number:
            _flush_day(prev_day_number, t)
        prev_day_number = day_n

        if adapt and (not init_done) and t >= init_end:
            matured = filter_matured_for_learning(pred_rows, asof_ts=t, horizon_hours=horizon_h)
            window = filter_rolling_window(matured, t, roll_days) if not matured.empty else matured
            prev = schedule.active_at(t).weights
            new_w, stats = estimate_weights_from_window(
                window, prev,
                half_life_days=float(wu.get("time_decay_half_life_days", 45)),
                weight_min=float(wu.get("weight_min", 0.05)),
                weight_max=float(wu.get("weight_max", 0.40)),
                min_obs_total=int(wu.get("min_observations_total", 200)),
                min_obs_per_factor=int(wu.get("min_observations_per_factor", 50)),
                stability_blend=float(wu.get("stability_blend", 0.25)),
                horizon=horizon_h,
            )
            eff = _next_bar_ts(btc_df, i)
            if new_w is not None and eff is not None:
                vid = f"init_{init_end.date()}"
                schedule.add(WeightVersion(
                    version_id=vid, weights=new_w, calculated_at=str(t), effective_from=str(eff),
                    learning_window_start=stats.get("learning_window_start"),
                    learning_window_end=stats.get("learning_window_end"),
                    n_samples=stats.get("n_samples"), phase="init", update_number=0,
                    stats={"status": "OK", "ycol": stats.get("ycol")},
                ))
                _append_weight_hist(weight_hist, prev, new_w, stats, t_local, vid, "OK", "init", day_number=day_n)
                update_log.append({
                    "update_number": 0, "phase": "init", "calculated_at": str(t),
                    "effective_from": str(eff),
                    "learning_window_start": stats.get("learning_window_start"),
                    "learning_window_end": stats.get("learning_window_end"),
                    "n_samples": stats.get("n_samples"), "status": "OK",
                    "weights": new_w, "old_weights": prev,
                })
                logger.info("INIT weights ready calculated_at=%s effective_from=%s n=%s", t, eff, stats.get("n_samples"))
            else:
                _append_weight_hist(weight_hist, prev, prev, stats, t_local, f"init_failed_{t.date()}", "FAILED", "init", day_number=day_n)
                update_log.append({
                    "update_number": 0, "phase": "init", "calculated_at": str(t),
                    "effective_from": None, "status": "FAILED", "notes": stats.get("reason"),
                    "n_samples": stats.get("n_samples"), "weights": prev, "old_weights": prev,
                })
                logger.warning("INIT weight update FAILED (%s) — keeping config weights", stats.get("reason"))
            init_done = True

        if adapt and phase == "daily" and init_done:
            if (t_local.hour, t_local.minute) >= (int(wu.get("hour", 23)), int(wu.get("minute", 0))):
                local_date = t_local.date().isoformat()
                if local_date != last_daily_local_date:
                    matured = filter_matured_for_learning(pred_rows, asof_ts=t, horizon_hours=horizon_h)
                    window = filter_rolling_window(matured, t, roll_days) if not matured.empty else matured
                    prev = schedule.active_at(t).weights
                    new_w, stats = estimate_weights_from_window(
                        window, prev,
                        half_life_days=float(wu.get("time_decay_half_life_days", 45)),
                        weight_min=float(wu.get("weight_min", 0.05)),
                        weight_max=float(wu.get("weight_max", 0.40)),
                        min_obs_total=int(wu.get("min_observations_total", 200)),
                        min_obs_per_factor=int(wu.get("min_observations_per_factor", 50)),
                        stability_blend=float(wu.get("stability_blend", 0.25)),
                        horizon=horizon_h,
                    )
                    daily_update_number += 1
                    eff = _next_bar_ts(btc_df, i)
                    vid = f"daily_{local_date}"
                    if new_w is not None and eff is not None:
                        schedule.add(WeightVersion(
                            version_id=vid, weights=new_w, calculated_at=str(t), effective_from=str(eff),
                            learning_window_start=stats.get("learning_window_start"),
                            learning_window_end=stats.get("learning_window_end"),
                            n_samples=stats.get("n_samples"), phase="daily",
                            update_number=daily_update_number, stats={"status": "OK"},
                        ))
                        _append_weight_hist(weight_hist, prev, new_w, stats, t_local, vid, "OK", "daily", day_number=day_n)
                        update_log.append({
                            "update_number": daily_update_number, "phase": "daily",
                            "calculated_at": str(t), "effective_from": str(eff),
                            "learning_window_start": stats.get("learning_window_start"),
                            "learning_window_end": stats.get("learning_window_end"),
                            "n_samples": stats.get("n_samples"), "status": "OK",
                            "weights": new_w, "old_weights": prev,
                        })
                    else:
                        _append_weight_hist(weight_hist, prev, prev, stats, t_local, vid, "FAILED", "daily", day_number=day_n)
                        update_log.append({
                            "update_number": daily_update_number, "phase": "daily",
                            "calculated_at": str(t), "effective_from": None,
                            "status": "FAILED", "notes": stats.get("reason"),
                            "n_samples": stats.get("n_samples"), "weights": prev, "old_weights": prev,
                        })
                    last_daily_local_date = local_date

        active = schedule.active_at(t)
        weights = active.weights

        for _pol in policies:
            _advance_book(open_books[_pol], sms[_pol], panels, btc_close, costs, sim, t, leg_rows)
            max_simultaneous[_pol] = max(max_simultaneous[_pol], sms[_pol].n_open())

        btc_hist = btc_df.iloc[: i + 1].copy()
        dom_pct, dom_obs, _dom_st = dom_series.observation_at(t)
        dom_changes = dom_series.dom_changes_at(t)
        # Age checks must use simulation time t — never wall-clock now
        # (otherwise every historical bar looks STALE / BTC_D_UNAVAILABLE).
        _now = t.to_pydatetime() if hasattr(t, "to_pydatetime") else t
        health = evaluate_health(
            sim_cfg=sim,
            dominance_pct=dom_pct,
            dominance_ts=dom_obs.to_pydatetime() if hasattr(dom_obs, "to_pydatetime") else dom_obs,
            dominance_source=(dom_series.meta or {}).get("source"),
            decision_candle_ts=_now,
            now=_now,
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
            late_score = None
            late_class = None
            if cache_key in own_cache and isinstance(own_cache[cache_key], dict) and "factor_scores" in own_cache[cache_key]:
                factor_scores = dict(own_cache[cache_key]["factor_scores"])
                late_score = own_cache[cache_key].get("late_entry_score")
                late_class = own_cache[cache_key].get("late_entry_class")
            else:
                row = predict_coin_at_bar(
                    rel_hist, alt_vol_hist, btc_hist, dom_pct, dom_changes, cfg, interval
                )
                if row is None:
                    continue
                factor_scores = extract_factor_scores(row["factors"])
                late_score = row.get("late_entry_score")
                late_class = row.get("late_entry_class")
                own_cache[cache_key] = {
                    "factor_scores": dict(factor_scores),
                    "late_entry_score": late_score,
                    "late_entry_class": late_class,
                }

            scored = combined_score(factor_scores, weights)
            S = float(scored["S"])
            pair = coin["symbol"]
            logical_pair = coin.get("logical_pair", base)
            resolved_market = coin.get("resolved_market", pair)

            # Shared prediction row fields; per-policy decisions appended below
            policy_decisions = {}
            for pol in policies:
                sm = sms[pol]
                decision = sm.evaluate(pair, S)
                ep = evaluate_entry_policy(
                    policy=pol,
                    sm_decision=decision,
                    late_entry_score=float(late_score) if late_score is not None else None,
                    late_entry_class=late_class,
                    alert_threshold=late_thr,
                    health_allow_new_trades=bool(health.allow_new_trades),
                    health_btc_d_available=bool(health.btc_d_available),
                )
                trade_opened = False
                rejection = ep["rejection_reason"]
                opportunity_id = None
                entry_classification = ep["entry_classification"]

                if ep["trade_suggested"]:
                    future = rel_full[rel_full["timestamp"] > t]
                    if future.empty:
                        rejection = "NO_NEXT_BAR"
                        entry_classification = "NO_NEXT_BAR"
                    else:
                        entry_bar = future.iloc[0]
                        entry_mid = float(entry_bar["open"])
                        entry_ts = _utc(entry_bar["timestamp"])
                        btc_hist_px = btc_df[btc_df["timestamp"] <= entry_ts]
                        btc_usdt = float(btc_hist_px["close"].iloc[-1]) if not btc_hist_px.empty else float(btc_hist["close"].iloc[-1])
                        opportunity_id = f"opp_{pol[:3]}_{uuid.uuid4().hex[:10]}"
                        legs = open_opportunity_legs(
                            alt_btc_entry_mid=entry_mid,
                            btc_usdt=btc_usdt,
                            notional_usd=float(sim.get("notional_usd", 100.0)),
                            costs=costs,
                            specs=specs,
                            entry_ts=entry_ts,
                        )
                        open_books[pol][opportunity_id] = {
                            "opportunity_id": opportunity_id,
                            "symbol": pair,
                            "base": base,
                            "logical_pair": logical_pair,
                            "resolved_market": resolved_market,
                            "S": S,
                            "opened_ts": str(t),
                            "entry_fill_ts": str(entry_ts),
                            "entry_alt_btc_mid": entry_mid,
                            "legs": legs,
                            "last_processed_ts": str(entry_ts),
                            "weight_version_id": active.version_id,
                            "weights_at_entry": dict(weights),
                            "entry_policy": pol,
                            "entry_classification": entry_classification,
                            "recovered_by_late_allowed": bool(ep["recovered_by_late_allowed"]),
                            "late_extended": bool(ep["late_extended"]),
                        }
                        sm.register_open(pair, opportunity_id)
                        trade_opened = True
                        rejection = None
                        if ep["recovered_by_late_allowed"]:
                            n_recovered += 1
                        opp_rows.append({
                            "opportunity_id": opportunity_id,
                            "opened_ts": str(t),
                            "signal_timestamp": str(t),
                            "day_number": day_n,
                            "symbol": pair,
                            "base": base,
                            "logical_pair": logical_pair,
                            "resolved_market": resolved_market,
                            "S": S,
                            "threshold": sm.long_threshold,
                            "entry_fill_ts": str(entry_ts),
                            "entry_alt_btc_mid": entry_mid,
                            "entry_btc_usdt": btc_usdt,
                            "notional_usd": float(sim.get("notional_usd", 100.0)),
                            "status": "OPEN",
                            "weight_version_id": active.version_id,
                            "weights_calculated_at": active.calculated_at,
                            "weights_effective_from": active.effective_from,
                            "weight_mode": weight_mode,
                            "entry_policy": pol,
                            "entry_classification": entry_classification,
                            "late_extended": bool(ep["late_extended"]),
                            "recovered_by_late_allowed": bool(ep["recovered_by_late_allowed"]),
                            "late_entry_score": late_score,
                            "late_entry_class": late_class,
                            **{f"weight_{k}": weights.get(k) for k in FACTOR_KEYS},
                        })
                if rejection == "MAX_OPEN_TRADES":
                    max_open_rejects[pol] += 1

                policy_decisions[pol] = {
                    "trade_opened": trade_opened,
                    "rejection_reason": rejection,
                    "entry_classification": entry_classification,
                    "opportunity_id": opportunity_id,
                    "signal_generated": bool(decision["signal_generated"]),
                    "late_extended": bool(ep["late_extended"]),
                    "normal_would_reject_for_late": bool(ep["normal_would_reject_for_late"]),
                    "recovered_by_late_allowed": bool(ep["recovered_by_late_allowed"]),
                    "crossed_into": bool(decision.get("crossed_into")),
                    "zone_state": decision.get("zone"),
                }

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
                        outcome_ts = rel_full.iloc[pos + bars_4h]["timestamp"]
                        outcome_timestamp = str(outcome_ts)

            # One prediction row per entry policy (same S/factors; policy-specific open/reject)
            for pol, pd_dec in policy_decisions.items():
                pred_rows.append({
                    "timestamp": str(t),
                    "day_number": day_n,
                    "symbol": pair,
                    "base": base,
                    "logical_pair": logical_pair,
                    "resolved_market": resolved_market,
                    "phase": phase,
                    "weight_mode": weight_mode,
                    "entry_policy": pol,
                    "S": S,
                    "long_threshold": float(sim["long_threshold"]),
                    "signal_generated": pd_dec["signal_generated"],
                    "trade_opened": pd_dec["trade_opened"],
                    "rejection_reason": None if pd_dec["trade_opened"] else pd_dec["rejection_reason"],
                    "entry_classification": pd_dec["entry_classification"],
                    "late_extended": pd_dec["late_extended"],
                    "normal_would_reject_for_late": pd_dec["normal_would_reject_for_late"],
                    "recovered_by_late_allowed": pd_dec["recovered_by_late_allowed"],
                    "late_entry_score": late_score,
                    "late_entry_class": late_class,
                    "opportunity_id": pd_dec["opportunity_id"],
                    "weight_version_id": active.version_id,
                    "weights_effective_from": active.effective_from,
                    "prediction_horizon_hours": horizon_h,
                    "future_return_4h": future_return_4h,
                    "outcome_timestamp": outcome_timestamp,
                    "outperformed_4h": int(future_return_4h > 0) if future_return_4h is not None else None,
                    "alt_btc_price": float(rel_hist["close"].iloc[-1]),
                    "btc_price": float(btc_hist["close"].iloc[-1]),
                    "btc_dominance": dom_pct,
                    "btc_d_status": health.btc_d_status,
                    "btc_d_age_seconds": health.btc_d_age_seconds,
                    "health_allow_new_trades": bool(health.allow_new_trades),
                    "btc_d_available": bool(health.btc_d_available),
                    "zone_state": pd_dec.get("zone_state"),
                    **{f"factor_{k}": factor_scores.get(k) for k in FACTOR_KEYS},
                    **{f"weight_{k}": scored["weights"].get(k) for k in FACTOR_KEYS},
                    **{f"signed_{k}": scored["signed"].get(k) for k in FACTOR_KEYS},
                })

        if n_done and n_done % 250 == 0:
            logger.info(
                "Sim %s %d/%d phase=%s day=%s open=%d preds=%d",
                weight_mode, n_done, len(decision_indices), phase, day_n,
                sum(s.n_open() for s in sms.values()), len(pred_rows),
            )

    # Flush final simulated day
    if prev_day_number is not None and decision_indices:
        _flush_day(prev_day_number, _utc(btc_df.iloc[decision_indices[-1]]["timestamp"]))

    for pol in policies:
        for oid, opp in list(open_books[pol].items()):
            coin = panels["coins"].get(opp["base"])
            mid = float(coin["rel"].iloc[-1]["close"]) if coin is not None else opp["entry_alt_btc_mid"]
            btc_usdt = float(btc_df.iloc[-1]["close"])
            for leg in opp["legs"]:
                if not leg.closed:
                    leg.closed = True
                    leg.exit_reason = "END_OF_BACKTEST"
                    leg.exit_ts = btc_df.iloc[-1]["timestamp"]
                    leg.exit_result = close_long_alt_btc(
                        position=leg.position, alt_btc_mid=mid, btc_usdt=btc_usdt, costs=costs
                    )
                rec = leg_to_record(leg, oid)
                rec["entry_policy"] = pol
                rec["recovered_by_late_allowed"] = bool(opp.get("recovered_by_late_allowed"))
                rec["entry_classification"] = opp.get("entry_classification")
                leg_rows.append(rec)
            sms[pol].register_close(oid, opp.get("symbol"))
        open_books[pol].clear()

    closed_ids = {r["opportunity_id"] for r in leg_rows if r.get("closed")}
    for o in opp_rows:
        if o["opportunity_id"] in closed_ids:
            o["status"] = "CLOSED"

    if weight_mode == "static" and pred_rows:
        for k in FACTOR_KEYS:
            vals = {round(float(r[f"weight_{k}"]), 10) for r in pred_rows}
            if len(vals) > 1:
                raise RuntimeError(f"STATIC weights changed for {k}: {vals}")
    if weight_mode == "equal" and pred_rows:
        expected = 1.0 / len(FACTOR_KEYS)
        for r in pred_rows:
            for k in FACTOR_KEYS:
                if abs(float(r[f"weight_{k}"]) - expected) > 1e-9:
                    raise RuntimeError(f"EQUAL weight drift {k}={r[f'weight_{k}']}")

    pred_df = pd.DataFrame(pred_rows)
    opp_df = pd.DataFrame(opp_rows)
    legs_df = pd.DataFrame(leg_rows)
    wh_df = pd.DataFrame(weight_hist)
    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    opp_df.to_csv(out_dir / "opportunities.csv", index=False)
    legs_df.to_csv(out_dir / "strategy_legs.csv", index=False)
    wh_df.to_csv(out_dir / "weight_history.csv", index=False)
    (out_dir / "weight_updates.json").write_text(json.dumps(update_log, indent=2, default=str), encoding="utf-8")
    (out_dir / "weight_schedule.json").write_text(json.dumps(schedule.history(), indent=2, default=str), encoding="utf-8")

    # Tag legs from opportunities for recovered analysis
    if not legs_df.empty and not opp_df.empty and "opportunity_id" in legs_df.columns:
        meta = opp_df.set_index("opportunity_id")[
            [c for c in ("entry_policy", "recovered_by_late_allowed", "entry_classification", "late_extended") if c in opp_df.columns]
        ]
        for c in meta.columns:
            if c not in legs_df.columns:
                legs_df[c] = legs_df["opportunity_id"].map(meta[c])

    summary = _summarize(
        pred_df, legs_df, float(sim["long_threshold"]), days, dom_series,
        max_simultaneous=max(max_simultaneous.values()) if max_simultaneous else 0,
        max_open_rejects=sum(max_open_rejects.values()) if max_open_rejects else 0,
    )
    summary["max_simultaneous_by_entry_policy"] = max_simultaneous
    summary["max_open_rejects_by_entry_policy"] = max_open_rejects
    summary["n_recovered_late_allowed"] = int(n_recovered)
    summary["entry_policies"] = list(policies)
    summary.update({
        "weight_mode": weight_mode,
        "decision_bars": len(decision_indices),
        "eval_start": str(eval_start),
        "eval_end": str(eval_end),
        "init_start": str(eval_start),
        "init_end": str(init_end),
        "init_days": init_days,
        "roll_days": roll_days,
        "daily_phase_start": str(init_end),
        "n_weight_updates": len(update_log),
        "n_daily_updates": daily_update_number,
        "protocol": (
            "90d_init_then_daily_rolling_90d" if adapt
            else ("static_fixed_weights" if weight_mode == "static" else "equal_fixed_weights")
        ),
        "universe_policy": sim.get("universe_policy"),
        "pair_coverage": panels.get("coverage") or {},
        "coverage_note": (
            "Fixed research universe of 20 bases. Individual pairs may list mid-window; "
            "predictions are emitted only when >=100 bars of history exist at decision time. "
            "No fabricated / forward-filled candles. Short-history pairs (e.g. CKBTC, DATA) "
            "do not contribute observations before first usable prediction timestamp and "
            "therefore cannot contaminate adaptive weight updates for earlier periods. "
            "Adaptive learning still uses configured min_observations_* gates."
        ),
        "requested_days": days,
        "actual_normal_pair_coverage_days": _normal_pair_coverage_days(panels),
        "experiment_label": _experiment_label(days, panels),
        "btc_d": {
            "source": (dom_series.meta or {}).get("source"),
            "representation": (dom_series.meta or {}).get("representation"),
            "calibration": (dom_series.meta or {}).get("calibration"),
            "n_points": (dom_series.meta or {}).get("n_points"),
            "status": (dom_series.meta or {}).get("status"),
            "days_requested": days,
        },
        "seed_weights": seed_w,
    })
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out_dir / "pair_coverage.json").write_text(
        json.dumps(panels.get("coverage") or {}, indent=2, default=str), encoding="utf-8"
    )
    (out_dir / "report.md").write_text(_report_md(summary, sim, update_log), encoding="utf-8")
    write_run_fingerprint(
        out_dir,
        root=root,
        sim_cfg=sim,
        signal_cfg=cfg,
        meta=summary,
        candle_dir=Path(cfg["backtest_data"]["candle_dir"]),
        dominance_cache_dir=Path(cfg["backtest_data"]["dominance_cache"]),
    )
    # Formal live handoff checkpoint
    try:
        import subprocess
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except Exception:
        git_commit = None
    fp = None
    fp_path = out_dir / "fingerprint.json"
    if fp_path.exists():
        fp = json.loads(fp_path.read_text(encoding="utf-8"))
    final = write_final_checkpoint(
        out_dir,
        summary=summary,
        fingerprint=fp,
        last_day_number=last_flushed_day or prev_day_number,
        git_commit=git_commit or (fp or {}).get("git_commit"),
    )
    v = verify_final_checkpoint(final)
    summary["final_checkpoint"] = {"path": str(final), "verify": v}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info("Sim backtest complete mode=%s → %s (FINAL_CHECKPOINT ok=%s)", weight_mode, out_dir, v.get("ok"))
    return out_dir


def _normal_pair_coverage_days(panels: dict) -> float | None:
    """Median available_days among non-listing-short synthetic pairs."""
    cov = panels.get("coverage") or {}
    days = [
        float(v["available_days"])
        for v in cov.values()
        if v.get("available_days") and float(v["available_days"]) >= 300
    ]
    if not days:
        days = [float(v["available_days"]) for v in cov.values() if v.get("available_days")]
    if not days:
        return None
    days_sorted = sorted(days)
    return days_sorted[len(days_sorted) // 2]


def _experiment_label(requested_days: int, panels: dict) -> str:
    actual = _normal_pair_coverage_days(panels)
    if actual is None:
        return f"requested_{requested_days}d_historical_backtest"
    return (
        f"approximately 1-year historical backtest "
        f"with approximately {actual:.0f} days of normal-pair coverage "
        f"(requested_days={requested_days})"
    )


def _append_weight_hist(hist, old_w, new_w, stats, t_local, vid, status, phase, day_number=None):
    for ind in (new_w or old_w):
        hist.append({
            "update_timestamp": t_local.isoformat(),
            "day_number": day_number,
            "update_id": vid,
            "phase": phase,
            "indicator": ind,
            "old_weight": (old_w or {}).get(ind),
            "new_weight": (new_w or old_w or {}).get(ind),
            "indicator_ic": (stats.get("ics") or {}).get(ind),
            "indicator_n": (stats.get("ns") or {}).get(ind),
            "n_samples": stats.get("n_samples"),
            "learning_window_start": stats.get("learning_window_start"),
            "learning_window_end": stats.get("learning_window_end"),
            "update_status": status,
            "notes": stats.get("reason"),
        })


def _advance_book(open_book, sm, panels, btc_close, costs, sim, asof_t, leg_rows) -> None:
    done = []
    for oid, opp in open_book.items():
        coin = panels["coins"].get(opp["base"])
        if coin is None:
            continue
        rel = coin["rel"].copy()
        rel["timestamp"] = pd.to_datetime(rel["timestamp"], utc=True)
        last = _utc(opp.get("last_processed_ts") or opp["entry_fill_ts"])
        bars = rel[(rel["timestamp"] > last) & (rel["timestamp"] <= asof_t)]
        if bars.empty:
            continue
        process_bars_until_closed(
            opp["legs"], bars,
            btc_usdt_series=btc_close,
            default_btc_usdt=float(btc_close.iloc[-1]),
            costs=costs,
            same_candle_conflict=str(sim.get("same_candle_conflict", "assume_sl_first")),
        )
        opp["last_processed_ts"] = str(bars.iloc[-1]["timestamp"])
        if all(leg.closed for leg in opp["legs"]):
            for leg in opp["legs"]:
                rec = leg_to_record(leg, oid)
                rec["entry_policy"] = opp.get("entry_policy")
                rec["recovered_by_late_allowed"] = bool(opp.get("recovered_by_late_allowed"))
                rec["entry_classification"] = opp.get("entry_classification")
                if rec.get("exit_ts") is not None:
                    try:
                        # day_number relative to eval requires caller context; leave for flush stamp
                        rec.setdefault("day_number", None)
                    except Exception:
                        pass
                leg_rows.append(rec)
            sm.register_close(oid, opp.get("symbol"))
            done.append(oid)
    for oid in done:
        del open_book[oid]


def _max_drawdown(pnl_series: pd.Series) -> float | None:
    if pnl_series is None or len(pnl_series) == 0:
        return None
    equity = pnl_series.cumsum()
    peak = equity.cummax()
    dd = equity - peak
    return float(dd.min()) if len(dd) else None


def _summarize(pred_df, legs_df, threshold, days, dom_series, *, max_simultaneous, max_open_rejects) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "days": days,
        "threshold": threshold,
        "n_predictions": int(len(pred_df)),
        "n_signals": int(pred_df["signal_generated"].sum()) if not pred_df.empty else 0,
        "n_trades_opened": int(pred_df["trade_opened"].sum()) if not pred_df.empty else 0,
        "max_simultaneous_opportunities": int(max_simultaneous),
        "n_rejected_max_open_trades": int(max_open_rejects),
        "dominance_status": (dom_series.meta or {}).get("status"),
        "dominance_source": (dom_series.meta or {}).get("source"),
        "same_candle_conflict": "assume_sl_first",
        "strategies": {},
        "rejection_counts": {},
    }
    if not pred_df.empty:
        if "rejection_reason" in pred_df.columns:
            summary["rejection_counts"] = pred_df["rejection_reason"].value_counts(dropna=True).to_dict()
        sig = pred_df[pred_df["signal_generated"] == True]  # noqa: E712
        if not sig.empty and sig["future_return_4h"].notna().any():
            summary["signal_hit_rate_4h"] = float((sig["future_return_4h"] > 0).mean())
            summary["mean_signal_future_return_4h"] = float(sig["future_return_4h"].mean())
            summary["prediction_accuracy_sign"] = float(
                ((sig["S"] >= 0) == (sig["future_return_4h"] > 0)).mean()
            )
    if not legs_df.empty and "strategy_key" in legs_df.columns:
        for key, g in legs_df.groupby("strategy_key"):
            g = g.sort_values("entry_ts") if "entry_ts" in g.columns else g
            pnl = g["pnl_btc"].dropna() if "pnl_btc" in g.columns else pd.Series(dtype=float)
            pnl_usd = g["pnl_usd_equiv"].dropna() if "pnl_usd_equiv" in g.columns else pd.Series(dtype=float)
            wins = int((pnl > 0).sum()) if len(pnl) else 0
            losses = int((pnl <= 0).sum()) if len(pnl) else 0
            gp = float(pnl[pnl > 0].sum()) if len(pnl) else 0.0
            gl = float((-pnl[pnl <= 0]).sum()) if len(pnl) else 0.0
            pct = g["pnl_pct"].dropna() if "pnl_pct" in g.columns else pd.Series(dtype=float)
            summary["strategies"][str(key)] = {
                "n_trades": int(len(g)),
                "opportunities": int(g["opportunity_id"].nunique()) if "opportunity_id" in g.columns else int(len(g)),
                "winning_trades": wins,
                "losing_trades": losses,
                "win_rate": float(wins / len(pnl)) if len(pnl) else None,
                "average_return_pct": float(pct.mean()) if len(pct) else None,
                "mean_pnl_btc": float(pnl.mean()) if len(pnl) else None,
                "sum_pnl_btc": float(pnl.sum()) if len(pnl) else None,
                "net_btc_pnl": float(pnl.sum()) if len(pnl) else None,
                "cumulative_btc": float(pnl.sum()) if len(pnl) else None,
                "sum_pnl_usd_equiv": float(pnl_usd.sum()) if len(pnl_usd) else None,
                "max_drawdown_btc": _max_drawdown(pnl.reset_index(drop=True)),
                "profit_factor": (gp / gl) if gl > 0 else None,
                "mean_holding_hours": float(g["holding_hours"].mean()) if "holding_hours" in g.columns else None,
                "exit_reasons": g["exit_reason"].value_counts().to_dict() if "exit_reason" in g.columns else {},
            }
        if "entry_policy" in legs_df.columns:
            summary["by_entry_policy"] = {}
            for pol, pg in legs_df.groupby("entry_policy"):
                summary["by_entry_policy"][str(pol)] = {}
                for key, g in pg.groupby("strategy_key"):
                    pnl = pd.to_numeric(g.get("pnl_btc"), errors="coerce").dropna()
                    wins = int((pnl > 0).sum()) if len(pnl) else 0
                    gp = float(pnl[pnl > 0].sum()) if len(pnl) else 0.0
                    gl = float((-pnl[pnl <= 0]).sum()) if len(pnl) else 0.0
                    summary["by_entry_policy"][str(pol)][str(key)] = {
                        "n_trades": int(len(g)),
                        "win_rate": float(wins / len(pnl)) if len(pnl) else None,
                        "sum_pnl_btc": float(pnl.sum()) if len(pnl) else None,
                        "max_drawdown_btc": _max_drawdown(pnl.reset_index(drop=True)),
                        "profit_factor": (gp / gl) if gl > 0 else None,
                        "mean_holding_hours": float(g["holding_hours"].mean()) if "holding_hours" in g.columns else None,
                    }
            if "recovered_by_late_allowed" in legs_df.columns:
                rec = legs_df[legs_df["recovered_by_late_allowed"] == True]  # noqa: E712
                summary["recovered_trades"] = {}
                for key, g in rec.groupby("strategy_key"):
                    pnl = pd.to_numeric(g.get("pnl_btc"), errors="coerce").dropna()
                    wins = int((pnl > 0).sum()) if len(pnl) else 0
                    gp = float(pnl[pnl > 0].sum()) if len(pnl) else 0.0
                    gl = float((-pnl[pnl <= 0]).sum()) if len(pnl) else 0.0
                    summary["recovered_trades"][str(key)] = {
                        "n_trades": int(len(g)),
                        "winning_trades": wins,
                        "losing_trades": int(len(pnl) - wins) if len(pnl) else 0,
                        "win_rate": float(wins / len(pnl)) if len(pnl) else None,
                        "mean_pnl_btc": float(pnl.mean()) if len(pnl) else None,
                        "sum_pnl_btc": float(pnl.sum()) if len(pnl) else None,
                        "sum_pnl_usd_equiv": float(pd.to_numeric(g.get("pnl_usd_equiv"), errors="coerce").dropna().sum()) if "pnl_usd_equiv" in g.columns else None,
                        "max_drawdown_btc": _max_drawdown(pnl.reset_index(drop=True)),
                        "profit_factor": (gp / gl) if gl > 0 else None,
                        "mean_holding_hours": float(g["holding_hours"].mean()) if "holding_hours" in g.columns else None,
                    }
    if not pred_df.empty and "entry_classification" in pred_df.columns:
        summary["entry_classification_counts"] = pred_df["entry_classification"].value_counts(dropna=True).to_dict()
    return summary


def _report_md(summary: dict[str, Any], sim: dict[str, Any], update_log: list) -> str:
    lines = [
        "# BTCC Adaptive V2 Backtest Report",
        "",
        f"**PAPER ONLY — LONG ALT/BTC — {summary.get('experiment_label', 'historical backtest')}**",
        "",
        f"- Protocol: `{summary.get('protocol')}`",
        f"- Eval: {summary.get('eval_start')} → {summary.get('eval_end')}",
        f"- Init: {summary.get('init_start')} → {summary.get('init_end')}",
        f"- Daily phase starts: {summary.get('daily_phase_start')}",
        f"- Threshold: {summary.get('threshold')}",
        f"- Signals / opens: {summary.get('n_signals')} / {summary.get('n_trades_opened')}",
        f"- Max simultaneous: {summary.get('max_simultaneous_opportunities')}",
        f"- MAX_OPEN rejects: {summary.get('n_rejected_max_open_trades')}",
        f"- BTC.D: {summary.get('dominance_source')} calib={(summary.get('btc_d') or {}).get('calibration')}",
        f"- Universe: {(summary.get('universe_policy') or {}).get('mode')} "
        f"(bias={(summary.get('universe_policy') or {}).get('survivorship_bias')})",
        "",
        "## Time-varying pair availability",
        "",
        summary.get("coverage_note") or "",
        "",
        f"- Requested days: {summary.get('requested_days')}",
        f"- Normal-pair coverage (approx): {summary.get('actual_normal_pair_coverage_days')} days",
        "",
    ]
    cov = summary.get("pair_coverage") or {}
    if cov:
        lines.append("| logical | resolved | first | last | days | candles | init90 | daily |")
        lines.append("|---|---|---|---|---:|---:|---|---|")
        for base in sorted(cov):
            c = cov[base]
            lines.append(
                f"| {c.get('logical_pair', base)} | {c.get('resolved_market')} | "
                f"{c.get('first_timestamp')} | {c.get('last_timestamp')} | "
                f"{c.get('available_days'):.1f} | {c.get('candle_count')} | "
                f"{c.get('usable_for_initial_90d')} | {c.get('usable_for_daily_adaptation')} |"
            )
        lines.append("")
    lines += [
        "## Weight updates",
        "",
    ]
    for u in update_log[:20]:
        lines.append(
            f"- #{u.get('update_number')} {u.get('phase')} calc={u.get('calculated_at')} "
            f"eff={u.get('effective_from')} status={u.get('status')} n={u.get('n_samples')}"
        )
    if len(update_log) > 20:
        lines.append(f"- ... ({len(update_log) - 20} more)")
    lines += ["", "## Strategies", ""]
    for k, st in (summary.get("strategies") or {}).items():
        lines.append(
            f"- **{k}**: n={st.get('n_trades')} win={st.get('win_rate')} "
            f"sum_btc={st.get('sum_pnl_btc')} DD={st.get('max_drawdown_btc')} "
            f"PF={st.get('profit_factor')} exits={st.get('exit_reasons')}"
        )
    return "\n".join(lines)


def run_threshold_sweep(
    *,
    days: int = 90,
    thresholds: list[float] | None = None,
    force_download: bool = False,
    weight_modes: list[str] | None = None,
) -> Path:
    sim = load_sim_config()
    thresholds = thresholds or list((sim.get("threshold_sweep") or {}).get("values") or [0.6])
    weight_modes = weight_modes or ["static", "equal", "adaptive"]
    root = Path(sim.get("_root") or Path(__file__).resolve().parents[2])
    sweep_dir = root / "results" / f"threshold_sweep_{days}d_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    first = True
    for mode in weight_modes:
        for thr in thresholds:
            logger.info("Threshold sweep mode=%s thr=%.2f", mode, thr)
            out = run_adaptive_sim_backtest(
                days=days,
                sim_cfg=sim,
                force_download=force_download and first,
                long_threshold=float(thr),
                weight_mode=mode,
                out_root=sweep_dir,
            )
            first = False
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            row = {
                "weight_mode": mode,
                "threshold": thr,
                "out_dir": str(out),
                "n_predictions": summary.get("n_predictions"),
                "n_signals": summary.get("n_signals"),
                "n_opportunities": summary.get("n_trades_opened"),
                "signal_hit_rate_4h": summary.get("signal_hit_rate_4h"),
                "prediction_accuracy_sign": summary.get("prediction_accuracy_sign"),
                "mean_signal_future_return_4h": summary.get("mean_signal_future_return_4h"),
                "max_simultaneous_opportunities": summary.get("max_simultaneous_opportunities"),
                "n_rejected_max_open_trades": summary.get("n_rejected_max_open_trades"),
                "entry_policies": summary.get("entry_policies"),
                "n_recovered_late_allowed": summary.get("n_recovered_late_allowed"),
            }
            for sk, st in (summary.get("strategies") or {}).items():
                row[f"{sk}_n"] = st.get("n_trades")
                row[f"{sk}_win_rate"] = st.get("win_rate")
                row[f"{sk}_sum_pnl_btc"] = st.get("sum_pnl_btc")
                row[f"{sk}_sum_pnl_usd"] = st.get("sum_pnl_usd_equiv")
                row[f"{sk}_mean_pnl_btc"] = st.get("mean_pnl_btc")
                row[f"{sk}_max_drawdown_btc"] = st.get("max_drawdown_btc")
                row[f"{sk}_profit_factor"] = st.get("profit_factor")
                row[f"{sk}_mean_holding_hours"] = st.get("mean_holding_hours")
            if summary.get("by_entry_policy"):
                for pol, by_sk in summary["by_entry_policy"].items():
                    for sk, pst in by_sk.items():
                        row[f"{pol}_{sk}_n"] = pst.get("n_trades")
                        row[f"{pol}_{sk}_sum_pnl_btc"] = pst.get("sum_pnl_btc")
            rows.append(row)
    pd.DataFrame(rows).to_csv(sweep_dir / "threshold_sweep_summary.csv", index=False)
    (sweep_dir / "threshold_sweep_summary.json").write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    write_run_fingerprint(
        sweep_dir,
        root=root,
        sim_cfg=sim,
        signal_cfg=load_backtest_config(),
        meta={
            "days": days,
            "thresholds": thresholds,
            "weight_modes": weight_modes,
            "kind": "threshold_sweep",
            "note": "Research only — primary experiment fixed at threshold=0.60; no auto-selection",
        },
    )
    return sweep_dir
