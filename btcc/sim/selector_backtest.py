"""Selector experiment backtest — fixed trails + dynamic selectors A–F.

Historical only. Chronological selection from counterfactual T history. No live handoff.
Strategy universe comes from the loaded sim config (T1–T12 for 1y; T1–T10 for 3y).
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
from btcc.backtest.dominance_history import HistoricalDominanceSeries, btc_d_manifest_fields
from btcc.backtest.predict import predict_coin_at_bar
from btcc.sim.accounting import CostModel, close_long_alt_btc
from btcc.sim.checkpoint import verify_final_checkpoint, write_daily_checkpoint, write_final_checkpoint
from btcc.sim.day_axis import day_number_at
from btcc.sim.exits import (
    StrategySpec,
    leg_to_record,
    open_opportunity_legs,
    process_bars_until_closed,
    specs_from_config,
)
from btcc.sim.fingerprint import write_run_fingerprint
from btcc.sim.health import evaluate_health
from btcc.sim.regime import classify_regime
from btcc.sim.score import FACTOR_KEYS, combined_score, extract_factor_scores, static_factor_weights
from btcc.sim.selector_config import (
    _label_for_key,
    active_fixed_arm_labels,
    active_fixed_strategy_keys,
    load_selector_experiment_config,
    validate_selector_experiment,
)
from btcc.sim.selector_engine import (
    CounterfactualHistory,
    build_selector_group,
    oracle_best_counterfactual,
)
from btcc.sim.state_machine import CrossingStateMachine
from btcc.sim.trail_entry import evaluate_trail_entry
from btcc.sim.backtest import _experiment_label, _normal_pair_coverage_days, _summarize
from btcc.series.relative import horizon_bars

logger = logging.getLogger(__name__)


def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _spec_map(sim: dict[str, Any]) -> dict[str, StrategySpec]:
    return {s.key: s for s in specs_from_config(sim)}


def _advance_selector_book(
    open_book: dict[str, dict[str, Any]],
    sm,
    panels,
    btc_close,
    costs,
    sim,
    asof_t,
    leg_rows: list[dict],
    opp_rows: list[dict],
    cf_history: CounterfactualHistory,
    history_recorded: set[tuple[str, str]],
) -> None:
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
        legs_only = [item["leg"] for item in opp["leg_items"]]
        process_bars_until_closed(
            legs_only,
            bars,
            btc_usdt_series=btc_close,
            default_btc_usdt=float(btc_close.iloc[-1]),
            costs=costs,
            same_candle_conflict=str(sim.get("same_candle_conflict", "assume_sl_first")),
        )
        opp["last_processed_ts"] = str(bars.iloc[-1]["timestamp"])
        regime = opp.get("regime")
        for item in opp["leg_items"]:
            leg = item["leg"]
            if not leg.closed:
                continue
            sk = leg.spec.key
            key = (str(oid), sk)
            if item.get("is_counterfactual") and key not in history_recorded:
                pnl = float((leg.exit_result or {}).get("pnl_pct") or 0.0)
                cf_history.record(
                    opportunity_id=oid,
                    strategy_key=sk,
                    exit_ts=leg.exit_ts,
                    pnl_pct=pnl,
                    regime=regime,
                )
                history_recorded.add(key)
        if all(item["leg"].closed for item in opp["leg_items"]):
            cf_pnls: dict[str, float] = {}
            for item in opp["leg_items"]:
                leg = item["leg"]
                rec = leg_to_record(leg, oid)
                rec["arm_key"] = item["arm_key"]
                rec["is_counterfactual"] = bool(item.get("is_counterfactual"))
                rec["selector_id"] = item.get("selector_id")
                rec["regime"] = regime
                rec["entry_policy"] = "COMMON"
                if item.get("is_counterfactual"):
                    cf_pnls[leg.spec.key] = float((leg.exit_result or {}).get("pnl_pct") or 0.0)
                leg_rows.append(rec)
            best_k, best_v = oracle_best_counterfactual(cf_pnls)
            oracle = {
                "oracle_strategy_key": best_k,
                "oracle_arm_label": _label_for_key(best_k),
                "oracle_pnl_pct": best_v,
            }
            opp["_oracle"] = oracle
            for o in opp_rows:
                if o.get("opportunity_id") == oid:
                    o.update(oracle)
                    o["status"] = "CLOSED"
            sm.register_close(oid, opp.get("symbol"))
            done.append(oid)
    for oid in done:
        del open_book[oid]


def _save_selector_state(out_dir: Path, *, cf_history: CounterfactualHistory, selectors: dict, sm: CrossingStateMachine) -> None:
    payload = {
        "counterfactual_history": cf_history.to_dict(),
        "selectors": {k: v.to_dict() for k, v in selectors.items()},
        "state_machine": sm.to_dict() if hasattr(sm, "to_dict") else None,
    }
    (out_dir / "selector_runtime_state.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _load_resume_state(out_dir: Path) -> dict[str, Any] | None:
    p = out_dir / "selector_runtime_state.json"
    if not p.exists():
        ck_dirs = sorted((out_dir / "daily_checkpoints").glob("day_*"), key=lambda x: int(x.name.split("_")[1]))
        if ck_dirs:
            p2 = ck_dirs[-1] / "selector_runtime_state.json"
            if p2.exists():
                p = p2
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def run_selector_experiment_backtest(
    *,
    days: int = 365,
    force_download: bool = False,
    out_root: Path | None = None,
    out_dir: Path | None = None,
    sim_cfg: dict[str, Any] | None = None,
    resume: bool = True,
) -> Path:
    sim = dict(sim_cfg or load_selector_experiment_config())
    errs = validate_selector_experiment(sim)
    if errs:
        raise ValueError("Selector experiment validation failed: " + "; ".join(errs))
    fixed_keys = active_fixed_strategy_keys(sim)
    fixed_labels = active_fixed_arm_labels(sim)
    if not fixed_keys:
        raise ValueError("No fixed strategy keys configured")

    bt_cfg = load_backtest_config()
    # Experiment-level interval / warmup overrides (e.g. 5m threshold sweep).
    se0 = sim.get("selector_experiment") or {}
    interval_override = sim.get("candle_interval") or se0.get("candle_interval") or se0.get("interval")
    if interval_override:
        bt_cfg = {**bt_cfg, "backtest": {**(bt_cfg.get("backtest") or {}), "interval": str(interval_override)}}
    warmup_override = sim.get("min_warmup_bars")
    if warmup_override is None:
        warmup_override = se0.get("min_warmup_bars")
    if warmup_override is not None:
        bt_cfg = {
            **bt_cfg,
            "backtest": {**(bt_cfg.get("backtest") or {}), "min_warmup_bars": int(warmup_override)},
        }
    # Separate candle root (e.g. Binance Vision) — do not overwrite MEXC cache.
    candle_dir_override = sim.get("candle_dir") or se0.get("candle_dir")
    if candle_dir_override:
        p = Path(candle_dir_override)
        if not p.is_absolute():
            p = Path(bt_cfg.get("_backtest_root") or Path(__file__).resolve().parents[2]) / p
        bt_cfg = {
            **bt_cfg,
            "backtest_data": {**(bt_cfg.get("backtest_data") or {}), "candle_dir": str(p)},
        }
    # Universe subset override (drop pairs missing on Binance, etc.)
    bases_override = sim.get("universe_bases") or se0.get("universe_bases")
    if bases_override:
        univ = dict(bt_cfg.get("universe") or {})
        univ["bases"] = list(bases_override)
        # Clear native-only list entries not in bases
        native = [b for b in (univ.get("native_btc_only") or []) if b in univ["bases"]]
        univ["native_btc_only"] = native
        bt_cfg = {**bt_cfg, "universe": univ}

    cfg = {**bt_cfg, "sim": sim}
    root = Path(bt_cfg.get("_root") or Path(__file__).resolve().parents[2])
    out_root = Path(out_root) if out_root else root / "results"

    if out_dir is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        prefix = (sim.get("selector_experiment") or {}).get("experiment_id_prefix") or f"selector_compare_{days}d_thr0p60"
        out_dir = out_root / f"{prefix}_{run_id}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lo = float(sim["long_threshold"])
    hi = sim.get("upper_threshold")
    thr_strict = bool(sim.get("threshold_strict") or se0.get("threshold_strict"))
    interval = bt_cfg["backtest"]["interval"]
    min_warmup = int(bt_cfg["backtest"]["min_warmup_bars"])
    horizon_h = int(sim.get("primary_horizon_hours", 4))
    bars_4h = horizon_bars(horizon_h, interval)

    weights = static_factor_weights(cfg)
    (out_dir / "fixed_weights.json").write_text(
        json.dumps({"weight_mode": "selector_static", "weights": weights}, indent=2),
        encoding="utf-8",
    )

    pinned_start = sim.get("eval_start") or se0.get("eval_start")
    pinned_end = sim.get("eval_end") or se0.get("eval_end")
    offline = bool(sim.get("offline_candles") or se0.get("offline_candles"))
    venue = str(sim.get("candle_venue") or se0.get("candle_venue") or ("binance" if offline else "mexc"))

    panels = download_panels(
        cfg,
        days,
        min_warmup,
        force=force_download,
        eval_start=pinned_start,
        eval_end=pinned_end,
        offline_candles=offline,
    )
    btc_df = panels["btc"].copy()
    btc_df["timestamp"] = pd.to_datetime(btc_df["timestamp"], utc=True)
    eval_start = _utc(panels["window"]["eval_start"])
    eval_end = _utc(panels["window"]["eval_end"])
    btc_close = btc_df.set_index("timestamp")["close"]

    dom_series = HistoricalDominanceSeries.fetch_for_backtest(
        days=int(days),
        cache_dir=bt_cfg["backtest_data"]["dominance_cache"],
        sim_cfg=sim,
        force=force_download,
    )

    ts_all = pd.to_datetime(btc_df["timestamp"], utc=True)
    decision_indices = [
        i for i in btc_df.index.tolist()
        if i >= min_warmup and i + 1 < len(btc_df)
        and ts_all.iloc[i] >= eval_start and ts_all.iloc[i] <= eval_end
    ]

    spec_by_key = _spec_map(sim)
    fixed_specs = [spec_by_key[k] for k in fixed_keys]
    regime_rules = (sim.get("selector_experiment") or {}).get("regime", {}).get("rules") or {}
    sw = (sim.get("selector_experiment") or {}).get("switching") or {}

    sm = CrossingStateMachine(
        long_threshold=lo,
        upper_threshold=hi,
        max_open=int(sim.get("max_open_opportunities", 10)),
        one_per_pair=bool(sim.get("one_opportunity_per_pair", True)),
        threshold_strict=thr_strict,
    )
    selectors = build_selector_group(sim)
    cf_history = CounterfactualHistory()
    history_recorded: set[tuple[str, str]] = set()

    pred_rows: list[dict] = []
    opp_rows: list[dict] = []
    leg_rows: list[dict] = []
    selection_rows: list[dict] = []
    open_book: dict[str, dict[str, Any]] = {}
    prev_day_number: int | None = None
    last_flushed_day: int | None = None
    start_decision_idx = 0
    max_simultaneous = 0
    max_open_rejects = 0
    own_cache: dict = {}

    resume_state = _load_resume_state(out_dir) if resume else None
    if resume_state:
        cf_history = CounterfactualHistory.from_dict(resume_state.get("counterfactual_history") or {})
        for sid, raw in (resume_state.get("selectors") or {}).items():
            from btcc.sim.selector_engine import SelectorState
            selectors[sid] = SelectorState.from_dict(raw)
        sm_data = resume_state.get("state_machine")
        if sm_data:
            sm = CrossingStateMachine.from_dict(sm_data)
        for name in ("predictions.csv", "opportunities.csv", "strategy_legs.csv", "selection_audit.csv"):
            p = out_dir / name
            if p.exists() and p.stat().st_size > 0:
                try:
                    df = pd.read_csv(p)
                except pd.errors.EmptyDataError:
                    continue
                if df is None or df.empty:
                    continue
                rows = df.to_dict("records")
                if name == "predictions.csv":
                    pred_rows = rows
                elif name == "opportunities.csv":
                    opp_rows = rows
                elif name == "strategy_legs.csv":
                    leg_rows = rows
                elif name == "selection_audit.csv":
                    selection_rows = rows
        for rec in leg_rows:
            if rec.get("is_counterfactual") in (True, "True", "true", 1):
                history_recorded.add((str(rec["opportunity_id"]), str(rec["strategy_key"])))
        if pred_rows:
            last_day = max(int(r.get("day_number") or 0) for r in pred_rows)
            last_flushed_day = last_day
            prev_day_number = last_day
            last_ts = max(str(r.get("timestamp")) for r in pred_rows if r.get("timestamp"))
            for idx, i in enumerate(decision_indices):
                if str(btc_df.iloc[i]["timestamp"]) > last_ts:
                    start_decision_idx = idx
                    break
            logger.info("Resuming selector experiment from day=%s decision_idx=%s", last_day, start_decision_idx)

    costs = CostModel(
        fee_rate_per_side=float(sim.get("fee_rate_per_side", 0.001)),
        slippage_rate_per_side=float(sim.get("slippage_rate_per_side", 0.0005)),
    )

    def _stamp_entry_days() -> None:
        oid_day = {
            str(o["opportunity_id"]): int(o["day_number"])
            for o in opp_rows
            if o.get("opportunity_id") is not None and o.get("day_number") is not None
        }
        for rec in leg_rows:
            if rec.get("entry_day_number") is None:
                oid = rec.get("opportunity_id")
                if oid is not None and str(oid) in oid_day:
                    rec["entry_day_number"] = oid_day[str(oid)]

    def _flush_day(day_n: int, ts_end: pd.Timestamp) -> None:
        nonlocal last_flushed_day
        if last_flushed_day == day_n:
            return
        _stamp_entry_days()
        pd.DataFrame(selection_rows).to_csv(out_dir / "selection_audit.csv", index=False)
        open_snap = {
            oid: {
                **{k: v for k, v in opp.items() if k not in ("leg_items", "_oracle")},
                "n_open_legs": sum(1 for it in opp.get("leg_items", []) if not it["leg"].closed),
            }
            for oid, opp in open_book.items()
        }
        for rec in leg_rows:
            if rec.get("day_number") is None and rec.get("exit_ts"):
                try:
                    rec["day_number"] = day_number_at(rec["exit_ts"], eval_start)
                except Exception:
                    rec["day_number"] = day_n
        ddir = write_daily_checkpoint(
            out_dir,
            day_number=day_n,
            simulated_timestamp=str(ts_end),
            pred_rows=pred_rows,
            opp_rows=opp_rows,
            leg_rows=leg_rows,
            weight_hist=[],
            update_log=[],
            schedule_history={"updates": []},
            schedule_active={"version_id": "selector_static", "weights": weights},
            open_books_snapshot={"COMMON": open_snap},
            weight_mode="selector_static",
            init_days=0,
            eval_start=str(eval_start),
            eval_end=str(eval_end),
            coverage=panels.get("coverage"),
            refresh_analytics=True,
            telegram_enabled=False,
            starting_capital_usd=float(sim.get("starting_capital_usd", 1000.0)),
        )
        _save_selector_state(out_dir, cf_history=cf_history, selectors=selectors, sm=sm)
        if ddir:
            (ddir / "selector_runtime_state.json").write_text(
                (out_dir / "selector_runtime_state.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        last_flushed_day = day_n

    for n_done, i in enumerate(decision_indices[start_decision_idx:], start=start_decision_idx):
        t = _utc(btc_df.iloc[i]["timestamp"])
        day_n = day_number_at(t, eval_start)
        if prev_day_number is not None and day_n != prev_day_number:
            _flush_day(prev_day_number, t)
        prev_day_number = day_n

        _advance_selector_book(
            open_book, sm, panels, btc_close, costs, sim, t, leg_rows, opp_rows,
            cf_history, history_recorded,
        )
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

                    cf_legs = open_opportunity_legs(
                        alt_btc_entry_mid=entry_mid,
                        btc_usdt=btc_usdt,
                        notional_usd=float(sim.get("notional_usd", 100.0)),
                        costs=costs,
                        specs=fixed_specs,
                        entry_ts=entry_ts,
                    )
                    leg_items: list[dict[str, Any]] = []
                    for arm_label, leg in zip(fixed_labels, cf_legs):
                        leg_items.append({
                            "leg": leg,
                            "arm_key": arm_label,
                            "is_counterfactual": True,
                            "selector_id": None,
                        })

                    selector_picks: dict[str, dict] = {}
                    for sel in selectors.values():
                        pick = sel.select(cf_history, entry_ts, regime_info["regime"], strategy_keys=fixed_keys)
                        selector_picks[sel.selector_id] = pick
                        sk = pick["selected_strategy_key"]
                        sel_spec = spec_by_key[sk]
                        sel_legs = open_opportunity_legs(
                            alt_btc_entry_mid=entry_mid,
                            btc_usdt=btc_usdt,
                            notional_usd=float(sim.get("notional_usd", 100.0)),
                            costs=costs,
                            specs=[sel_spec],
                            entry_ts=entry_ts,
                        )
                        leg_items.append({
                            "leg": sel_legs[0],
                            "arm_key": sel.arm_label,
                            "is_counterfactual": False,
                            "selector_id": sel.selector_id,
                        })
                        selection_rows.append({
                            "opportunity_id": opportunity_id,
                            "entry_ts": str(entry_ts),
                            "day_number": day_n,
                            "selector_id": sel.selector_id,
                            "arm_label": sel.arm_label,
                            "regime": regime_info["regime"],
                            "selected_strategy_key": sk,
                            "selected_arm_label": pick["selected_arm_label"],
                            "selected_score": pick["selected_score"],
                            "second_best_strategy_key": pick["second_best_strategy_key"],
                            "second_best_score": pick["second_best_score"],
                            "strategy_rank": pick["strategy_rank"],
                            "switched": pick["switched"],
                            **{f"score_{k}": pick["scores"].get(k) for k in fixed_keys},
                        })

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
                        "leg_items": leg_items,
                        "last_processed_ts": str(entry_ts),
                        "regime": regime_info["regime"],
                        "regime_adx": regime_info["adx"],
                        "regime_natr": regime_info["natr"],
                        "selector_picks": selector_picks,
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
                        "weight_mode": "selector_static",
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
            logger.info(
                "Selector sim %d/%d day=%s open=%d opps=%d cf_trades=%d",
                n_done, len(decision_indices), day_n, sm.n_open(), len(opp_rows), len(cf_history.trades),
            )

    if prev_day_number is not None and decision_indices:
        _flush_day(prev_day_number, _utc(btc_df.iloc[decision_indices[-1]]["timestamp"]))

    for oid, opp in list(open_book.items()):
        coin = panels["coins"].get(opp["base"])
        mid = float(coin["rel"].iloc[-1]["close"]) if coin else opp["entry_alt_btc_mid"]
        btc_usdt = float(btc_df.iloc[-1]["close"])
        cf_pnls: dict[str, float] = {}
        for item in opp["leg_items"]:
            leg = item["leg"]
            if not leg.closed:
                leg.closed = True
                leg.exit_reason = "END_OF_BACKTEST"
                leg.exit_ts = btc_df.iloc[-1]["timestamp"]
                leg.exit_result = close_long_alt_btc(position=leg.position, alt_btc_mid=mid, btc_usdt=btc_usdt, costs=costs)
            rec = leg_to_record(leg, oid)
            rec["arm_key"] = item["arm_key"]
            rec["is_counterfactual"] = bool(item.get("is_counterfactual"))
            rec["selector_id"] = item.get("selector_id")
            rec["regime"] = opp.get("regime")
            rec["day_number"] = day_number_at(rec.get("exit_ts") or eval_end, eval_start)
            if item.get("is_counterfactual"):
                cf_pnls[leg.spec.key] = float((leg.exit_result or {}).get("pnl_pct") or 0.0)
            leg_rows.append(rec)
        best_k, best_v = oracle_best_counterfactual(cf_pnls)
        for o in opp_rows:
            if o.get("opportunity_id") == oid:
                o["oracle_strategy_key"] = best_k
                o["oracle_arm_label"] = _label_for_key(best_k)
                o["oracle_pnl_pct"] = best_v
        sm.register_close(oid, opp.get("symbol"))
    open_book.clear()
    _stamp_entry_days()

    pred_df = pd.DataFrame(pred_rows)
    opp_df = pd.DataFrame(opp_rows)
    legs_df = pd.DataFrame(leg_rows)
    sel_df = pd.DataFrame(selection_rows)

    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    opp_df.to_csv(out_dir / "opportunities.csv", index=False)
    legs_df.to_csv(out_dir / "strategy_legs.csv", index=False)
    sel_df.to_csv(out_dir / "selection_audit.csv", index=False)

    summary = _summarize(pred_df, legs_df, lo, days, dom_series, max_simultaneous=max_simultaneous, max_open_rejects=max_open_rejects)
    summary.update({
        "experiment_kind": sim.get("experiment_kind") or "selector_experiment_v1",
        "weight_mode": "selector_static",
        "upper_threshold": hi,
        "threshold_strict": thr_strict,
        "threshold_operator": ">" if thr_strict else ">=",
        "candle_interval": interval,
        "min_warmup_bars": min_warmup,
        "n_arms": len(fixed_labels) + len(selectors),
        "fixed_strategies": list(fixed_labels),
        "selectors": [s.arm_label for s in selectors.values()],
        "n_opportunities": len(opp_df),
        "requested_days": days,
        "actual_normal_pair_coverage_days": _normal_pair_coverage_days(panels),
        "experiment_label": _experiment_label(days, panels),
        "switching": sw,
        "enable_period_analysis": bool(sim.get("enable_period_analysis", False)),
        "candle_venue": venue,
        "offline_candles": offline,
        "compare_baseline": se0.get("compare_baseline"),
        "data_window": {
            "data_start": str(panels["window"]["data_start"]),
            "eval_start": str(eval_start),
            "eval_end": str(eval_end),
            "n_btc_candles": int(len(btc_df)),
            "candle_dir": panels["window"].get("candle_dir"),
            "unavailable_bases": list(panels.get("unavailable") or []),
        },
    })
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    manifest = {
        "experiment_name": (sim.get("selector_experiment") or {}).get("experiment_id_prefix") or f"selector_compare_{days}d_thr0p60",
        "experiment_kind": sim.get("experiment_kind") or "selector_experiment_v1",
        "eval_start": str(eval_start),
        "eval_end": str(eval_end),
        "candle_interval": interval,
        "candle_venue": venue,
        "offline_candles": offline,
        "compare_baseline": se0.get("compare_baseline"),
        "universe_bases": list((cfg.get("universe") or {}).get("bases") or []),
        "entry": {
            "long_threshold": lo,
            "upper_threshold": hi,
            "threshold_strict": thr_strict,
            "operator": ">" if thr_strict else ">=",
        },
        "strategies": sim.get("strategies"),
        "fixed_strategy_keys": list(fixed_keys),
        "selectors": (sim.get("selector_experiment") or {}).get("selectors"),
        "switching": sw,
        "starting_capital_usd": float(sim.get("starting_capital_usd", 1000.0)),
        "notional_usd": float(sim.get("notional_usd", 100.0)),
        "enable_period_analysis": bool(sim.get("enable_period_analysis", False)),
        "fees": {"fee_rate_per_side": sim.get("fee_rate_per_side"), "slippage_rate_per_side": sim.get("slippage_rate_per_side")},
        **btc_d_manifest_fields(sim, dom_series),
    }
    (out_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

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
    manifest["git_sha"] = git_commit
    (out_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    fp = json.loads((out_dir / "fingerprint.json").read_text(encoding="utf-8")) if (out_dir / "fingerprint.json").exists() else None
    final = write_final_checkpoint(
        out_dir, summary=summary, fingerprint=fp,
        last_day_number=last_flushed_day or prev_day_number,
        git_commit=git_commit,
    )
    v = verify_final_checkpoint(final)
    summary["final_checkpoint"] = {"path": str(final), "verify": v, "status": "COMPLETED"}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    _save_selector_state(out_dir, cf_history=cf_history, selectors=selectors, sm=sm)

    if bool(sim.get("enable_period_analysis", False)):
        try:
            from btcc.analytics.selector_period_analysis import write_period_analysis
            from btcc.analytics.selector_period_plots import generate_period_plots

            paths = write_period_analysis(
                out_dir,
                legs=legs_df,
                selection=sel_df,
                eval_start=eval_start,
                eval_end=eval_end,
                starting_capital_usd=float(sim.get("starting_capital_usd", 1000.0)),
            )
            monthly = pd.read_csv(paths["monthly_results.csv"])
            yearly = pd.read_csv(paths["yearly_results.csv"])
            rankings = pd.read_csv(paths["monthly_rankings.csv"])
            generate_period_plots(
                monthly=monthly,
                yearly=yearly,
                rankings=rankings,
                out_dir=out_dir / "analytics" / "plots" / "periods",
            )
            summary["period_analysis"] = {k: str(v) for k, v in paths.items()}
            (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
            logger.info("Period analysis written → %s", out_dir / "analytics" / "periods")
        except Exception:
            logger.exception("Period analysis failed (experiment results still saved)")

    logger.info("Selector experiment complete → %s verify=%s", out_dir, v.get("ok"))
    return out_dir
