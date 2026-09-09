"""21-day entry-variant experiment: BASE vs E1–E5 early-breakout scoring.

Signals 15m, exits 1m, T1–T20 counterfactual legs per variant (no selectors).
Research only — does not touch live bot / orders / LIVE config.
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

TOOLS = Path(__file__).resolve().parent
REPO_ROOT = TOOLS.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from run_t_strategy_21d_1m_experiment import (  # noqa: E402
    ALLOC_FRAC,
    CANDLE_DIR,
    DAYS,
    EXIT_INTERVAL,
    LONG_THRESHOLD,
    MAX_OPEN,
    REPO,
    SIGNAL_INTERVAL,
    STARTING_BTC,
    WARMUP_BARS,
    _utc,
    download_candles,
    live_universe_bases,
    load_exit_panels,
    strategy_metrics,
    t11_t20_research,
    t1_t10_from_live,
    week_of,
)

from btcc.research.early_breakout_entry import (  # noqa: E402
    E3_MAX_15M_RET,
    E3_MAX_PCT_B,
    E3_MODERATE_BOOST,
    E3_RVOL_MIN,
    E3_STRONG_BOOST,
    score_all_variants,
)

VARIANTS = ("BASE", "E1", "E2", "E3", "E4", "E5")
OUT_DIR = REPO / "results" / "early_breakout_entry_21d_1m"
LABELS_T20 = tuple(f"T{i}" for i in range(1, 21))
MATCH_WINDOW_H = pd.Timedelta(hours=6)
LOOKBACK_H = pd.Timedelta(hours=12)
LATE_MIN_LEAD = pd.Timedelta(minutes=15)
FALSE_BREAKOUT_MFE = 0.005

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("early_breakout_entry_21d_1m")


def build_sim() -> dict[str, Any]:
    from btcc.sim.config import load_sim_config

    strategies = {**t1_t10_from_live(), **t11_t20_research()}
    sim = dict(load_sim_config())
    sim.update(
        {
            "experiment_kind": "early_breakout_entry_21d_1m",
            "long_threshold": LONG_THRESHOLD,
            "upper_threshold": None,
            "starting_capital_usd": 1000.0,
            "notional_usd": 125.0,
            "compound_portfolio": False,
            "max_open_opportunities": MAX_OPEN,
            "one_opportunity_per_pair": True,
            "fee_rate_per_side": 0.0,
            "slippage_rate_per_side": 0.0,
            "same_candle_conflict": "assume_sl_first",
            "disable_late_entry_rejection": True,
            "disable_weight_updates": True,
            "weight_mode": "static",
            "allow_trading": False,
            "telegram": "OFF",
            "btc_d": {"enabled": False},
            "btc_d_health": {"require_for_new_trades": False},
            "data_health": {"max_candle_age_seconds": 86400, "min_relative_bars": 100},
            "strategies": strategies,
            "benchmark_strategy_key": "trail_1",
        }
    )
    return sim


def _factor_fields(diag: dict[str, Any]) -> dict[str, Any]:
    return {
        "bb_breakout": bool(diag.get("bb_breakout")),
        "rsi_cross_70": bool(diag.get("rsi_cross_70")),
        "rvol": diag.get("rvol"),
        "ret_15m": diag.get("ret_15m"),
        "structure_breakout": bool(diag.get("structure_breakout")),
        "percent_b": diag.get("percent_b"),
    }


def _e3_fields(e3: dict[str, Any] | None) -> dict[str, Any]:
    if not e3:
        return {
            "e3_boost": None,
            "e3_n_conditions": None,
            "e3_chase_flag": None,
            "e3_rejected_chase": None,
            "e3_strong_breakout": None,
            "e3_moderate_breakout": None,
        }
    return {
        "e3_boost": e3.get("boost"),
        "e3_n_conditions": e3.get("n_conditions"),
        "e3_chase_flag": e3.get("chase_flag"),
        "e3_rejected_chase": e3.get("rejected_chase"),
        "e3_strong_breakout": e3.get("strong_breakout"),
        "e3_moderate_breakout": e3.get("moderate_breakout"),
    }


def milestone_times_from_1m(
    rel: pd.DataFrame,
    entry_ts: pd.Timestamp,
    entry_price: float,
    *,
    thresholds: tuple[tuple[str, float], ...] = (
        ("p0_5", 0.005),
        ("p0_75", 0.0075),
        ("p1", 0.01),
        ("p1_5", 0.015),
        ("p2", 0.02),
        ("p3", 0.03),
    ),
) -> dict[str, float | None]:
    """Minutes from entry to first 1m high reaching each return threshold."""
    out: dict[str, float | None] = {k: None for k, _ in thresholds}
    if entry_price <= 0 or rel is None or rel.empty:
        return out
    bars = rel[rel["timestamp"] > _utc(entry_ts)].copy()
    if bars.empty:
        return out
    entry_ts = _utc(entry_ts)
    for _, row in bars.iterrows():
        high = float(row["high"])
        ts = _utc(row["timestamp"])
        mins = (ts - entry_ts).total_seconds() / 60.0
        ret = (high / entry_price) - 1.0
        for label, thr in thresholds:
            if out[label] is None and ret >= thr:
                out[label] = mins
        if all(v is not None for v in out.values()):
            break
    return out


def _nearest_match(
    base_ts: pd.Timestamp,
    candidates: pd.DataFrame,
    *,
    window: pd.Timedelta = MATCH_WINDOW_H,
) -> pd.Series | None:
    if candidates.empty:
        return None
    c = candidates.copy()
    c["_ts"] = pd.to_datetime(c["entry_ts"], utc=True)
    c["_delta"] = (c["_ts"] - _utc(base_ts)).abs()
    c = c[c["_delta"] <= window]
    if c.empty:
        return None
    return c.sort_values("_delta").iloc[0]


def _capture_class(base_present: bool, exp_present: bool) -> str:
    if base_present and exp_present:
        return "BOTH"
    if base_present:
        return "BASE_ONLY"
    if exp_present:
        return "EXP_ONLY"
    return "NEITHER"


def _first_event_times(
    base: str,
    signal_ts: pd.Timestamp,
    diag_history: list[tuple[pd.Timestamp, dict[str, Any], dict[str, Any] | None]],
) -> dict[str, Any]:
    """Look back up to 12h before signal_ts for first breakout-related events."""
    signal_ts = _utc(signal_ts)
    window_start = signal_ts - LOOKBACK_H
    events = {
        "first_bb_breakout_ts": None,
        "first_rsi_cross_70_ts": None,
        "first_rvol_1_5_ts": None,
        "first_structure_breakout_ts": None,
        "first_e3_boost_ts": None,
    }
    for ts, diag, e3 in diag_history:
        if ts < window_start or ts > signal_ts:
            continue
        if events["first_bb_breakout_ts"] is None and diag.get("bb_breakout"):
            events["first_bb_breakout_ts"] = str(ts)
        if events["first_rsi_cross_70_ts"] is None and diag.get("rsi_cross_70"):
            events["first_rsi_cross_70_ts"] = str(ts)
        rvol = diag.get("rvol")
        if events["first_rvol_1_5_ts"] is None and rvol is not None and float(rvol) >= 1.5:
            events["first_rvol_1_5_ts"] = str(ts)
        if events["first_structure_breakout_ts"] is None and diag.get("structure_breakout"):
            events["first_structure_breakout_ts"] = str(ts)
        if events["first_e3_boost_ts"] is None and e3 is not None and float(e3.get("boost") or 0) > 0:
            events["first_e3_boost_ts"] = str(ts)
    return events


def run() -> Path:
    from btcc.backtest.config import load_backtest_config
    from btcc.backtest.data_loader import download_panels
    from btcc.backtest.dominance_history import HistoricalDominanceSeries
    from btcc.sim.accounting import CostModel
    from btcc.sim.day_axis import day_number_at
    from btcc.sim.exits import (
        leg_to_record,
        open_opportunity_legs,
        process_bars_until_closed,
        specs_from_config,
    )
    from btcc.sim.health import evaluate_health
    from btcc.sim.regime import classify_regime
    from btcc.sim.score import static_factor_weights
    from btcc.sim.state_machine import CrossingStateMachine
    from btcc.sim.trail_entry import evaluate_trail_entry

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    eval_end = _utc(datetime.now(timezone.utc)).floor("15min")
    eval_start = eval_end - pd.Timedelta(days=DAYS)

    bases = live_universe_bases()
    sim = build_sim()
    download_candles(bases, eval_start, eval_end)

    bt_cfg = load_backtest_config()
    bt_cfg = {
        **bt_cfg,
        "backtest": {
            **(bt_cfg.get("backtest") or {}),
            "interval": SIGNAL_INTERVAL,
            "min_warmup_bars": WARMUP_BARS,
        },
        "backtest_data": {**(bt_cfg.get("backtest_data") or {}), "candle_dir": str(CANDLE_DIR)},
        "universe": {
            **(bt_cfg.get("universe") or {}),
            "bases": list(bases),
            "quote": "USDT",
            "btc_symbol": "BTCUSDT",
            "native_btc_only": [],
            "btc_markets": {b: f"{b}BTC" for b in bases},
            "symbol_overrides": {},
        },
    }
    cfg = {**bt_cfg, "sim": sim}

    panels = download_panels(
        cfg,
        days=DAYS,
        warmup_bars=WARMUP_BARS,
        force=False,
        eval_start=eval_start,
        eval_end=eval_end,
        offline_candles=True,
    )
    exit_raw = load_exit_panels(bases)
    exit_panels = {
        "coins": {
            base: {**coin, "rel": exit_raw["coins"][base]["rel"]}
            for base, coin in panels["coins"].items()
            if base in exit_raw["coins"]
        }
    }
    btc_close_1m = exit_raw["btc_close"]
    btc_df = panels["btc"].copy()
    btc_df["timestamp"] = pd.to_datetime(btc_df["timestamp"], utc=True)
    eval_start = _utc(panels["window"]["eval_start"])
    eval_end = _utc(panels["window"]["eval_end"])

    keys_t20 = tuple(f"trail_{i}" for i in range(1, 21))
    spec_by_key = {s.key: s for s in specs_from_config(sim)}
    fixed_specs = [spec_by_key[k] for k in keys_t20]

    weights = static_factor_weights(cfg)
    costs = CostModel(0.0, 0.0)
    sms = {v: CrossingStateMachine(LONG_THRESHOLD, None, MAX_OPEN, True, False) for v in VARIANTS}
    prev_s_by_variant: dict[str, dict[str, float]] = {v: {} for v in VARIANTS}
    open_books: dict[str, dict[str, dict[str, Any]]] = {v: {} for v in VARIANTS}

    dom_series = HistoricalDominanceSeries.fetch_for_backtest(
        days=DAYS + 14,
        cache_dir=bt_cfg["backtest_data"]["dominance_cache"],
        sim_cfg=sim,
        force=False,
    )

    ts_all = pd.to_datetime(btc_df["timestamp"], utc=True)
    decision_indices = [
        i
        for i in btc_df.index.tolist()
        if i >= 100 and i + 1 < len(btc_df) and ts_all.iloc[i] >= eval_start and ts_all.iloc[i] <= eval_end
    ]

    n_15m = int(((ts_all >= eval_start) & (ts_all <= eval_end)).sum())
    n_1m = int(
        (
            (exit_raw["btc"]["timestamp"] >= eval_start - pd.Timedelta(hours=1))
            & (exit_raw["btc"]["timestamp"] <= eval_end + pd.Timedelta(hours=6))
        ).sum()
    )

    logger.info(
        "Window %s → %s | decisions=%d | 15m=%d | 1m≈%d | coins=%d | variants=%d",
        eval_start,
        eval_end,
        len(decision_indices),
        n_15m,
        n_1m,
        len(panels["coins"]),
        len(VARIANTS),
    )

    score_cache: dict[tuple[str, pd.Timestamp], dict[str, dict[str, Any]]] = {}
    diag_history_by_base: dict[str, list[tuple[pd.Timestamp, dict[str, Any], dict[str, Any] | None]]] = {
        b: [] for b in panels["coins"]
    }

    leg_rows: list[dict[str, Any]] = []
    opp_rows: list[dict[str, Any]] = []
    max_simultaneous: dict[str, int] = {v: 0 for v in VARIANTS}

    def notional_usd_at(entry_ts: pd.Timestamp) -> float:
        px = btc_close_1m
        try:
            hist = px[px.index <= entry_ts]
            btc_px = float(hist.iloc[-1]) if len(hist) else float(px.iloc[0])
        except Exception:
            btc_px = 100000.0
        return float(STARTING_BTC * ALLOC_FRAC * btc_px)

    def advance_variant(variant: str, asof_t: pd.Timestamp) -> None:
        book = open_books[variant]
        sm = sms[variant]
        done: list[str] = []
        for oid, opp in book.items():
            coin = exit_panels["coins"].get(opp["base"])
            if coin is None:
                continue
            rel = coin["rel"]
            last = _utc(opp.get("last_processed_ts") or opp["entry_fill_ts"])
            bars = rel[(rel["timestamp"] > last) & (rel["timestamp"] <= asof_t)]
            if bars.empty:
                continue
            legs_only = [item["leg"] for item in opp["leg_items"]]
            process_bars_until_closed(
                legs_only,
                bars,
                btc_usdt_series=btc_close_1m,
                default_btc_usdt=float(btc_close_1m.iloc[-1]),
                costs=costs,
                same_candle_conflict="assume_sl_first",
            )
            opp["last_processed_ts"] = str(bars.iloc[-1]["timestamp"])
            if all(item["leg"].closed for item in opp["leg_items"]):
                regime = opp.get("regime")
                for item in opp["leg_items"]:
                    leg = item["leg"]
                    rec = leg_to_record(leg, oid)
                    rec["variant"] = variant
                    rec["arm_key"] = item["arm_key"]
                    rec["is_counterfactual"] = True
                    rec["regime"] = regime
                    rec["base"] = opp.get("base")
                    rec["symbol"] = opp.get("symbol")
                    rec["signal_ts"] = opp.get("opened_ts")
                    rec["entry_S"] = opp.get("S")
                    rec["S_effective"] = opp.get("S_effective")
                    rec["prev_S"] = opp.get("prev_S")
                    rec["day_number"] = day_number_at(leg.exit_ts, eval_start) if leg.exit_ts is not None else None
                    rec["week"] = week_of(leg.exit_ts, eval_start) if leg.exit_ts is not None else None
                    leg_rows.append(rec)
                for o in opp_rows:
                    if o.get("opportunity_id") == oid:
                        o["status"] = "CLOSED"
                sm.register_close(oid, opp.get("symbol"))
                done.append(oid)
        for oid in done:
            del book[oid]
        max_simultaneous[variant] = max(max_simultaneous[variant], sm.n_open())

    def advance_all(asof_t: pd.Timestamp) -> None:
        for variant in VARIANTS:
            advance_variant(variant, asof_t)

    for n_done, i in enumerate(decision_indices):
        t = _utc(btc_df.iloc[i]["timestamp"])
        if n_done and n_done % 100 == 0:
            open_total = sum(sms[v].n_open() for v in VARIANTS)
            logger.info(
                "Progress %d/%d @ %s open=%d closed_legs=%d",
                n_done,
                len(decision_indices),
                t,
                open_total,
                len(leg_rows),
            )
        advance_all(t)

        btc_hist = btc_df.iloc[: i + 1].copy()
        dom_pct, dom_obs, _ = dom_series.observation_at(t)
        health = evaluate_health(
            sim_cfg=sim,
            dominance_pct=dom_pct,
            dominance_ts=dom_obs.to_pydatetime() if hasattr(dom_obs, "to_pydatetime") else dom_obs,
            dominance_source=(dom_series.meta or {}).get("source"),
            decision_candle_ts=t.to_pydatetime(),
            now=t.to_pydatetime(),
        )
        day_n = day_number_at(t, eval_start)

        for base, coin in panels["coins"].items():
            if base not in exit_panels["coins"]:
                continue
            rel_full = coin["rel"].copy()
            rel_full["timestamp"] = pd.to_datetime(rel_full["timestamp"], utc=True)
            rel_hist = rel_full[rel_full["timestamp"] <= t]
            if len(rel_hist) < 100:
                continue
            alt_vol = coin["alt_for_volume"].copy()
            alt_vol["timestamp"] = pd.to_datetime(alt_vol["timestamp"], utc=True)
            alt_vol_hist = alt_vol[alt_vol["timestamp"] <= t]

            cache_key = (base, t)
            if cache_key not in score_cache:
                scored = score_all_variants(
                    alt_btc=rel_hist,
                    alt_usdt=alt_vol_hist,
                    btc_usdt=btc_hist,
                    weights=weights,
                    interval=SIGNAL_INTERVAL,
                )
                score_cache[cache_key] = scored
                base_diag = scored["BASE"].get("diagnostics") or {}
                e3_assess = scored["E3"].get("e3")
                diag_history_by_base[base].append((t, base_diag, e3_assess))
            scored = score_cache[cache_key]

            pair = coin["symbol"]
            future = rel_full[rel_full["timestamp"] > t]
            if future.empty:
                continue

            for variant in VARIANTS:
                sv = scored[variant]
                s_eff = float(sv["S_effective"])
                sm = sms[variant]
                prev_s = prev_s_by_variant[variant].get(pair)
                prev_s_by_variant[variant][pair] = s_eff
                decision = sm.evaluate(pair, s_eff)
                ep = evaluate_trail_entry(
                    sm_decision=decision,
                    health_allow_new_trades=bool(health.allow_new_trades),
                )
                if not ep["trade_suggested"]:
                    continue

                entry_bar = future.iloc[0]
                entry_mid = float(entry_bar["open"])
                entry_ts = _utc(entry_bar["timestamp"])
                btc_hist_px = btc_df[btc_df["timestamp"] <= entry_ts]
                btc_usdt = (
                    float(btc_hist_px["close"].iloc[-1])
                    if not btc_hist_px.empty
                    else float(btc_hist["close"].iloc[-1])
                )
                notional = notional_usd_at(entry_ts)
                opportunity_id = f"opp_{variant.lower()}_{uuid.uuid4().hex[:10]}"
                # classify_regime expects nested factor blocks (trend/volatility), not flat diagnostics
                regime_info = classify_regime(
                    {
                        "trend": sv.get("trend") or {},
                        "volatility": sv.get("volatility") or {},
                    },
                    rules={},
                )
                diag = sv.get("diagnostics") or {}
                e3 = sv.get("e3")

                cf_legs = open_opportunity_legs(
                    alt_btc_entry_mid=entry_mid,
                    btc_usdt=btc_usdt,
                    notional_usd=notional,
                    costs=costs,
                    specs=fixed_specs,
                    entry_ts=entry_ts,
                )
                leg_items = [
                    {"leg": leg, "arm_key": arm_label}
                    for arm_label, leg in zip(LABELS_T20, cf_legs)
                ]

                open_books[variant][opportunity_id] = {
                    "opportunity_id": opportunity_id,
                    "variant": variant,
                    "symbol": pair,
                    "base": base,
                    "S": float(sv["S"]),
                    "S_effective": s_eff,
                    "prev_S": prev_s,
                    "opened_ts": str(t),
                    "entry_fill_ts": str(entry_ts),
                    "leg_items": leg_items,
                    "last_processed_ts": str(entry_ts),
                    "regime": regime_info["regime"],
                }
                sm.register_open(pair, opportunity_id)

                opp_row: dict[str, Any] = {
                    "opportunity_id": opportunity_id,
                    "variant": variant,
                    "base": base,
                    "symbol": pair,
                    "S": float(sv["S"]),
                    "S_effective": s_eff,
                    "prev_S": prev_s,
                    "signal_ts": str(t),
                    "entry_ts": str(entry_ts),
                    "entry_price": entry_mid,
                    "day_number": day_n,
                    "week": week_of(entry_ts, eval_start),
                    "regime": regime_info["regime"],
                    "status": "OPEN",
                    "notional_usd": notional,
                }
                opp_row.update(_factor_fields(diag))
                opp_row.update(_e3_fields(e3 if variant in ("E3", "E5") else None))
                if e3 and variant in ("E3", "E5"):
                    opp_row["e3_conditions"] = json.dumps(e3.get("conditions") or {})
                opp_rows.append(opp_row)

    last_1m = _utc(exit_raw["btc"]["timestamp"].max())
    advance_all(max(eval_end + pd.Timedelta(days=2), last_1m))

    for variant in VARIANTS:
        book = open_books[variant]
        if not book:
            continue
        logger.warning("%s: %d opportunities still open — force-closing", variant, len(book))
        from btcc.sim.exits import _close_leg as force_close_leg

        for oid, opp in list(book.items()):
            coin = exit_panels["coins"].get(opp["base"])
            if coin is None:
                continue
            rel = coin["rel"]
            last = _utc(opp.get("last_processed_ts") or opp["entry_fill_ts"])
            bars = rel[rel["timestamp"] > last]
            if not bars.empty:
                process_bars_until_closed(
                    [item["leg"] for item in opp["leg_items"]],
                    bars,
                    btc_usdt_series=btc_close_1m,
                    default_btc_usdt=float(btc_close_1m.iloc[-1]),
                    costs=costs,
                    same_candle_conflict="assume_sl_first",
                )
            last_bar = rel.iloc[-1]
            exit_mid = float(last_bar["close"])
            exit_ts = _utc(last_bar["timestamp"])
            try:
                btc_px = (
                    float(btc_close_1m.loc[exit_ts])
                    if exit_ts in btc_close_1m.index
                    else float(btc_close_1m.iloc[-1])
                )
            except Exception:
                btc_px = float(btc_close_1m.iloc[-1])
            for item in opp["leg_items"]:
                leg = item["leg"]
                if not leg.closed:
                    force_close_leg(
                        leg,
                        exit_mid=exit_mid,
                        btc_usdt=btc_px,
                        costs=costs,
                        exit_ts=exit_ts,
                        reason="END_OF_DATA",
                    )
            regime = opp.get("regime")
            for item in opp["leg_items"]:
                leg = item["leg"]
                rec = leg_to_record(leg, oid)
                rec["variant"] = variant
                rec["arm_key"] = item["arm_key"]
                rec["is_counterfactual"] = True
                rec["regime"] = regime
                rec["base"] = opp.get("base")
                rec["symbol"] = opp.get("symbol")
                rec["signal_ts"] = opp.get("opened_ts")
                rec["entry_S"] = opp.get("S")
                rec["S_effective"] = opp.get("S_effective")
                rec["prev_S"] = opp.get("prev_S")
                rec["day_number"] = day_number_at(leg.exit_ts, eval_start) if leg.exit_ts is not None else None
                rec["week"] = week_of(leg.exit_ts, eval_start) if leg.exit_ts is not None else None
                leg_rows.append(rec)
            for o in opp_rows:
                if o.get("opportunity_id") == oid:
                    o["status"] = "CLOSED_END_OF_DATA"
            sms[variant].register_close(oid, opp.get("symbol"))
            del book[oid]

    legs_df = pd.DataFrame(leg_rows)
    opp_df = pd.DataFrame(opp_rows)

    # ---- Artifacts ----
    run_meta = {
        "eval_start_utc": str(eval_start),
        "eval_end_utc": str(eval_end),
        "days": DAYS,
        "warmup_bars": WARMUP_BARS,
        "signal_interval": SIGNAL_INTERVAL,
        "exit_interval": EXIT_INTERVAL,
        "universe_bases": bases,
        "variants": list(VARIANTS),
        "long_threshold": LONG_THRESHOLD,
        "max_open": MAX_OPEN,
        "alloc_frac": ALLOC_FRAC,
        "starting_btc": STARTING_BTC,
        "fees": 0.0,
        "slippage": 0.0,
        "selector": "NONE",
        "strategies": "T1-T20 counterfactual per entry",
        "n_15m_bars": n_15m,
        "n_1m_bars": n_1m,
        "e3_thresholds": {
            "E3_MAX_15M_RET": E3_MAX_15M_RET,
            "E3_MAX_PCT_B": E3_MAX_PCT_B,
            "E3_RVOL_MIN": E3_RVOL_MIN,
            "E3_STRONG_BOOST": E3_STRONG_BOOST,
            "E3_MODERATE_BOOST": E3_MODERATE_BOOST,
        },
        "match_window_hours": MATCH_WINDOW_H.total_seconds() / 3600.0,
        "late_lookback_hours": LOOKBACK_H.total_seconds() / 3600.0,
        "false_breakout_mfe_pct": FALSE_BREAKOUT_MFE * 100.0,
        "methodology_notes": [
            "Six independent CrossingStateMachines (BASE, E1-E5) fed S_effective from score_all_variants",
            "Entry at next 15m open; exits on 1m OHLC with T1-T20 legs",
            "No selectors; fees/slippage zero",
            "Research only — live bot and frozen btcc/factors untouched",
        ],
    }
    (OUT_DIR / "run_meta.json").write_text(json.dumps(run_meta, indent=2, default=str), encoding="utf-8")
    opp_df.to_csv(OUT_DIR / "opportunities.csv", index=False)
    legs_df.to_csv(OUT_DIR / "strategy_legs.csv", index=False)

    t1_legs = legs_df[legs_df["arm_key"] == "T1"].copy() if not legs_df.empty else pd.DataFrame()

    # variant_summary (T1 primary)
    summary_rows = []
    for variant in VARIANTS:
        sub = t1_legs[t1_legs["variant"] == variant] if not t1_legs.empty else pd.DataFrame()
        m = strategy_metrics(sub, starting_btc=STARTING_BTC, label=variant)
        fb_rate = None
        if not sub.empty:
            mfe = pd.to_numeric(sub["mfe_pct"], errors="coerce").fillna(0.0)
            fb_rate = round(100.0 * float((mfe < FALSE_BREAKOUT_MFE).mean()), 2)
        m["False Breakout Rate %"] = fb_rate
        summary_rows.append(m)
    variant_summary = pd.DataFrame(summary_rows)
    variant_summary.to_csv(OUT_DIR / "variant_summary.csv", index=False)

    # Full variant x T1-T20 matrix
    matrix_rows = []
    for variant in VARIANTS:
        for arm in LABELS_T20:
            sub = legs_df[(legs_df["variant"] == variant) & (legs_df["arm_key"] == arm)]
            m = strategy_metrics(sub, starting_btc=STARTING_BTC, label=f"{variant}_{arm}")
            m["variant"] = variant
            m["arm_key"] = arm
            matrix_rows.append(m)
    variant_t_exit_matrix = pd.DataFrame(matrix_rows)
    variant_t_exit_matrix.to_csv(OUT_DIR / "variant_t_exit_matrix.csv", index=False)

    # Daily / weekly breakdown on T1
    daily_rows = []
    weekly_rows = []
    if not t1_legs.empty:
        for variant in VARIANTS:
            g = t1_legs[t1_legs["variant"] == variant]
            for w in (1, 2, 3):
                gw = g[g["week"] == w]
                m = strategy_metrics(gw, starting_btc=STARTING_BTC, label=variant)
                weekly_rows.append(
                    {
                        "variant": variant,
                        "week": w,
                        "trades": m["Trades"],
                        "net_btc": m["Net BTC"],
                        "win_pct": m["Win %"],
                        "expectancy": m["Expectancy"],
                        "max_dd": m["Max DD"],
                    }
                )
            if "day_number" in g.columns:
                for d, gd in g.groupby("day_number"):
                    daily_rows.append(
                        {
                            "variant": variant,
                            "day": int(d),
                            "net_btc": float(pd.to_numeric(gd["pnl_btc"], errors="coerce").fillna(0).sum()),
                            "trades": len(gd),
                            "win_pct": strategy_metrics(gd, starting_btc=STARTING_BTC, label=variant)["Win %"],
                        }
                    )
    pd.DataFrame(daily_rows).to_csv(OUT_DIR / "variant_daily_breakdown.csv", index=False)
    pd.DataFrame(weekly_rows).to_csv(OUT_DIR / "variant_weekly_breakdown.csv", index=False)

    # Trade-level comparison (BASE anchor, ±6h match)
    trade_cmp_rows: list[dict[str, Any]] = []
    base_opps = opp_df[opp_df["variant"] == "BASE"].copy() if not opp_df.empty else pd.DataFrame()
    if not base_opps.empty:
        base_opps["entry_ts_dt"] = pd.to_datetime(base_opps["entry_ts"], utc=True)
        for _, brow in base_opps.iterrows():
            base = str(brow["base"])
            base_ts = _utc(brow["entry_ts"])
            base_price = float(brow["entry_price"])
            base_oid = brow["opportunity_id"]
            base_t1 = t1_legs[(t1_legs["variant"] == "BASE") & (t1_legs["opportunity_id"] == base_oid)]
            base_leg = base_t1.iloc[0] if len(base_t1) else None

            row: dict[str, Any] = {
                "base": base,
                "BASE_opportunity_id": base_oid,
                "BASE_signal_ts": brow["signal_ts"],
                "BASE_entry_ts": str(base_ts),
                "BASE_entry_price": base_price,
            }
            if base_leg is not None:
                row["BASE_mfe_pct"] = float(base_leg.get("mfe_pct") or 0)
                row["BASE_mae_pct"] = float(base_leg.get("mae_pct") or 0)
                row["BASE_pnl_pct"] = float(base_leg.get("pnl_pct") or 0)
                for col in (
                    "time_to_p1_min",
                    "time_to_p1_5_min",
                    "time_to_p2_min",
                    "time_to_p3_min",
                ):
                    row[f"BASE_{col}"] = base_leg.get(col)

            coin_rel = exit_panels["coins"].get(base, {}).get("rel")
            if base_leg is not None and coin_rel is not None:
                ms = milestone_times_from_1m(
                    coin_rel,
                    pd.Timestamp(base_leg["entry_ts"]),
                    float(base_leg.get("entry_fill_price") or base_price),
                )
                row["BASE_time_to_0_5pct_min"] = ms["p0_5"]
                row["BASE_time_to_0_75pct_min"] = ms["p0_75"]

            for variant in VARIANTS:
                if variant == "BASE":
                    row["BASE_capture_class"] = "BOTH" if base_leg is not None else "BASE_ONLY"
                    continue
                candidates = opp_df[(opp_df["variant"] == variant) & (opp_df["base"] == base)]
                match = _nearest_match(base_ts, candidates)
                exp_present = match is not None
                row[f"{variant}_capture_class"] = _capture_class(True, exp_present)
                if match is None:
                    row[f"{variant}_entry_ts"] = None
                    row[f"{variant}_entry_price"] = None
                    row[f"{variant}_minutes_earlier"] = None
                    row[f"{variant}_entry_price_diff_pct"] = None
                    continue
                exp_ts = _utc(match["entry_ts"])
                exp_price = float(match["entry_price"])
                row[f"{variant}_entry_ts"] = str(exp_ts)
                row[f"{variant}_entry_price"] = exp_price
                row[f"{variant}_minutes_earlier"] = (base_ts - exp_ts).total_seconds() / 60.0
                if base_price > 0:
                    row[f"{variant}_entry_price_diff_pct"] = 100.0 * (exp_price - base_price) / base_price
                exp_oid = match["opportunity_id"]
                exp_t1 = t1_legs[(t1_legs["variant"] == variant) & (t1_legs["opportunity_id"] == exp_oid)]
                if len(exp_t1):
                    el = exp_t1.iloc[0]
                    row[f"{variant}_mfe_pct"] = float(el.get("mfe_pct") or 0)
                    row[f"{variant}_mae_pct"] = float(el.get("mae_pct") or 0)
                    row[f"{variant}_pnl_pct"] = float(el.get("pnl_pct") or 0)
                    for col in (
                        "time_to_p1_min",
                        "time_to_p1_5_min",
                        "time_to_p2_min",
                        "time_to_p3_min",
                    ):
                        row[f"{variant}_{col}"] = el.get(col)
                    if coin_rel is not None:
                        ms = milestone_times_from_1m(
                            coin_rel,
                            pd.Timestamp(el["entry_ts"]),
                            float(el.get("entry_fill_price") or exp_price),
                        )
                        row[f"{variant}_time_to_0_5pct_min"] = ms["p0_5"]
                        row[f"{variant}_time_to_0_75pct_min"] = ms["p0_75"]

            trade_cmp_rows.append(row)
    trade_level_comparison = pd.DataFrame(trade_cmp_rows)
    trade_level_comparison.to_csv(OUT_DIR / "trade_level_comparison.csv", index=False)

    # Late BASE breakout diagnostics
    late_rows: list[dict[str, Any]] = []
    if not base_opps.empty and not t1_legs.empty:
        base_t1_all = t1_legs[t1_legs["variant"] == "BASE"]
        e3_opps = opp_df[opp_df["variant"] == "E3"] if not opp_df.empty else pd.DataFrame()
        e5_opps = opp_df[opp_df["variant"] == "E5"] if not opp_df.empty else pd.DataFrame()
        for _, brow in base_opps.iterrows():
            base = str(brow["base"])
            signal_ts = _utc(brow["signal_ts"])
            base_oid = brow["opportunity_id"]
            bt1 = base_t1_all[base_t1_all["opportunity_id"] == base_oid]
            if bt1.empty:
                continue
            leg = bt1.iloc[0]
            events = _first_event_times(base, signal_ts, diag_history_by_base.get(base, []))
            event_ts_values = [
                pd.Timestamp(v) for k, v in events.items() if v is not None and k.endswith("_ts")
            ]
            breakout_before = any(
                (signal_ts - ts) >= LATE_MIN_LEAD for ts in event_ts_values if ts <= signal_ts
            )
            e3_match = _nearest_match(_utc(brow["entry_ts"]), e3_opps[e3_opps["base"] == base]) if len(e3_opps) else None
            e5_match = _nearest_match(_utc(brow["entry_ts"]), e5_opps[e5_opps["base"] == base]) if len(e5_opps) else None
            e3_earlier = (
                e3_match is not None
                and _utc(e3_match["entry_ts"]) < _utc(brow["entry_ts"]) - LATE_MIN_LEAD
            )
            e5_earlier = (
                e5_match is not None
                and _utc(e5_match["entry_ts"]) < _utc(brow["entry_ts"]) - LATE_MIN_LEAD
            )
            is_late = bool(breakout_before and (e3_earlier or e5_earlier))
            late_rows.append(
                {
                    "base": base,
                    "BASE_opportunity_id": base_oid,
                    "BASE_signal_ts": str(signal_ts),
                    "BASE_entry_ts": brow["entry_ts"],
                    "BASE_entry_price": brow["entry_price"],
                    "BASE_mfe_pct": float(leg.get("mfe_pct") or 0),
                    "BASE_mae_pct": float(leg.get("mae_pct") or 0),
                    "BASE_pnl_pct": float(leg.get("pnl_pct") or 0),
                    **events,
                    "breakout_started_before_base": breakout_before,
                    "E3_entered_earlier": e3_earlier,
                    "E5_entered_earlier": e5_earlier,
                    "flagged_late_base_breakout": is_late,
                    "E3_entry_ts": str(e3_match["entry_ts"]) if e3_match is not None else None,
                    "E3_entry_price": float(e3_match["entry_price"]) if e3_match is not None else None,
                    "E5_entry_ts": str(e5_match["entry_ts"]) if e5_match is not None else None,
                    "E5_entry_price": float(e5_match["entry_price"]) if e5_match is not None else None,
                }
            )
    late_base_breakout_diagnostics = pd.DataFrame(late_rows)
    late_base_breakout_diagnostics.to_csv(OUT_DIR / "late_base_breakout_diagnostics.csv", index=False)

    # False breakout rates
    fb_rows = []
    for variant in VARIANTS:
        sub = t1_legs[t1_legs["variant"] == variant] if not t1_legs.empty else pd.DataFrame()
        n = len(sub)
        if n == 0:
            fb_rows.append({"variant": variant, "trades": 0, "false_breakouts": 0, "false_breakout_rate_pct": None})
            continue
        mfe = pd.to_numeric(sub["mfe_pct"], errors="coerce").fillna(0.0)
        fb = int((mfe < FALSE_BREAKOUT_MFE).sum())
        fb_rows.append(
            {
                "variant": variant,
                "trades": n,
                "false_breakouts": fb,
                "false_breakout_rate_pct": round(100.0 * fb / n, 2),
                "false_breakout_definition": f"T1 MFE < {FALSE_BREAKOUT_MFE * 100:.1f}%",
            }
        )
    false_breakout_rates = pd.DataFrame(fb_rows)
    false_breakout_rates.to_csv(OUT_DIR / "false_breakout_rates.csv", index=False)

    # Summary report
    def _df_md(df: pd.DataFrame) -> str:
        if df is None or df.empty:
            return "(empty)"
        try:
            return df.to_markdown(index=False)
        except Exception:
            return df.to_string(index=False)

    avg_minutes_earlier: dict[str, float | None] = {}
    if not trade_level_comparison.empty:
        for variant in VARIANTS:
            if variant == "BASE":
                continue
            col = f"{variant}_minutes_earlier"
            cap = f"{variant}_capture_class"
            if col not in trade_level_comparison.columns:
                continue
            both = trade_level_comparison[trade_level_comparison[cap] == "BOTH"]
            vals = pd.to_numeric(both[col], errors="coerce").dropna()
            avg_minutes_earlier[variant] = float(vals.mean()) if len(vals) else None

    base_row = variant_summary[variant_summary["Strategy"] == "BASE"].iloc[0] if len(variant_summary) else None
    ranked = variant_summary.sort_values("Net BTC", ascending=False)

    report_lines = [
        "# Early Breakout Entry Experiment (21d, 15m / 1m)",
        "",
        "## Window",
        f"- Start (UTC): `{eval_start}`",
        f"- End (UTC): `{eval_end}`",
        f"- Variants: {', '.join(VARIANTS)}",
        "",
        "## T1 headline comparison (vs BASE)",
        "",
        _df_md(
            variant_summary[
                [
                    "Strategy",
                    "Trades",
                    "Win %",
                    "Net BTC",
                    "Profit Factor",
                    "Max DD",
                    "Expectancy",
                    "False Breakout Rate %",
                ]
            ]
        ),
        "",
        "## Rankings (T1 Net BTC)",
        "",
        *[f"- {row['Strategy']}: {row['Net BTC']}" for _, row in ranked.iterrows()],
        "",
        "## False breakout rates",
        "",
        _df_md(false_breakout_rates),
        "",
        "## Matched BOTH trades — avg minutes earlier vs BASE",
        "",
    ]
    for v, m in avg_minutes_earlier.items():
        report_lines.append(f"- {v}: {m}")
    report_lines += [
        "",
        "## Late BASE breakout flags",
        f"- Total BASE T1 trades: {len(late_base_breakout_diagnostics)}",
        f"- Flagged late (breakout before BASE + E3/E5 earlier): "
        f"{int(late_base_breakout_diagnostics['flagged_late_base_breakout'].sum()) if not late_base_breakout_diagnostics.empty else 0}",
        "",
        "## Notes",
        "- Primary comparison uses T1 legs; full T1–T20 matrix in `variant_t_exit_matrix.csv`.",
        "- Research only — do not apply to live bot without further validation.",
        "",
        f"Artifacts: `{OUT_DIR}`",
    ]
    (OUT_DIR / "summary_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    logger.info("Wrote results → %s", OUT_DIR)
    if not variant_summary.empty:
        print(
            variant_summary[
                ["Strategy", "Trades", "Win %", "Net BTC", "Expectancy", "Max DD", "False Breakout Rate %"]
            ].to_string(index=False)
        )
    print(f"\nReport: {OUT_DIR / 'summary_report.md'}")
    return OUT_DIR


if __name__ == "__main__":
    run()
