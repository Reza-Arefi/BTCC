"""21-day EXIT-ONLY experiment: T1–T20 + selectors A–F (T1–T12 and T1–T20).

Same entry list for every arm. Signals 15m, exits 1m.
Research only — does not touch live bot / orders / LIVE config.
"""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
CANDLE_DIR = REPO / "data" / "backtest_candles_binance"
SIGNAL_INTERVAL = "15m"
EXIT_INTERVAL = "1m"
WARMUP_BARS = 1000
DAYS = 21
STARTING_BTC = 0.00783355
ALLOC_FRAC = 0.125
MAX_OPEN = 8
LONG_THRESHOLD = 0.65

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("t_strategy_21d_1m")


def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def live_universe_bases() -> list[str]:
    cfg = yaml.safe_load((REPO / "binance_btc_bot" / "config" / "binance_bot.yaml").read_text(encoding="utf-8"))
    pairs = list((cfg.get("universe") or {}).get("btc_pairs") or [])
    out = []
    for p in pairs:
        s = str(p).upper()
        out.append(s[:-3] if s.endswith("BTC") else s)
    return out


def t1_t10_from_live() -> dict[str, Any]:
    cfg = yaml.safe_load((REPO / "binance_btc_bot" / "config" / "binance_bot.yaml").read_text(encoding="utf-8"))
    raw = cfg.get("strategies") or {}
    out: dict[str, Any] = {}
    for i in range(1, 11):
        name = f"T{i}"
        p = raw[name]
        out[f"trail_{i}"] = {
            "name": name,
            "stop_loss_pct": float(p["arm_sl_activation_trail"]),
            "take_profit_pct": None,
            "trailing": {
                "activation_pct": float(p["activation"]),
                "distance_pct": float(p["trail_distance"]),
            },
        }
    return out


def t11_t20_research() -> dict[str, Any]:
    """Research-only exit geometries T11–T20 (do not replace live T1–T10)."""
    return {
        "trail_11": {
            "name": "T11",
            "stop_loss_pct": 0.0075,
            "take_profit_pct": None,
            "trailing": {"activation_pct": 0.0075, "distance_pct": 0.005},
            "adaptive_mode": "fixed",
        },
        "trail_12": {
            "name": "T12",
            "stop_loss_pct": 0.0075,
            "take_profit_pct": None,
            "trailing": {"activation_pct": 0.0075, "distance_pct": 0.0075},
            "adaptive_mode": "fixed",
        },
        "trail_13": {
            "name": "T13",
            "stop_loss_pct": 0.0075,
            "take_profit_pct": None,
            "trailing": {"activation_pct": 0.0075, "distance_pct": 0.0075},
            "adaptive_mode": "two_stage",
            "adaptive_cfg": {
                "trail_initial": 0.0075,
                "trail_at_1_5pct": 0.005,
                "trail_at_2pct": 0.0025,
            },
        },
        "trail_14": {
            "name": "T14",
            "stop_loss_pct": 0.0075,
            "take_profit_pct": None,
            "trailing": {"activation_pct": 0.0075, "distance_pct": 0.0075},
            "adaptive_mode": "profit_lock",
            "adaptive_cfg": {
                "trail_initial": 0.0075,
                "floor_at_1_5pct": 0.0025,
                "floor_at_2pct": 0.0075,
            },
        },
        "trail_15": {
            "name": "T15",
            "stop_loss_pct": 0.0075,
            "take_profit_pct": None,
            "trailing": {"activation_pct": 0.0075, "distance_pct": 0.0075},
            "adaptive_mode": "progressive",
        },
        "trail_16": {
            "name": "T16",
            "stop_loss_pct": 0.0075,
            "take_profit_pct": None,
            "trailing": {"activation_pct": 0.01, "distance_pct": 0.005},
            "adaptive_mode": "fixed",
        },
        "trail_17": {
            "name": "T17",
            "stop_loss_pct": 0.0075,
            "take_profit_pct": None,
            "trailing": {"activation_pct": 0.0125, "distance_pct": 0.0075},
            "adaptive_mode": "fixed",
        },
        "trail_18": {
            "name": "T18",
            "stop_loss_pct": 0.0075,
            "take_profit_pct": None,
            "trailing": {"activation_pct": 0.0075, "distance_pct": 0.0075},
            "adaptive_mode": "break_even_then_tight",
            "adaptive_cfg": {"trail_after_be": 0.0075},
        },
        "trail_19": {
            "name": "T19",
            "stop_loss_pct": 0.0075,
            "take_profit_pct": None,
            "trailing": {"activation_pct": 0.0075, "distance_pct": 0.0075},
            "adaptive_mode": "mfe_responsive",
        },
        "trail_20": {
            "name": "T20",
            "stop_loss_pct": 0.0075,
            "take_profit_pct": None,
            "trailing": {"activation_pct": 0.0075, "distance_pct": 0.0075},
            "adaptive_mode": "time_adaptive",
        },
    }


def selector_block() -> dict[str, Any]:
    return {
        "long_threshold": LONG_THRESHOLD,
        "upper_threshold": None,
        "allow_threshold_override": True,
        "disable_late_entry_rejection": True,
        "switching": {
            "minimum_selection_duration_hours": 6,
            "switch_margin": 0.0005,
        },
        "selectors": {
            "A": {"kind": "ewma_7d", "half_life_days": 7},
            "B": {"kind": "multi_horizon_ewma"},
            "C": {"kind": "regime_conditional", "half_life_days": 7, "min_regime_observations": 5},
            "D": {
                "kind": "recent_plus_regime",
                "half_life_days": 7,
                "min_regime_observations": 5,
                "recent_weight": 0.65,
                "regime_weight": 0.35,
            },
            "E": {"kind": "rank_ewma", "half_life_days": 7},
            "F": {"kind": "downside_aware", "half_life_days": 7, "downside_lambda": 0.5},
        },
    }


def build_sim() -> dict[str, Any]:
    from btcc.sim.config import load_sim_config

    strategies = {**t1_t10_from_live(), **t11_t20_research()}
    sim = dict(load_sim_config())
    sim.update(
        {
            "experiment_kind": "t_strategy_21d_1m_exit_only",
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
            "selector_experiment": selector_block(),
        }
    )
    return sim


def download_candles(bases: list[str], eval_start: pd.Timestamp, eval_end: pd.Timestamp) -> None:
    from btcc.backtest.data_loader import compute_window
    from btcc.data.binance_vision import download_universe, fetch_klines_api, _session, _as_utc
    from btcc.data.candles import candle_path, load_candles, save_candles

    data_start, _, _ = compute_window(
        days=DAYS, warmup_bars=WARMUP_BARS, interval=SIGNAL_INTERVAL, eval_start=eval_start, eval_end=eval_end
    )
    start_15 = _utc(data_start) - pd.Timedelta(days=1)
    end = _utc(eval_end) + pd.Timedelta(hours=6)
    symbols = ["BTCUSDT"] + [f"{b}USDT" for b in bases]
    CANDLE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s %s → %s (%d symbols)", SIGNAL_INTERVAL, start_15, end, len(symbols))
    r15 = download_universe(symbols, start=start_15, end=end, interval=SIGNAL_INTERVAL, candle_dir=CANDLE_DIR)
    logger.info("15m ok=%d/%d", sum(1 for v in r15.values() if v.get("ok")), len(r15))

    start_1m = _utc(eval_start) - pd.Timedelta(hours=2)
    logger.info("Downloading %s via API %s → %s", EXIT_INTERVAL, start_1m, end)
    sess = _session()
    ok = 0
    for i, sym in enumerate(symbols, 1):
        path = candle_path(CANDLE_DIR, sym, EXIT_INTERVAL)
        frames = []
        cached = load_candles(path)
        if cached is not None and not cached.empty:
            frames.append(cached)
        need_api = True
        if cached is not None and not cached.empty:
            w = cached[(cached["timestamp"] >= _as_utc(start_1m)) & (cached["timestamp"] <= _as_utc(end))]
            expected = max(1, int((_as_utc(end) - _as_utc(start_1m)).total_seconds() / 60))
            if len(w) >= expected * 0.90:
                need_api = False
        if need_api:
            for attempt in range(3):
                try:
                    api_df = fetch_klines_api(sym, EXIT_INTERVAL, start_1m, end, session=sess, sleep_s=0.03)
                    if not api_df.empty:
                        frames.append(api_df)
                    break
                except Exception as e:
                    logger.warning("%s 1m API attempt %d failed: %s", sym, attempt + 1, e)
                    time.sleep(1.5 * (attempt + 1))
        if not frames:
            logger.error("No 1m data for %s", sym)
            continue
        out = (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates("timestamp", keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        save_candles(out, path)
        ok += 1
        if i % 5 == 0 or i == len(symbols):
            logger.info("1m progress %d/%d (last=%s n=%d)", i, len(symbols), sym, len(out))
    logger.info("1m ok=%d/%d", ok, len(symbols))


def load_exit_panels(bases: list[str]) -> dict[str, Any]:
    from btcc.data.candles import candle_path, load_candles
    from btcc.series.relative import build_alt_btc

    btc = load_candles(candle_path(CANDLE_DIR, "BTCUSDT", EXIT_INTERVAL))
    if btc is None or btc.empty:
        raise RuntimeError("Missing BTCUSDT 1m")
    btc = btc.copy()
    btc["timestamp"] = pd.to_datetime(btc["timestamp"], utc=True)
    coins: dict[str, Any] = {}
    unavailable: list[str] = []
    for base in bases:
        alt = load_candles(candle_path(CANDLE_DIR, f"{base}USDT", EXIT_INTERVAL))
        if alt is None or alt.empty:
            unavailable.append(base)
            continue
        alt = alt.copy()
        alt["timestamp"] = pd.to_datetime(alt["timestamp"], utc=True)
        rel = build_alt_btc(alt, btc)
        if rel is None or rel.empty:
            unavailable.append(base)
            continue
        rel = rel.copy()
        rel["timestamp"] = pd.to_datetime(rel["timestamp"], utc=True)
        coins[base] = {"base": base, "symbol": f"{base}USDT", "rel": rel}
    return {"btc": btc, "coins": coins, "btc_close": btc.set_index("timestamp")["close"], "unavailable": unavailable}


def _drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = (equity - peak) / peak.replace(0, np.nan)
    return float(dd.min()) if len(dd) else 0.0


def strategy_metrics(legs: pd.DataFrame, *, starting_btc: float, label: str) -> dict[str, Any]:
    if legs.empty:
        return {
            "Strategy": label,
            "Trades": 0,
            "Wins": 0,
            "Losses": 0,
            "Flat": 0,
            "Win %": None,
            "Gross BTC": 0.0,
            "Net BTC": 0.0,
            "Avg Trade BTC": None,
            "Expectancy": None,
            "Avg Win": None,
            "Avg Loss": None,
            "Profit Factor": None,
            "Max DD": None,
            "Final Equity": starting_btc,
            "Return %": 0.0,
            "Avg Hold": None,
            "Median Hold": None,
            "Longest Hold": None,
            "Trail exits": 0,
            "SL exits": 0,
            "Same-candle SL": 0,
            "Trail %": None,
            "SL %": None,
            "Fallback %": None,
            "Fallback Count": 0,
            "Strategy Switch Count": 0,
        }
    g = legs.copy()
    pnl = pd.to_numeric(g["pnl_btc"], errors="coerce").fillna(0.0)
    pnl_pct = pd.to_numeric(g["pnl_pct"], errors="coerce").fillna(0.0)
    wins = pnl[pnl > 1e-12]
    losses = pnl[pnl < -1e-12]
    flats = pnl[(pnl >= -1e-12) & (pnl <= 1e-12)]
    n = len(pnl)
    n_dec = len(wins) + len(losses)
    gross = float(pnl[pnl > 0].sum())
    net = float(pnl.sum())
    avg_win = float(wins.mean()) if len(wins) else None
    avg_loss = float(losses.mean()) if len(losses) else None
    expect = float(pnl_pct.mean()) if n else None
    pf = None
    if losses.sum() < 0:
        pf = float(wins.sum() / abs(losses.sum())) if abs(losses.sum()) > 0 else None
    elif len(wins):
        pf = float("inf")
    hold = pd.to_numeric(g.get("holding_hours"), errors="coerce")
    reasons = g["exit_reason"].astype(str)
    trail_n = int(reasons.str.contains("TRAILING_STOP", na=False).sum())
    same_n = int(reasons.str.contains("STOP_LOSS_TRAIL_ACTIVATION_SAME_CANDLE", na=False).sum())
    sl_n = int(reasons.str.contains("STOP_LOSS", na=False).sum()) - same_n
    # equity path by exit order
    ordered = g.sort_values("exit_ts")
    eq = starting_btc + pd.to_numeric(ordered["pnl_btc"], errors="coerce").fillna(0.0).cumsum()
    max_dd = _drawdown(eq)
    final_eq = float(eq.iloc[-1]) if len(eq) else starting_btc
    return {
        "Strategy": label,
        "Trades": n,
        "Wins": int(len(wins)),
        "Losses": int(len(losses)),
        "Flat": int(len(flats)),
        "Win %": round(100.0 * len(wins) / n_dec, 2) if n_dec else None,
        "Gross BTC": gross,
        "Net BTC": net,
        "Avg Trade BTC": float(pnl.mean()) if n else None,
        "Expectancy": expect,
        "Avg Win": avg_win,
        "Avg Loss": avg_loss,
        "Profit Factor": pf,
        "Max DD": max_dd,
        "Final Equity": final_eq,
        "Return %": 100.0 * (final_eq / starting_btc - 1.0) if starting_btc else None,
        "Avg Hold": float(hold.mean()) if hold.notna().any() else None,
        "Median Hold": float(hold.median()) if hold.notna().any() else None,
        "Longest Hold": float(hold.max()) if hold.notna().any() else None,
        "Trail exits": trail_n,
        "SL exits": sl_n,
        "Same-candle SL": same_n,
        "Trail %": round(100.0 * trail_n / n, 2) if n else None,
        "SL %": round(100.0 * (sl_n + same_n) / n, 2) if n else None,
        "Fallback %": None,
        "Fallback Count": 0,
        "Strategy Switch Count": 0,
    }


def week_of(ts: pd.Timestamp, eval_start: pd.Timestamp) -> int:
    days = (_utc(ts) - _utc(eval_start)).total_seconds() / 86400.0
    if days < 7:
        return 1
    if days < 14:
        return 2
    return 3


def factor_diag(factors_raw: dict) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not factors_raw:
        return out
    # Best-effort: pull common research factor fields if present
    for k in (
        "rsi",
        "RSI",
        "rsi_14",
        "bb_pct_b",
        "bb_percent_b",
        "percent_b",
        "bb_upper",
        "bb_mid",
        "bb_lower",
        "close",
        "btc_regime",
        "regime",
    ):
        if k in factors_raw:
            out[k] = factors_raw[k]
    # nested
    for nest in ("rsi", "bollinger", "bb", "bands"):
        v = factors_raw.get(nest)
        if isinstance(v, dict):
            out.update({f"{nest}.{kk}": vv for kk, vv in v.items()})
    return out


def run() -> Path:
    from btcc.backtest.config import load_backtest_config
    from btcc.backtest.data_loader import download_panels
    from btcc.backtest.dominance_history import HistoricalDominanceSeries
    from btcc.backtest.predict import predict_coin_at_bar
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
    from btcc.sim.score import combined_score, extract_factor_scores, static_factor_weights
    from btcc.sim.selector_config import _label_for_key
    from btcc.sim.selector_engine import (
        CounterfactualHistory,
        SelectorState,
        build_selector_group,
        oracle_best_counterfactual,
    )
    from btcc.sim.state_machine import CrossingStateMachine
    from btcc.sim.trail_entry import evaluate_trail_entry

    eval_end = _utc(datetime.now(timezone.utc)).floor("15min")
    eval_start = eval_end - pd.Timedelta(days=DAYS)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = REPO / "results" / f"t_strategy_21d_1m_{run_id}"
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

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
    keys_t12 = tuple(f"trail_{i}" for i in range(1, 13))
    labels_t20 = tuple(f"T{i}" for i in range(1, 21))
    spec_by_key = {s.key: s for s in specs_from_config(sim)}
    fixed_specs = [spec_by_key[k] for k in keys_t20]

    selectors_t20 = build_selector_group(sim)
    selectors_t12 = build_selector_group(sim)
    # Distinct state objects for T12 universe
    for sid, sel in list(selectors_t12.items()):
        selectors_t12[sid] = SelectorState(
            selector_id=sel.selector_id + "_t12",
            arm_label=sel.arm_label + "_T12",
            kind=sel.kind,
            cfg=dict(sel.cfg),
            min_duration_hours=sel.min_duration_hours,
            switch_margin=sel.switch_margin,
        )
    for sid, sel in list(selectors_t20.items()):
        selectors_t20[sid] = SelectorState(
            selector_id=sel.selector_id + "_t20",
            arm_label=sel.arm_label + "_T20",
            kind=sel.kind,
            cfg=dict(sel.cfg),
            min_duration_hours=sel.min_duration_hours,
            switch_margin=sel.switch_margin,
        )

    cf_history = CounterfactualHistory()
    history_recorded: set[tuple[str, str]] = set()
    weights = static_factor_weights(cfg)
    costs = CostModel(0.0, 0.0)
    sm = CrossingStateMachine(LONG_THRESHOLD, None, MAX_OPEN, True, False)
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
        "Window %s → %s | decisions=%d | 15m=%d | 1m≈%d | coins=%d",
        eval_start,
        eval_end,
        len(decision_indices),
        n_15m,
        n_1m,
        len(panels["coins"]),
    )

    open_book: dict[str, dict[str, Any]] = {}
    leg_rows: list[dict] = []
    opp_rows: list[dict] = []
    selection_rows: list[dict] = []
    entry_diag_rows: list[dict] = []
    own_cache: dict = {}
    prev_s_by_pair: dict[str, float] = {}
    max_simultaneous = 0
    fallback_counts: dict[str, int] = {}

    def notional_usd_at(entry_ts: pd.Timestamp) -> float:
        px = btc_close_1m
        try:
            # last available BTCUSDT close at/before entry
            hist = px[px.index <= entry_ts]
            btc_px = float(hist.iloc[-1]) if len(hist) else float(px.iloc[0])
        except Exception:
            btc_px = 100000.0
        return float(STARTING_BTC * ALLOC_FRAC * btc_px)

    def advance(asof_t: pd.Timestamp) -> None:
        nonlocal max_simultaneous
        done = []
        for oid, opp in open_book.items():
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
                    rec["universe"] = item.get("universe")
                    rec["regime"] = regime
                    rec["base"] = opp.get("base")
                    rec["symbol"] = opp.get("symbol")
                    rec["signal_ts"] = opp.get("opened_ts")
                    rec["entry_S"] = opp.get("S")
                    rec["prev_S"] = opp.get("prev_S")
                    rec["day_number"] = day_number_at(leg.exit_ts, eval_start) if leg.exit_ts is not None else None
                    rec["week"] = week_of(leg.exit_ts, eval_start) if leg.exit_ts is not None else None
                    if item.get("is_counterfactual"):
                        cf_pnls[leg.spec.key] = float((leg.exit_result or {}).get("pnl_pct") or 0.0)
                    leg_rows.append(rec)
                best_k, best_v = oracle_best_counterfactual(cf_pnls)
                # Attach oracle synthetic row using chosen CF leg PnL in BTC
                oracle_item = next(
                    (it for it in opp["leg_items"] if it.get("is_counterfactual") and it["leg"].spec.key == best_k),
                    None,
                )
                if oracle_item is not None:
                    orec = leg_to_record(oracle_item["leg"], oid)
                    orec["arm_key"] = "ORACLE"
                    orec["is_counterfactual"] = False
                    orec["selector_id"] = "oracle"
                    orec["universe"] = "T1-T20"
                    orec["regime"] = regime
                    orec["base"] = opp.get("base")
                    orec["symbol"] = opp.get("symbol")
                    orec["signal_ts"] = opp.get("opened_ts")
                    orec["entry_S"] = opp.get("S")
                    orec["prev_S"] = opp.get("prev_S")
                    orec["day_number"] = day_number_at(oracle_item["leg"].exit_ts, eval_start)
                    orec["week"] = week_of(oracle_item["leg"].exit_ts, eval_start)
                    orec["oracle_strategy_key"] = best_k
                    leg_rows.append(orec)
                for o in opp_rows:
                    if o.get("opportunity_id") == oid:
                        o["oracle_strategy_key"] = best_k
                        o["oracle_arm_label"] = _label_for_key(best_k)
                        o["oracle_pnl_pct"] = best_v
                        o["status"] = "CLOSED"
                sm.register_close(oid, opp.get("symbol"))
                done.append(oid)
        for oid in done:
            del open_book[oid]
        max_simultaneous = max(max_simultaneous, sm.n_open())

    for n_done, i in enumerate(decision_indices):
        t = _utc(btc_df.iloc[i]["timestamp"])
        if n_done and n_done % 100 == 0:
            logger.info(
                "Progress %d/%d @ %s open=%d closed_legs=%d",
                n_done,
                len(decision_indices),
                t,
                sm.n_open(),
                len(leg_rows),
            )
        advance(t)

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
            cache_key = (str(t), coin["symbol"])
            if cache_key in own_cache:
                factor_scores = dict(own_cache[cache_key]["factor_scores"])
                factors_raw = own_cache[cache_key].get("factors_raw") or {}
            else:
                row = predict_coin_at_bar(
                    rel_hist, alt_vol_hist, btc_hist, dom_pct, dom_changes, cfg, SIGNAL_INTERVAL
                )
                if row is None:
                    continue
                factor_scores = extract_factor_scores(row["factors"])
                factors_raw = row.get("factors") or {}
                own_cache[cache_key] = {"factor_scores": dict(factor_scores), "factors_raw": factors_raw}

            s_val = float(combined_score(factor_scores, weights)["S"])
            pair = coin["symbol"]
            prev_s = prev_s_by_pair.get(pair)
            prev_s_by_pair[pair] = s_val
            decision = sm.evaluate(pair, s_val)
            ep = evaluate_trail_entry(sm_decision=decision, health_allow_new_trades=bool(health.allow_new_trades))
            if not ep["trade_suggested"]:
                continue
            future = rel_full[rel_full["timestamp"] > t]
            if future.empty:
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
            opportunity_id = f"opp_{uuid.uuid4().hex[:12]}"
            regime_info = classify_regime(factors_raw, rules={})

            cf_legs = open_opportunity_legs(
                alt_btc_entry_mid=entry_mid,
                btc_usdt=btc_usdt,
                notional_usd=notional,
                costs=costs,
                specs=fixed_specs,
                entry_ts=entry_ts,
            )
            leg_items: list[dict[str, Any]] = []
            for arm_label, leg in zip(labels_t20, cf_legs):
                leg_items.append(
                    {
                        "leg": leg,
                        "arm_key": arm_label,
                        "is_counterfactual": True,
                        "selector_id": None,
                        "universe": "T1-T20",
                    }
                )

            def _select_and_open(sel: SelectorState, keys: tuple[str, ...], universe: str) -> None:
                scores = sel.compute_scores(cf_history, entry_ts, regime_info["regime"], strategy_keys=keys)
                n_hist = sum(len(cf_history.prior(k, entry_ts)) for k in keys)
                is_fallback = n_hist == 0 or (max(scores.values()) - min(scores.values()) < 1e-15)
                pick = sel.select(cf_history, entry_ts, regime_info["regime"], strategy_keys=keys)
                if is_fallback:
                    fallback_counts[sel.arm_label] = fallback_counts.get(sel.arm_label, 0) + 1
                sk = pick["selected_strategy_key"]
                sel_legs = open_opportunity_legs(
                    alt_btc_entry_mid=entry_mid,
                    btc_usdt=btc_usdt,
                    notional_usd=notional,
                    costs=costs,
                    specs=[spec_by_key[sk]],
                    entry_ts=entry_ts,
                )
                leg_items.append(
                    {
                        "leg": sel_legs[0],
                        "arm_key": sel.arm_label,
                        "is_counterfactual": False,
                        "selector_id": sel.selector_id,
                        "universe": universe,
                    }
                )
                selection_rows.append(
                    {
                        "opportunity_id": opportunity_id,
                        "entry_ts": str(entry_ts),
                        "day_number": day_n,
                        "week": week_of(entry_ts, eval_start),
                        "selector_id": sel.selector_id,
                        "arm_label": sel.arm_label,
                        "universe": universe,
                        "regime": regime_info["regime"],
                        "selected_strategy_key": sk,
                        "selected_arm_label": _label_for_key(sk),
                        "selected_score": pick["selected_score"],
                        "switched": pick["switched"],
                        "fallback": is_fallback,
                        "n_history": n_hist,
                    }
                )

            for sel in selectors_t20.values():
                _select_and_open(sel, keys_t20, "T1-T20")
            for sel in selectors_t12.values():
                _select_and_open(sel, keys_t12, "T1-T12")

            open_book[opportunity_id] = {
                "opportunity_id": opportunity_id,
                "symbol": pair,
                "base": base,
                "S": s_val,
                "prev_S": prev_s,
                "opened_ts": str(t),
                "entry_fill_ts": str(entry_ts),
                "leg_items": leg_items,
                "last_processed_ts": str(entry_ts),
                "regime": regime_info["regime"],
            }
            sm.register_open(pair, opportunity_id)
            opp_rows.append(
                {
                    "opportunity_id": opportunity_id,
                    "base": base,
                    "symbol": pair,
                    "S": s_val,
                    "prev_S": prev_s,
                    "opened_ts": str(t),
                    "entry_fill_ts": str(entry_ts),
                    "entry_alt_btc_mid": entry_mid,
                    "day_number": day_n,
                    "week": week_of(entry_ts, eval_start),
                    "regime": regime_info["regime"],
                    "status": "OPEN",
                    "notional_usd": notional,
                }
            )
            diag = {
                "opportunity_id": opportunity_id,
                "symbol": pair,
                "base": base,
                "signal_ts": str(t),
                "entry_ts": str(entry_ts),
                "entry_price": entry_mid,
                "entry_S": s_val,
                "previous_S": prev_s,
                "regime": regime_info["regime"],
            }
            diag.update(factor_diag(factors_raw if isinstance(factors_raw, dict) else {}))
            entry_diag_rows.append(diag)

    last_1m = _utc(exit_raw["btc"]["timestamp"].max())
    advance(max(eval_end + pd.Timedelta(days=2), last_1m))
    if open_book:
        logger.warning("%d opportunities still open — force-closing at last 1m bar", len(open_book))
        from btcc.sim.exits import _close_leg as force_close_leg

        for oid, opp in list(open_book.items()):
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
                opp["last_processed_ts"] = str(bars.iloc[-1]["timestamp"])
            last_bar = rel.iloc[-1]
            exit_mid = float(last_bar["close"])
            exit_ts = _utc(last_bar["timestamp"])
            try:
                btc_px = float(btc_close_1m.loc[exit_ts]) if exit_ts in btc_close_1m.index else float(btc_close_1m.iloc[-1])
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
            cf_pnls: dict[str, float] = {}
            for item in opp["leg_items"]:
                leg = item["leg"]
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
                rec = leg_to_record(leg, oid)
                rec["arm_key"] = item["arm_key"]
                rec["is_counterfactual"] = bool(item.get("is_counterfactual"))
                rec["selector_id"] = item.get("selector_id")
                rec["universe"] = item.get("universe")
                rec["regime"] = regime
                rec["base"] = opp.get("base")
                rec["symbol"] = opp.get("symbol")
                rec["signal_ts"] = opp.get("opened_ts")
                rec["entry_S"] = opp.get("S")
                rec["prev_S"] = opp.get("prev_S")
                rec["day_number"] = day_number_at(leg.exit_ts, eval_start) if leg.exit_ts is not None else None
                rec["week"] = week_of(leg.exit_ts, eval_start) if leg.exit_ts is not None else None
                if item.get("is_counterfactual"):
                    cf_pnls[leg.spec.key] = float((leg.exit_result or {}).get("pnl_pct") or 0.0)
                leg_rows.append(rec)
            best_k, best_v = oracle_best_counterfactual(cf_pnls)
            oracle_item = next(
                (it for it in opp["leg_items"] if it.get("is_counterfactual") and it["leg"].spec.key == best_k),
                None,
            )
            if oracle_item is not None:
                orec = leg_to_record(oracle_item["leg"], oid)
                orec["arm_key"] = "ORACLE"
                orec["is_counterfactual"] = False
                orec["selector_id"] = "oracle"
                orec["universe"] = "T1-T20"
                orec["regime"] = regime
                orec["base"] = opp.get("base")
                orec["symbol"] = opp.get("symbol")
                orec["signal_ts"] = opp.get("opened_ts")
                orec["entry_S"] = opp.get("S")
                orec["prev_S"] = opp.get("prev_S")
                orec["day_number"] = day_number_at(oracle_item["leg"].exit_ts, eval_start)
                orec["week"] = week_of(oracle_item["leg"].exit_ts, eval_start)
                orec["oracle_strategy_key"] = best_k
                leg_rows.append(orec)
            for o in opp_rows:
                if o.get("opportunity_id") == oid:
                    o["oracle_strategy_key"] = best_k
                    o["oracle_arm_label"] = _label_for_key(best_k)
                    o["oracle_pnl_pct"] = best_v
                    o["status"] = "CLOSED_END_OF_DATA"
            sm.register_close(oid, opp.get("symbol"))
            del open_book[oid]

    legs_df = pd.DataFrame(leg_rows)
    opp_df = pd.DataFrame(opp_rows)
    sel_df = pd.DataFrame(selection_rows)
    entry_diag_df = pd.DataFrame(entry_diag_rows)

    n_entries = len(opp_df)
    logger.info("TOTAL UNIQUE ENTRY SIGNALS = %d", n_entries)

    # Verify same entry count across T1–T20
    fixed_legs = legs_df[legs_df["arm_key"].isin(labels_t20)] if not legs_df.empty else pd.DataFrame()
    if not fixed_legs.empty:
        counts = fixed_legs.groupby("arm_key")["opportunity_id"].nunique()
        for lab in labels_t20:
            c = int(counts.get(lab, 0))
            logger.info("%s entries = %d", lab, c)
            if c != n_entries:
                raise RuntimeError(f"Entry mismatch: {lab} has {c} != {n_entries}")
        logger.info("SAME ENTRY SET VERIFIED: all T1–T20 = %d", n_entries)

    # ---- Artifacts ----
    params = {
        "eval_start_utc": str(eval_start),
        "eval_end_utc": str(eval_end),
        "signal_interval": SIGNAL_INTERVAL,
        "exit_interval": EXIT_INTERVAL,
        "long_threshold": LONG_THRESHOLD,
        "max_open": MAX_OPEN,
        "alloc_frac": ALLOC_FRAC,
        "starting_btc": STARTING_BTC,
        "fees": 0.0,
        "slippage": 0.0,
        "n_15m_bars": n_15m,
        "n_1m_bars": n_1m,
        "unique_entries": n_entries,
        "strategies": 20,
        "strategies_detail": sim["strategies"],
        "selector_config": sim["selector_experiment"],
        "notes": [
            "EXIT-ONLY: identical entry list for all strategies",
            "T1–T10 from binance_bot.yaml; T11–T20 research-only",
            "Selectors A–F over T1–T12 and T1–T20; ORACLE is future-info upper bound only",
            "Same-candle trail activation + hard SL => STOP_LOSS_TRAIL_ACTIVATION_SAME_CANDLE",
            "Live bot untouched",
        ],
    }
    (out_dir / "strategy_parameters.json").write_text(json.dumps(params, indent=2, default=str), encoding="utf-8")

    master_rows = []
    for lab in labels_t20:
        sub = fixed_legs[fixed_legs["arm_key"] == lab] if not fixed_legs.empty else pd.DataFrame()
        master_rows.append(strategy_metrics(sub, starting_btc=STARTING_BTC, label=lab))

    # Selectors
    for arm in sorted(legs_df["arm_key"].unique()) if not legs_df.empty else []:
        if arm in labels_t20 or arm == "ORACLE":
            continue
        sub = legs_df[legs_df["arm_key"] == arm]
        m = strategy_metrics(sub, starting_btc=STARTING_BTC, label=arm)
        fb = int(fallback_counts.get(arm, 0))
        m["Fallback Count"] = fb
        m["Fallback %"] = round(100.0 * fb / max(len(sub), 1), 2)
        sw = int(sel_df[sel_df["arm_label"] == arm]["switched"].astype(bool).sum()) if not sel_df.empty else 0
        m["Strategy Switch Count"] = sw
        master_rows.append(m)

    if not legs_df.empty and (legs_df["arm_key"] == "ORACLE").any():
        master_rows.append(
            strategy_metrics(legs_df[legs_df["arm_key"] == "ORACLE"], starting_btc=STARTING_BTC, label="ORACLE")
        )

    master = pd.DataFrame(master_rows)
    # Sort primary: Net BTC, Expectancy, Max DD (less negative better)
    master["_dd_sort"] = pd.to_numeric(master["Max DD"], errors="coerce").fillna(-1)
    master = master.sort_values(
        by=["Net BTC", "Expectancy", "_dd_sort"], ascending=[False, False, False]
    ).drop(columns=["_dd_sort"])
    master.to_csv(out_dir / "master_strategy_comparison.csv", index=False)

    # Trade-level counterfactual wide table
    if not fixed_legs.empty:
        base_cols = ["opportunity_id", "symbol", "entry_ts", "entry_fill_price", "mfe_pct", "mae_pct"]
        wide = fixed_legs[base_cols + ["arm_key", "exit_ts", "exit_fill_price", "exit_reason", "pnl_pct", "pnl_btc"]].copy()
        pieces = []
        for oid, g in wide.groupby("opportunity_id"):
            row = {
                "trade_id": oid,
                "symbol": g["symbol"].iloc[0],
                "entry_timestamp": g["entry_ts"].iloc[0],
                "entry_price": g["entry_fill_price"].iloc[0],
                "MFE": g["mfe_pct"].iloc[0],
                "MAE": g["mae_pct"].iloc[0],
            }
            for _, r in g.iterrows():
                a = r["arm_key"]
                row[f"{a}_exit_ts"] = r["exit_ts"]
                row[f"{a}_exit_price"] = r["exit_fill_price"]
                row[f"{a}_exit_reason"] = r["exit_reason"]
                row[f"{a}_pnl_pct"] = r["pnl_pct"]
                row[f"{a}_pnl_btc"] = r["pnl_btc"]
            pieces.append(row)
        trade_cmp = pd.DataFrame(pieces)
        trade_cmp.to_csv(out_dir / "trade_level_comparison.csv", index=False)
    else:
        trade_cmp = pd.DataFrame()

    # MFE/MAE by trade (use T1 path for shared path stats + all strategies' exits)
    mfe_rows = []
    if not fixed_legs.empty:
        for oid, g in fixed_legs.groupby("opportunity_id"):
            t1 = g[g["arm_key"] == "T1"]
            ref = t1.iloc[0] if len(t1) else g.iloc[0]
            activated = bool(ref.get("trail_activated"))
            mfe = float(ref.get("mfe_pct") or 0)
            row = {
                "trade_id": oid,
                "symbol": ref.get("symbol"),
                "entry_ts": ref.get("entry_ts"),
                "entry_price": ref.get("entry_fill_price"),
                "MFE_pct": mfe,
                "MAE_pct": float(ref.get("mae_pct") or 0),
                "peak_price": ref.get("peak_price"),
                "time_to_peak_min": ref.get("time_to_peak_min"),
                "time_to_activation_min": ref.get("time_to_activation_min"),
                "time_to_1pct_min": ref.get("time_to_p1_min"),
                "time_to_1_5pct_min": ref.get("time_to_p1_5_min"),
                "time_to_2pct_min": ref.get("time_to_p2_min"),
                "time_to_3pct_min": ref.get("time_to_p3_min"),
                "trail_activated": activated,
                "reached_0_75": mfe >= 0.0075,
                "reached_1_0": mfe >= 0.01,
                "reached_1_5": mfe >= 0.015,
                "reached_2_0": mfe >= 0.02,
                "reached_3_0": mfe >= 0.03,
                "T1_exit_price": float(ref.get("exit_fill_price") or 0) if len(t1) else None,
                "T1_exit_reason": ref.get("exit_reason") if len(t1) else None,
                "T1_pnl_pct": float(ref.get("pnl_pct") or 0) if len(t1) else None,
                "leave_on_table_vs_mfe_T1": (mfe - float(ref.get("pnl_pct") or 0)) if len(t1) else None,
            }
            for lab in ("T11", "T12", "T13", "T15", "T18"):
                sub = g[g["arm_key"] == lab]
                if len(sub):
                    row[f"{lab}_pnl_pct"] = float(sub.iloc[0]["pnl_pct"] or 0)
                    row[f"{lab}_exit_reason"] = sub.iloc[0]["exit_reason"]
            mfe_rows.append(row)
    mfe_df = pd.DataFrame(mfe_rows)
    mfe_df.to_csv(out_dir / "mfe_mae_by_trade.csv", index=False)

    entry_diag_df.to_csv(out_dir / "entry_diagnostics.csv", index=False)

    # Weekly / daily breakdowns
    weekly_rows = []
    daily_rows = []
    for lab, g in (legs_df.groupby("arm_key") if not legs_df.empty else []):
        for w in (1, 2, 3):
            gw = g[g["week"] == w]
            m = strategy_metrics(gw, starting_btc=STARTING_BTC, label=lab)
            weekly_rows.append(
                {
                    "Strategy": lab,
                    "Week": w,
                    "Net BTC": m["Net BTC"],
                    "Win %": m["Win %"],
                    "Expectancy": m["Expectancy"],
                    "Max DD": m["Max DD"],
                    "Trades": m["Trades"],
                }
            )
        if "day_number" in g.columns:
            for d, gd in g.groupby("day_number"):
                daily_rows.append(
                    {
                        "Strategy": lab,
                        "Day": int(d),
                        "Net BTC": float(pd.to_numeric(gd["pnl_btc"], errors="coerce").fillna(0).sum()),
                        "Trades": len(gd),
                        "Win %": strategy_metrics(gd, starting_btc=STARTING_BTC, label=lab)["Win %"],
                    }
                )
    weekly_df = pd.DataFrame(weekly_rows)
    daily_df = pd.DataFrame(daily_rows)
    weekly_df.to_csv(out_dir / "strategy_weekly_breakdown.csv", index=False)
    daily_df.to_csv(out_dir / "strategy_daily_breakdown.csv", index=False)

    # Selector comparison (T20 universe primary)
    sel_t20_arms = [f"{c}_T20" for c in "ABCDEF"]
    sel_t12_arms = [f"{c}_T12" for c in "ABCDEF"]
    sel_cmp_rows = []
    for arm in sel_t20_arms + sel_t12_arms:
        sub = legs_df[legs_df["arm_key"] == arm] if not legs_df.empty else pd.DataFrame()
        m = strategy_metrics(sub, starting_btc=STARTING_BTC, label=arm)
        fb = int(fallback_counts.get(arm, 0))
        m["Fallback Count"] = fb
        m["Fallback %"] = round(100.0 * fb / max(len(sub), 1), 2)
        m["Candidate Universe"] = "T1-T20" if arm.endswith("_T20") else "T1-T12"
        m["Selector"] = arm.split("_")[0]
        sw = int(sel_df[sel_df["arm_label"] == arm]["switched"].astype(bool).sum()) if not sel_df.empty else 0
        m["Strategy Switch Count"] = sw
        sel_cmp_rows.append(m)
    sel_cmp = pd.DataFrame(sel_cmp_rows)
    sel_cmp.to_csv(out_dir / "selector_comparison.csv", index=False)

    # Choice distribution
    choice_rows = []
    if not sel_df.empty and not legs_df.empty:
        merged = sel_df.merge(
            legs_df[~legs_df["is_counterfactual"].astype(bool)][
                ["opportunity_id", "arm_key", "pnl_btc", "pnl_pct", "exit_reason"]
            ],
            left_on=["opportunity_id", "arm_label"],
            right_on=["opportunity_id", "arm_key"],
            how="left",
        )
        for (arm, sk), gg in merged.groupby(["arm_label", "selected_arm_label"]):
            pnl = pd.to_numeric(gg["pnl_pct"], errors="coerce").fillna(0)
            wins = pnl[pnl > 0]
            choice_rows.append(
                {
                    "Selector": arm,
                    "Selected": sk,
                    "Trades": len(gg),
                    "Selection %": round(100.0 * len(gg) / max(len(merged[merged["arm_label"] == arm]), 1), 2),
                    "Net BTC": float(pd.to_numeric(gg["pnl_btc"], errors="coerce").fillna(0).sum()),
                    "Win %": round(100.0 * len(wins) / max(len(pnl[pnl != 0]), 1), 2) if len(pnl) else None,
                }
            )
    choice_df = pd.DataFrame(choice_rows)
    choice_df.to_csv(out_dir / "selector_strategy_choices.csv", index=False)

    # Selector weekly
    sel_week_rows = []
    for arm in sel_t20_arms + sel_t12_arms:
        sub = legs_df[legs_df["arm_key"] == arm] if not legs_df.empty else pd.DataFrame()
        for w in (1, 2, 3):
            gw = sub[sub["week"] == w]
            m = strategy_metrics(gw, starting_btc=STARTING_BTC, label=arm)
            # selection distribution this week
            dist = ""
            if not sel_df.empty:
                sw = sel_df[(sel_df["arm_label"] == arm) & (sel_df["week"] == w)]
                if not sw.empty:
                    vc = sw["selected_arm_label"].value_counts()
                    dist = "; ".join(f"{k}:{v}" for k, v in vc.items())
            sel_week_rows.append(
                {
                    "Selector": arm,
                    "Week": w,
                    "Net BTC": m["Net BTC"],
                    "Win %": m["Win %"],
                    "Expectancy": m["Expectancy"],
                    "Max DD": m["Max DD"],
                    "Selection distribution": dist,
                }
            )
    sel_week_df = pd.DataFrame(sel_week_rows)
    sel_week_df.to_csv(out_dir / "selector_weekly_breakdown.csv", index=False)

    # Regret vs oracle
    regret_rows = []
    if not legs_df.empty and not opp_df.empty:
        oracle_map = {
            str(r["opportunity_id"]): float(r["oracle_pnl_pct"])
            for _, r in opp_df.iterrows()
            if r.get("oracle_pnl_pct") is not None and pd.notna(r.get("oracle_pnl_pct"))
        }
        oracle_btc = {}
        for oid, g in legs_df[legs_df["arm_key"] == "ORACLE"].groupby("opportunity_id"):
            oracle_btc[str(oid)] = float(g.iloc[0]["pnl_btc"] or 0)
        for _, r in legs_df[~legs_df["is_counterfactual"].astype(bool)].iterrows():
            if r["arm_key"] == "ORACLE":
                continue
            oid = str(r["opportunity_id"])
            if oid not in oracle_map:
                continue
            regret_rows.append(
                {
                    "opportunity_id": oid,
                    "arm_label": r["arm_key"],
                    "week": r.get("week"),
                    "pnl_pct": float(r["pnl_pct"] or 0),
                    "oracle_pnl_pct": oracle_map[oid],
                    "regret_pct": oracle_map[oid] - float(r["pnl_pct"] or 0),
                    "pnl_btc": float(r["pnl_btc"] or 0),
                    "oracle_pnl_btc": oracle_btc.get(oid),
                    "regret_btc": (oracle_btc.get(oid, 0) - float(r["pnl_btc"] or 0)),
                }
            )
    regret_df = pd.DataFrame(regret_rows)
    regret_df.to_csv(out_dir / "selector_regret.csv", index=False)

    # Selector vs fixed / T12 vs T20
    fixed_master = master[master["Strategy"].isin(labels_t20)].copy()
    best_fixed = fixed_master.iloc[0] if not fixed_master.empty else None
    vs_rows = []
    if best_fixed is not None:
        for arm in sel_t20_arms:
            row = master[master["Strategy"] == arm]
            if row.empty:
                continue
            r = row.iloc[0]
            vs_rows.append(
                {
                    "Comparison": f"{arm} vs best fixed ({best_fixed['Strategy']})",
                    "Selector Net BTC": r["Net BTC"],
                    "Best Fixed Net BTC": best_fixed["Net BTC"],
                    "Delta Net BTC": float(r["Net BTC"]) - float(best_fixed["Net BTC"]),
                    "Selector Expectancy": r["Expectancy"],
                    "Best Fixed Expectancy": best_fixed["Expectancy"],
                    "Selector Max DD": r["Max DD"],
                    "Best Fixed Max DD": best_fixed["Max DD"],
                }
            )
    vs_df = pd.DataFrame(vs_rows)
    vs_df.to_csv(out_dir / "selector_vs_fixed.csv", index=False)

    t12_vs_t20 = []
    for c in "ABCDEF":
        a12 = master[master["Strategy"] == f"{c}_T12"]
        a20 = master[master["Strategy"] == f"{c}_T20"]
        if a12.empty or a20.empty:
            continue
        t12_vs_t20.append(
            {
                "Selector": c,
                "T12 Net BTC": float(a12.iloc[0]["Net BTC"]),
                "T20 Net BTC": float(a20.iloc[0]["Net BTC"]),
                "Delta (T20-T12)": float(a20.iloc[0]["Net BTC"]) - float(a12.iloc[0]["Net BTC"]),
                "T12 Expectancy": a12.iloc[0]["Expectancy"],
                "T20 Expectancy": a20.iloc[0]["Expectancy"],
                "T12 Max DD": a12.iloc[0]["Max DD"],
                "T20 Max DD": a20.iloc[0]["Max DD"],
            }
        )
    t12_vs_df = pd.DataFrame(t12_vs_t20)
    t12_vs_df.to_csv(out_dir / "selector_t1_t12_vs_t1_t20.csv", index=False)

    oracle_df = master[master["Strategy"] == "ORACLE"].copy()
    oracle_df.to_csv(out_dir / "oracle_upper_bound.csv", index=False)

    # Plots
    try:
        plot_df = fixed_master.sort_values("Strategy", key=lambda s: s.map(lambda x: int(x[1:])))
        fig, ax = plt.subplots(figsize=(14, 5))
        vals = plot_df["Return %"].astype(float)
        colors = ["#2e8b57" if v >= 0 else "#d62728" for v in vals]
        ax.bar(plot_df["Strategy"], vals, color=colors)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title("T1–T20 return % (21d, 15m entries / 1m exits)")
        ax.set_ylabel("Return % of starting BTC equity")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(plots_dir / "fixed_return_by_strategy.png", dpi=130)
        plt.close(fig)
    except Exception as e:
        logger.warning("Plot failed: %s", e)

    # Persist raw tables before report generation (report deps optional)
    if not legs_df.empty:
        legs_df.to_csv(out_dir / "all_legs.csv", index=False)
    opp_df.to_csv(out_dir / "opportunities.csv", index=False)
    sel_df.to_csv(out_dir / "selection_audit.csv", index=False)

    # Summary report
    def _rank(df: pd.DataFrame, col: str, ascending: bool = False) -> list[str]:
        s = df.dropna(subset=[col]).sort_values(col, ascending=ascending)
        return [f"{r.Strategy} ({r[col]})" for r in s.itertuples(index=False)]

    fixed_only = master[master["Strategy"].isin(labels_t20)].copy()

    def _df_text(df: pd.DataFrame) -> str:
        if df is None or df.empty:
            return "(empty)"
        try:
            return df.to_markdown(index=False)
        except Exception:
            return df.to_string(index=False)

    report_lines = [
        "# T-Strategy 21-Day Exit Experiment (15m entries / 1m exits)",
        "",
        "## Window",
        f"- Start (UTC): `{eval_start}`",
        f"- End (UTC): `{eval_end}`",
        f"- Days: {DAYS}",
        "",
        "## Data availability",
        f"- 15m signal bars (eval window): **{n_15m}**",
        f"- 1m exit bars (approx BTCUSDT): **{n_1m}**",
        f"- Unique entries (shared): **{n_entries}**",
        f"- Strategies: **20** (T1–T20)",
        f"- Same-entry verification: PASS (all T1–T20 = {n_entries})",
        f"- Max simultaneous: {max_simultaneous}",
        "",
        "## Fees / sizing",
        f"- Fees = 0, slippage = 0",
        f"- Starting BTC equity = {STARTING_BTC}",
        f"- Allocation = {ALLOC_FRAC*100:.1f}% per trade, max open = {MAX_OPEN}",
        "",
        "## Master comparison (sorted Net BTC → Expectancy → Max DD)",
        "",
        _df_text(master),
        "",
        "## Rankings (fixed T1–T20)",
        f"- A Net BTC: {_rank(fixed_only, 'Net BTC')[:5]}",
        f"- B Expectancy: {_rank(fixed_only, 'Expectancy')[:5]}",
        f"- C Profit Factor: {_rank(fixed_only, 'Profit Factor')[:5]}",
        f"- D Max DD (least severe): {_rank(fixed_only, 'Max DD', ascending=False)[:5]}",
        "",
        "## Selector vs best fixed",
        "",
        _df_text(vs_df),
        "",
        "## Expanding universe T1–T12 → T1–T20",
        "",
        _df_text(t12_vs_df),
        "",
        "## Oracle upper bound (future information — NOT for live use)",
        "",
        _df_text(oracle_df),
        "",
    ]

    # Interpretation helpers
    best_net = fixed_only.iloc[0] if not fixed_only.empty else None
    best_exp = fixed_only.sort_values("Expectancy", ascending=False).iloc[0] if not fixed_only.empty else None
    best_dd = fixed_only.sort_values("Max DD", ascending=False).iloc[0] if not fixed_only.empty else None
    t1 = fixed_only[fixed_only["Strategy"] == "T1"]
    t11 = fixed_only[fixed_only["Strategy"] == "T11"]
    t12 = fixed_only[fixed_only["Strategy"] == "T12"]

    # Consistency: sum of ranks across weeks by Net BTC
    consistency = {}
    if not weekly_df.empty:
        for w in (1, 2, 3):
            ww = weekly_df[(weekly_df["Week"] == w) & (weekly_df["Strategy"].isin(labels_t20))]
            if ww.empty:
                continue
            order = ww.sort_values("Net BTC", ascending=False)["Strategy"].tolist()
            for i, s in enumerate(order):
                consistency[s] = consistency.get(s, 0) + i
        most_consistent = sorted(consistency.items(), key=lambda x: x[1])[:5]
    else:
        most_consistent = []

    activated = mfe_df[mfe_df["trail_activated"] == True] if not mfe_df.empty else pd.DataFrame()
    leave = float(activated["leave_on_table_vs_mfe_T1"].mean()) if len(activated) else None

    sel_beats = False
    if best_fixed is not None and not vs_df.empty:
        sel_beats = bool((vs_df["Delta Net BTC"] > 0).any())

    expand_helps = False
    if not t12_vs_df.empty:
        expand_helps = bool((t12_vs_df["Delta (T20-T12)"] > 0).sum() >= 3)

    report_lines += [
        "## Key answers",
        "",
        f"1. Best fixed by Net BTC: **{best_net['Strategy'] if best_net is not None else 'n/a'}** "
        f"({best_net['Net BTC'] if best_net is not None else ''})",
        f"2. Best by Expectancy: **{best_exp['Strategy'] if best_exp is not None else 'n/a'}**",
        f"3. Best by Max DD (least severe): **{best_dd['Strategy'] if best_dd is not None else 'n/a'}**",
        f"4. Most consistent across weeks (rank sum): {most_consistent}",
        f"5. T1 return%: {float(t1.iloc[0]['Return %']) if len(t1) else None}; "
        f"T11: {float(t11.iloc[0]['Return %']) if len(t11) else None}; "
        f"T12: {float(t12.iloc[0]['Return %']) if len(t12) else None}",
        f"6. Among trail-activated trades, mean T1 leave-on-table vs MFE: {leave}",
        f"7. Does any A–F (T1–T20) beat best fixed on Net BTC? **{sel_beats}**",
        f"8. Does expanding selector universe T12→T20 help (≥3 selectors improve)? **{expand_helps}**",
        "",
        "## Interpretation notes",
        "",
        "- This is evidence collection only — **do not change live T1** from this run.",
        "- ORACLE uses future P/L and is an upper bound only.",
        "- Fixed vs adaptive trailing should be judged by Net BTC, expectancy, drawdown, and weekly consistency together.",
        "",
        f"Artifacts directory: `{out_dir}`",
    ]

    (out_dir / "summary_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    logger.info("Wrote results → %s", out_dir)
    print(master[["Strategy", "Trades", "Win %", "Expectancy", "Net BTC", "Return %", "Max DD"]].to_string(index=False))
    print(f"\nReport: {out_dir / 'summary_report.md'}")
    return out_dir


if __name__ == "__main__":
    run()
