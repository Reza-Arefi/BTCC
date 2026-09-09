"""21-day Binance multi-arm backtest: T1–T10 + selectors A–F.

Signals/entries: 15m (S >= 0.65, live universe).
Exits: 1m OHLC trail sim (incl. same-candle trail-activation + hard SL => SL).

Saves results + plots under results/multiarm_1m_21d_*/.
Does not place orders or touch the live process.
"""

from __future__ import annotations

import json
import logging
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
STARTING_CAPITAL_USD = 1000.0
NOTIONAL_USD = 125.0  # 12.5% of starting capital
MAX_OPEN = 8

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("multiarm_1m_21d")


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


def t1_t10_strategies() -> dict[str, Any]:
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


def build_sim() -> dict[str, Any]:
    from btcc.sim.config import load_sim_config

    sim = dict(load_sim_config())
    strategies = t1_t10_strategies()
    sim.update(
        {
            "experiment_kind": "binance_multiarm_1m_21d",
            "long_threshold": 0.65,
            "upper_threshold": None,
            "starting_capital_usd": STARTING_CAPITAL_USD,
            "notional_usd": NOTIONAL_USD,
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
            "selector_experiment": {
                "long_threshold": 0.65,
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
            },
        }
    )
    return sim


def download_candles(bases: list[str], eval_start: pd.Timestamp, eval_end: pd.Timestamp) -> None:
    """Download 15m (signal) + 1m (exit) candles. Prefer API for 1m (Vision zips often timeout)."""
    from btcc.backtest.data_loader import compute_window
    from btcc.data.binance_vision import download_universe, fetch_klines_api, _session, _as_utc
    from btcc.data.candles import candle_path, load_candles, save_candles

    data_start, _, _ = compute_window(
        days=DAYS,
        warmup_bars=WARMUP_BARS,
        interval=SIGNAL_INTERVAL,
        eval_start=eval_start,
        eval_end=eval_end,
    )
    start_15 = _utc(data_start) - pd.Timedelta(days=1)
    end = _utc(eval_end) + pd.Timedelta(hours=6)
    symbols = ["BTCUSDT"] + [f"{b}USDT" for b in bases]
    CANDLE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s %s → %s (%d symbols)", SIGNAL_INTERVAL, start_15, end, len(symbols))
    r15 = download_universe(symbols, start=start_15, end=end, interval=SIGNAL_INTERVAL, candle_dir=CANDLE_DIR)
    logger.info("15m ok=%d/%d", sum(1 for v in r15.values() if v.get("ok")), len(r15))

    start_1m = _utc(eval_start) - pd.Timedelta(hours=2)
    logger.info("Downloading %s via API %s → %s (%d symbols)", EXIT_INTERVAL, start_1m, end, len(symbols))
    sess = _session()
    ok = 0
    for i, sym in enumerate(symbols, 1):
        path = candle_path(CANDLE_DIR, sym, EXIT_INTERVAL)
        frames = []
        cached = load_candles(path)
        if cached is not None and not cached.empty:
            frames.append(cached)
        # Skip Vision monthly zips for 1m (slow/timeouts); API fill the window.
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
    logger.info("1m exit panels coins=%d unavailable=%s", len(coins), unavailable[:12])
    return {"btc": btc, "coins": coins, "btc_close": btc.set_index("timestamp")["close"], "unavailable": unavailable}


def build_daily_cap(legs: pd.DataFrame, eval_start: pd.Timestamp) -> pd.DataFrame:
    """Per-arm daily equity path from closed legs (fixed notional book)."""
    if legs.empty:
        return pd.DataFrame()
    g = legs.copy()
    g["exit_ts"] = pd.to_datetime(g["exit_ts"], utc=True)
    g["pnl_usd"] = pd.to_numeric(g.get("pnl_usd_equiv"), errors="coerce")
    if g["pnl_usd"].isna().all():
        # Fallback: notional * pnl_pct
        g["pnl_usd"] = NOTIONAL_USD * pd.to_numeric(g["pnl_pct"], errors="coerce").fillna(0.0)
    else:
        g["pnl_usd"] = g["pnl_usd"].fillna(0.0)
    from btcc.sim.day_axis import day_number_at

    g["day_number"] = g["exit_ts"].map(lambda t: day_number_at(t, eval_start))
    rows = []
    max_day = int(g["day_number"].max())
    for arm, ag in g.groupby("arm_key"):
        by_day = ag.groupby("day_number")["pnl_usd"].sum()
        equity = STARTING_CAPITAL_USD
        for d in range(1, max_day + 1):
            pnl = float(by_day.get(d, 0.0))
            start_eq = equity
            equity = start_eq + pnl
            rows.append(
                {
                    "strategy_key": arm,
                    "day_number": d,
                    "daily_pnl_usd": pnl,
                    "ending_value": equity,
                    "cumulative_return_pct": 100.0 * (equity / STARTING_CAPITAL_USD - 1.0),
                }
            )
    return pd.DataFrame(rows)


def arm_summary(legs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm, g in legs.groupby("arm_key"):
        pnl = pd.to_numeric(g["pnl_pct"], errors="coerce").fillna(0.0)
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        n = len(pnl)
        n_dec = len(wins) + len(losses)
        usd = pd.to_numeric(g.get("pnl_usd_equiv"), errors="coerce")
        if usd.isna().all():
            usd = NOTIONAL_USD * pnl
        else:
            usd = usd.fillna(0.0)
        net_usd = float(usd.sum())
        rows.append(
            {
                "arm": arm,
                "trades": n,
                "wins": int(len(wins)),
                "losses": int(len(losses)),
                "win_rate_pct": round(100.0 * len(wins) / n_dec, 2) if n_dec else None,
                "avg_pnl_pct": float(pnl.mean() * 100.0),
                "avg_win_pct": float(wins.mean() * 100.0) if len(wins) else None,
                "avg_loss_pct": float(losses.mean() * 100.0) if len(losses) else None,
                "net_pnl_usd": net_usd,
                "net_return_pct_capital": 100.0 * net_usd / STARTING_CAPITAL_USD,
                "sum_win_pct": float(wins.sum() * 100.0) if len(wins) else 0.0,
                "sum_loss_pct": float(losses.sum() * 100.0) if len(losses) else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("arm").reset_index(drop=True)


def plot_summary_bars(summary: pd.DataFrame, out_dir: Path) -> None:
    arms = summary["arm"].tolist()
    x = np.arange(len(arms))

    def _bar(col: str, title: str, fname: str, ylabel: str) -> None:
        fig, ax = plt.subplots(figsize=(12, 4.5))
        vals = summary[col].astype(float).fillna(0.0)
        colors = ["#2e8b57" if v >= 0 else "#d62728" for v in vals]
        ax.bar(x, vals, color=colors, width=0.7)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(arms, rotation=45, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=130)
        plt.close(fig)

    _bar("net_return_pct_capital", "Net return (% of starting capital) — 21d 1m exits", "net_return_by_arm.png", "Return %")
    _bar("win_rate_pct", "Win rate by arm — 21d 1m exits", "win_rate_by_arm.png", "Win rate %")
    _bar("avg_pnl_pct", "Average P/L per trade (%) — 21d 1m exits", "avg_pnl_by_arm.png", "Avg trade %")
    _bar("trades", "Trade count by arm — 21d 1m exits", "trade_count_by_arm.png", "Trades")


def run() -> Path:
    from btcc.backtest.config import load_backtest_config
    from btcc.backtest.data_loader import download_panels
    from btcc.backtest.dominance_history import HistoricalDominanceSeries
    from btcc.backtest.predict import predict_coin_at_bar
    from btcc.sim.accounting import CostModel
    from btcc.sim.day_axis import day_number_at
    from btcc.sim.exits import open_opportunity_legs, specs_from_config
    from btcc.sim.health import evaluate_health
    from btcc.sim.regime import classify_regime
    from btcc.sim.score import combined_score, extract_factor_scores, static_factor_weights
    from btcc.sim.selector_config import _label_for_key, active_fixed_arm_labels, active_fixed_strategy_keys
    from btcc.sim.selector_engine import (
        CounterfactualHistory,
        build_selector_group,
        oracle_best_counterfactual,
    )
    from btcc.sim.state_machine import CrossingStateMachine
    from btcc.sim.trail_entry import evaluate_trail_entry
    from btcc.sim.exits import leg_to_record, process_bars_until_closed
    from btcc.analytics.selector_plots import generate_selector_plots

    eval_end = _utc(datetime.now(timezone.utc)).floor("15min")
    eval_start = eval_end - pd.Timedelta(days=DAYS)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = REPO / "results" / f"multiarm_1m_21d_{run_id}"
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

    fixed_keys = active_fixed_strategy_keys(sim)
    fixed_labels = active_fixed_arm_labels(sim)
    spec_by_key = {s.key: s for s in specs_from_config(sim)}
    fixed_specs = [spec_by_key[k] for k in fixed_keys]
    selectors = build_selector_group(sim)
    cf_history = CounterfactualHistory()
    history_recorded: set[tuple[str, str]] = set()

    weights = static_factor_weights(cfg)
    costs = CostModel(0.0, 0.0)
    sm = CrossingStateMachine(0.65, None, MAX_OPEN, True, False)
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
    logger.info(
        "Window %s → %s | decisions=%d | coins=%d | arms=%s + %s",
        eval_start,
        eval_end,
        len(decision_indices),
        len(panels["coins"]),
        fixed_labels,
        list("ABCDEF"),
    )

    open_book: dict[str, dict[str, Any]] = {}
    leg_rows: list[dict] = []
    opp_rows: list[dict] = []
    selection_rows: list[dict] = []
    own_cache: dict = {}
    max_simultaneous = 0

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
                    rec["regime"] = regime
                    rec["base"] = opp.get("base")
                    rec["symbol"] = opp.get("symbol")
                    rec["day_number"] = day_number_at(leg.exit_ts, eval_start) if leg.exit_ts is not None else None
                    if item.get("is_counterfactual"):
                        cf_pnls[leg.spec.key] = float((leg.exit_result or {}).get("pnl_pct") or 0.0)
                    leg_rows.append(rec)
                best_k, best_v = oracle_best_counterfactual(cf_pnls)
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
            opportunity_id = f"opp_{uuid.uuid4().hex[:12]}"
            regime_info = classify_regime(factors_raw, rules={})

            cf_legs = open_opportunity_legs(
                alt_btc_entry_mid=entry_mid,
                btc_usdt=btc_usdt,
                notional_usd=NOTIONAL_USD,
                costs=costs,
                specs=fixed_specs,
                entry_ts=entry_ts,
            )
            leg_items: list[dict[str, Any]] = []
            for arm_label, leg in zip(fixed_labels, cf_legs):
                leg_items.append(
                    {"leg": leg, "arm_key": arm_label, "is_counterfactual": True, "selector_id": None}
                )

            for sel in selectors.values():
                pick = sel.select(cf_history, entry_ts, regime_info["regime"], strategy_keys=fixed_keys)
                sk = pick["selected_strategy_key"]
                sel_legs = open_opportunity_legs(
                    alt_btc_entry_mid=entry_mid,
                    btc_usdt=btc_usdt,
                    notional_usd=NOTIONAL_USD,
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
                    }
                )
                selection_rows.append(
                    {
                        "opportunity_id": opportunity_id,
                        "entry_ts": str(entry_ts),
                        "day_number": day_n,
                        "selector_id": sel.selector_id,
                        "arm_label": sel.arm_label,
                        "regime": regime_info["regime"],
                        "selected_strategy_key": sk,
                        "selected_arm_label": pick["selected_arm_label"],
                        "selected_score": pick["selected_score"],
                        "switched": pick["switched"],
                    }
                )

            open_book[opportunity_id] = {
                "opportunity_id": opportunity_id,
                "symbol": pair,
                "base": base,
                "S": s_val,
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
                    "opened_ts": str(t),
                    "entry_fill_ts": str(entry_ts),
                    "entry_alt_btc_mid": entry_mid,
                    "day_number": day_n,
                    "regime": regime_info["regime"],
                    "status": "OPEN",
                }
            )

    advance(eval_end)
    if open_book:
        logger.warning("%d opportunities still open at eval_end", len(open_book))

    legs_df = pd.DataFrame(leg_rows)
    opp_df = pd.DataFrame(opp_rows)
    sel_df = pd.DataFrame(selection_rows)
    if not legs_df.empty:
        legs_df["entry_ts"] = pd.to_datetime(legs_df["entry_ts"], utc=True)
        legs_df["exit_ts"] = pd.to_datetime(legs_df["exit_ts"], utc=True)

    # Regret vs oracle for selector arms
    regret_rows = []
    if not legs_df.empty and not opp_df.empty:
        oracle_map = {
            str(r["opportunity_id"]): float(r["oracle_pnl_pct"])
            for _, r in opp_df.iterrows()
            if r.get("oracle_pnl_pct") is not None and pd.notna(r.get("oracle_pnl_pct"))
        }
        for _, r in legs_df[~legs_df["is_counterfactual"].astype(bool)].iterrows():
            oid = str(r["opportunity_id"])
            if oid not in oracle_map:
                continue
            regret_rows.append(
                {
                    "opportunity_id": oid,
                    "arm_label": r["arm_key"],
                    "pnl_pct": float(r["pnl_pct"] or 0),
                    "oracle_pnl_pct": oracle_map[oid],
                    "regret_pct": oracle_map[oid] - float(r["pnl_pct"] or 0),
                }
            )
    regret_df = pd.DataFrame(regret_rows)

    summary = arm_summary(legs_df) if not legs_df.empty else pd.DataFrame()
    daily_cap = build_daily_cap(legs_df, eval_start) if not legs_df.empty else pd.DataFrame()
    max_day = int(daily_cap["day_number"].max()) if not daily_cap.empty else DAYS

    meta = {
        "eval_start": str(eval_start),
        "eval_end": str(eval_end),
        "days": DAYS,
        "signal_interval": SIGNAL_INTERVAL,
        "exit_interval": EXIT_INTERVAL,
        "long_threshold": 0.65,
        "max_open": MAX_OPEN,
        "notional_usd": NOTIONAL_USD,
        "starting_capital_usd": STARTING_CAPITAL_USD,
        "fixed_arms": list(fixed_labels),
        "selector_arms": list("ABCDEF"),
        "n_decision_bars": len(decision_indices),
        "n_opportunities": len(opp_df),
        "n_closed_legs": int(len(legs_df)),
        "max_simultaneous": max_simultaneous,
        "n_coins": len(panels["coins"]),
        "fee_rate_per_side": 0.0,
        "notes": [
            "15m signals; 1m exits",
            "T1-T10 geometry from binance_bot.yaml",
            "Same-candle trail activation + hard SL => STOP_LOSS",
            "Selectors A-F choose among T1-T10 using counterfactual history",
        ],
    }

    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    legs_df.to_csv(out_dir / "strategy_legs.csv", index=False)
    opp_df.to_csv(out_dir / "opportunities.csv", index=False)
    sel_df.to_csv(out_dir / "selection_audit.csv", index=False)
    regret_df.to_csv(out_dir / "selection_regret.csv", index=False)
    summary.to_csv(out_dir / "arm_summary.csv", index=False)
    daily_cap.to_csv(out_dir / "daily_capital.csv", index=False)
    (out_dir / "arm_summary.json").write_text(
        summary.to_json(orient="records", indent=2), encoding="utf-8"
    )

    if not legs_df.empty:
        generate_selector_plots(
            legs_df,
            daily_cap=daily_cap,
            trade_cap=pd.DataFrame(),
            out_dir=plots_dir,
            max_day=max_day,
            starting_capital_usd=STARTING_CAPITAL_USD,
            opportunities=opp_df,
            selection=sel_df if not sel_df.empty else None,
            regret=regret_df if not regret_df.empty else None,
        )
        plot_summary_bars(summary, plots_dir)

    logger.info("Wrote results → %s", out_dir)
    if not summary.empty:
        print(summary.to_string(index=False))
    print(f"\nPlots: {plots_dir}")
    return out_dir


if __name__ == "__main__":
    run()
