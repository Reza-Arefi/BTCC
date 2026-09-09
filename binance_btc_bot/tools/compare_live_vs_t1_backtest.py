"""Run a short T1 backtest over the live window and diff vs live SQLite trades.

Uses research trail engine with live-aligned knobs:
  strategy T1 only, S>=0.65, no upper band, max 8, one-per-pair,
  15m signals/entries + 1m OHLC exits (closer stop/trail path),
  Binance Vision candles, fee/slippage 0 (match live recorded PnL).

Does not place orders or touch the live process.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
BRT = ZoneInfo("America/Sao_Paulo")
STARTING_EQUITY_BTC = 0.00783355
LIVE_DB = REPO / "data" / "binance_btc_bot" / "bot.sqlite3"
OUT_DIR = REPO / "results" / "live_vs_t1_backtest_1m"
CANDLE_DIR = REPO / "data" / "backtest_candles_binance"
SIGNAL_INTERVAL = "15m"
EXIT_INTERVAL = "1m"

# Live window (BRT first entry → last scored exit), padded in UTC.
EVAL_START = "2026-09-06T19:00:00Z"  # ~16:00 BRT
EVAL_END = "2026-09-08T18:15:00Z"  # after last exit 14:58 BRT
WARMUP_BARS = 1000

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("live_vs_t1")


def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def live_universe_bases() -> list[str]:
    cfg = yaml.safe_load((REPO / "binance_btc_bot" / "config" / "binance_bot.yaml").read_text(encoding="utf-8"))
    pairs = list((cfg.get("universe") or {}).get("btc_pairs") or [])
    bases = []
    for p in pairs:
        s = str(p).upper()
        if s.endswith("BTC"):
            bases.append(s[:-3])
        else:
            bases.append(s)
    return bases


def build_sim() -> dict[str, Any]:
    from btcc.sim.config import load_sim_config

    sim = dict(load_sim_config())
    sim.update(
        {
            "experiment_kind": "live_window_t1_compare",
            "long_threshold": 0.65,
            "upper_threshold": None,
            "starting_capital_usd": 612.0,
            "notional_usd": 76.5,  # ~12.5% of ~0.00783 BTC at ~$78k
            "compound_portfolio": False,
            "max_open_opportunities": 8,
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
            "strategies": {
                "trail_1": {
                    "name": "T1",
                    "stop_loss_pct": 0.0075,
                    "take_profit_pct": None,
                    "trailing": {"activation_pct": 0.0075, "distance_pct": 0.0025},
                }
            },
        }
    )
    return sim


def download_binance_candles(bases: list[str]) -> None:
    from btcc.backtest.data_loader import compute_window
    from btcc.data.binance_vision import download_universe

    data_start, _, _ = compute_window(
        days=3,
        warmup_bars=WARMUP_BARS,
        interval=SIGNAL_INTERVAL,
        eval_start=EVAL_START,
        eval_end=EVAL_END,
    )
    # Pad a bit for API / zip edges.
    start_15 = _utc(data_start) - pd.Timedelta(days=1)
    end = _utc(EVAL_END) + pd.Timedelta(hours=6)
    symbols = ["BTCUSDT"] + [f"{b}USDT" for b in bases]
    CANDLE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Downloading Binance Vision %s candles %s → %s (%d symbols)",
        SIGNAL_INTERVAL,
        start_15,
        end,
        len(symbols),
    )
    report = download_universe(
        symbols, start=start_15, end=end, interval=SIGNAL_INTERVAL, candle_dir=CANDLE_DIR
    )
    ok = sum(1 for v in report.values() if v.get("ok"))
    logger.info("%s download done: %d/%d ok", SIGNAL_INTERVAL, ok, len(report))

    # 1m only needed across the live/eval window for exit path resolution.
    start_1m = _utc(EVAL_START) - pd.Timedelta(hours=2)
    logger.info(
        "Downloading Binance Vision %s candles %s → %s (%d symbols)",
        EXIT_INTERVAL,
        start_1m,
        end,
        len(symbols),
    )
    report1 = download_universe(
        symbols, start=start_1m, end=end, interval=EXIT_INTERVAL, candle_dir=CANDLE_DIR
    )
    ok1 = sum(1 for v in report1.values() if v.get("ok"))
    logger.info("%s download done: %d/%d ok", EXIT_INTERVAL, ok1, len(report1))
    bad = {k: v for k, v in report1.items() if not v.get("ok")}
    if bad:
        logger.warning("Missing/failed 1m symbols: %s", sorted(bad)[:20])


def _load_exit_panels(bases: list[str], candle_dir: Path) -> dict[str, Any]:
    """Build 1m relative ALT/BTC panels for exit simulation."""
    from btcc.data.candles import load_candles, candle_path
    from btcc.series.relative import build_alt_btc

    btc = load_candles(candle_path(candle_dir, "BTCUSDT", EXIT_INTERVAL))
    if btc is None or btc.empty:
        raise RuntimeError("Missing BTCUSDT 1m candles for exit panels")
    btc = btc.copy()
    btc["timestamp"] = pd.to_datetime(btc["timestamp"], utc=True)
    coins: dict[str, Any] = {}
    unavailable: list[str] = []
    for base in bases:
        alt = load_candles(candle_path(candle_dir, f"{base}USDT", EXIT_INTERVAL))
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
        coins[base] = {
            "base": base,
            "symbol": f"{base}USDT",
            "rel": rel,
            "alt_for_volume": alt,
        }
    logger.info("1m exit panels: coins=%d unavailable=%s", len(coins), unavailable[:10])
    return {
        "btc": btc,
        "coins": coins,
        "unavailable": unavailable,
        "btc_close": btc.set_index("timestamp")["close"],
    }


def run_t1_backtest(bases: list[str], sim: dict[str, Any]) -> pd.DataFrame:
    from btcc.backtest.config import load_backtest_config
    from btcc.backtest.data_loader import download_panels
    from btcc.backtest.dominance_history import HistoricalDominanceSeries
    from btcc.backtest.predict import predict_coin_at_bar
    from btcc.sim.accounting import CostModel
    from btcc.sim.backtest import _advance_book
    from btcc.sim.day_axis import day_number_at
    from btcc.sim.exits import open_opportunity_legs, specs_from_config
    from btcc.sim.health import evaluate_health
    from btcc.sim.regime import classify_regime
    from btcc.sim.score import combined_score, extract_factor_scores, static_factor_weights
    from btcc.sim.state_machine import CrossingStateMachine
    from btcc.sim.trail_entry import evaluate_trail_entry

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
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    panels = download_panels(
        cfg,
        days=3,
        warmup_bars=WARMUP_BARS,
        force=False,
        eval_start=EVAL_START,
        eval_end=EVAL_END,
        offline_candles=True,
    )
    exit_panels = _load_exit_panels(bases, CANDLE_DIR)
    # Use 1m relative series for stop/trail path; keep 15m panels for signals/entry.
    exit_book_panels = {
        "coins": {
            base: {
                **coin,
                "rel": exit_panels["coins"][base]["rel"],
            }
            for base, coin in panels["coins"].items()
            if base in exit_panels["coins"]
        }
    }
    btc_df = panels["btc"].copy()
    btc_df["timestamp"] = pd.to_datetime(btc_df["timestamp"], utc=True)
    eval_start = _utc(panels["window"]["eval_start"])
    eval_end = _utc(panels["window"]["eval_end"])
    btc_close_1m = exit_panels["btc_close"]
    logger.info(
        "Panels 15m: coins=%d unavailable=%s | exits=%s window=%s→%s",
        len(panels["coins"]),
        panels.get("unavailable"),
        EXIT_INTERVAL,
        eval_start,
        eval_end,
    )

    dom_series = HistoricalDominanceSeries.fetch_for_backtest(
        days=14,
        cache_dir=bt_cfg["backtest_data"]["dominance_cache"],
        sim_cfg=sim,
        force=False,
    )
    weights = static_factor_weights(cfg)
    specs = specs_from_config(sim)
    costs = CostModel(fee_rate_per_side=0.0, slippage_rate_per_side=0.0)
    sm = CrossingStateMachine(
        long_threshold=0.65,
        upper_threshold=None,
        max_open=8,
        one_per_pair=True,
        threshold_strict=False,
    )

    ts_all = pd.to_datetime(btc_df["timestamp"], utc=True)
    decision_indices = [
        i
        for i in btc_df.index.tolist()
        if i >= 100
        and i + 1 < len(btc_df)
        and ts_all.iloc[i] >= eval_start
        and ts_all.iloc[i] <= eval_end
    ]
    logger.info("Decision bars (%s): %d", SIGNAL_INTERVAL, len(decision_indices))

    open_book: dict[str, dict[str, Any]] = {}
    leg_rows: list[dict] = []
    opp_rows: list[dict] = []
    own_cache: dict = {}
    max_simultaneous = 0
    max_open_rejects = 0
    interval = SIGNAL_INTERVAL

    for n_done, i in enumerate(decision_indices):
        t = _utc(btc_df.iloc[i]["timestamp"])
        if n_done and n_done % 50 == 0:
            logger.info("Progress %d/%d @ %s open=%d", n_done, len(decision_indices), t, sm.n_open())

        _advance_book(open_book, sm, exit_book_panels, btc_close_1m, costs, sim, t, leg_rows)
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
            if base not in exit_book_panels["coins"]:
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
                row = predict_coin_at_bar(rel_hist, alt_vol_hist, btc_hist, dom_pct, dom_changes, cfg, interval)
                if row is None:
                    continue
                factor_scores = extract_factor_scores(row["factors"])
                factors_raw = row.get("factors") or {}
                own_cache[cache_key] = {
                    "factor_scores": dict(factor_scores),
                    "factors_raw": factors_raw,
                }

            scored = combined_score(factor_scores, weights)
            s_val = float(scored["S"])
            pair = coin["symbol"]
            decision = sm.evaluate(pair, s_val)
            ep = evaluate_trail_entry(
                sm_decision=decision,
                health_allow_new_trades=bool(health.allow_new_trades),
            )
            if ep["rejection_reason"] == "MAX_OPEN_TRADES":
                max_open_rejects += 1
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
            legs = open_opportunity_legs(
                alt_btc_entry_mid=entry_mid,
                btc_usdt=btc_usdt,
                notional_usd=float(sim["notional_usd"]),
                costs=costs,
                specs=specs,
                entry_ts=entry_ts,
            )
            regime_info = classify_regime(factors_raw, rules={})
            open_book[opportunity_id] = {
                "opportunity_id": opportunity_id,
                "symbol": pair,
                "base": base,
                "logical_pair": base,
                "resolved_market": coin.get("resolved_market", pair),
                "S": s_val,
                "opened_ts": str(t),
                "entry_fill_ts": str(entry_ts),
                "entry_alt_btc_mid": entry_mid,
                "legs": legs,
                "last_processed_ts": str(entry_ts),
                "regime": regime_info["regime"],
                "entry_classification": ep["entry_classification"],
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
                    "day_number": day_number_at(t, eval_start),
                }
            )

    # Force-close remaining at eval_end on last available bar mid.
    _advance_book(open_book, sm, exit_book_panels, btc_close_1m, costs, sim, eval_end, leg_rows)
    still_open = list(open_book.keys())
    if still_open:
        logger.warning("Force-marking %d open opportunities at eval_end (leaving open)", len(still_open))

    legs_df = pd.DataFrame(leg_rows)
    if not legs_df.empty:
        legs_df["entry_ts"] = pd.to_datetime(legs_df["entry_ts"], utc=True)
        legs_df["exit_ts"] = pd.to_datetime(legs_df["exit_ts"], utc=True)
        # Attach base from opportunities
        oid_base = {o["opportunity_id"]: o["base"] for o in opp_rows}
        legs_df["base"] = legs_df["opportunity_id"].map(oid_base)
        legs_df["symbol_btc"] = legs_df["base"].map(lambda b: f"{b}BTC" if pd.notna(b) else None)

    meta = {
        "eval_start": str(eval_start),
        "eval_end": str(eval_end),
        "signal_interval": SIGNAL_INTERVAL,
        "exit_interval": EXIT_INTERVAL,
        "n_decision_bars": len(decision_indices),
        "n_coins": len(panels["coins"]),
        "unavailable": panels.get("unavailable"),
        "n_opportunities": len(opp_rows),
        "n_closed_legs": int(len(legs_df)),
        "max_simultaneous": max_simultaneous,
        "max_open_rejects": max_open_rejects,
        "starting_equity_btc_ref": STARTING_EQUITY_BTC,
        "notional_usd": sim["notional_usd"],
        "fee_rate_per_side": 0.0,
        "notes": [
            "15m signals/entries; 1m OHLC for stop/trail exits.",
            "Same-candle trail activation + hard SL => STOP_LOSS.",
            "Research bar-trail exits (not Binance-native OCO).",
            "Fee/slippage set to 0 to align with live recorded price PnL.",
        ],
    }
    (OUT_DIR / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    pd.DataFrame(opp_rows).to_csv(OUT_DIR / "opportunities.csv", index=False)
    legs_df.to_csv(OUT_DIR / "strategy_legs.csv", index=False)
    return legs_df


def load_live_trades() -> pd.DataFrame:
    con = sqlite3.connect(LIVE_DB)
    df = pd.read_sql_query(
        """
        SELECT trade_id, symbol, strategy, entry_time, exit_time,
               entry_price, exit_price, btc_value,
               realized_pnl_btc, realized_pnl_btc_equivalent
        FROM trades WHERE upper(status)='CLOSED'
        ORDER BY exit_time, entry_time
        """,
        con,
    )
    con.close()
    if df.empty:
        return df
    df["pnl_btc"] = pd.to_numeric(df["realized_pnl_btc"], errors="coerce")
    alt = pd.to_numeric(df["realized_pnl_btc_equivalent"], errors="coerce")
    df["pnl_btc"] = df["pnl_btc"].fillna(alt)
    df = df.dropna(subset=["pnl_btc"]).copy()
    df["entry_dt"] = pd.to_datetime(df["entry_time"], utc=True, errors="coerce")
    df["exit_dt"] = pd.to_datetime(df["exit_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["exit_dt"]).sort_values("exit_dt").reset_index(drop=True)
    df["base"] = df["symbol"].astype(str).str.replace(r"BTC$", "", regex=True)
    df["is_win"] = df["pnl_btc"] > 0
    df["is_loss"] = df["pnl_btc"] < 0
    df["pnl_pct_equity"] = 100.0 * df["pnl_btc"] / STARTING_EQUITY_BTC
    entry_btc = pd.to_numeric(df["btc_value"], errors="coerce")
    df["pnl_pct_trade"] = 100.0 * df["pnl_btc"] / entry_btc.replace(0, pd.NA)
    return df


def summarize_side(df: pd.DataFrame, pnl_col: str, label: str) -> dict[str, Any]:
    if df is None or df.empty:
        return {"label": label, "n": 0}
    wins = df[df[pnl_col] > 0]
    losses = df[df[pnl_col] < 0]
    flats = df[df[pnl_col] == 0]
    n_dec = len(wins) + len(losses)
    return {
        "label": label,
        "n": int(len(df)),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "flats": int(len(flats)),
        "win_rate_pct": round(100.0 * len(wins) / n_dec, 2) if n_dec else None,
        "avg_gain": float(wins[pnl_col].mean()) if len(wins) else None,
        "avg_loss": float(losses[pnl_col].mean()) if len(losses) else None,
        "avg_pnl": float(df[pnl_col].mean()),
        "net": float(df[pnl_col].sum()),
        "sum_wins": float(wins[pnl_col].sum()) if len(wins) else 0.0,
        "sum_losses": float(losses[pnl_col].sum()) if len(losses) else 0.0,
    }


def match_trades(live: pd.DataFrame, bt: pd.DataFrame) -> dict[str, Any]:
    """Match by base + nearest entry within ±45 minutes."""
    if live.empty or bt.empty:
        return {"matched": [], "live_only": list(live.index), "bt_only": list(bt.index)}

    bt = bt.copy()
    bt["entry_ts"] = pd.to_datetime(bt["entry_ts"], utc=True)
    used = set()
    matched = []
    live_only = []
    for i, lr in live.iterrows():
        best_j = None
        best_dt = None
        for j, br in bt.iterrows():
            if j in used:
                continue
            if str(br.get("base") or "").upper() != str(lr["base"]).upper():
                continue
            if pd.isna(lr["entry_dt"]) or pd.isna(br["entry_ts"]):
                continue
            dt = abs((lr["entry_dt"] - br["entry_ts"]).total_seconds())
            if dt <= 45 * 60 and (best_dt is None or dt < best_dt):
                best_dt = dt
                best_j = j
        if best_j is None:
            live_only.append(int(i) if isinstance(i, (int,)) else i)
            continue
        used.add(best_j)
        br = bt.loc[best_j]
        matched.append(
            {
                "base": lr["base"],
                "live_symbol": lr["symbol"],
                "live_entry": str(lr["entry_dt"]),
                "bt_entry": str(br["entry_ts"]),
                "entry_delta_sec": int(best_dt),
                "live_exit": str(lr["exit_dt"]),
                "bt_exit": str(br.get("exit_ts")),
                "live_pnl_btc": float(lr["pnl_btc"]),
                "bt_pnl_btc": float(br.get("pnl_btc") or 0.0),
                "live_pnl_pct_trade": float(lr["pnl_pct_trade"]) if pd.notna(lr.get("pnl_pct_trade")) else None,
                "bt_pnl_pct": float(br["pnl_pct"]) * 100.0 if pd.notna(br.get("pnl_pct")) else None,
                "same_sign": (float(lr["pnl_btc"]) > 0) == (float(br.get("pnl_btc") or 0) > 0)
                if float(lr["pnl_btc"]) != 0 and float(br.get("pnl_btc") or 0) != 0
                else float(lr["pnl_btc"]) == float(br.get("pnl_btc") or 0),
            }
        )
    bt_only = [int(j) if isinstance(j, (int,)) else j for j in bt.index if j not in used]
    return {"matched": matched, "live_only": live_only, "bt_only": bt_only}


def main() -> None:
    bases = live_universe_bases()
    sim = build_sim()
    download_binance_candles(bases)
    legs = run_t1_backtest(bases, sim)
    live = load_live_trades()

    # Backtest pnl columns from leg_to_record
    if not legs.empty:
        if "pnl_btc" not in legs.columns and "pnl_btc" not in legs:
            # leg_to_record usually has pnl_btc / pnl_pct
            pass
        legs["pnl_btc"] = pd.to_numeric(legs.get("pnl_btc"), errors="coerce").fillna(0.0)
        legs["pnl_pct"] = pd.to_numeric(legs.get("pnl_pct"), errors="coerce").fillna(0.0)
        # Scale BT notional PnL to live starting equity % for headline compare
        # Each live trade ~12.5% equity; BT notional fixed $76.5 — use trade % for avg W/L.
        legs["pnl_pct_trade"] = legs["pnl_pct"] * 100.0
        # Approximate equity % if each trade is 12.5% of book: equity_pct ≈ 0.125 * trade_pct
        legs["pnl_pct_equity_approx"] = 0.125 * legs["pnl_pct_trade"]

    live_sum = summarize_side(live, "pnl_btc", "LIVE_BTC")
    live_eq = summarize_side(live, "pnl_pct_equity", "LIVE_%EQUITY")
    bt_sum = summarize_side(legs, "pnl_btc", "BT_BTC") if not legs.empty else {"label": "BT_BTC", "n": 0}
    bt_tr = summarize_side(legs, "pnl_pct_trade", "BT_%TRADE") if not legs.empty else {"label": "BT_%TRADE", "n": 0}
    bt_eq = (
        summarize_side(legs, "pnl_pct_equity_approx", "BT_%EQUITY_approx")
        if not legs.empty
        else {"label": "BT_%EQUITY_approx", "n": 0}
    )

    matching = match_trades(live, legs if not legs.empty else pd.DataFrame())
    matched = matching["matched"]
    same_sign = sum(1 for m in matched if m.get("same_sign"))
    cmp = {
        "live": live_sum,
        "live_pct_equity": live_eq,
        "backtest": bt_sum,
        "backtest_pct_trade": bt_tr,
        "backtest_pct_equity_approx": bt_eq,
        "match": {
            "n_matched": len(matched),
            "n_live_only": len(matching["live_only"]),
            "n_bt_only": len(matching["bt_only"]),
            "same_sign_among_matched": same_sign,
            "same_sign_rate_pct": round(100.0 * same_sign / len(matched), 1) if matched else None,
        },
        "live_first_entry_brt": str(live["entry_dt"].iloc[0].tz_convert(BRT)) if len(live) else None,
        "live_last_exit_brt": str(live["exit_dt"].iloc[-1].tz_convert(BRT)) if len(live) else None,
        "bt_first_entry": str(legs["entry_ts"].min()) if not legs.empty else None,
        "bt_last_exit": str(legs["exit_ts"].max()) if not legs.empty else None,
    }
    (OUT_DIR / "comparison.json").write_text(json.dumps(cmp, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(matched).to_csv(OUT_DIR / "matched_trades.csv", index=False)

    print("\n=== LIVE vs T1 BACKTEST ===")
    print(json.dumps(cmp, indent=2, default=str))
    print(f"\nWrote artifacts to {OUT_DIR}")


if __name__ == "__main__":
    main()
