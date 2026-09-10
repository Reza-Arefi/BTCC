"""Stage 1 — pick entry first (BASE / E1–E5), with all exit trails T1–T10.

Order (do not reverse):
  1) Entry (this script) — compare entries on a shared T1–T10 counterfactual grid
  2) Exit T1–T10 on winning entry (refine; Stage 2)
  3) Selectors A–F vs best fixed T
  4) Sizing

Setup:
  - Signals: 15m
  - Exits: T1–T10 parallel counterfactual legs on 1s tape (same entries per variant)
  - Window: 90d with walk-forward 60d select / 30d holdout
  - No selectors (Stage 3)

Entry pick rule (select window):
  - Primary: mean Net BTC across T1–T10 (robust to one lucky trail)
  - Also report T1-only ranking (legacy freeze check)
  - Confirm on 30d holdout; do not pick on full 90d

Results → results/stage1_entry_90d_1s_<timestamp>/
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from run_t_strategy_21d_1m_experiment import (  # noqa: E402
    ALLOC_FRAC,
    LONG_THRESHOLD,
    MAX_OPEN,
    STARTING_BTC,
    WARMUP_BARS,
    _utc,
    live_universe_bases,
    strategy_metrics,
    t1_t10_from_live,
)

from btcc.research.early_breakout_entry import score_all_variants  # noqa: E402

CANDLE_DIR = REPO / "data" / "backtest_candles_binance"
SIGNAL_INTERVAL = "15m"
EXIT_INTERVAL = "1s"
DAYS = 90
SELECT_DAYS = 60
HOLDOUT_DAYS = 30
EXIT_LABELS = tuple(f"T{i}" for i in range(1, 11))
VARIANTS = ("BASE", "E1", "E2", "E3", "E4", "E5")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stage1_entry_90d_1s")


def build_sim() -> dict[str, Any]:
    from btcc.sim.config import load_sim_config

    strategies = t1_t10_from_live()  # all T1–T10
    sim = dict(load_sim_config())
    sim.update(
        {
            "experiment_kind": "stage1_entry_BASE_E1E5_x_T1T10_90d_1s",
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


def download_candles(bases: list[str], eval_start: pd.Timestamp, eval_end: pd.Timestamp) -> None:
    from btcc.backtest.data_loader import compute_window
    from btcc.data.binance_vision import download_universe

    data_start, _, _ = compute_window(
        days=DAYS,
        warmup_bars=WARMUP_BARS,
        interval=SIGNAL_INTERVAL,
        eval_start=eval_start,
        eval_end=eval_end,
    )
    start_15 = _utc(data_start) - pd.Timedelta(days=1)
    start_1s = _utc(eval_start) - pd.Timedelta(hours=2)
    end = _utc(eval_end) + pd.Timedelta(hours=6)
    symbols = ["BTCUSDT"] + [f"{b}USDT" for b in bases]
    CANDLE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading %s %s → %s (%d symbols)", SIGNAL_INTERVAL, start_15, end, len(symbols))
    r15 = download_universe(symbols, start=start_15, end=end, interval=SIGNAL_INTERVAL, candle_dir=CANDLE_DIR)
    logger.info("15m ok=%d/%d", sum(1 for v in r15.values() if v.get("ok")), len(r15))

    logger.info("Downloading full-window %s %s → %s", EXIT_INTERVAL, start_1s, end)
    r1s = download_universe(
        symbols,
        start=start_1s,
        end=end,
        interval=EXIT_INTERVAL,
        candle_dir=CANDLE_DIR,
        prefer_vision=True,
        min_coverage=0.90,
    )
    mean_cov = sum(float(v.get("coverage") or 0.0) for v in r1s.values()) / max(1, len(r1s))
    logger.info("1s ok=%d/%d mean_cov=%.1f%%", sum(1 for v in r1s.values() if v.get("ok")), len(r1s), 100 * mean_cov)


def _phase(ts: pd.Timestamp, select_end: pd.Timestamp) -> str:
    return "select" if _utc(ts) <= select_end else "holdout"


def _summarize(legs: pd.DataFrame, label: str) -> dict[str, Any]:
    m = strategy_metrics(legs, starting_btc=STARTING_BTC, label=label)
    # also capital-% style fields for ranking tables
    if not legs.empty and "pnl_usd_equiv" in legs.columns:
        usd = pd.to_numeric(legs["pnl_usd_equiv"], errors="coerce")
        if usd.notna().any():
            net_usd = float(usd.fillna(0.0).sum())
        else:
            net_usd = float(pd.to_numeric(legs["pnl_pct"], errors="coerce").fillna(0.0).sum() * 125.0)
    else:
        net_usd = None
    m["net_usd"] = net_usd
    m["net_return_pct_capital"] = (
        None if net_usd is None else 100.0 * net_usd / 1000.0
    )
    return m


def _plot_bars(summary: pd.DataFrame, col: str, title: str, path: Path, ylabel: str) -> None:
    if summary.empty or col not in summary.columns:
        return
    df = summary.copy()
    # strategy_metrics already has a "Strategy" label; renaming variant→Strategy can duplicate it
    if isinstance(df["Strategy"], pd.DataFrame):
        labels = df["Strategy"].iloc[:, 0]
    else:
        labels = df["Strategy"]
    if isinstance(df[col], pd.DataFrame):
        vals_raw = df[col].iloc[:, 0]
    else:
        vals_raw = df[col]
    arms = labels.astype(str).tolist()
    vals = pd.to_numeric(vals_raw, errors="coerce").fillna(0.0).tolist()
    colors = ["#2e8b57" if v >= 0 else "#d62728" for v in vals]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(np.arange(len(arms)), vals, color=colors, width=0.7)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(np.arange(len(arms)))
    ax.set_xticklabels(arms, rotation=0)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run() -> Path:
    from btcc.backtest.config import load_backtest_config
    from btcc.backtest.data_loader import download_panels
    from btcc.backtest.dominance_history import HistoricalDominanceSeries
    from btcc.data.exit_tape_1s import ExitTape1s
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

    assert SELECT_DAYS + HOLDOUT_DAYS == DAYS

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = REPO / "results" / f"stage1_entry_90d_1s_{run_id}"
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    eval_end = _utc(datetime.now(timezone.utc)).floor("15min")
    eval_start = eval_end - pd.Timedelta(days=DAYS)
    select_end = eval_start + pd.Timedelta(days=SELECT_DAYS) - pd.Timedelta(seconds=1)

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
    exit_tape = ExitTape1s(CANDLE_DIR, bases)

    btc_df = panels["btc"].copy()
    btc_df["timestamp"] = pd.to_datetime(btc_df["timestamp"], utc=True)
    eval_start = _utc(panels["window"]["eval_start"])
    eval_end = _utc(panels["window"]["eval_end"])
    select_end = eval_start + pd.Timedelta(days=SELECT_DAYS) - pd.Timedelta(seconds=1)

    spec_by_key = {s.key: s for s in specs_from_config(sim)}
    fixed_keys = tuple(f"trail_{i}" for i in range(1, 11))
    fixed_specs = [spec_by_key[k] for k in fixed_keys]
    if len(fixed_specs) != 10:
        raise RuntimeError(f"expected T1–T10, got {len(fixed_specs)}")

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
    logger.info(
        "Stage1 ENTRY×T1–T10 | window %s → %s | select_end=%s | decisions=%d | coins=%d",
        eval_start,
        eval_end,
        select_end,
        len(decision_indices),
        len(panels["coins"]),
    )

    score_cache: dict[tuple[str, pd.Timestamp], dict[str, dict[str, Any]]] = {}
    leg_rows: list[dict[str, Any]] = []
    opp_rows: list[dict[str, Any]] = []
    max_simultaneous: dict[str, int] = {v: 0 for v in VARIANTS}

    def notional_usd_at(entry_ts: pd.Timestamp) -> float:
        btc_px = exit_tape.btc_close_near(entry_ts, default=100000.0)
        return float(STARTING_BTC * ALLOC_FRAC * btc_px)

    def advance_variant(variant: str, asof_t: pd.Timestamp) -> None:
        book = open_books[variant]
        sm = sms[variant]
        done: list[str] = []
        for oid, opp in book.items():
            base = opp["base"]
            if not exit_tape.has(base):
                continue
            last = _utc(opp.get("last_processed_ts") or opp["entry_fill_ts"])
            bars, btc_close = exit_tape.rel_and_btc_close(base, last, asof_t)
            if bars.empty:
                continue
            default_btc = float(btc_close.iloc[-1]) if len(btc_close) else exit_tape.btc_close_near(asof_t, 1.0)
            process_bars_until_closed(
                [item["leg"] for item in opp["leg_items"]],
                bars,
                btc_usdt_series=btc_close,
                default_btc_usdt=default_btc,
                costs=costs,
                same_candle_conflict="assume_sl_first",
            )
            opp["last_processed_ts"] = str(bars.iloc[-1]["timestamp"])
            if all(item["leg"].closed for item in opp["leg_items"]):
                for item in opp["leg_items"]:
                    leg = item["leg"]
                    rec = leg_to_record(leg, oid)
                    rec["arm_key"] = item["arm_key"]
                    rec["variant"] = variant
                    rec["base"] = base
                    rec["symbol"] = opp.get("symbol")
                    rec["phase"] = opp.get("phase")
                    rec["day_number"] = day_number_at(leg.exit_ts, eval_start) if leg.exit_ts is not None else None
                    leg_rows.append(rec)
                for o in opp_rows:
                    if o.get("opportunity_id") == oid:
                        o["status"] = "CLOSED"
                sm.register_close(oid, opp.get("symbol"))
                done.append(oid)
        for oid in done:
            del book[oid]
        max_simultaneous[variant] = max(max_simultaneous[variant], sm.n_open())

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
        for variant in VARIANTS:
            advance_variant(variant, t)

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
        phase = _phase(t, select_end)

        for base, coin in panels["coins"].items():
            if not exit_tape.has(base):
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
                score_cache[cache_key] = score_all_variants(
                    alt_btc=rel_hist,
                    alt_usdt=alt_vol_hist,
                    btc_usdt=btc_hist,
                    weights=weights,
                    interval=SIGNAL_INTERVAL,
                )
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
                btc_usdt = exit_tape.btc_close_near(entry_ts, default=float(btc_hist["close"].iloc[-1]))
                notional = notional_usd_at(entry_ts)
                opportunity_id = f"opp_{variant.lower()}_{uuid.uuid4().hex[:10]}"
                regime_info = classify_regime(
                    {"trend": sv.get("trend") or {}, "volatility": sv.get("volatility") or {}},
                    rules={},
                )
                legs = open_opportunity_legs(
                    alt_btc_entry_mid=entry_mid,
                    btc_usdt=btc_usdt,
                    notional_usd=notional,
                    costs=costs,
                    specs=fixed_specs,
                    entry_ts=entry_ts,
                )
                leg_items = [
                    {"leg": leg, "arm_key": arm}
                    for arm, leg in zip(EXIT_LABELS, legs)
                ]
                open_books[variant][opportunity_id] = {
                    "opportunity_id": opportunity_id,
                    "variant": variant,
                    "symbol": pair,
                    "base": base,
                    "entry_fill_ts": str(entry_ts),
                    "leg_items": leg_items,
                    "last_processed_ts": str(entry_ts),
                    "phase": phase,
                    "regime": regime_info["regime"],
                }
                sm.register_open(pair, opportunity_id)
                opp_rows.append(
                    {
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
                        "phase": phase,
                        "regime": regime_info["regime"],
                        "status": "OPEN",
                        "exit_arms": list(EXIT_LABELS),
                    }
                )

    for variant in VARIANTS:
        advance_variant(variant, eval_end)

    legs_df = pd.DataFrame(leg_rows)
    opp_df = pd.DataFrame(opp_rows)
    if not legs_df.empty:
        legs_df["entry_ts"] = pd.to_datetime(legs_df["entry_ts"], utc=True)
        legs_df["exit_ts"] = pd.to_datetime(legs_df["exit_ts"], utc=True)

    # --- summaries: matrix entry × T, plus entry aggregates ---
    matrix_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for arm in EXIT_LABELS:
            if legs_df.empty:
                sub = pd.DataFrame()
            else:
                sub = legs_df[(legs_df["variant"] == variant) & (legs_df["arm_key"] == arm)]
            for phase_name, phase_df in (
                ("full", sub),
                ("select", sub[sub["phase"] == "select"] if not sub.empty and "phase" in sub.columns else pd.DataFrame()),
                ("holdout", sub[sub["phase"] == "holdout"] if not sub.empty and "phase" in sub.columns else pd.DataFrame()),
            ):
                m = _summarize(phase_df, f"{variant}_{arm}")
                m["variant"] = variant
                m["arm"] = arm
                m["phase"] = phase_name
                matrix_rows.append(m)
    matrix_df = pd.DataFrame(matrix_rows)

    def _entry_aggregate(phase: str) -> pd.DataFrame:
        rows = []
        for variant in VARIANTS:
            sub = matrix_df[(matrix_df["variant"] == variant) & (matrix_df["phase"] == phase)]
            nets = pd.to_numeric(sub["Net BTC"], errors="coerce")
            rets = pd.to_numeric(sub["Return %"], errors="coerce")
            dds = pd.to_numeric(sub["Max DD"], errors="coerce")
            t1 = sub[sub["arm"] == "T1"]
            t1_net = float(t1["Net BTC"].iloc[0]) if not t1.empty else None
            best_idx = nets.idxmax() if nets.notna().any() else None
            best_arm = str(sub.loc[best_idx, "arm"]) if best_idx is not None else None
            rows.append(
                {
                    "Strategy": variant,
                    "mean_net_btc_T1T10": float(nets.mean()) if nets.notna().any() else None,
                    "median_net_btc_T1T10": float(nets.median()) if nets.notna().any() else None,
                    "best_T": best_arm,
                    "best_T_net_btc": float(nets.max()) if nets.notna().any() else None,
                    "T1_net_btc": t1_net,
                    "mean_return_pct": float(rets.mean()) if rets.notna().any() else None,
                    "worst_max_dd": float(dds.min()) if dds.notna().any() else None,
                    "Trades_T1": int(t1["Trades"].iloc[0]) if not t1.empty else 0,
                }
            )
        return pd.DataFrame(rows)

    agg_select = _entry_aggregate("select")
    agg_holdout = _entry_aggregate("holdout")
    agg_full = _entry_aggregate("full")

    # Primary entry pick: mean Net BTC across T1–T10 on SELECT
    rank_sel = agg_select.sort_values(
        by=["mean_net_btc_T1T10", "T1_net_btc", "worst_max_dd"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    winner_select = str(rank_sel.iloc[0]["Strategy"]) if not rank_sel.empty else None
    hold_row = agg_holdout[agg_holdout["Strategy"] == winner_select]
    winner_hold = hold_row.iloc[0].to_dict() if not hold_row.empty else {}

    # T1-only ranking (reference)
    t1_sel = matrix_df[(matrix_df["phase"] == "select") & (matrix_df["arm"] == "T1")].copy()
    t1_sel = t1_sel.sort_values(by=["Net BTC", "Expectancy"], ascending=[False, False])

    decision = {
        "stage": 1,
        "exit_arms": list(EXIT_LABELS),
        "variants": list(VARIANTS),
        "select_days": SELECT_DAYS,
        "holdout_days": HOLDOUT_DAYS,
        "primary_rule": "mean Net BTC across T1–T10 on 60d select",
        "winner_on_select": winner_select,
        "winner_select_mean_net_btc": None
        if not winner_select
        else float(rank_sel.iloc[0]["mean_net_btc_T1T10"] or 0),
        "winner_select_best_T": None if not winner_select else rank_sel.iloc[0].get("best_T"),
        "winner_select_T1_net_btc": None if not winner_select else rank_sel.iloc[0].get("T1_net_btc"),
        "winner_holdout_mean_net_btc": winner_hold.get("mean_net_btc_T1T10"),
        "winner_holdout_T1_net_btc": winner_hold.get("T1_net_btc"),
        "winner_holdout_worst_max_dd": winner_hold.get("worst_max_dd"),
        "t1_only_select_winner": str(t1_sel.iloc[0]["variant"]) if not t1_sel.empty else None,
        "rule": "Pick entry on 60d select (mean across T1–T10); confirm on 30d holdout. Do not pick on full 90d. Stage 2 still re-selects exit on the winning entry.",
        "next_stage": "Freeze winning entry; re-rank T1–T10 on that entry stream (stage 2).",
    }

    meta = {
        "eval_start": str(eval_start),
        "eval_end": str(eval_end),
        "select_end": str(select_end),
        "days": DAYS,
        "select_days": SELECT_DAYS,
        "holdout_days": HOLDOUT_DAYS,
        "signal_interval": SIGNAL_INTERVAL,
        "exit_interval": EXIT_INTERVAL,
        "exit_arms": list(EXIT_LABELS),
        "long_threshold": LONG_THRESHOLD,
        "max_open": MAX_OPEN,
        "variants": list(VARIANTS),
        "n_decision_bars": len(decision_indices),
        "n_opportunities": int(len(opp_df)),
        "n_closed_legs": int(len(legs_df)),
        "max_simultaneous": max_simultaneous,
        "n_coins": len(panels["coins"]),
        "decision": decision,
        "notes": [
            "Stage 1: entry sweep BASE/E1–E5 with counterfactual T1–T10 on each entry",
            "15m signals; 1s trail exits",
            "Walk-forward: first 60d=select, last 30d=holdout",
            "No selectors (A–F are stage 3)",
        ],
    }

    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    (out_dir / "decision.json").write_text(json.dumps(decision, indent=2, default=str), encoding="utf-8")
    legs_df.to_csv(out_dir / "strategy_legs.csv", index=False)
    opp_df.to_csv(out_dir / "opportunities.csv", index=False)
    matrix_df.to_csv(out_dir / "entry_x_T_matrix.csv", index=False)
    agg_full.to_csv(out_dir / "entry_summary_full90d.csv", index=False)
    agg_select.to_csv(out_dir / "entry_summary_select60d.csv", index=False)
    agg_holdout.to_csv(out_dir / "entry_summary_holdout30d.csv", index=False)
    rank_sel.to_csv(out_dir / "entry_ranking_select60d.csv", index=False)
    t1_sel.to_csv(out_dir / "entry_ranking_T1_only_select60d.csv", index=False)

    # Heatmap: select Net BTC entry × T
    try:
        heat = (
            matrix_df[matrix_df["phase"] == "select"]
            .pivot(index="variant", columns="arm", values="Net BTC")
            .reindex(index=list(VARIANTS), columns=list(EXIT_LABELS))
        )
        fig, ax = plt.subplots(figsize=(12, 4.5))
        im = ax.imshow(heat.astype(float).fillna(0.0).values, aspect="auto", cmap="RdYlGn")
        ax.set_xticks(range(len(EXIT_LABELS)))
        ax.set_xticklabels(EXIT_LABELS)
        ax.set_yticks(range(len(VARIANTS)))
        ax.set_yticklabels(VARIANTS)
        ax.set_title("Stage1 select Net BTC — entry × T1–T10")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(plots_dir / "select_heatmap_entry_x_T.png", dpi=140)
        plt.close(fig)
    except Exception as e:
        logger.warning("heatmap failed: %s", e)

    _plot_bars(
        rank_sel.rename(columns={"mean_net_btc_T1T10": "Net BTC"}),
        "Net BTC",
        "Stage1 entry — mean Net BTC across T1–T10 (60d select)",
        plots_dir / "select_mean_net_btc_by_entry.png",
        "Mean Net BTC",
    )
    _plot_bars(
        agg_holdout.rename(columns={"mean_net_btc_T1T10": "Net BTC"}),
        "Net BTC",
        "Stage1 entry — mean Net BTC across T1–T10 (30d holdout)",
        plots_dir / "holdout_mean_net_btc_by_entry.png",
        "Mean Net BTC",
    )
    if not t1_sel.empty:
        t1_plot = t1_sel.copy()
        if "variant" in t1_plot.columns:
            t1_plot["Strategy"] = t1_plot["variant"].astype(str)
        # drop duplicate Strategy columns if present
        if isinstance(t1_plot["Strategy"], pd.DataFrame):
            t1_plot = t1_plot.loc[:, ~t1_plot.columns.duplicated()]
            t1_plot["Strategy"] = t1_sel["variant"].astype(str)
        _plot_bars(
            t1_plot,
            "Net BTC",
            "Stage1 entry — T1 only Net BTC (60d select)",
            plots_dir / "select_T1_net_btc_by_entry.png",
            "Net BTC",
        )

    report = [
        "# Stage 1 — Entry selection (BASE/E1–E5 × T1–T10)",
        "",
        f"- Window: `{eval_start}` → `{eval_end}` (90d)",
        f"- Select: first {SELECT_DAYS}d | Holdout: last {HOLDOUT_DAYS}d",
        f"- Exits: **{', '.join(EXIT_LABELS)}** on **1s** tape (counterfactual per entry)",
        f"- Variants: {', '.join(VARIANTS)}",
        "",
        "## Decision (select-window winner)",
        f"- **Primary winner (mean T1–T10):** `{winner_select}`",
        f"- Select mean Net BTC: `{decision['winner_select_mean_net_btc']}`",
        f"- Select best T for winner: `{decision['winner_select_best_T']}`",
        f"- Select T1 Net BTC: `{decision['winner_select_T1_net_btc']}`",
        f"- Holdout mean Net BTC: `{decision['winner_holdout_mean_net_btc']}`",
        f"- T1-only select winner (reference): `{decision['t1_only_select_winner']}`",
        "",
        "## Select ranking by mean Net BTC across T1–T10",
        "```",
        rank_sel[
            ["Strategy", "mean_net_btc_T1T10", "T1_net_btc", "best_T", "best_T_net_btc", "worst_max_dd", "Trades_T1"]
        ].to_string(index=False)
        if not rank_sel.empty
        else "(empty)",
        "```",
        "",
        "## Holdout confirmation",
        "```",
        agg_holdout[
            ["Strategy", "mean_net_btc_T1T10", "T1_net_btc", "best_T", "worst_max_dd", "Trades_T1"]
        ].to_string(index=False)
        if not agg_holdout.empty
        else "(empty)",
        "```",
        "",
        "## Next",
        "Freeze the winning entry; run Stage 2 to re-rank T1–T10 on that fixed entry stream.",
        "Do not promote the Stage-1 `best_T` without Stage-2 confirmation.",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")

    logger.info("Wrote Stage1 results → %s", out_dir)
    print(rank_sel[["Strategy", "mean_net_btc_T1T10", "T1_net_btc", "best_T"]].to_string(index=False))
    print(f"\nWinner on select (mean T1–T10): {winner_select}")
    print(f"Results: {out_dir}")
    return out_dir


if __name__ == "__main__":
    run()
