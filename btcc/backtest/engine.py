"""90-day walk-forward prediction research engine — NO trading, NO Telegram."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from btcc.backtest.data_loader import compute_window, download_panels
from btcc.backtest.dominance_history import HistoricalDominanceSeries
from btcc.backtest.outcomes import future_outcomes
from btcc.backtest.predict import predict_coin_at_bar
from btcc.ranking.ranker import rank_signals
from btcc.series.relative import horizon_bars

logger = logging.getLogger(__name__)


def _align_btc_index(btc_df: pd.DataFrame, eval_start: pd.Timestamp, eval_end: pd.Timestamp,
                     min_warmup: int, max_fwd_bars: int) -> tuple[pd.DatetimeIndex, int, int]:
    """Decision timestamps within [eval_start, eval_end] with enough history and forward bars."""
    ts = pd.to_datetime(btc_df["timestamp"], utc=True)
    mask = (ts >= eval_start) & (ts <= eval_end)
    indices = btc_df.index[mask].tolist()
    valid = [i for i in indices if i >= min_warmup and i + max_fwd_bars < len(btc_df)]
    return ts, min(valid) if valid else 0, max(valid) if valid else -1


def run_backtest(cfg: dict[str, Any], days: int | None = None, force_download: bool = False) -> Path:
    days = days or int(cfg["backtest"]["days"])
    interval = cfg["backtest"]["interval"]
    min_warmup = int(cfg["backtest"]["min_warmup_bars"])
    top_n = int(cfg["backtest"]["top_n"])
    horizons = list(cfg["backtest"]["horizons_hours"])
    max_fwd = max(horizon_bars(h, interval) for h in horizons)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg["backtest_output"]["results_root"]) / f"backtest_90d_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    _, eval_start, eval_end = compute_window(days, min_warmup)

    logger.info("Downloading panels for %d days...", days)
    panels = download_panels(cfg, days, min_warmup, force=force_download)
    btc_df = panels["btc"]
    eval_start = panels["window"]["eval_start"]
    eval_end = panels["window"]["eval_end"]
    data_start = panels["window"]["data_start"]

    logger.info("Fetching historical BTC dominance...")
    # CoinGecko: days<=90 → hourly; days>90 → daily. Prefer hourly for 15m alignment.
    dom_days = min(int(days), 90)
    if days > 90:
        logger.warning(
            "Dominance fetch capped at 90 days for hourly CoinGecko resolution "
            "(requested %d). Early bars may lack BTC.D.",
            days,
        )
    dom_series = HistoricalDominanceSeries.fetch_coingecko(
        days=dom_days,
        cache_dir=cfg["backtest_data"]["dominance_cache"],
        force=force_download,
    )
    dom_report = dom_series.coverage_report(eval_start, eval_end)
    if dom_series.df.empty:
        logger.warning(
            "BTC dominance history unavailable — BTC Regime will use INSUFFICIENT_DATA "
            "(no fabrication). Set BTCC_COINGECKO_API_KEY for full history."
        )

    ts_index, i_start, i_end = _align_btc_index(btc_df, eval_start, eval_end, min_warmup, max_fwd)
    if i_end < i_start:
        raise RuntimeError("No valid decision bars in evaluation window")

    decision_indices = list(range(i_start, i_end + 1))
    logger.info(
        "Backtest window: %s → %s | decision bars: %d",
        eval_start, eval_end, len(decision_indices),
    )

    all_predictions: list[dict] = []
    top5_predictions: list[dict] = []
    n_skipped_dom = 0

    for step, i in enumerate(decision_indices, 1):
        ts = pd.Timestamp(ts_index.iloc[i])
        btc_hist = btc_df.iloc[: i + 1].copy()
        dom_pct, dom_obs_ts, dom_status = dom_series.observation_at(ts)
        dom_changes = dom_series.dom_changes_at(ts)
        dom_age_h = None
        if dom_obs_ts is not None:
            dom_age_h = (ts - dom_obs_ts).total_seconds() / 3600.0

        rows = []
        for base, coin in panels["coins"].items():
            rel = coin["rel"]
            # Align rel to btc timestamp at i
            rel_at = rel[rel["timestamp"] <= ts]
            if len(rel_at) < 100:
                continue
            alt_vol = coin["alt_for_volume"]
            alt_vol_at = alt_vol[alt_vol["timestamp"] <= ts] if alt_vol is not None else rel_at

            pred = predict_coin_at_bar(
                rel_at, alt_vol_at, btc_hist, dom_pct, dom_changes, cfg, interval, calib=None
            )
            if pred is None:
                continue
            pred.update({
                "timestamp": ts,
                "base": base,
                "symbol": coin["symbol"],
                "construction": coin["construction"],
                "btc_dominance": dom_pct,
                "btc_dominance_obs_ts": str(dom_obs_ts) if dom_obs_ts is not None else None,
                "btc_dominance_age_hours": dom_age_h,
                "dominance_status": dom_status,
            })
            rows.append({
                **pred,
                "p_1h": pred["probability_1h"],
                "p_4h": pred["probability_4h"],
                "p_8h": pred["probability_8h"],
                "p_12h": pred["probability_12h"],
                "p_24h": pred["probability_24h"],
            })

        if not rows:
            continue

        ranked = rank_signals(rows, cfg["probability"]["primary_rank_horizon"])
        for r in ranked[:top_n]:
            rel_full = panels["coins"][r["base"]]["rel"]
            rel_close = rel_full["close"]
            pos = int((rel_full["timestamp"] <= ts).sum()) - 1
            if pos < 0:
                continue

            outs = future_outcomes(rel_close, pos, interval, horizons)
            record = {k: v for k, v in r.items() if k not in ("factors", "late_entry", "p_1h", "p_8h", "p_12h", "p_24h")}
            record.update(outs)
            record["rank"] = r["rank"]
            if dom_status != "OK":
                n_skipped_dom += 1
            top5_predictions.append(record)

        if step % 200 == 0 or step == len(decision_indices):
            logger.info("Progress %d/%d decision bars | top5 rows: %d", step, len(decision_indices), len(top5_predictions))

    # Save raw outputs
    meta = {
        "run_id": run_id,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "eval_start": str(eval_start),
        "eval_end": str(eval_end),
        "data_start": str(data_start),
        "days": days,
        "interval": interval,
        "decision_bars": len(decision_indices),
        "top5_predictions": len(top5_predictions),
        "universe_size": len(cfg["universe"]["bases"]),
        "valid_symbols": len(panels["coins"]),
        "unavailable_symbols": panels["unavailable"],
        "construction": panels["construction"],
        "dominance": dom_report,
        "dominance_status": dom_report.get("status", "OK" if dom_report.get("n_points") else "INSUFFICIENT_DATA"),
        "dominance_missing_at_decisions": n_skipped_dom,
        "baseline_comparison_run": "backtest_90d_20260828_005424",
        "lookahead_policy": (
            "At decision t, all indicators use OHLCV with timestamp <= t only. "
            "Rolling stats computed on truncated history. "
            "BTC.D uses last observation with timestamp <= t (recorded as "
            "btc_dominance_obs_ts); never interpolated; never uses future BTC.D. "
            "Outcomes use future bars only for labels, never for features."
        ),
        "probability_note": "baseline_model_probability — NOT validated empirical probability",
        "orders_sent": 0,
        "telegram_sent": 0,
        "strategy_unchanged": True,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    top5_df = pd.DataFrame(top5_predictions)
    if not top5_df.empty:
        top5_df.to_csv(out_dir / "top5_predictions.csv", index=False)
        top5_df.to_csv(out_dir / "predictions.csv", index=False)

    logger.info("Saved %d Top-5 predictions to %s", len(top5_predictions), out_dir)
    return out_dir
