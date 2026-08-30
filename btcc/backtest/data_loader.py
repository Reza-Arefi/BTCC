"""Download historical 15m candles for backtest window."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from btcc.data.websocket import MexcPublicREST
from btcc.series.relative import build_alt_btc, native_btc_as_relative
from btcc.universe import resolve_btc_market, resolve_data_market, resolve_usdt_symbol

logger = logging.getLogger(__name__)

BARS_PER_DAY_15M = 96


def compute_window(days: int, warmup_bars: int) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Return (data_start, eval_start, eval_end) pinned at run time UTC."""
    eval_end = pd.Timestamp(datetime.now(timezone.utc))
    eval_start = eval_end - pd.Timedelta(days=days)
    data_start = eval_start - pd.Timedelta(minutes=15 * warmup_bars)
    return data_start, eval_start, eval_end


def required_bars(days: int, warmup_bars: int) -> int:
    return days * BARS_PER_DAY_15M + warmup_bars + 100


def pair_coverage_row(
    *,
    logical_pair: str,
    resolved_market: str,
    construction: str,
    df: pd.DataFrame | None,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    init_days: int = 90,
    min_pred_bars: int = 100,
) -> dict:
    """Coverage metadata for one universe pair (no fabricated history)."""
    row: dict = {
        "logical_pair": logical_pair,
        "resolved_market": resolved_market,
        "construction": construction,
        "first_timestamp": None,
        "last_timestamp": None,
        "available_days": 0.0,
        "candle_count": 0,
        "first_usable_prediction_ts": None,
        "usable_prediction_start": None,
        "usable_prediction_end": None,
        "insufficient_history_before": None,
        "usable_for_initial_90d": False,
        "usable_for_daily_adaptation": False,
        "note": "",
    }
    if df is None or df.empty:
        row["note"] = "NO_DATA"
        return row
    ts = pd.to_datetime(df["timestamp"], utc=True)
    first, last = ts.min(), ts.max()
    row["first_timestamp"] = str(first)
    row["last_timestamp"] = str(last)
    row["available_days"] = float((last - first).total_seconds() / 86400.0)
    row["candle_count"] = int(len(df))
    # Predictions require min_pred_bars of history ending at decision bar
    if len(df) < min_pred_bars:
        row["note"] = "INSUFFICIENT_BARS_FOR_ANY_PREDICTION"
        row["insufficient_history_before"] = str(last)
        return row
    first_usable = ts.iloc[min_pred_bars - 1]
    row["first_usable_prediction_ts"] = str(first_usable)
    row["insufficient_history_before"] = str(first_usable)
    usable_start = max(first_usable, eval_start)
    usable_end = min(last, eval_end)
    if usable_start <= usable_end:
        row["usable_prediction_start"] = str(usable_start)
        row["usable_prediction_end"] = str(usable_end)
        row["usable_for_daily_adaptation"] = True
        init_end = eval_start + pd.Timedelta(days=init_days)
        # Can contribute during the init window if usable before init_end
        row["usable_for_initial_90d"] = bool(usable_start < init_end)
    else:
        row["note"] = "NO_OVERLAP_WITH_EVAL_WINDOW"
    return row


def download_panels(cfg: dict, days: int, warmup_bars: int, force: bool = False) -> dict:
    """Download BTC + universe symbols; build relative panels.

    Uses shared ``resolve_data_market`` so native-only bases (e.g. CKBTC→CKBTCBTC)
    never probe an invalid USDT symbol.
    """
    data_start, eval_start, eval_end = compute_window(days, warmup_bars)
    lookback = required_bars(days, warmup_bars)
    interval = cfg["backtest"]["interval"]
    candle_dir = cfg["backtest_data"]["candle_dir"]
    rest = MexcPublicREST(cfg["data"]["mexc_rest"])

    btc_sym = cfg["universe"]["btc_symbol"]
    btc_df = rest.bootstrap_symbol(btc_sym, interval, lookback, candle_dir, force=force)
    if btc_df is None or btc_df.empty:
        raise RuntimeError(f"DATA_UNAVAILABLE: {btc_sym}")

    panels: dict = {
        "btc": btc_df,
        "coins": {},
        "unavailable": [],
        "construction": {},
        "coverage": {},
    }

    init_days = int(((cfg.get("sim") or {}).get("walk_forward") or {}).get("init_days", 90))
    min_pred = 100

    for base in cfg["universe"]["bases"]:
        meta = resolve_data_market(base, cfg)
        resolved = meta["resolved_market"]
        mode = meta["construction"]
        usdt = resolve_usdt_symbol(base, cfg)
        btc_mkt = resolve_btc_market(base, cfg)

        alt_df = None
        rel = None
        alt_for_vol = None
        sym_key = resolved

        if meta["mode"] == "native_btc":
            native = rest.bootstrap_symbol(resolved, interval, lookback, candle_dir, force=force)
            if native is None or native.empty:
                panels["unavailable"].append(base)
                panels["construction"][base] = "DATA_UNAVAILABLE"
                panels["coverage"][base] = pair_coverage_row(
                    logical_pair=base, resolved_market=resolved, construction=mode,
                    df=None, eval_start=eval_start, eval_end=eval_end, init_days=init_days,
                    min_pred_bars=min_pred,
                )
                logger.error("DATA_UNAVAILABLE: native %s (logical=%s)", resolved, base)
                continue
            rel = native_btc_as_relative(native)
            alt_for_vol = native
            alt_df = native
        else:
            alt_df = rest.bootstrap_symbol(resolved, interval, lookback, candle_dir, force=force)
            if alt_df is not None and not alt_df.empty:
                rel = build_alt_btc(alt_df, btc_df)
                alt_for_vol = alt_df
            else:
                # Unexpected missing USDT for a synthetic pair — try native once
                # (not used for native_btc_only bases, which never enter this branch).
                native = rest.bootstrap_symbol(btc_mkt, interval, lookback, candle_dir, force=force)
                if native is None or native.empty:
                    panels["unavailable"].append(base)
                    panels["construction"][base] = "DATA_UNAVAILABLE"
                    panels["coverage"][base] = pair_coverage_row(
                        logical_pair=base, resolved_market=resolved, construction=mode,
                        df=None, eval_start=eval_start, eval_end=eval_end, init_days=init_days,
                        min_pred_bars=min_pred,
                    )
                    logger.error("DATA_UNAVAILABLE: %s and %s", usdt, btc_mkt)
                    continue
                rel = native_btc_as_relative(native)
                alt_for_vol = native
                alt_df = native
                sym_key = btc_mkt
                mode = f"native {btc_mkt}"
                resolved = btc_mkt

        if rel is None or rel.empty:
            panels["unavailable"].append(base)
            panels["construction"][base] = "INSUFFICIENT_DATA"
            panels["coverage"][base] = pair_coverage_row(
                logical_pair=base, resolved_market=resolved, construction=mode,
                df=None, eval_start=eval_start, eval_end=eval_end, init_days=init_days,
                min_pred_bars=min_pred,
            )
            continue

        panels["coins"][base] = {
            "base": base,
            "logical_pair": base,
            "resolved_market": resolved,
            "symbol": sym_key,
            "usdt": usdt,
            "btc_market": btc_mkt,
            "construction": mode,
            "alt_usdt": alt_df if meta["mode"] == "synthetic_usdt" else None,
            "rel": rel,
            "alt_for_volume": alt_for_vol,
        }
        panels["construction"][base] = mode
        panels["coverage"][base] = pair_coverage_row(
            logical_pair=base,
            resolved_market=resolved,
            construction=mode,
            df=rel,
            eval_start=eval_start,
            eval_end=eval_end,
            init_days=init_days,
            min_pred_bars=min_pred,
        )

    panels["window"] = {
        "data_start": data_start,
        "eval_start": eval_start,
        "eval_end": eval_end,
    }
    return panels
