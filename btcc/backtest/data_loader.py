"""Download historical candles for backtest window (interval-aware: 5m/15m/…)."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from btcc.data.websocket import INTERVAL_MS, MexcPublicREST
from btcc.series.relative import build_alt_btc, native_btc_as_relative
from btcc.universe import resolve_btc_market, resolve_data_market, resolve_usdt_symbol

logger = logging.getLogger(__name__)

BARS_PER_DAY_15M = 96  # backward-compatible alias
BARS_PER_DAY = {"5m": 288, "15m": 96, "1h": 24, "4h": 6, "1d": 1}


def _interval_minutes(interval: str) -> float:
    ms = INTERVAL_MS.get(interval)
    if ms is None:
        # Fallback parse like "5m", "15m"
        if interval.endswith("m") and interval[:-1].isdigit():
            return float(interval[:-1])
        if interval.endswith("h") and interval[:-1].isdigit():
            return float(interval[:-1]) * 60.0
        return 15.0
    return float(ms) / 60_000.0


def bars_per_day(interval: str = "15m") -> int:
    if interval in BARS_PER_DAY:
        return int(BARS_PER_DAY[interval])
    mins = _interval_minutes(interval)
    return max(1, int(round(1440.0 / mins)))


def compute_window(
    days: int,
    warmup_bars: int,
    interval: str = "15m",
    *,
    eval_end: pd.Timestamp | datetime | str | None = None,
    eval_start: pd.Timestamp | datetime | str | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Return (data_start, eval_start, eval_end).

    Default: eval_end = now UTC, eval_start = eval_end - days.
    Optional pinned eval_end / eval_start for historical windows (e.g. prior year).
    If both pinned, ``days`` is ignored for the eval span (still used by callers for labels).
    """
    if eval_end is not None:
        ee = pd.Timestamp(eval_end)
        if ee.tzinfo is None:
            ee = ee.tz_localize("UTC")
        else:
            ee = ee.tz_convert("UTC")
    else:
        ee = pd.Timestamp(datetime.now(timezone.utc))

    if eval_start is not None:
        es = pd.Timestamp(eval_start)
        if es.tzinfo is None:
            es = es.tz_localize("UTC")
        else:
            es = es.tz_convert("UTC")
    else:
        es = ee - pd.Timedelta(days=days)

    data_start = es - pd.Timedelta(minutes=_interval_minutes(interval) * warmup_bars)
    return data_start, es, ee


def required_bars(days: int, warmup_bars: int, interval: str = "15m") -> int:
    return days * bars_per_day(interval) + warmup_bars + 100


def _load_symbol_offline(candle_dir: str, symbol: str, interval: str) -> pd.DataFrame | None:
    from btcc.data.candles import candle_path, load_candles

    path = candle_path(candle_dir, symbol, interval)
    return load_candles(path)


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


def download_panels(
    cfg: dict,
    days: int,
    warmup_bars: int,
    force: bool = False,
    *,
    eval_start: pd.Timestamp | datetime | str | None = None,
    eval_end: pd.Timestamp | datetime | str | None = None,
    offline_candles: bool = False,
) -> dict:
    """Download BTC + universe symbols; build relative panels.

    Uses shared ``resolve_data_market`` so native-only bases (e.g. CKBTC→CKBTCBTC)
    never probe an invalid USDT symbol.

    ``offline_candles=True`` loads only from local parquet/csv (no exchange fetch) —
    used for Binance Vision pre-downloaded historical windows.
    """
    interval = cfg["backtest"]["interval"]
    data_start, eval_start_ts, eval_end_ts = compute_window(
        days,
        warmup_bars,
        interval=interval,
        eval_start=eval_start,
        eval_end=eval_end,
    )
    lookback = required_bars(days, warmup_bars, interval=interval)
    candle_dir = cfg["backtest_data"]["candle_dir"]
    rest = None if offline_candles else MexcPublicREST(cfg["data"]["mexc_rest"])

    btc_sym = cfg["universe"]["btc_symbol"]
    if offline_candles:
        btc_df = _load_symbol_offline(candle_dir, btc_sym, interval)
    else:
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

    def _fetch(sym: str) -> pd.DataFrame | None:
        if offline_candles:
            return _load_symbol_offline(candle_dir, sym, interval)
        return rest.bootstrap_symbol(sym, interval, lookback, candle_dir, force=force)

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
            native = _fetch(resolved)
            if native is None or native.empty:
                panels["unavailable"].append(base)
                panels["construction"][base] = "DATA_UNAVAILABLE"
                panels["coverage"][base] = pair_coverage_row(
                    logical_pair=base, resolved_market=resolved, construction=mode,
                    df=None, eval_start=eval_start_ts, eval_end=eval_end_ts, init_days=init_days,
                    min_pred_bars=min_pred,
                )
                logger.error("DATA_UNAVAILABLE: native %s (logical=%s)", resolved, base)
                continue
            rel = native_btc_as_relative(native)
            alt_for_vol = native
            alt_df = native
        else:
            alt_df = _fetch(resolved)
            if alt_df is not None and not alt_df.empty:
                rel = build_alt_btc(alt_df, btc_df)
                alt_for_vol = alt_df
            else:
                # Unexpected missing USDT for a synthetic pair — try native once
                # (not used for native_btc_only bases, which never enter this branch).
                native = _fetch(btc_mkt)
                if native is None or native.empty:
                    panels["unavailable"].append(base)
                    panels["construction"][base] = "DATA_UNAVAILABLE"
                    panels["coverage"][base] = pair_coverage_row(
                        logical_pair=base, resolved_market=resolved, construction=mode,
                        df=None, eval_start=eval_start_ts, eval_end=eval_end_ts, init_days=init_days,
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
                df=None, eval_start=eval_start_ts, eval_end=eval_end_ts, init_days=init_days,
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
            eval_start=eval_start_ts,
            eval_end=eval_end_ts,
            init_days=init_days,
            min_pred_bars=min_pred,
        )

    panels["window"] = {
        "data_start": data_start,
        "eval_start": eval_start_ts,
        "eval_end": eval_end_ts,
        "offline_candles": bool(offline_candles),
        "candle_dir": candle_dir,
    }
    return panels
