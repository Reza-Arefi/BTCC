"""Binance public klines download (Vision monthly/daily zips + REST fill)."""

from __future__ import annotations

import io
import logging
import time
import zipfile
import json
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import pandas as pd
import requests

from btcc.data.candles import candle_path, load_candles, save_candles

logger = logging.getLogger(__name__)

PUBLIC_REST = "https://data-api.binance.vision"
VISION_DATA = "https://data.binance.vision"
INTERVAL_MS = {
    "1s": 1_000,
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


def _as_utc(ts: Any) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _session() -> requests.Session:
    sess = requests.Session()
    sess.headers.update({"User-Agent": "BTCC-backtest/0.1 (research)"})
    return sess


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])


def _klines_to_df(raw: list) -> pd.DataFrame:
    if not raw:
        return _empty()
    rows = [
        {
            "timestamp": pd.to_datetime(int(k[0]), unit="ms", utc=True),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
        }
        for k in raw
    ]
    return (
        pd.DataFrame(rows)
        .drop_duplicates("timestamp", keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def _csv_bytes_to_df(raw_csv: bytes) -> pd.DataFrame:
    raw = pd.read_csv(io.BytesIO(raw_csv), header=None)
    if raw.empty:
        return _empty()
    # Vision CSV: open_time, open, high, low, close, volume, ...
    # 1s monthly/daily zips use microseconds; coarser intervals often use ms.
    ts = raw.iloc[:, 0]
    if pd.api.types.is_numeric_dtype(ts):
        vmax = float(pd.to_numeric(ts, errors="coerce").max())
        if vmax >= 1e17:  # nanoseconds
            unit = "ns"
        elif vmax >= 1e14:  # microseconds (Binance Vision 1s)
            unit = "us"
        elif vmax >= 1e11:  # milliseconds
            unit = "ms"
        else:
            unit = "s"
        timestamp = pd.to_datetime(ts, unit=unit, utc=True)
    else:
        timestamp = pd.to_datetime(ts, utc=True)
    out = pd.DataFrame(
        {
            "timestamp": timestamp,
            "open": pd.to_numeric(raw.iloc[:, 1], errors="coerce"),
            "high": pd.to_numeric(raw.iloc[:, 2], errors="coerce"),
            "low": pd.to_numeric(raw.iloc[:, 3], errors="coerce"),
            "close": pd.to_numeric(raw.iloc[:, 4], errors="coerce"),
            "volume": pd.to_numeric(raw.iloc[:, 5], errors="coerce"),
        }
    )
    return (
        out.dropna(subset=["timestamp", "close"])
        .drop_duplicates("timestamp", keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def fetch_klines_api(
    symbol: str,
    interval: str,
    start: Any,
    end: Any,
    *,
    session: requests.Session | None = None,
    sleep_s: float = 0.05,
    base_url: str = PUBLIC_REST,
) -> pd.DataFrame:
    """Paginate ``/api/v3/klines`` (max 1000 per call) over [start, end]."""
    sess = session or _session()
    sym = str(symbol).upper()
    iv = str(interval)
    if iv not in INTERVAL_MS:
        raise ValueError(f"unsupported interval: {interval}")
    start_ms = int(_as_utc(start).timestamp() * 1000)
    end_ms = int(_as_utc(end).timestamp() * 1000)
    step = INTERVAL_MS[iv]
    frames: list[pd.DataFrame] = []
    cursor = start_ms
    url = urljoin(base_url.rstrip("/") + "/", "api/v3/klines")
    while cursor < end_ms:
        params = {
            "symbol": sym,
            "interval": iv,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": 1000,
        }
        raw: list | None = None
        for attempt in range(4):
            try:
                r = sess.get(url, params=params, timeout=60)
                if r.status_code == 429:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                r.raise_for_status()
                raw = r.json()
                break
            except Exception as e:
                if attempt == 3:
                    raise
                logger.warning("%s %s API retry %d: %s", sym, iv, attempt + 1, e)
                time.sleep(1.5 * (attempt + 1))
        if not isinstance(raw, list) or not raw:
            break
        frames.append(_klines_to_df(raw))
        last_open = int(raw[-1][0])
        nxt = last_open + step
        if nxt <= cursor:
            break
        cursor = nxt
        if len(raw) < 1000:
            break
        if sleep_s:
            time.sleep(sleep_s)
    if not frames:
        return _empty()
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("timestamp", keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def _month_range(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[int, int]]:
    cur = pd.Timestamp(year=start.year, month=start.month, day=1, tz="UTC")
    end_m = pd.Timestamp(year=end.year, month=end.month, day=1, tz="UTC")
    out: list[tuple[int, int]] = []
    while cur <= end_m:
        out.append((cur.year, cur.month))
        if cur.month == 12:
            cur = pd.Timestamp(year=cur.year + 1, month=1, day=1, tz="UTC")
        else:
            cur = pd.Timestamp(year=cur.year, month=cur.month + 1, day=1, tz="UTC")
    return out


def _day_range(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    s = _as_utc(start).floor("D")
    e = _as_utc(end).floor("D")
    return list(pd.date_range(s, e, freq="D", tz="UTC"))


def _fetch_vision_zip(url: str, session: requests.Session, timeout: int = 180) -> pd.DataFrame:
    try:
        r = session.get(url, timeout=timeout)
        if r.status_code != 200:
            return _empty()
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            members = [m for m in zf.namelist() if m.endswith(".csv")]
            if not members:
                return _empty()
            with zf.open(members[0]) as fh:
                return _csv_bytes_to_df(fh.read())
    except Exception as e:
        logger.debug("Vision miss %s: %s", url, e)
        return _empty()


def fetch_vision_month(
    symbol: str,
    interval: str,
    year: int,
    month: int,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    sess = session or _session()
    sym = symbol.upper()
    name = f"{sym}-{interval}-{year}-{month:02d}.zip"
    url = f"{VISION_DATA}/data/spot/monthly/klines/{sym}/{interval}/{name}"
    return _fetch_vision_zip(url, sess, timeout=300)


def fetch_vision_day(
    symbol: str,
    interval: str,
    day: pd.Timestamp,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    sess = session or _session()
    sym = symbol.upper()
    d = _as_utc(day)
    name = f"{sym}-{interval}-{d.strftime('%Y-%m-%d')}.zip"
    url = f"{VISION_DATA}/data/spot/daily/klines/{sym}/{interval}/{name}"
    return _fetch_vision_zip(url, sess, timeout=120)


def coverage_ratio(df: pd.DataFrame | None, start: Any, end: Any, interval: str) -> float:
    if df is None or df.empty or interval not in INTERVAL_MS:
        return 0.0
    s, e = _as_utc(start), _as_utc(end)
    w = df[(df["timestamp"] >= s) & (df["timestamp"] <= e)]
    expected = max(1, int((e - s).total_seconds() * 1000 / INTERVAL_MS[interval]))
    return float(len(w) / expected)


def partitioned_1s_dir(candle_dir: Path | str, symbol: str) -> Path:
    return Path(candle_dir) / "1s" / str(symbol).upper()


def save_candles_partitioned_1s(df: pd.DataFrame, candle_dir: Path | str, symbol: str) -> Path:
    """Write 1s candles as daily parquet parts under ``1s/{SYMBOL}/YYYY-MM-DD.parquet``."""
    root = partitioned_1s_dir(candle_dir, symbol)
    root.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = (
        out.dropna(subset=["timestamp", "close"])
        .drop_duplicates("timestamp", keep="last")
        .sort_values("timestamp")
    )
    out["_date"] = out["timestamp"].dt.strftime("%Y-%m-%d")
    for date, g in out.groupby("_date", sort=True):
        part = root / f"{date}.parquet"
        save_candles(g.drop(columns=["_date"]), part)
    # Also write a thin manifest for quick existence checks
    manifest = {
        "symbol": str(symbol).upper(),
        "interval": "1s",
        "n_rows": int(len(out)),
        "first": str(out["timestamp"].iloc[0]) if not out.empty else None,
        "last": str(out["timestamp"].iloc[-1]) if not out.empty else None,
        "n_days": int(out["_date"].nunique()) if not out.empty else 0,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return root


def load_candles_range_1s(
    candle_dir: Path | str,
    symbol: str,
    start: Any,
    end: Any,
) -> pd.DataFrame:
    """Load only the daily 1s parts overlapping [start, end]."""
    root = partitioned_1s_dir(candle_dir, symbol)
    s, e = _as_utc(start), _as_utc(end)
    if not root.exists():
        # Fallback: single-file cache
        single = load_candles(candle_path(candle_dir, symbol, "1s"))
        if single is None or single.empty:
            return _empty()
        return single[(single["timestamp"] >= s) & (single["timestamp"] <= e)].reset_index(drop=True)

    frames: list[pd.DataFrame] = []
    for day in _day_range(s, e):
        part = root / f"{day.strftime('%Y-%m-%d')}.parquet"
        if not part.exists():
            continue
        df = load_candles(part)
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        return _empty()
    out = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("timestamp", keep="last")
        .sort_values("timestamp")
    )
    return out[(out["timestamp"] >= s) & (out["timestamp"] <= e)].reset_index(drop=True)


def load_partitioned_1s_all(candle_dir: Path | str, symbol: str) -> pd.DataFrame | None:
    root = partitioned_1s_dir(candle_dir, symbol)
    if not root.exists():
        return load_candles(candle_path(candle_dir, symbol, "1s"))
    parts = sorted(root.glob("????-??-??.parquet"))
    if not parts:
        return None
    frames = [load_candles(p) for p in parts]
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return None
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("timestamp", keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def download_symbol_1s(
    symbol: str,
    start: Any,
    end: Any,
    candle_dir: Path | str,
    *,
    session: requests.Session | None = None,
    min_coverage: float = 0.90,
) -> dict[str, Any]:
    """Download full-window 1s via Vision monthly/daily zips + API gap fill."""
    sess = session or _session()
    sym = str(symbol).upper()
    start_ts, end_ts = _as_utc(start), _as_utc(end)
    frames: list[pd.DataFrame] = []

    cached = load_partitioned_1s_all(candle_dir, sym)
    if cached is not None and not cached.empty:
        frames.append(cached)
        cov = coverage_ratio(cached, start_ts, end_ts, "1s")
        if cov >= min_coverage:
            logger.info("%s 1s cache hit coverage=%.1f%%", sym, 100 * cov)
            save_candles_partitioned_1s(cached, candle_dir, sym)
            return {"ok": True, "n": int(len(cached)), "coverage": cov, "source": "cache"}

    # Monthly Vision zips (large, preferred)
    vision_months = 0
    for y, m in _month_range(start_ts, end_ts):
        df = fetch_vision_month(sym, "1s", y, m, session=sess)
        if not df.empty:
            frames.append(df)
            vision_months += 1
            logger.info("%s 1s vision month %04d-%02d rows=%d", sym, y, m, len(df))
        time.sleep(0.05)

    merged = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("timestamp", keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
        if frames
        else _empty()
    )
    cov = coverage_ratio(merged, start_ts, end_ts, "1s")

    # Daily Vision for sparse gaps / current month
    if cov < min_coverage:
        vision_days = 0
        for day in _day_range(start_ts, end_ts):
            # Skip days already well covered
            day_end = day + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
            day_cov = coverage_ratio(merged, day, day_end, "1s")
            if day_cov >= min_coverage:
                continue
            df = fetch_vision_day(sym, "1s", day, session=sess)
            if not df.empty:
                frames.append(df)
                vision_days += 1
            time.sleep(0.03)
        if vision_days:
            merged = (
                pd.concat(frames, ignore_index=True)
                .drop_duplicates("timestamp", keep="last")
                .sort_values("timestamp")
                .reset_index(drop=True)
            )
            cov = coverage_ratio(merged, start_ts, end_ts, "1s")
            logger.info("%s 1s after daily vision coverage=%.1f%% days_fetched=%d", sym, 100 * cov, vision_days)

    # API fill remaining gaps (esp. today / last hours)
    if cov < min_coverage:
        try:
            api_df = fetch_klines_api(sym, "1s", start_ts, end_ts, session=sess, sleep_s=0.02)
            if not api_df.empty:
                frames.append(api_df)
                merged = (
                    pd.concat(frames, ignore_index=True)
                    .drop_duplicates("timestamp", keep="last")
                    .sort_values("timestamp")
                    .reset_index(drop=True)
                )
                cov = coverage_ratio(merged, start_ts, end_ts, "1s")
                logger.info("%s 1s after API fill coverage=%.1f%%", sym, 100 * cov)
        except Exception as e:
            logger.warning("%s 1s API fill failed: %s", sym, e)

    if merged.empty:
        return {"ok": False, "n": 0, "coverage": 0.0, "source": "none"}

    pad_start = start_ts - pd.Timedelta(hours=1)
    pad_end = end_ts + pd.Timedelta(hours=1)
    merged = merged[(merged["timestamp"] >= pad_start) & (merged["timestamp"] <= pad_end)].reset_index(drop=True)
    save_candles_partitioned_1s(merged, candle_dir, sym)
    # Compatibility single-file pointer (optional small sample not needed)
    return {
        "ok": cov >= min_coverage * 0.8,
        "n": int(len(merged)),
        "coverage": float(cov),
        "source": "vision+api",
        "first": str(merged["timestamp"].iloc[0]),
        "last": str(merged["timestamp"].iloc[-1]),
        "vision_months": vision_months,
    }


def download_universe(
    symbols: list[str],
    *,
    start: Any,
    end: Any,
    interval: str,
    candle_dir: Path | str,
    session: requests.Session | None = None,
    prefer_vision: bool = True,
    min_coverage: float = 0.90,
) -> dict[str, dict[str, Any]]:
    """Download OHLCV for many symbols into local candle cache."""
    sess = session or _session()
    start_ts = _as_utc(start)
    end_ts = _as_utc(end)
    root = Path(candle_dir)
    root.mkdir(parents=True, exist_ok=True)
    report: dict[str, dict[str, Any]] = {}

    if interval == "1s":
        for i, symbol in enumerate(symbols, 1):
            rep = download_symbol_1s(
                symbol, start_ts, end_ts, root, session=sess, min_coverage=min_coverage
            )
            report[str(symbol).upper()] = rep
            logger.info(
                "1s %d/%d %s ok=%s n=%s cov=%.1f%%",
                i,
                len(symbols),
                symbol,
                rep.get("ok"),
                rep.get("n"),
                100 * float(rep.get("coverage") or 0.0),
            )
        return report

    months = _month_range(start_ts, end_ts) if prefer_vision else []
    for i, symbol in enumerate(symbols, 1):
        sym = str(symbol).upper()
        path = candle_path(root, sym, interval)
        frames: list[pd.DataFrame] = []
        cached = load_candles(path)
        if cached is not None and not cached.empty:
            frames.append(cached)
            if coverage_ratio(cached, start_ts, end_ts, interval) >= min_coverage:
                report[sym] = {
                    "ok": True,
                    "n": int(len(cached)),
                    "path": str(path),
                    "coverage": coverage_ratio(cached, start_ts, end_ts, interval),
                    "source": "cache",
                }
                continue

        vision_ok = 0
        if prefer_vision:
            for y, m in months:
                df = fetch_vision_month(sym, interval, y, m, session=sess)
                if not df.empty:
                    frames.append(df)
                    vision_ok += 1
                time.sleep(0.05)

        try:
            api_df = fetch_klines_api(sym, interval, start_ts, end_ts, session=sess, sleep_s=0.04)
            if not api_df.empty:
                frames.append(api_df)
        except Exception as e:
            logger.warning("%s %s API fill failed: %s", sym, interval, e)

        if not frames:
            report[sym] = {"ok": False, "n": 0, "path": str(path), "vision_months": vision_ok}
            logger.error("No data for %s %s", sym, interval)
            continue

        out = (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates("timestamp", keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        pad_start = start_ts - pd.Timedelta(days=2)
        pad_end = end_ts + pd.Timedelta(days=1)
        out = out[(out["timestamp"] >= pad_start) & (out["timestamp"] <= pad_end)].reset_index(drop=True)
        save_candles(out, path)
        cov = coverage_ratio(out, start_ts, end_ts, interval)
        report[sym] = {
            "ok": cov > 0,
            "n": int(len(out)),
            "coverage": cov,
            "path": str(path),
            "vision_months": vision_ok,
            "first": str(out["timestamp"].iloc[0]) if not out.empty else None,
            "last": str(out["timestamp"].iloc[-1]) if not out.empty else None,
        }
        if i % 5 == 0 or i == len(symbols):
            logger.info("download_universe %s %d/%d %s n=%d cov=%.1f%%", interval, i, len(symbols), sym, len(out), 100 * cov)
    return report
