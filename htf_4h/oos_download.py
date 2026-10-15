"""HTF4H out-of-sample data: Binance Vision monthly 1m spot klines, 2025-02 .. 2026-02.

Stored separately from the research cache so the in-sample data stays untouched:
  data/oos_2025_candles/1m/{SYMBOL}.parquet  (same columns as the main candle store)

    python -m htf_4h.oos_download
"""

from __future__ import annotations

import io
import logging
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

from btcc.data.candles import COLS, candle_path, save_candles
from htf_4h.common import REPO, bases

OOS_DIR = REPO / "data" / "oos_2025_candles"
MONTHS = [m.strftime("%Y-%m") for m in pd.date_range("2025-02-01", "2026-02-01", freq="MS")]
URL = "https://data.binance.vision/data/spot/monthly/klines/{s}/1m/{s}-1m-{m}.zip"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("htf4h.oos_download")


def _month(sess: requests.Session, symbol: str, month: str) -> pd.DataFrame | None:
    r = sess.get(URL.format(s=symbol, m=month), timeout=180)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        df = pd.read_csv(zf.open(zf.namelist()[0]), header=None, usecols=range(6))
    df = df[pd.to_numeric(df[0], errors="coerce").notna()].astype(float)
    ts = df[0].astype("int64")
    ts = ts.where(ts < 10**14, ts // 1000)
    out = pd.DataFrame({"timestamp": pd.to_datetime(ts, unit="ms", utc=True)})
    for k, c in zip(range(1, 6), COLS[1:]):
        out[c] = df[k].to_numpy()
    return out


def download_symbol(symbol: str) -> tuple[str, int, list[str]]:
    path = candle_path(OOS_DIR, symbol, "1m")
    if path.exists():
        return symbol, -1, []
    sess = requests.Session()
    sess.headers.update({"User-Agent": "BTCC-research-htf4h/0.1"})
    parts, missing = [], []
    for m in MONTHS:
        df = _month(sess, symbol, m)
        if df is None:
            missing.append(m)
        else:
            parts.append(df)
    if parts:
        save_candles(pd.concat(parts, ignore_index=True), path)
    return symbol, sum(len(p) for p in parts), missing


def extend_symbol(symbol: str, months: list[str]) -> tuple[str, int, list[str]]:
    """Add extra months to an existing OOS file (e.g. warm-up months before 2025-02)."""
    path = candle_path(OOS_DIR, symbol, "1m")
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=COLS)
    have = set(pd.to_datetime(old["timestamp"], utc=True).dt.strftime("%Y-%m")) if len(old) else set()
    sess = requests.Session()
    sess.headers.update({"User-Agent": "BTCC-research-htf4h/0.1"})
    parts, missing = [old], []
    for m in months:
        if m in have:
            continue
        df = _month(sess, symbol, m)
        if df is None:
            missing.append(m)
        else:
            parts.append(df)
    if len(parts) > 1:
        save_candles(pd.concat(parts, ignore_index=True), path)
    return symbol, sum(len(p) for p in parts[1:]), missing


def main() -> int:
    import sys

    extra = sys.argv[1:]
    symbols = ["BTCUSDT"] + [f"{b}USDT" for b in bases()]
    report = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        if extra:
            futs = [ex.submit(extend_symbol, s, extra) for s in symbols]
        else:
            futs = [ex.submit(download_symbol, s) for s in symbols]
        for fut in as_completed(futs):
            s, n, missing = fut.result()
            report[s] = (n, missing)
            log.info("%s rows=%s missing_months=%s", s, "cached" if n < 0 else n, missing or "-")
    short = {s: v[1] for s, v in report.items() if v[1]}
    print(f"symbols={len(symbols)}  with missing months={len(short)}")
    for s, m in sorted(short.items()):
        print(f"  {s}: missing {len(m)} months ({m[0]} .. {m[-1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
