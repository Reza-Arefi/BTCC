"""How should the live bot build its 4h ALT/BTC bars?

Research built ALT/BTC on 1m bars and resampled to 4h. Live can only cheaply fetch coarser klines, and
``build_alt_btc`` takes high = alt_high / btc_low, so a ratio built from 4h legs has much wider highs/lows.
This compares S and NEW_CROSS signals from three live-feasible constructions against the research S:

  4h    ratio of 4h USDT klines              (1 request per leg)
  1h    ratio of 1h USDT klines -> 4h        (2 requests per leg for 400 bars)
  15m   ratio of 15m USDT klines -> 4h       (7 requests per leg for 400 bars)

    python -m htf_4h.check_live_bars
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from btcc.series.relative import build_alt_btc
from btcc_m.data.load import resample_ohlcv
from htf_4h.common import BAR, RESULTS, THRESHOLD, _load_1m, score_variant, weights
from htf_4h.step6_lookback import DIRS, HISTORY

START = pd.Timestamp("2025-09-01", tz="UTC")
END = pd.Timestamp("2026-09-24 23:00", tz="UTC")
LOAD_START = START - 420 * BAR
COINS = ("ZEC", "DASH", "UNI", "WLD", "NEAR", "FET", "SOL", "ETH", "FIL", "LTC", "AVAX", "RENDER")
OUT = RESULTS / "check_live_bars"


def _rel4(alt: pd.DataFrame, btc: pd.DataFrame, rule: str) -> pd.DataFrame:
    if rule == "1m":
        return resample_ohlcv(build_alt_btc(alt, btc), "4h")
    a, b = resample_ohlcv(alt, rule), resample_ohlcv(btc, rule)
    rel = build_alt_btc(a, b)
    return rel if rule == "4h" else resample_ohlcv(rel, "4h")


def _coin(base: str) -> pd.DataFrame:
    btc = _load_1m(DIRS, "BTCUSDT")
    alt = _load_1m(DIRS, f"{base}USDT")
    btc = btc[(btc["timestamp"] >= LOAD_START) & (btc["timestamp"] <= END)].reset_index(drop=True)
    alt = alt[(alt["timestamp"] >= LOAD_START) & (alt["timestamp"] <= END)].reset_index(drop=True)
    alt4 = resample_ohlcv(alt, "4h").set_index("timestamp")
    btc4 = resample_ohlcv(btc, "4h").set_index("timestamp")
    w = weights()
    out = {}
    for rule in ("1m", "15m", "1h", "4h"):
        r4 = _rel4(alt, btc, rule)
        r4 = r4[r4["timestamp"] + BAR <= END + pd.Timedelta(minutes=1)].reset_index(drop=True)
        idx = pd.DatetimeIndex(r4["timestamp"])
        a4 = alt4.reindex(idx).ffill().reset_index()
        b4 = btc4.reindex(idx).ffill().reset_index()
        vals = {}
        for i in range(HISTORY - 1, len(r4)):
            ts = r4["timestamp"].iloc[i]
            if ts < START:
                continue
            lo = i + 1 - HISTORY
            vals[ts] = score_variant(alt_btc=r4.iloc[lo:i + 1], alt_usdt=a4.iloc[lo:i + 1],
                                     btc_usdt=b4.iloc[lo:i + 1], weights=w, interval="15m", use_e2=True)["S"]
        out[rule] = pd.Series(vals)
    df = pd.DataFrame(out)
    df["base"] = base
    return df.rename_axis("timestamp").reset_index()


def _crosses(s: pd.Series) -> pd.Series:
    z = s >= THRESHOLD
    return z & ~z.shift(1, fill_value=False)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor() as ex:
        df = pd.concat(ex.map(_coin, COINS), ignore_index=True)
    df.to_parquet(OUT / "s_by_construction.parquet")
    rows = []
    for rule in ("15m", "1h", "4h"):
        d = (df[rule] - df["1m"]).abs()
        ref_x = df.groupby("base")["1m"].transform(_crosses)
        x = df.groupby("base")[rule].transform(_crosses)
        rows.append({
            "construction": rule, "mean_abs_dS": d.mean(), "p95_abs_dS": d.quantile(0.95), "max_abs_dS": d.max(),
            "research_signals": int(ref_x.sum()), "live_signals": int(x.sum()),
            "same_bar": int((ref_x & x).sum()),
            "precision": (ref_x & x).sum() / max(x.sum(), 1), "recall": (ref_x & x).sum() / max(ref_x.sum(), 1),
        })
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "summary.csv", index=False)
    print(res.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
