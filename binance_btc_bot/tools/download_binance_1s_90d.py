"""Download Binance 15m + full-window 1s candles for the live universe (90d).

Caches under data/backtest_candles_binance/:
  - 15m/{SYMBOL}.parquet
  - 1s/{SYMBOL}/YYYY-MM-DD.parquet  (day-partitioned)

No backtest — download only. Safe to re-run (skips symbols with good coverage).
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from btcc.backtest.data_loader import compute_window  # noqa: E402
from btcc.data.binance_vision import download_universe  # noqa: E402

CANDLE_DIR = REPO / "data" / "backtest_candles_binance"
DAYS = 90
WARMUP_BARS = 1000
SIGNAL_INTERVAL = "15m"
EXIT_INTERVAL = "1s"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_1s_90d")


def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def bases() -> list[str]:
    cfg = yaml.safe_load((REPO / "binance_btc_bot" / "config" / "binance_bot.yaml").read_text(encoding="utf-8"))
    pairs = list((cfg.get("universe") or {}).get("btc_pairs") or [])
    out = []
    for p in pairs:
        s = str(p).upper()
        out.append(s[:-3] if s.endswith("BTC") else s)
    return out


def main() -> None:
    eval_end = _utc(datetime.now(timezone.utc)).floor("15min")
    eval_start = eval_end - pd.Timedelta(days=DAYS)
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
    symbols = ["BTCUSDT"] + [f"{b}USDT" for b in bases()]
    CANDLE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Window eval %s → %s | symbols=%d", eval_start, eval_end, len(symbols))
    logger.info("Downloading 15m…")
    r15 = download_universe(symbols, start=start_15, end=end, interval=SIGNAL_INTERVAL, candle_dir=CANDLE_DIR)
    logger.info("15m ok=%d/%d", sum(1 for v in r15.values() if v.get("ok")), len(r15))

    logger.info("Downloading full-window 1s (Vision monthly/daily + API fill)…")
    r1s = download_universe(
        symbols,
        start=start_1s,
        end=end,
        interval=EXIT_INTERVAL,
        candle_dir=CANDLE_DIR,
        prefer_vision=True,
        min_coverage=0.90,
    )
    ok = sum(1 for v in r1s.values() if v.get("ok"))
    mean_cov = sum(float(v.get("coverage") or 0.0) for v in r1s.values()) / max(1, len(r1s))
    logger.info("1s ok=%d/%d mean_coverage=%.1f%% → %s", ok, len(r1s), 100 * mean_cov, CANDLE_DIR / "1s")


if __name__ == "__main__":
    main()
