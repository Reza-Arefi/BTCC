"""Public REST/websocket helpers (MEXC-compatible stubs for offline Binance runs)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from btcc.data.candles import candle_path, load_candles, save_candles

logger = logging.getLogger(__name__)

INTERVAL_MS: dict[str, int] = {
    "1s": 1_000,
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


class MexcPublicREST:
    """Minimal MEXC public REST used by live signal path / online backtests.

    Binance multi-arm tools use ``offline_candles=True`` after pre-download, so
    this class is only needed for import compatibility and optional online fills.
    """

    def __init__(self, base_url: str = "https://api.mexc.com"):
        self.base_url = str(base_url).rstrip("/")

    def bootstrap_symbol(
        self,
        symbol: str,
        interval: str,
        lookback: int,
        candle_dir: str | Path,
        force: bool = False,
    ) -> pd.DataFrame | None:
        path = candle_path(candle_dir, symbol, interval)
        if not force:
            cached = load_candles(path)
            if cached is not None and not cached.empty:
                if len(cached) >= max(10, int(lookback * 0.5)):
                    return cached
        logger.warning(
            "MexcPublicREST.bootstrap_symbol online fetch not implemented here; "
            "missing cache for %s %s at %s",
            symbol,
            interval,
            path,
        )
        return load_candles(path)
