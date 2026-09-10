"""Local OHLCV candle cache (parquet preferred, CSV fallback)."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

PathLike = Union[str, Path]

OHLCV_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


def candle_path(candle_dir: PathLike, symbol: str, interval: str) -> Path:
    """Return ``{candle_dir}/{interval}/{SYMBOL}.parquet``."""
    root = Path(candle_dir)
    return root / str(interval) / f"{str(symbol).upper()}.parquet"


def load_candles(path: PathLike) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        # Allow CSV sibling written by older runs
        csv = p.with_suffix(".csv")
        if csv.exists():
            p = csv
        else:
            return None
    try:
        if p.suffix.lower() == ".parquet":
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    out = df.copy()
    if "timestamp" not in out.columns:
        return None
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = (
        out.dropna(subset=["timestamp", "close"])
        .drop_duplicates("timestamp", keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return out


def save_candles(df: pd.DataFrame, path: PathLike) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    keep = [c for c in OHLCV_COLS if c in out.columns]
    out = (
        out[keep]
        .drop_duplicates("timestamp", keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    out.to_parquet(p, index=False)
    return p
