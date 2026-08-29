"""Local OHLCV candle store (15m)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

COLS = ["timestamp", "open", "high", "low", "close", "volume"]


def candle_path(root: str | Path, symbol: str, interval: str) -> Path:
    d = Path(root) / interval
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{symbol}.parquet"


def save_candles(df: pd.DataFrame, path: Path) -> None:
    out = df[COLS].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out.sort_values("timestamp").drop_duplicates("timestamp")
    try:
        out.to_parquet(path, index=False)
    except Exception:
        out.to_csv(path.with_suffix(".csv"), index=False)


def load_candles(path: Path) -> pd.DataFrame | None:
    p = path if path.suffix else path.with_suffix(".parquet")
    if p.exists():
        try:
            df = pd.read_parquet(p)
        except Exception:
            df = None
        if df is not None:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            return df.sort_values("timestamp").reset_index(drop=True)
    csv = path.with_suffix(".csv")
    if csv.exists():
        df = pd.read_csv(csv)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df.sort_values("timestamp").reset_index(drop=True)
    return None


def upsert_candle(df: pd.DataFrame, row: dict) -> pd.DataFrame:
    ts = pd.Timestamp(row["timestamp"], tz="UTC")
    new = pd.DataFrame([{**row, "timestamp": ts}])
    if df is None or df.empty:
        return new
    out = pd.concat([df, new], ignore_index=True)
    return out.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
