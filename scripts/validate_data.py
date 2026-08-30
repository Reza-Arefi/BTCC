"""Validate backtest candle / BTC.D coverage for Adaptive V2."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def validate_candles(candle_dir: Path, interval: str = "15m") -> dict:
    d = candle_dir / interval
    files = sorted(d.glob("*.parquet"))
    rows = []
    for f in files:
        df = pd.read_parquet(f)
        ts = pd.to_datetime(df["timestamp"], utc=True)
        gaps = int((ts.sort_values().diff() > pd.Timedelta(minutes=20)).sum())
        bad = 0
        if set(["open", "high", "low", "close"]) <= set(df.columns):
            bad = int(
                (
                    (df["high"] < df[["open", "close"]].max(axis=1))
                    | (df["low"] > df[["open", "close"]].min(axis=1))
                    | (df["open"] <= 0)
                    | (df["close"] <= 0)
                ).sum()
            )
        zero_vol = int((df["volume"] <= 0).sum()) if "volume" in df.columns else None
        rows.append({
            "pair": f.stem,
            "first_timestamp": str(ts.min()),
            "last_timestamp": str(ts.max()),
            "calendar_days": round((ts.max() - ts.min()).total_seconds() / 86400, 2),
            "candles": len(df),
            "gaps": gaps,
            "duplicates": int(ts.duplicated().sum()),
            "bad_ohlc": bad,
            "zero_volume": zero_vol,
        })
    out = {"n_files": len(rows), "pairs": rows}
    if rows:
        out["global_first"] = min(r["first_timestamp"] for r in rows)
        out["global_last"] = max(r["last_timestamp"] for r in rows)
        out["min_pair_calendar_days"] = min(r["calendar_days"] for r in rows)
        out["min_pair_candles"] = min(r["candles"] for r in rows)
        out["max_pair_calendar_days"] = max(r["calendar_days"] for r in rows)
    return out


def validate_dominance(cache_dir: Path, days: int) -> dict:
    path = cache_dir / f"btc_dominance_{days}d.parquet"
    meta_path = cache_dir / f"btc_dominance_{days}d_meta.json"
    if not path.exists():
        return {"exists": False, "path": str(path)}
    df = pd.read_parquet(path)
    ts = pd.to_datetime(df["timestamp"], utc=True)
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    spacing_h = None
    if len(ts) >= 2:
        spacing_h = float(ts.diff().dropna().dt.total_seconds().median() / 3600.0)
    resolution = "daily" if spacing_h and spacing_h >= 20 else ("hourly" if spacing_h and spacing_h <= 2 else "mixed")
    return {
        "exists": True,
        "n_points": len(df),
        "first_timestamp": str(ts.min()),
        "last_timestamp": str(ts.max()),
        "calendar_days": round((ts.max() - ts.min()).total_seconds() / 86400, 2),
        "duplicates": int(ts.duplicated().sum()),
        "median_spacing_hours": spacing_h,
        "resolution": resolution,
        "source": meta.get("source"),
        "representation": meta.get("representation"),
        "calibration": meta.get("calibration"),
        "present_day_scale": str(meta.get("calibration", "")).startswith("scaled_to_global"),
    }


def print_candle_table(candles: dict) -> None:
    print("\nPAIR\tFIRST_TIMESTAMP\tLAST_TIMESTAMP\tCALENDAR_DAYS\tCANDLES\tGAPS\tDUPLICATES\tBAD_OHLC\tZERO_VOLUME")
    for r in sorted(candles.get("pairs") or [], key=lambda x: x["pair"]):
        print(
            f"{r['pair']}\t{r['first_timestamp'][:19]}\t{r['last_timestamp'][:19]}\t"
            f"{r['calendar_days']}\t{r['candles']}\t{r['gaps']}\t{r['duplicates']}\t"
            f"{r['bad_ohlc']}\t{r['zero_volume']}"
        )
    if candles.get("global_first"):
        print(
            f"\nUSABLE_PERIOD: {candles['global_first'][:19]} → {candles['global_last'][:19]} "
            f"(min pair {candles.get('min_pair_calendar_days')}d, "
            f"min candles {candles.get('min_pair_candles')})"
        )


def main() -> int:
    min_days = int(sys.argv[1]) if len(sys.argv) > 1 else 360
    candles = validate_candles(ROOT / "data" / "backtest_candles")
    print_candle_table(candles)
    print(json.dumps({"candles_summary": {
        k: candles.get(k) for k in (
            "n_files", "global_first", "global_last",
            "min_pair_calendar_days", "min_pair_candles", "max_pair_calendar_days",
        )
    }}, indent=2))
    for d in (90, 365):
        dom = validate_dominance(ROOT / "data" / "backtest_dominance", d)
        print(json.dumps({"dominance_days": d, **dom}, indent=2))
    ok = (
        candles.get("min_pair_calendar_days", 0) >= min_days
        and candles.get("min_pair_candles", 0) >= min_days * 96 - 200
    )
    print("READY_FOR_1Y_CANDLES" if ok else "NOT_READY_CANDLES")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
