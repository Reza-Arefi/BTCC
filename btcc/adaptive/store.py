"""Adaptive prediction store — every symbol every cycle + delayed outcome labels."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from btcc.series.relative import horizon_bars

logger = logging.getLogger(__name__)

PRED_COLS = [
    "prediction_id", "timestamp", "symbol", "base", "rank",
    "model_version", "model_type", "data_source", "probability_kind",
    "signal_score",
    # individual indicators
    "indicator_momentum", "indicator_ema", "indicator_macd", "indicator_ichimoku",
    "indicator_adx", "indicator_btc_regime", "indicator_btc_dominance",
    "indicator_rvol", "indicator_bollinger", "indicator_atr", "indicator_natr",
    "indicator_rsi", "indicator_structure",
    # factors
    "factor_momentum", "factor_trend", "factor_btc_regime", "factor_volume",
    "factor_volatility", "factor_rsi", "factor_structure",
    # probabilities
    "probability_1h", "probability_4h", "probability_8h", "probability_12h", "probability_24h",
    # late entry
    "late_entry_score", "late_entry_class",
    # market
    "alt_btc_price", "btc_price", "alt_usdt_price", "btc_dominance", "btc_dominance_obs_ts",
    "btc_dominance_age_hours", "candle_timestamp", "data_age",
    # outcomes (filled later)
    "future_return_1h", "future_return_4h", "future_return_8h",
    "future_return_12h", "future_return_24h",
    "outperformed_1h", "outperformed_4h", "outperformed_8h",
    "outperformed_12h", "outperformed_24h",
]


class AdaptivePredictionStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.df = pd.read_csv(self.path)
            if "timestamp" in self.df.columns:
                self.df["timestamp"] = pd.to_datetime(self.df["timestamp"], utc=True)
        else:
            self.df = pd.DataFrame(columns=PRED_COLS)

    def save(self) -> None:
        self.df.to_csv(self.path, index=False)

    def append_rows(self, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0
        for r in rows:
            r.setdefault("prediction_id", str(uuid.uuid4()))
        if not self.df.empty and "timestamp" in self.df.columns and "symbol" in self.df.columns:
            keys = {
                (str(t), str(s))
                for t, s in zip(self.df["timestamp"].astype(str), self.df["symbol"].astype(str))
            }
            before = len(rows)
            rows = [r for r in rows if (str(r.get("timestamp")), str(r.get("symbol"))) not in keys]
            skipped = before - len(rows)
            if skipped:
                logger.info("Skipped %d duplicate adaptive predictions (timestamp,symbol)", skipped)
            if not rows:
                return 0
        self.df = pd.concat([self.df, pd.DataFrame(rows)], ignore_index=True)
        self.save()
        return len(rows)

    def backfill_outcomes(
        self,
        panels: dict[str, pd.DataFrame],
        interval: str = "15m",
        horizons: list[int] | None = None,
    ) -> int:
        """Attach future ALT/BTC returns only after the horizon has elapsed. No overwrite of predictions."""
        horizons = horizons or [1, 4, 8, 12, 24]
        if self.df.empty:
            return 0
        updated = 0
        for i, row in self.df.iterrows():
            sym = row["symbol"]
            if sym not in panels:
                continue
            df = panels[sym]
            ts = pd.Timestamp(row["timestamp"])
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            prev = df[df["timestamp"] <= ts]
            if prev.empty:
                continue
            i0 = prev.index[-1]
            pos = df.index.get_loc(i0)
            if isinstance(pos, slice):
                continue
            px0 = float(df.loc[i0, "close"])
            if px0 == 0:
                continue
            for h in horizons:
                rcol = f"future_return_{h}h"
                ocol = f"outperformed_{h}h"
                if pd.notna(row.get(rcol)):
                    continue
                bars = horizon_bars(h, interval)
                j = pos + bars
                if j >= len(df):
                    continue
                px1 = float(df.iloc[j]["close"])
                ret = px1 / px0 - 1.0
                self.df.at[i, rcol] = ret
                self.df.at[i, ocol] = int(ret > 0)
                updated += 1
        if updated:
            self.save()
            logger.info("Adaptive store backfilled %d outcome fields", updated)
        return updated

    def mature_count(self, horizon: int = 4) -> int:
        col = f"outperformed_{horizon}h"
        if col not in self.df.columns or self.df.empty:
            return 0
        return int(self.df[col].notna().sum())

    def labeled(self, horizon: int = 4) -> pd.DataFrame:
        pcol = f"probability_{horizon}h"
        ycol = f"outperformed_{horizon}h"
        rcol = f"future_return_{horizon}h"
        if self.df.empty:
            return self.df.copy()
        return self.df.dropna(subset=[pcol, ycol, rcol]).copy()
