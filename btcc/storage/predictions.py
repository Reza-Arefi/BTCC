"""Historical prediction store + outcome backfill."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from btcc.series.relative import horizon_bars

logger = logging.getLogger(__name__)

PRED_COLS = [
    "timestamp", "symbol",
    "probability_1h", "probability_4h", "probability_8h", "probability_12h", "probability_24h",
    "signal_score",
    "momentum_score", "trend_score", "btc_regime_score", "volume_score",
    "volatility_score", "rsi_score", "structure_score",
    "late_entry_score", "late_entry_class",
    "alt_btc_price", "btc_price", "btc_dominance",
    "future_rel_return_1h", "future_rel_return_4h", "future_rel_return_8h",
    "future_rel_return_12h", "future_rel_return_24h",
    "future_hit_1h", "future_hit_4h", "future_hit_8h", "future_hit_12h", "future_hit_24h",
]


class PredictionStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.df = pd.read_csv(self.path)
            if "timestamp" in self.df.columns:
                self.df["timestamp"] = pd.to_datetime(self.df["timestamp"], utc=True)
        else:
            self.df = pd.DataFrame(columns=PRED_COLS)

    def append(self, row: dict[str, Any]) -> None:
        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        self.save()

    def append_many(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        self.df = pd.concat([self.df, pd.DataFrame(rows)], ignore_index=True)
        self.save()

    def save(self) -> None:
        self.df.to_csv(self.path, index=False)

    def backfill_outcomes(
        self,
        panels: dict[str, pd.DataFrame],
        interval: str = "15m",
    ) -> int:
        """Fill future relative returns when enough bars have elapsed. No lookahead at prediction time."""
        if self.df.empty:
            return 0
        updated = 0
        horizons = [1, 4, 8, 12, 24]
        for i, row in self.df.iterrows():
            sym = row["symbol"]
            if sym not in panels:
                continue
            df = panels[sym]
            ts = pd.Timestamp(row["timestamp"])
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            # find prediction bar index
            idx = df.index[df["timestamp"] == ts]
            if len(idx) == 0:
                # nearest at or before
                prev = df[df["timestamp"] <= ts]
                if prev.empty:
                    continue
                i0 = prev.index[-1]
            else:
                i0 = idx[0]
            pos = df.index.get_loc(i0)
            if isinstance(pos, slice):
                continue
            px0 = float(df.loc[i0, "close"])
            for h in horizons:
                col = f"future_rel_return_{h}h"
                if pd.notna(row.get(col)):
                    continue
                bars = horizon_bars(h, interval)
                j = pos + bars
                if j >= len(df):
                    continue
                px1 = float(df.iloc[j]["close"])
                ret = px1 / px0 - 1.0 if px0 else None
                self.df.at[i, col] = ret
                self.df.at[i, f"future_hit_{h}h"] = int(ret > 0) if ret is not None else None
                updated += 1
        if updated:
            self.save()
            logger.info("Backfilled %d outcome fields", updated)
        return updated
