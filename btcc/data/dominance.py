"""Live BTC.D feed — SAME relative top-N definition as backtest.

Formula:
  BTC.D_relative = 100 * BTC_mcap / sum(available top-N mcaps)

Source: CoinGecko `/coins/markets` for the fixed TOP_COIN_IDS list.
Not CoinGecko `/global` absolute dominance (that was the live/backtest mismatch).

Timestamp rule (no future leakage):
  Snapshots are stamped with the closed decision-candle timestamp (`as_of`)
  when provided by the cycle. Feature resolution uses last observation with
  timestamp <= decision_ts (identical to HistoricalDominanceSeries.observation_at).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from btcc.data.relative_btc_d import (
    MIN_COINS_FOR_VALID,
    RELATIVE_SOURCE,
    TOP_COIN_IDS,
    compute_relative_btc_d_pct,
)

logger = logging.getLogger(__name__)


@dataclass
class DominanceSnapshot:
    timestamp: datetime
    btc_dominance_pct: float
    source: str
    raw: dict[str, Any]


class DominanceFeed:
    def __init__(
        self,
        url: str = "",
        source_name: str = RELATIVE_SOURCE,
        poll_seconds: int = 300,
        history_path: str | Path | None = None,
        markets_url: str = "https://api.coingecko.com/api/v3/coins/markets",
        min_coins: int = MIN_COINS_FOR_VALID,
    ):
        # `url` kept for config compat; live relative uses markets_url.
        self.url = url
        self.markets_url = markets_url
        self.source_name = source_name or RELATIVE_SOURCE
        self.poll_seconds = poll_seconds
        self.min_coins = min_coins
        self.history: list[DominanceSnapshot] = []
        self._last_fetch = 0.0
        self.history_path = Path(history_path) if history_path else None
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "BTCC-signal/0.1"})
        self.last_meta: dict[str, Any] = {}
        self._load_history()

    def _load_history(self) -> None:
        if not self.history_path or not self.history_path.exists():
            return
        try:
            rows = json.loads(self.history_path.read_text(encoding="utf-8"))
            for r in rows:
                src = r.get("source", self.source_name)
                # Ignore legacy absolute /global snapshots — different definition
                if src in ("coingecko_global", "coingecko"):
                    continue
                if src != RELATIVE_SOURCE and "relative" not in str(src):
                    # Keep only relative-compatible rows
                    if r.get("representation") != "relative_btc_share_of_top_n":
                        continue
                self.history.append(
                    DominanceSnapshot(
                        timestamp=datetime.fromisoformat(r["timestamp"]),
                        btc_dominance_pct=float(r["btc_dominance_pct"]),
                        source=src,
                        raw=r.get("raw", {}),
                    )
                )
            logger.info(
                "Loaded %d relative BTC.D snapshots from %s",
                len(self.history),
                self.history_path,
            )
        except Exception as e:
            logger.warning("Could not load dominance history: %s", e)

    def _save_history(self) -> None:
        if not self.history_path:
            return
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        payload = []
        for s in self.history[-2000:]:
            payload.append({
                "timestamp": s.timestamp.isoformat(),
                "btc_dominance_pct": s.btc_dominance_pct,
                "source": s.source,
                "representation": "relative_btc_share_of_top_n",
                "calibration": "none_no_present_day_scaling",
                "raw": s.raw,
            })
        self.history_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _fetch_market_caps(self) -> dict[str, float] | None:
        """One CoinGecko markets call for the fixed top-N id list."""
        ids = ",".join(TOP_COIN_IDS)
        last_err: Exception | None = None
        for attempt in range(4):
            try:
                r = self.session.get(
                    self.markets_url,
                    params={
                        "vs_currency": "usd",
                        "ids": ids,
                        "per_page": len(TOP_COIN_IDS),
                        "page": 1,
                    },
                    timeout=30,
                )
                if r.status_code == 429:
                    time.sleep(min(60.0, 5.0 * (2 ** attempt)))
                    continue
                r.raise_for_status()
                rows = r.json()
                if not isinstance(rows, list):
                    return None
                caps: dict[str, float] = {}
                for row in rows:
                    cid = row.get("id")
                    mc = row.get("market_cap")
                    if cid and mc is not None:
                        caps[str(cid)] = float(mc)
                return caps
            except requests.RequestException as e:
                last_err = e
                time.sleep(min(30.0, 2.0 * (2 ** attempt)))
        logger.error("Relative BTC.D markets fetch failed: %s", last_err)
        return None

    def fetch(
        self,
        force: bool = False,
        as_of: datetime | None = None,
    ) -> DominanceSnapshot | None:
        """Fetch relative BTC.D; stamp with as_of (decision candle) when provided."""
        now = time.time()
        if not force and self.history and (now - self._last_fetch) < self.poll_seconds:
            return self.observation_at(as_of)[0] if as_of else self.history[-1]

        caps = self._fetch_market_caps()
        if caps is None:
            self.last_meta = {"status": "API_FAILURE"}
            # Fall back to last usable observation
            if as_of is not None:
                snap, _obs, _st = self.observation_at(as_of)
                return snap
            return self.history[-1] if self.history else None

        pct, meta = compute_relative_btc_d_pct(caps, min_coins=self.min_coins)
        self.last_meta = meta
        if pct is None:
            logger.error("Relative BTC.D invalid: %s", meta.get("status"))
            if as_of is not None:
                snap, _, _ = self.observation_at(as_of)
                return snap
            return self.history[-1] if self.history else None

        if as_of is not None:
            ts = as_of if as_of.tzinfo else as_of.replace(tzinfo=timezone.utc)
            # Never stamp in the future beyond wall clock
            wall = datetime.now(timezone.utc)
            if ts > wall:
                ts = wall
        else:
            ts = datetime.now(timezone.utc)

        snap = DominanceSnapshot(
            timestamp=ts,
            btc_dominance_pct=float(pct),
            source=RELATIVE_SOURCE,
            raw={
                "representation": "relative_btc_share_of_top_n",
                "calibration": "none_no_present_day_scaling",
                "n_coins_used": meta.get("n_coins_used"),
                "coins_used": meta.get("coins_used"),
                "btc_market_cap": meta.get("btc_market_cap"),
                "total_top_n_cap": meta.get("total_top_n_cap"),
            },
        )
        # Replace same decision-candle stamp; else append
        if self.history and self.history[-1].timestamp == snap.timestamp:
            self.history[-1] = snap
        elif self.history and abs((snap.timestamp - self.history[-1].timestamp).total_seconds()) < 1:
            self.history[-1] = snap
        else:
            self.history.append(snap)
        self._last_fetch = now
        self._save_history()
        logger.info(
            "BTC_DOMINANCE source=%s value=%.4f%% ts=%s coins=%s role=SLOW_MACRO_FACTOR relative_proxy",
            RELATIVE_SOURCE,
            pct,
            snap.timestamp.isoformat(),
            meta.get("n_coins_used"),
        )
        return snap

    def observation_at(
        self, ts: datetime | pd.Timestamp | None
    ) -> tuple[DominanceSnapshot | None, datetime | None, str]:
        """Last snapshot with timestamp <= ts. Never uses a future observation."""
        if ts is None:
            if not self.history:
                return None, None, "INSUFFICIENT_DATA"
            s = self.history[-1]
            return s, s.timestamp, "OK"
        t = pd.Timestamp(ts)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        else:
            t = t.tz_convert("UTC")
        t_dt = t.to_pydatetime()
        prior = [s for s in self.history if s.timestamp <= t_dt]
        if not prior:
            return None, None, "INSUFFICIENT_DATA"
        s = prior[-1]
        return s, s.timestamp, "OK"

    def change(self, hours: float, as_of: datetime | None = None) -> tuple[float | None, str]:
        """Return (change_pp, status) using last-known <= as_of (or latest)."""
        if as_of is not None:
            latest, latest_ts, st = self.observation_at(as_of)
            if st != "OK" or latest is None or latest_ts is None:
                return None, "INSUFFICIENT_DATA"
        else:
            if len(self.history) < 2:
                return None, "INSUFFICIENT_DATA"
            latest = self.history[-1]
            latest_ts = latest.timestamp

        target = latest_ts.timestamp() - hours * 3600
        prior = None
        for s in reversed(self.history):
            if s.timestamp.timestamp() <= target and s.timestamp < latest_ts:
                prior = s
                break
        if prior is None:
            return None, "INSUFFICIENT_DATA"
        age_ok = (latest_ts.timestamp() - prior.timestamp.timestamp()) >= hours * 3600 * 0.9
        if not age_ok:
            return None, "INSUFFICIENT_DATA"
        return latest.btc_dominance_pct - prior.btc_dominance_pct, "OK"

    def summary(self) -> dict[str, Any]:
        snap = self.history[-1] if self.history else None
        c4, s4 = self.change(4)
        c24, s24 = self.change(24)
        return {
            "source": self.source_name,
            "representation": "relative_btc_share_of_top_n",
            "calibration": "none_no_present_day_scaling",
            "role": "slow_moving_macro_factor",
            "btc_dominance_pct": snap.btc_dominance_pct if snap else None,
            "asof": snap.timestamp.isoformat() if snap else None,
            "n_snapshots": len(self.history),
            "change_4h_pp": c4,
            "change_4h_status": s4,
            "change_24h_pp": c24,
            "change_24h_status": s24,
            "last_meta": self.last_meta,
        }
