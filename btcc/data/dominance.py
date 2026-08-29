"""BTC dominance — slow-moving macro factor (CoinGecko global).

MEXC has no native BTC.D. We NEVER invent or interpolate missing history.
If not enough real observations exist for a horizon change, return None /
INSUFFICIENT_DATA.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

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
        url: str,
        source_name: str = "coingecko_global",
        poll_seconds: int = 300,
        history_path: str | Path | None = None,
    ):
        self.url = url
        self.source_name = source_name
        self.poll_seconds = poll_seconds
        self.history: list[DominanceSnapshot] = []
        self._last_fetch = 0.0
        self.history_path = Path(history_path) if history_path else None
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "BTCC-signal/0.1"})
        self._load_history()

    def _load_history(self) -> None:
        if not self.history_path or not self.history_path.exists():
            return
        try:
            rows = json.loads(self.history_path.read_text(encoding="utf-8"))
            for r in rows:
                self.history.append(
                    DominanceSnapshot(
                        timestamp=datetime.fromisoformat(r["timestamp"]),
                        btc_dominance_pct=float(r["btc_dominance_pct"]),
                        source=r.get("source", self.source_name),
                        raw=r.get("raw", {}),
                    )
                )
            logger.info(
                "Loaded %d BTC.D snapshots from %s (source=%s)",
                len(self.history), self.history_path, self.source_name,
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
                "raw": s.raw,
            })
        self.history_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def fetch(self, force: bool = False) -> DominanceSnapshot | None:
        now = time.time()
        if not force and self.history and (now - self._last_fetch) < self.poll_seconds:
            return self.history[-1]
        try:
            r = self.session.get(self.url, timeout=20)
            r.raise_for_status()
            data = r.json()
            pct = float(data["data"]["market_cap_percentage"]["btc"])
            snap = DominanceSnapshot(
                timestamp=datetime.now(timezone.utc),
                btc_dominance_pct=pct,
                source=self.source_name,
                raw={"market_cap_percentage_btc": pct},
            )
            # Avoid duplicate near-identical timestamps
            if self.history and (snap.timestamp - self.history[-1].timestamp).total_seconds() < 60:
                self.history[-1] = snap
            else:
                self.history.append(snap)
            self._last_fetch = now
            self._save_history()
            logger.info(
                "BTC_DOMINANCE source=%s value=%.4f%% ts=%s role=SLOW_MACRO_FACTOR",
                self.source_name, pct, snap.timestamp.isoformat(),
            )
            return snap
        except Exception as e:
            logger.error("BTC dominance fetch failed (%s): %s", self.source_name, e)
            return self.history[-1] if self.history else None

    def change(self, hours: float) -> tuple[float | None, str]:
        """Return (change_pp, status). status=OK | INSUFFICIENT_DATA.

        Requires a real observation at or before (now - hours). Never uses a
        newer snapshot as a fake prior.
        """
        if len(self.history) < 2:
            return None, "INSUFFICIENT_DATA"
        latest = self.history[-1]
        target = latest.timestamp.timestamp() - hours * 3600
        prior = None
        for s in reversed(self.history[:-1]):
            if s.timestamp.timestamp() <= target:
                prior = s
                break
        if prior is None:
            return None, "INSUFFICIENT_DATA"
        # Also reject if prior is much newer than requested window (already handled)
        age_ok = (latest.timestamp.timestamp() - prior.timestamp.timestamp()) >= hours * 3600 * 0.9
        if not age_ok:
            return None, "INSUFFICIENT_DATA"
        return latest.btc_dominance_pct - prior.btc_dominance_pct, "OK"

    def summary(self) -> dict[str, Any]:
        snap = self.history[-1] if self.history else None
        c4, s4 = self.change(4)
        c24, s24 = self.change(24)
        return {
            "source": self.source_name,
            "role": "slow_moving_macro_factor",
            "btc_dominance_pct": snap.btc_dominance_pct if snap else None,
            "asof": snap.timestamp.isoformat() if snap else None,
            "n_snapshots": len(self.history),
            "change_4h_pp": c4,
            "change_4h_status": s4,
            "change_24h_pp": c24,
            "change_24h_status": s24,
        }
