"""Data / BTC.D health checks before opening new simulated opportunities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass
class HealthReport:
    ok: bool
    allow_new_trades: bool
    reasons: list[str]
    btc_d_value: float | None = None
    btc_d_timestamp: str | None = None
    btc_d_age_seconds: float | None = None
    btc_d_available: bool = False
    btc_d_source: str | None = None
    btc_d_status: str = "UNKNOWN"
    candle_age_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "allow_new_trades": self.allow_new_trades,
            "reasons": list(self.reasons),
            "btc_d_value": self.btc_d_value,
            "btc_d_timestamp": self.btc_d_timestamp,
            "btc_d_age_seconds": self.btc_d_age_seconds,
            "btc_d_available": self.btc_d_available,
            "btc_d_source": self.btc_d_source,
            "btc_d_status": self.btc_d_status,
            "candle_age_seconds": self.candle_age_seconds,
        }


def evaluate_health(
    *,
    sim_cfg: dict[str, Any],
    dominance_pct: float | None,
    dominance_ts: datetime | None,
    dominance_source: str | None,
    decision_candle_ts: datetime | None,
    now: datetime | None = None,
    n_relative_bars: int | None = None,
    indicator_ok: bool = True,
) -> HealthReport:
    now = now or datetime.now(timezone.utc)
    reasons: list[str] = []
    bd = sim_cfg.get("btc_d_health") or {}
    dh = sim_cfg.get("data_health") or {}

    # Candle freshness
    candle_age = None
    if decision_candle_ts is not None:
        ts = decision_candle_ts
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        candle_age = (now - ts).total_seconds()
        max_age = float(dh.get("max_candle_age_seconds", 1200))
        if candle_age > max_age:
            reasons.append(f"STALE_CANDLES age_s={candle_age:.0f}>{max_age:.0f}")

    min_bars = int(dh.get("min_relative_bars", 200))
    if n_relative_bars is not None and n_relative_bars < min_bars:
        reasons.append(f"SHORT_HISTORY bars={n_relative_bars}<{min_bars}")

    if not indicator_ok:
        reasons.append("INDICATOR_CALC_FAILED")

    # BTC.D
    btc_d_available = False
    btc_d_status = "MISSING"
    btc_d_age = None
    btc_d_ts_str = None
    if dominance_pct is None:
        reasons.append("BTC_D_UNAVAILABLE")
        btc_d_status = "UNAVAILABLE"
    else:
        lo = float(bd.get("min_valid_pct", 20.0))
        hi = float(bd.get("max_valid_pct", 80.0))
        if not (lo <= float(dominance_pct) <= hi):
            reasons.append(f"BTC_D_OUT_OF_RANGE value={dominance_pct}")
            btc_d_status = "OUT_OF_RANGE"
        else:
            btc_d_available = True
            btc_d_status = "OK"
        if dominance_ts is not None:
            dts = dominance_ts if dominance_ts.tzinfo else dominance_ts.replace(tzinfo=timezone.utc)
            btc_d_age = (now - dts).total_seconds()
            btc_d_ts_str = dts.isoformat()
            max_bd_age = float(bd.get("max_age_seconds", 900))
            if btc_d_age > max_bd_age:
                reasons.append(f"BTC_D_STALE age_s={btc_d_age:.0f}>{max_bd_age:.0f}")
                btc_d_status = "STALE"
                btc_d_available = False

    require_bd = bool(bd.get("require_for_new_trades", True))
    allow = True
    block_reasons = [r for r in reasons if not r.startswith("SHORT_HISTORY")]
    # Short history alone can still allow research predictions but blocks trades if below min
    if any(r.startswith("STALE_CANDLES") for r in reasons):
        allow = False
    if any(r.startswith("INDICATOR") for r in reasons):
        allow = False
    if require_bd and not btc_d_available:
        allow = False
    if any(r.startswith("SHORT_HISTORY") for r in reasons):
        allow = False

    ok = len(reasons) == 0
    return HealthReport(
        ok=ok,
        allow_new_trades=allow,
        reasons=reasons or (["OK"] if ok else block_reasons),
        btc_d_value=float(dominance_pct) if dominance_pct is not None else None,
        btc_d_timestamp=btc_d_ts_str,
        btc_d_age_seconds=btc_d_age,
        btc_d_available=btc_d_available,
        btc_d_source=dominance_source,
        btc_d_status=btc_d_status,
        candle_age_seconds=candle_age,
    )
