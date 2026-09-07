"""Per-position / per-symbol T1 price monitors for bounded multi-position sessions.

Each open position gets an independent T1PriceMonitor (or shares a symbol monitor
that fans ticks out to every position on that symbol). Failures on one monitor
must not silently drop protection for others — caller HALTs on start failure.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

from btcc.execution.price_monitor import (
    DEFAULT_MAX_STALE_AGE_S,
    DEFAULT_POLL_INTERVAL_S,
    PriceSource,
    PriceTick,
    T1PriceMonitor,
)

logger = logging.getLogger(__name__)

OnPositionTick = Callable[[str, PriceTick], None]  # position_id, tick
OnSymbolStale = Callable[[str, float], None]
OnSymbolReconnect = Callable[[str, int], None]


class MultiPositionMonitorRegistry:
    """Own one T1PriceMonitor per symbol; map position_id → symbol."""

    def __init__(
        self,
        *,
        source_factory: Callable[[], PriceSource],
        on_position_tick: OnPositionTick,
        on_stale: OnSymbolStale | None = None,
        on_reconnect: OnSymbolReconnect | None = None,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
        max_stale_age_s: float = DEFAULT_MAX_STALE_AGE_S,
        require_live_ticker: bool = False,
    ) -> None:
        self._source_factory = source_factory
        self._on_position_tick = on_position_tick
        self._on_stale = on_stale
        self._on_reconnect = on_reconnect
        self.poll_interval_s = float(poll_interval_s)
        self.max_stale_age_s = float(max_stale_age_s)
        self.require_live_ticker = bool(require_live_ticker)
        self._lock = threading.RLock()
        self._monitors: dict[str, T1PriceMonitor] = {}
        self._positions: dict[str, str] = {}  # position_id -> symbol
        self._symbol_positions: dict[str, set[str]] = {}

    def _normalize(self, symbol: str) -> str:
        return str(symbol).upper().replace("/", "").replace("-", "").replace("_", "")

    def _new_source(self, symbol: str) -> PriceSource:
        src = self._source_factory()
        if self.require_live_ticker:
            from btcc.execution.live_price_guard import assert_production_price_source

            assert_production_price_source(src, context=f"symbol={symbol}")
        return src

    def attach_position(self, position_id: str, symbol: str) -> T1PriceMonitor:
        sym = self._normalize(symbol)
        with self._lock:
            self._positions[position_id] = sym
            self._symbol_positions.setdefault(sym, set()).add(position_id)
            if sym in self._monitors:
                mon = self._monitors[sym]
                if mon._thread is None or not mon._thread.is_alive():
                    mon.start()
                return mon
            mon = T1PriceMonitor(
                source=self._new_source(sym),
                symbol=sym,
                # Default-arg capture avoids late-binding to the last created symbol.
                on_tick=lambda tick, s=sym: self._fanout(s, tick),
                on_stale=self._on_stale,
                on_reconnect=self._on_reconnect,
                poll_interval_s=self.poll_interval_s,
                max_stale_age_s=self.max_stale_age_s,
            )
            self._monitors[sym] = mon
            mon.start()
            return mon

    def _fanout(self, symbol: str, tick: PriceTick) -> None:
        from btcc.execution.live_price_guard import validate_tick_basic

        basic = validate_tick_basic(tick, expected_symbol=symbol)
        if not basic.ok:
            logger.error("monitor tick rejected symbol=%s reason=%s", symbol, basic.reason)
            if self._on_stale is not None:
                self._on_stale(symbol, float(self.max_stale_age_s))
            return
        with self._lock:
            pids = list(self._symbol_positions.get(symbol, set()))
        for pid in pids:
            try:
                self._on_position_tick(pid, tick)
            except Exception as e:  # noqa: BLE001
                logger.error("monitor fanout failed position=%s: %s", pid, e)

    def detach_position(self, position_id: str) -> None:
        with self._lock:
            sym = self._positions.pop(position_id, None)
            if not sym:
                return
            bucket = self._symbol_positions.get(sym)
            if bucket is not None:
                bucket.discard(position_id)
            if bucket:
                return
            self._symbol_positions.pop(sym, None)
            mon = self._monitors.pop(sym, None)
        if mon is not None:
            mon.stop()

    def stop_all(self) -> None:
        with self._lock:
            mons = list(self._monitors.values())
            self._monitors.clear()
            self._positions.clear()
            self._symbol_positions.clear()
        for mon in mons:
            try:
                mon.stop()
            except Exception as e:  # noqa: BLE001
                logger.warning("monitor stop failed: %s", e)

    def halt_all(self, reason: str) -> None:
        with self._lock:
            mons = list(self._monitors.values())
        for mon in mons:
            try:
                mon.halt(reason)
            except Exception:  # noqa: BLE001
                pass

    @property
    def active_symbols(self) -> list[str]:
        with self._lock:
            return sorted(self._monitors.keys())

    @property
    def position_count(self) -> int:
        with self._lock:
            return len(self._positions)

    def monitor_for(self, position_id: str) -> T1PriceMonitor | None:
        with self._lock:
            sym = self._positions.get(position_id)
            if not sym:
                return None
            return self._monitors.get(sym)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "symbols": {
                    s: {
                        "state": m.state.value,
                        "ticks": m.ticks_received,
                        "source_type": type(m.source).__name__,
                        "positions": sorted(self._symbol_positions.get(s, set())),
                    }
                    for s, m in self._monitors.items()
                },
                "positions": dict(self._positions),
                "require_live_ticker": self.require_live_ticker,
            }
