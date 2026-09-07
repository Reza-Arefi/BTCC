"""Live price monitoring for REAL T1 canary — independent of the 15m signal loop.

Bot-managed T1 uses this monitor (default 1s poll), NOT MEXC native trailing.
Public documented endpoint: GET /api/v3/ticker/price
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

SIGNAL_EVALUATION_INTERVAL_S = 900.0
DEFAULT_POLL_INTERVAL_S = 1.0
DEFAULT_MAX_STALE_AGE_S = 5.0
DEFAULT_RECONNECT_BASE_S = 0.5
DEFAULT_RECONNECT_MAX_S = 10.0


class PriceMonitorState(str, Enum):
    STOPPED = "STOPPED"
    RUNNING = "RUNNING"
    RECONNECTING = "RECONNECTING"
    STALE = "STALE"
    HALTED = "HALTED"


@dataclass(frozen=True)
class PriceTick:
    symbol: str
    price: float
    ts_monotonic: float
    ts_wall: float
    source: str = "ticker"


class PriceSource(Protocol):
    def fetch_price(self, symbol: str) -> PriceTick: ...


class MexcPublicTickerSource:
    """Live MEXC public ticker. Never fabricates a price on failure."""

    is_live_mexc_ticker = True
    is_production_forbidden = False

    def __init__(
        self,
        *,
        base_url: str = "https://api.mexc.com",
        timeout_s: float = 5.0,
        opener: Any | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self._opener = opener

    def fetch_price(self, symbol: str) -> PriceTick:
        sym = str(symbol).upper().replace("/", "").replace("-", "").replace("_", "")
        if not sym:
            raise RuntimeError("empty symbol for ticker fetch")
        url = f"{self.base_url}/api/v3/ticker/price?symbol={sym}"
        if self._opener is not None:
            raw = self._opener(url)
        else:
            req = Request(url, headers={"User-Agent": "BTCC-t1-price-monitor/0.1"}, method="GET")
            with urlopen(req, timeout=self.timeout_s) as resp:
                # HTTP errors raise; never synthesize a price.
                raw = resp.read().decode("utf-8")
        if raw is None or raw == "":
            raise RuntimeError(f"empty ticker body for {sym}")
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError as e:
            raise RuntimeError(f"malformed ticker JSON for {sym}: {e}") from e
        if not isinstance(data, dict):
            raise RuntimeError(f"malformed ticker response type for {sym}: {type(data).__name__}")
        if "price" not in data:
            raise RuntimeError(f"malformed ticker response for {sym}: missing price")
        # When exchange echoes symbol, it must match the requested market.
        if "symbol" in data and data["symbol"] is not None:
            got = str(data["symbol"]).upper().replace("/", "").replace("-", "").replace("_", "")
            if got and got != sym:
                raise RuntimeError(f"ticker symbol mismatch: requested={sym} got={got}")
        try:
            px = float(data["price"])
        except (TypeError, ValueError) as e:
            raise RuntimeError(f"non-numeric ticker price for {sym}: {data.get('price')!r}") from e
        if not math.isfinite(px) or px <= 0:
            raise RuntimeError(f"non-positive ticker price for {sym}: {px}")
        return PriceTick(
            symbol=sym,
            price=px,
            ts_monotonic=time.monotonic(),
            ts_wall=time.time(),
            source="ticker/price",
        )


class SequencePriceSource:
    """TEST/dry-run only. Hard-forbidden on armed REAL production protection."""

    is_live_mexc_ticker = False
    is_production_forbidden = True

    def __init__(self, prices: list[float] | Callable[[], float], *, symbol: str = "ETHBTC") -> None:
        self._symbol = symbol
        if callable(prices):
            self._fn: Callable[[], float] = prices
            self._seq: list[float] | None = None
            self._i = 0
        else:
            self._seq = list(prices)
            self._i = 0
            self._fn = lambda: 0.0

    def fetch_price(self, symbol: str) -> PriceTick:
        if self._seq is not None:
            if self._i >= len(self._seq):
                raise RuntimeError("SequencePriceSource exhausted")
            px = float(self._seq[self._i])
            self._i += 1
        else:
            px = float(self._fn())
        return PriceTick(
            symbol=str(symbol or self._symbol).upper(),
            price=px,
            ts_monotonic=time.monotonic(),
            ts_wall=time.time(),
            source="sequence",
        )


OnTick = Callable[[PriceTick], None]
OnStale = Callable[[str, float], None]
OnReconnect = Callable[[str, int], None]


@dataclass
class T1PriceMonitor:
    source: PriceSource
    symbol: str
    on_tick: OnTick
    on_stale: OnStale | None = None
    on_reconnect: OnReconnect | None = None
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S
    max_stale_age_s: float = DEFAULT_MAX_STALE_AGE_S
    reconnect_base_s: float = DEFAULT_RECONNECT_BASE_S
    reconnect_max_s: float = DEFAULT_RECONNECT_MAX_S
    state: PriceMonitorState = PriceMonitorState.STOPPED
    last_tick: PriceTick | None = None
    last_error: str | None = None
    reconnect_attempts: int = 0
    ticks_received: int = 0
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if float(self.poll_interval_s) >= SIGNAL_EVALUATION_INTERVAL_S:
            raise ValueError(
                f"T1 price monitor poll_interval_s={self.poll_interval_s} must be "
                f"<< signal evaluation interval ({SIGNAL_EVALUATION_INTERVAL_S}s)."
            )
        if float(self.poll_interval_s) <= 0:
            raise ValueError("poll_interval_s must be > 0")
        if float(self.max_stale_age_s) < float(self.poll_interval_s):
            raise ValueError("max_stale_age_s should be >= poll_interval_s")

    @property
    def is_independent_of_signal_loop(self) -> bool:
        return float(self.poll_interval_s) < SIGNAL_EVALUATION_INTERVAL_S

    @property
    def age_s(self) -> float | None:
        if self.last_tick is None:
            return None
        return max(0.0, time.monotonic() - float(self.last_tick.ts_monotonic))

    @property
    def is_stale(self) -> bool:
        age = self.age_s
        if age is None:
            return self.state in {PriceMonitorState.STALE, PriceMonitorState.RECONNECTING}
        return age > float(self.max_stale_age_s)

    def start(self, *, daemon: bool = True) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self.state = PriceMonitorState.RUNNING
        self._thread = threading.Thread(target=self._loop, name="T1PriceMonitor", daemon=daemon)
        self._thread.start()

    def request_stop(self) -> None:
        """Signal the loop to exit without joining (safe from on_tick / monitor thread)."""
        self._stop.set()
        if self.state != PriceMonitorState.HALTED:
            self.state = PriceMonitorState.STOPPED

    def stop(self, *, join_timeout_s: float = 2.0) -> None:
        """Stop the monitor. Idempotent and safe to call from the monitor thread.

        When invoked from inside on_tick / the poll loop, only signals shutdown and
        does **not** join the current thread (avoids RuntimeError: cannot join
        current thread). An external caller may join afterward.
        """
        self.request_stop()
        t = self._thread
        if t is None:
            return
        # Never join the thread we are currently running on.
        if threading.current_thread() is t:
            return
        if t.is_alive():
            t.join(timeout=join_timeout_s)
        if self._thread is t and not t.is_alive():
            self._thread = None
        if self.state != PriceMonitorState.HALTED:
            self.state = PriceMonitorState.STOPPED

    def halt(self, reason: str) -> None:
        self.last_error = reason
        self.state = PriceMonitorState.HALTED
        self._stop.set()

    def poll_once(self) -> PriceTick | None:
        if self.state == PriceMonitorState.HALTED:
            return None
        if self._stop.is_set() and self.state == PriceMonitorState.STOPPED:
            return None
        try:
            tick = self.source.fetch_price(self.symbol)
        except Exception as e:  # noqa: BLE001
            self._handle_fetch_failure(e)
            return None
        return self._accept_tick(tick)

    def _accept_tick(self, tick: PriceTick) -> PriceTick:
        self.last_tick = tick
        self.last_error = None
        self.ticks_received += 1
        if self.reconnect_attempts > 0 and self.on_reconnect is not None:
            self.on_reconnect(self.symbol, self.reconnect_attempts)
        self.reconnect_attempts = 0
        if self.state != PriceMonitorState.HALTED and not self._stop.is_set():
            self.state = PriceMonitorState.RUNNING
        try:
            self.on_tick(tick)
        except Exception as e:  # noqa: BLE001
            logger.exception("on_tick failed: %s", e)
            self.last_error = f"ON_TICK_FAILED:{e}"
        return tick

    def _handle_fetch_failure(self, err: Exception) -> None:
        self.last_error = f"{type(err).__name__}:{err}"
        self.reconnect_attempts += 1
        self.state = PriceMonitorState.RECONNECTING
        age = self.age_s
        if age is None or age > float(self.max_stale_age_s):
            self.state = PriceMonitorState.STALE
            if self.on_stale is not None:
                self.on_stale(self.symbol, float(age if age is not None else self.max_stale_age_s))

    def _reconnect_sleep_s(self) -> float:
        delay = float(self.reconnect_base_s) * (2 ** max(0, self.reconnect_attempts - 1))
        return min(float(self.reconnect_max_s), delay)

    def _loop(self) -> None:
        try:
            while not self._stop.is_set():
                if self.state == PriceMonitorState.HALTED:
                    break
                self.poll_once()
                if self._stop.is_set():
                    break
                if self.is_stale and self.state == PriceMonitorState.RUNNING:
                    self.state = PriceMonitorState.STALE
                    if self.on_stale is not None:
                        self.on_stale(self.symbol, float(self.age_s or self.max_stale_age_s))
                sleep_s = (
                    self._reconnect_sleep_s()
                    if self.state == PriceMonitorState.RECONNECTING
                    else float(self.poll_interval_s)
                )
                end = time.monotonic() + sleep_s
                while not self._stop.is_set() and time.monotonic() < end:
                    time.sleep(min(0.05, end - time.monotonic()))
        finally:
            if self.state != PriceMonitorState.HALTED:
                self.state = PriceMonitorState.STOPPED
            if self._thread is threading.current_thread():
                self._thread = None
