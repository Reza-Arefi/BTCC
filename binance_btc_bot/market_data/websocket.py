"""Binance combined-stream websocket client with reconnect.

Uses public streams only (prefer data-stream.binance.vision). Disconnects must
not crash the bot — callers poll `last_prices` and treat staleness via SafetySystem.
Authenticated user-data streams live in ``user_stream.py``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Public market-data WS (works from many restricted regions; same family as Vision REST).
DEFAULT_PUBLIC_WS_BASE = "wss://data-stream.binance.vision"
# Official stream host — may return HTTP 451 from ineligible networks.
OFFICIAL_WS_BASE = "wss://stream.binance.com:9443"


class BinanceMarketWebsocket:
    """Background websocket price feed (requires ``websockets`` package)."""

    def __init__(
        self,
        symbols: list[str],
        *,
        ws_base: str = DEFAULT_PUBLIC_WS_BASE,
        on_price: Callable[[str, float], None] | None = None,
    ) -> None:
        self.symbols = [s.lower() for s in symbols]
        self.ws_base = ws_base.rstrip("/")
        self.on_price = on_price
        self.last_prices: dict[str, float] = {}
        self.last_message_at: float = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.connected = False
        self.last_error: str | None = None
        self.reconnects: int = 0

    @property
    def stream_url(self) -> str:
        streams = "/".join(f"{s}@trade" for s in self.symbols)
        return f"{self.ws_base}/stream?streams={streams}"

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="binance-md-ws", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def age_sec(self) -> float:
        if self.last_message_at <= 0:
            return float("inf")
        return time.time() - self.last_message_at

    def _run(self) -> None:
        try:
            import websockets  # type: ignore
        except ImportError:
            self.last_error = "websockets package not installed"
            logger.error(self.last_error)
            return

        async def _loop() -> None:
            backoff = 1.0
            while not self._stop.is_set():
                try:
                    async with websockets.connect(
                        self.stream_url, ping_interval=20, open_timeout=10
                    ) as ws:
                        self.connected = True
                        backoff = 1.0
                        logger.info(
                            "WS connected streams=%d base=%s", len(self.symbols), self.ws_base
                        )
                        while not self._stop.is_set():
                            raw = await asyncio.wait_for(ws.recv(), timeout=60)
                            self._handle(raw)
                except Exception as e:  # noqa: BLE001
                    was = self.connected
                    self.connected = False
                    self.last_error = str(e)
                    if was:
                        self.reconnects += 1
                    logger.warning("WS disconnect: %s; retry in %.1fs", e, backoff)
                    await asyncio.sleep(backoff)
                    backoff = min(60.0, backoff * 2)

        asyncio.run(_loop())

    def _handle(self, raw: str | bytes) -> None:
        data = json.loads(raw)
        payload: dict[str, Any] = (
            data.get("data") if isinstance(data, dict) and "data" in data else data
        )
        if not isinstance(payload, dict):
            return
        symbol = str(payload.get("s") or "").upper()
        price_raw = payload.get("p") or payload.get("c")
        if not symbol or price_raw is None:
            return
        price = float(price_raw)
        self.last_prices[symbol] = price
        self.last_message_at = time.time()
        if self.on_price:
            self.on_price(symbol, price)
