"""Authenticated Binance user-data WebSocket (Spot WebSocket API).

Current Binance Spot path (listenKey discontinued 2026-02-20):
  wss://ws-api.binance.com:443/ws-api/v3
  → userDataStream.subscribe.signature (Ed25519)
  → events: executionReport, listStatus, outboundAccountPosition, ...

Legacy listenKey REST (POST/PUT/DELETE /api/v3/userDataStream) is obsolete and
must not be used on the LIVE path (returns HTTP 410).

Dry-run / unit tests use the in-process UserDataStream bus without private WS.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

from binance_btc_bot.exchange.signing import Ed25519RequestSigner, build_signature_payload
from binance_btc_bot.secrets import scrub_exception, scrub_obj, scrub_text

logger = logging.getLogger(__name__)

EventHandler = Callable[[dict[str, Any]], None]

# Official Spot WebSocket API (not Vision public market streams; not listenKey host).
DEFAULT_WS_API_BASE = "wss://ws-api.binance.com:443/ws-api/v3"
SUBSCRIBE_METHOD = "userDataStream.subscribe.signature"
UNSUBSCRIBE_METHOD = "userDataStream.unsubscribe"


def parse_user_stream_event(raw: str | bytes | dict[str, Any]) -> dict[str, Any] | None:
    """Parse a Binance user-data payload into a normalized event.

    Supports:
      - legacy/direct event objects: {"e": "executionReport", ...}
      - combined stream wrapper: {"stream": "...", "data": {...}}
      - modern WS API wrapper: {"subscriptionId": 0, "event": {...}}
    """
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="replace")
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
    else:
        data = raw
    if not isinstance(data, dict):
        return None

    subscription_id = data.get("subscriptionId")
    # Modern WS API user-data envelope
    if isinstance(data.get("event"), dict):
        payload = data["event"]
    # Combined stream wrapper
    elif "data" in data and isinstance(data.get("data"), dict) and (
        "stream" in data or "e" not in data
    ):
        payload = data["data"]
        if subscription_id is None:
            subscription_id = data.get("subscriptionId")
    else:
        payload = data

    if not isinstance(payload, dict):
        return None
    etype = str(payload.get("e") or payload.get("eventType") or payload.get("type") or "").strip()
    if not etype:
        return None
    event_id = (
        str(payload.get("i") or "")
        or str(payload.get("E") or "")
        or str(payload.get("event_id") or "")
    )
    if etype == "executionReport":
        trade_id = payload.get("t")
        order_id = payload.get("i")
        status = payload.get("X")
        event_id = f"er:{order_id}:{status}:{trade_id}:{payload.get('E')}"
    elif etype == "listStatus":
        event_id = f"ls:{payload.get('g')}:{payload.get('l')}:{payload.get('E')}"
    elif etype == "outboundAccountPosition":
        event_id = f"oap:{payload.get('u') or payload.get('E')}"
    elif etype == "balanceUpdate":
        event_id = f"bu:{payload.get('a')}:{payload.get('E')}:{payload.get('T')}"
    elif etype == "eventStreamTerminated":
        event_id = f"est:{payload.get('E')}"
    return {
        "type": etype,
        "event_id": event_id,
        "event_time": payload.get("E"),
        "symbol": str(payload.get("s") or "").upper() or None,
        "subscription_id": subscription_id,
        "payload": payload,
    }


def build_subscribe_signature_params(
    *,
    api_key: str,
    timestamp_ms: int,
    recv_window_ms: int = 5000,
) -> dict[str, Any]:
    """Unsigned params for userDataStream.subscribe.signature (before signing)."""
    return {
        "apiKey": str(api_key),
        "timestamp": int(timestamp_ms),
        "recvWindow": int(recv_window_ms),
    }


def build_subscribe_signature_request(
    *,
    api_key: str,
    signer: Ed25519RequestSigner,
    recv_window_ms: int = 5000,
    timestamp_ms: int | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Build a signed WS API request for userDataStream.subscribe.signature.

    Does not place secrets in the WebSocket URL — credentials stay in the
    JSON params body only (apiKey id + signature; private key never leaves signer).
    """
    ts = int(timestamp_ms if timestamp_ms is not None else time.time() * 1000)
    params = build_subscribe_signature_params(
        api_key=api_key,
        timestamp_ms=ts,
        recv_window_ms=recv_window_ms,
    )
    _payload, sig_b64 = signer.sign_params(params)
    params = dict(params)
    params["signature"] = sig_b64
    return {
        "id": request_id or str(uuid.uuid4()),
        "method": SUBSCRIBE_METHOD,
        "params": params,
    }


# Deprecated alias — kept for test migration; do not use on LIVE path.
def build_listen_key_signed_params(*, timestamp_ms: int, recv_window_ms: int = 5000) -> dict[str, Any]:
    """DEPRECATED: legacy listenKey REST params (endpoint returns HTTP 410)."""
    return {"timestamp": int(timestamp_ms), "recvWindow": int(recv_window_ms)}


@dataclass
class UserDataStream:
    """In-process user-data bus with disconnect/reconnect + REST fallback."""

    on_event: EventHandler | None = None
    on_disconnect: Callable[[], None] | None = None
    on_reconnect: Callable[[], None] | None = None
    rest_reconcile: Callable[[], dict[str, Any]] | None = None
    connected: bool = False
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _queue: deque[dict[str, Any]] = field(default_factory=deque, repr=False)
    _seen_ids: set[str] = field(default_factory=set, repr=False)

    def connect(self) -> None:
        with self._lock:
            was = self.connected
            self.connected = True
        if not was and self.on_reconnect:
            try:
                self.on_reconnect()
            except Exception:  # noqa: BLE001
                logger.warning("on_reconnect failed")

    def disconnect(self) -> None:
        with self._lock:
            was = self.connected
            self.connected = False
        if was and self.on_disconnect:
            try:
                self.on_disconnect()
            except Exception:  # noqa: BLE001
                logger.warning("on_disconnect failed")
        if self.rest_reconcile:
            try:
                self.rest_reconcile()
            except Exception:  # noqa: BLE001
                logger.warning("rest_reconcile on disconnect failed")

    def publish(self, event: dict[str, Any]) -> None:
        """Publish an execution event (deduped by event_id if present)."""
        eid = str(event.get("event_id") or event.get("id") or "")
        with self._lock:
            if eid and eid in self._seen_ids:
                return
            if eid:
                self._seen_ids.add(eid)
            self._queue.append(event)
        self._drain()

    def _drain(self) -> None:
        while True:
            with self._lock:
                if not self.connected or not self._queue:
                    return
                ev = self._queue.popleft()
            if self.on_event:
                try:
                    self.on_event(ev)
                except Exception:  # noqa: BLE001
                    logger.warning("user-data handler failed")

    def reconnect(self) -> dict[str, Any]:
        """Reconnect bus and run REST reconciliation (no duplicate notifications)."""
        report: dict[str, Any] = {"ok": True, "rest": None}
        self.connect()
        if self.rest_reconcile:
            report["rest"] = self.rest_reconcile()
        self._drain()
        return report


class BinanceUserDataWebsocket:
    """Authenticated user-data stream via Spot WebSocket API subscribe.signature.

    LIVE path: connect to ws-api → Ed25519 userDataStream.subscribe.signature →
    receive account events. On disconnect / eventStreamTerminated: REST reconcile
    then resubscribe. Never places orders. Never puts private key material in the URL.

    Dry-run: local UserDataStream bus only (inject_message for tests).
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        signer: Ed25519RequestSigner | None = None,
        ws_api_base: str = DEFAULT_WS_API_BASE,
        recv_window_ms: int = 5000,
        on_event: EventHandler | None = None,
        rest_reconcile: Callable[[], dict[str, Any]] | None = None,
        on_auth_failure: Callable[[str], None] | None = None,
        dry_run: bool = True,
        # Deprecated — ignored for LIVE; kept so old call sites fail closed.
        listen_key: str | None = None,
        ws_base: str | None = None,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.signer = signer
        self.ws_api_base = (ws_api_base or DEFAULT_WS_API_BASE).rstrip("/")
        self.recv_window_ms = int(recv_window_ms)
        self.on_event = on_event
        self.rest_reconcile = rest_reconcile
        self.on_auth_failure = on_auth_failure
        self.dry_run = bool(dry_run)
        self._deprecated_listen_key = listen_key
        # Legacy alias unused on LIVE path
        self.ws_base = (ws_base or self.ws_api_base).rstrip("/")
        self.bus = UserDataStream(
            on_event=on_event,
            rest_reconcile=rest_reconcile,
            on_disconnect=lambda: None,
        )
        self.connected = False
        self.subscribed = False
        self.subscription_id: int | None = None
        self.last_error: str | None = None
        self.last_event_at: float = 0.0
        self.events_received: int = 0
        self.auth_ok: bool = False
        self._uncertain: bool = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def stream_url(self) -> str | None:
        """WS API endpoint URL — never includes credentials or listenKey."""
        return self.ws_api_base

    @property
    def entries_uncertain(self) -> bool:
        """True while disconnected / reconciling — callers must not open new risk."""
        return bool(self._uncertain) or not self.connected

    def start(self) -> None:
        if self.dry_run or not self.api_key or self.signer is None:
            # Local bus only — no private network call.
            if self._deprecated_listen_key and not self.dry_run:
                self.last_error = "listenKey path discontinued; use subscribe.signature"
            self.bus.connect()
            self.connected = True
            self._uncertain = False
            return
        if self._deprecated_listen_key:
            logger.warning(
                "listenKey ignored — Spot listenKey API discontinued; using WS API subscribe.signature"
            )
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="binance-userdata-wsapi", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self.connected:
            self.bus.disconnect()
        self.connected = False
        self.subscribed = False
        if self._thread:
            self._thread.join(timeout=8)
            self._thread = None

    def inject_message(self, raw: str | bytes | dict[str, Any]) -> dict[str, Any] | None:
        """Parse and publish a message (tests / dry-run simulation)."""
        ev = parse_user_stream_event(raw)
        if ev is None:
            return None
        self.events_received += 1
        self.last_event_at = time.time()
        if ev.get("type") == "eventStreamTerminated":
            self._handle_stream_terminated(ev)
            return ev
        self.bus.publish(ev)
        return ev

    def probe_subscribe(
        self,
        *,
        timeout_sec: float = 20.0,
        unsubscribe: bool = True,
    ) -> dict[str, Any]:
        """One-shot connect + subscribe.signature + optional unsubscribe (preflight).

        Does not place or cancel orders. Returns a sanitized status dict.
        """
        report: dict[str, Any] = {
            "ok": False,
            "connected": False,
            "subscribed": False,
            "subscription_id": None,
            "method": SUBSCRIBE_METHOD,
            "error": None,
        }
        if self.dry_run:
            report["error"] = "DRY_RUN"
            return report
        if not self.api_key or self.signer is None:
            report["error"] = "MISSING_CREDENTIALS"
            return report

        async def _once() -> dict[str, Any]:
            import websockets

            out = dict(report)
            try:
                async with websockets.connect(
                    self.ws_api_base,
                    open_timeout=min(15.0, timeout_sec),
                    close_timeout=5,
                    ping_interval=20,
                ) as ws:
                    out["connected"] = True
                    req = build_subscribe_signature_request(
                        api_key=self.api_key,
                        signer=self.signer,  # type: ignore[arg-type]
                        recv_window_ms=self.recv_window_ms,
                    )
                    # Never log params (contain apiKey + signature).
                    logger.info("WS API sending %s id=%s", SUBSCRIBE_METHOD, req.get("id"))
                    await ws.send(json.dumps(req))
                    raw = await asyncio.wait_for(ws.recv(), timeout=timeout_sec)
                    resp = json.loads(raw) if isinstance(raw, str) else raw
                    if not isinstance(resp, dict):
                        out["error"] = "MALFORMED_RESPONSE"
                        return out
                    status = resp.get("status")
                    result = resp.get("result") if isinstance(resp.get("result"), dict) else {}
                    err = resp.get("error") if isinstance(resp.get("error"), dict) else None
                    if status == 200 and result is not None:
                        out["subscribed"] = True
                        out["ok"] = True
                        out["subscription_id"] = result.get("subscriptionId")
                        self.auth_ok = True
                        self.subscription_id = (
                            int(result["subscriptionId"])
                            if result.get("subscriptionId") is not None
                            else None
                        )
                        if unsubscribe:
                            unsub = {
                                "id": str(uuid.uuid4()),
                                "method": UNSUBSCRIBE_METHOD,
                                "params": (
                                    {"subscriptionId": self.subscription_id}
                                    if self.subscription_id is not None
                                    else {}
                                ),
                            }
                            try:
                                await ws.send(json.dumps(unsub))
                                await asyncio.wait_for(ws.recv(), timeout=5)
                            except Exception:  # noqa: BLE001
                                pass
                    else:
                        code = err.get("code") if err else None
                        msg = scrub_text(str(err.get("msg") if err else resp))
                        out["error"] = f"SUBSCRIBE_FAIL status={status} code={code} msg={msg}"
                        self._fail_auth(out["error"])
            except Exception as e:  # noqa: BLE001
                out["error"] = scrub_exception(e)
                self._fail_auth(out["error"])
            return scrub_obj(out)

        try:
            return asyncio.run(_once())
        except Exception as e:  # noqa: BLE001
            report["error"] = scrub_exception(e)
            self._fail_auth(report["error"])
            return scrub_obj(report)

    def _fail_auth(self, reason: str) -> None:
        self.auth_ok = False
        self.last_error = scrub_text(reason)
        self._uncertain = True
        if self.on_auth_failure:
            try:
                self.on_auth_failure(self.last_error or "AUTH_FAILURE")
            except Exception:  # noqa: BLE001
                logger.warning("on_auth_failure callback failed")

    def _handle_stream_terminated(self, ev: dict[str, Any]) -> None:
        logger.warning("user-data eventStreamTerminated — reconciling via REST")
        self.subscribed = False
        self._uncertain = True
        self.bus.publish(ev)
        try:
            self.bus.disconnect()  # triggers REST reconcile
        except Exception:  # noqa: BLE001
            pass
        self.connected = False

    def _run(self) -> None:
        try:
            import websockets  # type: ignore  # noqa: F401
        except ImportError:
            self.last_error = "websockets package not installed"
            logger.error(self.last_error)
            self._fail_auth(self.last_error)
            return

        async def _loop() -> None:
            backoff = 1.0
            while not self._stop.is_set():
                try:
                    self._uncertain = True
                    import websockets

                    async with websockets.connect(
                        self.ws_api_base,
                        ping_interval=20,
                        open_timeout=15,
                        close_timeout=5,
                    ) as ws:
                        req = build_subscribe_signature_request(
                            api_key=self.api_key,
                            signer=self.signer,  # type: ignore[arg-type]
                            recv_window_ms=self.recv_window_ms,
                        )
                        logger.info("user-data WS API connected; subscribing via %s", SUBSCRIBE_METHOD)
                        await ws.send(json.dumps(req))
                        raw = await asyncio.wait_for(ws.recv(), timeout=20)
                        resp = json.loads(raw) if isinstance(raw, str) else raw
                        if not isinstance(resp, dict) or resp.get("status") != 200:
                            err = resp.get("error") if isinstance(resp, dict) else None
                            reason = scrub_text(
                                str(
                                    (err or {}).get("msg")
                                    if isinstance(err, dict)
                                    else resp
                                )
                            )
                            self._fail_auth(f"SUBSCRIBE_REJECTED {reason}")
                            await asyncio.sleep(backoff)
                            backoff = min(60.0, backoff * 2)
                            continue
                        result = resp.get("result") or {}
                        self.subscription_id = (
                            int(result["subscriptionId"])
                            if isinstance(result, dict) and result.get("subscriptionId") is not None
                            else None
                        )
                        self.auth_ok = True
                        self.subscribed = True
                        self.connected = True
                        self._uncertain = False
                        self.bus.connect()
                        # Reconcile after reconnect/resubscribe (source of truth).
                        if self.rest_reconcile:
                            try:
                                self.rest_reconcile()
                            except Exception:  # noqa: BLE001
                                logger.warning("REST reconcile after subscribe failed")
                        backoff = 1.0
                        logger.info(
                            "user-data subscribed subscriptionId=%s",
                            self.subscription_id,
                        )
                        while not self._stop.is_set():
                            try:
                                frame = await asyncio.wait_for(ws.recv(), timeout=60)
                            except asyncio.TimeoutError:
                                continue
                            # Ignore WS API request/response echoes if any; parse events.
                            if isinstance(frame, (str, bytes, bytearray, dict)):
                                try:
                                    maybe = json.loads(frame) if isinstance(frame, str) else frame
                                except json.JSONDecodeError:
                                    maybe = frame
                                # Skip JSON-RPC style responses (have status/id without event)
                                if (
                                    isinstance(maybe, dict)
                                    and "event" not in maybe
                                    and "e" not in maybe
                                    and ("status" in maybe or "result" in maybe)
                                ):
                                    continue
                            ev = self.inject_message(frame)
                            if ev and ev.get("type") == "eventStreamTerminated":
                                break
                except Exception as e:  # noqa: BLE001
                    self.connected = False
                    self.subscribed = False
                    self._uncertain = True
                    self.last_error = scrub_exception(e)
                    logger.warning(
                        "user-data WS disconnect: %s; retry in %.1fs",
                        self.last_error,
                        backoff,
                    )
                    try:
                        self.bus.disconnect()  # triggers REST reconcile
                    except Exception:  # noqa: BLE001
                        pass
                    await asyncio.sleep(backoff)
                    backoff = min(60.0, backoff * 2)

        try:
            asyncio.run(_loop())
        except Exception as e:  # noqa: BLE001
            self.last_error = scrub_exception(e)
            self._fail_auth(self.last_error)
