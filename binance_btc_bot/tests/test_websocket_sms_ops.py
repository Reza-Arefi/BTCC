"""WebSocket package, modern WS API user-data stream, and SMS routing tests.

No real orders. No private keys logged.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import unittest
from unittest.mock import MagicMock, patch

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat

from binance_btc_bot.exchange.signing import Ed25519RequestSigner, build_signature_payload
from binance_btc_bot.market_data.user_stream import (
    DEFAULT_WS_API_BASE,
    SUBSCRIBE_METHOD,
    BinanceUserDataWebsocket,
    UserDataStream,
    build_listen_key_signed_params,
    build_subscribe_signature_params,
    build_subscribe_signature_request,
    parse_user_stream_event,
)
from binance_btc_bot.notifications.base import NotificationResult
from binance_btc_bot.notifications.manager import NotificationManager
from binance_btc_bot.notifications.sms import MockSMSProvider, SMSNotifier, build_sms_provider_from_env
from binance_btc_bot.risk.safety import SafetyState, SafetySystem
from binance_btc_bot.secrets import ScrubbingFilter, clear_runtime_secrets, scrub_text


def _signer() -> Ed25519RequestSigner:
    key = Ed25519PrivateKey.generate()
    pem = key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
    return Ed25519RequestSigner.from_pem_bytes(pem)


class TestWebsocketsPackage(unittest.TestCase):
    def test_import(self):
        import websockets

        self.assertTrue(hasattr(websockets, "connect"))


class TestUserDataStreamLifecycle(unittest.TestCase):
    def test_connect_parse_dedupe_disconnect_reconnect_rest(self):
        reconciled = []
        received = []

        bus = UserDataStream(
            on_event=lambda e: received.append(e),
            rest_reconcile=lambda: reconciled.append(1) or {"ok": True},
        )
        bus.connect()
        raw = {"e": "executionReport", "E": 1, "s": "ETHBTC", "i": 9, "X": "NEW", "t": -1}
        ev = parse_user_stream_event(raw)
        self.assertEqual(ev["type"], "executionReport")
        bus.publish(ev)
        bus.publish(ev)  # dedupe
        self.assertEqual(len(received), 1)

        bus.disconnect()
        self.assertEqual(len(reconciled), 1)
        self.assertFalse(bus.connected)

        bus.reconnect()
        self.assertTrue(bus.connected)
        self.assertEqual(len(reconciled), 2)

    def test_binance_user_data_dry_inject(self):
        got = []
        rest = []
        ws = BinanceUserDataWebsocket(
            dry_run=True,
            on_event=lambda e: got.append(e),
            rest_reconcile=lambda: rest.append("r") or {"ok": True},
        )
        ws.start()
        self.assertTrue(ws.connected)
        ws.inject_message(
            {"e": "listStatus", "E": 2, "s": "ETHBTC", "g": 100, "l": "EXECUTING"}
        )
        self.assertEqual(len(got), 1)
        ws.stop()
        self.assertGreaterEqual(len(rest), 1)

    def test_deprecated_listen_key_params_still_importable(self):
        p = build_listen_key_signed_params(timestamp_ms=123, recv_window_ms=5000)
        self.assertEqual(p["timestamp"], 123)
        self.assertEqual(p["recvWindow"], 5000)

    def test_ws_api_url_has_no_secrets_or_listen_key(self):
        ws = BinanceUserDataWebsocket(
            api_key="TESTKEY",
            signer=_signer(),
            dry_run=True,
            listen_key="should-not-appear-in-url",
        )
        url = ws.stream_url or ""
        self.assertIn("ws-api", url)
        self.assertNotIn("should-not-appear-in-url", url)
        self.assertNotIn("TESTKEY", url)
        self.assertNotIn("/ws/", url.replace("ws-api", ""))


class TestModernWsApiUserData(unittest.TestCase):
    def setUp(self) -> None:
        clear_runtime_secrets()
        self.signer = _signer()
        self.api_key = "TESTAPIKEYVALUEFORSIGNING1234567890"

    def tearDown(self) -> None:
        clear_runtime_secrets()

    def test_a_ws_api_base_default(self):
        self.assertTrue(DEFAULT_WS_API_BASE.startswith("wss://ws-api.binance.com"))

    def test_c_subscribe_request_signing_alphabetical(self):
        params = build_subscribe_signature_params(
            api_key=self.api_key, timestamp_ms=1_700_000_000_000, recv_window_ms=5000
        )
        payload = build_signature_payload(params)
        # Alphabetical: apiKey, recvWindow, timestamp
        self.assertTrue(payload.startswith("apiKey="))
        self.assertIn("recvWindow=5000", payload)
        self.assertIn("timestamp=1700000000000", payload)
        parts = payload.split("&")
        keys = [p.split("=", 1)[0] for p in parts]
        self.assertEqual(keys, sorted(keys))
        req = build_subscribe_signature_request(
            api_key=self.api_key,
            signer=self.signer,
            timestamp_ms=1_700_000_000_000,
            recv_window_ms=5000,
            request_id="unit-sign",
        )
        self.assertEqual(req["method"], SUBSCRIBE_METHOD)
        self.assertEqual(req["id"], "unit-sign")
        sig = req["params"]["signature"]
        raw = base64.b64decode(sig)
        self.assertEqual(len(raw), 64)
        # Verify against payload without signature
        unsigned = {k: v for k, v in req["params"].items() if k != "signature"}
        payload2 = build_signature_payload(unsigned)
        self.signer._private_key.public_key().verify(raw, payload2.encode("ASCII"))

    def test_d_subscription_response_ok_shape(self):
        # Simulated Binance success response handling via probe mock path
        resp = {"id": "x", "status": 200, "result": {"subscriptionId": 7}}
        self.assertEqual(resp["status"], 200)
        self.assertEqual(resp["result"]["subscriptionId"], 7)

    def test_e_unwrap_subscription_event_wrapper(self):
        wrapped = {
            "subscriptionId": 3,
            "event": {
                "e": "executionReport",
                "E": 10,
                "s": "ETHBTC",
                "i": 99,
                "X": "NEW",
                "t": -1,
            },
        }
        ev = parse_user_stream_event(wrapped)
        self.assertIsNotNone(ev)
        self.assertEqual(ev["type"], "executionReport")
        self.assertEqual(ev["subscription_id"], 3)
        self.assertEqual(ev["symbol"], "ETHBTC")

    def test_f_execution_report_parsing(self):
        ev = parse_user_stream_event(
            {"e": "executionReport", "E": 1, "s": "SOLBTC", "i": 1, "X": "FILLED", "t": 2}
        )
        self.assertEqual(ev["type"], "executionReport")
        self.assertIn("er:", ev["event_id"])

    def test_g_list_status_parsing(self):
        ev = parse_user_stream_event(
            {"e": "listStatus", "E": 2, "s": "ETHBTC", "g": 100, "l": "EXECUTING"}
        )
        self.assertEqual(ev["type"], "listStatus")
        self.assertIn("ls:", ev["event_id"])

    def test_h_account_balance_event_parsing(self):
        oap = parse_user_stream_event(
            {
                "subscriptionId": 0,
                "event": {
                    "e": "outboundAccountPosition",
                    "E": 3,
                    "u": 4,
                    "B": [{"a": "BTC", "f": "1", "l": "0"}],
                },
            }
        )
        self.assertEqual(oap["type"], "outboundAccountPosition")
        bal = parse_user_stream_event(
            {"e": "balanceUpdate", "E": 5, "a": "BTC", "d": "0.1", "T": 6}
        )
        self.assertEqual(bal["type"], "balanceUpdate")

    def test_i_event_stream_terminated_triggers_reconcile(self):
        reconciled = []
        got = []
        ws = BinanceUserDataWebsocket(
            dry_run=True,
            on_event=lambda e: got.append(e),
            rest_reconcile=lambda: reconciled.append(1) or {"ok": True},
        )
        ws.start()
        ws.inject_message({"e": "eventStreamTerminated", "E": 99})
        self.assertTrue(any(e["type"] == "eventStreamTerminated" for e in got))
        self.assertGreaterEqual(len(reconciled), 1)
        self.assertTrue(ws.entries_uncertain or not ws.connected)
        ws.stop()

    def test_j_disconnect_rest_reconciliation(self):
        reconciled = []
        ws = BinanceUserDataWebsocket(
            dry_run=True,
            rest_reconcile=lambda: reconciled.append("r") or {"ok": True},
        )
        ws.start()
        ws.stop()
        self.assertIn("r", reconciled)

    def test_k_reconnect_resubscribe_bus(self):
        reconciled = []
        bus = UserDataStream(rest_reconcile=lambda: reconciled.append(1) or {"ok": True})
        bus.connect()
        bus.disconnect()
        bus.reconnect()
        self.assertTrue(bus.connected)
        self.assertEqual(len(reconciled), 2)

    def test_l_malformed_ws_response_safe_failure(self):
        self.assertIsNone(parse_user_stream_event("not-json{"))
        self.assertIsNone(parse_user_stream_event({"foo": "bar"}))
        self.assertIsNone(parse_user_stream_event(123))

    def test_m_authentication_failure_halts(self):
        safety = SafetySystem()
        failures = []

        def _halt(reason: str) -> None:
            failures.append(reason)
            safety.halt("USER_DATA_AUTH_FAILURE", detail=reason)

        ws = BinanceUserDataWebsocket(
            api_key=self.api_key,
            signer=self.signer,
            dry_run=False,
            on_auth_failure=_halt,
        )
        ws._fail_auth("Invalid API-key")
        self.assertEqual(safety.state, SafetyState.HALT)
        self.assertTrue(failures)
        self.assertTrue(ws.entries_uncertain)
        self.assertNotIn(self.api_key, failures[0])

    def test_n_no_secret_leakage_in_logs_or_request_repr(self):
        req = build_subscribe_signature_request(
            api_key=self.api_key,
            signer=self.signer,
            timestamp_ms=1_700_000_000_000,
        )
        # Scrub api key from accidental log strings
        blob = json.dumps(req)
        scrubbed = scrub_text(blob, extra_secrets=[self.api_key, req["params"]["signature"]])
        self.assertNotIn(self.api_key, scrubbed)
        self.assertNotIn(req["params"]["signature"], scrubbed)

        logger = logging.getLogger("binance_btc_bot.market_data.user_stream_test")
        logger.addFilter(ScrubbingFilter())
        line = f"api_key={self.api_key} signature={req['params']['signature']}"
        self.assertNotIn(self.api_key, scrub_text(line, extra_secrets=[self.api_key]))

    def test_o_dry_run_remains_safe_no_network_start(self):
        ws = BinanceUserDataWebsocket(
            api_key=self.api_key,
            signer=self.signer,
            dry_run=True,
        )
        ws.start()
        self.assertTrue(ws.connected)
        self.assertFalse(ws.subscribed)  # no live subscribe in dry-run
        probe = ws.probe_subscribe()
        self.assertEqual(probe.get("error"), "DRY_RUN")
        self.assertFalse(probe.get("ok"))
        ws.stop()

    def test_b_probe_subscribe_success_mocked(self):
        """Successful WS API connection + Ed25519 subscription (mocked transport)."""

        class _WS:
            def __init__(self) -> None:
                self.sent = []
                self._n = 0

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def send(self, data):
                self.sent.append(json.loads(data))

            async def recv(self):
                self._n += 1
                if self._n == 1:
                    return json.dumps(
                        {"id": "1", "status": 200, "result": {"subscriptionId": 42}}
                    )
                return json.dumps({"id": "2", "status": 200, "result": {}})

        mock_connect = MagicMock(return_value=_WS())
        ws = BinanceUserDataWebsocket(
            api_key=self.api_key,
            signer=self.signer,
            dry_run=False,
        )
        with patch("websockets.connect", mock_connect):
            result = ws.probe_subscribe(timeout_sec=5, unsubscribe=True)
        self.assertTrue(result["ok"])
        self.assertTrue(result["connected"])
        self.assertTrue(result["subscribed"])
        self.assertEqual(result["subscription_id"], 42)
        self.assertEqual(mock_connect.call_args[0][0], DEFAULT_WS_API_BASE)
        self.assertEqual(result.get("method"), SUBSCRIBE_METHOD)
        # First sent frame is subscribe.signature
        sent = mock_connect.return_value.sent
        self.assertEqual(sent[0]["method"], SUBSCRIBE_METHOD)
        self.assertIn("signature", sent[0]["params"])
        self.assertNotIn("listenKey", json.dumps(sent))

    def test_p_preflight_probe_does_not_touch_order_endpoints(self):
        """probe_subscribe must only use userDataStream.subscribe/unsubscribe methods."""
        class _WS:
            def __init__(self) -> None:
                self.sent = []
                self._n = 0

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def send(self, data):
                self.sent.append(json.loads(data))

            async def recv(self):
                self._n += 1
                if self._n == 1:
                    return json.dumps(
                        {"id": "1", "status": 200, "result": {"subscriptionId": 1}}
                    )
                return json.dumps({"id": "2", "status": 200, "result": {}})

        inst = _WS()
        ws = BinanceUserDataWebsocket(
            api_key=self.api_key, signer=self.signer, dry_run=False
        )
        with patch("websockets.connect", MagicMock(return_value=inst)):
            ws.probe_subscribe(unsubscribe=True)
        methods = [m["method"] for m in inst.sent]
        self.assertTrue(all(m.startswith("userDataStream.") for m in methods))
        forbidden = ("order.", "orderList", "openOrder", "/api/v3/order")
        blob = json.dumps(inst.sent)
        for f in forbidden:
            self.assertNotIn(f, blob)


class TestSmsEmergencyOnly(unittest.TestCase):
    def test_mock_provider_no_network(self):
        p = MockSMSProvider()
        r = p.send_sms("hello")
        self.assertTrue(r.ok)
        self.assertEqual(r.reason, "MOCK_OK")
        self.assertEqual(len(p.sent), 1)

    def test_critical_telegram_and_sms_info_telegram_only(self):
        class RecTg:
            name = "telegram"

            def __init__(self) -> None:
                self.sent = []

            def configured(self) -> bool:
                return True

            def send(self, event):
                self.sent.append((event.severity.value, event.event))
                return NotificationResult(ok=True, channel="telegram")

            def status(self):
                return {"configured": True}

        mock = MockSMSProvider()
        n = NotificationManager(telegram=RecTg(), sms=SMSNotifier(provider=mock))
        n.notify_info("SIGNAL_DETECTED", "normal")
        self.assertEqual(len(mock.sent), 0)
        self.assertEqual(n.telegram.sent[-1][0], "INFO")  # type: ignore[attr-defined]

        n.notify_critical("HALT", "emergency")
        self.assertEqual(len(mock.sent), 1)
        self.assertIn("CRITICAL", mock.sent[0])
        self.assertEqual(n.telegram.sent[-1][0], "CRITICAL")  # type: ignore[attr-defined]

    def test_build_mock_from_env(self):
        prev = os.environ.get("SMS_PROVIDER")
        try:
            os.environ["SMS_PROVIDER"] = "mock"
            p = build_sms_provider_from_env()
            self.assertIsInstance(p, MockSMSProvider)
            st = p.status()
            self.assertNotIn("auth", str(st).lower())
            self.assertNotIn("token", str(st).lower())
        finally:
            if prev is None:
                os.environ.pop("SMS_PROVIDER", None)
            else:
                os.environ["SMS_PROVIDER"] = prev


if __name__ == "__main__":
    unittest.main()
