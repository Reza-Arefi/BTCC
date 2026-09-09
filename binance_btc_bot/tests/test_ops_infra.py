"""Operational infrastructure tests — credentials, notifications, provider, scrubbing."""

from __future__ import annotations

import io
import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from binance_btc_bot.config_loader import load_config
from binance_btc_bot.credentials import (
    BinanceCredentials,
    CredentialError,
    load_binance_credentials_from_env,
    validate_binance_credentials,
)
from binance_btc_bot.exchange.binance import BinanceExchange
from binance_btc_bot.notifications.base import NotificationEvent, Severity
from binance_btc_bot.notifications.manager import NotificationManager
from binance_btc_bot.notifications.sms import NullSMSProvider, SMSNotifier, TwilioSMSProvider
from binance_btc_bot.notifications.telegram import TelegramNotifier
from binance_btc_bot.risk.safety import SafetySystem
from binance_btc_bot.secrets import ScrubbingFilter, scrub_text
from binance_btc_bot.strategy.entries import LiveEntryEngine
from binance_btc_bot.strategy.provider import FixedStrategyProvider, build_strategy_provider


class TestBinanceCredentials(unittest.TestCase):
    def test_missing_key(self):
        with tempfile.NamedTemporaryFile("wb", suffix=".pem", delete=False) as f:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
            from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat

            key = Ed25519PrivateKey.generate()
            f.write(key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()))
            path = f.name
        try:
            creds = BinanceCredentials(api_key="", private_key_path=path)
            with self.assertRaises(CredentialError) as ctx:
                validate_binance_credentials(creds, required=True)
            self.assertIn("BINANCE_API_KEY", str(ctx.exception))
        finally:
            Path(path).unlink(missing_ok=True)

    def test_missing_private_key_path(self):
        creds = BinanceCredentials(api_key="keyvalue", private_key_path="")
        with self.assertRaises(CredentialError) as ctx:
            validate_binance_credentials(creds, required=True)
        self.assertIn("BINANCE_ED25519_PRIVATE_KEY_PATH", str(ctx.exception))
        self.assertNotIn("keyvalue", str(ctx.exception))

    def test_credentials_present(self):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat

        key = Ed25519PrivateKey.generate()
        pem = key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
        with tempfile.NamedTemporaryFile("wb", suffix=".pem", delete=False) as f:
            f.write(pem)
            path = f.name
        try:
            creds = BinanceCredentials(api_key="abcd", private_key_path=path)
            out = validate_binance_credentials(creds, required=True)
            self.assertTrue(out.present)
            self.assertTrue(out.signer_ready)
            masked = out.masked_summary()
            self.assertEqual(masked["api_key"], "SET")
            self.assertEqual(masked["ed25519_private_key"], "SET")
            self.assertNotIn("abcd", str(masked))
            self.assertNotIn(path, str(masked))
        finally:
            Path(path).unlink(missing_ok=True)

    def test_not_required_when_dry(self):
        creds = BinanceCredentials(api_key="", private_key_path="")
        out = validate_binance_credentials(creds, required=False)
        self.assertFalse(out.present)

    def test_credentials_never_logged(self):
        os.environ["BINANCE_API_KEY"] = "SUPERSECRETKEY1234"
        try:
            logger = logging.getLogger("scrub_test_creds")
            logger.handlers.clear()
            logger.setLevel(logging.INFO)
            buf = io.StringIO()
            handler = logging.StreamHandler(buf)
            handler.addFilter(ScrubbingFilter())
            logger.addHandler(handler)
            logger.info("key=%s", os.environ["BINANCE_API_KEY"])
            text = buf.getvalue()
            self.assertNotIn("SUPERSECRETKEY1234", text)
            self.assertIn("***REDACTED***", text)
        finally:
            os.environ.pop("BINANCE_API_KEY", None)


class TestLiveDisabledBlocksOrders(unittest.TestCase):
    def test_live_enabled_false_blocks_submission(self):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat
        from binance_btc_bot.exchange.signing import Ed25519RequestSigner

        key = Ed25519PrivateKey.generate()
        pem = key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
        signer = Ed25519RequestSigner.from_pem_bytes(pem)
        ex = BinanceExchange(
            api_key="k",
            signer=signer,
            dry_run=False,
            live_enabled=False,
        )
        allowed, reason = ex._writes_allowed()
        self.assertFalse(allowed)
        self.assertEqual(reason, "LIVE_DISABLED")

    def test_dry_run_blocks_even_with_live_flag(self):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat
        from binance_btc_bot.exchange.signing import Ed25519RequestSigner

        key = Ed25519PrivateKey.generate()
        pem = key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
        signer = Ed25519RequestSigner.from_pem_bytes(pem)
        ex = BinanceExchange(api_key="k", signer=signer, dry_run=True, live_enabled=True)
        allowed, reason = ex._writes_allowed()
        self.assertFalse(allowed)
        self.assertEqual(reason, "DRY_RUN")


class TestTelegram(unittest.TestCase):
    def test_missing_configuration(self):
        tg = TelegramNotifier(bot_token="", chat_id="", enabled=True)
        self.assertFalse(tg.configured())
        res = tg.send(
            NotificationEvent(event="BOT_STARTED", severity=Severity.INFO, message="hi")
        )
        self.assertTrue(res.skipped)
        self.assertEqual(res.reason, "MISSING_CONFIG")

    def test_valid_message_mocked(self):
        tg = TelegramNotifier(bot_token="token", chat_id="123", enabled=True)
        with patch("urllib.request.urlopen") as urlopen:
            resp = MagicMock()
            resp.read.return_value = b'{"ok": true, "result": {}}'
            resp.__enter__.return_value = resp
            resp.__exit__.return_value = False
            urlopen.return_value = resp
            res = tg.send(
                NotificationEvent(event="ENTRY_FILLED", severity=Severity.INFO, message="filled")
            )
            self.assertTrue(res.ok)

    def test_api_failure(self):
        tg = TelegramNotifier(bot_token="token", chat_id="123", enabled=True)
        with patch("urllib.request.urlopen", side_effect=RuntimeError("boom")):
            res = tg.send(
                NotificationEvent(event="API_ERROR", severity=Severity.ERROR, message="x")
            )
            self.assertFalse(res.ok)
            self.assertIn("RuntimeError", res.reason)


class TestSMS(unittest.TestCase):
    def test_missing_configuration(self):
        sms = SMSNotifier(provider=TwilioSMSProvider(account_sid="", auth_token="", enabled=True))
        self.assertFalse(sms.configured())
        res = sms.send(
            NotificationEvent(event="HALT", severity=Severity.CRITICAL, message="halted")
        )
        self.assertTrue(res.skipped or not res.ok)
        self.assertEqual(res.reason, "MISSING_CONFIG")

    def test_valid_message_mocked(self):
        provider = TwilioSMSProvider(
            account_sid="ACxxx",
            auth_token="tok",
            from_number="+1000",
            to_number="+2000",
            enabled=True,
        )
        with patch("urllib.request.urlopen") as urlopen:
            resp = MagicMock()
            resp.read.return_value = b'{"sid": "SMxxx", "status": "queued"}'
            resp.__enter__.return_value = resp
            resp.__exit__.return_value = False
            urlopen.return_value = resp
            res = provider.send_sms("BOT HALTED")
            self.assertTrue(res.ok)

    def test_api_failure(self):
        provider = TwilioSMSProvider(
            account_sid="ACxxx",
            auth_token="tok",
            from_number="+1000",
            to_number="+2000",
            enabled=True,
        )
        with patch("urllib.request.urlopen", side_effect=RuntimeError("sms down")):
            res = provider.send_sms("fail")
            self.assertFalse(res.ok)


class TestNotificationManager(unittest.TestCase):
    def test_notification_failure_does_not_crash(self):
        from binance_btc_bot.notifications.base import NotificationResult

        tg = MagicMock()
        tg.send.side_effect = RuntimeError("tg crash")
        tg.status.return_value = {"configured": True}
        sms = MagicMock()
        sms.send.return_value = NotificationResult(ok=True, channel="sms", reason="OK")
        sms.status.return_value = {"configured": True}
        mgr = NotificationManager(telegram=tg, sms=sms)
        results = mgr.notify_info("BOT_STARTED", "hello")
        self.assertTrue(any(not r.ok for r in results))

    def test_critical_attempts_telegram_and_sms(self):
        from binance_btc_bot.notifications.base import NotificationResult

        tg = MagicMock()
        tg.send.return_value = NotificationResult(ok=True, channel="telegram")
        sms = MagicMock()
        sms.send.return_value = NotificationResult(ok=True, channel="sms")
        mgr = NotificationManager(telegram=tg, sms=sms)
        mgr.notify_critical("HALT", "BOT HALTED")
        tg.send.assert_called_once()
        sms.send.assert_called_once()

    def test_info_does_not_sms(self):
        from binance_btc_bot.notifications.base import NotificationResult

        tg = MagicMock()
        tg.send.return_value = NotificationResult(ok=True, channel="telegram")
        sms = MagicMock()
        mgr = NotificationManager(telegram=tg, sms=sms)
        mgr.notify_info("ENTRY_FILLED", "filled")
        tg.send.assert_called_once()
        sms.send.assert_not_called()


class TestStrategyProvider(unittest.TestCase):
    def test_fixed_provider_returns_t7(self):
        cfg = load_config()
        provider = build_strategy_provider(cfg)
        self.assertIsInstance(provider, FixedStrategyProvider)
        strat = provider.get_strategy()
        self.assertEqual(strat.key, "T4")
        self.assertIsNone(provider.selector_key())
        self.assertAlmostEqual(strat.trail_distance, 0.005)

    def test_entry_engine_uses_provider_strategy(self):
        provider = FixedStrategyProvider("T1")
        eng = LiveEntryEngine(
            strategy_key=provider.strategy_key(),
            selector_key=provider.selector_key(),
            long_threshold=0.6,
        )
        d = eng.evaluate(symbol="ETHBTC", score=0.7, open_symbols=[], open_count=0)
        self.assertEqual(d.strategy, provider.strategy_key())
        self.assertEqual(d.strategy, "T1")
        self.assertIsNone(d.selector)


class TestSecretScrubbing(unittest.TestCase):
    def test_scrub_patterns(self):
        text = scrub_text("api_key=abcd1234 and Bearer tokentokentoken")
        self.assertNotIn("abcd1234", text)
        self.assertIn("***REDACTED***", text)

    def test_telegram_token_scrubbed_from_env(self):
        os.environ["TELEGRAM_BOT_TOKEN"] = "123456:ABCDEF-telegram-token"
        try:
            out = scrub_text("token was 123456:ABCDEF-telegram-token used")
            self.assertNotIn("ABCDEF-telegram-token", out)
        finally:
            os.environ.pop("TELEGRAM_BOT_TOKEN", None)


class TestSafetyNotifyCritical(unittest.TestCase):
    def test_halt_triggers_critical_callback(self):
        seen = []

        def on_notify(event, message, **kwargs):
            seen.append((event, kwargs.get("severity"), message))

        safety = SafetySystem(on_notify=on_notify)
        safety.halt("TEST_HALT")
        self.assertEqual(seen[0][0], "HALT")
        self.assertEqual(seen[0][1], "CRITICAL")


if __name__ == "__main__":
    unittest.main()
