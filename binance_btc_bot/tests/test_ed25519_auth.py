"""Ed25519 authentication tests for Binance Spot REST signing."""

from __future__ import annotations

import base64
import io
import logging
import os
import tempfile
import unittest
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)

from binance_btc_bot.credentials import (
    BinanceCredentials,
    CredentialError,
    load_binance_credentials_from_env,
    validate_binance_credentials,
)
from binance_btc_bot.exchange.binance import BinanceExchange
from binance_btc_bot.exchange.signing import (
    Ed25519RequestSigner,
    SigningError,
    build_signature_payload,
    encode_signature_for_query,
)
from binance_btc_bot.secrets import ScrubbingFilter, clear_runtime_secrets, scrub_text


def _make_pem_pair() -> tuple[bytes, bytes]:
    key = Ed25519PrivateKey.generate()
    priv = key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
    pub = key.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
    return priv, pub


class TestEd25519Signing(unittest.TestCase):
    def setUp(self) -> None:
        clear_runtime_secrets()
        self.priv_pem, self.pub_pem = _make_pem_pair()

    def tearDown(self) -> None:
        clear_runtime_secrets()

    def test_ed25519_signing_works(self):
        signer = Ed25519RequestSigner.from_pem_bytes(self.priv_pem)
        params = {
            "symbol": "BTCUSDT",
            "side": "SELL",
            "type": "LIMIT",
            "timeInForce": "GTC",
            "quantity": "1",
            "price": "0.2",
            "timestamp": 1668481559918,
            "recvWindow": 5000,
        }
        payload, sig_b64 = signer.sign_params(params)
        self.assertIn("symbol=BTCUSDT", payload)
        self.assertIn("timestamp=1668481559918", payload)
        # Base64 signature, case-sensitive, verifiable
        raw = base64.b64decode(sig_b64)
        self.assertEqual(len(raw), 64)
        pub = signer._private_key.public_key()
        pub.verify(raw, payload.encode("ASCII"))
        # Query encoding percent-encodes +/= 
        enc = encode_signature_for_query(sig_b64 + "==")
        self.assertIn("%3D", enc)

    def test_public_key_pem_spki_format(self):
        signer = Ed25519RequestSigner.from_pem_bytes(self.priv_pem)
        pem = signer.public_key_pem()
        self.assertTrue(pem.startswith("-----BEGIN PUBLIC KEY-----"))
        self.assertIn("-----END PUBLIC KEY-----", pem)

    def test_missing_private_key_fails_safely(self):
        with self.assertRaises(SigningError):
            Ed25519RequestSigner.from_pem_file("/tmp/does-not-exist-binance-ed25519.pem")
        creds = BinanceCredentials(api_key="abc", private_key_path="")
        with self.assertRaises(CredentialError) as ctx:
            validate_binance_credentials(creds, required=True)
        msg = str(ctx.exception)
        self.assertIn("BINANCE_ED25519_PRIVATE_KEY_PATH", msg)
        self.assertNotIn("BEGIN", msg)

    def test_invalid_private_key_fails_safely(self):
        with self.assertRaises(SigningError) as ctx:
            Ed25519RequestSigner.from_pem_bytes(b"not-a-pem-key")
        self.assertNotIn("not-a-pem-key", str(ctx.exception))

        with tempfile.NamedTemporaryFile("w", suffix=".pem", delete=False) as f:
            f.write("-----BEGIN PRIVATE KEY-----\nAAAA\n-----END PRIVATE KEY-----\n")
            path = f.name
        try:
            creds = BinanceCredentials(api_key="abc", private_key_path=path)
            with self.assertRaises(CredentialError) as ctx:
                validate_binance_credentials(creds, required=True)
            self.assertNotIn("AAAA", str(ctx.exception))
        finally:
            Path(path).unlink(missing_ok=True)

    def test_private_key_never_logged(self):
        pem_text = self.priv_pem.decode("ascii")
        os.environ["BINANCE_API_KEY"] = "TESTAPIKEYVALUE999"
        try:
            logger = logging.getLogger("ed25519_scrub_test")
            logger.handlers.clear()
            logger.setLevel(logging.INFO)
            buf = io.StringIO()
            handler = logging.StreamHandler(buf)
            handler.addFilter(ScrubbingFilter())
            logger.addHandler(handler)
            # Simulate accidental leak attempts
            logger.info("pem=%s", pem_text)
            logger.info("api_key=%s", os.environ["BINANCE_API_KEY"])
            text = buf.getvalue()
            # Placeholder headers may remain; PEM body must not.
            for line in pem_text.splitlines():
                if line and not line.startswith("-----"):
                    self.assertNotIn(line, text)
            self.assertNotIn("TESTAPIKEYVALUE999", text)
            self.assertIn("***REDACTED***", text)
        finally:
            os.environ.pop("BINANCE_API_KEY", None)

    def test_live_disabled_blocks_orders_with_valid_creds(self):
        signer = Ed25519RequestSigner.from_pem_bytes(self.priv_pem)
        ex = BinanceExchange(
            api_key="validkey",
            signer=signer,
            dry_run=False,
            live_enabled=False,
        )
        allowed, reason = ex._writes_allowed()
        self.assertFalse(allowed)
        self.assertEqual(reason, "LIVE_DISABLED")

    def test_status_masked_summary_hides_path_and_pem(self):
        with tempfile.NamedTemporaryFile("wb", suffix=".pem", delete=False) as f:
            f.write(self.priv_pem)
            path = f.name
        try:
            os.environ["BINANCE_API_KEY"] = "KEY1234567890"
            os.environ["BINANCE_ED25519_PRIVATE_KEY_PATH"] = path
            creds = load_binance_credentials_from_env()
            validate_binance_credentials(creds, required=True)
            masked = creds.masked_summary()
            blob = str(masked)
            self.assertEqual(masked["api_key"], "SET")
            self.assertEqual(masked["ed25519_private_key"], "SET")
            self.assertEqual(masked["signer"], "READY")
            self.assertNotIn(path, blob)
            self.assertNotIn("KEY1234567890", blob)
            self.assertNotIn("BEGIN", blob)
            cleaned = scrub_text(f"pem={self.priv_pem.decode()}")
            for line_pem in self.priv_pem.decode().splitlines():
                if line_pem and not line_pem.startswith("-----"):
                    self.assertNotIn(line_pem, cleaned)
        finally:
            os.environ.pop("BINANCE_API_KEY", None)
            os.environ.pop("BINANCE_ED25519_PRIVATE_KEY_PATH", None)
            Path(path).unlink(missing_ok=True)

    def test_payload_percent_encodes_non_ascii(self):
        payload = build_signature_payload({"symbol": "１２３４５６", "timestamp": 1})
        self.assertIn("%EF%BC%91", payload)


class TestEngineStatusNoSecrets(unittest.TestCase):
    def test_status_text_no_pem(self):
        from binance_btc_bot.config_loader import load_config
        from binance_btc_bot.execution.engine import BinanceBotEngine

        priv, _ = _make_pem_pair()
        with tempfile.NamedTemporaryFile("wb", suffix=".pem", delete=False) as f:
            f.write(priv)
            path = f.name
        os.environ["BINANCE_API_KEY"] = "STATUSKEYHIDE"
        os.environ["BINANCE_ED25519_PRIVATE_KEY_PATH"] = path
        os.environ["DRY_RUN"] = "true"
        try:
            engine = BinanceBotEngine(load_config())
            text = engine.status_text()
            self.assertIn("DISABLED", text)
            self.assertIn("T1", text)
            self.assertIn("NONE", text)
            self.assertNotIn("STATUSKEYHIDE", text)
            self.assertNotIn(path, text)
            self.assertNotIn("BEGIN PRIVATE", text)
            self.assertIn("ed25519=SET", text)
        finally:
            os.environ.pop("BINANCE_API_KEY", None)
            os.environ.pop("BINANCE_ED25519_PRIVATE_KEY_PATH", None)
            Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
