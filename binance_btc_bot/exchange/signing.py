"""Binance Spot REST Ed25519 request signing.

Per Binance Spot API docs (Ed25519 keys):
  1. Build parameter=value&... payload (url-encoded / percent-encoded).
  2. Sign payload bytes with Ed25519 private key (PEM PKCS#8).
  3. Base64-encode the raw signature (case-sensitive).
  4. Percent-encode the base64 string when placing it in a query string.

This is NOT HMAC. Do not use BINANCE_API_SECRET for Ed25519 keys.
"""

from __future__ import annotations

import base64
import urllib.parse
from pathlib import Path
from typing import Any, Mapping

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
    load_pem_private_key,
)


class SigningError(RuntimeError):
    """Safe error for missing/invalid private keys — never includes key material."""


class Ed25519RequestSigner:
    """Loads an Ed25519 PEM private key from disk and signs Binance payloads."""

    def __init__(self, private_key: Ed25519PrivateKey, *, source_label: str = "memory") -> None:
        self._private_key = private_key
        self.source_label = source_label  # path or "memory" — never the PEM itself

    @classmethod
    def from_pem_file(
        cls,
        path: str | Path,
        *,
        password: str | bytes | None = None,
    ) -> "Ed25519RequestSigner":
        p = Path(path).expanduser()
        if not p.is_file():
            raise SigningError("Ed25519 private key file not found")
        try:
            raw = p.read_bytes()
        except OSError as e:
            raise SigningError(f"Cannot read Ed25519 private key file: {type(e).__name__}") from e
        return cls.from_pem_bytes(raw, password=password, source_label="file")

    @classmethod
    def from_pem_bytes(
        cls,
        pem: bytes,
        *,
        password: str | bytes | None = None,
        source_label: str = "memory",
    ) -> "Ed25519RequestSigner":
        if not pem or not pem.strip():
            raise SigningError("Ed25519 private key PEM is empty")
        pwd: bytes | None
        if password is None or password == "":
            pwd = None
        elif isinstance(password, bytes):
            pwd = password
        else:
            pwd = str(password).encode("utf-8")
        try:
            key = load_pem_private_key(pem, password=pwd)
        except Exception as e:  # noqa: BLE001 — never surface PEM/password contents
            raise SigningError(f"Invalid Ed25519 private key PEM ({type(e).__name__})") from None
        if not isinstance(key, Ed25519PrivateKey):
            raise SigningError("Private key is not an Ed25519 key")
        return cls(key, source_label=source_label)

    def public_key_pem(self) -> str:
        """SPKI PEM public key — the format Binance expects for upload."""
        pem = self._private_key.public_key().public_bytes(
            Encoding.PEM,
            PublicFormat.SubjectPublicKeyInfo,
        )
        return pem.decode("ascii")

    def sign_payload(self, payload: str) -> str:
        """Return base64 signature for an already-constructed payload string."""
        sig = self._private_key.sign(payload.encode("ASCII"))
        return base64.b64encode(sig).decode("ascii")

    def sign_params(self, params: Mapping[str, Any]) -> tuple[str, str]:
        """Return (payload, base64_signature) for Binance signed endpoints."""
        payload = build_signature_payload(params)
        return payload, self.sign_payload(payload)


def build_signature_payload(params: Mapping[str, Any]) -> str:
    """Parameter string used as the Ed25519 signature payload.

    Binance requires params (excluding ``signature``) sorted alphabetically by
    name, formatted as ``key=value`` pairs joined with ``&``.
    REST callers historically used urllib percent-encoding; alphabetical order
    is mandatory for WebSocket API signed methods and is safe for REST.
    """
    items = [
        (str(k), _stringify_param(v))
        for k, v in sorted(params.items(), key=lambda kv: str(kv[0]))
        if v is not None and str(k) != "signature"
    ]
    # quote_via=quote keeps %HH style (required for non-ASCII; fine for ASCII).
    return urllib.parse.urlencode(items, doseq=True, quote_via=urllib.parse.quote)


def _stringify_param(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def encode_signature_for_query(signature_b64: str) -> str:
    """Percent-encode base64 signature for use in a URL query string."""
    return urllib.parse.quote(signature_b64, safe="")
