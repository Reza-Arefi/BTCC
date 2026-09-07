"""Binance API credential loading — Ed25519 (not HMAC secret).

Environment:
  BINANCE_API_KEY=                    # API key id from Binance after uploading public key
  BINANCE_ED25519_PRIVATE_KEY_PATH=   # path to PKCS#8 PEM private key on this server
  BINANCE_ED25519_PRIVATE_KEY_PASSPHRASE=  # optional, if PEM is encrypted

Never treat the Ed25519 private key as BINANCE_API_SECRET.
Never log or return PEM contents.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from binance_btc_bot.exchange.signing import Ed25519RequestSigner, SigningError


class CredentialError(RuntimeError):
    """Raised when credentials are missing/invalid for a required operation."""


@dataclass
class BinanceCredentials:
    api_key: str
    private_key_path: str
    auth_mode: str = "ed25519"
    _signer: Ed25519RequestSigner | None = field(default=None, repr=False, compare=False)
    _load_error: str | None = field(default=None, repr=False, compare=False)

    @property
    def present(self) -> bool:
        return bool(self.api_key) and bool(self.private_key_path)

    @property
    def signer_ready(self) -> bool:
        return self.get_signer() is not None

    def masked_summary(self) -> dict[str, str]:
        """Safe for status/logs — never includes key material or filesystem path."""
        path_state = "MISSING"
        if self.private_key_path:
            path_state = "SET" if Path(self.private_key_path).expanduser().is_file() else "PATH_MISSING"
        signer_state = "READY" if self.signer_ready else ("ERROR" if self._load_error else "NOT_LOADED")
        return {
            "api_key": "SET" if self.api_key else "MISSING",
            "ed25519_private_key": path_state,
            "signer": signer_state,
            "auth_mode": self.auth_mode,
            "present": "true" if self.present else "false",
        }

    def get_signer(self) -> Ed25519RequestSigner | None:
        if self._signer is not None:
            return self._signer
        if not self.private_key_path:
            return None
        try:
            passphrase = os.environ.get("BINANCE_ED25519_PRIVATE_KEY_PASSPHRASE")
            self._signer = Ed25519RequestSigner.from_pem_file(
                self.private_key_path,
                password=passphrase if passphrase else None,
            )
            self._load_error = None
            # Register PEM contents for scrubbing without exposing them.
            try:
                from binance_btc_bot.secrets import register_runtime_secret

                pem = Path(self.private_key_path).expanduser().read_text(encoding="utf-8", errors="ignore")
                register_runtime_secret(pem)
                if passphrase:
                    register_runtime_secret(passphrase)
            except Exception:  # noqa: BLE001
                pass
            return self._signer
        except SigningError as e:
            self._load_error = str(e)
            return None
        except Exception as e:  # noqa: BLE001
            self._load_error = f"{type(e).__name__}"
            return None


def load_binance_credentials_from_env() -> BinanceCredentials:
    key = (os.environ.get("BINANCE_API_KEY") or "").strip()
    path = (os.environ.get("BINANCE_ED25519_PRIVATE_KEY_PATH") or "").strip()
    return BinanceCredentials(api_key=key, private_key_path=path, auth_mode="ed25519")


def validate_binance_credentials(
    creds: BinanceCredentials | None = None,
    *,
    required: bool,
) -> BinanceCredentials:
    """Validate credential presence / loadability.

    When required=False (public/dry paths), missing keys are allowed.
    When required=True, API key + readable Ed25519 private key are required.
    Never returns or logs private key material.
    """
    c = creds or load_binance_credentials_from_env()
    if not required:
        # Best-effort preload for scrubbing if path is set; ignore failures.
        if c.private_key_path:
            c.get_signer()
        return c
    if not c.api_key and not c.private_key_path:
        raise CredentialError(
            "BINANCE_API_KEY and BINANCE_ED25519_PRIVATE_KEY_PATH are required"
        )
    if not c.api_key:
        raise CredentialError("BINANCE_API_KEY is required")
    if not c.private_key_path:
        raise CredentialError("BINANCE_ED25519_PRIVATE_KEY_PATH is required")
    signer = c.get_signer()
    if signer is None:
        raise CredentialError(c._load_error or "Ed25519 private key could not be loaded")
    return c


def credentials_required_for_mode(*, live_enabled: bool, dry_run: bool, need_account: bool = False) -> bool:
    """Credentials needed for signed private endpoints."""
    if need_account:
        return True
    if live_enabled and not dry_run:
        return True
    return False
