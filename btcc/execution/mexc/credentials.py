"""MEXC Spot credential loading — secrets never logged or committed.

Environment variables (preferred):
  MEXC_API_KEY
  MEXC_API_SECRET

Optional aliases:
  BTCC_MEXC_API_KEY
  BTCC_MEXC_API_SECRET

Optional:
  MEXC_API_BASE_URL   (default https://api.mexc.com)
  MEXC_RECV_WINDOW_MS (default 5000, max 60000)

Stage 3 requires a READ-ONLY key with Spot permissions such as:
  SPOT_ACCOUNT_READ
  SPOT_DEAL_READ   (needed for openOrders / query order)

Do NOT grant SPOT_DEAL_WRITE for Stage 3 keys.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


class MissingCredentialsError(RuntimeError):
    """Raised when read-only MEXC credentials are not configured."""


@dataclass(frozen=True)
class MexcCredentials:
    api_key: str
    api_secret: str
    base_url: str = "https://api.mexc.com"
    recv_window_ms: int = 5000

    def __repr__(self) -> str:
        return (
            f"MexcCredentials(api_key='***{self.api_key[-4:] if len(self.api_key) >= 4 else ''}***', "
            f"api_secret='***REDACTED***', base_url={self.base_url!r}, "
            f"recv_window_ms={self.recv_window_ms})"
        )

    def __str__(self) -> str:
        return self.__repr__()


def load_mexc_credentials(*, require: bool = True) -> MexcCredentials | None:
    key = (os.environ.get("MEXC_API_KEY") or os.environ.get("BTCC_MEXC_API_KEY") or "").strip()
    secret = (os.environ.get("MEXC_API_SECRET") or os.environ.get("BTCC_MEXC_API_SECRET") or "").strip()
    base = (os.environ.get("MEXC_API_BASE_URL") or "https://api.mexc.com").strip().rstrip("/")
    recv_raw = (os.environ.get("MEXC_RECV_WINDOW_MS") or "5000").strip()
    try:
        recv = int(recv_raw)
    except ValueError as e:
        raise MissingCredentialsError("MEXC_RECV_WINDOW_MS must be an integer") from e
    if recv <= 0 or recv > 60000:
        raise MissingCredentialsError("MEXC_RECV_WINDOW_MS must be in 1..60000")

    if not key or not secret:
        if require:
            raise MissingCredentialsError(
                "MEXC read-only credentials missing. Set MEXC_API_KEY and MEXC_API_SECRET "
                "(or BTCC_MEXC_API_KEY / BTCC_MEXC_API_SECRET)."
            )
        return None
    return MexcCredentials(api_key=key, api_secret=secret, base_url=base, recv_window_ms=recv)


def redact_secrets(text: str, creds: MexcCredentials | None = None) -> str:
    """Remove credential material from strings before logging."""
    out = str(text)
    if creds is not None:
        if creds.api_secret:
            out = out.replace(creds.api_secret, "***REDACTED***")
        if creds.api_key:
            out = out.replace(creds.api_key, "***API_KEY***")
    for env_name in ("MEXC_API_SECRET", "BTCC_MEXC_API_SECRET", "MEXC_API_KEY", "BTCC_MEXC_API_KEY"):
        val = os.environ.get(env_name)
        if val:
            out = out.replace(val, "***REDACTED***")
    return out
