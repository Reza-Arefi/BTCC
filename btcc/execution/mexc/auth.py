"""HMAC-SHA256 request signing for MEXC Spot v3 SIGNED endpoints.

Official rule (Spot v3 Introduction):
  signature = HMAC_SHA256(secret, totalParams).hexdigest()  # lowercase only
  totalParams = query_string (+ request body if any)

We only issue GET signed requests in Stage 3 (read-only).
"""

from __future__ import annotations

import hashlib
import hmac
import time
from typing import Any
from urllib.parse import urlencode


def server_timestamp_ms() -> int:
    return int(time.time() * 1000)


def build_signed_query(
    params: dict[str, Any],
    *,
    api_secret: str,
    recv_window_ms: int = 5000,
    timestamp_ms: int | None = None,
) -> str:
    """Return URL-encoded query string including timestamp, recvWindow, signature."""
    payload = {k: v for k, v in params.items() if v is not None}
    payload["timestamp"] = int(timestamp_ms if timestamp_ms is not None else server_timestamp_ms())
    payload["recvWindow"] = int(recv_window_ms)
    # Stable order for tests: sort keys (MEXC accepts any order; signature covers exact string).
    query = urlencode(sorted((str(k), str(v)) for k, v in payload.items()))
    sig = hmac.new(
        api_secret.encode("utf-8"),
        query.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"{query}&signature={sig}"


def sign_query_string(query: str, api_secret: str) -> str:
    """Sign an already-built query string (without signature=)."""
    return hmac.new(api_secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256).hexdigest()
