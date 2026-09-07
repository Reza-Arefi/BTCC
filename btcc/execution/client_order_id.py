"""Deterministic client_order_id contract (exchange-neutral, restart-safe).

Format (ASCII, deterministic):

    BTCC-{KIND}-{INTENT_ID_HEX16}-{SEQ}

where:
  KIND ∈ {ENT, EXT, STP}  (entry / exit / protective stop)
  INTENT_ID_HEX16 = first 16 hex chars of sha256(canonical intent key)
  SEQ = zero-padded attempt sequence (001, 002, ...)

Canonical intent key (UTF-8, '|' separated):
  strategy_version|symbol|side|signal_ts|candle_ts|selected_exit|kind|intent_nonce

Uniqueness:
  - Same logical action → same INTENT_ID_HEX16 across restarts
  - SEQ increments only for explicit retries of the same intent+kind
  - Never rely on in-memory UUID alone

Length is kept short and alphanumeric+hyphen for broad exchange compatibility;
exchange-specific max length checks belong to a later broker stage.
"""

from __future__ import annotations

import hashlib
import re
from enum import Enum


class OrderKind(str, Enum):
    ENTRY = "ENT"
    EXIT = "EXT"
    PROTECTIVE_STOP = "STP"


_CLIENT_ID_RE = re.compile(r"^BTCC-(ENT|EXT|STP)-[0-9a-f]{16}-\d{3}$")


def intent_fingerprint(
    *,
    strategy_version: str,
    symbol: str,
    side: str,
    signal_ts: str,
    candle_ts: str,
    selected_exit: str,
    kind: OrderKind | str,
    intent_nonce: str = "",
) -> str:
    """Return 16-hex fingerprint used inside client_order_id / durable intent."""
    k = OrderKind(kind) if not isinstance(kind, OrderKind) else kind
    canonical = "|".join(
        [
            str(strategy_version).strip(),
            str(symbol).strip().upper(),
            str(side).strip().upper(),
            str(signal_ts).strip(),
            str(candle_ts).strip(),
            str(selected_exit).strip(),
            k.value,
            str(intent_nonce).strip(),
        ]
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def build_client_order_id(
    *,
    strategy_version: str,
    symbol: str,
    side: str,
    signal_ts: str,
    candle_ts: str,
    selected_exit: str,
    kind: OrderKind | str,
    attempt: int = 1,
    intent_nonce: str = "",
) -> str:
    if attempt < 1 or attempt > 999:
        raise ValueError("attempt must be in 1..999")
    fp = intent_fingerprint(
        strategy_version=strategy_version,
        symbol=symbol,
        side=side,
        signal_ts=signal_ts,
        candle_ts=candle_ts,
        selected_exit=selected_exit,
        kind=kind,
        intent_nonce=intent_nonce,
    )
    k = OrderKind(kind) if not isinstance(kind, OrderKind) else kind
    return f"BTCC-{k.value}-{fp}-{attempt:03d}"


def parse_client_order_id(client_order_id: str) -> dict[str, str | int]:
    if not _CLIENT_ID_RE.match(client_order_id):
        raise ValueError(f"Invalid client_order_id format: {client_order_id!r}")
    _, kind, fp, seq = client_order_id.split("-")
    return {"kind": kind, "fingerprint": fp, "attempt": int(seq)}


def is_valid_client_order_id(client_order_id: str) -> bool:
    return bool(_CLIENT_ID_RE.match(client_order_id or ""))
