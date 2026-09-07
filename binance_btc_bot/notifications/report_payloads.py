"""Build Telegram report payloads from authoritative trade/lifecycle fields.

No P/L invention — only maps already-computed accounting / order fields.
"""

from __future__ import annotations

from typing import Any, Mapping


def trade_open_payload(
    *,
    symbol: str,
    strategy: str,
    selector: str | None,
    signal: Mapping[str, Any] | None = None,
    entry: Mapping[str, Any],
    fees: Mapping[str, Any] | None = None,
    protection: Mapping[str, Any] | None = None,
    portfolio_before: Mapping[str, Any] | None = None,
    portfolio_after: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "kind": "trade_open",
        "payload": {
            "symbol": symbol,
            "strategy": strategy,
            "selector": selector or "NONE",
            "signal": dict(signal or {}),
            "entry": dict(entry),
            "fees": dict(fees or {}),
            "protection": dict(protection or {}),
            "portfolio_before": dict(portfolio_before or {}),
            "portfolio_after": dict(portfolio_after or {}),
        },
    }


def trade_close_payload(
    *,
    symbol: str,
    strategy: str,
    selector: str | None,
    close_reason: str,
    entry: Mapping[str, Any],
    exit: Mapping[str, Any],
    result: Mapping[str, Any],
    trailing: Mapping[str, Any] | None = None,
    duration: str | None = None,
    portfolio_before: Mapping[str, Any] | None = None,
    portfolio_after: Mapping[str, Any] | None = None,
    reconciliation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "kind": "trade_close",
        "payload": {
            "symbol": symbol,
            "strategy": strategy,
            "selector": selector or "NONE",
            "duration": duration,
            "close_reason": close_reason,
            "entry": dict(entry),
            "exit": dict(exit),
            "trailing": dict(trailing or {}),
            "result": dict(result),
            "portfolio_before": dict(portfolio_before or {}),
            "portfolio_after": dict(portfolio_after or {}),
            "reconciliation": dict(reconciliation or {}),
        },
    }


def protection_payload(
    *,
    kind: str,
    symbol: str,
    quantity: Any = None,
    reason: str | None = None,
    protection_state: str | None = None,
    binance_verified: bool | None = None,
    new_entries_status: str | None = None,
    oco_id: Any = None,
    timestamp: Any = None,
) -> dict[str, Any]:
    return {
        "kind": "protection",
        "symbol": symbol,
        "quantity": quantity,
        "reason": reason,
        "protection_state": protection_state,
        "binance_verified": binance_verified,
        "new_entries_status": new_entries_status,
        "oco_id": oco_id,
        "timestamp": timestamp,
        "event": kind,
    }
