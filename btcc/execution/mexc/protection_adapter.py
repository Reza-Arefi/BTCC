"""MEXC Spot exchange-resident protection adapter (LEGACY / OPTIONAL).

**********************************************************************
Production T1 path is BOT-MANAGED and does NOT use this adapter.

Native Spot protective-stop / trailing create APIs remain unconfirmed.
This module stays fail-closed so it cannot silently invent endpoints.
It is retained only for optional/legacy EXCHANGE_RESIDENT tests.
**********************************************************************
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from btcc.execution.mexc.write_client import MexcWriteClient
from btcc.execution.write_gate import WriteGate, CLOSED_WRITE_GATE
from btcc.safety.no_trading import TradingForbiddenError


class SpotProtectionApiNotConfirmedError(TradingForbiddenError):
    """Raised until MEXC confirms Spot protection create/cancel/replace API."""


@dataclass(frozen=True)
class ProtectionEndpointConfig:
    create_path: str = ""
    cancel_path: str = ""
    query_path: str = ""
    order_type: str = ""
    extra_create_params: dict[str, Any] | None = None

    @property
    def is_configured(self) -> bool:
        return bool(self.create_path and self.order_type)


def load_protection_endpoint_from_env() -> ProtectionEndpointConfig:
    return ProtectionEndpointConfig(
        create_path=(os.environ.get("BTCC_MEXC_PROTECTION_CREATE_PATH") or "").strip(),
        cancel_path=(os.environ.get("BTCC_MEXC_PROTECTION_CANCEL_PATH") or "").strip(),
        query_path=(os.environ.get("BTCC_MEXC_PROTECTION_QUERY_PATH") or "").strip(),
        order_type=(os.environ.get("BTCC_MEXC_PROTECTION_ORDER_TYPE") or "").strip(),
    )


class MexcSpotProtectionAdapter:
    """Legacy ProtectionBroker — NOT used by bot-managed T1 production canary."""

    def __init__(
        self,
        write_client: MexcWriteClient | None = None,
        *,
        gate: WriteGate | None = None,
        endpoint: ProtectionEndpointConfig | None = None,
        transport: Any | None = None,
    ) -> None:
        self.write_client = write_client
        self.gate = gate or CLOSED_WRITE_GATE
        self.endpoint = endpoint or load_protection_endpoint_from_env()
        self._transport = transport
        self._store: dict[str, dict[str, Any]] = {}

    def _assert_ready(self, action: str) -> None:
        self.gate.assert_protection_write(action)
        if not self.endpoint.is_configured and self._transport is None:
            raise SpotProtectionApiNotConfirmedError(
                f"MEXC Spot protection API not confirmed/configured for {action}. "
                "Production T1 uses bot-managed T1PriceMonitor instead."
            )

    def place_protective_stop(
        self,
        *,
        symbol: str,
        quantity: float,
        stop_price: float,
        client_order_id: str,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._assert_ready("place_protective_stop")
        if self._transport is not None:
            ack = self._transport(
                "place",
                symbol=symbol,
                quantity=quantity,
                stop_price=stop_price,
                client_order_id=client_order_id,
                meta=meta or {},
            )
            pid = str(ack.get("protection_id") or ack.get("order_id") or client_order_id)
            self._store[pid] = dict(ack)
            return {"protection_id": pid, **ack}
        raise SpotProtectionApiNotConfirmedError("Protection HTTP not live without transport")

    def cancel_protective_stop(self, *, symbol: str, protection_id: str) -> dict[str, Any]:
        self._assert_ready("cancel_protective_stop")
        if self._transport is not None:
            return self._transport("cancel", symbol=symbol, protection_id=protection_id)
        raise SpotProtectionApiNotConfirmedError("cancel protection HTTP not live without transport")

    def replace_protective_stop(
        self,
        *,
        symbol: str,
        old_protection_id: str,
        quantity: float,
        stop_price: float,
        client_order_id: str,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._assert_ready("replace_protective_stop")
        if self._transport is not None:
            return self._transport(
                "replace",
                symbol=symbol,
                old_protection_id=old_protection_id,
                quantity=quantity,
                stop_price=stop_price,
                client_order_id=client_order_id,
                meta=meta or {},
            )
        raise SpotProtectionApiNotConfirmedError("replace protection HTTP not live without transport")

    def get_protective_stop(self, *, symbol: str, protection_id: str) -> dict[str, Any] | None:
        if self._transport is not None:
            return self._transport("get", symbol=symbol, protection_id=protection_id)
        return self._store.get(protection_id)
