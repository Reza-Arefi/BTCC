"""Configurable failure scenarios for the Stage-4 simulated exchange."""

from __future__ import annotations

from enum import Enum


class SimScenario(str, Enum):
    NORMAL = "NORMAL"
    REJECTED = "REJECTED"
    TIMEOUT_BEFORE_ACCEPTANCE = "TIMEOUT_BEFORE_ACCEPTANCE"
    TIMEOUT_AFTER_ACCEPTANCE = "TIMEOUT_AFTER_ACCEPTANCE"
    PARTIAL_FILL = "PARTIAL_FILL"
    DELAYED_FILL = "DELAYED_FILL"
    MULTI_FILL = "MULTI_FILL"
    DUPLICATE_REQUEST = "DUPLICATE_REQUEST"
    EXCHANGE_STATE_LOST = "EXCHANGE_STATE_LOST"
    NETWORK_FAILURE = "NETWORK_FAILURE"


class SimBrokerError(RuntimeError):
    """Base simulated-broker failure."""


class SimTimeoutError(SimBrokerError):
    """Network timeout. accepted=True means exchange already recorded the order."""

    def __init__(
        self,
        message: str,
        *,
        accepted: bool,
        exchange_order_id: str | None = None,
        client_order_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.accepted = bool(accepted)
        self.exchange_order_id = exchange_order_id
        self.client_order_id = client_order_id


class SimNetworkError(SimBrokerError):
    """Generic network / DNS failure before outcome is known."""


class SimRejectedError(SimBrokerError):
    """Exchange explicitly rejected the order."""
