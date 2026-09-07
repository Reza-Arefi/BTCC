"""Notification channel abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class NotificationEvent:
    event: str
    severity: Severity
    message: str
    symbol: str | None = None
    trade_id: str | None = None
    order_id: str | None = None
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NotificationResult:
    ok: bool
    channel: str
    reason: str = "OK"
    skipped: bool = False


class NotificationChannel(ABC):
    name: str = "base"

    @abstractmethod
    def configured(self) -> bool:
        ...

    @abstractmethod
    def send(self, event: NotificationEvent) -> NotificationResult:
        """Must never raise into the trading engine — return failure instead."""
        ...

    def status(self) -> dict[str, Any]:
        return {"channel": self.name, "configured": self.configured()}
