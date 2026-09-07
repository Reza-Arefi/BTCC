"""Canonical T1 exit geometry — production target (exchange-neutral math).

Single source of truth for production T1 parameters. Strategy layer has no
MEXC-specific API logic. T1 is bot-managed via T1PriceMonitor — do NOT rewrite
parameters to fit exchange trailing minima.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_FLOOR
from enum import Enum
from typing import Any, Mapping


_FROZEN_STOP_LOSS_PCT = 0.0075  # -0.75%
_FROZEN_ACTIVATION_PCT = 0.0075  # +0.75%
_FROZEN_TRAILING_PCT = 0.0025  # 0.25% trail


@dataclass(frozen=True)
class T1Config:
    """Immutable canonical T1 definition (production source of truth)."""

    stop_loss_pct: float = _FROZEN_STOP_LOSS_PCT
    activation_pct: float = _FROZEN_ACTIVATION_PCT
    trailing_pct: float = _FROZEN_TRAILING_PCT
    name: str = "T1_sl0p75_act0p75_dist0p25"
    key: str = "trail_1"

    def __post_init__(self) -> None:
        if abs(float(self.stop_loss_pct) - _FROZEN_STOP_LOSS_PCT) > 1e-15:
            raise ValueError(f"T1 stop_loss_pct is frozen at {_FROZEN_STOP_LOSS_PCT}")
        if abs(float(self.activation_pct) - _FROZEN_ACTIVATION_PCT) > 1e-15:
            raise ValueError(f"T1 activation_pct is frozen at {_FROZEN_ACTIVATION_PCT}")
        if abs(float(self.trailing_pct) - _FROZEN_TRAILING_PCT) > 1e-15:
            raise ValueError(
                f"T1 trailing_pct is frozen at {_FROZEN_TRAILING_PCT} (0.25%); "
                f"got {self.trailing_pct}. Do NOT change T1 to fit the exchange."
            )

    @property
    def distance_pct(self) -> float:
        return self.trailing_pct


T1 = T1Config()
T1Geometry = T1Config


class T1State(str, Enum):
    INITIAL_STOP = "INITIAL_STOP"
    TRAILING_INACTIVE = "TRAILING_INACTIVE"
    TRAILING_ACTIVE = "TRAILING_ACTIVE"
    EXIT_TRIGGERED = "EXIT_TRIGGERED"
    CLOSED = "CLOSED"


class T1ParameterUnsupportedError(RuntimeError):
    """Raised when an exchange cannot implement frozen T1 (legacy native path)."""


def initial_stop_price(entry_price: float, geo: T1Config = T1) -> float:
    if entry_price <= 0:
        raise ValueError("entry_price must be > 0")
    return float(entry_price) * (1.0 - geo.stop_loss_pct)


def activation_price(entry_price: float, geo: T1Config = T1) -> float:
    if entry_price <= 0:
        raise ValueError("entry_price must be > 0")
    return float(entry_price) * (1.0 + geo.activation_pct)


def trailing_stop_from_high(high_water: float, geo: T1Config = T1) -> float:
    if high_water <= 0:
        raise ValueError("high_water must be > 0")
    return float(high_water) * (1.0 - geo.trailing_pct)


def ratchet_stop(current_stop: float, candidate: float) -> float:
    return max(float(current_stop), float(candidate))


def initial_stop_price_decimal(entry_price: Decimal, geo: T1Config = T1) -> Decimal:
    if entry_price <= 0:
        raise ValueError("entry_price must be > 0")
    return entry_price * (Decimal("1") - Decimal(str(geo.stop_loss_pct)))


def activation_price_decimal(entry_price: Decimal, geo: T1Config = T1) -> Decimal:
    if entry_price <= 0:
        raise ValueError("entry_price must be > 0")
    return entry_price * (Decimal("1") + Decimal(str(geo.activation_pct)))


def trailing_stop_from_high_decimal(high_water: Decimal, geo: T1Config = T1) -> Decimal:
    if high_water <= 0:
        raise ValueError("high_water must be > 0")
    return high_water * (Decimal("1") - Decimal(str(geo.trailing_pct)))


def round_stop_conservative(stop: Decimal, tick: Decimal) -> Decimal:
    if tick <= 0:
        return stop
    units = (stop / tick).to_integral_value(rounding=ROUND_FLOOR)
    floored = units * tick
    if floored < stop:
        return floored + tick
    return floored


def assert_exchange_supports_t1_trailing(
    *,
    min_trailing_pct: float | None,
    supports_trailing: bool | None = None,
    supports_initial_stop: bool | None = None,
    geo: T1Config = T1,
) -> None:
    """Legacy helper for native-exchange path (not used by bot-managed T1 production)."""
    if supports_initial_stop is False:
        raise T1ParameterUnsupportedError("Exchange does not support initial stop")
    if supports_trailing is False:
        raise T1ParameterUnsupportedError("Exchange does not support trailing")
    if min_trailing_pct is not None and float(min_trailing_pct) > float(geo.trailing_pct) + 1e-15:
        raise T1ParameterUnsupportedError(
            f"Exchange min trailing {min_trailing_pct} > frozen T1 {geo.trailing_pct}"
        )


@dataclass
class T1Machine:
    """Pure bot-managed T1 state machine — exchange-independent."""

    intent_id: str
    symbol: str
    entry_vwap: float
    executed_quantity: float
    client_order_id: str = ""
    requested_quantity: float | None = None
    side: str = "BUY"
    config: T1Config = field(default_factory=lambda: T1)
    state: T1State = T1State.INITIAL_STOP
    activated: bool = False
    highest_price: float = 0.0
    stop_price: float = 0.0
    initial_stop_price_value: float = 0.0
    activation_price_value: float = 0.0
    protection_order_id: str | None = None
    protection_status: str | None = None
    timestamps: dict[str, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.executed_quantity <= 0:
            raise ValueError("executed_quantity must be > 0 (never use requested qty)")
        if self.entry_vwap <= 0:
            raise ValueError("entry_vwap must be > 0")
        if self.highest_price <= 0:
            self.highest_price = float(self.entry_vwap)
        if self.initial_stop_price_value <= 0:
            self.initial_stop_price_value = initial_stop_price(self.entry_vwap, self.config)
        if self.activation_price_value <= 0:
            self.activation_price_value = activation_price(self.entry_vwap, self.config)
        if self.stop_price <= 0:
            self.stop_price = float(self.initial_stop_price_value)

    @classmethod
    def from_fill(
        cls,
        *,
        intent_id: str,
        symbol: str,
        entry_vwap: float,
        executed_quantity: float,
        client_order_id: str = "",
        requested_quantity: float | None = None,
        side: str = "BUY",
        config: T1Config = T1,
        timestamp: str | None = None,
    ) -> "T1Machine":
        m = cls(
            intent_id=intent_id,
            symbol=symbol,
            entry_vwap=float(entry_vwap),
            executed_quantity=float(executed_quantity),
            client_order_id=client_order_id,
            requested_quantity=requested_quantity,
            side=side,
            config=config,
            state=T1State.INITIAL_STOP,
            activated=False,
            highest_price=float(entry_vwap),
            stop_price=initial_stop_price(float(entry_vwap), config),
            initial_stop_price_value=initial_stop_price(float(entry_vwap), config),
            activation_price_value=activation_price(float(entry_vwap), config),
        )
        if timestamp:
            m.timestamps["entry"] = timestamp
        m.state = T1State.TRAILING_INACTIVE
        return m

    def mark_protection_established(
        self, protection_order_id: str, *, status: str = "BOT_MANAGED_ACTIVE", timestamp: str | None = None
    ) -> None:
        self.protection_order_id = protection_order_id
        self.protection_status = status
        if timestamp:
            self.timestamps["protection_established"] = timestamp
        if self.state == T1State.INITIAL_STOP:
            self.state = T1State.TRAILING_INACTIVE

    def on_price(self, observed_price: float) -> T1State:
        if self.state in {T1State.EXIT_TRIGGERED, T1State.CLOSED}:
            return self.state
        px = float(observed_price)
        if px <= 0:
            raise ValueError("observed_price must be > 0")

        if not self.activated:
            if px >= self.highest_price:
                self.highest_price = max(self.highest_price, px)
            if px >= self.activation_price_value:
                self.activated = True
                self.highest_price = max(self.highest_price, px)
                candidate = trailing_stop_from_high(self.highest_price, self.config)
                self.stop_price = ratchet_stop(self.stop_price, candidate)
                self.state = T1State.TRAILING_ACTIVE
            else:
                self.stop_price = float(self.initial_stop_price_value)
                self.state = T1State.TRAILING_INACTIVE
        else:
            self.highest_price = max(self.highest_price, px)
            candidate = trailing_stop_from_high(self.highest_price, self.config)
            self.stop_price = ratchet_stop(self.stop_price, candidate)
            self.state = T1State.TRAILING_ACTIVE

        if px <= self.stop_price:
            self.state = T1State.EXIT_TRIGGERED
        return self.state

    def mark_closed(self, *, timestamp: str | None = None) -> T1State:
        self.state = T1State.CLOSED
        if timestamp:
            self.timestamps["closed"] = timestamp
        return self.state

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "side": self.side,
            "requested_quantity": self.requested_quantity,
            "executed_quantity": self.executed_quantity,
            "entry_vwap": self.entry_vwap,
            "initial_stop_price": self.initial_stop_price_value,
            "activation_price": self.activation_price_value,
            "activated": self.activated,
            "highest_price": self.highest_price,
            "trailing_stop_price": self.stop_price,
            "protection_order_id": self.protection_order_id,
            "protection_status": self.protection_status,
            "state": self.state.value,
            "timestamps": dict(self.timestamps),
            "config": {
                "stop_loss_pct": self.config.stop_loss_pct,
                "activation_pct": self.config.activation_pct,
                "trailing_pct": self.config.trailing_pct,
                "key": self.config.key,
                "name": self.config.name,
            },
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "T1Machine":
        cfg_raw = data.get("config") or {}
        cfg = T1Config(
            stop_loss_pct=float(cfg_raw.get("stop_loss_pct", T1.stop_loss_pct)),
            activation_pct=float(cfg_raw.get("activation_pct", T1.activation_pct)),
            trailing_pct=float(cfg_raw.get("trailing_pct", T1.trailing_pct)),
            name=str(cfg_raw.get("name", T1.name)),
            key=str(cfg_raw.get("key", T1.key)),
        )
        return cls(
            intent_id=str(data["intent_id"]),
            symbol=str(data["symbol"]),
            entry_vwap=float(data["entry_vwap"]),
            executed_quantity=float(data["executed_quantity"]),
            client_order_id=str(data.get("client_order_id") or ""),
            requested_quantity=(
                float(data["requested_quantity"]) if data.get("requested_quantity") is not None else None
            ),
            side=str(data.get("side") or "BUY"),
            config=cfg,
            state=T1State(str(data["state"])),
            activated=bool(data.get("activated", False)),
            highest_price=float(data.get("highest_price") or data["entry_vwap"]),
            stop_price=float(
                data.get("trailing_stop_price")
                or data.get("stop_price")
                or data.get("initial_stop_price")
                or 0.0
            ),
            initial_stop_price_value=float(
                data.get("initial_stop_price") or initial_stop_price(float(data["entry_vwap"]), cfg)
            ),
            activation_price_value=float(
                data.get("activation_price") or activation_price(float(data["entry_vwap"]), cfg)
            ),
            protection_order_id=(str(data["protection_order_id"]) if data.get("protection_order_id") else None),
            protection_status=(str(data["protection_status"]) if data.get("protection_status") else None),
            timestamps=dict(data.get("timestamps") or {}),
            meta=dict(data.get("meta") or {}),
        )

    def log_fields(self, *, current_price: float | None = None) -> dict[str, Any]:
        out = {
            "symbol": self.symbol,
            "intent_id": self.intent_id,
            "client_order_id": self.client_order_id,
            "entry_vwap": self.entry_vwap,
            "filled_quantity": self.executed_quantity,
            "initial_stop": self.initial_stop_price_value,
            "activation_price": self.activation_price_value,
            "activated": self.activated,
            "highest_price": self.highest_price,
            "trailing_stop": self.stop_price,
            "protection_order_id": self.protection_order_id,
            "protection_status": self.protection_status,
            "state": self.state.value,
        }
        if current_price is not None:
            out["current_price"] = float(current_price)
        return out
