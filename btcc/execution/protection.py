"""Exchange-neutral T1 protection manager.

Production T1 path is BOT-MANAGED:
    T1Machine + T1PriceMonitor → MARKET SELL on stop cross

Does NOT require MEXC native protective-stop / trailing APIs.
Legacy EXCHANGE_RESIDENT mode remains available for adapter tests only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from btcc.execution.t1 import (
    T1,
    T1Config,
    T1Machine,
    T1State,
    activation_price,
    initial_stop_price,
    ratchet_stop,
    trailing_stop_from_high,
)


class ProtectionMode(str, Enum):
    BOT_MANAGED = "BOT_MANAGED"
    EXCHANGE_RESIDENT = "EXCHANGE_RESIDENT"  # legacy / tests only


class ProtectionState(str, Enum):
    UNPROTECTED = "UNPROTECTED"
    # Bot-managed: fill + local T1 geometry exist, but live mark not yet verified.
    AWAITING_LIVE_PRICE = "AWAITING_LIVE_PRICE"
    PROTECTED = "PROTECTED"
    REPLACEMENT_PENDING = "REPLACEMENT_PENDING"
    PROTECTION_UNKNOWN = "PROTECTION_UNKNOWN"
    EXITING = "EXITING"
    CLOSED = "CLOSED"
    HALTED = "HALTED"


def _t1_to_protection_state(t1: T1State, *, halted: bool = False) -> ProtectionState:
    if halted:
        return ProtectionState.HALTED
    if t1 == T1State.CLOSED:
        return ProtectionState.CLOSED
    if t1 == T1State.EXIT_TRIGGERED:
        return ProtectionState.EXITING
    if t1 in {T1State.INITIAL_STOP, T1State.TRAILING_INACTIVE, T1State.TRAILING_ACTIVE}:
        return ProtectionState.PROTECTED
    return ProtectionState.PROTECTION_UNKNOWN


@dataclass
class ProtectionRecord:
    position_id: str
    symbol: str
    quantity: float
    entry_price: float
    state: ProtectionState = ProtectionState.UNPROTECTED
    stop_price: float | None = None
    activated: bool = False
    high_water: float | None = None
    exchange_protection_id: str | None = None
    client_order_id: str | None = None
    geometry: T1Config = field(default_factory=lambda: T1)
    halt_reason: str | None = None
    mode: ProtectionMode = ProtectionMode.BOT_MANAGED
    monitor_armed: bool = False
    t1: T1Machine | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_halted(self) -> bool:
        return self.state == ProtectionState.HALTED

    @property
    def t1_state(self) -> T1State | None:
        return self.t1.state if self.t1 is not None else None

    @property
    def is_bot_managed(self) -> bool:
        return self.mode == ProtectionMode.BOT_MANAGED


class ProtectionBroker(Protocol):
    def place_protective_stop(
        self,
        *,
        symbol: str,
        quantity: float,
        stop_price: float,
        client_order_id: str,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    def cancel_protective_stop(self, *, symbol: str, protection_id: str) -> dict[str, Any]: ...

    def replace_protective_stop(
        self,
        *,
        symbol: str,
        old_protection_id: str,
        quantity: float,
        stop_price: float,
        client_order_id: str,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    def get_protective_stop(self, *, symbol: str, protection_id: str) -> dict[str, Any] | None: ...


class T1StateJournal:
    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._latest: dict[str, dict[str, Any]] = {}
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                for line in self.path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    pid = str(row.get("intent_id") or "")
                    if pid:
                        self._latest[pid] = row

    def persist(self, machine: T1Machine) -> None:
        payload = machine.to_dict()
        self._latest[machine.intent_id] = payload
        if self.path is None:
            return
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str) + "\n")

    def get(self, intent_id: str) -> dict[str, Any] | None:
        return self._latest.get(intent_id)

    def load_machine(self, intent_id: str) -> T1Machine | None:
        row = self.get(intent_id)
        return T1Machine.from_dict(row) if row else None


class ProtectiveExitManager:
    """T1 lifecycle. Production default = BOT_MANAGED (no native MEXC protection)."""

    def __init__(
        self,
        broker: ProtectionBroker | None = None,
        *,
        geometry: T1Config = T1,
        journal: T1StateJournal | None = None,
        mode: ProtectionMode = ProtectionMode.BOT_MANAGED,
    ) -> None:
        self.broker = broker
        self.geometry = geometry
        self.journal = journal
        self.mode = mode
        self._positions: dict[str, ProtectionRecord] = {}

    def get(self, position_id: str) -> ProtectionRecord | None:
        return self._positions.get(position_id)

    def establish(
        self,
        *,
        position_id: str,
        symbol: str,
        quantity: float,
        entry_price: float,
        client_order_id: str,
        requested_quantity: float | None = None,
    ) -> ProtectionRecord:
        """Establish T1. BOT_MANAGED never calls place_protective_stop."""
        if self.mode == ProtectionMode.BOT_MANAGED:
            return self.establish_bot_managed(
                position_id=position_id,
                symbol=symbol,
                quantity=quantity,
                entry_price=entry_price,
                client_order_id=client_order_id,
                requested_quantity=requested_quantity,
            )
        return self._establish_exchange_resident(
            position_id=position_id,
            symbol=symbol,
            quantity=quantity,
            entry_price=entry_price,
            client_order_id=client_order_id,
            requested_quantity=requested_quantity,
        )

    def establish_bot_managed(
        self,
        *,
        position_id: str,
        symbol: str,
        quantity: float,
        entry_price: float,
        client_order_id: str,
        requested_quantity: float | None = None,
    ) -> ProtectionRecord:
        if quantity <= 0:
            raise ValueError("protection quantity must be executed qty > 0")
        if requested_quantity is not None and quantity > float(requested_quantity) + 1e-15:
            raise ValueError("protection quantity must not exceed requested quantity")

        machine = T1Machine.from_fill(
            intent_id=position_id,
            symbol=symbol,
            entry_vwap=float(entry_price),
            executed_quantity=float(quantity),
            client_order_id=client_order_id,
            requested_quantity=requested_quantity,
            config=self.geometry,
        )
        bot_id = f"BOT_MANAGED:{position_id}"
        machine.mark_protection_established(bot_id, status="BOT_MANAGED_ACTIVE")
        rec = ProtectionRecord(
            position_id=position_id,
            symbol=symbol,
            quantity=float(quantity),
            entry_price=float(entry_price),
            stop_price=machine.stop_price,
            high_water=machine.highest_price,
            client_order_id=client_order_id,
            geometry=self.geometry,
            # Never claim PROTECTED until a valid live mark for this symbol arrives.
            state=ProtectionState.AWAITING_LIVE_PRICE,
            exchange_protection_id=bot_id,
            mode=ProtectionMode.BOT_MANAGED,
            monitor_armed=False,
            t1=machine,
            meta={
                "protection_mode": ProtectionMode.BOT_MANAGED.value,
                "live_price_verified": False,
            },
        )
        self._positions[position_id] = rec
        self._persist(machine)
        return rec

    def mark_monitor_armed(self, position_id: str) -> ProtectionRecord:
        rec = self._require(position_id)
        rec.monitor_armed = True
        if rec.t1 is not None:
            rec.t1.meta["monitor_armed"] = True
            rec.t1.protection_status = "BOT_MANAGED_MONITOR_ARMED"
            self._persist(rec.t1)
        return rec

    def confirm_live_price(self, position_id: str, *, mark: float, source: str) -> ProtectionRecord:
        """Promote AWAITING_LIVE_PRICE → PROTECTED after a validated live mark."""
        rec = self._require(position_id)
        if rec.state == ProtectionState.PROTECTED:
            rec.meta["live_price_verified"] = True
            rec.meta["live_price_confirm_mark"] = float(mark)
            rec.meta["live_price_confirm_source"] = source
            return rec
        if rec.state != ProtectionState.AWAITING_LIVE_PRICE:
            raise RuntimeError(
                f"confirm_live_price invalid state={rec.state.value} position={position_id}"
            )
        rec.state = ProtectionState.PROTECTED
        rec.meta["live_price_verified"] = True
        rec.meta["live_price_confirm_mark"] = float(mark)
        rec.meta["live_price_confirm_source"] = source
        if rec.t1 is not None:
            rec.t1.meta["live_price_verified"] = True
            rec.t1.protection_status = "BOT_MANAGED_LIVE_VERIFIED"
            self._persist(rec.t1)
        return rec

    def _establish_exchange_resident(
        self,
        *,
        position_id: str,
        symbol: str,
        quantity: float,
        entry_price: float,
        client_order_id: str,
        requested_quantity: float | None = None,
    ) -> ProtectionRecord:
        if self.broker is None:
            raise RuntimeError("EXCHANGE_RESIDENT mode requires ProtectionBroker")
        if quantity <= 0:
            raise ValueError("protection quantity must be executed qty > 0")
        machine = T1Machine.from_fill(
            intent_id=position_id,
            symbol=symbol,
            entry_vwap=float(entry_price),
            executed_quantity=float(quantity),
            client_order_id=client_order_id,
            requested_quantity=requested_quantity,
            config=self.geometry,
        )
        stop = machine.stop_price
        rec = ProtectionRecord(
            position_id=position_id,
            symbol=symbol,
            quantity=float(quantity),
            entry_price=float(entry_price),
            stop_price=stop,
            high_water=machine.highest_price,
            client_order_id=client_order_id,
            geometry=self.geometry,
            state=ProtectionState.PROTECTION_UNKNOWN,
            mode=ProtectionMode.EXCHANGE_RESIDENT,
            t1=machine,
        )
        self._positions[position_id] = rec
        self._persist(machine)
        try:
            ack = self.broker.place_protective_stop(
                symbol=symbol,
                quantity=float(quantity),
                stop_price=stop,
                client_order_id=client_order_id,
                meta={"kind": "INITIAL_SL", "geometry": self.geometry.name},
            )
        except Exception as e:  # noqa: BLE001
            rec.state = ProtectionState.HALTED
            rec.halt_reason = f"PROTECTION_CREATE_FAILED:{type(e).__name__}:{e}"
            raise
        rec.exchange_protection_id = str(ack.get("protection_id") or ack.get("order_id") or "")
        if not rec.exchange_protection_id:
            rec.state = ProtectionState.HALTED
            rec.halt_reason = "PROTECTION_CREATE_NO_ID"
            raise RuntimeError(rec.halt_reason)
        machine.mark_protection_established(rec.exchange_protection_id, status="ACTIVE")
        rec.state = ProtectionState.PROTECTED
        self._persist(machine)
        return rec

    def on_mark_price(self, position_id: str, mark: float) -> ProtectionRecord:
        rec = self._require(position_id)
        if rec.state in {ProtectionState.HALTED, ProtectionState.CLOSED, ProtectionState.EXITING}:
            return rec
        # Marks must not drive T1 while still awaiting a verified live feed.
        if rec.state == ProtectionState.AWAITING_LIVE_PRICE:
            return rec
        if rec.state != ProtectionState.PROTECTED:
            return rec
        if rec.is_bot_managed and not rec.monitor_armed:
            rec.state = ProtectionState.HALTED
            rec.halt_reason = "MARK_WITHOUT_ARMED_MONITOR"
            return rec
        if rec.t1 is None:
            rec.t1 = T1Machine(
                intent_id=position_id,
                symbol=rec.symbol,
                entry_vwap=rec.entry_price,
                executed_quantity=rec.quantity,
                client_order_id=rec.client_order_id or "",
                config=rec.geometry,
                state=T1State.TRAILING_ACTIVE if rec.activated else T1State.TRAILING_INACTIVE,
                activated=rec.activated,
                highest_price=float(rec.high_water or rec.entry_price),
                stop_price=float(rec.stop_price or initial_stop_price(rec.entry_price, rec.geometry)),
                initial_stop_price_value=initial_stop_price(rec.entry_price, rec.geometry),
                activation_price_value=activation_price(rec.entry_price, rec.geometry),
                protection_order_id=rec.exchange_protection_id,
                protection_status="ACTIVE",
            )

        prev_stop = float(rec.t1.stop_price)
        prev_activated = bool(rec.t1.activated)
        new_state = rec.t1.on_price(float(mark))
        rec.activated = rec.t1.activated
        rec.high_water = rec.t1.highest_price
        rec.stop_price = rec.t1.stop_price
        self._persist(rec.t1)

        if new_state == T1State.EXIT_TRIGGERED:
            rec.state = ProtectionState.EXITING
            rec.meta["exit_trigger_price"] = float(mark)
            return rec

        # Bot-managed: no exchange cancel/replace — local ratchet only.
        if rec.is_bot_managed:
            return rec

        if rec.t1.activated and float(rec.t1.stop_price) > prev_stop + 1e-15:
            reason = "ACTIVATE_TRAIL" if (rec.t1.activated and not prev_activated) else "RATCHET"
            return self._replace_stop(rec, float(rec.t1.stop_price), reason=reason)
        return rec

    def restore_from_journal(self, intent_id: str) -> ProtectionRecord | None:
        if self.journal is None:
            return None
        machine = self.journal.load_machine(intent_id)
        if machine is None:
            return None
        mode = ProtectionMode.BOT_MANAGED
        if str((machine.meta or {}).get("protection_mode") or "").upper() == "EXCHANGE_RESIDENT":
            mode = ProtectionMode.EXCHANGE_RESIDENT
        rec = ProtectionRecord(
            position_id=machine.intent_id,
            symbol=machine.symbol,
            quantity=machine.executed_quantity,
            entry_price=machine.entry_vwap,
            state=_t1_to_protection_state(machine.state),
            stop_price=machine.stop_price,
            activated=machine.activated,
            high_water=machine.highest_price,
            exchange_protection_id=machine.protection_order_id,
            client_order_id=machine.client_order_id,
            geometry=machine.config,
            mode=mode,
            monitor_armed=bool((machine.meta or {}).get("monitor_armed")),
            t1=machine,
        )
        self._positions[intent_id] = rec
        return rec

    def assert_protected_or_halt(self, position_id: str) -> ProtectionRecord:
        rec = self._require(position_id)
        if rec.state != ProtectionState.PROTECTED:
            rec.state = ProtectionState.HALTED
            rec.halt_reason = rec.halt_reason or "POSITION_EXISTS_WITHOUT_PROTECTION"
            return rec
        if rec.is_bot_managed and not rec.monitor_armed:
            rec.state = ProtectionState.HALTED
            rec.halt_reason = "POSITION_EXISTS_WITHOUT_ARMED_MONITOR"
        elif not rec.is_bot_managed and not rec.exchange_protection_id:
            rec.state = ProtectionState.HALTED
            rec.halt_reason = "POSITION_EXISTS_WITHOUT_PROTECTION"
        return rec

    def mark_closed(self, position_id: str) -> ProtectionRecord:
        rec = self._require(position_id)
        rec.state = ProtectionState.CLOSED
        if rec.t1 is not None:
            rec.t1.mark_closed()
            self._persist(rec.t1)
        return rec

    def _replace_stop(self, rec: ProtectionRecord, new_stop: float, *, reason: str) -> ProtectionRecord:
        if rec.is_bot_managed:
            rec.stop_price = float(new_stop)
            return rec
        if self.broker is None or not rec.exchange_protection_id:
            rec.state = ProtectionState.HALTED
            rec.halt_reason = "REPLACE_WITHOUT_EXISTING_PROTECTION"
            return rec
        if rec.stop_price is not None and float(new_stop) + 1e-15 < float(rec.stop_price):
            rec.state = ProtectionState.HALTED
            rec.halt_reason = "STOP_DECREASE_FORBIDDEN"
            return rec
        old_id = rec.exchange_protection_id
        rec.state = ProtectionState.REPLACEMENT_PENDING
        coid = f"{rec.client_order_id or 'prot'}-{reason[:8]}"
        try:
            ack = self.broker.replace_protective_stop(
                symbol=rec.symbol,
                old_protection_id=old_id,
                quantity=rec.quantity,
                stop_price=float(new_stop),
                client_order_id=coid,
                meta={"reason": reason},
            )
        except Exception as e:  # noqa: BLE001
            rec.state = ProtectionState.HALTED
            rec.halt_reason = f"REPLACE_FAILED:{type(e).__name__}:{e}"
            return rec
        new_id = str(ack.get("protection_id") or ack.get("order_id") or "")
        if not new_id:
            rec.state = ProtectionState.HALTED
            rec.halt_reason = "REPLACE_NO_ID"
            return rec
        rec.exchange_protection_id = new_id
        rec.stop_price = float(new_stop)
        rec.state = ProtectionState.PROTECTED
        rec.meta["last_replace_reason"] = reason
        if rec.t1 is not None:
            rec.t1.protection_order_id = new_id
            rec.t1.stop_price = float(new_stop)
            self._persist(rec.t1)
        return rec

    def _persist(self, machine: T1Machine) -> None:
        if self.journal is not None:
            self.journal.persist(machine)

    def _require(self, position_id: str) -> ProtectionRecord:
        rec = self._positions.get(position_id)
        if rec is None:
            raise KeyError(f"unknown position_id={position_id}")
        return rec
