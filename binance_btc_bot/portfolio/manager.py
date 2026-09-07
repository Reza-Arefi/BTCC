"""Central portfolio manager — slots, allocation, duplicate symbols, atomic reserve.

Production defaults (config-driven, not hard-coded call sites):
  max_simultaneous_trades = 8
  allocation_per_trade = 0.125
  max_total_allocation = 1.0
Supports configured max simultaneous trades in [3, 30].
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

from binance_btc_bot.portfolio.trade_state import ACTIVE_SLOT_STATES, TradeStatus

MIN_SIMULTANEOUS = 1  # 1 allowed only for controlled first-trade oneshot; production default remains 8
MAX_SIMULTANEOUS = 30
PRODUCTION_MAX_SIMULTANEOUS = 8


@dataclass(frozen=True)
class PortfolioConfig:
    max_simultaneous_trades: int
    allocation_per_trade: float
    max_total_allocation: float = 1.0
    one_position_per_symbol: bool = True

    @property
    def total_planned_allocation(self) -> float:
        return float(self.max_simultaneous_trades) * float(self.allocation_per_trade)


def validate_portfolio_config(raw: dict[str, Any] | None) -> PortfolioConfig:
    """Validate portfolio YAML; reject configs that can exceed 100% total allocation."""
    if not raw:
        raise ValueError("portfolio section is required")
    try:
        max_n = int(raw.get("max_simultaneous_trades"))
    except (TypeError, ValueError) as e:
        raise ValueError("portfolio.max_simultaneous_trades must be an integer") from e
    if max_n < MIN_SIMULTANEOUS or max_n > MAX_SIMULTANEOUS:
        raise ValueError(
            f"portfolio.max_simultaneous_trades must be in [{MIN_SIMULTANEOUS}, {MAX_SIMULTANEOUS}], got {max_n}"
        )
    try:
        alloc = float(raw.get("allocation_per_trade"))
    except (TypeError, ValueError) as e:
        raise ValueError("portfolio.allocation_per_trade must be a float fraction") from e
    if not (0.0 < alloc <= 1.0):
        raise ValueError(f"portfolio.allocation_per_trade must be in (0, 1], got {alloc}")
    try:
        max_total = float(raw.get("max_total_allocation", 1.0))
    except (TypeError, ValueError) as e:
        raise ValueError("portfolio.max_total_allocation must be a float") from e
    if not (0.0 < max_total <= 1.0 + 1e-12):
        raise ValueError(f"portfolio.max_total_allocation must be in (0, 1], got {max_total}")

    planned = max_n * alloc
    if planned > max_total + 1e-12:
        raise ValueError(
            f"invalid portfolio: {max_n} × {alloc} = {planned:.6f} exceeds "
            f"max_total_allocation={max_total}"
        )

    return PortfolioConfig(
        max_simultaneous_trades=max_n,
        allocation_per_trade=alloc,
        max_total_allocation=max_total,
        one_position_per_symbol=bool(raw.get("one_position_per_symbol", True)),
    )


@dataclass
class Reservation:
    reservation_id: str
    symbol: str
    status: TradeStatus = TradeStatus.RESERVED
    trade_id: str | None = None
    allocation_pct: float = 0.0


@dataclass(frozen=True)
class ReserveResult:
    ok: bool
    reservation: Reservation | None = None
    reason: str = ""


@dataclass
class PortfolioManager:
    """Authoritative portfolio slot + allocation controller.

    Thread-safe: two concurrent try_reserve() calls cannot both take the last slot.
    """

    config: PortfolioConfig
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _reservations: dict[str, Reservation] = field(default_factory=dict, repr=False)
    _by_symbol: dict[str, str] = field(default_factory=dict, repr=False)  # symbol -> reservation_id

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> PortfolioManager:
        portfolio = validate_portfolio_config(cfg.get("portfolio"))
        return cls(config=portfolio)

    @property
    def max_simultaneous_trades(self) -> int:
        return int(self.config.max_simultaneous_trades)

    @property
    def allocation_per_trade(self) -> float:
        return float(self.config.allocation_per_trade)

    @property
    def max_total_allocation(self) -> float:
        return float(self.config.max_total_allocation)

    def set_max_simultaneous_trades(self, n: int) -> int:
        """Runtime slot-ceiling change (NEW entries only).

        Does not close or modify existing reservations/positions.
        Allocation per trade and max_total_allocation stay frozen — if
        ``n * allocation > max_total``, capacity remains limited by allocation.
        """
        max_n = int(n)
        if max_n < MIN_SIMULTANEOUS or max_n > MAX_SIMULTANEOUS:
            raise ValueError(
                f"max_simultaneous_trades must be in [{MIN_SIMULTANEOUS}, {MAX_SIMULTANEOUS}]"
            )
        with self._lock:
            cfg = self.config
            self.config = PortfolioConfig(
                max_simultaneous_trades=max_n,
                allocation_per_trade=cfg.allocation_per_trade,
                max_total_allocation=cfg.max_total_allocation,
                one_position_per_symbol=cfg.one_position_per_symbol,
            )
            return max_n

    def slots_used(self) -> int:
        with self._lock:
            return sum(1 for r in self._reservations.values() if r.status in ACTIVE_SLOT_STATES)

    def slots_remaining(self) -> int:
        return max(0, self.max_simultaneous_trades - self.slots_used())

    def open_symbols(self) -> set[str]:
        with self._lock:
            return {
                r.symbol
                for r in self._reservations.values()
                if r.status in ACTIVE_SLOT_STATES
            }

    def allocated_pct(self) -> float:
        """Sum of reserved/active allocation fractions (planned per-slot)."""
        with self._lock:
            return sum(
                float(r.allocation_pct)
                for r in self._reservations.values()
                if r.status in ACTIVE_SLOT_STATES
            )

    def remaining_allocation_pct(self) -> float:
        return max(0.0, self.max_total_allocation - self.allocated_pct())

    def try_reserve(self, symbol: str) -> ReserveResult:
        """Atomically reserve one portfolio slot for ``symbol``.

        Rejects duplicate symbols and when no slots remain.
        """
        sym = str(symbol).upper()
        with self._lock:
            if self.config.one_position_per_symbol and sym in self._by_symbol:
                return ReserveResult(ok=False, reason="SAME_PAIR_ALREADY_OPEN")
            used = sum(1 for r in self._reservations.values() if r.status in ACTIVE_SLOT_STATES)
            if used >= self.max_simultaneous_trades:
                return ReserveResult(ok=False, reason="MAX_OPEN_TRADES")
            if self.allocated_pct() + self.allocation_per_trade > self.max_total_allocation + 1e-12:
                return ReserveResult(ok=False, reason="MAX_TOTAL_ALLOCATION")

            rid = f"res_{uuid.uuid4().hex[:16]}"
            reservation = Reservation(
                reservation_id=rid,
                symbol=sym,
                status=TradeStatus.RESERVED,
                allocation_pct=self.allocation_per_trade,
            )
            self._reservations[rid] = reservation
            self._by_symbol[sym] = rid
            return ReserveResult(ok=True, reservation=reservation, reason="OK")

    def transition(self, reservation_id: str, status: TradeStatus, *, trade_id: str | None = None) -> bool:
        with self._lock:
            res = self._reservations.get(reservation_id)
            if res is None:
                return False
            res.status = status
            if trade_id is not None:
                res.trade_id = trade_id
            if status not in ACTIVE_SLOT_STATES:
                self._by_symbol.pop(res.symbol, None)
                self._reservations.pop(reservation_id, None)
            return True

    def release(self, reservation_id: str) -> bool:
        """Release a reservation / close out a slot."""
        with self._lock:
            res = self._reservations.pop(reservation_id, None)
            if res is None:
                return False
            if self._by_symbol.get(res.symbol) == reservation_id:
                del self._by_symbol[res.symbol]
            return True

    def release_symbol(self, symbol: str) -> bool:
        with self._lock:
            rid = self._by_symbol.get(str(symbol).upper())
            if not rid:
                return False
        return self.release(rid)

    def mark_protected(self, reservation_id: str, trade_id: str) -> bool:
        return self.transition(reservation_id, TradeStatus.PROTECTED, trade_id=trade_id)

    def rehydrate_from_trades(self, trades: list[dict[str, Any]]) -> dict[str, Any]:
        """Rebuild slot state from persisted open trades (restart recovery).

        Binance/DB remain source of truth for positions; this restores local
        capacity accounting so a ninth entry cannot slip through after restart.
        """
        with self._lock:
            self._reservations.clear()
            self._by_symbol.clear()
            restored = 0
            skipped = 0
            for t in trades:
                sym = str(t.get("symbol") or "").upper()
                if not sym:
                    skipped += 1
                    continue
                if sym in self._by_symbol:
                    skipped += 1
                    continue
                st_raw = str(t.get("status") or "PROTECTED").upper()
                try:
                    status = TradeStatus(st_raw) if st_raw in TradeStatus.__members__ else TradeStatus.PROTECTED
                except Exception:  # noqa: BLE001
                    status = TradeStatus.PROTECTED
                if st_raw in {"DRY_RUN_PROTECTED", "PROTECTED_EMERGENCY"}:
                    status = (
                        TradeStatus.PROTECTED_EMERGENCY
                        if st_raw == "PROTECTED_EMERGENCY"
                        else TradeStatus.PROTECTED
                    )
                if status not in ACTIVE_SLOT_STATES and st_raw in {
                    "DRY_RUN_PROTECTED",
                    "OPEN",
                    "DRY_RUN",
                }:
                    status = TradeStatus.PROTECTED
                if status not in ACTIVE_SLOT_STATES:
                    skipped += 1
                    continue
                if restored >= self.max_simultaneous_trades:
                    skipped += 1
                    continue
                rid = f"rehy_{sym.lower()}_{restored}"
                self._reservations[rid] = Reservation(
                    reservation_id=rid,
                    symbol=sym,
                    status=status if status in ACTIVE_SLOT_STATES else TradeStatus.PROTECTED,
                    trade_id=str(t.get("trade_id") or "") or None,
                    allocation_pct=self.allocation_per_trade,
                )
                self._by_symbol[sym] = rid
                restored += 1
            return {
                "restored": restored,
                "skipped": skipped,
                "slots_used": self.slots_used(),
                "slots_remaining": self.slots_remaining(),
            }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "max_simultaneous_trades": self.max_simultaneous_trades,
                "allocation_per_trade": self.allocation_per_trade,
                "max_total_allocation": self.max_total_allocation,
                "slots_used": self.slots_used(),
                "slots_remaining": self.slots_remaining(),
                "allocated_pct": self.allocated_pct(),
                "open_symbols": sorted(self.open_symbols()),
                "reservations": [
                    {
                        "reservation_id": r.reservation_id,
                        "symbol": r.symbol,
                        "status": r.status.value,
                        "trade_id": r.trade_id,
                        "allocation_pct": r.allocation_pct,
                    }
                    for r in self._reservations.values()
                ],
            }
